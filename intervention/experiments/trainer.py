"""Training loop shared by every intervention method.

``InterventionTrainer`` implements fit / evaluate / prediction over a frozen repeat
model. Each method supplies its own batch step by overriding ``_run_batch`` (the scale
method uses the base implementation; DAS overrides it to swap encoder states), so the
early-stopping loop, metrics, and I/O are written once.

Metrics are exact (sequence-accuracy = correct / total sequences; loss weighted by the
number of sequences), which removes the small-final-batch bias of averaging per-batch
means.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from intervention.models.repeat_model_utils import decode_with_hidden, get_encoder_hidden


def count_correct(preds: torch.Tensor, targets: torch.Tensor, seq_lens: torch.Tensor) -> int:
    """Number of sequences predicted exactly right up to their (unpadded) length."""
    correct = 0
    for i in range(preds.shape[0]):
        sl = int(seq_lens[i])
        if torch.equal(preds[i, :sl], targets[i, :sl]):
            correct += 1
    return correct


class InterventionTrainer:
    def __init__(
        self,
        repeat_model: nn.Module,
        intervention: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        pad_id: int,
        teacher_forcing: bool = False,
    ):
        self.repeat_model = repeat_model
        self.intervention = intervention
        self.optimizer = optimizer
        self.device = device
        self.pad_id = pad_id
        self.teacher_forcing = teacher_forcing
        self.history: dict[str, list[float]] = {}

    # -- per-batch step (methods override this) ------------------------------ #
    def _compute_loss(self, logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=self.pad_id
        )

    def _run_batch(self, batch: dict[str, torch.Tensor]):
        """Scale-style step: intervene from the (old_token -> new_token) delta."""
        input_ids = batch["input"].to(self.device)
        target_ids = batch["target"].to(self.device)
        old_token = batch["old_token"].to(self.device)
        new_token = batch["new_token"].to(self.device)
        position = batch["position"].to(self.device)

        h, c = get_encoder_hidden(self.repeat_model, input_ids, self.device)
        h_mod, c_mod = self.intervention.intervene(h, c, old_token, new_token, position)
        logits = decode_with_hidden(self.repeat_model, h_mod, c_mod, target_ids, self.device, self.teacher_forcing)

        loss = self._compute_loss(logits, target_ids)
        return loss, logits.argmax(dim=-1), target_ids, batch["seq_len"]

    # -- loops --------------------------------------------------------------- #
    def _run_loader(self, loader: DataLoader, training: bool) -> tuple[float, float]:
        self.intervention.train(training)
        # self.repeat_model.train(training)
        self.repeat_model.eval()

        loss_sum, correct, total = 0.0, 0, 0
        for batch in loader:
            if training:
                self.optimizer.zero_grad()
            loss, preds, targets, seq_lens = self._run_batch(batch)
            if training:
                loss.backward()
                self.optimizer.step()

            n = preds.shape[0]
            loss_sum += loss.item() * n
            correct += count_correct(preds, targets, seq_lens)
            total += n

        if total == 0:  # empty loader (e.g. a tiny test split) -> report NaN, don't crash
            return math.nan, math.nan
        return loss_sum / total, correct / total

    def train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        return self._run_loader(loader, training=True)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        return self._run_loader(loader, training=False)

    @torch.no_grad()
    def evaluate_with_predictions(
        self,
        loader: DataLoader,
        id_to_phoneme: dict[int, str],
        edit_id_to_str: dict[int, str] | None = None,
    ) -> pd.DataFrame:
        """One row per test example. ``edit_id_to_str`` labels ``old/new_token`` when they
        index something other than the phoneme vocab (n-gram edits); defaults to phonemes."""
        edit_id_to_str = edit_id_to_str or id_to_phoneme
        self.intervention.eval()
        self.repeat_model.eval()
        records: list[dict[str, object]] = []
        for batch in loader:
            _, preds, targets, seq_lens = self._run_batch(batch)
            for i in range(preds.shape[0]):
                sl = int(seq_lens[i])
                pred_ids = preds[i, :sl].cpu().tolist()
                target_ids = targets[i, :sl].cpu().tolist()
                input_ids = batch["input"][i, :sl].tolist()
                records.append({
                    "input": " ".join(id_to_phoneme[t] for t in input_ids),
                    "source": " ".join(id_to_phoneme[t] for t in batch["source"][i, :sl].tolist()),
                    "target": " ".join(id_to_phoneme[t] for t in target_ids),
                    "prediction": " ".join(id_to_phoneme[p] for p in pred_ids),
                    "position": batch["position"][i].item(),
                    "old_phoneme": edit_id_to_str[batch["old_token"][i].item()],
                    "new_phoneme": edit_id_to_str[batch["new_token"][i].item()],
                    "seq_len": sl,
                    "match": pred_ids == target_ids,
                    "token_acc": sum(p == t for p, t in zip(pred_ids, target_ids)) / sl,
                })
        return pd.DataFrame(records)

    # -- fit with early stopping -------------------------------------------- #
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        patience: int = 5,
        min_delta: float = 1e-4,
        test_loader: DataLoader | None = None,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        splits = {"train": train_loader, "val": val_loader}
        if test_loader is not None and len(test_loader) > 0:
            splits["test"] = test_loader
        self.history = {f"{name}_{m}": [] for name in splits for m in ("loss", "acc")}

        best_val, best_epoch = float("inf"), 0
        best_state = copy.deepcopy(self.intervention.state_dict())
        stale = 0

        for epoch in range(num_epochs + 1):  # epoch 0 = evaluation before any training
            for name, loader in splits.items():
                loss, acc = self.train_epoch(loader) if (epoch > 0 and name == "train") else self.evaluate(loader)
                self.history[f"{name}_loss"].append(loss)
                self.history[f"{name}_acc"].append(acc)
            if verbose:
                self._log(epoch, splits)

            val_loss = self.history["val_loss"][-1]
            if val_loss + min_delta < best_val:
                best_val, best_epoch, stale = val_loss, epoch, 0
                best_state = copy.deepcopy(self.intervention.state_dict())
            elif epoch > 0:
                stale += 1
                if stale >= patience:
                    print(f"Early stopping at epoch {epoch}: no val improvement for {patience} epochs.")
                    break

        self.intervention.load_state_dict(best_state)
        if verbose:
            print(f"Loaded best state from epoch {best_epoch} (val_loss={best_val:.4f}).")
        return self.history

    def _log(self, epoch: int, splits: dict) -> None:
        parts = [
            f"{name} loss={self.history[f'{name}_loss'][-1]:.4f} acc={self.history[f'{name}_acc'][-1]:.4f}"
            for name in splits
        ]
        print(f"Epoch {epoch:3d} | " + " | ".join(parts))

    def save_params(self, save_dir: Path) -> None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        np.savez(Path(save_dir) / "params.npz", **self.intervention.get_parameters())
