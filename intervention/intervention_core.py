from __future__ import annotations

from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


import copy
import torch.nn.functional as F
from scale_intervention import ScaleIntervention

def create_dataloader(
    df: pd.DataFrame,
    phoneme_col: str,
    phoneme_to_id: dict[str, int],
    batch_size: int = 32,
    shuffle: bool = True,
    max_len: int = 20,
    max_pos: int = 9,
    random_replace_pos: bool = True,
    test_run: bool = False,
) -> DataLoader:
    sequences: list[list[int]] = []
    for phonemes in df[phoneme_col]:
        if isinstance(phonemes, str):
            phonemes = literal_eval(phonemes)
        sequences.append([phoneme_to_id[p] for p in phonemes])

    dataset = InterventionDataset(
        sequences,
        len(phoneme_to_id),
        phoneme_to_id["<PAD>"],
        phoneme_to_id["<EOS>"],
        phoneme_to_id["<SOS>"],
        max_len=max_len,
        max_pos=max_pos,
        random_replace_pos=random_replace_pos,
        test_run=test_run,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class InterventionDataset(Dataset):
    def __init__(
        self,
        phoneme_sequences: list[list[int]],
        vocab_size: int,
        pad_id: int,
        eos_id: int,
        start_id: int,
        max_len: int = 20,
        max_pos: int = 9,
        random_replace_pos: bool = True,
        test_run: bool = False,
    ):
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.start_id = start_id
        self.max_len = max_len
        self.max_pos = max_pos
        self.special_tokens = {pad_id, eos_id, start_id}
        self.random_replace_pos = random_replace_pos
        self.test_run = test_run

        self.examples: list[dict[str, int | list[int]]] = []
        for seq in phoneme_sequences:
            seq = seq.copy()
            if seq[-1] != self.eos_id:
                seq.append(self.eos_id)

            if self.random_replace_pos:
                replace_pos = np.random.randint(0, min(self.max_pos, len(seq) - 1))
                pos_list = [replace_pos]
            else:
                pos_list = list(range(min(self.max_pos, len(seq) - 1)))

            for replace_pos in pos_list:
                old_token = seq[replace_pos]
                modified_seq = seq.copy()
                if self.test_run:
                    new_token = old_token
                else:
                    valid_tokens = [
                        t
                        for t in range(self.vocab_size)
                        if t != old_token and t not in self.special_tokens
                    ]
                    new_token = np.random.choice(valid_tokens)
                    modified_seq[replace_pos] = new_token

                self.examples.append(
                    {
                        "seq": seq,
                        "modified_seq": modified_seq,
                        "replace_pos": replace_pos,
                        "old_token": old_token,
                        "new_token": new_token,
                    }
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.examples[idx]
        seq = example["seq"]
        modified_seq = example["modified_seq"]

        input_padded = seq + [self.pad_id] * max(0, self.max_len - len(seq))
        target_padded = modified_seq + [self.pad_id] * max(0, self.max_len - len(modified_seq))

        return {
            "input": torch.tensor(input_padded[: self.max_len], dtype=torch.long),
            "target": torch.tensor(target_padded[: self.max_len], dtype=torch.long),
            "old_token": torch.tensor(example["old_token"], dtype=torch.long),
            "new_token": torch.tensor(example["new_token"], dtype=torch.long),
            "position": torch.tensor(example["replace_pos"], dtype=torch.long),
            "seq_len": torch.tensor(len(seq), dtype=torch.long),
        }





def get_encoder_hidden(model: nn.Module, input_ids: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        h, c = model.encoder(input_ids.to(device))
        return h[-1], c[-1]


def decode_with_hidden(
    model: nn.Module,
    h: torch.Tensor,
    c: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
    teacher_forcing: bool = False,
) -> torch.Tensor:
    h, c = h.unsqueeze(0), c.unsqueeze(0)
    batch_size, max_len = target.shape

    outputs: list[torch.Tensor] = []
    hidden = (h, c)
    inp = torch.full((batch_size, 1), model.start_token_id, device=device, dtype=torch.long)

    for t in range(max_len):
        embedded = model.decoder.dropout(model.decoder.embedding(inp))
        out, hidden = model.decoder.recurrent(embedded, hidden)
        logits = out @ model.decoder.embedding.weight.T
        outputs.append(logits)
        if teacher_forcing:
            inp = target[:, t : t + 1]
        else:
            inp = logits.argmax(dim=-1)

    return torch.cat(outputs, dim=1)


def _accuracy(preds: torch.Tensor, targets: torch.Tensor, seq_lens: torch.Tensor) -> float:
    batch_size = preds.shape[0]
    seq_correct = 0
    for i in range(batch_size):
        sl = seq_lens[i].item()
        if torch.equal(preds[i, :sl], targets[i, :sl]):
            seq_correct += 1
    return seq_correct / batch_size


class InterventionTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        intervention: ScaleIntervention,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        pad_id: int,
    ):
        self.model = model
        self.intervention = intervention
        self.optimizer = optimizer
        self.device = device
        self.pad_id = pad_id
        self.history: dict[str, list[float]] = {}

    def _compute_loss(self, logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=self.pad_id,
        )

    def _run_batch(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = batch["input"].to(self.device)
        target_ids = batch["target"].to(self.device)
        old_token = batch["old_token"].to(self.device)
        new_token = batch["new_token"].to(self.device)
        position = batch["position"].to(self.device)
        seq_len = batch["seq_len"]

        h, c = get_encoder_hidden(self.model, input_ids, self.device)
        h_mod, c_mod = self.intervention.intervene(h, c, old_token, new_token, position)
        logits = decode_with_hidden(self.model, h_mod, c_mod, target_ids, self.device)

        loss = self._compute_loss(logits, target_ids)
        preds = logits.argmax(dim=-1)
        return loss, preds, target_ids, seq_len

    def _run_loader(self, loader: DataLoader, training: bool) -> tuple[float, float]:
        if training:
            self.intervention.train()
        else:
            self.intervention.eval()
            self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for batch in loader:
            if training:
                self.optimizer.zero_grad()

            loss, preds, targets, seq_lens = self._run_batch(batch)

            if training:
                loss.backward()
                self.optimizer.step()

            acc = _accuracy(preds, targets, seq_lens)
            total_loss += loss.item()
            total_acc += acc
            n_batches += 1

        return total_loss / n_batches, total_acc / n_batches

    def train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        return self._run_loader(loader, training=True)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        return self._run_loader(loader, training=False)

    @torch.no_grad()
    def evaluate_with_predictions(self, loader: DataLoader, id_to_phoneme: dict[int, str]) -> pd.DataFrame:
        self.intervention.eval()
        self.model.eval()

        records: list[dict[str, object]] = []
        for batch in loader:
            _, preds, targets, seq_lens = self._run_batch(batch)
            batch_size = preds.shape[0]
            for i in range(batch_size):
                sl = seq_lens[i].item()
                pred_ids = preds[i, :sl].cpu().tolist()
                target_ids = targets[i, :sl].cpu().tolist()
                input_ids = batch["input"][i, :sl].tolist()

                records.append(
                    {
                        "input": " ".join([id_to_phoneme[t] for t in input_ids]),
                        "target": " ".join([id_to_phoneme[t] for t in target_ids]),
                        "prediction": " ".join([id_to_phoneme[p] for p in pred_ids]),
                        "position": batch["position"][i].item(),
                        "old_phoneme": id_to_phoneme[batch["old_token"][i].item()],
                        "new_phoneme": id_to_phoneme[batch["new_token"][i].item()],
                        "seq_len": sl,
                        "match": pred_ids == target_ids,
                        "token_acc": sum(p == t for p, t in zip(pred_ids, target_ids)) / sl,
                    }
                )

        return pd.DataFrame(records)

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
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "test_loss": [],
            "test_acc": [],
        }

        best_val_loss = float("inf")
        best_state = copy.deepcopy(self.intervention.state_dict())
        epochs_without_improvement = 0

        train_loss, train_acc = self.evaluate(train_loader)
        val_loss, val_acc = self.evaluate(val_loader)
        test_loss, test_acc = (None, None)
        if test_loader is not None:
            test_loss, test_acc = self.evaluate(test_loader)

        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)
        if test_loader is not None:
            self.history["test_loss"].append(test_loss)
            self.history["test_acc"].append(test_acc)
        if verbose:
            print(
                f"Epoch {0:3d} | "
                f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"Val loss={val_loss:.4f} acc={val_acc:.4f}"
                + (f" | Test loss={test_loss:.4f} acc={test_acc:.4f}" if test_loader is not None else "")
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(self.intervention.state_dict())

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            test_loss, test_acc = (None, None)
            if test_loader is not None:
                test_loss, test_acc = self.evaluate(test_loader)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            if test_loader is not None:
                self.history["test_loss"].append(test_loss)
                self.history["test_acc"].append(test_acc)
            if verbose:
                print(
                    f"Epoch {epoch:3d} | "
                    f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
                    f"Val loss={val_loss:.4f} acc={val_acc:.4f}"
                    + (f" | Test loss={test_loss:.4f} acc={test_acc:.4f}" if test_loader is not None else "")
                )

            if val_loss + min_delta < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.intervention.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping after {epoch} epochs: "
                    f"no improvement on val set for {patience} epochs."
                )
                break

        self.intervention.load_state_dict(best_state)
        return self.history

    def save_scale_params(self, save_dir: Path) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        params = self.intervention.scale_parameters()
        np.savez(save_dir / "scale_params.npz", **params)
