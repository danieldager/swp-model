"""Distributed Alignment Search (DAS) intervention for the phoneme repeat model.

  * a learned orthogonal **rotation matrix** R maps the RNN state into a
    disentangled space where each sequence position owns a contiguous block
    ("variable");
  * an optional **alignment matrix** (a soft per-channel mask) selects which
    rotated channels actually carry the causal variable.

Real-real counterfactual: a minimal pair of real words (`base`, `source`) that
differs at a single `position`.  We rotate both encoder states, swap the block
of the edited position from `source` into `base`, rotate back, and decode.  The
rotation (+ alignment) is trained so the decoded word equals `source`.

Pipeline (`run_das`): frozen repeat model -> real-real minimal-pair loaders ->
train rotation -> evaluate / save.

Run from the `intervention/` directory:
    python DAS.py --state_mode concat --epochs 200
"""
from __future__ import annotations

import os

# The orthogonal rotation uses `matrix_exp`, which is unimplemented on Apple MPS;
# fall back to CPU for that one small (D x D) op. The native-MPS alternatives in
# torch 2.10 are unusable: orthogonal_map="cayley" crashes in linalg_lu_solve and
# "householder" produces NaN gradients. Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import sys
from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
for _p in (ROOT, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from swp.datasets.phonemes import get_phoneme_to_id
from swp.utils.datasets import get_train_dataset
from swp.utils.models import get_model
from swp.utils.setup import set_device

from intervention.core import (
    InterventionTrainer,
    create_real_real_dataloader,
    decode_with_hidden,
    get_encoder_hidden,
)


# --------------------------------------------------------------------------- #
# DAS module: rotation matrix + alignment matrix
# --------------------------------------------------------------------------- #
def gumbel_sigmoid(logits: torch.Tensor, tau: float = 1.0, hard: bool = True, eps: float = 1e-10) -> torch.Tensor:
    """Differentiable Bernoulli sample (concrete relaxation) used to train the mask."""
    u = torch.rand_like(logits)
    logistic = torch.log(u + eps) - torch.log(1 - u + eps)
    y = torch.sigmoid((logits + logistic) / tau)
    if hard:
        y = (y > 0.5).float() + y - y.detach()
    return y


class DASIntervention(nn.Module):
    """Rotation-based interchange intervention on the (frozen) RNN state.

    state_mode selects what is intervened on: 'concat' = [h, c], else 'h'/'c'/'both'.
    """

    def __init__(
        self,
        state_dim: int,
        n_variables: int,
        var_size: int | None = None,
        masking: bool = True,
        reg_loss: float = 1e-3,
        state_mode: str = "concat",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_variables = n_variables
        self.var_size = var_size or state_dim // n_variables
        self.intervention_size = self.var_size * self.n_variables
        if self.intervention_size > state_dim:
            raise ValueError(
                f"var_size*n_variables={self.intervention_size} exceeds state_dim={state_dim}"
            )
        self.masking = masking
        self.reg_loss = reg_loss
        self.state_mode = state_mode

        # Rotation matrix R (orthogonal) -- Uses the default matrix_exp map;
        self.rotation = nn.utils.parametrizations.orthogonal(
            nn.Linear(state_dim, state_dim, bias=False)
        )
        # Alignment matrix: soft per-channel mask over the intervened region.
        if masking:
            self.mask = nn.Parameter(torch.zeros(self.intervention_size))
        self.register_buffer("_reg", torch.zeros(()), persistent=False)

    # -- core: rotate -> swap edited block -> (align) -> rotate back -------- #
    def _rotate_swap(self, base: torch.Tensor, source: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        B = base.shape[0]
        rb = self.rotation(base)      # (B, D)  base in DAS basis
        rs = self.rotation(source)    # (B, D)  source in DAS basis
        I = self.intervention_size

        bb = rb[:, :I].view(B, self.n_variables, self.var_size)
        ss = rs[:, :I].view(B, self.n_variables, self.var_size)

        # Replace only the block of the edited position with the source's block.
        sel = F.one_hot(position.clamp(max=self.n_variables - 1), self.n_variables).bool()[..., None]
        swapped = torch.where(sel, ss, bb).reshape(B, I)

        if self.masking:
            swapped = self._align(swapped, rb[:, :I])

        out = torch.cat([swapped, rb[:, I:]], dim=-1)   # untouched dims kept from base
        return F.linear(out, self.rotation.weight.t())  # inverse rotation (R orthogonal)

    def _align(self, intervened: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        m = gumbel_sigmoid(self.mask) if self.training else (self.mask > 0).float()
        self._reg = self._reg + self.reg_loss * torch.sigmoid(self.mask).sum()
        return m * intervened + (1 - m) * original

    def intervene(
        self,
        h: torch.Tensor,
        c: torch.Tensor,
        h_src: torch.Tensor,
        c_src: torch.Tensor,
        position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._reg = torch.zeros((), device=h.device)
        if self.state_mode == "concat":
            out = self._rotate_swap(torch.cat([h, c], -1), torch.cat([h_src, c_src], -1), position)
            return out.chunk(2, dim=-1)
        if self.state_mode == "h":
            return self._rotate_swap(h, h_src, position), c
        if self.state_mode == "c":
            return h, self._rotate_swap(c, c_src, position)
        if self.state_mode == "both":
            return self._rotate_swap(h, h_src, position), self._rotate_swap(c, c_src, position)
        raise ValueError(f"unknown state_mode: {self.state_mode}")

    def reg(self) -> torch.Tensor:
        return self._reg

    @torch.no_grad()
    def get_parameters(self) -> dict[str, np.ndarray]:
        params = {
            "rotation": self.rotation.weight.detach().cpu().numpy(),
            "n_variables": np.int64(self.n_variables),
            "var_size": np.int64(self.var_size),
            "state_mode": self.state_mode,
        }
        if self.masking:
            params["mask"] = torch.sigmoid(self.mask).detach().cpu().numpy()
        return params


# --------------------------------------------------------------------------- #
# Trainer: base = input word, source = target word (differ only at `position`)
# --------------------------------------------------------------------------- #
class DASTrainer(InterventionTrainer):
    """Reuses InterventionTrainer's fit/evaluate loops; only the batch differs:
    DAS needs the *source* encoder state, so we encode both words and swap."""

    def _run_batch(self, batch: dict[str, torch.Tensor]):
        base_ids = batch["input"].to(self.device)     # base word (a)
        source_ids = batch["target"].to(self.device)  # source word (b) == counterfactual target
        position = batch["position"].to(self.device)
        seq_len = batch["seq_len"]

        h, c = get_encoder_hidden(self.repeat_model, base_ids, self.device)
        h_src, c_src = get_encoder_hidden(self.repeat_model, source_ids, self.device)
        h_mod, c_mod = self.intervention.intervene(h, c, h_src, c_src, position)
        logits = decode_with_hidden(self.repeat_model, h_mod, c_mod, source_ids, self.device, self.teacher_forcing)

        loss = self._compute_loss(logits, source_ids) + self.intervention.reg()
        preds = logits.argmax(dim=-1)
        return loss, preds, source_ids, seq_len


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
@dataclass
class DASConfig:
    model_name: str = "Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1"
    weights_path: str = "reproduce/weights/1024_75.pth"  # resolved against repo root
    hidden_size: int = 128
    state_mode: str = "concat"        # concat | h | c | both
    var_size: int | None = None       # dims per position-variable (default: state_dim // n_variables)
    masking: bool = False             # learn the alignment matrix
    reg_loss: float = 1e-3
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 200
    patience: int = 5
    min_delta: float = 1e-6
    max_seq_len: int = 20
    teacher_forcing: bool = False
    seed: int = 42
    save_dir: str = "results/das_real_real"


def run_das(config: DASConfig, verbose: bool = True) -> dict[str, list[float]]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = set_device()
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- frozen repeat model ---
    weights_path = Path(config.weights_path)
    if not weights_path.is_absolute():
        weights_path = REPO_ROOT / weights_path
    repeat_model = get_model(config.model_name)
    repeat_model.load_state_dict(torch.load(weights_path, map_location=device))
    repeat_model.to(device).eval()
    for p in repeat_model.parameters():
        p.requires_grad = False

    phoneme_to_id = get_phoneme_to_id()
    id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}
    pad_id = phoneme_to_id["<PAD>"]

    # --- real-real minimal-pair data ---
    df = get_train_dataset().copy()
    wfe_path = ROOT / "datasets/wfe_with_repetition.csv"
    if wfe_path.exists():
        wfe_df = pd.read_csv(wfe_path, converters={"Phonemes": literal_eval, "No_Stress": literal_eval})
        df = df[~df["Word"].isin(wfe_df["Word"])].copy()
    df["Length"] = df["No_Stress"].apply(len)
    n_variables = int(df["Length"].max())

    train_df, holdout_df = train_test_split(df, test_size=0.4, random_state=config.seed, shuffle=True)
    val_df, test_df = train_test_split(holdout_df, test_size=0.7, random_state=config.seed, shuffle=True)

    loaders = {
        name: create_real_real_dataloader(
            part, "No_Stress", phoneme_to_id,
            batch_size=config.batch_size, shuffle=(name == "train"), max_len=config.max_seq_len,
        )
        for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]
    }
    if verbose:
        for name, loader in loaders.items():
            print(f"{name}_loader: {len(loader)} batches")

    # --- DAS intervention (rotation + optional alignment) ---
    state_dim = config.hidden_size * (2 if config.state_mode == "concat" else 1)
    intervention = DASIntervention(
        state_dim=state_dim, n_variables=n_variables, var_size=config.var_size,
        masking=config.masking, reg_loss=config.reg_loss, state_mode=config.state_mode,
    ).to(device)

    optimizer = torch.optim.Adam(intervention.parameters(), lr=config.learning_rate)
    trainer = DASTrainer(repeat_model, intervention, optimizer, device, pad_id, config.teacher_forcing)

    print(
        f"DAS: state_dim={state_dim} n_variables={n_variables} var_size={intervention.var_size} "
        f"masking={config.masking} | trainable params="
        f"{sum(p.numel() for p in intervention.parameters() if p.requires_grad)}"
    )

    history = trainer.fit(
        loaders["train"], loaders["val"], num_epochs=config.num_epochs,
        patience=config.patience, min_delta=config.min_delta,
        test_loader=loaders["test"], verbose=verbose,
    )

    test_loss, test_acc = trainer.evaluate(loaders["test"])
    print(f"Final test: loss={test_loss:.4f} acc={test_acc:.4f}")

    # --- save ---
    torch.save(intervention.state_dict(), save_dir / "das_intervention.pth")
    np.savez(save_dir / "das_params.npz", **intervention.get_parameters())
    trainer.evaluate_with_predictions(loaders["test"], id_to_phoneme).to_csv(save_dir / "predictions.csv", index=False)
    pd.DataFrame(history).to_csv(save_dir / "history.csv", index=False)
    print(f"Saved to {save_dir}")
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DAS real-real intervention")
    parser.add_argument("--state_mode", default="concat", choices=["concat", "h", "c", "both"])
    parser.add_argument("--var_size", type=int, default=None)
    parser.add_argument("--masking", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", default="results/das_real_real")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_das(DASConfig(
        state_mode=args.state_mode, var_size=args.var_size, masking=args.masking,
        num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
        save_dir=args.save_dir,
    ))
