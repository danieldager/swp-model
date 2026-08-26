"""Untrained control: does the delta-norm decline need training, or is it just the LSTM?

``surprisal.py`` rules out one confound. This rules out the plainer one: an LSTM's
forget gate contracts and its tanh saturates, so state updates shrink with depth
whether or not anything was learned. The same architecture is run at random
initialisation over the same words, and the two curves are put side by side.

Shapes, not magnitudes, are the comparison -- random weights land on an arbitrary
scale -- so every curve also appears normalised to its own position-1 value. Several
seeds, because one draw of random weights could be idiosyncratic.

States are collected by unrolling the encoder step by step over a padded batch, which
is equivalent to ``StateExtractor.extract_sequential`` (an LSTM from a zero state gives
the same (h, c) whether the prefix is re-fed or the state is carried) but O(L) per word
rather than O(L^2), so all 30k words take seconds instead of minutes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from intervention.models.repeat_model import get_model
from intervention.paths import get_phoneme_to_id, get_train_dataset, resolve_weights
from intervention.utils import seed_everything, set_device

MODEL_NAME = "Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1"
WEIGHTS = "resources/weights/1024_75.pth"
TARGETS = ["delta_h", "delta_c", "delta_state"]
SEEDS = [0, 1, 2]
MAX_POS = 12  # past this the per-position n is in the double digits and the curve is noise
PLOTS_DIR = Path(__file__).resolve().parents[1] / "plots" / "random_baseline"


@torch.no_grad()
def delta_norms(model, seqs, phoneme_to_id, device, batch_size: int = 512) -> pd.DataFrame:
    """||delta|| at every step of every sequence, long format (target, position, norm)."""
    pad_id = phoneme_to_id["<PAD>"]
    frames = []
    for start in range(0, len(seqs), batch_size):
        batch = seqs[start : start + batch_size]
        lengths = np.array([len(s) for s in batch])
        width = lengths.max()

        ids = torch.full((len(batch), width), pad_id, dtype=torch.long, device=device)
        for i, seq in enumerate(batch):
            ids[i, : len(seq)] = torch.tensor([phoneme_to_id[p] for p in seq], device=device)

        embedded = model.encoder.embedding(ids)
        state, hs, cs = None, [], []
        for t in range(width):
            _, state = model.encoder.recurrent(embedded[:, t : t + 1], state)
            hs.append(state[0][-1])
            cs.append(state[1][-1])
        h, c = torch.stack(hs, dim=1), torch.stack(cs, dim=1)  # (B, width, hidden)

        keep = np.arange(width)[None, :] < lengths[:, None]
        positions = np.broadcast_to(np.arange(width), keep.shape)[keep]
        for name, arr in (("delta_h", h), ("delta_c", c),
                          ("delta_state", torch.cat([h, c], dim=-1))):
            # delta[0] = state[0], matching the convention in states_extract
            delta = torch.cat([arr[:, :1], arr.diff(dim=1)], dim=1)
            norms = delta.norm(dim=-1).cpu().numpy()
            frames.append(pd.DataFrame({"target": name, "position": positions,
                                        "norm": norms[keep]}))
    return pd.concat(frames, ignore_index=True)


def build_model(device, weights: str | None = None, seed: int | None = None):
    """The trained model, or a fresh random-initialised one at ``seed``."""
    if seed is not None:
        seed_everything(seed)
    model = get_model(MODEL_NAME)
    if weights is not None:
        model.load_state_dict(
            torch.load(resolve_weights(weights), map_location=device, weights_only=True)
        )
    return model.to(device).eval()


def profiles(seqs, phoneme_to_id, device) -> pd.DataFrame:
    """Mean norm by position for the trained model and for each random seed."""
    runs = {"trained": build_model(device, weights=WEIGHTS)}
    runs |= {f"random s{s}": build_model(device, seed=s) for s in SEEDS}

    out = []
    for label, model in runs.items():
        df = delta_norms(model, seqs, phoneme_to_id, device)
        out.append(df.groupby(["target", "position"], as_index=False)["norm"].mean()
                     .assign(model=label))
        print(f"  {label}: done")
    return pd.concat(out, ignore_index=True)


def plot_comparison(prof: pd.DataFrame, path: Path):
    """Raw norms on the top row, each curve rescaled to its own position 1 below."""
    prof = prof[(prof["position"] > 0) & (prof["position"] <= MAX_POS)]
    fig, axes = plt.subplots(2, len(TARGETS), figsize=(15, 8), sharex=True)

    for col, target in enumerate(TARGETS):
        for row, rescale in enumerate([False, True]):
            ax = axes[row, col]
            for label, grp in prof[prof["target"] == target].groupby("model", sort=False):
                y = grp.set_index("position")["norm"]
                if rescale:
                    y = y / y.iloc[0]
                is_trained = label == "trained"
                ax.plot(y.index, y.values, label=label,
                        color="#1565C0" if is_trained else "#9E9E9E",
                        lw=2.8 if is_trained else 1.3,
                        marker="o" if is_trained else None,
                        markersize=7, zorder=3 if is_trained else 1)
            ax.set_title(target if row == 0 else "")
            ax.set_ylabel(("||delta||" if row == 0 else "relative to position 1")
                          if col == 0 else "")
            if row == 1:
                ax.set_xlabel("position")
                ax.axhline(1.0, color="k", lw=0.6, ls=":")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Trained vs randomly initialised encoder: delta norms across position")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    device = set_device()
    phoneme_to_id = get_phoneme_to_id()
    seqs = [list(p) + ["<EOS>"] for p in get_train_dataset()["No_Stress"]]
    print(f"{len(seqs)} words, {sum(len(s) for s in seqs)} phoneme tokens")

    prof = profiles(seqs, phoneme_to_id, device)
    prof.to_csv(PLOTS_DIR / "profiles.csv", index=False)
    plot_comparison(prof, PLOTS_DIR / "trained_vs_random.png")

    # Headline number: how much of the decline is already there before any training?
    kept = prof[(prof["position"] >= 1) & (prof["position"] <= MAX_POS)]
    decay = kept.pivot_table(index=["target", "model"], columns="position", values="norm")
    decay = (decay[MAX_POS] / decay[1]).rename(f"norm[{MAX_POS}] / norm[1]")
    print("\n=== decay ratio ===")
    print(decay.round(3).to_string())
    print(f"\nplots + profiles.csv -> {PLOTS_DIR}")
