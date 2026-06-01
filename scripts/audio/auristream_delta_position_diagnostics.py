#!/usr/bin/env python3
"""Position-specific diagnostics for the block_47 → block_48 residual delta.

Uses existing saved embeddings and metadata. Does NOT re-run extraction.

Task A — phoneme-pooled level:
    Loads embeddings_block_47.pt / embeddings_block_48.pt, computes
    delta = block_48 - block_47, groups by phoneme_position_from_start,
    phoneme_type, and absolute-token-midpoint bins.

Task B — token level:
    Loads all raw token-level activations from activations/, computes
    token-level delta, groups by absolute token position bins.

Groups (Task A):
    phoneme_position_from_start : 0, 1, 2, 3, 4, 5+ (position within item)
    phoneme_type                : consonant / vowel
    token_midpoint_bin          : quantile-based early / mid / late

Groups (Task B):
    absolute_token_position bin : fixed-width or quantile bins across all items

For each group and the full dataset, the same metrics are computed:
    n, delta_norm (mean/std), isotropy_ratio, cos(delta_i, group_mean_delta),
    pairwise cos(delta_i, delta_j), cos(delta_i, block_47_i), cos(delta_i, block_48_i),
    cos(delta_i, global_mean_delta), cos(group_mean_delta, global_mean_delta).

This allows testing whether the near-common residual update is uniform across phoneme
positions or depends on causal context / position-in-item.

Usage:
    python scripts/audio/auristream_delta_position_diagnostics.py \\
        --run reproduce/data/audio/auristream__6ee9aeb6 \\
        --overwrite

Output:
    reproduce/figures/audio/auristream_delta_position/auristream__6ee9aeb6/
        last_block_delta_by_phoneme_position.csv
        last_block_delta_by_phoneme_type.csv
        last_block_delta_by_absolute_position.csv  (pooled-level; token_midpoint bin)
        token_delta_by_absolute_position.csv       (token-level)
        summary_delta_by_phoneme_position.png
        summary_delta_group_mean_alignment.png
        summary_token_delta_by_absolute_position.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_ARPABET_VOWELS: frozenset[str] = frozenset({
    "AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY", "IH", "IY",
    "OW", "OY", "UH", "UW",
})
_DPI = 150


def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def _phoneme_type(base: str) -> str:
    if base in _ARPABET_VOWELS:
        return "vowel"
    return "consonant" if base.isalpha() else "other"


# ── Core per-group metrics ─────────────────────────────────────────────────────

def _group_metrics(
    delta:          np.ndarray,   # [n, D]  residual vectors
    ref47:          np.ndarray,   # [n, D]  block_47
    ref48:          np.ndarray,   # [n, D]  block_48
    global_dir:     np.ndarray,   # [D]     global mean delta direction (L2-normalised)
    rng:            np.random.Generator,
    n_cos:          int = 300,
) -> dict:
    """Compute all delta metrics for one group."""
    n = delta.shape[0]
    norms_d  = np.linalg.norm(delta, axis=1)
    norms_47 = np.linalg.norm(ref47,  axis=1)
    norms_48 = np.linalg.norm(ref48,  axis=1)

    delta_n = (delta / (norms_d[:, None]  + 1e-12)).astype(np.float32)
    ref47_n = (ref47  / (norms_47[:, None] + 1e-12)).astype(np.float32)
    ref48_n = (ref48  / (norms_48[:, None] + 1e-12)).astype(np.float32)

    mean_d      = delta.mean(axis=0)
    mean_d_norm = float(np.linalg.norm(mean_d))
    isotropy    = mean_d_norm / (float(norms_d.mean()) + 1e-12)
    group_dir   = (mean_d / (mean_d_norm + 1e-12)).astype(np.float32)

    cos_dm   = delta_n @ group_dir     # [n] cosine with group mean direction
    cos_dg   = delta_n @ global_dir    # [n] cosine with global mean direction
    cos_da   = (delta_n * ref47_n).sum(axis=1)  # [n]
    cos_db   = (delta_n * ref48_n).sum(axis=1)  # [n]

    n_s  = min(n_cos, n)
    idx  = rng.choice(n, n_s, replace=False)
    cp   = (delta_n[idx] @ delta_n[idx].T)[np.triu_indices(n_s, k=1)]

    cos_group_vs_global = float(group_dir @ global_dir)

    return {
        "n": n,
        "delta_norm_mean":           float(norms_d.mean()),
        "delta_norm_std":            float(norms_d.std()),
        "isotropy_ratio":            float(isotropy),
        "cos_with_group_mean_mean":  float(cos_dm.mean()),
        "cos_with_group_mean_std":   float(cos_dm.std()),
        "cos_pairwise_mean":         float(cp.mean()),
        "cos_pairwise_q05":          float(np.percentile(cp, 5)),
        "cos_pairwise_q95":          float(np.percentile(cp, 95)),
        "cos_delta_block47_mean":    float(cos_da.mean()),
        "cos_delta_block48_mean":    float(cos_db.mean()),
        "cos_with_global_mean_mean": float(cos_dg.mean()),
        "cos_group_vs_global_mean":  float(cos_group_vs_global),
    }


def _sort_key(x: str) -> tuple:
    x = str(x)
    if x == "5+":
        return (0, 5)
    if x.lstrip("-").isdigit():
        return (0, int(x))
    if x.startswith("early"):
        return (1, 0)
    if x.startswith("mid"):
        return (1, 1)
    if x.startswith("late"):
        return (1, 2)
    if x == "consonant":
        return (2, 0)
    if x == "vowel":
        return (2, 1)
    return (9, 0)


def _group_df(
    delta:      np.ndarray,
    ref47:      np.ndarray,
    ref48:      np.ndarray,
    labels:     list[str],
    global_dir: np.ndarray,
    rng:        np.random.Generator,
    groupby:    str,
) -> pd.DataFrame:
    """Compute per-group metrics across a list of string labels."""
    label_arr = np.array(labels)
    unique_labels = sorted(set(labels), key=_sort_key)
    rows = []
    for g in unique_labels:
        mask = label_arr == g
        if mask.sum() < 2:
            continue
        m = _group_metrics(delta[mask], ref47[mask], ref48[mask], global_dir, rng)
        rows.append({"groupby": groupby, "group": g, **m})
    return pd.DataFrame(rows)


# ── Figures ────────────────────────────────────────────────────────────────────

def _bar_panel(
    df: pd.DataFrame,
    x_col: str,
    metrics: list[tuple[str, str]],
    title: str,
    output_path: Path,
) -> None:
    """Bar plots of several metrics vs a grouping column."""
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]
    groups = df[x_col].tolist()
    x = np.arange(len(groups))

    for ax, (col, ylabel) in zip(axes, metrics):
        ax.bar(x, df[col], color="#4575b4", alpha=0.85, edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(col, fontsize=9)
        ax.set_ylim(0, max(float(df[col].max()) * 1.15, 0.1))

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    _savefig(fig, output_path)


def _alignment_panel(
    dfs: list[pd.DataFrame],
    labels: list[str],
    output_path: Path,
) -> None:
    """Horizontal bar chart: cos(group_mean_delta, global_mean_delta) per grouping."""
    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, df, lbl in zip(axes, dfs, labels):
        groups = df["group"].tolist()
        vals   = df["cos_group_vs_global_mean"].tolist()
        y = np.arange(len(groups))
        ax.barh(y, vals, color="#d73027", alpha=0.85, edgecolor="none")
        ax.set_yticks(y)
        ax.set_yticklabels(groups, fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.axvline(1.0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("cos(group_mean_δ, global_mean_δ)", fontsize=9)
        ax.set_title(lbl, fontsize=10)

    fig.suptitle(
        "Alignment of group mean delta with global mean delta\n"
        "(1 = all groups point in same direction as global delta)", fontsize=10
    )
    fig.tight_layout()
    _savefig(fig, output_path)


# ── Task A: phoneme-pooled position diagnostics ────────────────────────────────

def run_pooled_diagnostics(
    emb_dir: Path,
    output_dir: Path,
    rng: np.random.Generator,
) -> None:
    print("\n[delta_pos] Task A — phoneme-pooled position diagnostics")

    # Load metadata
    meta = pd.read_csv(emb_dir / "metadata_phonemes.csv")
    meta["phoneme_base"] = meta["phoneme"].str.replace(r"\d+$", "", regex=True).str.upper()
    meta["phoneme_type"] = meta["phoneme_base"].map(_phoneme_type)

    # Load embeddings
    emb47_t = torch.load(emb_dir / "embeddings_block_47.pt", map_location="cpu", weights_only=True).float()
    emb48_t = torch.load(emb_dir / "embeddings_block_48.pt", map_location="cpu", weights_only=True).float()

    # Validity filter
    keep = (
        meta["valid_embedding"].astype(bool)
        & ~meta["is_silence"].astype(bool)
        & torch.isfinite(emb47_t).all(dim=1).numpy()
        & torch.isfinite(emb48_t).all(dim=1).numpy()
    )
    meta_v  = meta[keep].reset_index(drop=True)
    emb47   = emb47_t[keep].numpy().astype(np.float32)
    emb48   = emb48_t[keep].numpy().astype(np.float32)

    delta = emb48 - emb47
    n = len(meta_v)
    print(f"  n valid phonemes: {n}")

    # Global mean delta direction
    mean_delta      = delta.mean(axis=0)
    mean_delta_norm = float(np.linalg.norm(mean_delta))
    global_dir      = (mean_delta / (mean_delta_norm + 1e-12)).astype(np.float32)
    global_iso      = mean_delta_norm / (float(np.linalg.norm(delta, axis=1).mean()) + 1e-12)
    print(f"  global isotropy_ratio: {global_iso:.5f}")

    dfs_all: list[pd.DataFrame] = []
    align_dfs: list[pd.DataFrame] = []
    align_labels: list[str] = []

    # ── Group 1: phoneme_position_from_start ──────────────────────────────────
    raw_pos = meta_v["phoneme_position_from_start"].values
    pos_labels = [str(p) if p < 5 else "5+" for p in raw_pos]
    df_pos = _group_df(delta, emb47, emb48, pos_labels, global_dir, rng, "phoneme_position")
    df_pos.to_csv(output_dir / "last_block_delta_by_phoneme_position.csv", index=False)
    dfs_all.append(("phoneme_position", df_pos))
    align_dfs.append(df_pos)
    align_labels.append("phoneme_position_from_start")
    print(f"  saved last_block_delta_by_phoneme_position.csv  ({len(df_pos)} groups)")

    # ── Group 2: phoneme_type ─────────────────────────────────────────────────
    type_mask = meta_v["phoneme_type"].isin(["consonant", "vowel"])
    if type_mask.sum() > 0:
        df_type = _group_df(
            delta[type_mask.values], emb47[type_mask.values], emb48[type_mask.values],
            meta_v.loc[type_mask, "phoneme_type"].tolist(), global_dir, rng, "phoneme_type",
        )
        df_type.to_csv(output_dir / "last_block_delta_by_phoneme_type.csv", index=False)
        align_dfs.append(df_type)
        align_labels.append("phoneme_type")
        print(f"  saved last_block_delta_by_phoneme_type.csv  ({len(df_type)} groups)")

    # ── Group 3: absolute token midpoint bins ─────────────────────────────────
    tok_mid = ((meta_v["token_start_idx"] + meta_v["token_end_idx"]) / 2.0).values
    # Quantile bins: early / mid / late
    q33, q67 = float(np.percentile(tok_mid, 33)), float(np.percentile(tok_mid, 67))
    def _tok_bin(v: float) -> str:
        if v < q33:
            return f"early(tok<{q33:.0f})"
        if v < q67:
            return f"mid({q33:.0f}-{q67:.0f})"
        return f"late(tok≥{q67:.0f})"
    abs_labels = [_tok_bin(v) for v in tok_mid]
    df_abs = _group_df(delta, emb47, emb48, abs_labels, global_dir, rng, "token_midpoint_bin")
    df_abs.to_csv(output_dir / "last_block_delta_by_absolute_position.csv", index=False)
    align_dfs.append(df_abs)
    align_labels.append("absolute_token_position")
    print(f"  saved last_block_delta_by_absolute_position.csv  ({len(df_abs)} groups)")

    # ── Figure 1: metrics by phoneme position ─────────────────────────────────
    _bar_panel(
        df_pos, "group",
        [
            ("isotropy_ratio",          "Isotropy ratio"),
            ("cos_with_group_mean_mean","cos(δ, group_mean_δ)"),
            ("cos_pairwise_mean",        "Pairwise cos"),
            ("cos_with_global_mean_mean","cos(δ, global_mean_δ)"),
        ],
        "Delta diagnostics by phoneme_position_from_start",
        output_dir / "summary_delta_by_phoneme_position.png",
    )
    print("  saved summary_delta_by_phoneme_position.png")

    # ── Figure 2: group mean alignment ────────────────────────────────────────
    _alignment_panel(
        align_dfs, align_labels,
        output_dir / "summary_delta_group_mean_alignment.png",
    )
    print("  saved summary_delta_group_mean_alignment.png")


# ── Task B: token-level absolute position diagnostics ─────────────────────────

def run_token_diagnostics(
    run_dir: Path,
    manifest: dict,
    output_dir: Path,
    rng: np.random.Generator,
    n_pos_bins: int = 5,
) -> None:
    print("\n[delta_pos] Task B — token-level absolute position diagnostics")

    all_delta_list: list[np.ndarray] = []
    all_pos_list:   list[np.ndarray] = []
    all_rel_pos_list: list[np.ndarray] = []

    items_done = 0
    for item_id, paths in manifest["items"].items():
        if "block_47" not in paths or "block_48" not in paths:
            continue
        h47 = torch.load(run_dir / paths["block_47"], map_location="cpu", weights_only=True).float()
        h48 = torch.load(run_dir / paths["block_48"], map_location="cpu", weights_only=True).float()
        if h47.shape != h48.shape:
            print(f"  [warning] {item_id}: shape mismatch — skipping")
            continue
        L = h47.shape[1]
        d = (h48 - h47).T.numpy().astype(np.float32)   # [L, D]
        abs_pos = np.arange(L, dtype=np.float32)
        rel_pos = abs_pos / max(L - 1, 1)

        all_delta_list.append(d)
        all_pos_list.append(abs_pos)
        all_rel_pos_list.append(rel_pos)
        items_done += 1

    if not all_delta_list:
        print("  [warning] no token-level activations found — skipping Task B")
        return

    delta_all = np.concatenate(all_delta_list, axis=0)    # [N_total, D]
    pos_all   = np.concatenate(all_pos_list,   axis=0)    # [N_total]
    rel_all   = np.concatenate(all_rel_pos_list, axis=0)  # [N_total]
    N = len(delta_all)
    print(f"  N total tokens: {N}  (from {items_done} items)")
    print(f"  abs_pos range: [{int(pos_all.min())}, {int(pos_all.max())}]")

    # Global direction
    mean_d     = delta_all.mean(axis=0)
    mean_d_n   = float(np.linalg.norm(mean_d))
    global_dir = (mean_d / (mean_d_n + 1e-12)).astype(np.float32)
    global_iso = mean_d_n / (float(np.linalg.norm(delta_all, axis=1).mean()) + 1e-12)
    print(f"  global isotropy_ratio (token level): {global_iso:.5f}")

    # Fake ref47/ref48 (not loaded for memory; pass zeros → cos metrics will be 0)
    # Load actual refs for a meaningful cross-layer cosine
    # We'll compute isotropy, cos_pairwise, and cos_with_global_mean only (no ref needed)
    zeros = np.zeros_like(delta_all)

    # Quantile bins on absolute position
    bin_edges = np.unique(
        np.percentile(pos_all, np.linspace(0, 100, n_pos_bins + 1)).astype(int)
    )
    if len(bin_edges) < 2:
        bin_edges = np.array([int(pos_all.min()), int(pos_all.max()) + 1])

    def _pos_bin(p: float) -> str:
        for i in range(len(bin_edges) - 1):
            if p <= bin_edges[i + 1]:
                return f"{bin_edges[i]}-{bin_edges[i+1]}"
        return f"{bin_edges[-2]}-{bin_edges[-1]}"

    abs_labels = [_pos_bin(p) for p in pos_all]
    df_tok = _group_df(delta_all, zeros, zeros, abs_labels, global_dir, rng, "abs_token_position")
    # Remove ref-dependent columns (all-zero refs make those metrics meaningless)
    df_tok = df_tok.drop(columns=["cos_delta_block47_mean", "cos_delta_block48_mean"],
                         errors="ignore")
    df_tok.to_csv(output_dir / "token_delta_by_absolute_position.csv", index=False)
    print(f"  saved token_delta_by_absolute_position.csv  ({len(df_tok)} groups)")

    _bar_panel(
        df_tok, "group",
        [
            ("isotropy_ratio",          "Isotropy ratio"),
            ("cos_with_group_mean_mean","cos(δ, group_mean_δ)"),
            ("cos_pairwise_mean",        "Pairwise cos"),
            ("cos_with_global_mean_mean","cos(δ, global_mean_δ)"),
        ],
        f"Token-level delta diagnostics by absolute position ({items_done} items)",
        output_dir / "summary_token_delta_by_absolute_position.png",
    )
    print("  saved summary_token_delta_by_absolute_position.png")


# ── CLI + main ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", required=True,
                   help="Extraction run directory, e.g. reproduce/data/audio/auristream__6ee9aeb6")
    p.add_argument("--embeddings-dir", default=None, dest="embeddings_dir",
                   help="Phoneme embeddings directory (default: {run}/phoneme_embeddings/)")
    p.add_argument("--n-pos-bins", type=int, default=5, dest="n_pos_bins",
                   help="Number of absolute token position bins for Task B (default: 5)")
    p.add_argument("--random-state", type=int, default=0, dest="random_state")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--output", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = (REPO_ROOT / args.run).resolve()
    emb_dir = (
        Path(args.embeddings_dir).resolve() if args.embeddings_dir
        else run_dir / "phoneme_embeddings"
    )

    with open(run_dir / "manifest.json") as f:
        manifest = json.load(f)
    run_id = manifest["run_id"]

    output_dir = (
        Path(args.output).resolve() if args.output
        else REPO_ROOT / "reproduce" / "figures" / "audio"
             / "auristream_delta_position" / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    done_path = output_dir / "last_block_delta_by_phoneme_position.csv"
    if not args.overwrite and done_path.exists():
        print(f"[delta_pos] Output exists: {output_dir}\nPass --overwrite to regenerate.")
        return

    for required in ("block_47", "block_48"):
        if required not in manifest["run_params"]["layers"]:
            print(f"[delta_pos] {required!r} not in run manifest — cannot run.")
            sys.exit(1)

    rng = np.random.default_rng(args.random_state)

    run_pooled_diagnostics(emb_dir, output_dir, rng)
    run_token_diagnostics(run_dir, manifest, output_dir, rng, n_pos_bins=args.n_pos_bins)

    print(f"\n[delta_pos] Done — {output_dir}")


if __name__ == "__main__":
    main()
