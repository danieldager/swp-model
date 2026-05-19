#!/usr/bin/env python3
"""PC1 diagnostics for AuriStream block_48 phoneme embeddings.

Investigates why block_48 PC1 explains ~83 % of variance (vs ~11–13 % for
other layers) by correlating PC1 scores with norm, position, duration, and
n_tokens, and by re-running PCA on L2-normalised embeddings as a comparison.

This script works on any layer (--layer), but is primarily intended for block_48.
It generalises straightforwardly to phoneme embeddings from other
representation-only speech models that produce the same output format.

Inputs (all produced by scripts/audio/auristream_phoneme_pca_norm.py +
        scripts/audio/build_phoneme_embeddings.py):
  --figures-dir   analysis output directory
                  (e.g. reproduce/figures/audio/auristream_phonemes/auristream__9d3f269f/)
  --embeddings-dir phoneme embeddings directory
                  (e.g. reproduce/data/audio/auristream__9d3f269f/phoneme_embeddings/)

Run from swp-model/ repo root:

    python scripts/audio/auristream_block48_diagnostics.py \\
        --figures-dir   reproduce/figures/audio/auristream_phonemes/auristream__9d3f269f/ \\
        --embeddings-dir reproduce/data/audio/auristream__9d3f269f/phoneme_embeddings/

Outputs (inside {figures-dir}/{layer}/):
    {layer}_pc1_diagnostics.csv
    {layer}_pc1_correlations.csv
    {layer}_normalized_pca_comparison.json
    {layer}_pc1_vs_norm.png
    {layer}_pc1_vs_position_start.png
    {layer}_pc1_vs_n_tokens.png
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
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _pearson(a: np.ndarray, b: np.ndarray) -> dict:
    mask = np.isfinite(a) & np.isfinite(b)
    r, p = stats.pearsonr(a[mask], b[mask])
    rho, p_sp = stats.spearmanr(a[mask], b[mask])
    return {"pearson_r": round(float(r), 4), "pearson_p": float(p),
            "spearman_r": round(float(rho), 4), "spearman_p": float(p_sp),
            "n": int(mask.sum())}


def run_diagnostics(
    figures_dir: Path,
    embeddings_dir: Path,
    layer: str,
    overwrite: bool,
) -> None:
    layer_dir = figures_dir / layer
    if not layer_dir.exists():
        sys.exit(f"Layer directory not found: {layer_dir}")

    # ── Load scores and norms CSVs ────────────────────────────────────────────
    scores_csv = layer_dir / f"phoneme_pca_scores_{layer}.csv"
    norms_csv  = layer_dir / f"phoneme_norms_{layer}.csv"
    for p in (scores_csv, norms_csv):
        if not p.exists():
            sys.exit(f"Required CSV not found: {p}\nRun auristream_phoneme_pca_norm.py first.")

    scores_df = pd.read_csv(scores_csv)
    norms_df  = pd.read_csv(norms_csv)

    # Merge on item_id + phoneme_index (row-aligned, but merge is safer)
    diag = scores_df.merge(
        norms_df[["item_id", "phoneme_index", "norm"]],
        on=["item_id", "phoneme_index"],
        how="inner",
    )
    assert len(diag) == len(scores_df), "Merge lost rows — check CSV alignment."

    # ── A: diagnostics CSV ───────────────────────────────────────────────────
    keep = [
        "item_id", "phoneme", "phoneme_position_from_start",
        "phoneme_position_from_end", "duration_s", "n_tokens", "norm",
        "pc1", "pc2",
    ]
    for col in ("lexicality", "length_bin"):
        if col in diag.columns:
            keep.append(col)
    diag_csv = layer_dir / f"{layer}_pc1_diagnostics.csv"
    diag[keep].to_csv(diag_csv, index=False)
    print(f"  Saved: {diag_csv.name}")

    # ── B: correlations ──────────────────────────────────────────────────────
    pc1 = diag["pc1"].values
    targets = {
        "norm":                          diag["norm"].values,
        "phoneme_position_from_start":   diag["phoneme_position_from_start"].values,
        "phoneme_position_from_end":     diag["phoneme_position_from_end"].values,
        "duration_s":                    diag["duration_s"].values,
        "n_tokens":                      diag["n_tokens"].values,
    }
    corr_rows = []
    for name, vals in targets.items():
        row = {"variable": name}
        row.update(_pearson(pc1, vals.astype(float)))
        corr_rows.append(row)
        print(f"    pc1 ~ {name:<35} r={row['pearson_r']:+.3f}  ρ={row['spearman_r']:+.3f}")

    corr_csv = layer_dir / f"{layer}_pc1_correlations.csv"
    pd.DataFrame(corr_rows).to_csv(corr_csv, index=False)
    print(f"  Saved: {corr_csv.name}")

    # ── C: scatter plots ─────────────────────────────────────────────────────
    pos = diag["phoneme_position_from_start"].values
    cmap = matplotlib.colormaps["viridis"].resampled(int(pos.max()) + 1)

    def _scatter(x, y, xlabel, ylabel, title, fname):
        fig, ax = plt.subplots(figsize=(5, 4))
        sc = ax.scatter(x, y, c=pos, cmap="viridis", s=8, alpha=0.55)
        plt.colorbar(sc, ax=ax, label="position (from start)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(layer_dir / fname, dpi=120)
        plt.close(fig)
        print(f"  Saved: {fname}")

    _scatter(diag["norm"].values, pc1,
             "L2 norm", "PC1",
             f"{layer}: PC1 vs norm", f"{layer}_pc1_vs_norm.png")

    jitter = np.random.default_rng(0).uniform(-0.2, 0.2, len(pc1))
    _scatter(pos + jitter, pc1,
             "phoneme position (from start, jittered)", "PC1",
             f"{layer}: PC1 vs position", f"{layer}_pc1_vs_position_start.png")

    _scatter(diag["n_tokens"].values.astype(float), pc1,
             "n_tokens", "PC1",
             f"{layer}: PC1 vs n_tokens", f"{layer}_pc1_vs_n_tokens.png")

    # ── D: PCA on L2-normalised embeddings ───────────────────────────────────
    emb_path = embeddings_dir / f"embeddings_{layer}.pt"
    if not emb_path.exists():
        print(f"  Skipping normalised PCA — {emb_path} not found.")
        return

    X = torch.load(emb_path, map_location="cpu", weights_only=True).numpy()
    # Align to the filtered rows used in the CSVs (all valid in this run → direct use)
    if X.shape[0] != len(diag):
        print(
            f"  Warning: embeddings have {X.shape[0]} rows but diagnostics CSV has "
            f"{len(diag)}. Skipping normalised PCA."
        )
        return

    # Original PCA (StandardScaler)
    ev_orig = diag[["pc1", "pc2"]].var(axis=0)   # just to read back; re-fit for accuracy
    scaler = StandardScaler()
    pca_orig = PCA(n_components=2, random_state=0)
    pca_orig.fit(scaler.fit_transform(X))
    ev_before = pca_orig.explained_variance_ratio_.tolist()

    # L2-normalise each row then StandardScaler + PCA
    norms_arr = np.linalg.norm(X, axis=1, keepdims=True)
    norms_arr = np.where(norms_arr == 0, 1.0, norms_arr)   # avoid /0
    X_l2 = X / norms_arr
    scaler2 = StandardScaler()
    pca_norm = PCA(n_components=2, random_state=0)
    pca_norm.fit(scaler2.fit_transform(X_l2))
    ev_after = pca_norm.explained_variance_ratio_.tolist()

    comparison = {
        "layer": layer,
        "before_l2_normalisation": {
            "pc1_explained_variance": round(ev_before[0], 4),
            "pc2_explained_variance": round(ev_before[1], 4),
        },
        "after_l2_normalisation": {
            "pc1_explained_variance": round(ev_after[0], 4),
            "pc2_explained_variance": round(ev_after[1], 4),
        },
        "interpretation": (
            "If PC1 drops substantially after L2 normalisation, the dominant PC1 "
            "primarily reflects embedding magnitude rather than direction."
        ),
    }
    comp_json = layer_dir / f"{layer}_normalized_pca_comparison.json"
    with open(comp_json, "w") as f:
        json.dump(comparison, f, indent=2)
    print(
        f"  Saved: {comp_json.name}\n"
        f"    Before normalisation: PC1={ev_before[0]*100:.1f}%  PC2={ev_before[1]*100:.1f}%\n"
        f"    After  normalisation: PC1={ev_after[0]*100:.1f}%  PC2={ev_after[1]*100:.1f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PC1 diagnostics for a specific AuriStream phoneme embedding layer."
    )
    parser.add_argument(
        "--figures-dir", required=True, dest="figures_dir",
        help=(
            "Analysis output directory "
            "(e.g. reproduce/figures/audio/auristream_phonemes/auristream__9d3f269f/)"
        ),
    )
    parser.add_argument(
        "--embeddings-dir", required=True, dest="embeddings_dir",
        help=(
            "Phoneme embeddings directory "
            "(e.g. reproduce/data/audio/auristream__9d3f269f/phoneme_embeddings/)"
        ),
    )
    parser.add_argument(
        "--layer", default="block_48",
        help="Layer to diagnose (default: block_48).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output files.",
    )
    args = parser.parse_args()

    figures_dir   = Path(args.figures_dir)
    embeddings_dir = Path(args.embeddings_dir)

    for p in (figures_dir, embeddings_dir):
        if not p.exists():
            sys.exit(f"Path not found: {p}")

    print(f"Layer        : {args.layer}")
    print(f"Figures dir  : {figures_dir}")
    print(f"Embeddings   : {embeddings_dir}")

    run_diagnostics(figures_dir, embeddings_dir, args.layer, args.overwrite)
    print("\nDone.")


if __name__ == "__main__":
    main()
