#!/usr/bin/env python3
"""PCA visualization of audio codec hidden states — one point per word.

For each (run_id, layer) combination, loads per-item activation tensors,
mean-pools over the time dimension to produce one [D]-dimensional vector
per item, then fits a 2-component PCA and saves scatter plots and a tidy
summary CSV.

Supported layers: encoder_out, decoder_in (both are 2D [C, T] tensors).
decoder_out is skipped (waveform-like shape, not meaningful for PCA here).

Method
------
For each item:
    1. Load tensor [C, T] from the extraction run.
    2. Mean over T  →  one vector [C]  (pooled representation).
Build matrix [n_items, C], standardize features (StandardScaler),
fit PCA(n_components=2).  One PCA is fit per (run_id, layer).

Tensor shape assumption
-----------------------
Activations saved by scripts/audio/extract.py have shape [C, T] for
encoder_out and decoder_in (C = number of channels, T = frame count).
This is the convention of both the EnCodec and DAC wrappers in this repo.

Output (per run, per layer)
---------------------------
reproduce/figures/audio/pca/<run_id>/<layer>/
    point_per_word_mean.csv            — tidy CSV: item_id, layer, pc1, pc2,
                                         lexicality, length_bin, morphology,
                                         frequency_bin, n_frames
    point_per_word_mean_summary.json   — explained variance ratio and item counts
    point_per_word_mean.png            — all items, color=lexicality, marker=length_bin
    point_per_word_mean_short_only.png — short items only, color=lexicality
    point_per_word_mean_long_only.png  — long items only, color=lexicality

Usage
-----
    python reproduce/scripts/audio/pca_analysis.py \\
        --run reproduce/data/audio/encodec__7f7d3b97/

    # Custom output root
    python reproduce/scripts/audio/pca_analysis.py \\
        --run reproduce/data/audio/encodec__7f7d3b97/ \\
        --output /tmp/pca_figures/

    # Restrict to a single layer
    python reproduce/scripts/audio/pca_analysis.py \\
        --run reproduce/data/audio/encodec__7f7d3b97/ \\
        --layers encoder_out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch

# Resolve repo root: reproduce/scripts/audio/ → 3 levels up → swp-model/
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Layers supported by this script (must be 2D [C, T] with C > 1)
SUPPORTED_LAYERS: list[str] = ["encoder_out", "decoder_in"]

FACTOR_COLS: list[str] = ["lexicality", "length_bin", "morphology", "frequency_bin"]
META_COLS: list[str] = ["speaker", "duration_s"]

# Colorblind-safe palette (matches seaborn "colorblind" used elsewhere)
_LEX_COLORS: dict[str, str] = {"word": "#0072B2", "nonword": "#E69F00"}
_LEN_MARKERS: dict[str, str] = {"short": "o", "long": "^"}


# ── Data structure ────────────────────────────────────────────────────────────


class LayerPCAResult(NamedTuple):
    """Everything produced by one (run_id, layer) PCA run."""

    df: pd.DataFrame           # Tidy row-per-item DataFrame
    explained_variance: list[float]  # [ev_pc1, ev_pc2]
    n_items: int


# ── Data loading ──────────────────────────────────────────────────────────────


def load_manifest(run_dir: Path) -> dict:
    """Load and validate manifest.json from a run directory.

    Args:
        run_dir: absolute path to the extraction run directory.

    Returns:
        Parsed manifest dictionary.

    Raises:
        FileNotFoundError: if manifest.json is absent.
        ValueError: if required top-level keys are missing.
    """
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found in {run_dir}. "
            "Expected a run directory produced by scripts/audio/extract.py."
        )
    with open(manifest_path) as f:
        manifest = json.load(f)
    required_keys = {"run_id", "run_params", "items"}
    missing = required_keys - manifest.keys()
    if missing:
        raise ValueError(f"manifest.json is missing keys: {missing}")
    return manifest


def load_paradigm_factors(manifest: dict, repo_root: Path) -> pd.DataFrame:
    """Load factor columns from the dataset CSV referenced in the manifest.

    Args:
        manifest:  parsed manifest.json
        repo_root: repository root used to resolve the relative dataset_path

    Returns:
        DataFrame with columns [item_id, *FACTOR_COLS, <available META_COLS>].

    Raises:
        FileNotFoundError: if the dataset CSV cannot be located.
        ValueError: if required factor columns are absent from the CSV.
    """
    dataset_rel = manifest["run_params"]["dataset_path"]
    dataset_abs = (repo_root / dataset_rel).resolve()
    if not dataset_abs.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found: {dataset_abs}\n"
            f"(recorded in manifest as '{dataset_rel}')"
        )
    df = pd.read_csv(dataset_abs)
    required = ["item_id", *FACTOR_COLS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset CSV is missing required columns: {missing}\n"
            f"Found: {list(df.columns)}"
        )
    keep = ["item_id", *FACTOR_COLS, *[c for c in META_COLS if c in df.columns]]
    return df[keep]


# ── PCA computation ───────────────────────────────────────────────────────────


def _is_waveform(t: torch.Tensor) -> bool:
    """Return True if tensor is waveform-like: 1D [T] or 2D [1, T]."""
    return t.dim() == 1 or (t.dim() == 2 and t.shape[0] == 1)


def _load_mean_activations(
    manifest: dict,
    run_dir: Path,
    layer: str,
) -> tuple[list[str], np.ndarray, list[int]]:
    """Load activations for all items at one layer and mean-pool over time.

    For each item the [C, T] tensor is averaged across T, yielding one
    representative vector [C] per item.  Waveform-like tensors are skipped.

    Args:
        manifest: parsed manifest.json
        run_dir:  absolute path to run directory
        layer:    layer name (e.g. "encoder_out")

    Returns:
        item_ids  — list of item IDs with valid tensors (length n)
        matrix    — float32 array [n, C] of mean-pooled activations
        n_frames  — list of T values (frame counts) per item (length n)

    Raises:
        RuntimeError: if no valid 2D tensors are found for the layer.
    """
    items = manifest["items"]
    item_ids: list[str] = []
    vectors: list[np.ndarray] = []
    n_frames: list[int] = []
    skipped = 0

    for item_id, layer_paths in items.items():
        if layer not in layer_paths:
            continue

        pt_path = run_dir / layer_paths[layer]
        if not pt_path.exists():
            raise FileNotFoundError(
                f"Activation file not found: {pt_path}\n"
                f"(manifest entry: item={item_id}, layer={layer})"
            )

        t = torch.load(pt_path, map_location="cpu").float()

        if _is_waveform(t):
            print(
                f"[pca_analysis] WARNING: {item_id}/{layer} is waveform-like "
                f"(shape {tuple(t.shape)}) — skipped."
            )
            skipped += 1
            continue

        # Tensor shape: [C, T].  Mean over dim=1 (time) → [C].
        C, T = t.shape
        mean_vec = t.mean(dim=1).numpy()  # [C]
        item_ids.append(item_id)
        vectors.append(mean_vec)
        n_frames.append(T)

    if skipped:
        print(f"[pca_analysis] Skipped {skipped} waveform-like tensor(s) for layer={layer}.")

    if not vectors:
        raise RuntimeError(
            f"No valid 2D tensors found for layer={layer}. "
            "Verify that this layer is present in the run."
        )

    return item_ids, np.stack(vectors, axis=0).astype(np.float32), n_frames


def run_pca(
    manifest: dict,
    run_dir: Path,
    paradigm_df: pd.DataFrame,
    layer: str,
) -> LayerPCAResult:
    """Fit a 2-component PCA for one (run_id, layer) pair.

    Steps:
        1. Load and mean-pool activations → [n_items, C] matrix.
        2. Standardize features with StandardScaler.
        3. Fit PCA(n_components=2, random_state=0).
        4. Merge paradigm factor columns.

    Args:
        manifest:    parsed manifest.json
        run_dir:     absolute path to run directory
        paradigm_df: DataFrame with item_id and factor columns
        layer:       layer name to process

    Returns:
        LayerPCAResult with tidy DataFrame, explained variance, and item count.

    Raises:
        ValueError: if any item_id fails to match the paradigm DataFrame.
    """
    n_total = len(manifest["items"])
    print(f"[pca_analysis] Layer={layer} — loading {n_total} item activations …")

    item_ids, matrix, n_frames = _load_mean_activations(manifest, run_dir, layer)
    n_items = len(item_ids)

    print(
        f"[pca_analysis] Layer={layer} — matrix shape: {matrix.shape}  "
        f"(n_items={n_items}, n_dims={matrix.shape[1]})"
    )

    # Standardize then PCA
    matrix_scaled = StandardScaler().fit_transform(matrix)
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(matrix_scaled)  # [n_items, 2]
    ev = pca.explained_variance_ratio_.tolist()

    print(
        f"[pca_analysis] Layer={layer} — explained variance: "
        f"PC1={ev[0]:.3f}  PC2={ev[1]:.3f}  total={sum(ev):.3f}"
    )

    # Assemble tidy DataFrame and merge factor columns
    result_df = pd.DataFrame({
        "item_id": item_ids,
        "layer": layer,
        "pc1": coords[:, 0],
        "pc2": coords[:, 1],
        "n_frames": n_frames,
    })
    merged = result_df.merge(paradigm_df, on="item_id", how="left")

    n_unmatched = merged["lexicality"].isna().sum()
    if n_unmatched > 0:
        raise ValueError(
            f"{n_unmatched} item(s) in layer={layer} did not match the dataset CSV. "
            "Ensure the run and dataset CSV originate from the same extraction."
        )

    # Enforce column order per spec
    ordered = ["item_id", "layer", "pc1", "pc2", *FACTOR_COLS, "n_frames"]
    ordered += [c for c in merged.columns if c not in ordered]
    return LayerPCAResult(df=merged[ordered], explained_variance=ev, n_items=n_items)


# ── Figures ───────────────────────────────────────────────────────────────────


def _scatter_pca(
    df: pd.DataFrame,
    ev: list[float],
    out_path: Path,
    title: str,
    by_length: bool,
) -> None:
    """Save a publication-style PCA scatter plot.

    Args:
        df:        DataFrame with pc1, pc2, lexicality, and (if by_length) length_bin.
        ev:        [ev_pc1, ev_pc2] explained variance ratios.
        out_path:  destination file path.
        title:     plot title string.
        by_length: if True, encode length_bin via marker shape in addition to color.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    lex_levels = sorted(df["lexicality"].dropna().unique())

    if by_length:
        length_levels = sorted(df["length_bin"].dropna().unique())
        for lex in lex_levels:
            for lb in length_levels:
                mask = (df["lexicality"] == lex) & (df["length_bin"] == lb)
                sub = df[mask]
                if sub.empty:
                    continue
                ax.scatter(
                    sub["pc1"], sub["pc2"],
                    color=_LEX_COLORS.get(lex, "#999999"),
                    marker=_LEN_MARKERS.get(lb, "s"),
                    s=40, alpha=0.75, linewidths=0,
                    label=f"{lex} / {lb}",
                )
        # Group centroids (one "+" per (lex, lb) cell)
        for lex in lex_levels:
            for lb in length_levels:
                mask = (df["lexicality"] == lex) & (df["length_bin"] == lb)
                sub = df[mask]
                if sub.empty:
                    continue
                ax.scatter(
                    sub["pc1"].mean(), sub["pc2"].mean(),
                    color=_LEX_COLORS.get(lex, "#999999"),
                    marker="+", s=180, linewidths=2, zorder=5,
                )
    else:
        for lex in lex_levels:
            sub = df[df["lexicality"] == lex]
            ax.scatter(
                sub["pc1"], sub["pc2"],
                color=_LEX_COLORS.get(lex, "#999999"),
                s=40, alpha=0.75, linewidths=0,
                label=lex,
            )
        # Group centroids
        for lex in lex_levels:
            sub = df[df["lexicality"] == lex]
            if sub.empty:
                continue
            ax.scatter(
                sub["pc1"].mean(), sub["pc2"].mean(),
                color=_LEX_COLORS.get(lex, "#999999"),
                marker="+", s=180, linewidths=2, zorder=5,
            )

    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f}% var)", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_figures(
    result: LayerPCAResult,
    out_dir: Path,
    run_id: str,
    layer: str,
) -> None:
    """Save three PCA scatter plots for one (run_id, layer).

    Plots produced:
        1. All items  — color=lexicality, marker=length_bin, with group centroids.
        2. Short only — color=lexicality.
        3. Long only  — color=lexicality.

    Args:
        result:  LayerPCAResult from run_pca().
        out_dir: destination directory (must exist).
        run_id:  used in plot titles.
        layer:   used in plot titles.
    """
    df = result.df
    ev = result.explained_variance
    base_title = f"{run_id} — {layer}"

    # 1. Global scatter
    path = out_dir / "point_per_word_mean.png"
    _scatter_pca(
        df, ev, path,
        title=f"{base_title}\nAll items (n={len(df)})",
        by_length=True,
    )
    print(f"[pca_analysis] Written : {path}")

    # 2. Short items only
    short_df = df[df["length_bin"] == "short"]
    path = out_dir / "point_per_word_mean_short_only.png"
    _scatter_pca(
        short_df, ev, path,
        title=f"{base_title}\nShort items only (n={len(short_df)})",
        by_length=False,
    )
    print(f"[pca_analysis] Written : {path}")

    # 3. Long items only
    long_df = df[df["length_bin"] == "long"]
    path = out_dir / "point_per_word_mean_long_only.png"
    _scatter_pca(
        long_df, ev, path,
        title=f"{base_title}\nLong items only (n={len(long_df)})",
        by_length=False,
    )
    print(f"[pca_analysis] Written : {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "PCA visualization of audio codec hidden states — one point per word.\n"
            "Fits a separate 2-component PCA per (run_id, layer) using mean-over-time "
            "activations.  Supports encoder_out and decoder_in only."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run",
        required=True,
        help="Path to run directory (e.g. reproduce/data/audio/encodec__7f7d3b97/)",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=SUPPORTED_LAYERS,
        choices=SUPPORTED_LAYERS,
        metavar="LAYER",
        help=(
            f"Layers to process (default: all supported = {SUPPORTED_LAYERS}). "
            "decoder_out is excluded — it is waveform-like."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output root directory "
            "(default: reproduce/figures/audio/pca/<run_id>/ relative to repo root). "
            "One subdirectory per layer is created inside."
        ),
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repo root for resolving the dataset path from manifest (default: auto-detected).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run).resolve()
    if not run_dir.exists():
        print(f"[pca_analysis] ERROR: run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else REPO_ROOT
    manifest = load_manifest(run_dir)
    run_id = manifest["run_id"]

    out_root = (
        Path(args.output).resolve()
        if args.output
        else repo_root / "reproduce" / "figures" / "audio" / "pca" / run_id
    )

    print(f"[pca_analysis] Run       : {run_id}")
    print(f"[pca_analysis] Run dir   : {run_dir}")
    print(f"[pca_analysis] Layers    : {args.layers}")
    print(f"[pca_analysis] Repo root : {repo_root}")
    print(f"[pca_analysis] Output    : {out_root}")

    paradigm_df = load_paradigm_factors(manifest, repo_root)

    # Apply the same whitegrid theme used across audio scripts
    plt.rcParams.update({
        "axes.grid": True,
        "axes.grid.axis": "both",
        "axes.facecolor": "white",
        "grid.color": "0.85",
        "grid.linewidth": 0.8,
        "font.size": 10,
    })

    for layer in args.layers:
        print(f"\n[pca_analysis] ── Processing layer={layer} ──────────────────────")

        try:
            result = run_pca(manifest, run_dir, paradigm_df, layer)
        except RuntimeError as exc:
            print(f"[pca_analysis] WARNING: skipping layer={layer}: {exc}")
            continue

        layer_out = out_root / layer
        layer_out.mkdir(parents=True, exist_ok=True)

        # Tidy CSV
        csv_path = layer_out / "point_per_word_mean.csv"
        result.df.to_csv(csv_path, index=False)
        print(f"[pca_analysis] Written : {csv_path}")

        # JSON summary
        summary = {
            "run_id": run_id,
            "layer": layer,
            "n_items": result.n_items,
            "n_words": int((result.df["lexicality"] == "word").sum()),
            "n_nonwords": int((result.df["lexicality"] == "nonword").sum()),
            "n_short": int((result.df["length_bin"] == "short").sum()),
            "n_long": int((result.df["length_bin"] == "long").sum()),
            "explained_variance_pc1": result.explained_variance[0],
            "explained_variance_pc2": result.explained_variance[1],
            "explained_variance_total": sum(result.explained_variance),
        }
        json_path = layer_out / "point_per_word_mean_summary.json"
        json_path.write_text(json.dumps(summary, indent=2))
        print(f"[pca_analysis] Written : {json_path}")

        # Figures
        save_figures(result, layer_out, run_id, layer)

    print(f"\n[pca_analysis] Done. All outputs under {out_root}")


if __name__ == "__main__":
    main()