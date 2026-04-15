#!/usr/bin/env python3
"""PCA visualization of audio codec hidden states.

Two complementary modes are available, both working per (run_id, layer):

point mode (default)
--------------------
"One point per word" PCA using mean-over-time activations.
For each item the [C, T] tensor is averaged over T to produce one [C] vector,
then a 2-component PCA is fit on the [n_items, C] matrix.
Useful for asking: do items cluster by condition in a global summary?

trajectory mode
---------------
Frame-level trajectory PCA that tests whether lexicality emerges in
temporal dynamics rather than in mean-pooled representations.
All frames from all items are pooled to fit a shared 2D PCA space.
Each item's frame sequence is then projected → [T_item, 2] trajectory,
resampled to a fixed number of normalized time steps (default: 30), and
condition-mean trajectories for the 4 (lexicality × length_bin) cells are
visualized.

both mode
---------
Runs point then trajectory for each layer.

Tensor shape assumption
-----------------------
Activations saved by scripts/audio/extract.py have shape [C, T] for
encoder_out and decoder_in (C = channels, T = frame count, C > 1).
decoder_out is waveform-like and is excluded.

Output locations
----------------
reproduce/figures/audio/pca/<run_id>/<layer>/

point mode outputs:
    point_per_word_mean.csv
    point_per_word_mean_summary.json
    point_per_word_mean.png
    point_per_word_mean_short_only.png
    point_per_word_mean_long_only.png

trajectory mode outputs:
    trajectory_condition_means.png
    trajectory_condition_means_with_examples.png
    trajectory_condition_means_summary.json
    trajectory_condition_means_centroids.csv
    trajectory_projected_items.csv

Usage
-----
    # Static "point per word" PCA (unchanged default)
    python reproduce/scripts/audio/pca_analysis.py \\
        --run reproduce/data/audio/encodec__7f7d3b97/

    # Trajectory mode only
    python reproduce/scripts/audio/pca_analysis.py \\
        --run reproduce/data/audio/encodec__7f7d3b97/ \\
        --mode trajectory

    # Both modes
    python reproduce/scripts/audio/pca_analysis.py \\
        --run reproduce/data/audio/encodec__7f7d3b97/ \\
        --mode both

    # Trajectory with custom resampling and example overlays
    python reproduce/scripts/audio/pca_analysis.py \\
        --run reproduce/data/audio/encodec__7f7d3b97/ \\
        --mode trajectory --resample-steps 50 --overlay-per-condition 12

    # Single layer
    python reproduce/scripts/audio/pca_analysis.py \\
        --run reproduce/data/audio/encodec__7f7d3b97/ \\
        --layers decoder_in --mode both
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

# Colorblind-safe palette (matches seaborn "colorblind" used elsewhere in repo)
_LEX_COLORS: dict[str, str] = {"word": "#0072B2", "nonword": "#E69F00"}
_LEN_MARKERS: dict[str, str] = {"short": "o", "long": "^"}
_LEN_LINESTYLES: dict[str, str] = {"short": "-", "long": "--"}

# 4 condition cells for trajectory grouping
CONDITIONS: list[tuple[str, str]] = [
    ("word", "short"),
    ("word", "long"),
    ("nonword", "short"),
    ("nonword", "long"),
]


# ── Shared data structures ────────────────────────────────────────────────────


class LayerPCAResult(NamedTuple):
    """Everything produced by one (run_id, layer) static point PCA run."""

    df: pd.DataFrame            # Tidy row-per-item DataFrame
    explained_variance: list[float]  # [ev_pc1, ev_pc2]
    n_items: int


class LayerTrajectoryResult(NamedTuple):
    """Everything produced by one (run_id, layer) trajectory PCA run."""

    items_df: pd.DataFrame      # Per-item per-frame rows (projected, raw time)
    centroids_df: pd.DataFrame  # Per-condition per-normalized-step rows
    explained_variance: list[float]  # [ev_pc1, ev_pc2]
    n_items: int
    n_frames_total: int
    resample_steps: int


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


def _is_waveform(t: torch.Tensor) -> bool:
    """Return True if tensor is waveform-like: 1D [T] or 2D [1, T]."""
    return t.dim() == 1 or (t.dim() == 2 and t.shape[0] == 1)


def _load_tensors(
    manifest: dict,
    run_dir: Path,
    layer: str,
) -> tuple[list[str], list[torch.Tensor]]:
    """Load raw [C, T] tensors for all items at one layer.

    Waveform-like tensors are skipped with a warning.

    Args:
        manifest: parsed manifest.json
        run_dir:  absolute path to run directory
        layer:    layer name (e.g. "encoder_out")

    Returns:
        item_ids — list of item IDs with valid 2D tensors
        tensors  — corresponding list of float32 tensors [C, T]

    Raises:
        RuntimeError: if no valid 2D tensors are found.
    """
    items = manifest["items"]
    item_ids: list[str] = []
    tensors: list[torch.Tensor] = []
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
        item_ids.append(item_id)
        tensors.append(t)

    if skipped:
        print(f"[pca_analysis] Skipped {skipped} waveform-like tensor(s) for layer={layer}.")
    if not tensors:
        raise RuntimeError(
            f"No valid 2D tensors found for layer={layer}. "
            "Verify that this layer is present in the run."
        )
    return item_ids, tensors


# ── Point PCA (static "one point per word") ───────────────────────────────────


def _load_mean_activations(
    manifest: dict,
    run_dir: Path,
    layer: str,
) -> tuple[list[str], np.ndarray, list[int]]:
    """Load activations for all items at one layer and mean-pool over time.

    For each item the [C, T] tensor is averaged across T, yielding one
    representative vector [C] per item.

    Args:
        manifest: parsed manifest.json
        run_dir:  absolute path to run directory
        layer:    layer name (e.g. "encoder_out")

    Returns:
        item_ids  — list of n item IDs
        matrix    — float32 array [n, C] of mean-pooled activations
        n_frames  — list of T values (frame counts) per item
    """
    item_ids, tensors = _load_tensors(manifest, run_dir, layer)
    vectors = [t.mean(dim=1).numpy() for t in tensors]   # [C] per item
    n_frames = [t.shape[1] for t in tensors]
    return item_ids, np.stack(vectors, axis=0).astype(np.float32), n_frames


def run_pca(
    manifest: dict,
    run_dir: Path,
    paradigm_df: pd.DataFrame,
    layer: str,
) -> LayerPCAResult:
    """Fit a 2-component PCA for one (run_id, layer) pair — point mode.

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
    print(f"[pca_analysis] [point] Layer={layer} — loading {n_total} item activations …")

    item_ids, matrix, n_frames = _load_mean_activations(manifest, run_dir, layer)
    n_items = len(item_ids)

    print(
        f"[pca_analysis] [point] Layer={layer} — matrix shape: {matrix.shape}  "
        f"(n_items={n_items}, n_dims={matrix.shape[1]})"
    )

    matrix_scaled = StandardScaler().fit_transform(matrix)
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(matrix_scaled)   # [n_items, 2]
    ev = pca.explained_variance_ratio_.tolist()

    print(
        f"[pca_analysis] [point] Layer={layer} — explained variance: "
        f"PC1={ev[0]:.3f}  PC2={ev[1]:.3f}  total={sum(ev):.3f}"
    )

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

    ordered = ["item_id", "layer", "pc1", "pc2", *FACTOR_COLS, "n_frames"]
    ordered += [c for c in merged.columns if c not in ordered]
    return LayerPCAResult(df=merged[ordered], explained_variance=ev, n_items=n_items)


# ── Trajectory PCA (frame-level) ──────────────────────────────────────────────


def _resample_trajectory(traj: np.ndarray, n_steps: int) -> np.ndarray:
    """Linearly resample a 2D trajectory to n_steps equally spaced time points.

    Args:
        traj:    float array [T, 2] — original trajectory in PCA space.
        n_steps: number of normalized time steps to output.

    Returns:
        float array [n_steps, 2] — resampled trajectory.
    """
    T = traj.shape[0]
    if T == 1:
        return np.tile(traj, (n_steps, 1))
    t_orig = np.linspace(0.0, 1.0, T)
    t_new = np.linspace(0.0, 1.0, n_steps)
    pc1 = np.interp(t_new, t_orig, traj[:, 0])
    pc2 = np.interp(t_new, t_orig, traj[:, 1])
    return np.stack([pc1, pc2], axis=1)   # [n_steps, 2]


def run_trajectory_pca(
    manifest: dict,
    run_dir: Path,
    paradigm_df: pd.DataFrame,
    layer: str,
    resample_steps: int = 30,
) -> LayerTrajectoryResult:
    """Fit a frame-level 2-component PCA and build per-item projected trajectories.

    Method:
        1. Load all item tensors [C, T] for the layer.
        2. Transpose each to [T, C] (frames × channels).
        3. Concatenate all frames: [total_frames, C].
        4. Standardize with StandardScaler, fit PCA(n_components=2).
        5. Project each item's frames back → [T_item, 2] trajectory.
        6. Resample each projected trajectory to n_steps via linear interpolation.
        7. Group by (lexicality × length_bin), compute condition-mean trajectories.

    Resampling happens after projection to ensure temporal normalization is
    performed in the common 2D PCA space, not in the high-dimensional feature space.

    Args:
        manifest:       parsed manifest.json
        run_dir:        absolute path to run directory
        paradigm_df:    DataFrame with item_id and factor columns
        layer:          layer name (e.g. "encoder_out")
        resample_steps: number of normalized time steps (default 30)

    Returns:
        LayerTrajectoryResult with per-item and per-condition DataFrames.

    Raises:
        ValueError: if item_ids fail to match the paradigm DataFrame.
    """
    n_total = len(manifest["items"])
    print(
        f"[pca_analysis] [trajectory] Layer={layer} — loading {n_total} item tensors …"
    )

    item_ids, tensors = _load_tensors(manifest, run_dir, layer)
    n_items = len(item_ids)

    # Build frame-level matrix: concatenate [T_i, C] across items
    frames_list = [t.permute(1, 0).numpy() for t in tensors]   # list of [T_i, C]
    all_frames = np.concatenate(frames_list, axis=0).astype(np.float32)
    n_frames_total = all_frames.shape[0]

    print(
        f"[pca_analysis] [trajectory] Layer={layer} — "
        f"all_frames shape: {all_frames.shape}  "
        f"(n_items={n_items}, total_frames={n_frames_total})"
    )

    # Fit shared PCA on all frames
    scaler = StandardScaler()
    all_frames_scaled = scaler.fit_transform(all_frames)
    pca = PCA(n_components=2, random_state=0)
    pca.fit(all_frames_scaled)
    ev = pca.explained_variance_ratio_.tolist()

    print(
        f"[pca_analysis] [trajectory] Layer={layer} — explained variance: "
        f"PC1={ev[0]:.3f}  PC2={ev[1]:.3f}  total={sum(ev):.3f}"
    )

    # Project each item's frames and resample to normalized time
    item_rows: list[dict] = []
    resampled: dict[str, np.ndarray] = {}   # item_id → [resample_steps, 2]

    for item_id, frames in zip(item_ids, frames_list):
        frames_scaled = scaler.transform(frames)               # [T, C]
        proj = pca.transform(frames_scaled)                    # [T, 2]
        T = proj.shape[0]

        # Store raw projected frames for the per-item CSV
        for t_idx in range(T):
            item_rows.append({
                "item_id": item_id,
                "layer": layer,
                "time_idx_raw": t_idx,
                "pc1": float(proj[t_idx, 0]),
                "pc2": float(proj[t_idx, 1]),
            })
        resampled[item_id] = _resample_trajectory(proj, resample_steps)

    # Build per-item raw-frame DataFrame and merge factor columns
    items_df = pd.DataFrame(item_rows)
    items_df = items_df.merge(
        paradigm_df[["item_id", *FACTOR_COLS]].assign(
            n_frames_raw=pd.Series(
                {iid: t.shape[1] for iid, t in zip(item_ids, tensors)}
            ).reset_index().rename(columns={"index": "item_id", 0: "n_frames_raw"})["n_frames_raw"].values
        ),
        on="item_id", how="left",
    )
    # n_frames_raw: broadcast per-item value to all frame rows
    n_frames_map = {iid: t.shape[1] for iid, t in zip(item_ids, tensors)}
    items_df["n_frames_raw"] = items_df["item_id"].map(n_frames_map)
    # Reorder columns
    items_col_order = [
        "item_id", "layer", *FACTOR_COLS, "n_frames_raw", "time_idx_raw", "pc1", "pc2",
    ]
    for c in items_col_order:
        if c not in items_df.columns:
            items_df[c] = np.nan
    items_df = items_df[items_col_order]

    n_unmatched = items_df["lexicality"].isna().sum()
    if n_unmatched > 0:
        raise ValueError(
            f"{n_unmatched} frame row(s) in layer={layer} did not match the dataset CSV."
        )

    # Build paradigm lookup keyed by item_id for condition grouping
    para_idx = paradigm_df.set_index("item_id")

    # Compute condition-mean trajectories from resampled paths
    centroid_rows: list[dict] = []
    condition_counts: dict[str, int] = {}

    t_norm_values = np.linspace(0.0, 1.0, resample_steps)

    for lex, lb in CONDITIONS:
        cond_items = [
            iid for iid in item_ids
            if iid in para_idx.index
            and para_idx.loc[iid, "lexicality"] == lex
            and para_idx.loc[iid, "length_bin"] == lb
        ]
        cond_key = f"{lex}/{lb}"
        condition_counts[cond_key] = len(cond_items)

        if not cond_items:
            print(
                f"[pca_analysis] [trajectory] WARNING: no items for condition "
                f"{cond_key} — skipping."
            )
            continue

        traj_stack = np.stack(
            [resampled[iid] for iid in cond_items], axis=0
        )  # [n_cond, resample_steps, 2]
        mean_traj = traj_stack.mean(axis=0)   # [resample_steps, 2]

        for step_idx in range(resample_steps):
            centroid_rows.append({
                "run_id": manifest["run_id"],
                "layer": layer,
                "lexicality": lex,
                "length_bin": lb,
                "time_idx": step_idx,
                "time_norm": float(t_norm_values[step_idx]),
                "pc1": float(mean_traj[step_idx, 0]),
                "pc2": float(mean_traj[step_idx, 1]),
                "n_items_in_condition": len(cond_items),
            })

    centroids_df = pd.DataFrame(centroid_rows)

    return LayerTrajectoryResult(
        items_df=items_df,
        centroids_df=centroids_df,
        explained_variance=ev,
        n_items=n_items,
        n_frames_total=n_frames_total,
        resample_steps=resample_steps,
    )


# ── Point PCA figures ─────────────────────────────────────────────────────────


def _scatter_pca(
    df: pd.DataFrame,
    ev: list[float],
    out_path: Path,
    title: str,
    by_length: bool,
) -> None:
    """Save a publication-style static PCA scatter plot.

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


def save_point_figures(
    result: LayerPCAResult,
    out_dir: Path,
    run_id: str,
    layer: str,
) -> None:
    """Save the three static point PCA scatter plots.

    Args:
        result:  LayerPCAResult from run_pca().
        out_dir: destination directory (must exist).
        run_id:  used in plot titles.
        layer:   used in plot titles.
    """
    df = result.df
    ev = result.explained_variance
    base_title = f"{run_id} — {layer}"

    path = out_dir / "point_per_word_mean.png"
    _scatter_pca(df, ev, path, f"{base_title}\nAll items (n={len(df)})", by_length=True)
    print(f"[pca_analysis] Written : {path}")

    short_df = df[df["length_bin"] == "short"]
    path = out_dir / "point_per_word_mean_short_only.png"
    _scatter_pca(
        short_df, ev, path,
        f"{base_title}\nShort items only (n={len(short_df)})",
        by_length=False,
    )
    print(f"[pca_analysis] Written : {path}")

    long_df = df[df["length_bin"] == "long"]
    path = out_dir / "point_per_word_mean_long_only.png"
    _scatter_pca(
        long_df, ev, path,
        f"{base_title}\nLong items only (n={len(long_df)})",
        by_length=False,
    )
    print(f"[pca_analysis] Written : {path}")


# ── Trajectory PCA figures ────────────────────────────────────────────────────


def _condition_label(lex: str, lb: str) -> str:
    return f"{lex} / {lb}"


def _draw_trajectory_axes(
    ax: plt.Axes,
    centroids_df: pd.DataFrame,
    ev: list[float],
    title: str,
) -> None:
    """Draw condition-mean trajectories on an existing Axes object.

    Each of the 4 (lexicality × length_bin) conditions is drawn as a line
    with start (circle) and end (square) markers.  Color encodes lexicality;
    linestyle encodes length_bin.

    Args:
        ax:           matplotlib Axes to draw on.
        centroids_df: per-condition per-step DataFrame from LayerTrajectoryResult.
        ev:           [ev_pc1, ev_pc2] explained variance ratios.
        title:        axes title string.
    """
    for lex, lb in CONDITIONS:
        sub = centroids_df[
            (centroids_df["lexicality"] == lex) & (centroids_df["length_bin"] == lb)
        ].sort_values("time_idx")
        if sub.empty:
            continue

        color = _LEX_COLORS.get(lex, "#999999")
        ls = _LEN_LINESTYLES.get(lb, "-")
        label = _condition_label(lex, lb)

        ax.plot(sub["pc1"], sub["pc2"], color=color, linestyle=ls,
                linewidth=2.2, label=label, zorder=3)
        # Start marker (filled circle)
        ax.scatter(sub["pc1"].iloc[0], sub["pc2"].iloc[0],
                   color=color, marker="o", s=80, zorder=5, edgecolors="white",
                   linewidths=1.2)
        # End marker (filled square)
        ax.scatter(sub["pc1"].iloc[-1], sub["pc2"].iloc[-1],
                   color=color, marker="s", s=80, zorder=5, edgecolors="white",
                   linewidths=1.2)

    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f}% var)", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, framealpha=0.8, title="lexicality / length")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_trajectory_figures(
    result: LayerTrajectoryResult,
    out_dir: Path,
    run_id: str,
    layer: str,
    overlay_per_condition: int = 8,
    rng_seed: int = 0,
) -> None:
    """Save both trajectory PCA figures for one (run_id, layer).

    Figure 1 — condition means only.
    Figure 2 — condition means + alpha-blended individual trajectory overlays.

    Args:
        result:                LayerTrajectoryResult from run_trajectory_pca().
        out_dir:               destination directory (must exist).
        run_id:                used in plot titles.
        layer:                 used in plot titles.
        overlay_per_condition: max individual trajectories overlaid per condition.
        rng_seed:              random seed for reproducible overlay sampling.
    """
    ev = result.explained_variance
    centroids_df = result.centroids_df
    items_df = result.items_df
    base_title = f"{run_id} — {layer}  [trajectory PCA, T_norm={result.resample_steps}]"
    rng = np.random.default_rng(rng_seed)

    # ── Figure 1: condition means only ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    _draw_trajectory_axes(ax, centroids_df, ev, base_title)
    plt.tight_layout()
    path = out_dir / "trajectory_condition_means.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[pca_analysis] Written : {path}")

    # ── Figure 2: condition means + individual overlays ───────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))

    # Draw individual trajectories first (background)
    for lex, lb in CONDITIONS:
        color = _LEX_COLORS.get(lex, "#999999")
        mask = (items_df["lexicality"] == lex) & (items_df["length_bin"] == lb)
        cond_items = items_df[mask]["item_id"].unique()

        n_sample = min(overlay_per_condition, len(cond_items))
        sample_ids = rng.choice(cond_items, size=n_sample, replace=False)

        for iid in sample_ids:
            sub = items_df[items_df["item_id"] == iid].sort_values("time_idx_raw")
            ax.plot(sub["pc1"], sub["pc2"], color=color, linewidth=0.8,
                    alpha=0.20, zorder=1)

    # Draw condition means on top
    _draw_trajectory_axes(ax, centroids_df, ev, base_title)
    plt.tight_layout()
    path = out_dir / "trajectory_condition_means_with_examples.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[pca_analysis] Written : {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "PCA visualization of audio codec hidden states.\n\n"
            "Modes:\n"
            "  point      — static 'one point per word' PCA (mean-over-time)\n"
            "  trajectory — frame-level trajectory PCA (temporal dynamics)\n"
            "  both       — run point then trajectory"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run",
        required=True,
        help="Path to run directory (e.g. reproduce/data/audio/encodec__7f7d3b97/)",
    )
    parser.add_argument(
        "--mode",
        choices=["point", "trajectory", "both"],
        default="point",
        help="PCA mode: 'point' (default), 'trajectory', or 'both'.",
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
        "--resample-steps",
        type=int,
        default=30,
        metavar="N",
        help=(
            "Number of normalized time steps for trajectory resampling "
            "(trajectory/both mode only, default: 30)."
        ),
    )
    parser.add_argument(
        "--overlay-per-condition",
        type=int,
        default=8,
        metavar="N",
        help=(
            "Max individual trajectories overlaid per condition in the "
            "with-examples figure (trajectory/both mode only, default: 8)."
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
    print(f"[pca_analysis] Mode      : {args.mode}")
    print(f"[pca_analysis] Layers    : {args.layers}")
    print(f"[pca_analysis] Repo root : {repo_root}")
    print(f"[pca_analysis] Output    : {out_root}")

    paradigm_df = load_paradigm_factors(manifest, repo_root)

    plt.rcParams.update({
        "axes.grid": True,
        "axes.grid.axis": "both",
        "axes.facecolor": "white",
        "grid.color": "0.85",
        "grid.linewidth": 0.8,
        "font.size": 10,
    })

    do_point = args.mode in ("point", "both")
    do_trajectory = args.mode in ("trajectory", "both")

    for layer in args.layers:
        print(f"\n[pca_analysis] ── Processing layer={layer} ──────────────────────")
        layer_out = out_root / layer
        layer_out.mkdir(parents=True, exist_ok=True)

        # ── Point mode ────────────────────────────────────────────────────────
        if do_point:
            try:
                point_result = run_pca(manifest, run_dir, paradigm_df, layer)
            except RuntimeError as exc:
                print(f"[pca_analysis] WARNING: [point] skipping layer={layer}: {exc}")
            else:
                csv_path = layer_out / "point_per_word_mean.csv"
                point_result.df.to_csv(csv_path, index=False)
                print(f"[pca_analysis] Written : {csv_path}")

                ev = point_result.explained_variance
                summary = {
                    "run_id": run_id,
                    "layer": layer,
                    "pca_type": "mean_pooled_point",
                    "n_items": point_result.n_items,
                    "n_words": int((point_result.df["lexicality"] == "word").sum()),
                    "n_nonwords": int((point_result.df["lexicality"] == "nonword").sum()),
                    "n_short": int((point_result.df["length_bin"] == "short").sum()),
                    "n_long": int((point_result.df["length_bin"] == "long").sum()),
                    "explained_variance_pc1": ev[0],
                    "explained_variance_pc2": ev[1],
                    "explained_variance_total": sum(ev),
                }
                json_path = layer_out / "point_per_word_mean_summary.json"
                json_path.write_text(json.dumps(summary, indent=2))
                print(f"[pca_analysis] Written : {json_path}")

                save_point_figures(point_result, layer_out, run_id, layer)

        # ── Trajectory mode ───────────────────────────────────────────────────
        if do_trajectory:
            try:
                traj_result = run_trajectory_pca(
                    manifest, run_dir, paradigm_df, layer,
                    resample_steps=args.resample_steps,
                )
            except RuntimeError as exc:
                print(f"[pca_analysis] WARNING: [trajectory] skipping layer={layer}: {exc}")
                continue

            # Per-item raw-frame CSV
            items_csv = layer_out / "trajectory_projected_items.csv"
            traj_result.items_df.to_csv(items_csv, index=False)
            print(f"[pca_analysis] Written : {items_csv}")

            # Condition centroids CSV
            centroids_csv = layer_out / "trajectory_condition_means_centroids.csv"
            traj_result.centroids_df.to_csv(centroids_csv, index=False)
            print(f"[pca_analysis] Written : {centroids_csv}")

            # Summary JSON
            ev = traj_result.explained_variance
            cond_counts = {}
            for lex, lb in CONDITIONS:
                key = f"{lex}/{lb}"
                sub = traj_result.centroids_df[
                    (traj_result.centroids_df["lexicality"] == lex)
                    & (traj_result.centroids_df["length_bin"] == lb)
                ]
                cond_counts[key] = int(sub["n_items_in_condition"].iloc[0]) if len(sub) > 0 else 0

            summary = {
                "run_id": run_id,
                "layer": layer,
                "pca_type": "frame_level_trajectory",
                "n_items": traj_result.n_items,
                "n_frames_total": traj_result.n_frames_total,
                "explained_variance_pc1": ev[0],
                "explained_variance_pc2": ev[1],
                "explained_variance_total": sum(ev),
                "resample_steps": traj_result.resample_steps,
                "condition_counts": cond_counts,
            }
            json_path = layer_out / "trajectory_condition_means_summary.json"
            json_path.write_text(json.dumps(summary, indent=2))
            print(f"[pca_analysis] Written : {json_path}")

            # Figures
            save_trajectory_figures(
                traj_result, layer_out, run_id, layer,
                overlay_per_condition=args.overlay_per_condition,
            )

    print(f"\n[pca_analysis] Done. All outputs under {out_root}")


if __name__ == "__main__":
    main()