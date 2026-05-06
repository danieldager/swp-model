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

trajectory mode outputs (Cartesian):
    trajectory_condition_means.png
    trajectory_condition_means_with_examples.png
    trajectory_condition_means_summary.json
    trajectory_condition_means_centroids.csv
    trajectory_projected_items.csv

trajectory mode outputs (polar — auto-generated alongside Cartesian):
    trajectory_condition_means_polar.csv
    trajectory_condition_means_polar_summary.json
    trajectory_radius_over_time.png
    trajectory_angle_over_time.png
    trajectory_condition_means_polar.png
    trajectory_condition_means_polar_pca_origin.csv
    trajectory_condition_means_polar_pca_origin.png

trajectory mode outputs (condition distances — auto-generated alongside Cartesian):
    trajectory_condition_distances_over_time.png
    trajectory_condition_distances_over_time.csv
    trajectory_condition_distances_summary.csv

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

# Visual grammar: color = lexicality, marker/linestyle = length.
# Colorblind-safe palette consistent with other audio scripts.
_LEX_COLORS: dict[str, str] = {"word": "#0072B2", "nonword": "#ED2727"}
_LEN_MARKERS: dict[str, str] = {"short": "o", "long": "^"}
_COND_LINESTYLES: dict[str, str] = {"short": "-", "long": "--"}


def _display_lex(lex: str) -> str:
    """Map raw lexicality value to display label ('nonword' → 'pseudoword')."""
    return "pseudoword" if lex == "nonword" else lex

# 4 condition cells for trajectory grouping
CONDITIONS: list[tuple[str, str]] = [
    ("word", "short"),
    ("word", "long"),
    ("nonword", "short"),
    ("nonword", "long"),
]


# ── Shared data structures ──�����────────────────────────────────────────────────


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
) -> None:
    """Save a publication-style static PCA scatter plot.

    Color encodes lexicality (_LEX_COLORS); marker shape encodes length_bin
    (_LEN_MARKERS).  Group centroids are shown as "+" crosses.

    Args:
        df:       DataFrame with columns pc1, pc2, lexicality, length_bin.
        ev:       [ev_pc1, ev_pc2] explained variance ratios.
        out_path: destination file path.
        title:    plot title string.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    for lex, lb in CONDITIONS:
        mask = (df["lexicality"] == lex) & (df["length_bin"] == lb)
        sub = df[mask]
        if sub.empty:
            continue
        color = _LEX_COLORS.get(lex, "#999999")
        marker = _LEN_MARKERS.get(lb, "o")
        ax.scatter(
            sub["pc1"], sub["pc2"],
            color=color, marker=marker,
            s=40, alpha=0.75, linewidths=0,
            label=f"{_display_lex(lex)} / {lb}",
        )
        # Group centroid
        ax.scatter(
            sub["pc1"].mean(), sub["pc2"].mean(),
            color=color, marker="+", s=180, linewidths=2, zorder=5,
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
    base_title = f"{run_id} — {layer}  [mean-pooled PCA]"

    path = out_dir / "point_per_word_mean.png"
    _scatter_pca(df, ev, path, f"{base_title}\nAll items (n={len(df)})")
    print(f"[pca_analysis] Written : {path}")

    short_df = df[df["length_bin"] == "short"]
    path = out_dir / "point_per_word_mean_short_only.png"
    _scatter_pca(short_df, ev, path, f"{base_title}\nShort items only (n={len(short_df)})")
    print(f"[pca_analysis] Written : {path}")

    long_df = df[df["length_bin"] == "long"]
    path = out_dir / "point_per_word_mean_long_only.png"
    _scatter_pca(long_df, ev, path, f"{base_title}\nLong items only (n={len(long_df)})")
    print(f"[pca_analysis] Written : {path}")


# ── Trajectory PCA figures ────────────────────────────────────────────────────


def _condition_label(lex: str, lb: str) -> str:
    return f"{_display_lex(lex)} / {lb}"


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
        ls = _COND_LINESTYLES.get(lb, "-")
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
    base_title = f"{run_id} — {layer}  [trajectory PCA on raw frames, T_norm={result.resample_steps}]"
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


# ── Polar trajectory analysis ────────────────────────────────────────────────


def compute_polar_centroids(
    centroids_df: pd.DataFrame,
    run_id: str,
    layer: str,
) -> tuple[pd.DataFrame, float, float]:
    """Convert condition-mean Cartesian trajectories to polar coordinates.

    Polar origin definition
    -----------------------
    The reference center is the mean of the 4 condition trajectory starting
    points (time_idx == 0).  Concretely:

        center = mean over (lex, lb) of centroids_df[time_idx==0][[pc1, pc2]]

    This grounds the origin in the data rather than in the arbitrary PCA (0,0).

    Angle unwrapping
    ----------------
    Raw angles are computed with numpy.arctan2(pc2_centered, pc1_centered).
    numpy.unwrap is then applied **per condition** so that each angle
    trajectory is continuous (no ±π jumps across time steps).
    Angles are stored in radians; degrees are added for convenience.

    Args:
        centroids_df: per-condition per-step DataFrame (from LayerTrajectoryResult).
        run_id:       run identifier (written into every output row).
        layer:        layer name (written into every output row).

    Returns:
        polar_df   — tidy DataFrame with all required polar columns.
        center_pc1 — x coordinate of the common start center.
        center_pc2 — y coordinate of the common start center.
    """
    # Common start center: mean of first normalized time step across all conditions
    start_rows = centroids_df[centroids_df["time_idx"] == 0]
    center_pc1 = float(start_rows["pc1"].mean())
    center_pc2 = float(start_rows["pc2"].mean())

    rows: list[dict] = []
    for lex, lb in CONDITIONS:
        sub = centroids_df[
            (centroids_df["lexicality"] == lex) & (centroids_df["length_bin"] == lb)
        ].sort_values("time_idx").copy()
        if sub.empty:
            continue

        pc1_c = sub["pc1"].values - center_pc1
        pc2_c = sub["pc2"].values - center_pc2
        radius = np.sqrt(pc1_c ** 2 + pc2_c ** 2)

        # Unwrap angle per condition for continuity
        angle_raw = np.arctan2(pc2_c, pc1_c)
        angle_unwrapped = np.unwrap(angle_raw)

        n_items = int(sub["n_items_in_condition"].iloc[0])
        for i, row in enumerate(sub.itertuples(index=False)):
            rows.append({
                "run_id": run_id,
                "layer": layer,
                "lexicality": lex,
                "length_bin": lb,
                "time_idx": int(row.time_idx),
                "time_norm": float(row.time_norm),
                "pc1": float(row.pc1),
                "pc2": float(row.pc2),
                "center_pc1": center_pc1,
                "center_pc2": center_pc2,
                "pc1_centered": float(pc1_c[i]),
                "pc2_centered": float(pc2_c[i]),
                "radius": float(radius[i]),
                "angle_rad": float(angle_unwrapped[i]),
                "angle_deg": float(np.degrees(angle_unwrapped[i])),
                "n_items_in_condition": n_items,
            })

    col_order = [
        "run_id", "layer", "lexicality", "length_bin",
        "time_idx", "time_norm", "pc1", "pc2",
        "center_pc1", "center_pc2", "pc1_centered", "pc2_centered",
        "radius", "angle_rad", "angle_deg", "n_items_in_condition",
    ]
    return pd.DataFrame(rows)[col_order], center_pc1, center_pc2


def _plot_polar_timeseries(
    polar_df: pd.DataFrame,
    y_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    """Save a time-series line plot (x=time_norm, y=y_col) for all 4 conditions.

    Args:
        polar_df: DataFrame from compute_polar_centroids().
        y_col:    column to plot on the y axis ('radius' or 'angle_rad').
        ylabel:   y axis label string.
        title:    plot title.
        out_path: destination file path.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for lex, lb in CONDITIONS:
        sub = polar_df[
            (polar_df["lexicality"] == lex) & (polar_df["length_bin"] == lb)
        ].sort_values("time_norm")
        if sub.empty:
            continue
        color = _LEX_COLORS.get(lex, "#999999")
        ls = _COND_LINESTYLES.get(lb, "-")
        ax.plot(
            sub["time_norm"], sub[y_col],
            color=color, linestyle=ls, linewidth=2.2,
            label=f"{_display_lex(lex)} / {lb}",
        )
    ax.set_xlabel("Normalized time", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_polar_projection(
    polar_df: pd.DataFrame,
    title: str,
    out_path: Path,
) -> None:
    """Save a true polar (r, theta) projection of condition-mean trajectories.

    Uses matplotlib's 'polar' subplot projection.  Start and end of each
    condition trajectory are marked with circle and square respectively.

    Args:
        polar_df: DataFrame from compute_polar_centroids().
        title:    plot title.
        out_path: destination file path.
    """
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    for lex, lb in CONDITIONS:
        sub = polar_df[
            (polar_df["lexicality"] == lex) & (polar_df["length_bin"] == lb)
        ].sort_values("time_norm")
        if sub.empty:
            continue
        color = _LEX_COLORS.get(lex, "#999999")
        ls = _COND_LINESTYLES.get(lb, "-")
        theta = sub["angle_rad"].values
        r = sub["radius"].values
        ax.plot(theta, r, color=color, linestyle=ls, linewidth=2.2,
                label=f"{_display_lex(lex)} / {lb}")
        # Start marker (circle)
        ax.scatter(theta[0], r[0], color=color, marker="o", s=80, zorder=5,
                   edgecolors="white", linewidths=1.2)
        # End marker (square)
        ax.scatter(theta[-1], r[-1], color=color, marker="s", s=80, zorder=5,
                   edgecolors="white", linewidths=1.2)

    ax.set_title(title, fontsize=10, pad=15)
    ax.legend(fontsize=9, framealpha=0.8, loc="upper right",
              bbox_to_anchor=(1.35, 1.1))
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _compute_pca_origin_polar(polar_df: pd.DataFrame) -> pd.DataFrame:
    """Compute polar coordinates relative to the raw PCA origin (0, 0).

    Unlike the start-centered polar representation already stored in polar_df
    (which is centered on the mean of condition start points), this function
    uses the raw Cartesian PCA coordinates pc1 and pc2 directly.

    For every row:
        radius_pca_origin  = sqrt(pc1² + pc2²)
        angle_pca_origin   = atan2(pc2, pc1)   [raw, NOT unwrapped]

    No centering, no reference subtraction, no unwrapping — the angle is
    simply the position angle in the original 2D PCA space.

    Args:
        polar_df: DataFrame from compute_polar_centroids(), which contains
                  the raw 'pc1' and 'pc2' columns alongside the start-centered
                  polar columns.

    Returns:
        New DataFrame with columns required for the PCA-origin CSV output.
    """
    df = polar_df.copy()
    df["radius_pca_origin"] = np.sqrt(df["pc1"] ** 2 + df["pc2"] ** 2)
    angle_rad = np.arctan2(df["pc2"].values, df["pc1"].values)
    df["angle_pca_origin_rad"] = angle_rad
    df["angle_pca_origin_deg"] = np.degrees(angle_rad)

    col_order = [
        "run_id", "layer", "lexicality", "length_bin",
        "time_idx", "time_norm", "pc1", "pc2",
        "radius_pca_origin", "angle_pca_origin_rad", "angle_pca_origin_deg",
        "n_items_in_condition",
    ]
    return df[col_order]


def _plot_polar_pca_origin(
    pca_origin_df: pd.DataFrame,
    title: str,
    out_path: Path,
) -> None:
    """Save a full polar plot of condition-mean trajectories around the PCA origin.

    Angles and radii are computed from raw (pc1, pc2) coordinates — no
    centering or reference-angle subtraction is applied.  The full 360° polar
    space is shown (no sector zoom).

    Args:
        pca_origin_df: DataFrame from _compute_pca_origin_polar().
        title:         plot title.
        out_path:      destination file path.
    """
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    for lex, lb in CONDITIONS:
        sub = pca_origin_df[
            (pca_origin_df["lexicality"] == lex) & (pca_origin_df["length_bin"] == lb)
        ].sort_values("time_norm")
        if sub.empty:
            continue
        color = _LEX_COLORS.get(lex, "#999999")
        ls = _COND_LINESTYLES.get(lb, "-")
        theta = sub["angle_pca_origin_rad"].values
        r = sub["radius_pca_origin"].values
        ax.plot(theta, r, color=color, linestyle=ls, linewidth=2.2,
                label=f"{_display_lex(lex)} / {lb}")
        # Start marker (circle)
        ax.scatter(theta[0], r[0], color=color, marker="o", s=80, zorder=5,
                   edgecolors="white", linewidths=1.2)
        # End marker (square)
        ax.scatter(theta[-1], r[-1], color=color, marker="s", s=80, zorder=5,
                   edgecolors="white", linewidths=1.2)

    ax.set_title(title, fontsize=10, pad=15)
    ax.legend(fontsize=9, framealpha=0.8, loc="upper right",
              bbox_to_anchor=(1.35, 1.1))
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_polar_figures(
    polar_df: pd.DataFrame,
    center_pc1: float,
    center_pc2: float,
    out_dir: Path,
    run_id: str,
    layer: str,
    resample_steps: int,
    condition_counts: dict,
) -> None:
    """Save all polar outputs for one (run_id, layer).

    Files written:
        trajectory_condition_means_polar.csv           — start-centered polar coords
        trajectory_condition_means_polar_summary.json
        trajectory_radius_over_time.png                — radius (start-centered) vs time
        trajectory_angle_over_time.png                 — angle (start-centered) vs time
        trajectory_condition_means_polar.png           — full polar, start-centered
        trajectory_condition_means_polar_pca_origin.csv
        trajectory_condition_means_polar_pca_origin.png — full polar, raw PCA origin (0,0)

    Args:
        polar_df:         DataFrame from compute_polar_centroids() — contains both
                          raw pc1/pc2 and start-centered polar columns.
        center_pc1/pc2:   coordinates of the common start center (for JSON).
        out_dir:          destination directory (must already exist).
        run_id:           used in plot titles and JSON.
        layer:            used in plot titles and JSON.
        resample_steps:   stored in the JSON summary.
        condition_counts: dict of "lex/lb" → n_items (stored in the JSON summary).
    """
    base_title = f"{run_id} — {layer}  [polar]"

    # Polar CSV
    csv_path = out_dir / "trajectory_condition_means_polar.csv"
    polar_df.to_csv(csv_path, index=False)
    print(f"[pca_analysis] Written : {csv_path}")

    # Polar summary JSON
    summary = {
        "run_id": run_id,
        "layer": layer,
        "center_definition": "mean_of_condition_start_points",
        "center_pc1": center_pc1,
        "center_pc2": center_pc2,
        "resample_steps": resample_steps,
        "condition_counts": condition_counts,
    }
    json_path = out_dir / "trajectory_condition_means_polar_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"[pca_analysis] Written : {json_path}")

    # Radius over normalized time
    _plot_polar_timeseries(
        polar_df,
        y_col="radius",
        ylabel="Radius (distance from start center)",
        title=f"{base_title}\nRadius over normalized time",
        out_path=out_dir / "trajectory_radius_over_time.png",
    )
    print(f"[pca_analysis] Written : {out_dir / 'trajectory_radius_over_time.png'}")

    # Angle over normalized time (unwrapped, radians)
    _plot_polar_timeseries(
        polar_df,
        y_col="angle_rad",
        ylabel="Angle (radians, unwrapped)",
        title=f"{base_title}\nAngle over normalized time (unwrapped)",
        out_path=out_dir / "trajectory_angle_over_time.png",
    )
    print(f"[pca_analysis] Written : {out_dir / 'trajectory_angle_over_time.png'}")

    # Full polar projection — start-centered coordinates
    _plot_polar_projection(
        polar_df,
        title=base_title,
        out_path=out_dir / "trajectory_condition_means_polar.png",
    )
    print(f"[pca_analysis] Written : {out_dir / 'trajectory_condition_means_polar.png'}")

    # PCA-origin polar — raw (pc1, pc2) with no centering or reference subtraction
    pca_origin_df = _compute_pca_origin_polar(polar_df)
    pca_origin_csv = out_dir / "trajectory_condition_means_polar_pca_origin.csv"
    pca_origin_df.to_csv(pca_origin_csv, index=False)
    print(f"[pca_analysis] Written : {pca_origin_csv}")
    _plot_polar_pca_origin(
        pca_origin_df,
        title=f"{run_id} — {layer}  [polar, PCA origin]",
        out_path=out_dir / "trajectory_condition_means_polar_pca_origin.png",
    )
    print(f"[pca_analysis] Written : {out_dir / 'trajectory_condition_means_polar_pca_origin.png'}")


# ── Condition-distance analysis ───────────────────────────────────────────────

# Named pairs: (label, contrast_family, cond_A, cond_B)
_CONTRAST_PAIRS: list[tuple[str, str, tuple[str, str], tuple[str, str]]] = [
    ("lex_short",   "lexicality", ("word", "short"),    ("nonword", "short")),
    ("lex_long",    "lexicality", ("word", "long"),      ("nonword", "long")),
    ("len_word",    "length",     ("word", "short"),     ("word", "long")),
    ("len_nonword", "length",     ("nonword", "short"),  ("nonword", "long")),
]


def compute_condition_distances(
    centroids_df: pd.DataFrame,
    run_id: str,
    layer: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Euclidean distances between condition-mean trajectories over time.

    For each of the 4 named contrasts, at every normalized time step the
    distance between the two condition means in 2D PCA space is:

        distance(t) = sqrt((pc1_A(t) - pc1_B(t))² + (pc2_A(t) - pc2_B(t))²)

    Contrasts:
        lex_short   word/short vs nonword/short  (lexicality family)
        lex_long    word/long  vs nonword/long   (lexicality family)
        len_word    word/short vs word/long       (length family)
        len_nonword nonword/short vs nonword/long (length family)

    Args:
        centroids_df: per-condition per-step DataFrame (from LayerTrajectoryResult).
        run_id:       written into every output row.
        layer:        written into every output row.

    Returns:
        dist_df    — long-format DataFrame with one row per (contrast × time_step).
        summary_df — one row per contrast with aggregate metrics.
    """
    # Build a lookup: (lex, lb) → sorted DataFrame indexed by time_idx
    traj: dict[tuple[str, str], pd.DataFrame] = {}
    for lex, lb in CONDITIONS:
        sub = centroids_df[
            (centroids_df["lexicality"] == lex) & (centroids_df["length_bin"] == lb)
        ].sort_values("time_idx").reset_index(drop=True)
        traj[(lex, lb)] = sub

    dist_rows: list[dict] = []
    summary_rows: list[dict] = []

    for contrast, family, cond_a, cond_b in _CONTRAST_PAIRS:
        df_a = traj.get(cond_a)
        df_b = traj.get(cond_b)
        if df_a is None or df_b is None or df_a.empty or df_b.empty:
            print(
                f"[pca_analysis] WARNING: skipping contrast={contrast} "
                f"— one or both conditions missing."
            )
            continue

        d_pc1 = df_a["pc1"].values - df_b["pc1"].values
        d_pc2 = df_a["pc2"].values - df_b["pc2"].values
        distances = np.sqrt(d_pc1 ** 2 + d_pc2 ** 2)
        time_norm = df_a["time_norm"].values
        time_idx = df_a["time_idx"].values

        for i in range(len(distances)):
            dist_rows.append({
                "run_id": run_id,
                "layer": layer,
                "contrast": contrast,
                "contrast_family": family,
                "time_idx": int(time_idx[i]),
                "time_norm": float(time_norm[i]),
                "distance": float(distances[i]),
            })

        # Summary metrics
        idx_max = int(np.argmax(distances))
        auc = float(np.trapezoid(distances, time_norm))
        summary_rows.append({
            "run_id": run_id,
            "layer": layer,
            "contrast": contrast,
            "contrast_family": family,
            "n_steps": len(distances),
            "mean_distance": float(distances.mean()),
            "max_distance": float(distances[idx_max]),
            "time_idx_of_max": int(time_idx[idx_max]),
            "time_norm_of_max": float(time_norm[idx_max]),
            "auc_distance": auc,
        })

    dist_df = pd.DataFrame(dist_rows)
    summary_df = pd.DataFrame(summary_rows)
    return dist_df, summary_df


def save_distance_figures(
    dist_df: pd.DataFrame,
    out_dir: Path,
    run_id: str,
    layer: str,
) -> None:
    """Save the condition-distance-over-time line plot.

    Each of the 4 contrasts is shown as one curve.
    Lexicality contrasts use solid lines; length contrasts use dashed lines.
    Colors separate the two lexicality contrasts and two length contrasts.

    Args:
        dist_df:  long-format DataFrame from compute_condition_distances().
        out_dir:  destination directory (must already exist).
        run_id:   used in the plot title.
        layer:    used in the plot title.
    """
    # Visual encoding: color by contrast_family, linestyle by specific contrast
    _FAMILY_COLORS = {"lexicality": "#0072B2", "length": "#D62828"}
    _CONTRAST_LS = {
        "lex_short":   "-",
        "lex_long":    "--",
        "len_word":    "-",
        "len_nonword": "--",
    }
    _CONTRAST_LABELS = {
        "lex_short":   "lexicality @ short  (word vs pseudoword)",
        "lex_long":    "lexicality @ long   (word vs pseudoword)",
        "len_word":    "length @ word       (short vs long)",
        "len_nonword": "length @ pseudoword (short vs long)",
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    for contrast, family, _, _ in _CONTRAST_PAIRS:
        sub = dist_df[dist_df["contrast"] == contrast].sort_values("time_norm")
        if sub.empty:
            continue
        ax.plot(
            sub["time_norm"], sub["distance"],
            color=_FAMILY_COLORS[family],
            linestyle=_CONTRAST_LS[contrast],
            linewidth=2.2,
            label=_CONTRAST_LABELS[contrast],
        )

    ax.set_xlabel("Normalized time", fontsize=11)
    ax.set_ylabel("Euclidean distance in PCA space", fontsize=11)
    ax.set_title(
        f"{run_id} — {layer}\nCondition-mean distances over normalized time",
        fontsize=11,
    )
    ax.legend(fontsize=9, framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = out_dir / "trajectory_condition_distances_over_time.png"
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
    parser.add_argument(
        "--filter",
        nargs="+",
        default=None,
        metavar="KEY=VALUE",
        help=(
            "Restrict items to those matching all KEY=VALUE pairs before running PCA. "
            "E.g. --filter length_bin=short. "
            "Applied to both the paradigm metadata and the manifest item list."
        ),
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

    # ── Optional trial filter ─────────────────────────────────────────────────
    if args.filter:
        trial_filter: dict[str, str] = {}
        for item in args.filter:
            if "=" not in item:
                print(
                    f"[pca_analysis] ERROR: --filter items must be KEY=VALUE, "
                    f"got: {item!r}",
                    file=sys.stderr,
                )
                sys.exit(1)
            key, val = item.split("=", 1)
            trial_filter[key.strip()] = val.strip()

        n_before = len(paradigm_df)
        for col, val in trial_filter.items():
            paradigm_df = paradigm_df[paradigm_df[col] == val].reset_index(drop=True)
        allowed_ids: set[str] = set(paradigm_df["item_id"].values)
        manifest["items"] = {
            k: v for k, v in manifest["items"].items() if k in allowed_ids
        }
        print(
            f"[pca_analysis] --filter {trial_filter}: "
            f"{n_before} → {len(paradigm_df)} items"
        )

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

            # Cartesian trajectory figures
            save_trajectory_figures(
                traj_result, layer_out, run_id, layer,
                overlay_per_condition=args.overlay_per_condition,
            )

            # Polar extension — runs automatically after Cartesian trajectory
            print(f"[pca_analysis] [polar] Layer={layer} — computing polar coordinates …")
            polar_df, center_pc1, center_pc2 = compute_polar_centroids(
                traj_result.centroids_df, run_id, layer,
            )
            save_polar_figures(
                polar_df=polar_df,
                center_pc1=center_pc1,
                center_pc2=center_pc2,
                out_dir=layer_out,
                run_id=run_id,
                layer=layer,
                resample_steps=traj_result.resample_steps,
                condition_counts=cond_counts,
            )

            # Condition-distance analysis
            print(f"[pca_analysis] [distance] Layer={layer} — computing condition distances …")
            dist_df, dist_summary_df = compute_condition_distances(
                traj_result.centroids_df, run_id, layer,
            )
            dist_csv = layer_out / "trajectory_condition_distances_over_time.csv"
            dist_df.to_csv(dist_csv, index=False)
            print(f"[pca_analysis] Written : {dist_csv}")
            dist_summary_csv = layer_out / "trajectory_condition_distances_summary.csv"
            dist_summary_df.to_csv(dist_summary_csv, index=False)
            print(f"[pca_analysis] Written : {dist_summary_csv}")
            save_distance_figures(dist_df, layer_out, run_id, layer)

    print(f"\n[pca_analysis] Done. All outputs under {out_root}")


if __name__ == "__main__":
    main()