#!/usr/bin/env python3
"""Summary 2×2 panel figures for the audio PCA analysis.

Reads existing CSV outputs from pca_analysis.py and assembles compact
cross-model / cross-layer summary figures for presentation.
Does NOT recompute any PCA — purely a post-processing / plotting script.

Layout of each panel
--------------------
    rows  = models  : EnCodec (encodec__7f7d3b97) | DAC (dac__f104bbdd)
    cols  = layers  : encoder_out | decoder_in

Figures produced
----------------
summary_condition_distances_2x2.png
    4-panel distance-over-time plot.
    Source: trajectory_condition_distances_over_time.csv (one per panel).

summary_trajectory_means_2x2.png
    4-panel condition-mean trajectory plot.
    Source: trajectory_condition_means_centroids.csv (one per panel).

Output location
---------------
reproduce/figures/audio/pca/summary/

Usage
-----
    # Default: uses built-in run_ids and layers, auto-detects pca figure root
    python reproduce/scripts/audio/pca_summary.py

    # Custom pca figure root (if --output was used in pca_analysis.py)
    python reproduce/scripts/audio/pca_summary.py \\
        --pca-root /custom/path/to/pca/figures/

    # Custom list of run_ids (order = top row, bottom row)
    python reproduce/scripts/audio/pca_summary.py \\
        --runs encodec__7f7d3b97 dac__f104bbdd

    # Custom output directory
    python reproduce/scripts/audio/pca_summary.py \\
        --output /tmp/summary_figs/
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Resolve repo root: reproduce/scripts/audio/ → 3 levels up → swp-model/
REPO_ROOT = Path(__file__).resolve().parents[3]

# ── Style constants — must match pca_analysis.py exactly ─────────────────────

_CONDITIONS: list[tuple[str, str]] = [
    ("word",    "short"),
    ("word",    "long"),
    ("nonword", "short"),
    ("nonword", "long"),
]

_COND_COLORS: dict[tuple[str, str], str] = {
    ("word",    "short"): "#2EC4B6",
    ("word",    "long"):  "#1565C0",
    ("nonword", "short"): "#FFB703",
    ("nonword", "long"):  "#D62828",
}
_COND_LINESTYLES: dict[str, str] = {"short": "-", "long": "--"}

_CONTRAST_ORDER = ["lex_short", "lex_long", "len_word", "len_nonword"]
_FAMILY_COLORS  = {"lexicality": "#0072B2", "length": "#D62828"}
_CONTRAST_LS    = {"lex_short": "-", "lex_long": "--", "len_word": "-", "len_nonword": "--"}
_CONTRAST_LABELS = {
    "lex_short":   "lex @ short  (word vs nonword)",
    "lex_long":    "lex @ long   (word vs nonword)",
    "len_word":    "length @ word    (short vs long)",
    "len_nonword": "length @ nonword (short vs long)",
}

# Friendly model labels for panel row titles
_MODEL_LABELS: dict[str, str] = {
    "encodec__7f7d3b97": "EnCodec",
    "dac__f104bbdd":     "DAC",
}

_LAYER_LABELS: dict[str, str] = {
    "encoder_out": "encoder_out",
    "decoder_in":  "decoder_in",
}

DEFAULT_RUNS   = ["encodec__7f7d3b97", "dac__f104bbdd"]
DEFAULT_LAYERS = ["encoder_out", "decoder_in"]


# ── CSV loading helpers ───────────────────────────────────────────────────────


def _load_csv(path: Path, run_id: str, layer: str, pca_root: Path) -> pd.DataFrame | None:
    """Load a CSV, or print a helpful error and return None if missing.

    Args:
        path:     absolute path to the CSV file.
        run_id:   used only in the error message.
        layer:    used only in the error message.
        pca_root: used only in the error message.

    Returns:
        DataFrame if file exists, None otherwise.
    """
    if not path.exists():
        print(
            f"[pca_summary] MISSING: {path}\n"
            f"  → Run pca_analysis.py first:\n"
            f"      python reproduce/scripts/audio/pca_analysis.py \\\n"
            f"          --run reproduce/data/audio/{run_id}/ \\\n"
            f"          --mode trajectory --layers {layer}",
            file=sys.stderr,
        )
        return None
    return pd.read_csv(path)


# ── Panel drawing helpers ─────────────────────────────────────────────────────


def _draw_distance_panel(
    ax: plt.Axes,
    dist_df: pd.DataFrame,
    row_label: str,
    col_label: str,
    sharey_max: float | None,
) -> None:
    """Draw condition-distance curves on one axes panel.

    Args:
        ax:          target Axes.
        dist_df:     DataFrame from trajectory_condition_distances_over_time.csv.
        row_label:   model label for the row (used in panel title).
        col_label:   layer label for the column (used in panel title).
        sharey_max:  if not None, fix y-axis upper limit to this value.
    """
    for contrast in _CONTRAST_ORDER:
        sub = dist_df[dist_df["contrast"] == contrast].sort_values("time_norm")
        if sub.empty:
            continue
        family = sub["contrast_family"].iloc[0]
        ax.plot(
            sub["time_norm"], sub["distance"],
            color=_FAMILY_COLORS[family],
            linestyle=_CONTRAST_LS[contrast],
            linewidth=1.8,
            label=_CONTRAST_LABELS[contrast],
        )

    ax.set_title(f"{row_label} / {col_label}", fontsize=9)
    if sharey_max is not None:
        ax.set_ylim(bottom=0, top=sharey_max * 1.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_trajectory_panel(
    ax: plt.Axes,
    centroids_df: pd.DataFrame,
    row_label: str,
    col_label: str,
) -> None:
    """Draw condition-mean trajectories on one axes panel.

    Args:
        ax:           target Axes.
        centroids_df: DataFrame from trajectory_condition_means_centroids.csv.
        row_label:    model label (used in panel title).
        col_label:    layer label (used in panel title).
    """
    # Compute per-panel axis limits for consistent centering
    pc1_vals = centroids_df["pc1"]
    pc2_vals = centroids_df["pc2"]
    ax.axhline(0, color="grey", linewidth=0.4, linestyle="--", alpha=0.4)
    ax.axvline(0, color="grey", linewidth=0.4, linestyle="--", alpha=0.4)

    for lex, lb in _CONDITIONS:
        sub = centroids_df[
            (centroids_df["lexicality"] == lex) & (centroids_df["length_bin"] == lb)
        ].sort_values("time_idx")
        if sub.empty:
            continue
        color = _COND_COLORS.get((lex, lb), "#999999")
        ls = _COND_LINESTYLES.get(lb, "-")
        ax.plot(sub["pc1"], sub["pc2"], color=color, linestyle=ls,
                linewidth=1.8, label=f"{lex} / {lb}")
        # Start (circle) and end (square) markers
        ax.scatter(sub["pc1"].iloc[0], sub["pc2"].iloc[0],
                   color=color, marker="o", s=50, zorder=5,
                   edgecolors="white", linewidths=0.8)
        ax.scatter(sub["pc1"].iloc[-1], sub["pc2"].iloc[-1],
                   color=color, marker="s", s=50, zorder=5,
                   edgecolors="white", linewidths=0.8)

    ax.set_title(f"{row_label} / {col_label}", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Figure assembly ───────────────────────────────────────────────────────────


def make_distance_summary(
    runs: list[str],
    layers: list[str],
    pca_root: Path,
    out_dir: Path,
) -> None:
    """Build and save summary_condition_distances_2x2.png.

    Args:
        runs:     list of run_id strings (one per row).
        layers:   list of layer names (one per column).
        pca_root: root of the per-run/per-layer PCA figure tree.
        out_dir:  destination directory for the summary figure.
    """
    n_rows, n_cols = len(runs), len(layers)
    print(f"[pca_summary] Building distance 2×2 …")

    # Load all data first so we can compute a global y-max for sharey
    data: dict[tuple[str, str], pd.DataFrame] = {}
    any_loaded = False
    for run_id in runs:
        for layer in layers:
            csv = pca_root / run_id / layer / "trajectory_condition_distances_over_time.csv"
            df = _load_csv(csv, run_id, layer, pca_root)
            if df is not None:
                data[(run_id, layer)] = df
                any_loaded = True

    if not any_loaded:
        print("[pca_summary] No distance CSVs found — aborting distance panel.", file=sys.stderr)
        return

    global_ymax = max(
        df["distance"].max() for df in data.values()
    )

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 3.8 * n_rows),
        sharex=True, sharey=True,
        squeeze=False,
    )

    for r, run_id in enumerate(runs):
        for c, layer in enumerate(layers):
            ax = axes[r][c]
            row_label = _MODEL_LABELS.get(run_id, run_id)
            col_label = _LAYER_LABELS.get(layer, layer)
            df = data.get((run_id, layer))
            if df is None:
                ax.set_visible(False)
                continue
            _draw_distance_panel(ax, df, row_label, col_label, global_ymax)

            # Axis labels only on border panels
            if r == n_rows - 1:
                ax.set_xlabel("Normalized time", fontsize=9)
            if c == 0:
                ax.set_ylabel("Distance in PCA space", fontsize=9)

    # Single shared legend from the last valid panel
    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    fig.legend(handles, labels, loc="lower center", ncol=2,
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Condition-mean trajectory distances over normalized time",
                 fontsize=12, y=1.01)
    plt.tight_layout()

    out_path = out_dir / "summary_condition_distances_2x2.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[pca_summary] Written : {out_path}")


def make_trajectory_summary(
    runs: list[str],
    layers: list[str],
    pca_root: Path,
    out_dir: Path,
) -> None:
    """Build and save summary_trajectory_means_2x2.png.

    Args:
        runs:     list of run_id strings (one per row).
        layers:   list of layer names (one per column).
        pca_root: root of the per-run/per-layer PCA figure tree.
        out_dir:  destination directory for the summary figure.
    """
    n_rows, n_cols = len(runs), len(layers)
    print(f"[pca_summary] Building trajectory 2×2 …")

    data: dict[tuple[str, str], pd.DataFrame] = {}
    any_loaded = False
    for run_id in runs:
        for layer in layers:
            csv = pca_root / run_id / layer / "trajectory_condition_means_centroids.csv"
            df = _load_csv(csv, run_id, layer, pca_root)
            if df is not None:
                data[(run_id, layer)] = df
                any_loaded = True

    if not any_loaded:
        print("[pca_summary] No centroid CSVs found — aborting trajectory panel.", file=sys.stderr)
        return

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 5 * n_rows),
        squeeze=False,
    )

    for r, run_id in enumerate(runs):
        for c, layer in enumerate(layers):
            ax = axes[r][c]
            row_label = _MODEL_LABELS.get(run_id, run_id)
            col_label = _LAYER_LABELS.get(layer, layer)
            df = data.get((run_id, layer))
            if df is None:
                ax.set_visible(False)
                continue
            _draw_trajectory_panel(ax, df, row_label, col_label)

            if r == n_rows - 1:
                ax.set_xlabel("PC1", fontsize=9)
            if c == 0:
                ax.set_ylabel("PC2", fontsize=9)

    # Single shared legend
    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    fig.legend(handles, labels, loc="lower center", ncol=2,
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Condition-mean PCA trajectories", fontsize=12, y=1.01)
    plt.tight_layout()

    out_path = out_dir / "summary_trajectory_means_2x2.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[pca_summary] Written : {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build 2×2 summary figures from existing pca_analysis.py outputs.\n"
            "Does NOT recompute PCA — reads CSVs only."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=DEFAULT_RUNS,
        metavar="RUN_ID",
        help=(
            f"Run IDs to include, in row order "
            f"(default: {DEFAULT_RUNS})."
        ),
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=DEFAULT_LAYERS,
        metavar="LAYER",
        help=(
            f"Layers to include, in column order "
            f"(default: {DEFAULT_LAYERS})."
        ),
    )
    parser.add_argument(
        "--pca-root",
        default=None,
        help=(
            "Root directory of per-run/per-layer PCA figures "
            "(default: reproduce/figures/audio/pca/ relative to repo root)."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output directory for summary figures "
            "(default: reproduce/figures/audio/pca/summary/ relative to repo root)."
        ),
    )
    args = parser.parse_args()

    pca_root = (
        Path(args.pca_root).resolve()
        if args.pca_root
        else REPO_ROOT / "reproduce" / "figures" / "audio" / "pca"
    )
    out_dir = (
        Path(args.output).resolve()
        if args.output
        else pca_root / "summary"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[pca_summary] PCA root : {pca_root}")
    print(f"[pca_summary] Output   : {out_dir}")
    print(f"[pca_summary] Runs     : {args.runs}")
    print(f"[pca_summary] Layers   : {args.layers}")
    print()

    # Print input CSVs being read
    print("[pca_summary] Input CSVs:")
    for run_id in args.runs:
        for layer in args.layers:
            base = pca_root / run_id / layer
            print(f"  {base / 'trajectory_condition_distances_over_time.csv'}")
            print(f"  {base / 'trajectory_condition_means_centroids.csv'}")
    print()

    plt.rcParams.update({
        "axes.grid": True,
        "axes.grid.axis": "both",
        "axes.facecolor": "white",
        "grid.color": "0.88",
        "grid.linewidth": 0.7,
        "font.size": 9,
    })

    make_distance_summary(args.runs, args.layers, pca_root, out_dir)
    make_trajectory_summary(args.runs, args.layers, pca_root, out_dir)

    print(f"\n[pca_summary] Done. Figures written to {out_dir}")


if __name__ == "__main__":
    main()