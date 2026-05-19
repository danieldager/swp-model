#!/usr/bin/env python3
"""Summary panels comparing AuriStream phoneme analyses across all layers.

Reads per-layer CSV outputs produced by scripts/audio/auristream_phoneme_pca_norm.py
and assembles side-by-side comparison figures — one panel per analysis type.

This script works on any representation model's phoneme embeddings that follow
the same output layout (phoneme_pca_scores_{layer}.csv, etc.).

Run from swp-model/ repo root:

    python scripts/audio/auristream_phoneme_summary_panels.py \\
        --figures-dir reproduce/figures/audio/auristream_phonemes/auristream__9d3f269f/

Outputs (inside {figures-dir}/summary_panels/):
    summary_pca_by_position_all_layers.png
    summary_norm_by_position_start_all_layers.png
    summary_pca_by_lexicality_all_layers.png           (if lexicality present)
    summary_norm_by_position_lexicality_all_layers.png (if data present)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_LEX_COLORS: dict[str, str] = {"word": "#0072B2", "nonword": "#ED2727"}
_TYPE_COLORS: dict[str, str] = {
    "vowel":     "#E69F00",
    "consonant": "#56B4E9",
    "other":     "#999999",
}


def _display_lex(v: str) -> str:
    return "pseudoword" if v == "nonword" else v


def _layer_sort_key(name: str) -> tuple:
    if name == "embedding":
        return (0, 0)
    m = re.match(r"block_(\d+)$", name)
    return (1, int(m.group(1))) if m else (2, name)


def _read_pca_summary(layer_dir: Path, layer: str) -> dict | None:
    """Return explained variance from phoneme_pca_summary_{layer}.json, or None."""
    import json
    p = layer_dir / f"phoneme_pca_summary_{layer}.json"
    if p.exists():
        with open(p) as f:
            d = json.load(f)
        ev = d.get("explained_variance_ratio", [0, 0])
        return {"pc1": ev[0], "pc2": ev[1]}
    return None


# ── Panel 0: PCA by phoneme type ─────────────────────────────────────────────


def panel_pca_by_phoneme_type(layers: list[str], figures_dir: Path, out_path: Path) -> bool:
    """1 × n_layers scatter: PC1/PC2 coloured by phoneme type (vowel/consonant/other)."""
    n = len(layers)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), squeeze=False)
    drawn = False
    for col, layer in enumerate(layers):
        ax = axes[0, col]
        csv = figures_dir / layer / f"phoneme_pca_scores_{layer}.csv"
        if not csv.exists():
            ax.set_visible(False)
            continue
        df = pd.read_csv(csv)
        if "phoneme_type" not in df.columns:
            ax.set_visible(False)
            continue
        ev = _read_pca_summary(figures_dir / layer, layer) or {"pc1": 0, "pc2": 0}
        for ptype, color in _TYPE_COLORS.items():
            sel = df["phoneme_type"] == ptype
            if sel.any():
                label = ptype if col == n - 1 else None
                ax.scatter(df.loc[sel, "pc1"], df.loc[sel, "pc2"],
                           c=color, label=label, s=5, alpha=0.6)
        ax.axhline(0, color="lightgray", lw=0.5)
        ax.axvline(0, color="lightgray", lw=0.5)
        ax.set_title(layer, fontsize=10)
        ax.set_xlabel(f"PC1 ({ev['pc1']*100:.1f}%)", fontsize=8)
        ax.set_ylabel(f"PC2 ({ev['pc2']*100:.1f}%)" if col == 0 else "", fontsize=8)
        if col == n - 1:
            ax.legend(markerscale=2, fontsize=8)
        drawn = True
    if not drawn:
        plt.close(fig)
        return False
    fig.suptitle("PCA by phoneme type (vowel / consonant / other)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


# ── Panel 1: PCA by phoneme position ─────────────────────────────────────────


def panel_pca_by_position(layers: list[str], figures_dir: Path, out_path: Path) -> bool:
    """1 × n_layers scatter: PC1/PC2 coloured by phoneme_position_from_start."""
    n = len(layers)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), squeeze=False)
    drawn = False
    for col, layer in enumerate(layers):
        ax = axes[0, col]
        csv = figures_dir / layer / f"phoneme_pca_scores_{layer}.csv"
        if not csv.exists():
            ax.set_visible(False)
            continue
        df = pd.read_csv(csv)
        pos = df["phoneme_position_from_start"].values
        ev = _read_pca_summary(figures_dir / layer, layer) or {"pc1": 0, "pc2": 0}
        sc = ax.scatter(df["pc1"], df["pc2"], c=pos, cmap="viridis", s=5, alpha=0.55)
        plt.colorbar(sc, ax=ax, label="position" if col == n - 1 else "")
        ax.axhline(0, color="lightgray", lw=0.5)
        ax.axvline(0, color="lightgray", lw=0.5)
        ax.set_title(layer, fontsize=10)
        ax.set_xlabel(f"PC1 ({ev['pc1']*100:.1f}%)", fontsize=8)
        ax.set_ylabel(f"PC2 ({ev['pc2']*100:.1f}%)" if col == 0 else "", fontsize=8)
        drawn = True
    if not drawn:
        plt.close(fig)
        return False
    fig.suptitle("PCA by phoneme position (from start)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


# ── Panel 2: PCA by lexicality ────────────────────────────────────────────────


def panel_pca_by_lexicality(layers: list[str], figures_dir: Path, out_path: Path) -> bool:
    n = len(layers)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), squeeze=False)
    drawn = False
    for col, layer in enumerate(layers):
        ax = axes[0, col]
        csv = figures_dir / layer / f"phoneme_pca_scores_{layer}.csv"
        if not csv.exists():
            ax.set_visible(False)
            continue
        df = pd.read_csv(csv)
        if "lexicality" not in df.columns or df["lexicality"].isna().all():
            ax.set_visible(False)
            continue
        ev = _read_pca_summary(figures_dir / layer, layer) or {"pc1": 0, "pc2": 0}
        for lex, color in _LEX_COLORS.items():
            sel = df["lexicality"] == lex
            if sel.any():
                ax.scatter(df.loc[sel, "pc1"], df.loc[sel, "pc2"],
                           c=color, label=_display_lex(lex), s=5, alpha=0.6)
        ax.axhline(0, color="lightgray", lw=0.5)
        ax.axvline(0, color="lightgray", lw=0.5)
        ax.set_title(layer, fontsize=10)
        ax.set_xlabel(f"PC1 ({ev['pc1']*100:.1f}%)", fontsize=8)
        ax.set_ylabel(f"PC2 ({ev['pc2']*100:.1f}%)" if col == 0 else "", fontsize=8)
        if col == n - 1:
            ax.legend(markerscale=2, fontsize=8)
        drawn = True
    if not drawn:
        plt.close(fig)
        return False
    fig.suptitle("PCA by lexicality", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


# ── Panel 3: norm by position from start ─────────────────────────────────────


def panel_norm_by_position_start(layers: list[str], figures_dir: Path, out_path: Path) -> bool:
    n = len(layers)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    drawn = False

    # Compute shared y-range across layers for comparability
    all_means: list[float] = []
    for layer in layers:
        csv = figures_dir / layer / f"norm_by_position_{layer}.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            sub = df[df["pos_axis"] == "from_start"]
            all_means.extend((sub["mean"] + sub["sem"]).tolist())
            all_means.extend((sub["mean"] - sub["sem"]).tolist())
    ymax = max(all_means) * 1.05 if all_means else 1.0
    ymin = min(0.0, min(all_means)) if all_means else 0.0

    for col, layer in enumerate(layers):
        ax = axes[0, col]
        csv = figures_dir / layer / f"norm_by_position_{layer}.csv"
        if not csv.exists():
            ax.set_visible(False)
            continue
        df = pd.read_csv(csv)
        sub = df[df["pos_axis"] == "from_start"].sort_values("position")
        ax.plot(sub["position"], sub["mean"], marker="o", ms=4, lw=1.5, color="#333333")
        ax.fill_between(sub["position"],
                        sub["mean"] - sub["sem"],
                        sub["mean"] + sub["sem"],
                        alpha=0.25, color="#333333")
        ax.set_title(layer, fontsize=10)
        ax.set_xlabel("position (from start)", fontsize=8)
        ax.set_ylabel("L2 norm (mean ± SEM)" if col == 0 else "", fontsize=8)
        ax.set_xticks(sorted(sub["position"].unique()))
        ax.set_ylim(ymin, ymax)
        drawn = True
    if not drawn:
        plt.close(fig)
        return False
    fig.suptitle("L2 norm by phoneme position (from start)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


# ── Panel 3b: norm by position from start — free y-axis ─────────────────────


def panel_norm_by_position_start_free_y(layers: list[str], figures_dir: Path, out_path: Path) -> bool:
    """Same as panel_norm_by_position_start but each subplot has its own y-scale."""
    n = len(layers)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    drawn = False
    for col, layer in enumerate(layers):
        ax = axes[0, col]
        csv = figures_dir / layer / f"norm_by_position_{layer}.csv"
        if not csv.exists():
            ax.set_visible(False)
            continue
        df = pd.read_csv(csv)
        sub = df[df["pos_axis"] == "from_start"].sort_values("position")
        ax.plot(sub["position"], sub["mean"], marker="o", ms=4, lw=1.5, color="#333333")
        ax.fill_between(sub["position"],
                        sub["mean"] - sub["sem"],
                        sub["mean"] + sub["sem"],
                        alpha=0.25, color="#333333")
        ax.set_title(layer, fontsize=10)
        ax.set_xlabel("position (from start)", fontsize=8)
        ax.set_ylabel("L2 norm (mean ± SEM)" if col == 0 else "", fontsize=8)
        ax.set_xticks(sorted(sub["position"].unique()))
        drawn = True
    if not drawn:
        plt.close(fig)
        return False
    fig.suptitle("L2 norm by phoneme position (from start) — free y-axis", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


# ── Panel 3c: norm by position from start — normalised to position 0 ─────────


def panel_norm_by_position_start_normalized(layers: list[str], figures_dir: Path, out_path: Path) -> bool:
    """Norm curves divided by the position-0 mean, so all layers start at 1.0."""
    n = len(layers)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    drawn = False
    for col, layer in enumerate(layers):
        ax = axes[0, col]
        csv = figures_dir / layer / f"norm_by_position_{layer}.csv"
        if not csv.exists():
            ax.set_visible(False)
            continue
        df = pd.read_csv(csv)
        sub = df[df["pos_axis"] == "from_start"].sort_values("position")
        scale = sub.loc[sub["position"] == sub["position"].min(), "mean"].values
        if len(scale) == 0 or scale[0] == 0:
            ax.set_visible(False)
            continue
        scale = scale[0]
        mean_norm = sub["mean"] / scale
        sem_norm  = sub["sem"]  / scale
        ax.plot(sub["position"], mean_norm, marker="o", ms=4, lw=1.5, color="#333333")
        ax.fill_between(sub["position"],
                        mean_norm - sem_norm,
                        mean_norm + sem_norm,
                        alpha=0.25, color="#333333")
        ax.axhline(1.0, color="lightgray", lw=0.8, ls="--")
        ax.set_title(layer, fontsize=10)
        ax.set_xlabel("position (from start)", fontsize=8)
        ax.set_ylabel("norm / norm(pos 0)" if col == 0 else "", fontsize=8)
        ax.set_xticks(sorted(sub["position"].unique()))
        drawn = True
    if not drawn:
        plt.close(fig)
        return False
    fig.suptitle("L2 norm by phoneme position (normalised to position 0)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


# ── Panel 4: norm by position × lexicality ───────────────────────────────────


def panel_norm_by_position_lexicality(layers: list[str], figures_dir: Path, out_path: Path) -> bool:
    n = len(layers)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    drawn = False
    for col, layer in enumerate(layers):
        ax = axes[0, col]
        csv = figures_dir / layer / f"norm_by_position_lexicality_{layer}.csv"
        if not csv.exists():
            ax.set_visible(False)
            continue
        df = pd.read_csv(csv).sort_values(["factor", "position"])
        for lex, color in _LEX_COLORS.items():
            sub = df[df["factor"] == lex]
            if sub.empty:
                continue
            label = _display_lex(lex) if col == n - 1 else None
            line = ax.plot(sub["position"], sub["mean"],
                           marker="o", ms=4, lw=1.5, color=color, label=label)[0]
            ax.fill_between(sub["position"],
                            sub["mean"] - sub["sem"],
                            sub["mean"] + sub["sem"],
                            alpha=0.2, color=line.get_color())
        ax.set_title(layer, fontsize=10)
        ax.set_xlabel("position (from start)", fontsize=8)
        ax.set_ylabel("L2 norm (mean ± SEM)" if col == 0 else "", fontsize=8)
        positions = sorted(df["position"].unique())
        ax.set_xticks(positions)
        if col == n - 1:
            ax.legend(fontsize=8)
        drawn = True
    if not drawn:
        plt.close(fig)
        return False
    fig.suptitle("L2 norm by position × lexicality", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summary panels comparing phoneme embedding analyses across layers."
    )
    parser.add_argument(
        "--figures-dir", required=True, dest="figures_dir",
        help=(
            "Analysis output directory "
            "(e.g. reproduce/figures/audio/auristream_phonemes/auristream__9d3f269f/)"
        ),
    )
    parser.add_argument(
        "--layers", nargs="+", default=None,
        help="Layers to include. Default: all layer subdirectories found.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory for panels. Default: {figures-dir}/summary_panels/",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output directory.",
    )
    args = parser.parse_args()

    figures_dir = Path(args.figures_dir)
    if not figures_dir.exists():
        sys.exit(f"Figures directory not found: {figures_dir}")

    # Discover layers
    if args.layers:
        layers = sorted(args.layers, key=_layer_sort_key)
    else:
        found = [
            d.name for d in figures_dir.iterdir()
            if d.is_dir() and (d / f"phoneme_pca_scores_{d.name}.csv").exists()
        ]
        if not found:
            sys.exit("No layer subdirectories with phoneme_pca_scores_*.csv found.")
        layers = sorted(found, key=_layer_sort_key)

    output_dir = Path(args.output) if args.output else figures_dir / "summary_panels"
    if output_dir.exists() and not args.overwrite:
        sys.exit(
            f"Output directory already exists: {output_dir}\n"
            "Pass --overwrite to overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Figures dir : {figures_dir}")
    print(f"Layers      : {layers}")
    print(f"Output      : {output_dir}")

    panels = [
        (
            "summary_pca_by_phoneme_type_all_layers.png",
            panel_pca_by_phoneme_type,
        ),
        (
            "summary_pca_by_position_all_layers.png",
            panel_pca_by_position,
        ),
        (
            "summary_norm_by_position_start_all_layers.png",
            panel_norm_by_position_start,
        ),
        (
            "summary_norm_by_position_start_all_layers_free_y.png",
            panel_norm_by_position_start_free_y,
        ),
        (
            "summary_norm_by_position_start_all_layers_normalized.png",
            panel_norm_by_position_start_normalized,
        ),
        (
            "summary_pca_by_lexicality_all_layers.png",
            panel_pca_by_lexicality,
        ),
        (
            "summary_norm_by_position_lexicality_all_layers.png",
            panel_norm_by_position_lexicality,
        ),
    ]

    for fname, fn in panels:
        out = output_dir / fname
        ok = fn(layers, figures_dir, out)
        if ok:
            print(f"  Saved: {fname}")
        else:
            print(f"  Skipped (data not available): {fname}")

    print(f"\nDone: {output_dir}")


if __name__ == "__main__":
    main()
