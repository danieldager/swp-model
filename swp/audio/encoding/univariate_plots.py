"""Matplotlib visualisations for univariate encoding summary CSVs.

All public functions accept DataFrames, save a PNG, and close the figure.
No interactive display; Agg backend is set at import time.

Public API
----------
plot_score_over_time(score_summary, metric, output_path) -> bool
plot_weights_over_time(weight_summary, analysis_set, output_path)
plot_global_feature_ranking(feature_ranking, output_path)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # must come before pyplot import; no GUI needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_DPI = 200

# Consistent visual encoding across all figures
_LAYER_COLOR  = {"encoder_out": "#1f77b4", "decoder_in": "#ff7f0e"}
_ASET_LS      = {"all_items": "-",         "words_only": "--"}
_FEAT_COLOR   = {
    "lexicality":    "#2ca02c",
    "length_bin":    "#1f77b4",
    "morphology":    "#d62728",
    "frequency_bin": "#9467bd",
}

_LAYER_ORDER = ["encoder_out", "decoder_in"]
_ASET_ORDER  = ["all_items",   "words_only"]


def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


# ── Figure 1 & 2 ─────────────────────────────────────────────────────────────


def plot_score_over_time(
    score_summary: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> bool:
    """Line plot of mean_score ± std_score over relative_time.

    Four lines: 2 layers × 2 analysis_sets. Shaded band = ±1 std across neurons.
    Horizontal dotted line at y=0 for reference.

    Args:
        score_summary: DataFrame from score_summary_by_time.csv.
        metric:        'r2' or 'spearman'. If absent, returns False without saving.
        output_path:   Where to write the PNG.

    Returns:
        True if figure was saved, False if metric not present in data.
    """
    df = score_summary[score_summary["metric"] == metric]
    if df.empty:
        return False

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":", zorder=0)

    layers = [l for l in _LAYER_ORDER if l in df["layer"].unique()]
    asets  = [a for a in _ASET_ORDER  if a in df["analysis_set"].unique()]

    for layer in layers:
        for aset in asets:
            sub = (
                df[(df["layer"] == layer) & (df["analysis_set"] == aset)]
                .sort_values("relative_time")
            )
            if sub.empty:
                continue
            color = _LAYER_COLOR.get(layer)
            ls    = _ASET_LS.get(aset, "-")
            x = sub["relative_time"].values
            y = sub["mean_score"].values
            s = sub["std_score"].values
            ax.plot(x, y, color=color, linestyle=ls, linewidth=1.6,
                    label=f"{layer} / {aset}", marker="o", markersize=3)
            ax.fill_between(x, y - s, y + s, color=color, alpha=0.12)

    ax.set_xlabel("Relative time (bin centre)")
    ax.set_ylabel(f"Mean {metric.capitalize()} score across neurons")
    ax.set_title(f"Univariate encoding — {metric.capitalize()} over time")
    ax.legend(fontsize=8, loc="best")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    _savefig(fig, output_path)
    return True


# ── Figure 3 & 4 ─────────────────────────────────────────────────────────────


def plot_weights_over_time(
    weight_summary: pd.DataFrame,
    analysis_set: str,
    output_path: Path,
) -> None:
    """Two-panel plot of mean_abs_weight over relative_time, one panel per layer.

    Each panel shows one curve per feature. Y axes are shared across panels.

    Args:
        weight_summary: DataFrame from weight_summary_by_feature_time.csv.
        analysis_set:   'all_items' or 'words_only'.
        output_path:    Where to write the PNG.
    """
    df = weight_summary[weight_summary["analysis_set"] == analysis_set]
    layers = [l for l in _LAYER_ORDER if l in df["layer"].unique()]
    n_panels = len(layers)

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        sub_layer = df[df["layer"] == layer]
        # Consistent feature order: sort by overall mean desc for readability
        feat_order = (
            sub_layer.groupby("feature")["mean_abs_weight"].mean()
            .sort_values(ascending=False).index.tolist()
        )
        for feat in feat_order:
            sub = sub_layer[sub_layer["feature"] == feat].sort_values("relative_time")
            if sub.empty:
                continue
            color = _FEAT_COLOR.get(feat)
            x = sub["relative_time"].values
            y = sub["mean_abs_weight"].values
            ax.plot(x, y, color=color, linewidth=1.6,
                    label=feat, marker="o", markersize=3)

        ax.set_title(layer, fontsize=10)
        ax.set_xlabel("Relative time (bin centre)")
        if ax is axes[0]:
            ax.set_ylabel("Mean |weight| across neurons")
        ax.legend(fontsize=8, loc="best")
        ax.set_xlim(0, 1)

    fig.suptitle(f"Feature weights over time — {analysis_set}", fontsize=11)
    fig.tight_layout()
    _savefig(fig, output_path)


# ── Figure 5 ─────────────────────────────────────────────────────────────────


def plot_global_feature_ranking(
    feature_ranking: pd.DataFrame,
    output_path: Path,
) -> None:
    """2×2 bar plot of mean_abs_weight by feature, one panel per (layer × analysis_set).

    Bars are labelled with their numeric value. Features are ordered by rank.

    Args:
        feature_ranking: DataFrame from global_feature_ranking.csv.
        output_path:     Where to write the PNG.
    """
    layers = [l for l in _LAYER_ORDER if l in feature_ranking["layer"].unique()]
    asets  = [a for a in _ASET_ORDER  if a in feature_ranking["analysis_set"].unique()]

    n_rows, n_cols = len(asets), len(layers)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), sharey=False
    )

    # Normalise axes array to always be 2-D
    axes = np.array(axes)
    if axes.ndim == 0:
        axes = axes.reshape(1, 1)
    elif n_rows == 1 and n_cols > 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1 and n_rows > 1:
        axes = axes.reshape(-1, 1)

    for r, aset in enumerate(asets):
        for c, layer in enumerate(layers):
            ax = axes[r, c]
            sub = (
                feature_ranking[
                    (feature_ranking["layer"] == layer) &
                    (feature_ranking["analysis_set"] == aset)
                ]
                .sort_values("rank_by_mean_abs_weight")
            )
            features = sub["feature"].tolist()
            values   = sub["mean_abs_weight_over_time_and_neurons"].tolist()
            colors   = [_FEAT_COLOR.get(f, "#888888") for f in features]

            bars = ax.bar(features, values, color=colors, edgecolor="white", linewidth=0.5)
            ax.set_title(f"{layer}\n{aset}", fontsize=9)
            ax.set_ylabel("Mean |weight|" if c == 0 else "")
            ax.tick_params(axis="x", rotation=15, labelsize=8)

            y_max = max(values) if values else 1.0
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_max * 0.02,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7,
                )

    fig.suptitle("Global feature ranking — mean |weight| over time × neurons", fontsize=11)
    fig.tight_layout()
    _savefig(fig, output_path)