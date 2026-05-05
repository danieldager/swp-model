"""Matplotlib visualisations for univariate encoding summary CSVs.

All public functions accept DataFrames, save a PNG, and close the figure.
No interactive display; Agg backend is set at import time.

Public API
----------
plot_score_over_time(score_summary, metric, output_path) -> bool
plot_weights_over_time(weight_summary, analysis_set, output_path)
plot_global_feature_ranking(feature_ranking, output_path)
plot_encoding_fi(fi_df, scores_df, layer, analysis_set, metric, output_path) -> bool
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
    model_name: str = "",
    run_id: str = "",
    analysis_view: str = "",
) -> bool:
    """Line plot of mean_score ± std_score over relative_time.

    Four lines: 2 layers × 2 analysis_sets. Shaded band = ±1 std across neurons.
    Horizontal dotted line at y=0 for reference.

    Args:
        score_summary: DataFrame from score_summary_by_time.csv.
        metric:        'r2' or 'spearman'. If absent, returns False without saving.
        output_path:   Where to write the PNG.
        model_name:    Shown in title (e.g. 'encodec').
        run_id:        Shown as subtitle (e.g. 'encodec__7f7d3b97').

    Returns:
        True if figure was saved, False if metric not present in data.
    """
    df = score_summary[score_summary["metric"] == metric]
    if analysis_view and "analysis_view" in df.columns:
        df = df[df["analysis_view"] == analysis_view]
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
    _prefix = f" — {model_name}" if model_name else ""
    _view   = f" — {analysis_view}" if analysis_view else ""
    _sub    = f"\nrun_id: {run_id}" if run_id else ""
    ax.set_title(f"Univariate encoding{_prefix}{_view} — {metric.capitalize()} over time{_sub}")
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
    model_name: str = "",
    run_id: str = "",
    analysis_view: str = "",
) -> None:
    """Two-panel plot of mean_abs_weight over relative_time, one panel per layer.

    Each panel shows one curve per feature. Y axes are shared across panels.

    Args:
        weight_summary: DataFrame from weight_summary_by_feature_time.csv.
        analysis_set:   'all_items' or 'words_only'.
        output_path:    Where to write the PNG.
        model_name:     Shown in suptitle.
        run_id:         Shown as subtitle.
    """
    df = weight_summary[weight_summary["analysis_set"] == analysis_set]
    if analysis_view and "analysis_view" in df.columns:
        df = df[df["analysis_view"] == analysis_view]
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

    _prefix = f" — {model_name}" if model_name else ""
    _label  = analysis_view if analysis_view else analysis_set
    _sub    = f"\nrun_id: {run_id}" if run_id else ""
    fig.suptitle(f"Feature weights over time{_prefix} — {_label}{_sub}", fontsize=11)
    fig.tight_layout()
    _savefig(fig, output_path)


# ── Figure 5 ─────────────────────────────────────────────────────────────────


def plot_global_feature_ranking(
    feature_ranking: pd.DataFrame,
    output_path: Path,
    model_name: str = "",
    run_id: str = "",
    analysis_view: str = "",
) -> None:
    """2×2 bar plot of mean_abs_weight by feature, one panel per (layer × analysis_set).

    Bars are labelled with their numeric value. Features are ordered by rank.

    Args:
        feature_ranking: DataFrame from global_feature_ranking.csv.
        output_path:     Where to write the PNG.
        model_name:      Shown in suptitle.
        run_id:          Shown as subtitle.
    """
    if analysis_view and "analysis_view" in feature_ranking.columns:
        feature_ranking = feature_ranking[feature_ranking["analysis_view"] == analysis_view]
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

    _prefix = f" — {model_name}" if model_name else ""
    _view   = f" — {analysis_view}" if analysis_view else ""
    _sub    = f"\nrun_id: {run_id}" if run_id else ""
    fig.suptitle(f"Global feature ranking{_prefix}{_view}\nmean |weight| over time × neurons{_sub}", fontsize=11)
    fig.tight_layout()
    _savefig(fig, output_path)


# ── Figure 6 (optional, requires --compute-fi run) ───────────────────────────


def plot_encoding_fi(
    fi_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    layer: str,
    analysis_set: str,
    metric: str,
    output_path: Path,
    aggregation: str = "mean",
    model_name: str = "",
    run_id: str = "",
    analysis_view: str = "",
) -> bool:
    """Dual-axis plot: permutation FI (left) + encoding score (right) over time.

    Adapted from Intracranial Plotter.encoding_FI(). Instead of one electrode,
    shows the population mean over all neurons in the layer.
    Both Y axes are zero-aligned (convention from Intracranial).

    Args:
        fi_df:        feature_importance.csv from a run_univariate_encoding output dir.
                      Columns: neuron, time_bin, relative_time, feature, fi_mean, fi_std.
        scores_df:    scores.csv from the same run dir.
                      Columns: neuron, time_bin, relative_time, metric, score_median, score_std.
        layer:        e.g. 'encoder_out' or 'decoder_in'.
        analysis_set: 'all_items' or 'words_only'.
        metric:       'r2' or 'spearman'.
        output_path:  Where to write the PNG.
        aggregation:  Aggregation over neurons. Only 'mean' is supported in v1.
        model_name:   e.g. 'encodec' or 'dac' — shown in title.
        run_id:       e.g. 'encodec__7f7d3b97' — shown as subtitle if provided.

    Returns:
        True if figure was saved, False if no data for the requested combination.
    """
    fi_sub = fi_df[
        (fi_df["layer"] == layer) & (fi_df["analysis_set"] == analysis_set)
    ]
    if analysis_view and "analysis_view" in fi_sub.columns:
        fi_sub = fi_sub[fi_sub["analysis_view"] == analysis_view]
    sc_sub = scores_df[
        (scores_df["layer"] == layer) &
        (scores_df["analysis_set"] == analysis_set) &
        (scores_df["metric"] == metric)
    ]
    if analysis_view and "analysis_view" in sc_sub.columns:
        sc_sub = sc_sub[sc_sub["analysis_view"] == analysis_view]

    if fi_sub.empty or sc_sub.empty:
        return False

    n_neurons = int(fi_sub["neuron"].nunique())

    # Mean FI over neurons, per (time_bin, feature)
    fi_agg = (
        fi_sub.groupby(["time_bin", "relative_time", "feature"])["fi_mean"]
        .mean()
        .reset_index()
    )

    # Mean ± std (cross-neuron) of score_median per time_bin
    sc_agg = (
        sc_sub.groupby(["time_bin", "relative_time"])["score_median"]
        .agg(mean_score="mean", std_score="std")
        .reset_index()
        .sort_values("relative_time")
    )

    # Feature order: descending mean FI
    feat_order = (
        fi_agg.groupby("feature")["fi_mean"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    x  = sc_agg["relative_time"].values
    ys = sc_agg["mean_score"].values
    ss = sc_agg["std_score"].fillna(0).values

    fig, ax1 = plt.subplots(figsize=(9, 4))

    # Left axis: FI timecourses
    for feat in feat_order:
        sub = fi_agg[fi_agg["feature"] == feat].sort_values("relative_time")
        if sub.empty:
            continue
        ax1.plot(
            sub["relative_time"].values, sub["fi_mean"].values,
            color=_FEAT_COLOR.get(feat), linewidth=1.8,
            label=feat, marker="o", markersize=3,
        )

    ax1.axhline(0, color="gray", linewidth=0.6, linestyle=":", zorder=0)
    ax1.set_xlabel("Relative time (bin centre)")
    ax1.set_ylabel("Permutation importance (mean across neurons)", fontsize=9)
    ax1.set_xlim(0, 1)

    # Right axis: encoding score
    ax2 = ax1.twinx()
    ax2.plot(x, ys, color="black", linewidth=2.2, linestyle="--",
             label=f"{metric.capitalize()}")
    ax2.fill_between(x, ys - ss, ys + ss, color="black", alpha=0.10)
    ax2.set_ylabel(f"{metric.capitalize()} (mean across neurons)", fontsize=9)

    # Zero-align both axes (Intracranial convention)
    for ax in (ax1, ax2):
        lo, hi = ax.get_ylim()
        absmax = max(abs(lo), abs(hi))
        if absmax > 0:
            ax.set_ylim(-absmax, absmax)

    # Merged legend below figure
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=len(lines1 + lines2),
        frameon=False,
        fontsize=8,
    )

    model_prefix = f"{model_name} / " if model_name else ""
    title_line1  = f"FI + {metric.capitalize()} — {model_prefix}{layer} / {analysis_set}"
    title_line2  = f"mean over {n_neurons} neurons" + (f" — run_id: {run_id}" if run_id else "")
    ax1.set_title(f"{title_line1}\n{title_line2}", fontsize=10)

    fig.tight_layout()
    _savefig(fig, output_path)
    return True