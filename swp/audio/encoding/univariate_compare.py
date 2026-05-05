# LEGACY — not analysis_view-safe.
# Groups only by analysis_set; will silently mix all_items_short_only,
# all_items_long_only, all_items_all_speakers etc. into the same rows.
# Use summarize_length_controlled_encoding.py for cross-view comparisons.
"""Codec comparison figures and CSV summaries for univariate encoding results.

Reads pre-computed summary directories produced by summarize_univariate_encoding.py
(one per codec) and produces combined CSVs plus 8 comparison figures.

Public API
----------
load_summary_dir(path, label)           -> dict[str, pd.DataFrame]
combine_summaries(summaries)            -> dict[str, pd.DataFrame]
compute_length_dominance(global_fi)     -> pd.DataFrame
compare(summary_dirs, labels, output_dir, overwrite) -> dict
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_DPI     = 200
_EPSILON = 1e-12   # added to denominator in dominance ratios to avoid div-by-zero

_FEAT_COLOR = {
    "lexicality":    "#2ca02c",
    "length_bin":    "#1f77b4",
    "morphology":    "#d62728",
    "frequency_bin": "#9467bd",
}

_LAYER_ORDER = ["encoder_out", "decoder_in"]
_ASET_ORDER  = ["all_items",   "words_only"]

# Up to 4 codecs: distinct colors + linestyles
_CODEC_PALETTE = ["#e377c2", "#bcbd22", "#17becf", "#7f7f7f"]
_CODEC_LS      = ["-",       "--",      "-.",      ":"]


def _codec_styles(labels: list[str]) -> dict[str, tuple[str, str]]:
    """Map codec label → (hex_color, linestyle)."""
    return {
        lbl: (_CODEC_PALETTE[i % len(_CODEC_PALETTE)], _CODEC_LS[i % len(_CODEC_LS)])
        for i, lbl in enumerate(labels)
    }


def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


# ── I/O ───────────────────────────────────────────────────────────────────────


def load_summary_dir(path: Path, label: str) -> dict[str, pd.DataFrame]:
    """Load 4 CSVs from a univariate summary dir and tag each with codec_label."""
    path = Path(path)
    required = {
        "global_fi_ranking":      "global_fi_ranking.csv",
        "fi_summary":             "fi_summary_by_feature_time.csv",
        "score_summary":          "score_summary_by_time.csv",
        "global_feature_ranking": "global_feature_ranking.csv",
    }
    out: dict[str, pd.DataFrame] = {}
    for key, fname in required.items():
        p = path / fname
        if not p.exists():
            raise FileNotFoundError(
                f"Expected file not found: {p}\n"
                "Ensure this is a valid summarize_univariate_encoding output directory."
            )
        df = pd.read_csv(p)
        df.insert(0, "codec_label", label)
        out[key] = df
    return out


def combine_summaries(summaries: list[dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    """Concatenate N codec summary dicts into 4 combined DataFrames."""
    keys = ["global_fi_ranking", "fi_summary", "score_summary", "global_feature_ranking"]
    return {k: pd.concat([s[k] for s in summaries], ignore_index=True) for k in keys}


# ── Length dominance ──────────────────────────────────────────────────────────


def compute_length_dominance(global_fi: pd.DataFrame) -> pd.DataFrame:
    """Compute length_bin FI dominance over the other features per condition.

    For each (codec_label, run_id, model_name, layer, analysis_set):
      length_to_other_ratio     = length_mean_fi   / (mean other features mean_fi   + eps)
      length_to_other_abs_ratio = length_mean_abs_fi / (mean other features mean_abs_fi + eps)

    eps = 1e-12 to prevent division by zero when all other-feature FIs are exactly 0.

    The column ``other_features`` lists the non-length features as a pipe-joined sorted string
    (e.g. "lexicality|morphology") so the denominator composition is always explicit.
    """
    group_keys = ["codec_label", "run_id", "model_name", "layer", "analysis_set"]
    rows = []
    for group_vals, sub in global_fi.groupby(group_keys, sort=False):
        length_row = sub[sub["feature"] == "length_bin"]
        other_rows = sub[sub["feature"] != "length_bin"]
        if length_row.empty:
            continue

        l_fi    = float(length_row["mean_fi_over_time_and_neurons"].iloc[0])
        l_absfi = float(length_row["mean_abs_fi_over_time_and_neurons"].iloc[0])
        o_fi    = float(other_rows["mean_fi_over_time_and_neurons"].mean())    if not other_rows.empty else 0.0
        o_absfi = float(other_rows["mean_abs_fi_over_time_and_neurons"].mean()) if not other_rows.empty else 0.0

        row = dict(zip(group_keys, group_vals))
        row.update({
            "length_mean_fi":            l_fi,
            "other_mean_fi":             o_fi,
            "length_to_other_ratio":     l_fi    / (o_fi    + _EPSILON),
            "length_mean_abs_fi":        l_absfi,
            "other_mean_abs_fi":         o_absfi,
            "length_to_other_abs_ratio": l_absfi / (o_absfi + _EPSILON),
            "other_features":            "|".join(sorted(other_rows["feature"].tolist())),
        })
        rows.append(row)

    return pd.DataFrame(rows)


# ── Helpers for multi-panel layouts ───────────────────────────────────────────


def _make_2x2_axes(layers, asets, figsize_per_cell=(5, 4)):
    n_rows, n_cols = len(asets), len(layers)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
        squeeze=False,
    )
    return fig, axes, n_rows, n_cols


# ── Figure 1 & 2 : global FI bar charts ──────────────────────────────────────


def _plot_global_fi_compare(
    global_fi: pd.DataFrame,
    codec_labels: list[str],
    output_path: Path,
    use_abs: bool = False,
) -> None:
    col        = "mean_abs_fi_over_time_and_neurons" if use_abs else "mean_fi_over_time_and_neurons"
    y_label    = "Mean |FI| over time × neurons"    if use_abs else "Mean FI over time × neurons"
    metric_str = "mean |FI|"                         if use_abs else "mean FI"

    styles = _codec_styles(codec_labels)
    layers = [l for l in _LAYER_ORDER if l in global_fi["layer"].unique()]
    asets  = [a for a in _ASET_ORDER  if a in global_fi["analysis_set"].unique()]

    fig, axes, n_rows, n_cols = _make_2x2_axes(layers, asets)
    n_codecs = len(codec_labels)
    width    = 0.7 / n_codecs

    for r, aset in enumerate(asets):
        for c, layer in enumerate(layers):
            ax  = axes[r, c]
            sub = global_fi[
                (global_fi["layer"] == layer) & (global_fi["analysis_set"] == aset)
            ]
            # Feature order: descending average across codecs
            feat_order = (
                sub.groupby("feature")[col].mean()
                .sort_values(ascending=False).index.tolist()
            )
            x = np.arange(len(feat_order))
            for i, codec in enumerate(codec_labels):
                color, _ = styles[codec]
                vals = [
                    float(sub.loc[(sub["codec_label"] == codec) & (sub["feature"] == f), col].iloc[0])
                    if not sub[(sub["codec_label"] == codec) & (sub["feature"] == f)].empty
                    else 0.0
                    for f in feat_order
                ]
                offset = (i - n_codecs / 2 + 0.5) * width
                ax.bar(x + offset, vals, width=width, color=color,
                       label=codec, edgecolor="white", linewidth=0.5)

            ax.set_xticks(x)
            ax.set_xticklabels(feat_order, rotation=15, fontsize=8)
            ax.set_title(f"{layer} / {aset}", fontsize=9)
            if c == 0:
                ax.set_ylabel(y_label)
            ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
            if r == 0 and c == 0:
                ax.legend(fontsize=8)

    fig.suptitle(
        f"EnCodec vs DAC — Global FI ranking ({metric_str})\nby feature, layer, analysis set",
        fontsize=11,
    )
    fig.tight_layout()
    _savefig(fig, output_path)


# ── Figure 3 : FI timecourse for one feature (2×2) ───────────────────────────


def _plot_fi_over_time_compare_feature(
    fi_summary: pd.DataFrame,
    codec_labels: list[str],
    feature: str,
    output_path: Path,
) -> None:
    styles    = _codec_styles(codec_labels)
    sub_feat  = fi_summary[fi_summary["feature"] == feature]
    layers    = [l for l in _LAYER_ORDER if l in sub_feat["layer"].unique()]
    asets     = [a for a in _ASET_ORDER  if a in sub_feat["analysis_set"].unique()]

    fig, axes, _, _ = _make_2x2_axes(layers, asets)

    for r, aset in enumerate(asets):
        for c, layer in enumerate(layers):
            ax  = axes[r, c]
            sub = sub_feat[
                (sub_feat["layer"] == layer) & (sub_feat["analysis_set"] == aset)
            ]
            for codec in codec_labels:
                color, ls = styles[codec]
                rows = sub[sub["codec_label"] == codec].sort_values("relative_time")
                if rows.empty:
                    continue
                ax.plot(
                    rows["relative_time"], rows["mean_fi"],
                    color=color, linestyle=ls, linewidth=1.8,
                    label=codec, marker="o", markersize=3,
                )
            ax.axhline(0, color="black", linewidth=0.6, linestyle=":", zorder=0)
            ax.set_title(f"{layer} / {aset}", fontsize=9)
            ax.set_xlabel("Relative time (bin centre)")
            if c == 0:
                ax.set_ylabel("Mean FI across neurons")
            ax.set_xlim(0, 1)
            if r == 0 and c == 0:
                ax.legend(fontsize=8)

    fig.suptitle(
        f"EnCodec vs DAC — FI over time for '{feature}' (mean FI)\nby layer and analysis set",
        fontsize=11,
    )
    fig.tight_layout()
    _savefig(fig, output_path)


# ── Figures 4 & 5 : FI timecourses, all features (1×2 per layer) ─────────────


def _plot_fi_over_time_all_features(
    fi_summary: pd.DataFrame,
    codec_labels: list[str],
    analysis_set: str,
    output_path: Path,
) -> None:
    styles   = _codec_styles(codec_labels)
    sub_aset = fi_summary[fi_summary["analysis_set"] == analysis_set]
    layers   = [l for l in _LAYER_ORDER if l in sub_aset["layer"].unique()]
    n_cols   = len(layers)

    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4), squeeze=False)

    for c, layer in enumerate(layers):
        ax  = axes[0, c]
        sub = sub_aset[sub_aset["layer"] == layer]

        # Feature order: by mean FI descending (averaged over codecs)
        feat_order = (
            sub.groupby("feature")["mean_fi"].mean()
            .sort_values(ascending=False).index.tolist()
        )
        for feat in feat_order:
            feat_color = _FEAT_COLOR.get(feat, "#888888")
            for codec in codec_labels:
                _, ls = styles[codec]
                rows = (
                    sub[(sub["feature"] == feat) & (sub["codec_label"] == codec)]
                    .sort_values("relative_time")
                )
                if rows.empty:
                    continue
                ax.plot(
                    rows["relative_time"], rows["mean_fi"],
                    color=feat_color, linestyle=ls, linewidth=1.8,
                    label=f"{feat} ({codec})", marker="o", markersize=3,
                )

        ax.axhline(0, color="black", linewidth=0.6, linestyle=":", zorder=0)
        ax.set_title(layer, fontsize=10)
        ax.set_xlabel("Relative time (bin centre)")
        if c == 0:
            ax.set_ylabel("Mean FI across neurons")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=7, ncol=2, loc="best")

    fig.suptitle(
        f"EnCodec vs DAC — FI over time, {analysis_set} (mean FI)\ncolor = feature   style = codec",
        fontsize=11,
    )
    fig.tight_layout()
    _savefig(fig, output_path)


# ── Figures 6 & 7 : score timecourses (2×2) ──────────────────────────────────


def _plot_score_over_time_compare(
    score_summary: pd.DataFrame,
    codec_labels: list[str],
    metric: str,
    output_path: Path,
) -> bool:
    styles     = _codec_styles(codec_labels)
    sub_metric = score_summary[score_summary["metric"] == metric]
    if sub_metric.empty:
        return False

    layers = [l for l in _LAYER_ORDER if l in sub_metric["layer"].unique()]
    asets  = [a for a in _ASET_ORDER  if a in sub_metric["analysis_set"].unique()]

    fig, axes, _, _ = _make_2x2_axes(layers, asets)

    for r, aset in enumerate(asets):
        for c, layer in enumerate(layers):
            ax  = axes[r, c]
            sub = sub_metric[
                (sub_metric["layer"] == layer) & (sub_metric["analysis_set"] == aset)
            ]
            for codec in codec_labels:
                color, ls = styles[codec]
                rows = sub[sub["codec_label"] == codec].sort_values("relative_time")
                if rows.empty:
                    continue
                x = rows["relative_time"].values
                y = rows["mean_score"].values
                s = rows["std_score"].fillna(0).values
                ax.plot(x, y, color=color, linestyle=ls, linewidth=1.8,
                        label=codec, marker="o", markersize=3)
                ax.fill_between(x, y - s, y + s, color=color, alpha=0.12)

            ax.axhline(0, color="black", linewidth=0.6, linestyle=":", zorder=0)
            ax.set_title(f"{layer} / {aset}", fontsize=9)
            ax.set_xlabel("Relative time (bin centre)")
            if c == 0:
                ax.set_ylabel(f"Mean {metric.capitalize()} score")
            ax.set_xlim(0, 1)
            if r == 0 and c == 0:
                ax.legend(fontsize=8)

    fig.suptitle(
        f"EnCodec vs DAC — Encoding score over time ({metric.capitalize()})\nby layer and analysis set",
        fontsize=11,
    )
    fig.tight_layout()
    _savefig(fig, output_path)
    return True


# ── Figure 8 : length dominance ratio barplot ─────────────────────────────────


def _plot_length_dominance_ratio(
    dominance_df: pd.DataFrame,
    codec_labels: list[str],
    output_path: Path,
) -> None:
    styles  = _codec_styles(codec_labels)
    layers  = [l for l in _LAYER_ORDER if l in dominance_df["layer"].unique()]
    asets   = [a for a in _ASET_ORDER  if a in dominance_df["analysis_set"].unique()]

    combos_raw = [(l, a) for l in layers for a in asets]
    x_labels   = [f"{l}\n{a}" for l, a in combos_raw]
    x          = np.arange(len(combos_raw))
    n_codecs   = len(codec_labels)
    width      = 0.7 / n_codecs

    fig, ax = plt.subplots(figsize=(max(6, 2.2 * len(combos_raw)), 4))

    for i, codec in enumerate(codec_labels):
        color, _ = styles[codec]
        vals = []
        for layer, aset in combos_raw:
            row = dominance_df[
                (dominance_df["codec_label"] == codec) &
                (dominance_df["layer"] == layer) &
                (dominance_df["analysis_set"] == aset)
            ]
            vals.append(float(row["length_to_other_ratio"].iloc[0]) if not row.empty else 0.0)
        offset = (i - n_codecs / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width=width, color=color,
                      label=codec, edgecolor="white", linewidth=0.5)
        y_max = max(vals) if vals else 1.0
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_max * 0.02,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel("length_bin mean FI / other features mean FI")
    ax.set_title(
        "EnCodec vs DAC — length_bin FI dominance ratio\n"
        "(ratio > 1 → length_bin dominates other features)"
    )
    ax.axhline(1, color="black", linewidth=0.8, linestyle=":", zorder=0)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _savefig(fig, output_path)


# ── Main entry point ──────────────────────────────────────────────────────────


def compare(
    summary_dirs: list[Path],
    labels: list[str],
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load, combine, compute, and save all comparison CSVs and figures.

    Args:
        summary_dirs: Paths to summarize_univariate_encoding output directories,
                      one per codec.
        labels:       Short codec labels in the same order as summary_dirs.
        output_dir:   Directory where CSV and PNG outputs will be written.
        overwrite:    If False and any output exists, raise FileExistsError.

    Returns:
        dict with keys: 'combined_global_fi', 'combined_fi_summary',
        'combined_score_summary', 'length_dominance'.
    """
    if len(summary_dirs) != len(labels):
        raise ValueError(
            f"Number of summary dirs ({len(summary_dirs)}) must match "
            f"number of labels ({len(labels)})."
        )

    output_dir = Path(output_dir)

    out_csv: dict[str, Path] = {
        "combined_global_fi_ranking":          output_dir / "combined_global_fi_ranking.csv",
        "combined_fi_summary_by_feature_time": output_dir / "combined_fi_summary_by_feature_time.csv",
        "combined_score_summary_by_time":      output_dir / "combined_score_summary_by_time.csv",
        "length_dominance_ratio":              output_dir / "length_dominance_ratio.csv",
    }
    out_png: dict[str, Path] = {
        "global_fi_ranking_compare":     output_dir / "global_fi_ranking_compare.png",
        "global_abs_fi_ranking_compare": output_dir / "global_abs_fi_ranking_compare.png",
        "fi_over_time_length_bin":       output_dir / "fi_over_time_compare_length_bin.png",
        "fi_over_time_all_items":        output_dir / "fi_over_time_compare_all_features_all_items.png",
        "fi_over_time_words_only":       output_dir / "fi_over_time_compare_all_features_words_only.png",
        "score_r2":                      output_dir / "score_over_time_compare_r2.png",
        "score_spearman":                output_dir / "score_over_time_compare_spearman.png",
        "length_dominance_ratio":        output_dir / "length_dominance_ratio.png",
    }

    if not overwrite:
        existing = [p for p in {**out_csv, **out_png}.values() if p.exists()]
        if existing:
            raise FileExistsError(
                f"Output files already exist: {[p.name for p in existing]}. "
                "Pass overwrite=True / --overwrite to replace."
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and combine
    summaries = [load_summary_dir(d, lbl) for d, lbl in zip(summary_dirs, labels)]
    combined  = combine_summaries(summaries)

    global_fi     = combined["global_fi_ranking"]
    fi_summary    = combined["fi_summary"]
    score_summary = combined["score_summary"]

    # Compute derived table
    length_dominance = compute_length_dominance(global_fi)

    # Write CSVs
    global_fi.to_csv(out_csv["combined_global_fi_ranking"],          index=False)
    fi_summary.to_csv(out_csv["combined_fi_summary_by_feature_time"], index=False)
    score_summary.to_csv(out_csv["combined_score_summary_by_time"],   index=False)
    length_dominance.to_csv(out_csv["length_dominance_ratio"],        index=False)

    # Write figures
    _plot_global_fi_compare(global_fi, labels, out_png["global_fi_ranking_compare"], use_abs=False)
    _plot_global_fi_compare(global_fi, labels, out_png["global_abs_fi_ranking_compare"], use_abs=True)
    _plot_fi_over_time_compare_feature(fi_summary, labels, "length_bin", out_png["fi_over_time_length_bin"])
    _plot_fi_over_time_all_features(fi_summary, labels, "all_items",  out_png["fi_over_time_all_items"])
    _plot_fi_over_time_all_features(fi_summary, labels, "words_only", out_png["fi_over_time_words_only"])
    _plot_score_over_time_compare(score_summary, labels, "r2",       out_png["score_r2"])
    _plot_score_over_time_compare(score_summary, labels, "spearman", out_png["score_spearman"])
    _plot_length_dominance_ratio(length_dominance, labels, out_png["length_dominance_ratio"])

    return {
        "combined_global_fi":      global_fi,
        "combined_fi_summary":     fi_summary,
        "combined_score_summary":  score_summary,
        "length_dominance":        length_dominance,
    }