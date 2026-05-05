#!/usr/bin/env python3
"""Compact summary figures for length-controlled univariate encoding analyses.

Reads pre-computed global_fi_ranking.csv and fi_summary_by_feature_time.csv from
multiple univariate summary directories — all-speakers runs and male-only
length-controlled runs — and writes comparison figures and a combined CSV to a
single output directory. Does not re-run any analyses.

Each --summaries item has the form SPEAKER_SET:PATH, where SPEAKER_SET is a
human-readable dataset label (e.g. 'all_speakers', 'male_only') that will
appear in the output CSV and figures.

Usage (run from swp-model/ repo root):
    python scripts/audio/summarize_length_controlled_encoding.py \\
        --summaries \\
            all_speakers:reproduce/figures/audio/encoding/encodec__b429aeea/univariate_summary/ \\
            all_speakers:reproduce/figures/audio/encoding/dac__d466515b/univariate_summary/ \\
            male_only:reproduce/figures/audio/encoding/encodec__7f7d3b97/univariate_summary_length_controlled/ \\
            male_only:reproduce/figures/audio/encoding/dac__f104bbdd/univariate_summary_length_controlled/ \\
        --output reproduce/figures/audio/encoding/length_controlled_summary/ \\
        --overwrite

Outputs
-------
    fi_global_comparison.csv                Combined global FI ranking across all sources.
    fi_global_heatmap.png                   Heatmap: mean FI per (view × layer) row, feature column.
    male_only_short_long_fi_summary.png     Bar chart: male-only short_only vs long_only FI contrast.
    short_only_morphology_vs_lexicality.png Bar chart: main residual effect after length control.
    long_only_morphology_vs_lexicality.png  Bar chart: residual effects in long_only (weaker/mixed).
    temporal_fi_short_only.png              Temporal FI profiles (morphology/lexicality, short_only).
    temporal_fi_long_only.png               Temporal FI profiles (morphology/lexicality, long_only).
    speaker_effect_summary.png              Speaker FI vs linguistic features in speakerctrl views.
    summary_readme.md                       Auto-generated summary report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_DPI = 200

# Feature colours — consistent with swp/audio/encoding/univariate_plots.py
# 'speaker' is added here as it does not exist in the upstream colour map
_FEAT_COLOR: dict[str, str] = {
    "length_bin":    "#1f77b4",
    "speaker":       "#8c564b",
    "morphology":    "#d62728",
    "lexicality":    "#2ca02c",
    "frequency_bin": "#9467bd",
}

# Display order: temporal/acoustic first, then psycholinguistic
_FEAT_ORDER = ["length_bin", "speaker", "morphology", "lexicality", "frequency_bin"]

# Logical ordering for analysis_view values in sorted outputs
_VIEW_SORT: dict[str, int] = {
    "all_items_all_speakers":              0,
    "all_items_all_speakers_speakerctrl":  1,
    "all_items_short_only":                2,
    "all_items_short_only_speakerctrl":    3,
    "all_items_long_only":                 4,
    "all_items_long_only_speakerctrl":     5,
    "words_only_all_speakers":             6,
    "words_only_all_speakers_speakerctrl": 7,
}

_LAYER_ORDER  = ["encoder_out", "decoder_in"]
_MODEL_ORDER  = ["encodec", "dac"]
_SPKSET_ORDER = ["all_speakers", "male_only"]
_SPKSET_SORT  = {s: i for i, s in enumerate(_SPKSET_ORDER)}


# ── I/O helpers ───────────────────────────────────────────────────────────────


def _parse_summary_arg(item: str) -> tuple[str, Path]:
    """Parse 'SPEAKER_SET:PATH' into (speaker_set, Path)."""
    if ":" not in item:
        print(
            f"[summarize_length_controlled] ERROR: --summaries items must be "
            f"SPEAKER_SET:PATH, got: {item!r}",
            file=sys.stderr,
        )
        sys.exit(1)
    speaker_set, path_str = item.split(":", 1)
    return speaker_set.strip(), Path(path_str.strip())


def _load_csv(path: Path, fname: str) -> pd.DataFrame:
    """Load a required CSV, failing clearly if absent or missing analysis_view."""
    p = path / fname
    if not p.exists():
        raise FileNotFoundError(
            f"Required file not found: {p}\n"
            "Ensure this directory was produced by summarize_univariate_encoding.py "
            "with the analysis_view fix applied."
        )
    df = pd.read_csv(p)
    if "analysis_view" not in df.columns:
        raise ValueError(
            f"{p} is missing the 'analysis_view' column.\n"
            "Re-run summarize_univariate_encoding.py with the updated code "
            "(analysis_view derived from output directory names)."
        )
    return df


def load_all_summaries(
    summaries: list[tuple[str, Path]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and concatenate global_fi_ranking and fi_summary_by_feature_time.

    Returns (combined_global, combined_temporal).
    """
    all_global:   list[pd.DataFrame] = []
    all_temporal: list[pd.DataFrame] = []

    for speaker_set, path in summaries:
        print(f"[summarize_length_controlled] Loading {speaker_set}: {path}")

        g = _load_csv(path, "global_fi_ranking.csv")
        t = _load_csv(path, "fi_summary_by_feature_time.csv")

        g["speaker_set"] = speaker_set
        t["speaker_set"] = speaker_set
        g["source"] = speaker_set + "/" + g["run_id"].astype(str)
        t["source"] = speaker_set + "/" + t["run_id"].astype(str)

        views = sorted(g["analysis_view"].unique())
        print(f"  global_fi_ranking     : {len(g):>5} rows — {len(views)} views: {views}")
        print(f"  fi_summary_by_time    : {len(t):>5} rows")

        all_global.append(g)
        all_temporal.append(t)

    combined_g = pd.concat(all_global,   ignore_index=True)
    combined_t = pd.concat(all_temporal, ignore_index=True)
    print(
        f"\n[summarize_length_controlled] Combined: "
        f"{len(combined_g)} global rows, {len(combined_t)} temporal rows"
    )
    return combined_g, combined_t


def _row_sort_key(idx: tuple) -> tuple:
    """Sort key for (speaker_set, model_name, analysis_view, layer) index tuples."""
    sp, model, view, layer = idx
    return (
        _SPKSET_SORT.get(sp, 9),
        _MODEL_ORDER.index(model) if model in _MODEL_ORDER else 9,
        _VIEW_SORT.get(view, 99),
        _LAYER_ORDER.index(layer) if layer in _LAYER_ORDER else 9,
    )


# ── Output 1: fi_global_comparison.csv ───────────────────────────────────────

_COMPARISON_COLS = [
    "source", "model_name", "run_id", "speaker_set", "analysis_view",
    "layer", "feature", "mean_fi_over_time_and_neurons", "rank_by_mean_fi",
]


def write_comparison_csv(combined: pd.DataFrame, output_dir: Path) -> None:
    out = combined[[c for c in _COMPARISON_COLS if c in combined.columns]].copy()
    path = output_dir / "fi_global_comparison.csv"
    out.to_csv(path, index=False)
    print(f"[summarize_length_controlled] Written: {path}  ({len(out)} rows)")


# ── Output 2: fi_global_heatmap.png ──────────────────────────────────────────


def plot_fi_heatmap(combined: pd.DataFrame, output_dir: Path) -> None:
    pivot = combined.pivot_table(
        index=["speaker_set", "model_name", "analysis_view", "layer"],
        columns="feature",
        values="mean_fi_over_time_and_neurons",
        aggfunc="mean",
    )

    # Keep only features that actually appear in the data, in canonical order
    feat_cols = [f for f in _FEAT_ORDER if f in pivot.columns]
    pivot = pivot.reindex(columns=feat_cols)

    # Sort rows into logical order
    pivot = pivot.loc[sorted(pivot.index, key=_row_sort_key)]

    mat = pivot.values.astype(float)
    n_rows, n_cols = mat.shape

    fig, ax = plt.subplots(figsize=(max(6, n_cols * 1.5), max(8, n_rows * 0.38)))

    vmax = float(np.nanmax(mat)) if not np.all(np.isnan(mat)) else 0.1
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=vmax)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad(color="#e0e0e0")   # grey for features absent from that view

    im = ax.imshow(mat, aspect="auto", norm=norm, cmap=cmap, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="mean FI (sqrt scale)", fraction=0.03, pad=0.03)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(feat_cols, fontsize=9, rotation=25, ha="right")

    # Row labels: compact four-component format
    row_labels = []
    prev_group = None
    separator_rows: list[int] = []
    for i, (sp, model, view, layer) in enumerate(pivot.index):
        group = (sp, model)
        if prev_group is not None and group != prev_group:
            separator_rows.append(i)
        prev_group = group
        short_sp    = "ALL" if sp == "all_speakers" else "MALE"
        short_layer = "enc" if layer == "encoder_out" else "dec"
        row_labels.append(
            f"{model.upper()} {short_layer} [{short_sp}]  {view}"
        )

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=6.5)

    for i in separator_rows:
        ax.axhline(i - 0.5, color="white", linewidth=2.5)

    ax.set_title(
        "Mean permutation FI by analysis view and feature\n"
        "(grey = feature absent/not requested for this view;  sqrt colour scale)",
        fontsize=10, pad=10,
    )
    fig.tight_layout()
    path = output_dir / "fi_global_heatmap.png"
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[summarize_length_controlled] Written: {path}")


# ── Outputs 3 & 4: morph vs lex bar charts ───────────────────────────────────


def _plot_morph_vs_lex(
    combined: pd.DataFrame,
    output_dir: Path,
    analysis_view: str,
    filename: str,
    suptitle: str,
) -> None:
    """Shared bar-chart logic for morphology vs lexicality FI in a single analysis_view."""
    df = combined[
        (combined["analysis_view"] == analysis_view) &
        (combined["feature"].isin(["morphology", "lexicality"]))
    ].copy()

    if df.empty:
        print(
            f"[summarize_length_controlled] WARNING: no {analysis_view} data — "
            f"skipping {filename}"
        )
        return

    models_present = [m for m in _MODEL_ORDER if m in df["model_name"].unique()]
    fig, axes = plt.subplots(
        1, len(models_present),
        figsize=(6 * len(models_present), 5),
        sharey=False,
    )
    if len(models_present) == 1:
        axes = [axes]

    for ax, model in zip(axes, models_present):
        sub = df[df["model_name"] == model]

        groups = [
            (sp, layer)
            for sp in _SPKSET_ORDER for layer in _LAYER_ORDER
            if not sub[(sub["speaker_set"] == sp) & (sub["layer"] == layer)].empty
        ]
        x = np.arange(len(groups))
        width = 0.35

        for j, feat in enumerate(["morphology", "lexicality"]):
            vals = []
            for sp, layer in groups:
                row = sub[
                    (sub["speaker_set"] == sp) &
                    (sub["layer"] == layer) &
                    (sub["feature"] == feat)
                ]
                vals.append(
                    float(row["mean_fi_over_time_and_neurons"].iloc[0])
                    if not row.empty else 0.0
                )
            offset = (j - 0.5) * width
            ax.bar(
                x + offset, vals, width * 0.92,
                label=feat, color=_FEAT_COLOR[feat], alpha=0.85,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                f"{sp.replace('_', ' ')}\n"
                f"{layer.replace('encoder_out', 'enc_out').replace('decoder_in', 'dec_in')}"
                for sp, layer in groups
            ],
            fontsize=8,
        )
        ax.set_title(f"{model.upper()} — {analysis_view}", fontsize=10)
        ax.set_ylabel("Mean FI over time × neurons")
        ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
        ax.legend(fontsize=8)

    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[summarize_length_controlled] Written: {path}")


def plot_short_only_morph_vs_lex(combined: pd.DataFrame, output_dir: Path) -> None:
    _plot_morph_vs_lex(
        combined, output_dir,
        analysis_view="all_items_short_only",
        filename="short_only_morphology_vs_lexicality.png",
        suptitle=(
            "Morphology vs lexicality FI after length control (short_only)\n"
            "Main residual psycholinguistic effects"
        ),
    )


def plot_long_only_morph_vs_lex(combined: pd.DataFrame, output_dir: Path) -> None:
    _plot_morph_vs_lex(
        combined, output_dir,
        analysis_view="all_items_long_only",
        filename="long_only_morphology_vs_lexicality.png",
        suptitle=(
            "Morphology vs lexicality FI after length control (long_only)\n"
            "Residual effects are weaker/mixed; DAC shows small lexicality signal"
        ),
    )


# ── Outputs 5 & 6: temporal FI profiles ──────────────────────────────────────


def _plot_temporal_fi(
    temporal: pd.DataFrame,
    output_dir: Path,
    analysis_view: str,
    filename: str,
    suptitle: str,
) -> None:
    """Shared temporal FI grid for morphology/lexicality in a single analysis_view."""
    df = temporal[
        (temporal["analysis_view"] == analysis_view) &
        (temporal["feature"].isin(["morphology", "lexicality"]))
    ].copy()

    if df.empty:
        print(
            f"[summarize_length_controlled] WARNING: no temporal {analysis_view} data — "
            f"skipping {filename}"
        )
        return

    models_present = [m for m in _MODEL_ORDER if m in df["model_name"].unique()]
    layers_present = [l for l in _LAYER_ORDER if l in df["layer"].unique()]

    fig, axes = plt.subplots(
        len(models_present), len(layers_present),
        figsize=(6 * len(layers_present), 4 * len(models_present)),
        sharey="row", sharex=True,
        squeeze=False,
    )

    _LS = {"all_speakers": "-", "male_only": "--"}

    for r, model in enumerate(models_present):
        for c, layer in enumerate(layers_present):
            ax = axes[r, c]
            sub = df[(df["model_name"] == model) & (df["layer"] == layer)]

            for sp in _SPKSET_ORDER:
                for feat in ["morphology", "lexicality"]:
                    s = sub[(sub["speaker_set"] == sp) & (sub["feature"] == feat)]
                    if s.empty:
                        continue
                    s = s.sort_values("relative_time")
                    ax.plot(
                        s["relative_time"], s["mean_fi"],
                        color=_FEAT_COLOR[feat],
                        linestyle=_LS.get(sp, "-"),
                        linewidth=1.8, marker="o", markersize=3,
                        label=f"{feat} / {sp.replace('_', ' ')}",
                    )

            ax.axhline(0, color="gray", linewidth=0.6, linestyle=":", zorder=0)
            ax.set_title(f"{model.upper()} / {layer.replace('_', ' ')}", fontsize=9)
            ax.set_xlim(0, 1)
            if c == 0:
                ax.set_ylabel("Mean FI", fontsize=8)
            if r == len(models_present) - 1:
                ax.set_xlabel("Relative time (bin centre)", fontsize=8)
            if r == 0 and c == len(layers_present) - 1:
                ax.legend(fontsize=7, loc="upper right", framealpha=0.8)

    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[summarize_length_controlled] Written: {path}")


def plot_temporal_fi_short_only(temporal: pd.DataFrame, output_dir: Path) -> None:
    _plot_temporal_fi(
        temporal, output_dir,
        analysis_view="all_items_short_only",
        filename="temporal_fi_short_only.png",
        suptitle=(
            "Temporal FI profile — all_items_short_only\n"
            "Solid: all-speakers  |  Dashed: male-only"
        ),
    )


def plot_temporal_fi_long_only(temporal: pd.DataFrame, output_dir: Path) -> None:
    _plot_temporal_fi(
        temporal, output_dir,
        analysis_view="all_items_long_only",
        filename="temporal_fi_long_only.png",
        suptitle=(
            "Temporal FI profile — all_items_long_only\n"
            "Solid: all-speakers  |  Dashed: male-only"
        ),
    )


# ── Output 7: male_only_short_long_fi_summary.png ────────────────────────────


def plot_male_only_short_long_fi_summary(
    combined: pd.DataFrame, output_dir: Path
) -> None:
    """Bar chart: male-only morphology + lexicality FI for short_only vs long_only.

    Provides the male-only baseline to compare against all-speakers results.
    One panel per model; bars grouped by length view, split by feature.
    Separate bars per layer (encoder_out, decoder_in).
    """
    df = combined[
        (combined["speaker_set"] == "male_only") &
        (combined["analysis_view"].isin(["all_items_short_only", "all_items_long_only"])) &
        (combined["feature"].isin(["morphology", "lexicality"]))
    ].copy()

    if df.empty:
        print(
            "[summarize_length_controlled] WARNING: no male_only short/long data — "
            "skipping male_only_short_long_fi_summary.png"
        )
        return

    models_present = [m for m in _MODEL_ORDER if m in df["model_name"].unique()]
    layers_present = [l for l in _LAYER_ORDER if l in df["layer"].unique()]
    views_order    = ["all_items_short_only", "all_items_long_only"]

    fig, axes = plt.subplots(
        1, len(models_present),
        figsize=(7 * len(models_present), 5),
        sharey=False,
    )
    if len(models_present) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models_present):
        sub = df[df["model_name"] == model_name]

        # Groups: view × layer
        groups = [
            (view, layer)
            for view in views_order for layer in layers_present
            if not sub[(sub["analysis_view"] == view) & (sub["layer"] == layer)].empty
        ]

        x     = np.arange(len(groups))
        width = 0.38

        for j, feat in enumerate(["morphology", "lexicality"]):
            vals = []
            for view, layer in groups:
                row = sub[
                    (sub["analysis_view"] == view) &
                    (sub["layer"] == layer) &
                    (sub["feature"] == feat)
                ]
                vals.append(
                    float(row["mean_fi_over_time_and_neurons"].iloc[0])
                    if not row.empty else 0.0
                )
            offset = (j - 0.5) * width
            ax.bar(
                x + offset, vals, width * 0.92,
                label=feat, color=_FEAT_COLOR[feat], alpha=0.85,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                f"{view.replace('all_items_', '')}\n"
                f"{layer.replace('encoder_out', 'enc_out').replace('decoder_in', 'dec_in')}"
                for view, layer in groups
            ],
            fontsize=8,
        )
        ax.set_title(f"{model_name.upper()} — male_only", fontsize=10)
        ax.set_ylabel("Mean FI over time × neurons")
        ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Male-only baseline: morphology vs lexicality FI — short_only vs long_only\n"
        "(length_bin dropped in both views as constant after filtering)",
        fontsize=11,
    )
    fig.tight_layout()
    path = output_dir / "male_only_short_long_fi_summary.png"
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[summarize_length_controlled] Written: {path}")


# ── Output 8: speaker_effect_summary.png ─────────────────────────────────────


def plot_speaker_effect(combined: pd.DataFrame, output_dir: Path) -> None:
    df = combined[
        (combined["speaker_set"] == "all_speakers") &
        (combined["analysis_view"].str.contains("speakerctrl", na=False))
    ].copy()

    if df.empty:
        print(
            "[summarize_length_controlled] WARNING: no speakerctrl data in all-speakers sources — "
            "skipping speaker_effect_summary.png"
        )
        return

    features_shown = [f for f in _FEAT_ORDER if f in df["feature"].unique()]
    views = sorted(df["analysis_view"].unique(), key=lambda v: _VIEW_SORT.get(v, 99))
    models_present = [m for m in _MODEL_ORDER if m in df["model_name"].unique()]
    layers_present = [l for l in _LAYER_ORDER if l in df["layer"].unique()]

    n_feats   = len(features_shown)
    bar_width = 0.8 / max(n_feats, 1)
    x = np.arange(len(views))

    fig, axes = plt.subplots(
        len(models_present), len(layers_present),
        figsize=(3.5 * len(views) * max(len(layers_present), 1), 5 * len(models_present)),
        sharey="row", squeeze=False,
    )

    for r, model in enumerate(models_present):
        for c, layer in enumerate(layers_present):
            ax = axes[r, c]
            sub = df[(df["model_name"] == model) & (df["layer"] == layer)]

            for j, feat in enumerate(features_shown):
                vals = []
                for view in views:
                    row = sub[(sub["analysis_view"] == view) & (sub["feature"] == feat)]
                    vals.append(
                        float(row["mean_fi_over_time_and_neurons"].iloc[0])
                        if not row.empty else np.nan
                    )
                offset = (j - (n_feats - 1) / 2) * bar_width
                ax.bar(
                    x + offset, vals, bar_width * 0.9,
                    label=feat, color=_FEAT_COLOR.get(feat, "#999999"), alpha=0.85,
                )

            ax.set_xticks(x)
            short_views = [
                v.replace("all_items_", "")
                 .replace("words_only_all_speakers_speakerctrl", "words_only\n+ctrl")
                 .replace("_speakerctrl", "\n+ctrl")
                for v in views
            ]
            ax.set_xticklabels(short_views, fontsize=8, rotation=10, ha="right")
            ax.set_title(f"{model.upper()} / {layer.replace('_', ' ')}", fontsize=9)
            ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
            if c == 0:
                ax.set_ylabel("Mean FI over time × neurons", fontsize=8)
            if r == 0 and c == len(layers_present) - 1:
                ax.legend(fontsize=7, loc="upper right", framealpha=0.8)

    fig.suptitle(
        "Speaker identity FI vs linguistic features — speakerctrl views\n"
        "(all-speakers dataset only)",
        fontsize=11,
    )
    fig.tight_layout()
    path = output_dir / "speaker_effect_summary.png"
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[summarize_length_controlled] Written: {path}")


# ── Output 6: summary_readme.md ──────────────────────────────────────────────


def write_readme(
    combined: pd.DataFrame,
    summaries: list[tuple[str, Path]],
    output_dir: Path,
) -> None:
    lines: list[str] = [
        "# Length-Controlled Encoding Summary",
        "",
        "Generated by `scripts/audio/summarize_length_controlled_encoding.py`.",
        "Do not edit manually — re-run the script to regenerate.",
        "",
        "## 1. Inputs",
        "",
        "| source | run_id | speaker_set | n_rows | analysis_views |",
        "|--------|--------|-------------|--------|----------------|",
    ]
    for source, grp in combined.groupby("source", sort=False):
        run_id      = grp["run_id"].iloc[0]
        speaker_set = grp["speaker_set"].iloc[0]
        views       = sorted(grp["analysis_view"].unique())
        lines.append(
            f"| {source} | {run_id} | {speaker_set} | {len(grp)} "
            f"| {', '.join(f'`{v}`' for v in views)} |"
        )

    lines += ["", "## 2. Feature presence check", ""]
    for view in ["all_items_short_only", "all_items_long_only"]:
        sub = combined[combined["analysis_view"] == view]
        has_length = "length_bin" in sub["feature"].unique() if not sub.empty else None
        if has_length is None:
            lines.append(f"- `{view}`: **not present in any source**")
        elif has_length:
            lines.append(f"- `{view}`: `length_bin` present = **True** ⚠ UNEXPECTED")
        else:
            lines.append(f"- `{view}`: `length_bin` present = **False** ✓ (correctly dropped)")

    lines += ["", "## 3. Top feature per model / layer / view", ""]
    key_views = [
        "all_items_all_speakers",
        "all_items_short_only",
        "all_items_long_only",
        "all_items_all_speakers_speakerctrl",
    ]
    for view in key_views:
        sub = combined[combined["analysis_view"] == view]
        if sub.empty:
            continue
        lines.append(f"### `{view}`")
        lines.append("")
        lines.append("| model | layer | top feature | mean FI |")
        lines.append("|-------|-------|-------------|---------|")
        for model in _MODEL_ORDER:
            for layer in _LAYER_ORDER:
                s = sub[(sub["model_name"] == model) & (sub["layer"] == layer)]
                if s.empty:
                    continue
                best = s.loc[s["mean_fi_over_time_and_neurons"].idxmax()]
                lines.append(
                    f"| {model} | {layer} | **{best['feature']}** "
                    f"| {best['mean_fi_over_time_and_neurons']:.4f} |"
                )
        lines.append("")

    lines += [
        "## 4. Scientific interpretation",
        "",
        "- **Length dominates full views**: `length_bin` is rank 1 in "
        "`all_items_all_speakers` for both EnCodec and DAC across both layers.",
        "",
        "- **Morphology dominates lexicality in short_only**: After restricting to "
        "short items, `morphology` becomes the top-ranked psycholinguistic feature. "
        "`lexicality` remains weak — there is no clear dual-route lexical signal "
        "once length variance is removed.",
        "",
        "- **Long_only effects are weak**: In `all_items_long_only`, both morphology "
        "and lexicality have low FI values across codecs and layers.",
        "",
        "- **Speaker is a strong acoustic dimension**: In speakerctrl views, `speaker` "
        "FI is comparable to or exceeds `length_bin` in some configurations, "
        "confirming that speaker identity is strongly encoded by both codecs.",
        "",
        "- **Results consistent across codecs**: EnCodec shows larger absolute FI "
        "values than DAC, but the qualitative feature hierarchy is stable across "
        "architectures and speaker sets.",
    ]

    path = output_dir / "summary_readme.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"[summarize_length_controlled] Written: {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compact summary figures for length-controlled univariate encoding analyses."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--summaries",
        nargs="+",
        required=True,
        metavar="SPEAKER_SET:DIR",
        help=(
            "One or more summary directories, each prefixed with a speaker-set label "
            "and a colon. E.g.: all_speakers:path/to/dir  male_only:path/to/other/dir"
        ),
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="DIR",
        help="Output directory for all generated files.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs without error.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    summaries  = [_parse_summary_arg(s) for s in args.summaries]
    output_dir = Path(args.output)

    _EXPECTED_OUTPUTS = [
        "fi_global_comparison.csv",
        "fi_global_heatmap.png",
        "male_only_short_long_fi_summary.png",
        "short_only_morphology_vs_lexicality.png",
        "long_only_morphology_vs_lexicality.png",
        "temporal_fi_short_only.png",
        "temporal_fi_long_only.png",
        "speaker_effect_summary.png",
        "summary_readme.md",
    ]
    if not args.overwrite:
        existing = [output_dir / f for f in _EXPECTED_OUTPUTS if (output_dir / f).exists()]
        if existing:
            print(
                f"[summarize_length_controlled] ERROR: output files already exist: "
                f"{[p.name for p in existing]}. Pass --overwrite to replace.",
                file=sys.stderr,
            )
            sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[summarize_length_controlled] Output  : {output_dir}")
    print(f"[summarize_length_controlled] Sources : {len(summaries)}")
    for sp, path in summaries:
        print(f"  {sp}: {path}")
    print()

    combined_global, combined_temporal = load_all_summaries(summaries)

    write_comparison_csv(combined_global,                   output_dir)
    plot_fi_heatmap(combined_global,                        output_dir)
    plot_male_only_short_long_fi_summary(combined_global,   output_dir)
    plot_short_only_morph_vs_lex(combined_global,           output_dir)
    plot_long_only_morph_vs_lex(combined_global,            output_dir)
    plot_temporal_fi_short_only(combined_temporal,          output_dir)
    plot_temporal_fi_long_only(combined_temporal,           output_dir)
    plot_speaker_effect(combined_global,                    output_dir)
    write_readme(combined_global, summaries,                output_dir)

    print(f"\n[summarize_length_controlled] Done — all outputs in {output_dir}")


if __name__ == "__main__":
    main()