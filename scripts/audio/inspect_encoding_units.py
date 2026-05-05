#!/usr/bin/env python3
"""Unit-level inspection of univariate Ridge encoding results.

Reads feature_importance.csv and scores.csv from multiple univariate output
directories and answers: "Is lexicality truly absent globally, or concentrated
in a small subset of units?"

Each --fi-dirs item has the form SPEAKER_SET:PATH, where SPEAKER_SET labels the
dataset (e.g. 'all_speakers', 'male_only').

analysis_view is derived from the directory name (e.g. 'all_items_short_only')
since it is not stored in the CSV files themselves.

Usage (run from swp-model/ repo root):
    python scripts/audio/inspect_encoding_units.py \\
        --fi-dirs \\
            all_speakers:reproduce/figures/audio/encoding/encodec__b429aeea/encoder_out/univariate/all_items_all_speakers/ \\
            all_speakers:reproduce/figures/audio/encoding/encodec__b429aeea/encoder_out/univariate/all_items_short_only/ \\
            all_speakers:reproduce/figures/audio/encoding/encodec__b429aeea/encoder_out/univariate/all_items_long_only/ \\
            male_only:reproduce/figures/audio/encoding/encodec__7f7d3b97/encoder_out/univariate/all_items_short_only/ \\
        --output reproduce/figures/audio/encoding/unit_inspection/ \\
        --top-k 20 --min-fi-for-dominance 0.001 --min-dominance-margin 0.003 \\
        --overwrite

Outputs
-------
    unit_fi_stats.csv                    Per-unit × feature: max/mean FI, peak time, fi_std.
    unit_dominance.csv                   Per-unit: peak and mean dominant features + score stats.
    dominance_counts.csv                 Unit counts and fractions per dominant feature per view.
    top_units_by_feature.csv             Top-k units per feature × view (with score stats).
    lexicality_concentration.csv         Fraction of units above FI thresholds per view.
    unit_dominance_counts.png            Stacked bar chart (fractions) per analysis_view.
    lexicality_concentration.png         Threshold curves per view.
    lexicality_vs_morphology_scatter.png Scatter: morph FI vs lex FI per unit.
    peak_time_short_only.png             Peak score time for lex-dominant units (short_only).
    top_lexicality_units.png             FI-over-time heatmap for top-k lex units.
    speaker_unit_scatter.png             Speaker FI vs lex/morph FI (speakerctrl views).
    summary_readme.md                    Auto-generated summary report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_DPI = 200

_FEAT_COLOR: dict[str, str] = {
    "length_bin":    "#1f77b4",
    "speaker":       "#8c564b",
    "morphology":    "#d62728",
    "lexicality":    "#2ca02c",
    "frequency_bin": "#9467bd",
    "undecided":     "#aaaaaa",
}

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

_MODEL_ORDER  = ["encodec", "dac"]
_LAYER_ORDER  = ["encoder_out", "decoder_in"]

_UNIT_ID_COLS   = ["speaker_set", "model_name", "run_id", "layer", "analysis_view", "neuron"]
_LEX_THRESHOLDS = [0.001, 0.005, 0.01, 0.02, 0.05]


# ── Helpers ────────────────────────────────────────────────────────────────────


def _parse_fi_dir_arg(item: str) -> tuple[str, Path]:
    """Parse 'SPEAKER_SET:PATH' → (speaker_set, Path)."""
    if ":" not in item:
        print(
            f"[inspect_encoding_units] ERROR: --fi-dirs items must be SPEAKER_SET:PATH, "
            f"got: {item!r}",
            file=sys.stderr,
        )
        sys.exit(1)
    idx = item.index(":")
    return item[:idx].strip(), Path(item[idx + 1:].strip())


def _view_sort_key(v: str) -> int:
    return _VIEW_SORT.get(v, 99)


def _combo_sort_key(combo: tuple) -> tuple:
    model, layer = combo
    return (
        _MODEL_ORDER.index(model) if model in _MODEL_ORDER else 9,
        _LAYER_ORDER.index(layer) if layer in _LAYER_ORDER else 9,
    )


def _sorted_combos(df: pd.DataFrame) -> list[tuple[str, str]]:
    combos = df[["model_name", "layer"]].drop_duplicates().values.tolist()
    return sorted([tuple(c) for c in combos], key=_combo_sort_key)


def _sorted_views(values) -> list[str]:
    return sorted(values, key=_view_sort_key)


def _load_one_dir(speaker_set: str, fi_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load feature_importance.csv + scores.csv; inject analysis_view and speaker_set."""
    analysis_view = fi_dir.name

    fi_path = fi_dir / "feature_importance.csv"
    sc_path = fi_dir / "scores.csv"

    if not fi_path.exists():
        print(f"[WARN] feature_importance.csv not found: {fi_path}", file=sys.stderr)
        return pd.DataFrame(), pd.DataFrame()

    fi = pd.read_csv(fi_path)
    fi["analysis_view"] = analysis_view
    fi["speaker_set"]   = speaker_set

    sc = pd.DataFrame()
    if sc_path.exists():
        sc = pd.read_csv(sc_path)
        sc["analysis_view"] = analysis_view
        sc["speaker_set"]   = speaker_set

    return fi, sc


# ── Derived tables ─────────────────────────────────────────────────────────────


def _build_unit_fi_stats(fi_df: pd.DataFrame) -> pd.DataFrame:
    """Per-unit × feature: max FI, mean FI, peak relative time, fi_std stats."""
    grp_cols = _UNIT_ID_COLS + ["feature"]

    max_fi  = fi_df.groupby(grp_cols)["fi_mean"].max().rename("max_fi_over_time")
    mean_fi = fi_df.groupby(grp_cols)["fi_mean"].mean().rename("mean_fi_over_time")

    peak_rows = fi_df.loc[fi_df.groupby(grp_cols)["fi_mean"].idxmax()].set_index(grp_cols)
    peak_time = peak_rows["relative_time"].rename("peak_relative_time")

    parts = [max_fi, mean_fi, peak_time]

    if "fi_std" in fi_df.columns:
        fi_std_mean   = fi_df.groupby(grp_cols)["fi_std"].mean().rename("fi_std_mean")
        fi_std_peak   = peak_rows["fi_std"].rename("fi_std_at_peak")
        parts += [fi_std_mean, fi_std_peak]

    return pd.concat(parts, axis=1).reset_index()


def _build_score_stats(sc_df: pd.DataFrame) -> pd.DataFrame:
    """Per-unit: max/mean Spearman + R2 scores; peak time at max Spearman."""
    if sc_df.empty:
        return pd.DataFrame()

    r2 = sc_df[sc_df["metric"] == "r2"]
    sp = sc_df[sc_df["metric"] == "spearman"]

    parts: list[pd.Series] = []

    if not r2.empty:
        r2_grp = r2.groupby(_UNIT_ID_COLS)["score_median"]
        parts += [r2_grp.max().rename("max_score_r2"),
                  r2_grp.mean().rename("mean_score_r2")]

    if not sp.empty:
        sp_grp = sp.groupby(_UNIT_ID_COLS)["score_median"]
        parts += [sp_grp.max().rename("max_score_spearman"),
                  sp_grp.mean().rename("mean_score_spearman")]
        peak_rows = sp.loc[sp.groupby(_UNIT_ID_COLS)["score_median"].idxmax()].set_index(_UNIT_ID_COLS)
        parts.append(peak_rows["relative_time"].rename("peak_time_at_max_score_spearman"))

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, axis=1).reset_index()


def _dominant_for(
    grp_sorted: pd.DataFrame,
    fi_col: str,
    min_fi: float,
    min_margin: float,
) -> tuple[str, float, float, float]:
    """Return (dominant_feature, top_fi, second_fi, margin) for one unit group."""
    top_fi   = float(grp_sorted.iloc[0][fi_col])
    top_feat = str(grp_sorted.iloc[0]["feature"])

    if len(grp_sorted) >= 2:
        second_fi = float(grp_sorted.iloc[1][fi_col])
        margin    = top_fi - second_fi
    else:
        second_fi = float("nan")
        margin    = float("nan")

    if top_fi < min_fi or (not np.isnan(margin) and margin < min_margin):
        dominant = "undecided"
    else:
        dominant = top_feat

    return dominant, top_fi, second_fi, margin


def _build_unit_dominance(
    unit_fi_stats: pd.DataFrame,
    score_stats: pd.DataFrame,
    min_fi: float,
    min_margin: float,
) -> pd.DataFrame:
    """Per-unit: dominant_feature_peak, dominant_feature_mean, top/second/margin, score stats."""
    rows = []
    for key, grp in unit_fi_stats.groupby(_UNIT_ID_COLS, sort=False):
        row = {col: val for col, val in zip(_UNIT_ID_COLS, key)}

        for dtype, fi_col in [("peak", "max_fi_over_time"), ("mean", "mean_fi_over_time")]:
            grp_sorted = grp.sort_values(fi_col, ascending=False)
            dom, top, second, margin = _dominant_for(grp_sorted, fi_col, min_fi, min_margin)
            row[f"dominant_feature_{dtype}"]  = dom
            row[f"top_feature_{dtype}"]       = grp_sorted.iloc[0]["feature"]
            row[f"top_fi_{dtype}"]            = top
            row[f"second_fi_{dtype}"]         = second
            row[f"dominance_margin_{dtype}"]  = margin

        rows.append(row)

    result = pd.DataFrame(rows)

    if not score_stats.empty:
        result = result.merge(score_stats, on=_UNIT_ID_COLS, how="left")

    return result


def _build_dominance_counts(dominance_df: pd.DataFrame) -> pd.DataFrame:
    """Per (speaker_set, model, layer, analysis_view, dominant_feature, dominance_type):
    n_units, total_units, frac_units."""
    totals = (
        dominance_df.groupby(["speaker_set", "model_name", "layer", "analysis_view"])
        .size()
        .reset_index(name="total_units")
    )

    frames = []
    for dtype in ("peak", "mean"):
        col = f"dominant_feature_{dtype}"
        cnts = (
            dominance_df.groupby(
                ["speaker_set", "model_name", "layer", "analysis_view", col]
            )
            .size()
            .reset_index(name="n_units")
            .rename(columns={col: "dominant_feature"})
        )
        cnts["dominance_type"] = dtype
        cnts = cnts.merge(totals, on=["speaker_set", "model_name", "layer", "analysis_view"])
        cnts["frac_units"] = cnts["n_units"] / cnts["total_units"]
        frames.append(cnts)

    return pd.concat(frames, ignore_index=True)


def _build_top_units_by_feature(
    unit_fi_stats: pd.DataFrame,
    score_stats: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """Top-k units by max FI per feature × (speaker_set, model, layer, analysis_view)."""
    frames: list[pd.DataFrame] = []
    group_cols = ["speaker_set", "model_name", "layer", "analysis_view", "feature"]
    for _, grp in unit_fi_stats.groupby(group_cols, sort=False):
        top = grp.nlargest(top_k, "max_fi_over_time").copy()
        top["rank"] = range(1, len(top) + 1)
        frames.append(top)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)

    if not score_stats.empty:
        result = result.merge(score_stats, on=_UNIT_ID_COLS, how="left")

    return result


def _build_lexicality_concentration(unit_fi_stats: pd.DataFrame) -> pd.DataFrame:
    """Fraction of units with lexicality max FI above each threshold, per view."""
    lex = unit_fi_stats[unit_fi_stats["feature"] == "lexicality"]
    rows = []
    for (sp_set, model, run_id, layer, av), grp in lex.groupby(
        ["speaker_set", "model_name", "run_id", "layer", "analysis_view"], sort=False
    ):
        fi_vals = grp["max_fi_over_time"].values
        n = len(fi_vals)
        row: dict = dict(
            speaker_set=sp_set, model_name=model, run_id=run_id,
            layer=layer, analysis_view=av, n_units=n,
        )
        for t in _LEX_THRESHOLDS:
            key = f"frac_lex_max_fi_above_{str(t).replace('.', '_')}"
            row[key] = float((fi_vals > t).mean()) if n > 0 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ── Figures ────────────────────────────────────────────────────────────────────


def _plot_dominance_counts(dominance_counts: pd.DataFrame, output_dir: Path) -> None:
    """Stacked bar (fractions) per analysis_view — peak dominance only."""
    peak_counts = dominance_counts[dominance_counts["dominance_type"] == "peak"]
    combos = _sorted_combos(peak_counts)
    n = len(combos)
    if n == 0:
        return

    all_feats = [f for f in _FEAT_COLOR if f in peak_counts["dominant_feature"].values]

    fig, axes = plt.subplots(1, n, figsize=(max(6, 3.5 * n), 5), squeeze=False)
    for ax, (model, layer) in zip(axes[0], combos):
        sub = peak_counts[
            (peak_counts["model_name"] == model) & (peak_counts["layer"] == layer)
        ]
        views = _sorted_views(sub["analysis_view"].unique())
        pivot = sub.pivot_table(
            index="analysis_view", columns="dominant_feature",
            values="frac_units", fill_value=0,
        ).reindex(views)
        feats = [f for f in all_feats if f in pivot.columns]
        pivot = pivot.reindex(columns=feats, fill_value=0)

        bottoms = np.zeros(len(pivot))
        for feat in feats:
            vals = pivot[feat].values
            ax.bar(range(len(pivot)), vals, bottom=bottoms,
                   color=_FEAT_COLOR[feat], label=feat)
            bottoms += vals

        ax.set_xticks(range(len(pivot)))
        ax.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction of units")
        ax.set_title(f"{model} / {layer}", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Unit dominance (peak FI) — fraction per analysis_view", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "unit_dominance_counts.png", dpi=_DPI)
    plt.close(fig)


def _plot_lexicality_concentration(unit_fi_stats: pd.DataFrame, output_dir: Path) -> None:
    lex = unit_fi_stats[unit_fi_stats["feature"] == "lexicality"]
    if lex.empty:
        return

    combos = _sorted_combos(lex)
    n = len(combos)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for ax, (model, layer) in zip(axes[0], combos):
        sub = lex[(lex["model_name"] == model) & (lex["layer"] == layer)]
        for av in _sorted_views(sub["analysis_view"].unique()):
            fi_vals = sub[sub["analysis_view"] == av]["max_fi_over_time"].values
            fracs = [float((fi_vals > t).mean()) for t in _LEX_THRESHOLDS]
            ax.plot(_LEX_THRESHOLDS, fracs, marker="o", label=av)
        ax.set_xscale("log")
        ax.set_xlabel("FI threshold")
        ax.set_ylabel("Fraction of units")
        ax.set_title(f"{model} / {layer}", fontsize=9)
        ax.legend(fontsize=6, loc="upper right")

    fig.suptitle("Lexicality: fraction of units above FI threshold", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "lexicality_concentration.png", dpi=_DPI)
    plt.close(fig)


def _plot_lex_vs_morph_scatter(unit_fi_stats: pd.DataFrame, output_dir: Path) -> None:
    """One panel per (model, layer): scatter lex vs morph FI, coloured by analysis_view."""
    lex = (
        unit_fi_stats[unit_fi_stats["feature"] == "lexicality"]
        .set_index(_UNIT_ID_COLS)["max_fi_over_time"].rename("lex")
    )
    morph = (
        unit_fi_stats[unit_fi_stats["feature"] == "morphology"]
        .set_index(_UNIT_ID_COLS)["max_fi_over_time"].rename("morph")
    )
    combined = pd.concat([lex, morph], axis=1).dropna().reset_index()
    if combined.empty:
        return

    combos = _sorted_combos(combined)
    n = len(combos)
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    for ax, (model, layer) in zip(axes[0], combos):
        sub = combined[(combined["model_name"] == model) & (combined["layer"] == layer)]
        for idx, av in enumerate(_sorted_views(sub["analysis_view"].unique())):
            sv = sub[sub["analysis_view"] == av]
            ax.scatter(sv["morph"], sv["lex"], alpha=0.25, s=3,
                       color=cmap(idx % 10), label=av, rasterized=True)
        lim = max(sub["morph"].max(), sub["lex"].max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.4)
        ax.set_xlabel("Morphology max FI", fontsize=9)
        ax.set_ylabel("Lexicality max FI", fontsize=9)
        ax.set_title(f"{model} / {layer}", fontsize=9)
        ax.legend(fontsize=6, markerscale=3, loc="upper right")

    fig.suptitle("Lexicality vs Morphology max FI per unit", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "lexicality_vs_morphology_scatter.png", dpi=_DPI)
    plt.close(fig)


def _plot_peak_time_short_only(
    dominance_df: pd.DataFrame,
    sc_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Distribution of peak Spearman time for lex-dominant units in short_only views."""
    short_views = [v for v in dominance_df["analysis_view"].unique() if "short_only" in v]
    if not short_views:
        print("[INFO] No short_only views — skipping peak_time_short_only.png")
        return
    if sc_df.empty:
        print("[INFO] No scores.csv loaded — skipping peak_time_short_only.png")
        return

    lex_units = dominance_df[
        (dominance_df["dominant_feature_peak"] == "lexicality") &
        (dominance_df["analysis_view"].isin(short_views))
    ][_UNIT_ID_COLS].copy()

    if lex_units.empty:
        print("[INFO] No lex-dominant units in short_only — skipping peak_time_short_only.png")
        return

    score_stats = _build_score_stats(sc_df[sc_df["analysis_view"].isin(short_views)])
    merged = lex_units.merge(score_stats, on=_UNIT_ID_COLS, how="left")

    combos = _sorted_combos(merged)
    n = len(combos)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for ax, (model, layer) in zip(axes[0], combos):
        sub = merged[(merged["model_name"] == model) & (merged["layer"] == layer)]
        for av in _sorted_views(sub["analysis_view"].unique()):
            vals = (
                sub[sub["analysis_view"] == av]["peak_time_at_max_score_spearman"]
                .dropna().values
            )
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=10, range=(0, 1), alpha=0.5, label=av, density=True)
        ax.set_xlabel("Relative time of peak Spearman score")
        ax.set_ylabel("Density")
        ax.set_title(f"{model} / {layer}", fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle(
        "Peak score time — lexicality-dominant units (short_only, peak FI criterion)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "peak_time_short_only.png", dpi=_DPI)
    plt.close(fig)


def _plot_top_lexicality_units(
    fi_df: pd.DataFrame,
    unit_fi_stats: pd.DataFrame,
    top_k: int,
    output_dir: Path,
) -> None:
    """FI-over-time heatmap for top-k lex units — short_only views (without speakerctrl)."""
    short_views = [
        v for v in unit_fi_stats["analysis_view"].unique()
        if "short_only" in v and "speakerctrl" not in v
    ]
    if not short_views:
        short_views = [v for v in unit_fi_stats["analysis_view"].unique() if "short_only" in v]
    if not short_views:
        print("[INFO] No short_only views — skipping top_lexicality_units.png")
        return

    lex_stats = unit_fi_stats[
        (unit_fi_stats["feature"] == "lexicality") &
        (unit_fi_stats["analysis_view"].isin(short_views))
    ]
    if lex_stats.empty:
        return

    combos = _sorted_combos(lex_stats)
    views  = _sorted_views(short_views)
    n_cols = len(combos)
    n_rows = len(views)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(max(6, 5 * n_cols), max(3, 2.5 * n_rows)),
        squeeze=False,
    )

    for j, (model, layer) in enumerate(combos):
        for i, av in enumerate(views):
            ax = axes[i][j]
            sub_stats = lex_stats[
                (lex_stats["model_name"] == model) &
                (lex_stats["layer"] == layer) &
                (lex_stats["analysis_view"] == av)
            ].nlargest(top_k, "max_fi_over_time")

            if sub_stats.empty:
                ax.axis("off")
                continue

            top_neurons = sub_stats["neuron"].values
            sub_fi = fi_df[
                (fi_df["model_name"] == model) &
                (fi_df["layer"] == layer) &
                (fi_df["analysis_view"] == av) &
                (fi_df["feature"] == "lexicality") &
                (fi_df["neuron"].isin(top_neurons))
            ]
            pivot = (
                sub_fi
                .pivot_table(index="neuron", columns="time_bin", values="fi_mean", fill_value=0)
                .reindex(top_neurons)
            )

            im = ax.imshow(pivot.values, aspect="auto", cmap="Reds", interpolation="nearest")
            ax.set_yticks(range(len(pivot)))
            ax.set_yticklabels(pivot.index, fontsize=5)
            ax.set_xlabel("Time bin", fontsize=8)
            ax.set_title(f"{model}/{layer}\n{av}", fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Top-{top_k} lexicality units: FI over time (short_only)", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "top_lexicality_units.png", dpi=_DPI)
    plt.close(fig)


def _plot_speaker_unit_scatter(unit_fi_stats: pd.DataFrame, output_dir: Path) -> None:
    """Speaker FI vs lexicality/morphology FI — speakerctrl views only."""
    speakerctrl_views = [
        v for v in unit_fi_stats["analysis_view"].unique() if "speakerctrl" in v
    ]
    if not speakerctrl_views:
        print("[INFO] No speakerctrl views — skipping speaker_unit_scatter.png")
        return

    sub = unit_fi_stats[unit_fi_stats["analysis_view"].isin(speakerctrl_views)]
    pivot = sub.pivot_table(
        index=_UNIT_ID_COLS, columns="feature", values="max_fi_over_time"
    ).reset_index()

    if "speaker" not in pivot.columns or pivot.empty:
        print("[INFO] 'speaker' feature absent in speakerctrl views — skipping speaker_unit_scatter.png")
        return

    combos = _sorted_combos(pivot)
    views  = _sorted_views(speakerctrl_views)
    n_cols = len(combos)
    n_rows = len(views)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    for j, (model, layer) in enumerate(combos):
        for i, av in enumerate(views):
            ax = axes[i][j]
            sub_v = pivot[
                (pivot["model_name"] == model) &
                (pivot["layer"] == layer) &
                (pivot["analysis_view"] == av)
            ]
            if sub_v.empty:
                ax.axis("off")
                continue
            speaker_fi = sub_v["speaker"].fillna(0)
            for feat in ("lexicality", "morphology"):
                if feat in sub_v.columns:
                    ax.scatter(
                        speaker_fi, sub_v[feat].fillna(0),
                        alpha=0.25, s=3, color=_FEAT_COLOR[feat],
                        label=feat, rasterized=True,
                    )
            ax.set_xlabel("Speaker max FI", fontsize=8)
            ax.set_ylabel("Linguistic feature max FI", fontsize=8)
            ax.set_title(f"{model}/{layer}\n{av}", fontsize=8)
            ax.legend(fontsize=7, markerscale=3)

    fig.suptitle("Speaker FI vs linguistic features (speakerctrl views)", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "speaker_unit_scatter.png", dpi=_DPI)
    plt.close(fig)


# ── README ─────────────────────────────────────────────────────────────────────


def _write_readme(
    output_dir: Path,
    dominance_counts: pd.DataFrame,
    lex_conc: pd.DataFrame,
    top_k: int,
    min_fi: float,
    min_margin: float,
) -> None:
    lines = [
        "# Unit Inspection Summary",
        "",
        "Generated by `scripts/audio/inspect_encoding_units.py`",
        f"- `--top-k {top_k}`, `--min-fi-for-dominance {min_fi}`, "
        f"`--min-dominance-margin {min_margin}`",
        "",
        "## Dominance counts per analysis_view (peak FI criterion)",
        "",
    ]

    peak = dominance_counts[dominance_counts["dominance_type"] == "peak"]
    combos = sorted(
        peak[["model_name", "layer"]].drop_duplicates().values.tolist(),
        key=lambda c: _combo_sort_key(tuple(c)),
    )
    for model, layer in combos:
        lines.append(f"### {model} / {layer}")
        lines.append("")
        sub = peak[
            (peak["model_name"] == model) & (peak["layer"] == layer)
        ]
        pivot = sub.pivot_table(
            index=["speaker_set", "analysis_view"],
            columns="dominant_feature",
            values="frac_units",
            fill_value=0,
        )
        try:
            lines.append(pivot.to_markdown(floatfmt=".3f"))
        except Exception:
            lines.append(pivot.to_string())
        lines.append("")

    lines += [
        "## Lexicality concentration (fraction of units with max FI > threshold)",
        "",
    ]
    if not lex_conc.empty:
        try:
            lines.append(lex_conc.to_markdown(index=False, floatfmt=".4f"))
        except Exception:
            lines.append(lex_conc.to_string(index=False))
        lines.append("")

    lines += [
        "## Output files",
        "",
        "| File | Description |",
        "|---|---|",
        "| `unit_fi_stats.csv` | Per-unit × feature: max/mean FI, peak relative time, fi_std |",
        "| `unit_dominance.csv` | Per-unit: peak+mean dominant features, top/second FI, score stats |",
        "| `dominance_counts.csv` | Unit counts + fractions per dominant feature (peak & mean) |",
        "| `top_units_by_feature.csv` | Top-k units per feature × view, with score stats |",
        "| `lexicality_concentration.csv` | Fraction of units above 5 FI thresholds per view |",
        "| `unit_dominance_counts.png` | Stacked bar (fractions) — peak dominance per view |",
        "| `lexicality_concentration.png` | Threshold curves per view |",
        "| `lexicality_vs_morphology_scatter.png` | Scatter: morph vs lex FI per unit |",
        "| `peak_time_short_only.png` | Peak Spearman time — lex-dominant units (short_only) |",
        "| `top_lexicality_units.png` | FI heatmap for top-k lex units (short_only views) |",
        "| `speaker_unit_scatter.png` | Speaker FI vs linguistic features (speakerctrl views) |",
        "",
    ]

    (output_dir / "summary_readme.md").write_text("\n".join(lines))


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--fi-dirs", nargs="+", required=True, metavar="SPEAKER_SET:PATH",
        help="One or more 'SPEAKER_SET:PATH' items pointing to univariate output dirs.",
    )
    p.add_argument("--output", required=True, type=Path, metavar="DIR")
    p.add_argument(
        "--top-k", type=int, default=20, dest="top_k", metavar="K",
        help="Number of top units per feature to report (default 20).",
    )
    p.add_argument(
        "--min-fi-for-dominance", type=float, default=0.001, dest="min_fi",
        metavar="FI",
        help="Minimum top FI to assign a dominant feature (default 0.001).",
    )
    p.add_argument(
        "--min-dominance-margin", type=float, default=0.003, dest="min_margin",
        metavar="MARGIN",
        help="Minimum gap between top and second FI to call dominance (default 0.003).",
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    output_dir: Path = args.output
    if output_dir.exists() and not args.overwrite:
        p.error(
            f"Output directory already exists: {output_dir}. "
            "Pass --overwrite to replace."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────

    fi_frames: list[pd.DataFrame] = []
    sc_frames: list[pd.DataFrame] = []

    for item in args.fi_dirs:
        speaker_set, fi_dir = _parse_fi_dir_arg(item)
        print(f"[load] {speaker_set}:{fi_dir}")
        fi, sc = _load_one_dir(speaker_set, fi_dir)
        if not fi.empty:
            fi_frames.append(fi)
        if not sc.empty:
            sc_frames.append(sc)

    if not fi_frames:
        print("[ERROR] No feature_importance.csv files loaded.", file=sys.stderr)
        sys.exit(1)

    fi_df = pd.concat(fi_frames, ignore_index=True)
    sc_df = pd.concat(sc_frames, ignore_index=True) if sc_frames else pd.DataFrame()

    n_views       = fi_df["analysis_view"].nunique()
    n_unit_views  = fi_df[_UNIT_ID_COLS].drop_duplicates().shape[0]
    print(
        f"[info] Loaded FI: {len(fi_df)} rows | {n_views} analysis_views | "
        f"{n_unit_views} unit×view instances"
    )

    # ── Derived tables ────────────────────────────────────────────────────────

    print("[compute] Building unit FI stats …")
    unit_fi_stats = _build_unit_fi_stats(fi_df)

    print("[compute] Building score stats …")
    score_stats = _build_score_stats(sc_df)

    print("[compute] Building unit dominance …")
    dominance_df     = _build_unit_dominance(unit_fi_stats, score_stats, args.min_fi, args.min_margin)
    dominance_counts = _build_dominance_counts(dominance_df)
    top_units        = _build_top_units_by_feature(unit_fi_stats, score_stats, args.top_k)
    lex_conc         = _build_lexicality_concentration(unit_fi_stats)

    # ── CSVs ──────────────────────────────────────────────────────────────────

    unit_fi_stats.to_csv(output_dir / "unit_fi_stats.csv", index=False)
    dominance_df.to_csv(output_dir / "unit_dominance.csv", index=False)
    dominance_counts.to_csv(output_dir / "dominance_counts.csv", index=False)
    if not top_units.empty:
        top_units.to_csv(output_dir / "top_units_by_feature.csv", index=False)
    lex_conc.to_csv(output_dir / "lexicality_concentration.csv", index=False)
    print(
        "[csv] unit_fi_stats.csv  unit_dominance.csv  dominance_counts.csv  "
        "top_units_by_feature.csv  lexicality_concentration.csv"
    )

    # ── Figures ───────────────────────────────────────────────────────────────

    _plot_dominance_counts(dominance_counts, output_dir)
    print("[fig] unit_dominance_counts.png")

    _plot_lexicality_concentration(unit_fi_stats, output_dir)
    print("[fig] lexicality_concentration.png")

    _plot_lex_vs_morph_scatter(unit_fi_stats, output_dir)
    print("[fig] lexicality_vs_morphology_scatter.png")

    _plot_peak_time_short_only(dominance_df, sc_df, output_dir)
    print("[fig] peak_time_short_only.png")

    _plot_top_lexicality_units(fi_df, unit_fi_stats, args.top_k, output_dir)
    print("[fig] top_lexicality_units.png")

    _plot_speaker_unit_scatter(unit_fi_stats, output_dir)
    print("[fig] speaker_unit_scatter.png")

    # ── README ────────────────────────────────────────────────────────────────

    _write_readme(
        output_dir, dominance_counts, lex_conc, args.top_k, args.min_fi, args.min_margin
    )
    print("[readme] summary_readme.md")

    print(f"\n[done] All outputs written to: {output_dir}")


if __name__ == "__main__":
    main()