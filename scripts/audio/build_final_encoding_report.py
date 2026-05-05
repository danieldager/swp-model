#!/usr/bin/env python3
"""Assemble canonical final figures and write report.md for audio encoding analyses.

Copies the 8 main canonical figures from length_controlled_summary/ and
unit_inspection/ into a single final/ directory. Does NOT recompute any analyses.
Does NOT pull from diagnostic_plots/ or plots_aggregate_legacy/.

Usage (run from swp-model/ repo root):
    python scripts/audio/build_final_encoding_report.py \\
        --length-controlled-summary reproduce/figures/audio/encoding/length_controlled_summary/ \\
        --unit-inspection reproduce/figures/audio/encoding/unit_inspection/ \\
        --output reproduce/figures/audio/encoding/final/ \\
        --overwrite

Outputs (in --output directory)
--------------------------------
    01_fi_global_heatmap.png
    02_male_only_short_long_fi_summary.png
    03_short_only_morphology_vs_lexicality.png
    04_long_only_morphology_vs_lexicality.png
    05_speaker_effect_summary.png
    06_temporal_fi_short_only.png
    07_unit_dominance_counts.png
    08_lexicality_concentration.png
    report.md
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


# ── Canonical figure manifest ─────────────────────────────────────────────────
#
# Each entry: (numbered_output_name, source_dir_key, source_filename)
# source_dir_key: "lc" = length_controlled_summary, "ui" = unit_inspection

_MAIN_FIGURES: list[tuple[str, str, str]] = [
    ("01_fi_global_heatmap.png",                    "lc", "fi_global_heatmap.png"),
    ("02_male_only_short_long_fi_summary.png",      "lc", "male_only_short_long_fi_summary.png"),
    ("03_short_only_morphology_vs_lexicality.png",  "lc", "short_only_morphology_vs_lexicality.png"),
    ("04_long_only_morphology_vs_lexicality.png",   "lc", "long_only_morphology_vs_lexicality.png"),
    ("05_speaker_effect_summary.png",               "lc", "speaker_effect_summary.png"),
    ("06_temporal_fi_short_only.png",               "lc", "temporal_fi_short_only.png"),
    ("07_unit_dominance_counts.png",                "ui", "unit_dominance_counts.png"),
    ("08_lexicality_concentration.png",             "ui", "lexicality_concentration.png"),
]


# ── Report generation ─────────────────────────────────────────────────────────


def _load_fi_summary(lc_dir: Path) -> pd.DataFrame | None:
    """Load fi_global_comparison.csv from the length-controlled summary dir."""
    p = lc_dir / "fi_global_comparison.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


def _load_lex_conc(ui_dir: Path) -> pd.DataFrame | None:
    p = ui_dir / "lexicality_concentration.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


def _format_top_features(fi_df: pd.DataFrame) -> str:
    """Auto-populate a findings table from fi_global_comparison.csv."""
    lines = []
    key_views = [
        "all_items_all_speakers",
        "all_items_short_only",
        "all_items_long_only",
    ]
    _MODEL_ORDER = ["encodec", "dac"]
    _LAYER_ORDER = ["encoder_out", "decoder_in"]

    for view in key_views:
        sub = fi_df[fi_df["analysis_view"] == view]
        if sub.empty:
            continue
        lines.append(f"#### `{view}`")
        lines.append("")
        lines.append("| model | layer | top feature | mean FI |")
        lines.append("|---|---|---|---|")
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
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    lc_dir: Path,
    ui_dir: Path,
    figures_copied: list[str],
    figures_missing: list[str],
    fi_df: pd.DataFrame | None,
    lex_conc: pd.DataFrame | None,
) -> None:
    ts = datetime.now().isoformat(timespec="seconds")

    lines = [
        "# Audio Encoding Analysis — Final Report",
        "",
        f"Generated: {ts}  ",
        f"Script: `scripts/audio/build_final_encoding_report.py`",
        "",
        "---",
        "",
        "## 1. Methodology",
        "",
        "### analysis_set vs analysis_view",
        "",
        "- **`analysis_set`**: the feature family used in the Ridge design matrix.",
        "  Values: `all_items` (lexicality, length_bin, morphology) and",
        "  `words_only` (frequency_bin, length_bin, morphology).",
        "- **`analysis_view`**: the concrete experimental condition, encoding both",
        "  the feature family and the trial filter (and optional speaker covariate).",
        "  Examples: `all_items_short_only`, `all_items_all_speakers_speakerctrl`.",
        "",
        "  Never group summaries or plots by `analysis_set` alone when multiple",
        "  `analysis_view` values are present. `analysis_view` is the primary key.",
        "",
        "### speakerctrl",
        "",
        "  `speakerctrl` models speaker identity as an additional covariate in the",
        "  Ridge design matrix (effect-coded MALE=−1, FEMALE=+1). It does not remove",
        "  speaker information from activations; it allows the regression to explicitly",
        "  attribute variance to the speaker variable. Weights and FI for linguistic",
        "  features in speakerctrl views reflect the residual after accounting for speaker.",
        "",
        "### FI vs weights",
        "",
        "  **Permutation feature importance (FI)** is the canonical metric.",
        "  Ridge regression weights reflect the design matrix structure and can be",
        "  inflated by feature correlations. `global_feature_ranking.csv` and weight",
        "  plots are **diagnostic only**. `global_fi_ranking.csv` and FI figures are",
        "  **canonical**. Weight and FI rankings can disagree — for example, in DAC",
        "  long_only, the weight ranking differs from the FI ranking.",
        "",
        "---",
        "",
        "## 2. Key Findings",
        "",
        "### Full views",
        "  `length_bin` is rank 1 for both EnCodec and DAC across both layers in",
        "  `all_items_all_speakers`. Length dominates the population-level encoding.",
        "",
        "### Short-only (after length control)",
        "  Morphology is the robust residual psycholinguistic factor after restricting",
        "  to short items. The effect is consistent across both models and speaker sets.",
        "  Lexicality remains weak at the population level.",
        "",
        "### Long-only",
        "  Effects are weaker and more mixed. DAC shows a small lexicality-leading",
        "  signal in some configurations, but this is not clearly robust across",
        "  all-speakers and both models.",
        "",
        "### Speaker identity",
        "  Speaker identity is a very strong acoustic axis. In speakerctrl views,",
        "  speaker FI is comparable to or exceeds `length_bin` in some configurations.",
        "",
        "### Lexicality at unit level",
        "  Lexicality is sparse and localized. It is not absent globally, but",
        "  concentrated in a small fraction of units — predominantly in DAC decoder_in",
        "  short_only. At the population level it is dominated by morphology and length.",
        "",
        "### Scope of unit-level inspection",
        "  Unit-level inspection focuses on `all_items` views, because the main",
        "  unit-level question is lexicality vs morphology after length control.",
        "  `words_only`/frequency analyses remain available in per-run summaries and",
        "  diagnostic outputs (`diagnostic_plots/`), but are not part of the canonical",
        "  unit-level inspection.",
        "",
    ]

    if fi_df is not None:
        lines += [
            "### Top features per view (auto-populated from fi_global_comparison.csv)",
            "",
            _format_top_features(fi_df),
        ]

    if lex_conc is not None and not lex_conc.empty:
        lines += [
            "### Lexicality concentration",
            "",
            "See `unit_inspection/lexicality_concentration.csv` for full per-view data,",
            "and `08_lexicality_concentration.png` for the threshold curve figure.",
            "",
            "**Summary**: Many units exceed very small lexicality FI thresholds, but",
            "high-FI lexicality units remain sparse. This supports the interpretation",
            "that lexicality is not absent, but localized/sparse rather than a dominant",
            "population-level factor.",
            "",
        ]
        # Short filtered table: all_items_short_only only, key columns
        short_only = lex_conc[lex_conc["analysis_view"] == "all_items_short_only"].copy()
        key_cols = [c for c in [
            "speaker_set", "model_name", "layer", "n_units",
            "frac_lex_max_fi_above_0_02", "frac_lex_max_fi_above_0_05",
        ] if c in short_only.columns]
        if not short_only.empty and key_cols:
            lines.append("Fraction of units above key thresholds — `all_items_short_only`:")
            lines.append("")
            try:
                lines.append(short_only[key_cols].to_markdown(index=False, floatfmt=".4f"))
            except Exception:
                lines.append(short_only[key_cols].to_string(index=False))
            lines.append("")

    lines += [
        "---",
        "",
        "## 3. Canonical Figures",
        "",
        "All canonical figures use permutation FI as the primary metric.",
        "Weight-based plots and R² plots are diagnostic; see `diagnostic_plots/`.",
        "",
        "| # | Filename | Scientific message |",
        "|---|---|---|",
        "| 01 | `01_fi_global_heatmap.png` | Full picture across all views and models: length dominates full views; morphology is the residual factor after length control |",
        "| 02 | `02_male_only_short_long_fi_summary.png` | Male-only baseline: FI contrast between short_only and long_only |",
        "| 03 | `03_short_only_morphology_vs_lexicality.png` | After removing length variance: morphology > lexicality; consistent across models/speakers |",
        "| 04 | `04_long_only_morphology_vs_lexicality.png` | Long-only residual: weaker and mixed; small DAC lexicality-leading signal |",
        "| 05 | `05_speaker_effect_summary.png` | Speaker identity is a very strong acoustic axis; speakerctrl covariate captures it |",
        "| 06 | `06_temporal_fi_short_only.png` | Temporal profile of residual factors in short_only |",
        "| 07 | `07_unit_dominance_counts.png` | Fraction of units per view dominated by each feature: lex is sparse |",
        "| 08 | `08_lexicality_concentration.png` | Lexicality sparsity: only a small fraction of units exceed any FI threshold |",
        "",
    ]

    if figures_missing:
        lines += [
            "### Missing figures (source not found — re-run upstream scripts)",
            "",
        ]
        for f in figures_missing:
            lines.append(f"- `{f}`")
        lines.append("")

    lines += [
        "---",
        "",
        "## 4. Caveats",
        "",
        "- **Weight plots are diagnostic.** Do not use `global_feature_ranking_*.png`",
        "  or `weights_over_time_*.png` as primary evidence for feature importance.",
        "  Always prefer FI plots.",
        "",
        "- **Negative R² is expected** in per-neuron Ridge regressions when length is",
        "  removed and residual factors explain little average variance. It is not merely",
        "  a regularization artifact; it means the model predicts worse than the mean",
        "  baseline for that neuron. Speakerctrl can improve group-level R² because",
        "  speaker is a strong acoustic predictor, but this does not imply that",
        "  lexicality or morphology effects become stronger. Use Spearman for neuron-level",
        "  quality assessment.",
        "",
        "- **`short_only` / `long_only` views have ≈180 trials.** CV scores are less",
        "  stable than in full-view (360-trial) analyses.",
        "",
        "- **`words_only` short_only / long_only were not run** — too few trials (~45)",
        "  for reliable 5-fold CV.",
        "",
        "- **speakerctrl does not remove speaker from activations.** It partials out",
        "  speaker variance in the regression, but the codec activations still contain",
        "  full speaker information.",
        "",
        "---",
        "",
        "## 5. Source Directories",
        "",
        f"- `length_controlled_summary/`: `{lc_dir}`",
        f"- `unit_inspection/`: `{ui_dir}`",
        "",
        "## 6. Technical Debt",
        "",
        "- `unit_profiles.py` and `univariate_compare.py` lack `analysis_view` in their",
        "  grouping keys and are marked LEGACY. Do not use with multi-view summaries.",
        "- `global_feature_ranking.csv` is weight-based; consider renaming to",
        "  `global_weight_ranking.csv` in a future refactor.",
        "- `analysis_view` is derived from the output directory name at summarize time.",
        "  Never rename run output directories after analysis.",
        "",
    ]

    path = output_dir / "report.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"[build_final_report] Written: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--length-controlled-summary", required=True, type=Path,
        dest="lc_dir", metavar="DIR",
        help="Output directory of summarize_length_controlled_encoding.py.",
    )
    p.add_argument(
        "--unit-inspection", required=True, type=Path,
        dest="ui_dir", metavar="DIR",
        help="Output directory of inspect_encoding_units.py.",
    )
    p.add_argument(
        "--output", required=True, type=Path, metavar="DIR",
        help="Output directory for final figures and report.md.",
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    lc_dir     = Path(args.lc_dir)
    ui_dir     = Path(args.ui_dir)
    output_dir = Path(args.output)

    if not lc_dir.exists():
        print(f"[build_final_report] ERROR: --length-controlled-summary not found: {lc_dir}",
              file=sys.stderr)
        sys.exit(1)
    if not ui_dir.exists():
        print(f"[build_final_report] ERROR: --unit-inspection not found: {ui_dir}",
              file=sys.stderr)
        sys.exit(1)

    if output_dir.exists() and not args.overwrite:
        p.error(
            f"Output directory already exists: {output_dir}. "
            "Pass --overwrite to replace."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    src_dirs = {"lc": lc_dir, "ui": ui_dir}

    figures_copied:  list[str] = []
    figures_missing: list[str] = []

    for dest_name, src_key, src_name in _MAIN_FIGURES:
        src = src_dirs[src_key] / src_name
        dst = output_dir / dest_name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"[build_final_report] Copied: {src_name} → {dest_name}")
            figures_copied.append(dest_name)
        else:
            print(
                f"[build_final_report] WARNING: source not found: {src}  "
                f"(skipping {dest_name})",
                file=sys.stderr,
            )
            figures_missing.append(dest_name)

    fi_df    = _load_fi_summary(lc_dir)
    lex_conc = _load_lex_conc(ui_dir)

    write_report(
        output_dir, lc_dir, ui_dir,
        figures_copied, figures_missing,
        fi_df, lex_conc,
    )

    print(f"\n[build_final_report] Done — {len(figures_copied)} figure(s) copied, "
          f"{len(figures_missing)} missing.")
    print(f"[build_final_report] Output: {output_dir}")

    if figures_missing:
        print(
            f"[build_final_report] Missing figures: {figures_missing}\n"
            "Re-run summarize_length_controlled_encoding.py and/or "
            "inspect_encoding_units.py first.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
