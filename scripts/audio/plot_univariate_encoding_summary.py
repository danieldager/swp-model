"""CLI for plotting univariate encoding summary CSVs.

Reads score_summary_by_time.csv, weight_summary_by_feature_time.csv, and
global_feature_ranking.csv from --summary-dir, then writes PNG files to --output.

Usage — base figures only
--------------------------
    python scripts/audio/plot_univariate_encoding_summary.py \\
      --summary-dir reproduce/figures/audio/encoding/encodec__7f7d3b97/univariate_summary/ \\
      --output reproduce/figures/audio/encoding/encodec__7f7d3b97/univariate_summary/plots/ \\
      --overwrite

Usage — with encoding_FI figures (requires --compute-fi runs)
--------------------------------------------------------------
    python scripts/audio/plot_univariate_encoding_summary.py \\
      --summary-dir reproduce/figures/audio/encoding/encodec__7f7d3b97/univariate_summary/ \\
      --output reproduce/figures/audio/encoding/encodec__7f7d3b97/univariate_summary/plots/ \\
      --fi-dirs \\
        reproduce/figures/audio/encoding/encodec__7f7d3b97/encoder_out/univariate/all_items/ \\
        reproduce/figures/audio/encoding/encodec__7f7d3b97/encoder_out/univariate/words_only/ \\
        reproduce/figures/audio/encoding/encodec__7f7d3b97/decoder_in/univariate/all_items/ \\
        reproduce/figures/audio/encoding/encodec__7f7d3b97/decoder_in/univariate/words_only/ \\
      --overwrite

Outputs (in --output directory)
--------------------------------
    score_over_time_{model_name}_r2.png
    score_over_time_{model_name}_spearman.png      (skipped with warning if metric absent)
    weights_over_time_{model_name}_all_items.png
    weights_over_time_{model_name}_words_only.png
    global_feature_ranking_{model_name}.png
    encoding_fi_{model_name}_{layer}_{analysis_set}_{metric}.png  (one per run dir in --fi-dirs, if feature_importance.csv present)

model_name is extracted from the summary CSVs. If multiple model names are found in a
single --summary-dir, the script exits with an error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from swp.audio.encoding.univariate_plots import (
    plot_score_over_time,
    plot_weights_over_time,
    plot_global_feature_ranking,
    plot_encoding_fi,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot univariate encoding summary CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--summary-dir", required=True, type=Path,
        dest="summary_dir",
        metavar="DIR",
        help="Directory produced by summarize_univariate_encoding.py.",
    )
    p.add_argument(
        "--output", required=True, type=Path,
        metavar="DIR",
        help="Output directory for PNG figures (created if absent).",
    )
    p.add_argument(
        "--fi-dirs", nargs="*", type=Path, default=None,
        dest="fi_dirs",
        metavar="DIR",
        help=(
            "Optional: one or more run_univariate_encoding output directories "
            "containing feature_importance.csv and scores.csv. "
            "Produces encoding_fi_{layer}_{analysis_set}_{metric}.png per directory. "
            "Directories missing feature_importance.csv are skipped with a warning."
        ),
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing PNG files without error.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    summary_dir = Path(args.summary_dir)
    output_dir  = Path(args.output)

    # ── Load CSVs ──────────────────────────────────────────────────────────────

    required = {
        "score_summary_by_time.csv":           "score_summary",
        "weight_summary_by_feature_time.csv":  "weight_summary",
        "global_feature_ranking.csv":          "feature_ranking",
    }
    for fname in required:
        p = summary_dir / fname
        if not p.exists():
            print(
                f"[plot_univariate] ERROR: expected file not found: {p}",
                file=sys.stderr,
            )
            sys.exit(1)

    score_summary   = pd.read_csv(summary_dir / "score_summary_by_time.csv")
    weight_summary  = pd.read_csv(summary_dir / "weight_summary_by_feature_time.csv")
    feature_ranking = pd.read_csv(summary_dir / "global_feature_ranking.csv")

    # ── Extract model_name — error if summary mixes multiple models ───────────

    model_names = score_summary["model_name"].unique()
    if len(model_names) > 1:
        print(
            f"[plot_univariate] ERROR: --summary-dir contains multiple model names: "
            f"{sorted(model_names)}. Each summary directory must correspond to a "
            "single model. Run summarize_univariate_encoding.py per model.",
            file=sys.stderr,
        )
        sys.exit(1)
    model_name = str(model_names[0])
    print(f"[plot_univariate] model_name  : {model_name}")

    run_ids = score_summary["run_id"].unique()
    if len(run_ids) > 1:
        print(
            f"[plot_univariate] ERROR: --summary-dir contains multiple run_ids: "
            f"{sorted(run_ids)}. Each summary directory must correspond to a "
            "single run. Run summarize_univariate_encoding.py per run.",
            file=sys.stderr,
        )
        sys.exit(1)
    run_id = str(run_ids[0])
    print(f"[plot_univariate] run_id      : {run_id}")

    # ── Validate output dir ────────────────────────────────────────────────────

    output_dir.mkdir(parents=True, exist_ok=True)

    expected_pngs = [
        f"score_over_time_{model_name}_r2.png",
        f"score_over_time_{model_name}_spearman.png",
        f"weights_over_time_{model_name}_all_items.png",
        f"weights_over_time_{model_name}_words_only.png",
        f"global_feature_ranking_{model_name}.png",
    ]
    if not args.overwrite:
        existing = [output_dir / f for f in expected_pngs if (output_dir / f).exists()]
        if existing:
            print(
                f"[plot_univariate] ERROR: output files already exist: "
                f"{[p.name for p in existing]}. Use --overwrite to replace.",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"[plot_univariate] summary_dir : {summary_dir}")
    print(f"[plot_univariate] output      : {output_dir}")

    # ── Figures 1–5: per-analysis_view or legacy aggregate ───────────────────

    has_views = "analysis_view" in score_summary.columns

    if has_views:
        all_views = sorted(score_summary["analysis_view"].unique())
        print(
            f"\n[plot_univariate] analysis_view column detected "
            f"({len(all_views)} view(s)) — skipping aggregate figures; "
            "generating one set of figures per analysis_view."
        )
        for av in all_views:
            print(f"\n[plot_univariate] ── view: {av}")
            ss_v = score_summary[score_summary["analysis_view"] == av]
            ws_v = weight_summary[weight_summary["analysis_view"] == av]
            fr_v = feature_ranking[feature_ranking["analysis_view"] == av]

            # Each analysis_view maps to exactly one analysis_set
            aset = ws_v["analysis_set"].iloc[0] if not ws_v.empty else "all_items"

            for metric in ("r2", "spearman"):
                fname = f"score_over_time_{model_name}_{av}_{metric}.png"
                saved = plot_score_over_time(
                    ss_v, metric, output_dir / fname,
                    model_name=model_name, run_id=run_id,
                    analysis_view=av,
                )
                if saved:
                    print(f"  saved  {fname}")
                else:
                    print(f"  [warning] metric '{metric}' not found for view '{av}' — skipped")

            fname = f"weights_over_time_{model_name}_{av}.png"
            plot_weights_over_time(
                ws_v, aset, output_dir / fname,
                model_name=model_name, run_id=run_id,
                analysis_view=av,
            )
            print(f"  saved  {fname}")

            if not fr_v.empty:
                fname = f"global_feature_ranking_{model_name}_{av}.png"
                plot_global_feature_ranking(
                    fr_v, output_dir / fname,
                    model_name=model_name, run_id=run_id,
                    analysis_view=av,
                )
                print(f"  saved  {fname}")

    else:
        # ── Legacy aggregate figures (summary has no analysis_view column) ────
        for metric in ("r2", "spearman"):
            fname = f"score_over_time_{model_name}_{metric}.png"
            saved = plot_score_over_time(
                score_summary, metric, output_dir / fname,
                model_name=model_name, run_id=run_id,
            )
            if saved:
                print(f"  saved  {fname}")
            else:
                print(f"  [warning] metric '{metric}' not found in score_summary — {fname} skipped")

        for aset in ("all_items", "words_only"):
            fname = f"weights_over_time_{model_name}_{aset}.png"
            plot_weights_over_time(
                weight_summary, aset, output_dir / fname,
                model_name=model_name, run_id=run_id,
            )
            print(f"  saved  {fname}")

        fname = f"global_feature_ranking_{model_name}.png"
        plot_global_feature_ranking(
            feature_ranking, output_dir / fname,
            model_name=model_name, run_id=run_id,
        )
        print(f"  saved  {fname}")

    # ── Figure 6+: encoding_FI (optional, requires --fi-dirs) ────────────────

    if args.fi_dirs:
        for fi_dir in args.fi_dirs:
            fi_dir = Path(fi_dir)
            fi_path = fi_dir / "feature_importance.csv"
            sc_path = fi_dir / "scores.csv"

            if not fi_path.exists():
                print(
                    f"  [warning] feature_importance.csv not found in {fi_dir} — "
                    "run with --compute-fi to generate it; skipping"
                )
                continue
            if not sc_path.exists():
                print(
                    f"  [warning] scores.csv not found in {fi_dir} — skipping"
                )
                continue

            fi_df_run     = pd.read_csv(fi_path)
            scores_df_run = pd.read_csv(sc_path)

            # analysis_view is the directory name (e.g. "all_items_short_only_speakerctrl")
            analysis_view = fi_dir.name

            # Detect all (run_id, model_name, layer, analysis_set, metric) combos
            combos = (
                scores_df_run[["run_id", "model_name", "layer", "analysis_set", "metric"]]
                .drop_duplicates()
                .values.tolist()
            )
            for run_id, model_name, layer, analysis_set, metric in combos:
                fname = f"encoding_fi_{model_name}_{layer}_{analysis_view}_{metric}.png"
                saved = plot_encoding_fi(
                    fi_df_run, scores_df_run,
                    layer=layer,
                    analysis_set=analysis_set,
                    metric=metric,
                    output_path=output_dir / fname,
                    model_name=model_name,
                    run_id=run_id,
                )
                if saved:
                    print(f"  saved  {fname}")
                else:
                    print(
                        f"  [warning] no data for {model_name}/{layer}/{analysis_set}/{metric} "
                        f"in {fi_dir} — {fname} skipped"
                    )

    print(f"\n[plot_univariate] Done — {output_dir}")


if __name__ == "__main__":
    main()