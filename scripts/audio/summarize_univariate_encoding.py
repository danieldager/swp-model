"""CLI for summarising univariate Ridge encoding results.

Reads one or more run_univariate_encoding() output directories and produces
four aggregated CSV files in --output.

Usage
-----
    python scripts/audio/summarize_univariate_encoding.py \\
      --inputs \\
        reproduce/figures/audio/encoding/encodec__7f7d3b97/encoder_out/univariate/all_items/ \\
        reproduce/figures/audio/encoding/encodec__7f7d3b97/encoder_out/univariate/words_only/ \\
        reproduce/figures/audio/encoding/encodec__7f7d3b97/decoder_in/univariate/all_items/ \\
        reproduce/figures/audio/encoding/encodec__7f7d3b97/decoder_in/univariate/words_only/ \\
      --output reproduce/figures/audio/encoding/encodec__7f7d3b97/univariate_summary/ \\
      --top-k 20 \\
      --overwrite

Outputs (in --output directory)
--------------------------------
    score_summary_by_time.csv            Performance across neurons, per time bin.
    weight_summary_by_feature_time.csv   Weight stats across neurons, per feature × time bin.
    top_units_by_feature.csv             Top-k neurons by max |weight| per feature.
    global_feature_ranking.csv           Feature ranking by mean |weight| over time and neurons.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from swp.audio.encoding.univariate_summary import summarize


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarise univariate encoding outputs into aggregated CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--inputs", nargs="+", required=True, type=Path,
        metavar="DIR",
        help="One or more run_univariate_encoding output directories.",
    )
    p.add_argument(
        "--output", required=True, type=Path,
        metavar="DIR",
        help="Directory where summary CSVs will be written.",
    )
    p.add_argument(
        "--top-k", type=int, default=20,
        dest="top_k",
        help="Number of top neurons to keep per feature in top_units_by_feature.csv.",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing outputs without error.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print(f"[summarize_univariate] inputs  : {len(args.inputs)} director(ies)")
    for d in args.inputs:
        print(f"  {d}")
    print(f"[summarize_univariate] output  : {args.output}")
    print(f"[summarize_univariate] top_k   : {args.top_k}")

    out = summarize(
        input_dirs=args.inputs,
        output_dir=args.output,
        top_k=args.top_k,
        overwrite=args.overwrite,
    )

    score_summary   = out["score_summary"]
    weight_summary  = out["weight_summary"]
    top_units       = out["top_units"]
    feature_ranking = out["feature_ranking"]

    nan_checks = {
        "score_summary/mean_score":               score_summary["mean_score"].isna().any(),
        "weight_summary/mean_abs_weight":          weight_summary["mean_abs_weight"].isna().any(),
        "top_units/max_abs_weight":                top_units["max_abs_weight"].isna().any(),
        "feature_ranking/mean_abs_weight_…":       feature_ranking["mean_abs_weight_over_time_and_neurons"].isna().any(),
    }

    print("\n[summarize_univariate] NaN checks:")
    any_nan = False
    for key, has_nan in nan_checks.items():
        flag = "FAIL" if has_nan else "OK"
        print(f"  {flag}  {key}")
        any_nan = any_nan or has_nan

    print(f"\n[summarize_univariate] Shapes:")
    print(f"  score_summary   : {score_summary.shape}")
    print(f"  weight_summary  : {weight_summary.shape}")
    print(f"  top_units       : {top_units.shape}")
    print(f"  feature_ranking : {feature_ranking.shape}")

    print(f"\n[summarize_univariate] global_feature_ranking.csv (first rows):")
    print(feature_ranking.to_string(index=False))

    print(f"\n[summarize_univariate] score_summary_by_time.csv (first 8 rows):")
    print(score_summary.head(8).to_string(index=False))

    print(f"\n[summarize_univariate] Done — outputs in {args.output}")

    if any_nan:
        print(
            "[summarize_univariate] WARNING: unexpected NaN in outputs.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()