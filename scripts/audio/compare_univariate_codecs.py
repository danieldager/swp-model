"""CLI for comparing univariate encoding summaries across codecs.

Reads one summary directory per codec (produced by summarize_univariate_encoding.py)
and writes combined CSVs and comparison figures to --output.

Usage
-----
    python scripts/audio/compare_univariate_codecs.py \\
      --summaries \\
        reproduce/figures/audio/encoding/encodec__7f7d3b97/univariate_summary/ \\
        reproduce/figures/audio/encoding/dac__f104bbdd/univariate_summary/ \\
      --labels encodec dac \\
      --output reproduce/figures/audio/encoding/codec_comparison/ \\
      --overwrite

Outputs (in --output directory)
--------------------------------
CSVs
    combined_global_fi_ranking.csv
    combined_fi_summary_by_feature_time.csv
    combined_score_summary_by_time.csv
    length_dominance_ratio.csv

Figures
    global_fi_ranking_compare.png
    global_abs_fi_ranking_compare.png
    fi_over_time_compare_length_bin.png
    fi_over_time_compare_all_features_all_items.png
    fi_over_time_compare_all_features_words_only.png
    score_over_time_compare_r2.png
    score_over_time_compare_spearman.png
    length_dominance_ratio.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from swp.audio.encoding.univariate_compare import compare


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare univariate encoding summaries across codecs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--summaries", nargs="+", required=True, type=Path,
        metavar="DIR",
        help="One summary directory per codec (summarize_univariate_encoding output).",
    )
    p.add_argument(
        "--labels", nargs="+", required=True, type=str,
        metavar="LABEL",
        help="Short label for each codec, same order as --summaries (e.g. encodec dac).",
    )
    p.add_argument(
        "--output", required=True, type=Path,
        metavar="DIR",
        help="Output directory for CSV files and PNG figures.",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing outputs without error.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if len(args.summaries) != len(args.labels):
        print(
            f"[compare_univariate] ERROR: --summaries has {len(args.summaries)} entries "
            f"but --labels has {len(args.labels)}. They must match.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[compare_univariate] codecs   : {args.labels}")
    for lbl, d in zip(args.labels, args.summaries):
        print(f"  {lbl:12s}  →  {d}")
    print(f"[compare_univariate] output   : {args.output}")

    out = compare(
        summary_dirs=args.summaries,
        labels=args.labels,
        output_dir=args.output,
        overwrite=args.overwrite,
    )

    # ── Summary report ────────────────────────────────────────────────────────

    print("\n[compare_univariate] Shapes:")
    print(f"  combined_global_fi     : {out['combined_global_fi'].shape}")
    print(f"  combined_fi_summary    : {out['combined_fi_summary'].shape}")
    print(f"  combined_score_summary : {out['combined_score_summary'].shape}")
    print(f"  length_dominance       : {out['length_dominance'].shape}")

    nan_checks: dict[str, bool] = {
        "combined_global_fi / mean_fi":         out["combined_global_fi"]["mean_fi_over_time_and_neurons"].isna().any(),
        "combined_fi_summary / mean_fi":        out["combined_fi_summary"]["mean_fi"].isna().any(),
        "combined_score_summary / mean_score":  out["combined_score_summary"]["mean_score"].isna().any(),
        "length_dominance / length_to_other_ratio": out["length_dominance"]["length_to_other_ratio"].isna().any(),
    }
    print("\n[compare_univariate] NaN checks:")
    any_nan = False
    for key, has_nan in nan_checks.items():
        flag = "FAIL" if has_nan else "OK"
        print(f"  {flag}  {key}")
        any_nan = any_nan or has_nan

    print("\n[compare_univariate] length_dominance_ratio.csv:")
    print(out["length_dominance"].to_string(index=False))

    print(f"\n[compare_univariate] Done — {args.output}")

    if any_nan:
        print(
            "[compare_univariate] WARNING: unexpected NaN in outputs.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()