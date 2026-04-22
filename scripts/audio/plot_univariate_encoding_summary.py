"""CLI for plotting univariate encoding summary CSVs.

Reads score_summary_by_time.csv, weight_summary_by_feature_time.csv, and
global_feature_ranking.csv from --summary-dir, then writes 5 PNG files to --output.

Usage
-----
    python scripts/audio/plot_univariate_encoding_summary.py \\
      --summary-dir reproduce/figures/audio/encoding/encodec__7f7d3b97/univariate_summary/ \\
      --output reproduce/figures/audio/encoding/encodec__7f7d3b97/univariate_summary/plots/ \\
      --overwrite

Outputs (in --output directory)
--------------------------------
    score_over_time_r2.png
    score_over_time_spearman.png       (skipped with warning if metric absent)
    weights_over_time_all_items.png
    weights_over_time_words_only.png
    global_feature_ranking.png
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

    # ── Validate output dir ────────────────────────────────────────────────────

    output_dir.mkdir(parents=True, exist_ok=True)

    expected_pngs = [
        "score_over_time_r2.png",
        "score_over_time_spearman.png",
        "weights_over_time_all_items.png",
        "weights_over_time_words_only.png",
        "global_feature_ranking.png",
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

    # ── Figures 1 & 2: scores over time ───────────────────────────────────────

    for metric in ("r2", "spearman"):
        fname = f"score_over_time_{metric}.png"
        saved = plot_score_over_time(score_summary, metric, output_dir / fname)
        if saved:
            print(f"  saved  {fname}")
        else:
            print(f"  [warning] metric '{metric}' not found in score_summary — {fname} skipped")

    # ── Figures 3 & 4: weights over time ──────────────────────────────────────

    for aset in ("all_items", "words_only"):
        fname = f"weights_over_time_{aset}.png"
        plot_weights_over_time(weight_summary, aset, output_dir / fname)
        print(f"  saved  {fname}")

    # ── Figure 5: global feature ranking ──────────────────────────────────────

    plot_global_feature_ranking(feature_ranking, output_dir / "global_feature_ranking.png")
    print(f"  saved  global_feature_ranking.png")

    print(f"\n[plot_univariate] Done — {output_dir}")


if __name__ == "__main__":
    main()