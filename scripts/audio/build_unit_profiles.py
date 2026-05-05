#!/usr/bin/env python
# LEGACY — not analysis_view-safe. Do not use with multi-view summaries.
# Use scripts/audio/inspect_encoding_units.py instead.
# This script wraps unit_profiles.py which lacks analysis_view in _UNIT_COLS and will
# silently collapse short_only / long_only / all_speakers views into the same neuron row.
"""CLI: build unit feature importance profiles from per-neuron FI run directories."""

import warnings
warnings.warn(
    "build_unit_profiles.py is legacy and not analysis_view-safe. "
    "Use inspect_encoding_units.py instead.",
    UserWarning,
    stacklevel=1,
)

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from swp.audio.encoding.unit_profiles import build_profiles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build unit feature profiles from univariate FI run directories."
    )
    parser.add_argument(
        "--fi-dirs", nargs="+", required=True,
        help="Paths to univariate FI run directories (each must contain "
             "feature_importance.csv and config.json).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for CSV profiles and PNG plots.",
    )
    parser.add_argument(
        "--top-k", type=int, default=30,
        help="Number of top units to include in top_* CSV files (default: 30).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite output directory if it already exists.",
    )
    args = parser.parse_args()

    try:
        build_profiles(
            fi_dirs=args.fi_dirs,
            output_dir=args.output,
            top_k=args.top_k,
            overwrite=args.overwrite,
        )
    except FileExistsError as exc:
        print(f"[build_unit_profiles] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()