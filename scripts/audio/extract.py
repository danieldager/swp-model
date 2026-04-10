#!/usr/bin/env python3
"""CLI entry-point for the audio extraction pipeline.

Runs reconstruction metrics and activation extraction for every item in
a processed paradigm CSV, saving results under reproduce/data/audio/{run_id}/.

Example:
    python scripts/audio/extract.py \\
        --model encodec --model-arg bandwidth=6.0 \\
        --dataset data/external/paradigm/processed/subset_male.csv \\
        --layers encoder_out decoder_in decoder_out \\
        --output reproduce/data/audio/
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse

import swp.audio.models.encodec  # noqa: F401 — registers encodec in the registry
from swp.audio.datasets.base import ParadigmDataset
from swp.audio.models import get_model
from swp.audio.pipeline.extraction import run_extraction


def _parse_model_arg(s: str) -> tuple[str, int | float | bool | str]:
    """Parse 'key=value' into (key, typed_value).

    Tries int, then float, then bool ('true'/'false'), then falls back to str.
    """
    key, _, raw = s.partition("=")
    key = key.strip()
    raw = raw.strip()
    for cast in (int, float):
        try:
            return key, cast(raw)
        except ValueError:
            pass
    if raw.lower() in ("true", "false"):
        return key, raw.lower() == "true"
    return key, raw


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract audio metrics and activations for a paradigm dataset."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name (e.g. encodec)",
    )
    parser.add_argument(
        "--model-arg",
        dest="model_args",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Model constructor kwargs (repeatable). E.g. --model-arg bandwidth=6.0",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to processed subset.csv",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        required=True,
        help="Layer names to extract (space-separated stable names)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Base output directory (e.g. reproduce/data/audio/)",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Overwrite an existing run instead of skipping",
    )
    args = parser.parse_args()

    model_kwargs = dict(_parse_model_arg(a) for a in args.model_args)

    # Resolve dataset path and make repo-relative for paramhash
    dataset_abs = Path(args.dataset).resolve()
    try:
        dataset_rel = str(dataset_abs.relative_to(REPO_ROOT))
    except ValueError:
        # Dataset outside repo root — use absolute path
        dataset_rel = str(dataset_abs)

    print(f"[extract] Model       : {args.model}  kwargs={model_kwargs}")
    print(f"[extract] Dataset     : {dataset_rel}")
    print(f"[extract] Layers      : {args.layers}")
    print(f"[extract] Output base : {args.output}")

    model = get_model(args.model, **model_kwargs)
    dataset = ParadigmDataset(
        dataset_abs,
        target_sr=model.sample_rate,
        repo_root=REPO_ROOT,
    )
    print(f"[extract] Items       : {len(dataset)}")

    run_extraction(
        model=model,
        dataset=dataset,
        layers=args.layers,
        output_dir=Path(args.output),
        dataset_path=dataset_rel,
        model_kwargs=model_kwargs,
        regenerate=args.regenerate,
    )


if __name__ == "__main__":
    main()