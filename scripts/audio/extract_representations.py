#!/usr/bin/env python3
"""Extract continuous hidden-state representations for audio stimuli.

For representation-only models (AuriStream, HuBERT, WavLM, etc.) that expose
layer-wise hidden states but do not support audio reconstruction.

Activations are saved as [D, L] tensors under:
    reproduce/data/audio/{model}__{hash}/activations/

Run from swp-model/ repo root:

    python scripts/audio/extract_representations.py \\
        --model auristream \\
        --dataset data/external/paradigm/processed/subset_male.csv \\
        --layers embedding block_12 block_24 block_36 block_48 \\
        --output reproduce/data/audio/

Smoke test (3 items):

    python scripts/audio/extract_representations.py \\
        --model auristream \\
        --dataset data/external/paradigm/processed/subset_male.csv \\
        --layers embedding block_12 block_24 block_36 block_48 \\
        --output reproduce/data/audio/ \\
        --max-items 3
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse

from swp.audio.models import get_model
from swp.audio.datasets.base import ParadigmDataset
from swp.audio.pipeline.representation_extraction import run_representation_extraction


_MODEL_IMPORTS = {
    "auristream": "swp.audio.models.auristream",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract hidden-state representations for audio stimuli."
    )
    parser.add_argument("--model", required=True,
                        help="Model name (e.g. auristream)")
    parser.add_argument("--dataset", required=True,
                        help="Path to processed stimulus CSV")
    parser.add_argument("--layers", required=True, nargs="+",
                        help="Layer names to extract (e.g. embedding block_12 block_48)")
    parser.add_argument("--output", required=True,
                        help="Base output directory (run lands at output/{model}__{hash}/)")
    parser.add_argument("--model-arg", dest="model_args", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="Extra model kwargs, e.g. --model-arg wavcoch_id=TuKoResearch/WavCochV8192")
    parser.add_argument("--regenerate", action="store_true",
                        help="Overwrite an existing run")
    parser.add_argument("--max-items", dest="max_items", type=int, default=None,
                        help="Stop after N items (for smoke tests)")
    args = parser.parse_args()

    # Parse KEY=VALUE model kwargs
    model_kwargs: dict = {}
    for kv in args.model_args:
        if "=" not in kv:
            parser.error(f"--model-arg must be KEY=VALUE, got: {kv!r}")
        k, v = kv.split("=", 1)
        model_kwargs[k] = v

    # Trigger @register for the requested model
    import_path = _MODEL_IMPORTS.get(args.model)
    if import_path is not None:
        import importlib
        importlib.import_module(import_path)
    else:
        print(
            f"Warning: no known import path for model {args.model!r}. "
            "Attempting get_model() directly."
        )

    model = get_model(args.model, **model_kwargs)
    print(f"Model   : {model.name}")
    print(f"Device  : {model.device}")
    print(f"Layers  : {len(model.available_layers())} available")

    # Validate requested layers before loading the dataset
    available = set(model.available_layers())
    bad = [la for la in args.layers if la not in available]
    if bad:
        sys.exit(f"Unknown layer(s) for model {args.model!r}: {bad}\n"
                 f"Run with --model {args.model} to see available layers.")

    # Load dataset resampled to model's sample rate
    dataset_path = Path(args.dataset)
    dataset = ParadigmDataset(dataset_path, target_sr=model.sample_rate)
    n_items = len(dataset) if args.max_items is None else min(args.max_items, len(dataset))
    print(f"Dataset : {len(dataset)} items  ({n_items} will be processed)")

    # Dataset path relative to repo root (stored in manifest)
    try:
        dataset_rel = str(dataset_path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        dataset_rel = str(dataset_path)

    run_dir = run_representation_extraction(
        model=model,
        dataset=dataset,
        layers=args.layers,
        output_dir=Path(args.output),
        dataset_path=dataset_rel,
        model_kwargs=model_kwargs,
        regenerate=args.regenerate,
        max_items=args.max_items,
    )

    print(f"\nRun directory: {run_dir}")


if __name__ == "__main__":
    main()