from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import transformers


def _paramhash(params: dict) -> str:
    """8-char SHA-256 of a JSON-serialised dict. Matches extraction.py."""
    serialized = json.dumps(params, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:8]


def run_representation_extraction(
    model: Any,
    dataset: Any,
    layers: list[str],
    output_dir: Path,
    dataset_path: str,
    model_kwargs: dict,
    regenerate: bool = False,
    max_items: int | None = None,
) -> Path:
    """Extract and save temporal hidden-state activations for a representation model.

    For models that expose output_hidden_states but do not support reconstruct().
    Analogous to run_extraction() in extraction.py but without signal metrics.

    Output layout (parallel to codec extraction runs)::

        {output_dir}/{model}__{hash}/
            manifest.json
            extraction_summary.csv
            activations/{item_id}__{layer}.pt   shape [D, L] float32

    Args:
        model:         model with extract_activations(), available_layers(),
                       name, sample_rate, hop_length attributes
        dataset:       ParadigmDataset iterable; items already at model.sample_rate
        layers:        layer names to extract (subset of model.available_layers())
        output_dir:    base directory; run lands at output_dir / run_id
        dataset_path:  dataset CSV path relative to repo root (stored in manifest)
        model_kwargs:  model constructor kwargs (stored in manifest + used for hash)
        regenerate:    if True, overwrite an existing run
        max_items:     stop after this many items (for smoke tests)

    Returns:
        run_dir Path
    """
    layers_sorted = sorted(layers)
    params = {
        "model_name": model.name,
        "model_kwargs": model_kwargs,
        "dataset_path": dataset_path,
        "layers": layers_sorted,
        "sample_rate": model.sample_rate,
        "max_items": max_items,  # None = full run; int = partial/smoke-test run
    }
    run_id = f"{model.name}__{_paramhash(params)}"
    run_dir = output_dir / run_id

    if run_dir.exists() and not regenerate:
        print(f"Run {run_id} already exists. Pass --regenerate to overwrite.")
        return run_dir

    run_dir.mkdir(parents=True, exist_ok=True)
    activations_dir = run_dir / "activations"
    activations_dir.mkdir(exist_ok=True)

    hop: int = getattr(model, "hop_length", 80)
    token_rate: int = model.sample_rate // hop

    summary_rows: list[dict] = []
    manifest_items: dict[str, dict[str, str]] = {}

    for item_idx, (waveform, meta) in enumerate(dataset):
        if max_items is not None and item_idx >= max_items:
            break

        item_id: str = str(meta["item_id"])
        duration_s: float = float(meta["duration_s"])
        n_samples_16k: int = int(waveform.shape[-1])
        trailing_samples: int = n_samples_16k % hop

        activations = model.extract_activations(waveform, layers_sorted)

        layer_paths: dict[str, str] = {}
        for layer_name in layers_sorted:
            fname = f"{item_id}__{layer_name}.pt"
            fpath = activations_dir / fname
            torch.save(activations[layer_name], fpath)
            layer_paths[layer_name] = str(fpath.relative_to(run_dir))
        manifest_items[item_id] = layer_paths

        n_tokens: int = activations[layers_sorted[0]].shape[1]  # [D, L] → L
        D: int = activations[layers_sorted[0]].shape[0]

        row: dict = {
            "item_id": item_id,
            "duration_s": duration_s,
            "n_samples_16k": n_samples_16k,
            "n_tokens": n_tokens,
            "represented_duration_s": round(n_tokens / token_rate, 6),
            "trailing_samples": trailing_samples,
            "trailing_duration_s": round(trailing_samples / model.sample_rate, 6),
        }
        if "original_sample_rate" in meta:
            row["original_sample_rate"] = int(meta["original_sample_rate"])
        summary_rows.append(row)

        print(
            f"  [{item_idx + 1:>4}] {item_id:<35}  "
            f"n_tokens={n_tokens:4d}  shape=[{D}, {n_tokens}]"
        )

    # Write extraction_summary.csv with consistent column order
    col_order = [
        "item_id", "duration_s", "original_sample_rate",
        "n_samples_16k", "n_tokens", "represented_duration_s",
        "trailing_samples", "trailing_duration_s",
    ]
    df = pd.DataFrame(summary_rows)
    ordered_cols = [c for c in col_order if c in df.columns]
    df[ordered_cols].to_csv(run_dir / "extraction_summary.csv", index=False)

    # Write manifest.json
    run_params: dict = dict(params)
    # Model-specific identifiers
    for attr in ("wavcoch_id", "auristream_id"):
        if hasattr(model, attr):
            run_params[attr] = getattr(model, attr)
    # Temporal resolution metadata
    run_params.update({
        "token_rate_hz": token_rate,
        "hop_length": hop,
        "token_count_rule": "floor(n_samples / hop_length)",
        "token_center_rule": "(i + 0.5) / token_rate_hz",
        "transformers_version": transformers.__version__,
        "torch_dtype": "float32",
        "device": str(model.device),
    })
    manifest = {
        "run_id": run_id,
        "run_params": run_params,
        "items": manifest_items,
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    n_done = len(manifest_items)
    print(f"\nExtraction complete: {run_dir}")
    print(f"  Items: {n_done}  Layers: {layers_sorted}")
    return run_dir