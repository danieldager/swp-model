from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from swp.audio.metrics.signal import compute_all


def _paramhash(params: dict) -> str:
    """SHA-256 (truncated to 8 hex chars) of sorted JSON-serialized params."""
    serialized = json.dumps(params, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:8]


def run_extraction(
    model,
    dataset,
    layers: list[str],
    output_dir: Path | str,
    dataset_path: str,
    model_kwargs: dict,
    regenerate: bool = False,
) -> Path:
    """Run the extraction pipeline for a dataset.

    For each item in ``dataset``:
      1. Runs ``model.reconstruct()`` and computes signal metrics.
      2. Runs ``model.extract_activations()`` and saves one ``.pt`` file per layer.

    Outputs under ``output_dir / run_id /``:
      - ``manifest.json``   — run parameters + item → layer → relative path map
      - ``metrics.csv``     — one row per item: item_id, mel_distance, si_sdr
      - ``activations/``    — ``{item_id}__{layer_name}.pt`` files

    Args:
        model:        AudioModel instance (must implement reconstruct + extract_activations)
        dataset:      ParadigmDataset instance
        layers:       stable layer names to extract
        output_dir:   base output directory (e.g. ``reproduce/data/audio/``)
        dataset_path: repo-relative path to the processed CSV (for paramhash)
        model_kwargs: model constructor kwargs (for paramhash; e.g. ``{"bandwidth": 6.0}``)
        regenerate:   if True, overwrite an existing run; otherwise skip

    Returns:
        Path to the run output directory.
    """
    layers_sorted = sorted(layers)

    params = {
        "model_name": model.name,
        "model_kwargs": model_kwargs,
        "dataset_path": dataset_path,
        "layers": layers_sorted,
        "sample_rate": model.sample_rate,
    }
    phash = _paramhash(params)
    run_id = f"{model.name}__{phash}"
    run_dir = Path(output_dir) / run_id

    if run_dir.exists() and not regenerate:
        print(
            f"[extract] Cache hit: {run_dir} already exists. "
            "Pass --regenerate to overwrite."
        )
        return run_dir

    act_dir = run_dir / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict] = []
    manifest_items: dict[str, dict[str, str]] = {}

    n = len(dataset)
    for idx in range(n):
        waveform, metadata = dataset[idx]
        item_id = str(metadata["item_id"])

        if (idx + 1) % 10 == 0 or idx == 0 or idx == n - 1:
            print(f"[extract] {idx + 1}/{n}  item={item_id}")

        # Signal metrics
        result = model.reconstruct(waveform)
        metrics = compute_all(result["original"], result["reconstructed"], model.sample_rate)
        metrics_rows.append({"item_id": item_id, **metrics})

        # Activations
        activations = model.extract_activations(waveform, layers_sorted)
        layer_paths: dict[str, str] = {}
        for layer_name, tensor in activations.items():
            pt_filename = f"{item_id}__{layer_name}.pt"
            pt_path = act_dir / pt_filename
            torch.save(tensor, pt_path)
            # Store path relative to run_dir for portability
            layer_paths[layer_name] = str(pt_path.relative_to(run_dir))
        manifest_items[item_id] = layer_paths

    # metrics.csv
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(run_dir / "metrics.csv", index=False)

    # manifest.json
    manifest = {
        "run_id": run_id,
        "run_params": params,
        "items": manifest_items,
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[extract] Done. {n} items → {run_dir}")
    return run_dir