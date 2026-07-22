"""Grid search over configs, each evaluated by cross-validation.

A flat ``{key: [values]}`` grid expands (Cartesian product) into one ``ExperimentConfig``
per combination; every config is run across its seeds by :func:`intervention.experiments.cross_validation.run_cv`.
Configs are independent, so they parallelise across CPU processes (``n_jobs > 1``) — the
right axis of parallelism for a model this small. Results are rolled up into
``grid_summary.csv`` (one row per config, with ``mean ± std`` metrics).
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch

from intervention.config import ExperimentConfig
from intervention.experiments.cross_validation import run_cv

SUMMARY_FILENAME = "grid_summary.csv"


def grid_iter(grid: dict[str, Iterable]) -> Iterable[dict]:
    """Yield one flat config dict per combination in the Cartesian product."""
    keys = list(grid)
    for values in product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def load_grid_from_json(json_path: Path) -> dict[str, list]:
    grid = json.loads(Path(json_path).read_text())
    if not isinstance(grid, dict):
        raise ValueError("Grid JSON must be an object mapping parameter names to lists.")
    return grid


def _run_one(
    cfg: ExperimentConfig, base_dir: Path, cache_dir: Path | None,
    device: torch.device | None, skip_existing: bool, save_model: bool, verbose: bool,
) -> dict | None:
    run_dir = Path(base_dir) / cfg.run_name()
    summary_path = run_dir / "cv_summary.json"
    if skip_existing and summary_path.exists():
        print(f"Skipping existing run: {cfg.run_name()}")
        return json.loads(summary_path.read_text())
    torch.set_num_threads(1)  # avoid thread oversubscription across parallel workers
    return run_cv(cfg, base_dir, cache_dir=cache_dir, device=device,
                  save_model=save_model, verbose=verbose)


def run_grid(
    grid: dict[str, list] | Path,
    base_dir: Path,
    cache_dir: Path | None = None,
    n_jobs: int = 1,
    device: torch.device | None = None,
    skip_existing: bool = True,
    save_model: bool = False,
    verbose: bool = False,
) -> list[dict]:
    if isinstance(grid, (str, Path)):
        grid = load_grid_from_json(grid)
    grid = dict(grid)
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # `seeds` is the CV dimension, applied to every config -- not a swept axis.
    seeds = grid.pop("seeds", None)
    rows = list(grid_iter(grid))
    if seeds is not None:
        for row in rows:
            row["seeds"] = seeds
    configs = [ExperimentConfig.from_flat(row) for row in rows]
    configs = [c for c in configs if not c.method.is_trivial()]  # random frozen embedding => nothing to learn
    print(f"Grid: {len(configs)} configs x seeds, n_jobs={n_jobs}")

    if n_jobs == 1:
        summaries = [_run_one(c, base_dir, cache_dir, device, skip_existing, save_model, verbose) for c in configs]
    else:
        # Build each unique dataset ONCE up front. Without this, every worker rebuilds the
        # full dataset simultaneously (4x peak memory -> the "parallel doesn't work" hang).
        if cache_dir is not None:
            _prewarm_cache(configs, cache_dir, skip_existing)
        from joblib import Parallel, delayed
        # Workers run on CPU (MPS/CUDA don't share cleanly across processes).
        summaries = Parallel(n_jobs=n_jobs)(
            delayed(_run_one)(c, base_dir, cache_dir, torch.device("cpu"), skip_existing, save_model, verbose)
            for c in configs
        )

    summaries = [r for r in summaries if r is not None]
    save_grid_summary(summaries, base_dir)
    return summaries


def _prewarm_cache(configs: list[ExperimentConfig], cache_dir: Path, skip_existing: bool) -> None:
    """Populate the dataset cache for every unique (data, seed) once, sequentially, so
    parallel workers only train (and never rebuild the same dataset concurrently).
    ``build_loaders`` already no-ops when a split's cache file exists."""
    import torch as _torch
    from swp.datasets.phonemes import get_phoneme_to_id

    from intervention.data import build_loaders
    from intervention.experiments.runner import _load_repeat_model

    phoneme_to_id = get_phoneme_to_id()
    device = _torch.device("cpu")
    models: dict[tuple[str, str], object] = {}
    seen: set[tuple] = set()
    for cfg in configs:
        model_key = (cfg.train.model_name, cfg.train.weights_path)
        for seed in cfg.train.seeds:
            key = (json.dumps(cfg.data.cache_fields(), sort_keys=True), seed, *model_key)
            if key in seen:
                continue
            seen.add(key)
            if model_key not in models:
                models[model_key] = _load_repeat_model(cfg.train, device)
            build_loaders(cfg.data, seed, phoneme_to_id, models[model_key], device,
                          batch_size=cfg.train.batch_size, cache_dir=cache_dir)
    print(f"Pre-warmed dataset cache for {len(seen)} unique (data, seed) combos")


def save_grid_summary(rows: list[dict], base_dir: Path) -> None:
    """One row per config; de-duplicate on run_name, keeping the latest."""
    if not rows:
        return
    base_dir = Path(base_dir)
    path = base_dir / SUMMARY_FILENAME
    new_df = pd.DataFrame(rows)
    if path.exists():
        combined = pd.concat([pd.read_csv(path), new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["run_name"], keep="last")
    else:
        combined = new_df
    combined.to_csv(path, index=False)
