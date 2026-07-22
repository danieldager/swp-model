"""Cross-validation: run one config across several seeds and aggregate.

``run_cv`` trains ``cfg`` once per seed in ``cfg.train.seeds`` (each into its own
``seed_<n>/`` sub-directory, weights not saved) and writes:

    cv_metrics.csv          one row per seed (test acc/loss, val acc, best epoch)
    cv_summary.json         mean/std of each metric + the flat config, for grid rollups
    params_by_seed.npz      per-seed interpretable params stacked on a leading seed axis,
                            so mean +/- CI / bar plots over positions are one-liners later

The per-seed ``params.npz`` (gamma, scales, ...) are still written by the runner.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from intervention.config import ExperimentConfig
from intervention.experiments.runner import run_experiment

METRIC_KEYS = ("final_test_acc", "final_test_loss", "val_acc")


def run_cv(
    cfg: ExperimentConfig,
    base_dir: Path,
    cache_dir: Path | None = None,
    device: torch.device | None = None,
    save_model: bool = True,
    verbose: bool = False,
) -> dict[str, object]:
    """Train ``cfg`` over all its seeds and return the aggregated summary row."""
    cfg.validate()
    run_dir = Path(base_dir) / cfg.run_name()
    per_seed = [
        run_experiment(cfg, seed, run_dir / f"seed_{seed}", cache_dir=cache_dir,
                       device=device, save_model=save_model, verbose=verbose)
        for seed in cfg.train.seeds
    ]
    return summarize_cv(cfg, run_dir, per_seed)


def summarize_cv(cfg: ExperimentConfig, run_dir: Path, per_seed: list[dict]) -> dict[str, object]:
    run_dir = Path(run_dir)
    pd.DataFrame(per_seed).to_csv(run_dir / "cv_metrics.csv", index=False)

    agg: dict[str, object] = {
        "run_name": cfg.run_name(),
        "n_seeds": len(per_seed),
        "seeds": list(cfg.train.seeds),
    }
    metrics = pd.DataFrame(per_seed)
    for key in METRIC_KEYS:
        vals = metrics[key].to_numpy(dtype=float)
        agg[f"{key}_mean"] = float(np.nanmean(vals))
        agg[f"{key}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
    agg.update(cfg.to_flat())  # method/data identity, for grouping and grid rollups

    with open(run_dir / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    _stack_params(run_dir, cfg.train.seeds)

    print(f"[CV] {cfg.run_name()}: test_acc "
          f"{agg['final_test_acc_mean']:.4f} ± {agg['final_test_acc_std']:.4f} (n={len(per_seed)})")
    return agg


def _stack_params(run_dir: Path, seeds) -> None:
    """Stack matching per-seed params along a leading seed axis for later CI plots."""
    per_seed = [
        dict(np.load(run_dir / f"seed_{s}" / "params.npz", allow_pickle=True))
        for s in seeds
        if (run_dir / f"seed_{s}" / "params.npz").exists()
    ]
    if not per_seed:
        return

    common = set.intersection(*(set(p) for p in per_seed))
    stacked: dict[str, np.ndarray] = {}
    for key in common:
        arrs = [np.asarray(p[key]) for p in per_seed]
        if arrs[0].dtype.kind in "fiu" and all(a.shape == arrs[0].shape for a in arrs):
            stacked[key] = np.stack(arrs)  # (n_seeds, *param_shape)
    if stacked:
        np.savez(run_dir / "params_by_seed.npz", seeds=np.asarray(list(seeds)), **stacked)
