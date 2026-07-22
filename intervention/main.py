"""CLI entry point for intervention experiments.

    python main.py --mode grid                     # run grid_config.json (CV per config)
    python main.py --mode grid --n_jobs 4          # parallelise configs across CPU processes
    python main.py --mode single --seeds 42 43 44  # one config, cross-validated

Outputs go under ``outputs/`` (results + dataset cache), which is git-ignored. Plot the
results afterwards with ``python analysis_plots.py outputs/results``.
"""
from __future__ import annotations

import os

# DAS's orthogonal rotation uses matrix_exp, unimplemented on MPS; must be set pre-torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
for path in (ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from intervention.config import ExperimentConfig
from intervention.experiments.cross_validation import run_cv
from intervention.experiments.grid import run_grid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Intervention experiment runner")
    p.add_argument("--mode", choices=["single", "grid"], default="grid")
    p.add_argument("--grid", type=Path, default=ROOT / "grid_config.json")
    p.add_argument("--out", type=Path, default=ROOT /  "results"/ "n-gram")
    p.add_argument("--cache", type=Path, default=ROOT / "cache")
    p.add_argument("--n_jobs", type=int, default=6, help="parallel CPU workers over configs")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="override CV seeds (single mode)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "grid":
        run_grid(args.grid, args.out, cache_dir=args.cache, n_jobs=args.n_jobs, verbose=args.verbose)
        return

    # single: one sensible default config, cross-validated over --seeds.
    flat = {
        "model": "das",
        "dataset": "real-real", 
        "edit_ngram": 2, 
        "state_mode": "concat",
        "seeds" : [42, 43, 44,45],
    }
    # flat = {
    #     "dataset": "real-real", "edit_ngram": 2, 
    #     "model": "das",
    #     "state_mode": "concat",
    #     # "embedding_init": "delta_state_mean", 
    #     # "train_embedding": True,
    # }
    if args.seeds:
        flat["seeds"] = args.seeds
    cfg = ExperimentConfig.from_flat(flat)
    run_cv(cfg, args.out, cache_dir=args.cache, verbose=args.verbose)


if __name__ == "__main__":
    main()
