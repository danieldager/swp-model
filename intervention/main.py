"""CLI entry point for intervention experiments.

    python -m intervention.main --mode grid                     # run grid_config.json (CV per config)
    python -m intervention.main --mode grid --n_jobs 4          # parallelise configs across CPU processes
    python -m intervention.main --mode single --seeds 42 43 44  # one config, cross-validated

Outputs go under ``outputs/`` (results + dataset cache), which is git-ignored. Plot the
results afterwards with ``python analysis_plots.py outputs/results``.
"""
from __future__ import annotations

import os

# DAS's orthogonal rotation uses matrix_exp, unimplemented on MPS; must be set pre-torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

from intervention.config import ExperimentConfig
from intervention.experiments.cross_validation import run_cv
from intervention.experiments.grid import run_grid
from intervention.experiments.transfer import find_runs, run_transfer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Intervention experiment runner")
    p.add_argument("--mode", choices=["single", "grid", "transfer"], default="grid")
    p.add_argument("--grid", type=Path, default=ROOT / "grid_config.json")
    p.add_argument("--out", type=Path, default=ROOT /  "results"/ "filters")
    p.add_argument("--cache", type=Path, default=ROOT / "cache")
    p.add_argument("--n_jobs", type=int, default=6, help="parallel CPU workers over configs")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="override CV seeds (single mode)")
    p.add_argument("--save_model", action="store_true", help="keep weights (needed for --mode transfer)")
    p.add_argument("--eval_on", type=str, nargs="+", default=None,
                   help="transfer mode: test sets as flat JSON, e.g. '{\"dataset\":\"real-real\"}'")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "grid":
        run_grid(args.grid, args.out, cache_dir=args.cache, n_jobs=args.n_jobs,
                 save_model=args.save_model, verbose=args.verbose)
        return

    if args.mode == "transfer":
        runs = find_runs(args.out)
        if not runs:
            raise SystemExit(f"No runs with saved weights under {args.out}. "
                             "Re-run training with --save_model.")
        evals = [json.loads(s) for s in (args.eval_on or [
            '{"dataset":"real-real"}',
            '{"dataset":"source-modified"}',
            '{"dataset":"modified-source"}',
        ])]
        print(f"Transfer: {len(runs)} runs x {len(evals)} test sets")
        run_transfer(runs, evals, out_root=args.out / "transfer",
                     seeds=args.seeds, cache_dir=args.cache)
        return

    # single: one sensible default config, cross-validated over --seeds.
    flat = {
        "model": "das_auto",
        "dataset": "real-real",
        "edit_ngram": 1,
        "var_ngram": 2,
        "state_mode": "concat",
        "seeds": [42, 43, 44, 45],
    }
    if args.seeds:
        flat["seeds"] = args.seeds
    cfg = ExperimentConfig.from_flat(flat)
    run_cv(cfg, args.out, cache_dir=args.cache, verbose=args.verbose)


if __name__ == "__main__":
    main()
