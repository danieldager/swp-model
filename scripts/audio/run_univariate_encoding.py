"""CLI for univariate Ridge encoding of audio activations.

Usage
-----
    python scripts/audio/run_univariate_encoding.py \
        --xarray reproduce/data/audio/encodec__7f7d3b97/xarray/encoder_out.nc \
        --analysis-set all_items \
        --n-bins 10 \
        --metrics r2 \
        --output reproduce/figures/audio/encoding/encodec__7f7d3b97/encoder_out/univariate/all_items/ \
        --overwrite

Outputs (in --output directory)
--------------------------------
    scores.csv               run_id, model_name, layer, analysis_set, neuron, time_bin, relative_time, metric, score_median, score_std
    weights.csv              run_id, model_name, layer, analysis_set, neuron, time_bin, relative_time, feature, weight_mean, weight_std
    feature_importance.csv   (only if --compute-fi) neuron, time_bin, relative_time, feature, fi_mean, fi_std
    config.json              all run parameters
    qc_check.json            basic sanity checks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from swp.audio.encoding.univariate import run_univariate_encoding


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Univariate Ridge encoding of prebinned audio activations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--xarray", required=True, type=Path,
        metavar="PATH",
        help="Path to .nc xarray file (output of scripts/audio/build_xarray.py).",
    )
    p.add_argument(
        "--analysis-set", required=True, choices=["all_items", "words_only"],
        dest="analysis_set",
        help="Analysis set: 'all_items' (lexicality, length_bin, morphology) "
             "or 'words_only' (frequency_bin, length_bin, morphology).",
    )
    p.add_argument(
        "--n-bins", type=int, default=10,
        dest="n_bins",
        help="Number of relative-time bins per trial.",
    )
    p.add_argument(
        "--metrics", nargs="+", default=["r2"],
        choices=["r2", "spearman", "corr"],
        help="Scoring metrics.",
    )
    p.add_argument(
        "--compute-fi", action="store_true",
        dest="compute_fi",
        help="Compute permutation feature importance (significantly slower).",
    )
    p.add_argument(
        "--fi-n-repeats", type=int, default=10,
        dest="fi_n_repeats",
        help="Number of shuffles per feature in permutation importance.",
    )
    p.add_argument(
        "--n-jobs", type=int, default=1,
        dest="n_jobs",
        help="Number of parallel workers for the neuron loop.",
    )
    p.add_argument(
        "--n-outer-splits", type=int, default=5,
        dest="n_outer_splits",
        help="Number of outer CV folds.",
    )
    p.add_argument(
        "--n-inner-splits", type=int, default=5,
        dest="n_inner_splits",
        help="Number of inner CV folds (alpha selection).",
    )
    p.add_argument(
        "--alphas", nargs="+", type=float,
        default=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
        help="Ridge regularisation strengths to search.",
    )
    p.add_argument(
        "--random-state", type=int, default=0,
        dest="random_state",
        help="Random seed for reproducibility.",
    )
    p.add_argument(
        "--max-neurons", type=int, default=None,
        dest="max_neurons",
        metavar="N",
        help="Process only the first N neurons (useful for smoke-testing).",
    )
    p.add_argument(
        "--output", required=True, type=Path,
        metavar="DIR",
        help="Output directory (created if it does not exist).",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing outputs without error.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print(f"[run_univariate_encoding] xarray      : {args.xarray}")
    print(f"[run_univariate_encoding] analysis_set: {args.analysis_set}")
    print(f"[run_univariate_encoding] n_bins       : {args.n_bins}")
    print(f"[run_univariate_encoding] metrics      : {args.metrics}")
    print(f"[run_univariate_encoding] n_jobs       : {args.n_jobs}")
    if args.max_neurons is not None:
        print(f"[run_univariate_encoding] max_neurons  : {args.max_neurons}")
    print(f"[run_univariate_encoding] output       : {args.output}")

    out = run_univariate_encoding(
        xarray_path=args.xarray,
        analysis_set=args.analysis_set,
        n_bins=args.n_bins,
        metrics=tuple(args.metrics),
        compute_fi=args.compute_fi,
        fi_n_repeats=args.fi_n_repeats,
        n_jobs=args.n_jobs,
        n_outer_splits=args.n_outer_splits,
        n_inner_splits=args.n_inner_splits,
        alphas=args.alphas,
        random_state=args.random_state,
        max_neurons=args.max_neurons,
        output_dir=args.output,
        overwrite=args.overwrite,
    )

    qc = out["qc_check"]
    fi_line = f"  NaN in FI         : {qc['nan_in_fi']}\n" if args.compute_fi else ""
    print(
        f"\n[run_univariate_encoding] Done.\n"
        f"  trials selected   : {qc['n_trials_selected']} / {qc['n_trials_total']}\n"
        f"  neurons processed : {qc['n_neurons_processed']} / {qc['n_neurons_total']}\n"
        f"  NaN in scores     : {qc['nan_in_scores']}\n"
        f"  NaN in weights    : {qc['nan_in_weights']}\n"
        f"{fi_line}"
        f"  output            : {args.output}"
    )

    nan_detected = qc["nan_in_scores"] or qc["nan_in_weights"] or bool(qc.get("nan_in_fi"))
    if nan_detected:
        print(
            "[run_univariate_encoding] WARNING: NaN detected in outputs — "
            f"inspect {args.output / 'qc_check.json'}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()