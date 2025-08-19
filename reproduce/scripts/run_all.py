#!/usr/bin/env python3
"""
Master script to run all reproduction analyses.
"""

import argparse
import os
import subprocess
import sys

from utils import get_default_paths


def run_script(script_name, args_dict):
    """Run a script with given arguments."""
    cmd = [sys.executable, script_name]

    for arg, value in args_dict.items():
        if isinstance(value, bool) and value:
            cmd.append(f"--{arg}")
        elif isinstance(value, list):
            cmd.extend([f"--{arg}"] + [str(v) for v in value])
        elif value is not None:
            cmd.extend([f"--{arg}", str(value)])

    print("Running: " + "\n".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running {script_name}:")
        print(result.stderr)
        return False
    else:
        print(result.stdout)
        return True


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Run all reproduction analyses")

    # Model parameters
    parser.add_argument(
        "--model_name", default="Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1", help="Model name"
    )
    parser.add_argument(
        "--weights_path_1024", help="Path to 1024 batch model weights file"
    )
    parser.add_argument(
        "--weights_path_2048", help="Path to 2048 batch model weights file"
    )

    # I/O parameters
    parser.add_argument("--data_dir", help="Directory for cached data files")
    parser.add_argument("--figures_dir", help="Directory for output figures")

    # Analysis selection
    parser.add_argument(
        "--skip_behavioral",
        action="store_true",
        help="Skip behavioral analysis (Figure 2)",
    )
    parser.add_argument(
        "--skip_phoneme", action="store_true", help="Skip phoneme analysis (Figure 3)"
    )
    parser.add_argument(
        "--skip_ablation", action="store_true", help="Skip ablation search (Figure 4)"
    )
    parser.add_argument(
        "--skip_neuron", action="store_true", help="Skip neuron ablation (Figure 5)"
    )
    parser.add_argument(
        "--skip_univariate",
        action="store_true",
        help="Skip univariate feature importance analysis",
    )
    parser.add_argument(
        "--skip_error_analysis",
        action="store_true",
        help="Skip error analysis (Figure 15)",
    )

    # Control parameters
    parser.add_argument(
        "--regenerate", action="store_true", help="Regenerate all cached results"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Get default paths
    paths = get_default_paths()

    weights_1024 = args.weights_path_1024 or paths["weights"]["1024"]
    weights_2048 = args.weights_path_2048 or paths["weights"]["2048"]
    data_dir = args.data_dir or paths["data"]
    figures_dir = args.figures_dir or paths["figures"]

    # Common arguments
    common_args = {
        "model_name": args.model_name,
        "data_dir": data_dir,
        "figures_dir": figures_dir,
        "regenerate": args.regenerate,
        "seed": args.seed,
    }

    # Script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    success_count = 0
    total_count = 0

    # Run behavioral analysis (Figure 2)
    if not args.skip_behavioral:
        total_count += 1
        print("\n" + "=" * 50)
        print("RUNNING BEHAVIORAL ANALYSIS (FIGURE 2)")
        print("=" * 50)

        behavioral_args = {
            **common_args,
            "weights_path": weights_2048,
            "batch_size": 2048,
        }

        if run_script(
            os.path.join(script_dir, "behavioral_analysis.py"), behavioral_args
        ):
            success_count += 1

    # Run phoneme analysis (Figure 3)
    if not args.skip_phoneme:
        total_count += 1
        print("\\n" + "=" * 50)
        print("RUNNING PHONEME ANALYSIS (FIGURE 3)")
        print("=" * 50)

        phoneme_args = {
            **common_args,
            "weights_path": weights_1024,
            "batch_size": 1024,
        }

        if run_script(os.path.join(script_dir, "phoneme_analysis.py"), phoneme_args):
            success_count += 1

    # Run ablation search (Figure 4)
    if not args.skip_ablation:
        total_count += 1
        print("\\n" + "=" * 50)
        print("RUNNING ablation search (FIGURE 4)")
        print("=" * 50)

        ablation_args = {
            **common_args,
            "weights_path": weights_1024,
            "batch_size": 1024,
            "quiet": True,
        }

        if run_script(os.path.join(script_dir, "ablation_search.py"), ablation_args):
            success_count += 1

    # Run neuron ablation (Figure 5)
    if not args.skip_neuron:
        total_count += 1
        print("\\n" + "=" * 50)
        print("RUNNING NEURON ABLATION (FIGURE 5)")
        print("=" * 50)

        neuron_args = {
            **common_args,
            "weights_path": weights_2048,
            "batch_size": 2048,
            "neuron_ids": [49, 31],
        }

        if run_script(os.path.join(script_dir, "ablation_behavioral.py"), neuron_args):
            success_count += 1

    # Run feature importance analysis
    if not args.skip_univariate:
        total_count += 1
        print("\\n" + "=" * 50)
        print("RUNNING FEATURE IMPORTANCE ANALYSIS")
        print("=" * 50)

        feature_args = {
            **common_args,
            "weights_path": weights_1024,
            "batch_size": 1024,
            "hidden_size": 128,
        }

        if run_script(os.path.join(script_dir, "univariate_analysis.py"), feature_args):
            success_count += 1

    # # Run error analysis (Figure 15)
    # if not args.skip_error_analysis:
    #     total_count += 1
    #     print("\\n" + "=" * 50)
    #     print("RUNNING ERROR ANALYSIS (FIGURE 15)")
    #     print("=" * 50)

    #     error_args = {
    #         **common_args,
    #         "weights_path": weights_2048,
    #         "batch_size": 2048,
    #     }

    #     if run_script(os.path.join(script_dir, "error_analysis.py"), error_args):
    #         success_count += 1

    # Summary
    print("\\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Successfully completed: {success_count}/{total_count} analyses")

    if success_count == total_count:
        print("All analyses completed successfully!")
        return 0
    else:
        print(f"Some analyses failed. Check output above for details.")
        return 1


if __name__ == "__main__":
    exit(main())
