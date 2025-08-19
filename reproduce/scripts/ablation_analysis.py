#!/usr/bin/env python3
"""
Generate ablation analysis plots (Figure 4).
Includes lexical vs sublexical accuracy comparisons.
"""

import argparse
import os

import pandas as pd
from utils import (
    CONVERTERS,
    ensure_output_dir,
    get_default_paths,
    load_model,
    parse_figsize,
    setup_environment,
)

from swp.datasets.phonemes import get_phoneme_testloader
from swp.test.ablations import ablate
from swp.viz.ablation import ablation_plots


def generate_ablation_analysis(
    model_name: str,
    weights_path: str,
    batch_size: int,
    data_dir: str,
    figures_dir: str,
    include_stress: bool = False,
    regenerate: bool = False,
    seed: int = 42,
    quiet: bool = False,
):
    """Generate ablation analysis plots."""
    # Setup
    device = setup_environment(seed)

    # Get default paths
    paths = get_default_paths()

    # Check for cached ablation results
    ablated_data_path = os.path.join(data_dir, "wfe_ablated.csv")

    if os.path.exists(ablated_data_path) and not regenerate:
        print(f"Loading cached ablation results from {ablated_data_path}")
        ablated_results = pd.read_csv(ablated_data_path, converters=CONVERTERS)
    else:
        print("Generating new ablation results...")

        # Load model
        model = load_model(model_name, weights_path, device)

        # Load WFE dataset
        wfe_df = pd.read_csv(
            paths["datasets"]["wfe"], converters=CONVERTERS, index_col=0
        )
        wfe_loader = get_phoneme_testloader(batch_size=batch_size, dataset_df=wfe_df)

        # Run ablation analysis
        _, ablated_results = ablate(
            model=model,
            device=device,
            test_df=wfe_df,
            test_loader=wfe_loader,
            include_stress=include_stress,
            print_progress=not quiet,
        )

        # Save results
        os.makedirs(os.path.dirname(ablated_data_path), exist_ok=True)
        ablated_results.to_csv(ablated_data_path, index=False)

    # Create output directory
    output_path = ensure_output_dir(os.path.join(figures_dir, "ablations"))

    # Generate ablation plots
    print("Generating ablation plots...          ")
    ablation_plots(
        ablated_results,
        "real_accuracy",
        "pseudo_accuracy",
        "Real (Lexical)",
        "Pseudo (Sublexical)",
        "lex",
        model_dir=output_path,
    )

    print(f"Plots saved to: {output_path}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Generate ablation analysis plots (Figure 4)"
    )

    # Model parameters
    parser.add_argument(
        "--model_name", default="Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1", help="Model name"
    )
    parser.add_argument("--weights_path", help="Path to model weights file")
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="Batch size for evaluation"
    )

    # I/O parameters
    parser.add_argument("--data_dir", help="Directory for cached data files")
    parser.add_argument("--figures_dir", help="Directory for output figures")

    # Analysis parameters
    parser.add_argument(
        "--include_stress",
        action="store_true",
        help="Include stress in ablation analysis",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate ablation results even if cached data exists",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-neuron progress output",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Get default paths if not provided
    paths = get_default_paths()

    weights_path = args.weights_path or paths["weights"]["1024"]
    data_dir = args.data_dir or paths["data"]
    figures_dir = args.figures_dir or paths["figures"]

    # Generate plots
    generate_ablation_analysis(
        model_name=args.model_name,
        weights_path=weights_path,
        batch_size=args.batch_size,
        data_dir=data_dir,
        figures_dir=figures_dir,
        include_stress=args.include_stress,
        regenerate=args.regenerate,
        quiet=args.quiet,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
