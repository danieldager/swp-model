#!/usr/bin/env python3
"""
Generate behavioral analysis plots (Figure 2A-D).
Includes length errors, regressions, position errors, and sonority plots.
"""

import argparse
import os

from utils import (
    ensure_output_dir,
    get_default_paths,
    load_model,
    load_or_generate_dataset_results,
    parse_figsize,
    setup_environment,
)

from swp.viz.test.length import plot_length_errors
from swp.viz.test.position import plot_position_errors_smooth
from swp.viz.test.regressions import regression_plots
from swp.viz.test.sonority import plot_sonority_errors


def generate_behavioral_plots(
    model_name: str,
    weights_path: str,
    batch_size: int,
    data_dir: str,
    figures_dir: str,
    figsize: tuple = (7, 6),
    regenerate: bool = False,
    seed: int = 42,
):
    """Generate all behavioral analysis plots."""
    # Setup
    device = setup_environment(seed)
    model = load_model(model_name, weights_path, device)

    # Get default paths
    paths = get_default_paths()

    # Load/generate WFE results
    wfe_data_path = os.path.join(data_dir, "wfe_enriched.csv")
    wfe_results = load_or_generate_dataset_results(
        data_path=wfe_data_path,
        dataset_path=paths["datasets"]["wfe"],
        model=model,
        device=device,
        batch_size=batch_size,
        enrich=True,
        regenerate=regenerate,
    )

    # Load/generate SSP results
    ssp_data_path = os.path.join(data_dir, "ssp_enriched.csv")
    ssp_results = load_or_generate_dataset_results(
        data_path=ssp_data_path,
        dataset_path=paths["datasets"]["ssp"],
        model=model,
        device=device,
        batch_size=batch_size,
        enrich=True,
        regenerate=regenerate,
    )

    # Create output directory
    output_path = ensure_output_dir(os.path.join(figures_dir, "behavioral"))

    # Generate plots
    print("Generating behavioral analysis plots...")
    plot_length_errors(wfe_results, dir=output_path, figsize=figsize)
    regression_plots(wfe_results, path=output_path, figsize=figsize)
    plot_position_errors_smooth(wfe_results, dir=output_path, figsize=figsize)
    plot_sonority_errors(ssp_results, path=output_path, figsize=figsize)

    print(f"Plots saved to: {output_path}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Generate behavioral analysis plots (Figure 2A-D)"
    )

    # Model parameters
    parser.add_argument(
        "--model_name", default="Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1", help="Model name"
    )
    parser.add_argument("--weights_path", help="Path to model weights file")
    parser.add_argument(
        "--batch_size", type=int, default=2048, help="Batch size for evaluation"
    )

    # I/O parameters
    parser.add_argument("--data_dir", help="Directory for cached data files")
    parser.add_argument("--figures_dir", help="Directory for output figures")

    # Plot parameters
    parser.add_argument(
        "--figsize", default="7,6", help='Figure size as "width,height"'
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate results even if cached data exists",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Get default paths if not provided
    paths = get_default_paths()

    weights_path = args.weights_path or paths["weights"]["2048"]
    data_dir = args.data_dir or paths["data"]
    figures_dir = args.figures_dir or paths["figures"]

    # Parse figsize
    figsize = parse_figsize(args.figsize)

    # Generate plots
    generate_behavioral_plots(
        model_name=args.model_name,
        weights_path=weights_path,
        batch_size=args.batch_size,
        data_dir=data_dir,
        figures_dir=figures_dir,
        figsize=figsize,
    regenerate=args.regenerate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
