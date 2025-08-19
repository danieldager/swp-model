#!/usr/bin/env python3
"""
Generate specific neuron ablation search plots (Figure 5).
Analyzes effects of ablating specific neurons on behavioral measures.
"""

import argparse
import os
import pathlib
from typing import List

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
from swp.test.ablations import (
    ablate_lstm_neuron,
    cache_lstm_weights,
    restore_lstm_weights,
)
from swp.test.repetition import test
from swp.utils.datasets import enrich_for_plotting
from swp.viz.test.length import plot_length_errors
from swp.viz.test.position import plot_position_errors_smooth
from swp.viz.test.regressions import regression_plots
from swp.viz.test.sonority import plot_sonority_errors


def generate_ablation_behavioral_analysis(
    model_name: str,
    weights_path: str,
    batch_size: int,
    neuron_ids: List[int],
    data_dir: str,
    figures_dir: str,
    figsize: tuple = (7, 6),
    regenerate: bool = False,
    seed: int = 42,
):
    """Generate specific neuron ablation search plots."""
    # Setup
    device = setup_environment(seed)
    model = load_model(model_name, weights_path, device)

    # Get default paths
    paths = get_default_paths()

    # Load datasets
    wfe_df = pd.read_csv(paths["datasets"]["wfe"], converters=CONVERTERS, index_col=0)
    wfe_loader = get_phoneme_testloader(batch_size=batch_size, dataset_df=wfe_df)

    ssp_df = pd.read_csv(paths["datasets"]["ssp"], converters=CONVERTERS, index_col=0)
    ssp_loader = get_phoneme_testloader(batch_size=batch_size, dataset_df=ssp_df)

    # Cache original weights
    weights = cache_lstm_weights(model.encoder.recurrent)

    for neuron_id in neuron_ids:
        print(f"Processing neuron {neuron_id}...")

        # Check for cached results
        wfe_data_path = os.path.join(data_dir, f"wfe_{neuron_id}_results.csv")
        ssp_data_path = os.path.join(data_dir, f"ssp_{neuron_id}_results.csv")

        if (
            os.path.exists(wfe_data_path)
            and os.path.exists(ssp_data_path)
            and not regenerate
        ):
            print(f"Loading cached results for neuron {neuron_id}")
            wfe_results = pd.read_csv(wfe_data_path, converters=CONVERTERS, index_col=0)
            ssp_results = pd.read_csv(ssp_data_path, converters=CONVERTERS, index_col=0)
        else:
            print(f"Generating new results for neuron {neuron_id}")

            # Ablate the specific neuron
            ablate_lstm_neuron(model.encoder.recurrent, neuron_id, 128)

            # Test on WFE dataset
            wfe_results, _ = test(
                model=model,
                device=device,
                test_df=wfe_df,
                test_loader=wfe_loader,
            )
            wfe_results = enrich_for_plotting(wfe_results)

            # Test on SSP dataset
            ssp_results, _ = test(
                model=model,
                device=device,
                test_df=ssp_df,
                test_loader=ssp_loader,
            )
            ssp_results = enrich_for_plotting(ssp_results)

            # Save results
            os.makedirs(data_dir, exist_ok=True)
            wfe_results.to_csv(wfe_data_path)
            ssp_results.to_csv(ssp_data_path)

            # Restore weights for next neuron
            restore_lstm_weights(model.encoder.recurrent, weights)

        # Create output directory for this neuron
        neuron_output_path = ensure_output_dir(
            os.path.join(figures_dir, "ablations", str(neuron_id))
        )

        # Generate plots
        print(f"Generating plots for neuron {neuron_id}...")
        plot_length_errors(wfe_results, dir=neuron_output_path, figsize=figsize)
        regression_plots(wfe_results, path=neuron_output_path, figsize=figsize)
        plot_position_errors_smooth(
            wfe_results, dir=neuron_output_path, figsize=figsize
        )
        plot_sonority_errors(ssp_results, path=neuron_output_path, figsize=figsize)

        print(f"Plots for neuron {neuron_id} saved to: {neuron_output_path}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Generate specific neuron ablation search plots (Figure 5)"
    )

    # Model parameters
    parser.add_argument(
        "--model_name", default="Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1", help="Model name"
    )
    parser.add_argument("--weights_path", help="Path to model weights file")
    parser.add_argument(
        "--batch_size", type=int, default=2048, help="Batch size for evaluation"
    )

    # Neuron parameters
    parser.add_argument(
        "--neuron_ids",
        type=int,
        nargs="+",
        default=[49, 31],
        help="List of neuron IDs to ablate",
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
    generate_ablation_behavioral_analysis(
        model_name=args.model_name,
        weights_path=weights_path,
        batch_size=args.batch_size,
        neuron_ids=args.neuron_ids,
        data_dir=data_dir,
        figures_dir=figures_dir,
        figsize=figsize,
        regenerate=args.regenerate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
