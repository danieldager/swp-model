#!/usr/bin/env python3
"""
Generate error type analysis across all neurons (Figure 15).
Analyzes different types of errors produced by each neuron ablation.
"""

import argparse
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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


def is_early_eos(pred, gt):
    """Check if prediction ends early with EOS token."""
    for i in range(len(pred)):
        if pred[i] == "<EOS>":
            return True
        elif pred[i] != gt[i]:
            return False
    return False


def is_position_error(pred, gt):
    """Check if error is due to position/permutation (same phonemes, wrong order)."""
    gt_count = Counter(pred)
    pred_count = Counter(gt)
    delta = len(list((gt_count - pred_count).keys())) + len(
        list((pred_count - gt_count).keys())
    )
    return delta == 0


def is_identity_error(pred, gt):
    """Check if error involves identity substitution (wrong phonemes)."""
    for i in range(len(pred)):
        if pred[i] in {"<EOS>", "<SOS>", "<PAD>"}:
            return False
    return True


def generate_error_analysis(
    model_name: str,
    weights_path: str,
    batch_size: int,
    data_dir: str,
    figures_dir: str,
    min_errors: int = 50,
    regenerate: bool = False,
    seed: int = 42,
):
    """Generate error type analysis for all neurons."""
    # Setup
    device = setup_environment(seed)
    model = load_model(model_name, weights_path, device)

    # Get default paths
    paths = get_default_paths()

    # Check for cached results
    results_path = os.path.join(data_dir, "error_analysis.csv")

    if os.path.exists(results_path) and not regenerate:
        print(f"Loading cached error analysis from {results_path}")
        error_df = pd.read_csv(results_path)
    else:
        print("Generating new error analysis...")

        # Load WFE dataset
        wfe_df = pd.read_csv(
            paths["datasets"]["wfe"], converters=CONVERTERS, index_col=0
        )
        wfe_loader = get_phoneme_testloader(batch_size=batch_size, dataset_df=wfe_df)

        neurons = range(128)
        results = []

        # Cache original weights
        weights = cache_lstm_weights(model.encoder.recurrent)

        # Run ablation for each neuron
        for neuron in neurons:
            print(f"Processing neuron {neuron}...")

            ablate_lstm_neuron(model.encoder.recurrent, neuron, 128)
            wfe_results, _ = test(
                model=model,
                device=device,
                test_df=wfe_df,
                test_loader=wfe_loader,
            )
            wfe_results = enrich_for_plotting(wfe_results)
            results.append(wfe_results)

            # Restore weights for next iteration
            restore_lstm_weights(model.encoder.recurrent, weights)

        # Analyze errors
        df_dict = {
            "Unit Number": [],
            "Error Type": [],
            "Rate": [],
            "Number of Errors": [],
            "Unit Name": [],
        }
        errors_dict = {}

        for neuron, df in zip(neurons, results):
            df["Error_Rate"] = (df["Edit_Distance"] > 0).astype(int)
            errors = df["Error_Rate"].sum()
            errors_dict[neuron] = errors

            if errors >= min_errors:
                eos = 0
                permute = 0
                identity = 0
                other = 0

                for index, row in df.iterrows():
                    pred = row["Prediction"]
                    gt = row["No_Stress"]
                    is_other = True

                    if row["Error_Rate"] == 1:
                        if is_early_eos(pred, gt):
                            eos += 1
                            is_other = False
                        if is_position_error(pred, gt):
                            permute += 1
                            is_other = False
                        elif is_identity_error(pred, gt):
                            identity += 1
                            is_other = False
                        if is_other:
                            other += 1

                for errtype, errcount in [
                    ("Length", eos),
                    ("Position", permute),
                    ("Identity", identity),
                    ("Other", other),
                ]:
                    df_dict["Unit Number"].append(str(neuron))
                    df_dict["Error Type"].append(errtype)
                    df_dict["Number of Errors"].append(errcount)
                    df_dict["Rate"].append(round(errcount / errors * 100))
                    df_dict["Unit Name"].append(neuron)

        error_df = pd.DataFrame(df_dict)
        # Sort by total errors (descending)
        error_df = error_df.iloc[
            sorted(
                error_df.index,
                key=lambda k: errors_dict[error_df.loc[k, "Unit Name"]],
                reverse=True,
            )
        ]

        # Save results
        os.makedirs(data_dir, exist_ok=True)
        error_df.to_csv(results_path, index=False)

    # Create output directory
    output_path = ensure_output_dir(figures_dir)

    # Generate plot
    print("Generating error type analysis plot...")
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        error_df, y="Unit Number", x="Number of Errors", hue="Error Type", orient="h"
    )
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.title("Error Types by Neuron Ablation")
    plt.xlabel("Number of Errors")
    plt.ylabel("Neuron ID")
    plt.tight_layout()

    plot_path = os.path.join(output_path, "error_type_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {plot_path}")

    return error_df


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Generate error type analysis across all neurons (Figure 15)"
    )

    # Model parameters
    parser.add_argument(
        "--model_name", default="Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1", help="Model name"
    )
    parser.add_argument("--weights_path", help="Path to model weights file")
    parser.add_argument(
        "--batch_size", type=int, default=2048, help="Batch size for evaluation"
    )

    # Analysis parameters
    parser.add_argument(
        "--min_errors",
        type=int,
        default=50,
        help="Minimum number of errors to include a neuron in analysis",
    )

    # I/O parameters
    parser.add_argument("--data_dir", help="Directory for cached data files")
    parser.add_argument("--figures_dir", help="Directory for output figures")

    # Control parameters
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

    # Generate analysis
    generate_error_analysis(
        model_name=args.model_name,
        weights_path=weights_path,
        batch_size=args.batch_size,
        data_dir=data_dir,
        figures_dir=figures_dir,
        min_errors=args.min_errors,
        regenerate=args.regenerate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
