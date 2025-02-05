import argparse
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The PyTorch API of nested tensors is in prototype stage",
)

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from swp.test.repetition import test
from swp.utils.paths import get_ablations_dir
from swp.utils.setup import seed_everything
from swp.datasets.phonemes import get_phoneme_testloader
from swp.utils.models import get_model, get_model_args, get_train_args, load_weights
from swp.utils.datasets import (
    get_test_data,
    enrich_for_plotting,
    get_ablation_train_data,
    classify_error_positions,
)


def cache_lstm_weights(layer):
    """Cache LSTM weights and biases for later restoration"""
    weights = {
        "weight_ih_l0": layer.weight_ih_l0.clone(),
        "weight_hh_l0": layer.weight_hh_l0.clone(),
        "bias_ih_l0": layer.bias_ih_l0.clone(),
        "bias_hh_l0": layer.bias_hh_l0.clone(),
    }
    return weights


def restore_lstm_weights(layer, weights):
    """Restore LSTM weights and biases from cache"""
    with torch.no_grad():
        layer.weight_ih_l0.copy_(weights["weight_ih_l0"])
        layer.weight_hh_l0.copy_(weights["weight_hh_l0"])
        layer.bias_ih_l0.copy_(weights["bias_ih_l0"])
        layer.bias_hh_l0.copy_(weights["bias_hh_l0"])


def ablate_lstm_neuron(layer, neuron_idx, num_neurons):
    """Zero out all four gates of a single LSTM neuron"""
    with torch.no_grad():
        # Compute row indices for all four gates
        gate_indices = torch.tensor(
            [
                neuron_idx,
                neuron_idx + num_neurons,
                neuron_idx + num_neurons * 2,
                neuron_idx + num_neurons * 3,
            ]
        )
        # Zero out corresponding rows in weights and biases
        layer.weight_ih_l0[gate_indices] = 0
        layer.weight_hh_l0[gate_indices] = 0
        layer.bias_ih_l0[gate_indices] = 0
        layer.bias_hh_l0[gate_indices] = 0


def calc_accuracy(df, error_condition, total_condition):
    """
    Compute accuracy as 1 - (number of errors / total items) for a given condition.
    Returns NaN if no rows satisfy the total_condition.
    """
    total = df.loc[total_condition].shape[0]
    if total == 0:
        return np.nan
    errors = df.loc[error_condition].shape[0]
    return 1 - errors / total


def scatter_plot(results_df, x, y, xlabel, ylabel, filename, model_dir):
    """
    Produce and save a scatter plot for the specified x and y columns.
    """
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        data=results_df,
        x=x,
        y=y,
        hue="layer_name",
        palette={"encoder": "blue", "decoder": "red"},
        edgecolor="black",
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1)
    plt.legend(title="Layer")
    plt.grid(True)
    plt.savefig(model_dir / filename)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name string",
    )
    parser.add_argument(
        "--train_name",
        type=str,
        required=True,
        help="Training name string",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint to load",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )

    args = parser.parse_args()
    model_name = args.model_name
    train_name = args.train_name
    checkpoint = args.checkpoint

    model_args = get_model_args(model_name)
    train_args = get_train_args(train_name)
    include_stress = train_args["include_stress"]
    batch_size = train_args["batch_size"]

    seed_everything()
    device = torch.device("cpu")

    # Load and prepare data.
    test_data = get_test_data()
    train_data = get_ablation_train_data()
    train_data = train_data.sample(frac=0.3)
    ablation_data = pd.concat([test_data, train_data])
    ablation_loader = get_phoneme_testloader(batch_size, include_stress, ablation_data)

    # Load the model and weights.
    model = get_model(model_name)
    load_weights(
        model=model,
        model_name=model_name,
        train_name=train_name,
        checkpoint=checkpoint,
        device=device,
    )

    # Set up directories.
    ablations_dir = get_ablations_dir()
    model_dir = ablations_dir / f"{model_name}~{train_name}~{checkpoint}"
    model_dir.mkdir(exist_ok=True, parents=True)

    # check if results_df already exists
    if not (model_dir / "ablation_results.csv").exists():
        results_df = pd.read_csv(model_dir / "ablation_results.csv")

        # Loop over layers and neurons for ablation.
        ablation_results = []
        layers = [
            ("encoder", model.encoder.recurrent),
            ("decoder", model.decoder.recurrent),
        ]
        for layer_name, layer in layers:
            num_neurons = layer.hidden_size
            original_weights = cache_lstm_weights(layer)

            for neuron_idx in range(num_neurons):
                print(
                    f"Ablating neuron {neuron_idx+1}/{num_neurons} in {layer_name}",
                    end="\r",
                )
                ablate_lstm_neuron(layer, neuron_idx, num_neurons)
                df, _ = test(
                    model=model,
                    device=device,
                    test_df=ablation_data,
                    test_loader=ablation_loader,
                    include_stress=include_stress,
                )
                restore_lstm_weights(layer, original_weights)
                df = enrich_for_plotting(df, include_stress)
                df = classify_error_positions(df)

                # Compute accuracy values using our helper
                real_accuracy = calc_accuracy(
                    df,
                    (df["Lexicality"] == "real") & (df["Edit Distance"] > 0),
                    (df["Lexicality"] == "real"),
                )
                pseudo_accuracy = calc_accuracy(
                    df,
                    (df["Lexicality"] == "pseudo") & (df["Edit Distance"] > 0),
                    (df["Lexicality"] == "pseudo"),
                )
                low_freq_accuracy = calc_accuracy(
                    df,
                    (df["Lexicality"] == "real")
                    & (df["Zipf Frequency"] < 3.0)
                    & (df["Edit Distance"] > 0),
                    (df["Lexicality"] == "real") & (df["Zipf Frequency"] < 3.0),
                )
                high_freq_accuracy = calc_accuracy(
                    df,
                    (df["Lexicality"] == "real")
                    & (df["Zipf Frequency"] >= 3.0)
                    & (df["Edit Distance"] > 0),
                    (df["Lexicality"] == "real") & (df["Zipf Frequency"] >= 3.0),
                )
                simple_accuracy = calc_accuracy(
                    df,
                    (df["Lexicality"] == "real")
                    & (df["Morphology"] == "simple")
                    & (df["Edit Distance"] > 0),
                    (df["Lexicality"] == "real") & (df["Morphology"] == "simple"),
                )
                complex_accuracy = calc_accuracy(
                    df,
                    (df["Lexicality"] == "real")
                    & (df["Morphology"] == "complex")
                    & (df["Edit Distance"] > 0),
                    (df["Lexicality"] == "real") & (df["Morphology"] == "complex"),
                )
                primacy_accuracy = calc_accuracy(
                    df,
                    (df["Primacy Error"] > 0),
                    df["Edit Distance"] >= 0,
                )
                recency_accuracy = calc_accuracy(
                    df,
                    (df["Recency Error"] > 0),
                    (df["Edit Distance"] >= 0),
                )
                ablation_results.append(
                    {
                        "neuron_idx": neuron_idx,
                        "layer_name": layer_name,
                        "real_accuracy": real_accuracy,
                        "pseudo_accuracy": pseudo_accuracy,
                        "low_freq_accuracy": low_freq_accuracy,
                        "high_freq_accuracy": high_freq_accuracy,
                        "simple_accuracy": simple_accuracy,
                        "complex_accuracy": complex_accuracy,
                        "primacy_accuracy": primacy_accuracy,
                        "recency_accuracy": recency_accuracy,
                    }
                )

        # Save results.
        results_df = pd.DataFrame(ablation_results)
        results_df.to_csv(model_dir / "ablation_results.csv", index=False)

    results_df = pd.read_csv(model_dir / "ablation_results.csv")
    print("\nLowest accuracies:")
    for condition in [
        "real_accuracy",
        "pseudo_accuracy",
        "low_freq_accuracy",
        "high_freq_accuracy",
        "simple_accuracy",
        "complex_accuracy",
        "primacy_accuracy",
        "recency_accuracy",
    ]:
        # print the neuron indices as well
        print(
            results_df.sort_values(condition).head(3)[
                ["neuron_idx", "layer_name", condition]
            ]
        )

    # Produce scatter plots.
    scatter_plot(
        results_df,
        "real_accuracy",
        "pseudo_accuracy",
        "Real (Lexical Processing)",
        "Pseudo (Sublexical Processing)",
        "lex_scatter.png",
        model_dir,
    )

    scatter_plot(
        results_df,
        "low_freq_accuracy",
        "high_freq_accuracy",
        "Low Frequency",
        "High Frequency",
        "frq_scatter.png",
        model_dir,
    )

    scatter_plot(
        results_df,
        "simple_accuracy",
        "complex_accuracy",
        "Morphologically Simple",
        "Morphologically Complex",
        "mor_scatter.png",
        model_dir,
    )

    scatter_plot(
        results_df,
        "primacy_accuracy",
        "recency_accuracy",
        "Primacy",
        "Recency",
        "pos_scatter.png",
        model_dir,
    )
