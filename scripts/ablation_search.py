import argparse
import os
import sys
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The PyTorch API of nested tensors is in prototype stage",
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*set_ticklabels\(\) should only be used with a fixed number of ticks.*",
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from swp.datasets.phonemes import get_phoneme_testloader
from swp.test.repetition import test
from swp.utils.datasets import (
    classify_error_positions,
    enrich_for_plotting,
    get_test_data,
)
from swp.utils.models import get_model, get_model_args, get_train_args, load_weights
from swp.utils.paths import get_ablations_dir
from swp.utils.setup import seed_everything

sns.set_palette("colorblind")


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
    """
    total = df.loc[total_condition].shape[0]
    errors = df.loc[error_condition].shape[0]
    return 1 - errors / total


def scatter_plot(
    results_df, x, y, xlabel, ylabel, filename, model_dir, log_scale=False
):
    """
    Produce and save a scatter plot for the specified x and y columns.

    Parameters:
        results_df (DataFrame): DataFrame containing the data.
        x (str): Name of the x column.
        y (str): Name of the y column.
        xlabel (str): Label for the x axis.
        ylabel (str): Label for the y axis.
        filename (str): File name to save the plot.
        model_dir (Path): Directory where to save the plot.
        log_scale (bool): If True, set both axes to a combined linear/logarithmic scale.
                          The region from 0 to 0.01 is linear (with 0 shown as the lower tick),
                          and above that the scale is logarithmic. The lower and upper ticks are
                          forced to be 0 and 100 respectively, with intermediate ticks as whole numbers.
    """
    # Set text elements to font size 18
    plt.rcParams.update({"font.size": 18})

    fig, ax = plt.subplots(figsize=(8, 8))

    results_df = results_df.copy()
    results_df[x] = 1 - results_df[x]
    results_df[y] = 1 - results_df[y]
    results_df["distance"] = (results_df[y] - results_df[x]) / np.sqrt(2)

    # Plot the scatter points
    sns.scatterplot(
        data=results_df,
        x=x,
        y=y,
        hue="layer_name",
        palette={"encoder": "blue", "decoder": "red"},
        # edgecolor="black",
        alpha=0.9,
        s=50,
        ax=ax,
    )
    ax.set_xlabel(xlabel, fontsize=24, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=24)
    ax.grid(True)

    if log_scale:
        ax.set_xscale("symlog", linthresh=0.001)
        ax.set_yscale("symlog", linthresh=0.001)
        ax.set_xlim(-1e-4, 1)
        ax.set_ylim(-1e-4, 1)
        ticks = ["0", "0.1", "1", "10", "100"]
        log_grid = [i * 1e-4 for i in [0, 2.5, 5, 7.5]] + [
            i * 10 ** -(3 - j) for j in range(0, 4) for i in range(1, 10)
        ]
        for x in log_grid:
            ax.axhline(x, linestyle="--", color="k", alpha=0.1)
            ax.axvline(x, linestyle="--", color="k", alpha=0.1)
    else:
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        lin_grid = [i * 1e-1 for i in range(11)]
        ticks = ["0", "0", "20", "40", "60", "80", "100"]
        for x in lin_grid:
            ax.axhline(x, linestyle="--", color="k", alpha=0.1)
            ax.axvline(x, linestyle="--", color="k", alpha=0.1)

    ax.plot([-0.001, 1], [-0.001, 1], color="grey", linestyle="--", linewidth=1)
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)

    # Draw a diagonal reference line
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    plt.savefig(model_dir / f"{filename}_scatter.png", dpi=300)

    # Create a legend figure
    ax.legend(title="Layer")
    handles, labels = ax.get_legend_handles_labels()
    figLegend = plt.figure(figsize=(2, 2))
    figLegend.legend(handles, labels, loc="center", title="Layer")
    figLegend.canvas.draw()
    plt.axis("off")
    figLegend.savefig(model_dir / "legend.png", dpi=300, bbox_inches="tight")

    plt.close(fig)
    plt.close(figLegend)

    common_bins = np.linspace(
        results_df["distance"].min(), results_df["distance"].max(), 21
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    for ax, layer, color in zip(axes, ["encoder", "decoder"], ["blue", "red"]):
        subset = results_df[results_df["layer_name"] == layer]
        ax.hist(subset["distance"], bins=common_bins, color=color)
        ax.grid(True)
        xabs_max = abs(max(ax.get_xlim(), key=abs))
        ax.set_xlim(xmin=-xabs_max, xmax=xabs_max)

    xlabel = fig.supxlabel("Distance from Diagonal", fontsize=24)
    xlabel.set_position((0.54, 0.05))
    axes[0].set_ylabel("# of Neurons", fontsize=24, labelpad=10)
    plt.tight_layout()
    plt.savefig(model_dir / f"{filename}_histos.png", dpi=300)
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

    # Load and prepare data
    test_data = get_test_data()
    test_loader = get_phoneme_testloader(batch_size, include_stress)

    # Load the model and weights
    model = get_model(model_name)
    load_weights(
        model=model,
        model_name=model_name,
        train_name=train_name,
        checkpoint=checkpoint,
        device=device,
    )

    # Set up directories
    ablations_dir = get_ablations_dir()
    model_dir = ablations_dir / f"{model_name}~{train_name}~{checkpoint}"
    model_dir.mkdir(exist_ok=True, parents=True)

    # Check if results_df already exists
    if not (model_dir / "ablation_results.csv").exists():

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
                    test_df=test_data,
                    test_loader=test_loader,
                    include_stress=include_stress,
                )
                restore_lstm_weights(layer, original_weights)
                df = enrich_for_plotting(df, include_stress)
                df = classify_error_positions(df)

                # Compute accuracy values
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
                short_accuracy = calc_accuracy(
                    df,
                    (df["Size"] == "short") & (df["Edit Distance"] > 0),
                    (df["Size"] == "short"),
                )
                long_accuracy = calc_accuracy(
                    df,
                    (df["Size"] == "long") & (df["Edit Distance"] > 0),
                    (df["Size"] == "long"),
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
                        "short_accuracy": short_accuracy,
                        "long_accuracy": long_accuracy,
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
        "short_accuracy",
        "long_accuracy",
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
        "Real (Lexical)",
        "Pseudo (Sublexical)",
        "lex",
        model_dir,
    )

    scatter_plot(
        results_df,
        "low_freq_accuracy",
        "high_freq_accuracy",
        "Low Frequency",
        "High Frequency",
        "frq",
        model_dir,
    )

    scatter_plot(
        results_df,
        "simple_accuracy",
        "complex_accuracy",
        "Morphologically Simple",
        "Morphologically Complex",
        "mor",
        model_dir,
    )

    scatter_plot(
        results_df,
        "primacy_accuracy",
        "recency_accuracy",
        "Primacy (First Half)",
        "Recency (Second Half)",
        "pos",
        model_dir,
    )

    scatter_plot(
        results_df,
        "short_accuracy",
        "long_accuracy",
        "Short Words",
        "Long Words",
        "len",
        model_dir,
    )
