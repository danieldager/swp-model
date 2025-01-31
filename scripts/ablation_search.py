import os
import sys
import torch
import argparse
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The PyTorch API of nested tensors is in prototype stage",
)

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from swp.test.repetition import test
from swp.utils.setup import seed_everything
from swp.utils.paths import get_ablations_dir
from swp.utils.models import get_model, load_weights
from swp.datasets.phonemes import get_phoneme_testloader
from swp.utils.models import get_model_args, get_train_args
from swp.utils.datasets import get_test_data, get_train_data



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
        default=None,
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

    model_class, recur_type, model_args = get_model_args(model_name)
    train_args = get_train_args(train_name)
    include_stress = train_args["s"]
    batch_size = train_args["b"]

    seed_everything()
    # device = set_device()
    device = torch.device("cpu")

    test_data = get_test_data()
    test_data = test_data[test_data["Lexicality"] == "pseudo"]
    test_loader = get_phoneme_testloader(batch_size, include_stress, test_data)

    train_data = get_train_data()
    train_data = train_data.sample(len(test_data))
    train_loader = get_phoneme_testloader(batch_size, include_stress, train_data)

    model = get_model(args.model_name)
    load_weights(
        model=model,
        model_name=model_name,
        train_name=train_name,
        checkpoint=checkpoint,
        device=device,
    )

    ablations_dir = get_ablations_dir()
    model_dir = ablations_dir / f"{model_name}~{train_name}~{checkpoint}"
    model_dir.mkdir(exist_ok=True, parents=True)

    results = []
    layers = [
        ("encoder", model.encoder.recurrent),
        ("decoder", model.decoder.recurrent),
    ]
    for layer_name, layer in layers:
        print(f"Layer: {layer_name}")
        num_neurons = layer.hidden_size
        original_weights = cache_lstm_weights(layer)

        for neuron_idx in range(num_neurons):
            print(f"{neuron_idx + 1}/{num_neurons}", end="\r")
            ablate_lstm_neuron(layer, neuron_idx, num_neurons)
            _, train_error = test(
                model=model,
                device=device,
                test_df=train_data,
                test_loader=train_loader,
                include_stress=include_stress,
                verbose=args.verbose,
            )
            _, test_error = test(
                model=model,
                device=device,
                test_df=test_data,
                test_loader=test_loader,
                include_stress=include_stress,
                verbose=args.verbose,
            )
            restore_lstm_weights(layer, original_weights)

            results.append(
                {
                    "neuron_idx": neuron_idx,
                    "layer_name": layer_name,
                    "train_error": train_error,
                    "test_error": test_error,
                    "combined_error": train_error + test_error,
                }
            )
    results_df = pd.DataFrame(results)
    results_df.to_csv(model_dir / "errors.csv")
    print(results_df.nlargest(30, "combined_error"))

    # Create a scatter plot using seaborn
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=results_df,
        x="train_error",
        y="test_error",
        hue="layer_name",
        palette={"encoder": "blue", "decoder": "red"},
        alpha=0.5,
        edgecolor="black",
    )

    # Customize the plot
    plt.xlabel("Train Error")
    plt.ylabel("Test Error")
    plt.title("Ablation Impact on Train vs Test Error")
    plt.legend(title="Layer")
    plt.grid(True)

    # Save the plot
    plt.savefig(model_dir / "scatter.png")
