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
from swp.utils.datasets import get_test_data, get_train_data
from swp.utils.plots import enrich_for_plotting, error_plots, regression_plots
from swp.utils.models import (
    get_model,
    load_weights,
    get_model_args,
    get_train_args,
)
from swp.datasets.phonemes import (
    get_phoneme_testloader,
    get_sonority_dataset,
    get_phoneme_to_id,
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
        "--layer",
        type=str,
        required=True,
        help="Layer name to ablate",
    )
    parser.add_argument(
        "--neuron",
        type=int,
        required=True,
        help="Neuron index to ablate",
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
    layer_name = args.layer
    neuron_idx = args.neuron

    seed_everything()
    # device = set_device()
    device = torch.device("cpu")

    model_class, recur_type, model_args = get_model_args(model_name)
    train_args = get_train_args(train_name)
    include_stress = train_args["s"]
    batch_size = train_args["b"]
    phoneme_to_id = get_phoneme_to_id(include_stress)

    ablations_dir = get_ablations_dir()
    model_dir = ablations_dir / f"{model_name}~{train_name}~{checkpoint}"
    model_dir.mkdir(exist_ok=True, parents=True)

    model = get_model(args.model_name)
    load_weights(
        model=model,
        model_name=model_name,
        train_name=train_name,
        checkpoint=checkpoint,
        device=device,
    )

    layers = {
        "encoder": model.encoder.recurrent,
        "decoder": model.decoder.recurrent,
    }
    layer = layers[layer_name]
    num_neurons = layer.hidden_size

    ## Testing ##
    # original_weights = cache_lstm_weights(layer)
    ablate_lstm_neuron(layer, neuron_idx, num_neurons)

    test_df = get_test_data()
    test_loader = get_phoneme_testloader(batch_size, include_stress)
    test_results, test_error = test(
        model=model,
        device=device,
        test_df=test_df,
        test_loader=test_loader,
        include_stress=include_stress,
        verbose=args.verbose,
    )
    test_results.to_csv(model_dir / f"{checkpoint}~test.csv")

    ssp_df = get_sonority_dataset(include_stress=include_stress)
    ssp_loader = get_phoneme_testloader(batch_size, include_stress, ssp_df)
    ssp_results, ssp_error = test(
        model=model,
        device=device,
        test_df=ssp_df,
        test_loader=ssp_loader,
        include_stress=include_stress,
        verbose=args.verbose,
    )
    ssp_results.to_csv(model_dir / f"{checkpoint}~ssp.csv")

    train_df = get_train_data()
    train_loader = get_phoneme_testloader(batch_size, include_stress, train_df)
    train_results, train_error = test(
        model=model,
        device=device,
        test_df=train_df,
        test_loader=train_loader,
        include_stress=include_stress,
        verbose=args.verbose,
    )
    train_results.to_csv(model_dir / f"{checkpoint}~train.csv")

    # restore_lstm_weights(layer, original_weights)

    ## Plotting ##
    test_df = enrich_for_plotting(test_df, phoneme_to_id, include_stress)
    ssp_df = enrich_for_plotting(ssp_df, phoneme_to_id, include_stress)
    train_df = enrich_for_plotting(train_df, phoneme_to_id, include_stress)

    error_plots(
        test_df,
        ssp_df,
        args.model_name,
        args.train_name,
        checkpoint,
    )

    regression_plots(
        test_df,
        train_df,
        args.model_name,
        args.train_name,
        checkpoint,
    )
