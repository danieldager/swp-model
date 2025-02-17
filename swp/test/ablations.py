import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..models.autoencoder import Bimodel, Unimodel
from ..utils.datasets import classify_error_positions, enrich_for_plotting
from ..utils.metrics import calc_accuracy
from .repetition import test


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


def ablate(
    model: Unimodel,
    device: str | torch.device,
    test_df: pd.DataFrame,
    test_loader: DataLoader,
    include_stress: bool,
) -> pd.DataFrame:
    # TODO docstring
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
                test_df=test_df,
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
    return results_df
