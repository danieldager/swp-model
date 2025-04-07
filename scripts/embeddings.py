# TODO: extracting hidden states and activations should be added to the
# standard testing pipeline, no need for a separate script

import argparse
import os
import sys
import warnings
from ast import literal_eval
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The PyTorch API of nested tensors is in prototype stage",
)

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from swp.datasets.phonemes import (
    get_bigram_dataset,
    get_trigram_dataset,
    get_handmade_dataset,
    get_phoneme_dataset,
    get_phoneme_testloader,
    get_sonority_dataset,
)
from swp.models.metrics import classic_errors, free_gen_errors
from swp.test.ablations import ablate_lstm_neuron
from swp.test.repetition import test
from swp.utils.datasets import (
    enrich_for_plotting,
    get_evaluation_dataset,
    get_train_dataset,
)
from swp.utils.models import get_model, load_weights
from swp.utils.paths import get_evaluation_dir, get_figures_dir, get_weights_dir
from swp.utils.setup import backend_setup, seed_everything, set_device
from swp.viz.embeddings import dissim_matrix, mlem_importance, pca_mds

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
        "--batch_size",
        type=int,
        default=512,
        help="Test dataloader batch size",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint to load",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to test on",
        required=True,
    )
    parser.add_argument(
        "--include_stress",
        action="store_true",
        help="Include stress in phonemes",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )
    parser.add_argument(
        "--retest",
        action="store_true",
        help="Regenerate test results",
    )
    parser.add_argument(
        "--dmat",
        action="store_true",
        help="Plot dissimilarity matrix",
    )
    parser.add_argument(
        "--mlem",
        action="store_true",
        help="Plot MLEM feature importance",
    )
    parser.add_argument(
        "--pca",
        action="store_true",
        help="Plot PCA and MDS",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all visualizations",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        help="Distance metric for dissimilarity matrix and MLEM",
    )
    parser.add_argument(
        "--ablate_layer",
        type=str,
        default=None,
        help="Layer name to ablate",
    )
    parser.add_argument(
        "--ablate_neuron",
        type=int,
        default=None,
        help="Neuron index to ablate",
    )

    args = parser.parse_args()
    error_meter = free_gen_errors

    backend_setup()
    seed_everything()
    device = set_device()
    model = None

    def create_embeddings_LSTM_hook(
        embeddings: dict[str, list[np.ndarray]], is_batched: bool, num_layers: int
    ) -> Callable[
        [
            nn.Module,
            tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
            tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        ],
        None,
    ]:
        MAX_PAD = 20

        def embeddings_LSTM_hook(
            module: nn.Module,
            inputs: tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
            output: tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        ):
            """Hook function to capture the final hidden state."""
            out, _ = output  # h, c (L, B, H)

            if not is_batched:
                out = out.unsqueeze(0)

            out = out.detach().cpu().numpy()
            B, T, H = out.shape
            padded_out = np.zeros((B, MAX_PAD, H))
            padded_out[:, :T, :] = out
            embeddings[f"Hidden"].append(padded_out)

        return embeddings_LSTM_hook

    valid_datasets = [
        "evaluation",
        "sonority",
        "train",
        "phoneme",
        "bigram",
        "trigram",
    ]
    if args.dataset not in valid_datasets:
        raise ValueError(f"Dataset {args.dataset} not recognized")

    name = f"{args.model_name}~{args.train_name}"
    weights_dir = get_weights_dir() / name

    checkpoints = (
        [args.checkpoint]
        if args.checkpoint is not None
        else [f.stem.split(".")[-1] for f in weights_dir.glob("*.pth")]
    )

    for checkpoint in checkpoints:
        # TODO: redo the file structure
        results_dir = get_evaluation_dir() / name / "epochs" / f"{checkpoint}"
        figures_dir = get_figures_dir() / name / "embeddings" / f"{checkpoint}"

        ### LOAD AND HOOK ###

        if model == None:
            model = get_model(args.model_name)
        load_weights(
            model=model,
            model_name=args.model_name,
            train_name=args.train_name,
            checkpoint=checkpoint,
            device=device,
        )
        num_layers = model.encoder.num_layers

        embeddings = {"Hidden": []}
        for i in range(num_layers):
            embeddings[f"H{i+1}"] = []
            embeddings[f"C{i+1}"] = []
        is_batched = True if args.batch_size > 1 else False

        hook = create_embeddings_LSTM_hook(embeddings, is_batched, num_layers)
        hook_handle = model.encoder.recurrent.register_forward_hook(hook)

        ### ABLATIONS ###

        if args.ablate_layer is not None and args.ablate_neuron is not None:
            layer_name = args.ablate_layer
            neuron_idx = args.ablate_neuron
            layers = {
                "encoder": model.encoder.recurrent,
                "decoder": model.decoder.recurrent,
            }
            layer = layers[layer_name]
            num_neurons = layer.hidden_size
            ablate_lstm_neuron(layer, neuron_idx, num_neurons)
            results_dir = results_dir / f"{layer_name}_{neuron_idx}"
            figures_dir = figures_dir / f"{layer_name}_{neuron_idx}"

        elif args.ablate_layer is not None or args.ablate_neuron is not None:
            raise ValueError(
                "ablate_layer and ablate_neuron have to be passed together to run ablation"
            )

        else:
            results_dir = results_dir / "control"
            figures_dir = figures_dir / "control"

        results_dir.mkdir(exist_ok=True, parents=True)
        figures_dir.mkdir(exist_ok=True, parents=True)

        ### TESTING ###

        csv_path = results_dir / f"{args.dataset}.csv"
        npy_path = results_dir / f"{args.dataset}.npy"

        if args.retest or not csv_path.exists():

            if args.dataset == "phoneme":
                test_df = get_phoneme_dataset()
            elif args.dataset == "bigram":
                test_df = get_bigram_dataset()
            elif args.dataset == "trigram":
                test_df = get_trigram_dataset()
            elif args.dataset == "sonority":
                test_df = get_sonority_dataset()
            elif args.dataset == "evaluation":
                test_df = get_evaluation_dataset()
            elif args.dataset == "train":
                test_df = get_train_dataset()

            test_loader = get_phoneme_testloader(
                args.batch_size,
                args.include_stress,
                dataset_df=test_df,
            )

            results_df, _ = test(
                model=model,
                device=device,
                test_df=test_df,
                test_loader=test_loader,
                include_stress=args.include_stress,
                error_meter=error_meter,
                verbose=args.verbose,
            )
            # results_df = results_df.reset_index(drop=True)
            results_df.to_csv(csv_path)

            embeddings_np = np.concat(embeddings["Hidden"])

            lengths = results_df["Length"].to_numpy()
            pad = np.ones(embeddings_np.shape)

            # use lengths array to add zeros to the pad matrix



            np.save(npy_path, embeddings_np)

        ### PLOTTING ###

        converters = {
            "Phonemes": literal_eval,
            "No Stress": literal_eval,
            "Prediction": literal_eval,
        }

        df = pd.read_csv(csv_path, index_col=0, converters=converters)

        emb = np.load(npy_path)

        if args.all or args.dmat:
            dissim_matrix(df, args.dataset, num_layers, figures_dir, args.metric)

        if args.all or args.mlem:
            mlem_importance(df, args.dataset, num_layers, figures_dir, args.metric)

        if args.all or args.pca:
            pca_mds(df, args.dataset, num_layers, figures_dir)
            pca_mds(df, args.dataset, num_layers, figures_dir, last_token=True)

        if args.verbose:
            print("-" * 60)
