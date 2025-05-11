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
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The PyTorch API of nested tensors is in prototype stage",
)

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from swp.datasets.phonemes import get_dataset, get_phoneme_testloader
from swp.models.metrics import classic_errors, free_gen_errors
from swp.test.ablations import ablate_lstm_neuron
from swp.test.repetition import test
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
        required=True,
        help="Dataset to test on",
    )
    parser.add_argument(
        "--ngrams",
        type=int,
        default=None,
        help="Determines size of ngrams to test",
    )
    parser.add_argument(
        "--include_stress",
        action="store_true",
        help="Include stress in phonemes",
    )
    parser.add_argument(
        "--store_out",
        action="store_true",
        help="Store the 'out' variable of LSTM hook",
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

    model_name = args.model_name
    train_name = args.train_name
    error_meter = classic_errors

    backend_setup()
    seed_everything()
    device = set_device()
    model = None

    def create_embeddings_LSTM_hook(
        embeddings: dict[str, list[np.ndarray]],
        is_batched: bool,
        num_layers: int,
        store_out: bool = False,
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
            out, (h, c) = output  # h, c (L, B, H)

            if store_out and isinstance(out, PackedSequence):
                out, _ = pad_packed_sequence(out, batch_first=True)

            if not is_batched:
                out = out.unsqueeze(0)
                h = h.unsqueeze(1)
                c = c.unsqueeze(1)

            # Pad output to allow for concatenation
            if store_out:
                out = out.detach().cpu().numpy()
                B, T, H = out.shape
                padded_out = np.zeros((B, MAX_PAD, H))
                padded_out[:, :T, :] = out
                embeddings["Out"].append(padded_out)

            h = h.squeeze(0)
            h = h.detach().cpu().numpy()
            embeddings["Hidden"].append(h)

            c = c.squeeze(0)
            c = c.detach().cpu().numpy()
            embeddings["Cell"].append(c)

        return embeddings_LSTM_hook

    weights_dir = get_weights_dir() / model_name / train_name

    checkpoints = (
        [args.checkpoint]
        if args.checkpoint is not None
        else [f.stem.split(".")[-1] for f in weights_dir.glob("*.pth")]
    )

    for checkpoint in checkpoints:
        results_dir = get_evaluation_dir() / model_name / train_name / f"{checkpoint}"
        figures_dir = (
            get_figures_dir() / model_name / train_name / "embeddings" / f"{checkpoint}"
        )

        ### LOAD AND HOOK ###

        if model == None:
            model = get_model(model_name)
        load_weights(
            model=model,
            model_name=model_name,
            train_name=train_name,
            checkpoint=checkpoint,
            device=device,
        )
        num_layers = model.encoder.num_layers

        embeddings = {"Out": [], "Hidden": [], "Cell": []}
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

        if args.ngrams:
            dataset = f"{args.ngrams}{args.dataset[1:]}"
        else:
            dataset = args.dataset

        df_path = results_dir / f"{dataset}.csv"
        h_path = results_dir / f"{dataset}_h.npy"
        c_path = results_dir / f"{dataset}_c.npy"

        if args.retest or not h_path.exists():
            test_df = get_dataset(args.dataset, args.ngrams)
            test_loader = get_phoneme_testloader(
                args.batch_size,
                args.include_stress,
                dataset_df=test_df,
            )
            results, _ = test(
                model=model,
                device=device,
                test_df=test_df,
                test_loader=test_loader,
                include_stress=args.include_stress,
                error_meter=error_meter,
                verbose=args.verbose,
            )
            if not df_path.exists() or args.retest:
                results.to_csv(df_path)
            h_emb = np.concat(embeddings["Hidden"])
            c_emb = np.concat(embeddings["Cell"])

            if args.store_out:
                # if you want hidden state for each token in sequence
                # use lengths to zero out hidden states for pad tokens
                o_emb = np.concat(embeddings["Out"])
                _, T, _ = o_emb.shape
                lengths = results["Length"].to_numpy() + 1
                mask = np.arange(T)[None, :] < lengths[:, None]
                mask = mask[:, :, None]
                o_emb *= mask
                o_path = results_dir / f"{dataset}_o.npy"
                np.save(o_path, o_emb)

            print(f"Saving embeddings to {results_dir}")
            np.save(h_path, h_emb)
            np.save(c_path, c_emb)

        ### PLOTTING ###

        converters = {
            "Phonemes": literal_eval,
            "No_Stress": literal_eval,
            "Prediction": literal_eval,
        }

        df = pd.read_csv(df_path, index_col=0, converters=converters)

        h_emb = np.load(h_path)
        c_emb = np.load(c_path)

        if args.all or args.dmat:
            dissim_matrix(df, dataset, num_layers, figures_dir, args.metric)

        if args.all or args.mlem:
            mlem_importance(df, dataset, num_layers, figures_dir, args.metric)

        if args.all or args.pca:
            pca_mds(df, dataset, num_layers, figures_dir)
            pca_mds(df, dataset, num_layers, figures_dir, last_token=True)

        if args.verbose:
            print("-" * 60)
