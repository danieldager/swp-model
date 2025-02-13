import argparse
import os
import sys
import warnings
from ast import literal_eval

import pandas as pd

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The PyTorch API of nested tensors is in prototype stage",
)

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from swp.datasets.phonemes import get_phoneme_testloader, get_sonority_dataset
from swp.test.ablations import ablate_lstm_neuron
from swp.test.repetition import test
from swp.utils.datasets import enrich_for_plotting, get_test_data, get_train_data
from swp.utils.models import get_model, load_weights
from swp.utils.paths import (
    get_ablations_dir,
    get_figures_dir,
    get_test_dir,
    get_weights_dir,
)
from swp.utils.setup import backend_setup, seed_everything, set_device
from swp.viz.test import (
    plot_category_errors,
    plot_length_errors,
    plot_position_errors,
    plot_position_errors_bins,
    plot_position_smoothened_errors,
    plot_sonority_errors,
    regression_plots,
)

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
        "--plot",
        action="store_true",
        help="Generate plots",
    )
    parser.add_argument(
        "--abblate_layer",
        type=str,
        default=None,
        help="Layer name to ablate",
    )
    parser.add_argument(
        "--abblate_neuron",
        type=int,
        default=None,
        help="Neuron index to ablate",
    )
    parser.add_argument(
        "--test_train",
        action="store_true",
        help="Tests also on the training set",
    )

    args = parser.parse_args()
    model_name = args.model_name
    train_name = args.train_name
    batch_size = args.batch_size
    checkpoint = args.checkpoint
    include_stress = args.include_stress

    seed_everything()
    backend_setup()
    device = set_device()

    test_dir = get_test_dir()
    model_dir = test_dir / f"{model_name}~{train_name}"
    model_dir.mkdir(exist_ok=True, parents=True)
    weights_dir = get_weights_dir() / model_name / train_name

    if checkpoint is None:
        checkpoints = [f.stem.split(".")[-1] for f in weights_dir.glob("*.pth")]
    else:
        checkpoints = [checkpoint]

    for checkpoint in checkpoints:

        model = get_model(args.model_name)
        load_weights(
            model=model,
            model_name=model_name,
            train_name=train_name,
            checkpoint=checkpoint,
            device=device,
        )

        if args.abblate_layer is not None and args.abblate_neuron is not None:
            layer_name = args.abblate_layer
            neuron_idx = args.abblate_neuron
            layers = {
                "encoder": model.encoder.recurrent,
                "decoder": model.decoder.recurrent,
            }
            layer = layers[layer_name]
            num_neurons = layer.hidden_size
            ablate_lstm_neuron(layer, neuron_idx, num_neurons)
            ablations_dir = get_ablations_dir()
            model_dir = (
                ablations_dir
                / f"{model_name}~{train_name}~{checkpoint}"
                / f"{layer_name}_{neuron_idx}"
            )
            model_dir.mkdir(exist_ok=True, parents=True)
        elif args.abblate_layer is not None or args.abblate_neuron is not None:
            raise ValueError(
                "abblate_layer and abblate_neuron have to be passed together to run abblation"
            )

        # if the results datasets already exist, skip testing
        if not (model_dir / f"{checkpoint}.csv").exists():
            test_df = get_test_data()
            test_loader = get_phoneme_testloader(batch_size, include_stress)
            test_results, _ = test(
                model=model,
                device=device,
                test_df=test_df,
                test_loader=test_loader,
                include_stress=include_stress,
                verbose=args.verbose,
            )
            test_results.to_csv(model_dir / f"{checkpoint}.csv")

        if not (model_dir / f"{checkpoint}~ssp.csv").exists():
            ssp_df = get_sonority_dataset(include_stress=include_stress)
            ssp_loader = get_phoneme_testloader(batch_size, include_stress, ssp_df)
            ssp_results, _ = test(
                model=model,
                device=device,
                test_df=ssp_df,
                test_loader=ssp_loader,
                include_stress=include_stress,
                verbose=args.verbose,
            )
            ssp_results.to_csv(model_dir / f"{checkpoint}~ssp.csv")

        if args.test_train and not (model_dir / f"{checkpoint}~train.csv").exists():
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
            # train_results = enrich_for_plotting(train_results, include_stress)

        if args.plot:

            converters = {
                "Phonemes": literal_eval,
                "No Stress": literal_eval,
                "Prediction": literal_eval,
            }

            test_results = pd.read_csv(
                model_dir / f"{checkpoint}.csv", index_col=0, converters=converters
            )
            ssp_results = pd.read_csv(
                model_dir / f"{checkpoint}~ssp.csv", index_col=0, converters=converters
            )

            test_results = enrich_for_plotting(test_results, include_stress)
            ssp_results = enrich_for_plotting(ssp_results, include_stress)
            figures_dir = get_figures_dir() / f"{args.model_name}~{args.train_name}"
            figures_dir.mkdir(exist_ok=True)

            plot_length_errors(test_results, checkpoint, figures_dir)
            plot_position_smoothened_errors(test_results, checkpoint, figures_dir)
            plot_position_errors_bins(test_results, checkpoint, figures_dir, num_bins=3)
            plot_sonority_errors(ssp_results, checkpoint, figures_dir)
            # plot_category_errors(test_results, checkpoint, figures_dir)
            regression_plots(test_results, checkpoint, figures_dir, 1)
            regression_plots(test_results, checkpoint, figures_dir, 2)

        if args.verbose:
            print("-" * 60)
