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
from swp.models.metrics import classic_errors, free_gen_errors
from swp.test.ablations import ablate_lstm_neuron
from swp.test.repetition import test
from swp.utils.datasets import enrich_for_plotting, get_test_data, get_train_data
from swp.utils.models import get_model, load_weights
from swp.utils.paths import (
    get_ablations_dir,
    get_evaluation_dir,
    get_figures_dir,
    get_weights_dir,
)
from swp.utils.setup import backend_setup, seed_everything, set_device
from swp.viz.test import (
    plot_category_errors,
    plot_frequency_errors,
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
        "--retest",
        action="store_true",
        help="Regenerate test results",
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
    error_meter = free_gen_errors

    seed_everything()
    backend_setup()
    device = set_device()

    weights_dir = get_weights_dir() / model_name / train_name

    if checkpoint is None:
        checkpoints = [f.stem.split(".")[-1] for f in weights_dir.glob("*.pth")]
    else:
        checkpoints = [checkpoint]

    for checkpoint in checkpoints:

        results_dir = (
            get_evaluation_dir() / f"{model_name}~{train_name}" / f"{checkpoint}"
        )
        figures_dir = (
            get_figures_dir()
            / f"{model_name}~{train_name}"
            / f"{checkpoint}"
            / "evaluation"
        )

        model = get_model(args.model_name)
        load_weights(
            model=model,
            model_name=model_name,
            train_name=train_name,
            checkpoint=checkpoint,
            device=device,
        )

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

        if args.retest or not (results_dir / "fdd.csv").exists():
            test_df = get_test_data()
            test_loader = get_phoneme_testloader(batch_size, include_stress)
            test_results, _ = test(
                model=model,
                device=device,
                test_df=test_df,
                test_loader=test_loader,
                include_stress=include_stress,
                error_meter=error_meter,
                verbose=args.verbose,
            )
            test_results.to_csv(results_dir / "fdd.csv")

        if args.retest or not (results_dir / f"ssp.csv").exists():
            ssp_df = get_sonority_dataset(include_stress=include_stress)
            ssp_loader = get_phoneme_testloader(batch_size, include_stress, ssp_df)
            ssp_results, _ = test(
                model=model,
                device=device,
                test_df=ssp_df,
                test_loader=ssp_loader,
                include_stress=include_stress,
                error_meter=error_meter,
                verbose=args.verbose,
            )
            ssp_results.to_csv(results_dir / f"ssp.csv")

        if args.test_train and (
            args.retest or not (results_dir / f"train.csv").exists()
        ):
            train_df = get_train_data()
            train_loader = get_phoneme_testloader(batch_size, include_stress, train_df)
            train_results, train_error = test(
                model=model,
                device=device,
                test_df=train_df,
                test_loader=train_loader,
                include_stress=include_stress,
                error_meter=error_meter,
                verbose=args.verbose,
            )
            train_results.to_csv(results_dir / f"train.csv")
            train_results = enrich_for_plotting(train_results, include_stress)

        ### PLOTTING ###

        converters = {
            "Phonemes": literal_eval,
            "No Stress": literal_eval,
            "Prediction": literal_eval,
        }

        test_results = pd.read_csv(
            results_dir / "fdd.csv", index_col=0, converters=converters
        )
        ssp_results = pd.read_csv(
            results_dir / "ssp.csv", index_col=0, converters=converters
        )

        test_results = enrich_for_plotting(test_results, include_stress)
        ssp_results = enrich_for_plotting(ssp_results, include_stress)

        plot_length_errors(test_results, figures_dir)
        plot_frequency_errors(test_results, figures_dir)
        plot_sonority_errors(ssp_results, figures_dir)
        plot_position_smoothened_errors(test_results, figures_dir)
        plot_position_errors_bins(test_results, figures_dir, num_bins=3)

        plot_category_errors(test_results, figures_dir)
        regression_plots(test_results, figures_dir, "real")
        regression_plots(test_results, figures_dir, "both")

        if args.verbose:
            print("-" * 60)
