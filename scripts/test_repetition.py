import argparse
import os
import sys
import warnings
from ast import literal_eval

import pandas as pd
import torch

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The PyTorch API of nested tensors is in prototype stage",
)

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from swp.datasets.phonemes import get_phoneme_testloader, get_sonority_dataset
from swp.test.repetition import test
from swp.utils.datasets import (
    enrich_for_plotting,
    get_test_data,
    get_train_data,
    get_valid_fold,
)
from swp.utils.models import get_model, load_weights
from swp.utils.paths import get_figures_dir, get_test_dir, get_weights_dir
from swp.utils.plots import (
    plot_category_errors,
    plot_length_errors,
    plot_position_errors,
    plot_sonority_errors,
    regression_plots,
)
from swp.utils.setup import backend_setup, seed_everything, set_device

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

        # if not (model_dir / f"{checkpoint}~train.csv").exists():
        #     train_df = get_train_data()
        #     train_df = get_valid_fold(fold_id=0)  ### HARDCODE ###
        #     train_loader = get_phoneme_testloader(batch_size, include_stress, train_df)
        #     train_results, train_error = test(
        #         model=model,
        #         device=device,
        #         test_df=train_df,
        #         test_loader=train_loader,
        #         include_stress=include_stress,
        #         verbose=args.verbose,
        #     )
        #     train_results.to_csv(model_dir / f"{checkpoint}~train.csv")
        # if not (model_dir / f"{checkpoint}~train.csv").exists():
        #     train_df = get_train_data()
        #     train_df = get_valid_fold(fold_id=0)  ### HARDCODE ###
        #     train_loader = get_phoneme_testloader(batch_size, include_stress, train_df)
        #     train_results, train_error = test(
        #         model=model,
        #         device=device,
        #         test_df=train_df,
        #         test_loader=train_loader,
        #         include_stress=include_stress,
        #         verbose=args.verbose,
        #     )
        #     train_results.to_csv(model_dir / f"{checkpoint}~train.csv")

        #     print(f"Train error: {train_error}")
        #     print(f"Train error: {train_error}")

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
            # train_results = pd.read_csv(
            #     model_dir / f"{checkpoint}~train.csv", index_col=0, converters=converters
            # )

            test_results = enrich_for_plotting(test_results, include_stress)
            ssp_results = enrich_for_plotting(ssp_results, include_stress)
            # train_results = enrich_for_plotting(train_results, include_stress)

            figures_dir = get_figures_dir() / f"{args.model_name}~{args.train_name}"
            figures_dir.mkdir(exist_ok=True)
            # train_results = enrich_for_plotting(train_results, include_stress)

            plot_length_errors(test_results, checkpoint, figures_dir)
            plot_position_errors(test_results, checkpoint, figures_dir)
            plot_sonority_errors(ssp_results, checkpoint, figures_dir)
            plot_category_errors(test_results, checkpoint, figures_dir)
            regression_plots(test_results, checkpoint, figures_dir, 1)
            regression_plots(test_results, checkpoint, figures_dir, 2)
            # regression_plots(train_results, checkpoint, figures_dir, 3)

        if args.verbose:
            print("-" * 60)
