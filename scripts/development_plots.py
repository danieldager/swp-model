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
from swp.viz.development import development_plots

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
    ### TODO: Implement this
    parser.add_argument(
        "--save_all",
        action="store_true",
        help="Save test results for every checkpoint",
    )

    args = parser.parse_args()
    model_name = args.model_name
    train_name = args.train_name
    batch_size = args.batch_size
    checkpoint = args.checkpoint
    include_stress = args.include_stress
    error_meter = classic_errors

    seed_everything()
    backend_setup()
    device = set_device()

    conditions = [
        "RSCH",
        "RSCL",
        "RSSH",
        "RSSL",
        "RLCH",
        "RLCL",
        "RLSH",
        "RLSL",
        "PSC",
        "PSS",
        "PLC",
        "PLS",
    ]
    errors_by_condition = {"epoch": []}
    for condition in conditions:
        errors_by_condition[condition] = []
    edits_by_condition = {"epoch": []}
    for condition in conditions:
        edits_by_condition[condition] = []

    weights_dir = get_weights_dir() / model_name / train_name
    checkpoints = sorted([f.stem.split(".")[-1] for f in weights_dir.glob("*.pth")])
    for checkpoint in checkpoints:
        print(f"Epoch: {checkpoint:<3}", end="\r")
        results_dir = (
            get_evaluation_dir()
            / f"{model_name}~{train_name}"
            / "epochs"
            / f"{checkpoint}"
        )
        results_dir.mkdir(exist_ok=True, parents=True)
        model = get_model(args.model_name)
        load_weights(
            model=model,
            model_name=model_name,
            train_name=train_name,
            checkpoint=checkpoint,
            device=device,
        )

        ### TESTING ###

        if args.retest or not (results_dir / "evaluation.csv").exists():
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
            test_results.to_csv(results_dir / "evaluation.csv")

        # if args.retest or not (results_dir / f"sonority.csv").exists():
        #     ssp_df = get_sonority_dataset(include_stress=include_stress)
        #     ssp_loader = get_phoneme_testloader(batch_size, include_stress, ssp_df)
        #     ssp_results, _ = test(
        #         model=model,
        #         device=device,
        #         test_df=ssp_df,
        #         test_loader=ssp_loader,
        #         include_stress=include_stress,
        #         error_meter=error_meter,
        #         verbose=args.verbose,
        #     )
        #     ssp_results.to_csv(results_dir / f"sonority.csv")

        # if args.test_train and (
        #     args.retest or not (results_dir / f"train.csv").exists()
        # ):
        #     train_df = get_train_data()
        #     train_loader = get_phoneme_testloader(batch_size, include_stress, train_df)
        #     train_results, train_error = test(
        #         model=model,
        #         device=device,
        #         test_df=train_df,
        #         test_loader=train_loader,
        #         include_stress=include_stress,
        #         error_meter=error_meter,
        #         verbose=args.verbose,
        #     )
        #     train_results.to_csv(results_dir / f"train.csv")
        #     train_results = enrich_for_plotting(train_results, include_stress)

        ### PLOTTING ###

        converters = {
            "Phonemes": literal_eval,
            "No Stress": literal_eval,
            "Prediction": literal_eval,
        }
        test_results = pd.read_csv(
            results_dir / "evaluation.csv", index_col=0, converters=converters
        )
        test_results = enrich_for_plotting(test_results, include_stress)

        # ssp_results = pd.read_csv(
        #     results_dir / "ssp.csv", index_col=0, converters=converters
        # )
        # ssp_results = enrich_for_plotting(ssp_results, include_stress)

        errors_by_condition["epoch"].append(checkpoint)
        edits_by_condition["epoch"].append(checkpoint)
        for condition in conditions:
            edit_distances = test_results.loc[
                test_results["Condition"] == condition, "Edit Distance"
            ]
            error_count = (edit_distances > 0).sum()
            mean_edits = edit_distances.mean()
            errors_by_condition[condition].append(error_count)
            edits_by_condition[condition].append(mean_edits)

    results_dir = get_evaluation_dir() / f"{model_name}~{train_name}" / "development"
    results_dir.mkdir(exist_ok=True, parents=True)
    errors_df = pd.DataFrame(errors_by_condition)
    edits_df = pd.DataFrame(edits_by_condition)
    errors_df.to_csv(results_dir / "dev_errors.csv")
    edits_df.to_csv(results_dir / "dev_edits.csv")

    figures_dir = get_figures_dir() / f"{model_name}~{train_name}" / "development"
    figures_dir.mkdir(exist_ok=True, parents=True)
    development_plots(errors_df, figures_dir, "errors")
    development_plots(edits_df, figures_dir, "edits")
