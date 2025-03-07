import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


import argparse
import warnings
from ast import literal_eval

import pandas as pd

from swp.datasets.phonemes import get_phoneme_testloader
from swp.test.activations import neural_regressions
from swp.utils.datasets import get_test_data
from swp.utils.models import get_model, load_weights
from swp.utils.paths import get_evaluation_dir, get_figures_dir, get_weights_dir
from swp.utils.setup import backend_setup, seed_everything, set_device
from swp.viz.test.regressions import activation_plots

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The PyTorch API of nested tensors is in prototype stage",
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
        default=128,
        help="Test dataloader batch size",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint to load",
    )
    # parser.add_argument(
    #     "--mode",
    #     type=str,
    #     default=None,
    #     help="Activation analysis algorithm",
    # )
    parser.add_argument(
        "--layers",
        type=str,
        default="encoder",
        help="Layers of the model to include for analysis",
    )
    parser.add_argument(
        "--include_stress",
        action="store_true",
        help="Include stress in phonemes",
    )
    parser.add_argument(
        "--include_cell",
        action="store_true",
        help="Use cell states for trajectories as well",
    )
    parser.add_argument(
        "--retest",
        action="store_true",
        help="Recompute the feature importances",
    )

    args = parser.parse_args()
    model_name = args.model_name
    train_name = args.train_name
    batch_size = args.batch_size
    checkpoint = args.checkpoint
    include_stress = args.include_stress
    # mode = args.mode
    layers = args.layers
    include_cell = args.include_cell
    cell_str = "c" if include_cell else "h"

    seed_everything()
    backend_setup()
    device = set_device()
    model = None

    weights_dir = get_weights_dir() / model_name / train_name

    if checkpoint is None:
        checkpoints = [f.stem.split(".")[-1] for f in weights_dir.glob("*.pth")]
    else:
        checkpoints = [checkpoint]

    for checkpoint in checkpoints:

        eval_dir = get_evaluation_dir() / f"{model_name}~{train_name}" / f"{checkpoint}"
        eval_dir.mkdir(exist_ok=True, parents=True)
        csv_path = eval_dir / "activations.csv"

        # if the results datasets already exist, skip testing
        if args.retest or not csv_path.exists():
            print(f"Neural Regressions")
            if model is None:
                model = get_model(model_name)
            load_weights(
                model=model,
                model_name=model_name,
                train_name=train_name,
                checkpoint=checkpoint,
                device=device,
            )

            test_df = get_test_data()
            test_loader = get_phoneme_testloader(batch_size, include_stress)
            results = neural_regressions(
                model=model,  # type: ignore
                device=device,
                test_df=test_df,
                test_loader=test_loader,
                include_cell=include_cell,
                layers=layers,
            )
            results.to_csv(csv_path)

        figures_dir = (
            get_figures_dir()
            / f"{model_name}~{train_name}"
            / f"{checkpoint}"
            / "activations"
        )
        figures_dir.mkdir(exist_ok=True)

        results = pd.read_csv(csv_path, index_col=0)
        activation_plots(results, figures_dir)
