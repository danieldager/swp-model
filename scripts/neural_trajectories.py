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
from swp.test.trajectories import trajectories
from swp.utils.datasets import get_test_data
from swp.utils.models import get_model, load_weights
from swp.utils.paths import get_figures_dir, get_test_dir, get_weights_dir
from swp.utils.setup import backend_setup, seed_everything, set_device
from swp.viz.trajectory import plot_trajectories, sns_plot_trajectories

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
        default=1,
        help="Test dataloader batch size",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint to load",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Activation analysis algorithm",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Layers of the model to include for analysis",
    )
    parser.add_argument(
        "--include_stress",
        action="store_true",
        help="Include stress in phonemes",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots",
    )
    parser.add_argument(
        "--include_cell",
        action="store_true",
        help="Use cell states for trajectories as well",
    )
    parser.add_argument(
        "--include_start",
        action="store_true",
        help="Add zero position at the beginning of the trajectories",
    )

    args = parser.parse_args()
    model_name = args.model_name
    train_name = args.train_name
    batch_size = args.batch_size
    checkpoint = args.checkpoint
    include_stress = args.include_stress
    mode = args.mode
    layers = args.layers
    include_cell = args.include_cell
    cell_str = "c" if include_cell else "h"
    include_start = args.include_start
    start_str = "s" if include_start else "n"

    seed_everything()
    backend_setup()
    device = set_device()
    model = None

    # TODO think about saving directory
    test_dir = get_test_dir()
    model_dir = test_dir / f"{model_name}~{train_name}"
    model_dir.mkdir(exist_ok=True, parents=True)
    weights_dir = get_weights_dir() / model_name / train_name

    if checkpoint is None:
        checkpoints = [f.stem.split(".")[-1] for f in weights_dir.glob("*.pth")]
    else:
        checkpoints = [checkpoint]

    for checkpoint in checkpoints:
        traj_results = None
        csv_name = (
            f"trajectories_{checkpoint}_{layers}_{mode}_{cell_str}{start_str}.csv"
        )
        if not (model_dir / csv_name).exists():
            if model is None:
                model = get_model(model_name)
            load_weights(
                model=model,
                model_name=model_name,
                train_name=train_name,
                checkpoint=checkpoint,
                device=device,
            )

            # if the results datasets already exist, skip testing

            test_df = get_test_data()
            test_loader = get_phoneme_testloader(batch_size, include_stress)
            traj_results = trajectories(
                model=model,  # type: ignore
                device=device,
                test_df=test_df,
                test_loader=test_loader,
                mode=mode,
                include_cell=include_cell,
                include_start=include_start,
                layers=layers,
            )
            traj_results.to_csv(model_dir / csv_name)

        if args.plot:

            figures_dir = get_figures_dir() / f"{model_name}~{train_name}"
            figures_dir.mkdir(exist_ok=True)

            if traj_results is None:
                converters = {
                    "Phonemes": literal_eval,
                    "No Stress": literal_eval,
                    "Prediction": literal_eval,
                    "Trajectory": literal_eval,
                }

                traj_results = pd.read_csv(
                    model_dir / csv_name,
                    index_col=0,
                    converters=converters,
                )

            image_name = f"{mode}_{checkpoint}_{layers}_{cell_str}{start_str}_traj.png"

            plot_trajectories(
                traj_results,
                figures_dir,
                filename=image_name,
            )

            sns_plot_trajectories(
                traj_results,
                figures_dir,
                filename=f"sns_{image_name}",
            )
