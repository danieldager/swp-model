import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import argparse
import warnings

import pandas as pd
import seaborn as sns
import torch

from swp.datasets.phonemes import get_phoneme_testloader
from swp.test.ablations import ablate
from swp.utils.datasets import get_test_data
from swp.utils.models import get_model, get_model_args, get_train_args, load_weights
from swp.utils.paths import get_ablations_dir
from swp.utils.setup import seed_everything
from swp.viz.ablation import scatter_plot

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The PyTorch API of nested tensors is in prototype stage",
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*set_ticklabels\(\) should only be used with a fixed number of ticks.*",
)

sns.set_palette("colorblind")


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
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )

    args = parser.parse_args()
    model_name = args.model_name
    train_name = args.train_name
    checkpoint = args.checkpoint

    model_args = get_model_args(model_name)
    train_args = get_train_args(train_name)
    include_stress = train_args["include_stress"]
    batch_size = train_args["batch_size"]

    seed_everything()
    device = torch.device("cpu")

    # Load and prepare data
    test_data = get_test_data()
    test_loader = get_phoneme_testloader(batch_size, include_stress)

    # Load the model and weights
    model = get_model(model_name)
    load_weights(
        model=model,
        model_name=model_name,
        train_name=train_name,
        checkpoint=checkpoint,
        device=device,
    )

    # Set up directories
    ablations_dir = get_ablations_dir()
    model_dir = ablations_dir / f"{model_name}~{train_name}~{checkpoint}"
    model_dir.mkdir(exist_ok=True, parents=True)

    # Check if results_df already exists
    if not (model_dir / "ablation_results.csv").exists():
        results_df = ablate(
            model=model,
            device=device,
            test_df=test_data,
            test_loader=test_loader,
            include_stress=False,
        )
        results_df.to_csv(model_dir / "ablation_results.csv", index=False)

    results_df = pd.read_csv(model_dir / "ablation_results.csv")
    print("\nLowest accuracies:")
    for condition in [
        "real_accuracy",
        "pseudo_accuracy",
        "low_freq_accuracy",
        "high_freq_accuracy",
        "simple_accuracy",
        "complex_accuracy",
        "primacy_accuracy",
        "recency_accuracy",
        "short_accuracy",
        "long_accuracy",
    ]:
        # print the neuron indices as well
        print(
            results_df.sort_values(condition).head(3)[
                ["neuron_idx", "layer_name", condition]
            ]
        )

    # Produce scatter plots.
    scatter_plot(
        results_df,
        "real_accuracy",
        "pseudo_accuracy",
        "Real (Lexical)",
        "Pseudo (Sublexical)",
        "lex",
        model_dir,
    )

    scatter_plot(
        results_df,
        "low_freq_accuracy",
        "high_freq_accuracy",
        "Low Frequency",
        "High Frequency",
        "frq",
        model_dir,
    )

    scatter_plot(
        results_df,
        "simple_accuracy",
        "complex_accuracy",
        "Morphologically Simple",
        "Morphologically Complex",
        "mor",
        model_dir,
    )

    scatter_plot(
        results_df,
        "primacy_accuracy",
        "recency_accuracy",
        "Primacy (First Half)",
        "Recency (Second Half)",
        "pos",
        model_dir,
    )

    scatter_plot(
        results_df,
        "short_accuracy",
        "long_accuracy",
        "Short Words",
        "Long Words",
        "len",
        model_dir,
    )
