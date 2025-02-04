from itertools import product
from typing import Iterable, TypedDict

import pandas as pd

from .models import get_model_args, get_train_args
from .paths import get_gridsearch_dir, get_train_dir


def get_empty_training_log() -> pd.DataFrame:
    r"""Returns an empty DataFrame with column names set up for training logs"""
    columns = [
        "Model name",
        "Train name",
        "Model type",
        "Start token id",
        "Recurrent type",
        "Hidden size",
        "Num layers",
        "Dropout",
        "Tf ratio",
        "CNN hidden size",
        "CorNet model",
        "Batch size",
        "Learning rate",
        "Fold",
        "Include stress",
        "Epoch",
        "Train loss",
        "Valid loss",
        "Train errors",
        "Valid errors",
    ]
    df = pd.DataFrame(columns=columns)
    return df


def grid_search_log(
    train_losses: list,
    valid_losses: list,
    train_errors: list,
    valid_errors: list,
    model_name: str,
    train_name: str,
    num_epochs: int,
):
    r"""Create one csv logging the training and validation losses of each epoch
    for a given model (descibed through `model_name`) along a given training process
    (described through `train_name`).
    """
    # Initialize log
    logfile_path = get_train_dir() / f"{model_name}~{train_name}.csv"
    logfile_path.parent.mkdir(exist_ok=True, parents=True)
    log = None
    # Extract parameters from the model name
    model_args = get_model_args(model_name)
    train_args = get_train_args(train_name)
    cnn_args = model_args["cnn_args"]
    for epoch in range(num_epochs):
        row_dict = {
            "Model name": [model_name],
            "Train name": [train_name],
            "Model type": [model_args["model_class"]],
            "Start token id": [model_args["start_token_id"]],
            "Recurrent type": [model_args["recur_type"]],
            "Hidden size": [model_args["hidden_size"]],
            "Num layers": [model_args["num_layers"]],
            "Dropout": [model_args["droprate"]],
            "Tf ratio": [model_args["tf_ratio"]],
            "CNN hidden size": (
                [cnn_args] if cnn_args is None else [cnn_args["hidden_size"]]
            ),
            "CorNet model": [cnn_args] if cnn_args is None else [cnn_args["cnn_model"]],
            "Batch size": [train_args["batch_size"]],
            "Learning rate": [train_args["learning_rate"]],
            "Fold": [train_args["fold_id"]],
            "Include stress": [train_args["include_stress"]],
            "Epoch": [epoch + 1],
            "Train loss": [train_losses[epoch]],
            "Valid loss": [valid_losses[epoch]],
            "Train errors": [train_errors[epoch]],
            "Valid errors": [valid_errors[epoch]],
        }
        row_df = pd.DataFrame.from_dict(row_dict)
        if log is None:
            log = row_df
        else:
            log = pd.concat([log, row_df], ignore_index=True)

    if log is None:
        log = get_empty_training_log()
    # Save the DataFrame to a CSV file
    log.to_csv(logfile_path)


# TODO add log for tests


def grid_search_aggregate():
    r"""Aggregates all the training logs into one .csv file."""
    aggregated_file_path = get_gridsearch_dir() / "aggregated_training.csv"
    aggregated = None
    log_path = get_train_dir()
    for file in log_path.glob("*.csv"):
        log_df = pd.read_csv(file, index_col=0)
        if aggregated is None:
            aggregated = log_df
        else:
            aggregated = pd.concat([aggregated, log_df], ignore_index=False)
    if aggregated is None:
        aggregated = get_empty_training_log()
    # Save the DataFrame to a CSV file
    aggregated.to_csv(aggregated_file_path)


class Grid(
    TypedDict, total=True
):  # Wait for PEP 728 for proper typing : https://peps.python.org/pep-0728/
    model_class: Iterable[str]
    vocab_size: Iterable[int]
    start_token_id: Iterable[int]
    cnn_args: Iterable[dict | None]
    fold_id: Iterable[int | None]
    num_epochs: Iterable[int]
    batch_size: Iterable[int]
    recur_type: Iterable[str]
    hidden_size: Iterable[int]
    num_layers: Iterable[int]
    learning_rate: Iterable[float]
    droprate: Iterable[float]
    tf_ratio: Iterable[float]
    include_stress: Iterable[bool]


def grid_iter(grid: dict[str, Iterable[int | str | float | bool | None]]):
    r"""Iterate through `grid` and yields arg dicts corresponding
    to the configurations contained in the grid"""
    keys = grid.keys()
    for instance in product(*grid.values()):
        yield dict(zip(keys, instance))
