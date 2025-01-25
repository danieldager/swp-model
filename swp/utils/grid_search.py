import pandas as pd

from swp.utils.models import get_args_from_model_name, get_training_args

from .paths import get_gridsearch_dir, get_gridsearch_train_dir


def get_empty_training_log() -> pd.DataFrame:
    r"""Returns an empty DataFrame with column names set up for training logs"""
    columns = [
        "Model name",
        "Training name",
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
        "Validation loss",
    ]
    df = pd.DataFrame(columns=columns)
    return df


def grid_search_log(
    train_losses: list,
    valid_losses: list,
    model_name: str,
    training_name: str,
    num_epochs: int,
):
    r"""Create one csv logging the training and validation losses of each epoch
    for a given model (descibed through `model_name`) along a given training process
    (described through `training_name`).
    """
    # Initialize log
    logfile_path = get_gridsearch_train_dir() / f"{model_name}~{training_name}.csv"
    logfile_path.parent.mkdir(exist_ok=True, parents=True)
    log = None
    # Extract parameters from the model name
    model_type, recur_type, model_args = get_args_from_model_name(model_name)
    cnn_args = None
    if "c" in model_args:
        cnn_args = model_args["c"]
    training_args = get_training_args(training_name)
    for epoch in range(num_epochs):
        row_dict = {
            "Model name": [model_name],
            "Training name": [training_name],
            "Model type": [model_type],
            "Start token id": [model_args["s"]],
            "Recurrent type": [recur_type],
            "Hidden size": [model_args["h"]],
            "Num layers": [model_args["l"]],
            "Dropout": [model_args["d"]],
            "Tf ratio": [model_args["t"]],
            "CNN hidden size": [cnn_args] if cnn_args is None else [cnn_args["h"]],
            "CorNet model": [cnn_args] if cnn_args is None else [cnn_args["m"]],
            "Batch size": [training_args["b"]],
            "Learning rate": [training_args["l"]],
            "Fold": [training_args["f"]],
            "Include stress": [training_args["s"]],
            "Epoch": [epoch],
            "Train loss": [train_losses[epoch]],
            "Validation loss": [valid_losses[epoch]],
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
    aggregatedfile_path = get_gridsearch_dir() / "aggregated_training.csv"
    aggregated = None
    log_path = get_log_dir() / "train"
    for file in log_path.glob("*.csv"):
        log_df = pd.read_csv(file, index_col=0)
        if aggregated is None:
            aggregated = log_df
        else:
            aggregated = pd.concat([aggregated, log_df], ignore_index=True)
    if aggregated is None:
        aggregated = get_empty_training_log()
    # Save the DataFrame to a CSV file
    aggregated.to_csv(aggregatedfile_path)
