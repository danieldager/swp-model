import pandas as pd

from .paths import get_gridsearch_dir


def grid_search_log(train_losses, valid_losses, model, num_epochs):
    # TODO rework so that it doesn't break with simultaneous access
    logfile_path = get_gridsearch_dir() / f"{model}.csv"
    try:
        df = pd.read_csv(logfile_path)
    except FileNotFoundError:
        # Create a new DataFrame if the file doesn't exist
        print("\nCreating new grid search log")
        columns = [
            "name",
            "type",
            "fold",
            "h_size",
            "l_rate",
            "n_layers",
            "dropout",
            "tf_ratio",
        ]
        columns += [f"T{i}" for i in range(1, num_epochs + 1)]
        columns += [f"V{i}" for i in range(1, num_epochs + 1)]
        df = pd.DataFrame(columns=columns)

    # Extract parameters from the model name
    h, r, d, t, l, m, f = [p[1:] for p in model.split("_")]
    m = "rnn" if m == "n" else "lstm"
    df.loc[model] = [model, m, f, h, r, l, d, t] + train_losses + valid_losses
    print("model", model)

    # Save the DataFrame to a CSV file
    df.to_csv(logfile_path, index=False)
