import pandas as pd

from .paths import get_result_dir


def grid_search_log(train_losses, valid_losses, model, num_epochs):
    try:
        df = pd.read_csv(get_result_dir() / "grid_search.csv")
    except FileNotFoundError:
        # Create a new DataFrame if the file doesn't exist
        print("\nCreating new grid search log")
        columns = [
            "model",
            "h_size",
            "n_layers",
            "dropout",
            "tf_ratio",
            "l_rate",
        ]
        columns += [f"T{i}" for i in range(1, num_epochs + 1)]
        columns += [f"V{i}" for i in range(1, num_epochs + 1)]
        df = pd.DataFrame(columns=columns)

    # Extract parameters from the model name
    h, l, d, t, r = [p[1:] for p in model.split("_")[1:]]
    df.loc[model] = [model, h, l, d, t, r] + train_losses + valid_losses
    print("model", model)

    # Save the DataFrame to a CSV file
    df.to_csv(get_result_dir() / "grid_search.csv", index=False)
