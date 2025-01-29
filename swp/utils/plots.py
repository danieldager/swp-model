import numpy as np
import pandas as pd
import seaborn as sns
from ast import literal_eval
import matplotlib.pyplot as plt

from collections import Counter
from Levenshtein import editops

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from .paths import get_figures_dir, get_stimuli_dir

sns.set_palette("colorblind")


def enrich_for_plotting(
    df: pd.DataFrame, phoneme_to_id: dict, include_stress: bool
) -> pd.DataFrame:
    """
    Calculate error and bigram statistics for each row in the DataFrame and append them as new columns.

    Args:
        df (pd.DataFrame): The input DataFrame with columns for phonemes, predictions, and optionally stress.
        phoneme_to_id (dict): A mapping of phonemes to their corresponding IDs.
        include_stress (bool): Whether to use the "Phonemes" column or "No Stress" column.

    Returns:
        pd.DataFrame: The DataFrame with additional columns for error statistics.
    """
    phoneme_key = "Phonemes" if include_stress else "No Stress"
    df[phoneme_key] = df[phoneme_key].apply(literal_eval)
    df["Prediction"] = df["Prediction"].apply(literal_eval)
    df = df[df[phoneme_key].apply(len) > 1].copy()

    # Initialize lists to store results
    edit_distances = []
    insertions = []
    deletions = []
    substitutions = []
    sequence_lengths = []
    error_indices = []
    bigram_frequency = []

    stress = "sw" if include_stress else "sn"
    stats_dir = get_stimuli_dir() / "statistics"
    bigram_stats_df = pd.read_csv(stats_dir / f"bigram_stats_{stress}.csv")
    bigram_to_freq = dict(
        zip(bigram_stats_df["Bigram"], bigram_stats_df["Normalized Frequency"])
    )

    for _, row in df.iterrows():
        # Compute average bigram frequency for the sequence
        phonemes = row[phoneme_key]
        bigrams = [" ".join(phonemes[i : i + 2]) for i in range(len(phonemes) - 1)]
        bigram_freqs = [bigram_to_freq.get(bigram, 0) for bigram in bigrams]
        avg_bigram_freq = sum(bigram_freqs) / len(bigram_freqs)

        # Tally edit operations and identify error indices
        phonemes = [phoneme_to_id[p] for p in phonemes]
        prediction = [phoneme_to_id[p] for p in row["Prediction"]]
        errors = editops(phonemes, prediction)
        counts = Counter(op for op, _, _ in errors)
        mismatched_indices = [
            i + 1 for i, (j, k) in enumerate(zip(phonemes, prediction)) if j != k
        ]

        # Append results to the respective lists
        edit_distances.append(len(errors))
        insertions.append(counts["insert"])
        deletions.append(counts["delete"])
        substitutions.append(counts["replace"])
        sequence_lengths.append(len(phonemes))
        error_indices.append(mismatched_indices)
        bigram_frequency.append(avg_bigram_freq)

    # Add results as new columns to the DataFrame
    df["Edit Distance"] = edit_distances
    df["Insertions"] = insertions
    df["Deletions"] = deletions
    df["Substitutions"] = substitutions
    df["Sequence Length"] = sequence_lengths
    df["Error Indices"] = error_indices
    df["Bigram Frequency"] = bigram_frequency

    return df


# Plot the training and validation loss curves
def training_curves(train_losses: list, valid_losses: list, model: str, n_epochs: int):
    # Extract parameters from the model name
    h, r, d, t, l, m, f = [p[1:] for p in model.split("_")]
    m = "RNN" if m[0] == "n" else "LSTM"

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(1, n_epochs + 1), y=train_losses, label="Training")
    sns.lineplot(x=range(1, n_epochs + 1), y=valid_losses, label="Validation")
    plt.title(f"{m} (fold {f}): H={h}, LR={r}, L={l}, D={d}, TF={t}, LR={r}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.legend()
    plt.tight_layout()

    MODEL_FIGURES_DIR = get_figures_dir() / model
    MODEL_FIGURES_DIR.mkdir(exist_ok=True)
    filename = MODEL_FIGURES_DIR / "training.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


# Function to plot Edit Distance by Length
def plot_errors_by_length(df, ax=None):
    """Plot average edit distance by sequence length.

    Parameters:
        df (pd.DataFrame): Data containing 'Sequence Length', 'Lexicality',
            'Morphology', and 'Edit Distance'.
        ax (matplotlib.axes.Axes, optional): Axes object to draw the plot onto.
            If None, a new figure and axes are created.

    Returns:
        None
    """
    data = df.copy()
    data = data[data["Sequence Length"] != 10]
    grouped_df = (
        data.groupby(
            [
                "Sequence Length",
                "Lexicality",
                "Morphology",
            ],
            observed=True,
        )["Edit Distance"]
        .mean()
        .reset_index()
    )

    sns.lineplot(
        data=grouped_df,
        x="Sequence Length",
        y="Edit Distance",
        hue="Lexicality",
        style="Morphology",
        marker="o",
        markersize=8,
        linewidth=3,
        ax=ax,
    )

    if ax:
        ax.set_title("Edit Distance by Sequence Length", fontsize=14)
        ax.set_xlabel("Phoneme Sequence Length", fontsize=13)
        ax.set_ylabel("Average Edit Distance", fontsize=13)
        ax.legend(title="Lexicality & Morphology")
        ax.grid(True)

    else:
        plt.title("Edit Distance by Sequence Length", fontsize=14)
        plt.xlabel("Phoneme Sequence Length", fontsize=13)
        plt.ylabel("Average Edit Distance", fontsize=13)
        plt.legend(title="Lexicality & Morphology")
        plt.grid(True)
        plt.tight_layout()


# Function to plot Frequency vs Edit Distance
def plot_errors_by_frequency(df, ax=None):
    """Plot average edit distance by word frequency.

    Parameters:
        df (pd.DataFrame): Data containing 'Lexicality', 'Zipf Frequency', 'Size',
            'Morphology', and 'Edit Distance'.
        ax (matplotlib.axes.Axes, optional): Axes object to draw the plot onto.
            If None, a new figure and axes are created.

    Returns:
        None
    """
    data = df[df["Lexicality"].isin(["real", "pseudo"])].copy()
    data["Zipf Bin"] = pd.cut(
        data["Zipf Frequency"], bins=[1, 2, 3, 4, 5, 6, 7], right=False
    )

    grouped_df = (
        data.groupby(["Zipf Bin", "Size", "Morphology"], observed=True)["Edit Distance"]
        .mean()
        .reset_index()
    )
    grouped_df["Zipf Bin"] = grouped_df["Zipf Bin"].astype(str)

    sns.lineplot(
        data=grouped_df,
        x="Zipf Bin",
        y="Edit Distance",
        hue="Size",
        style="Morphology",
        marker="o",
        markersize=8,
        linewidth=3,
        ax=ax,
    )

    if ax:
        ax.set_title("Average Edit Distance by Word Frequency")
        ax.set_xlabel("Zipf Frequency")
        ax.set_ylabel("Average Edit Distance")
        ax.legend(title="Size & Morphology")
        ax.grid(True)

    else:
        plt.title("Average Edit Distance by Word Frequency")
        plt.xlabel("Zipf Frequency")
        plt.ylabel("Average Edit Distance")
        plt.legend(title="Size & Morphology")
        plt.grid(True)
        plt.tight_layout()


# Function to plot Errors by Test Category
def plot_errors_by_category(df, ax=None):
    """Plot average errors (deletions, insertions, substitutions) by word category.

    Parameters:
        df (pd.DataFrame): Data containing 'Lexicality', 'Size', 'Frequency',
            'Morphology', 'Deletions', 'Insertions', and 'Substitutions'.
        ax (matplotlib.axes.Axes, optional): Axes object to draw the plot onto.
            If None, a new figure and axes are created.

    Returns:
        None
    """
    data = df.copy()

    data["Category"] = data.apply(
        lambda row: (
            f"pseudo {row['Size']} {row['Morphology']}"
            if row["Lexicality"] == "pseudo"
            else f"real {row['Size']} {row['Frequency']} {row['Morphology']}"
        ),
        axis=1,
    )
    grouped = (
        data.groupby("Category")[["Deletions", "Insertions", "Substitutions"]]
        .mean()
        .reset_index()
    )
    melted = pd.melt(
        grouped,
        id_vars=["Category"],
        value_vars=["Deletions", "Insertions", "Substitutions"],
    )

    # Sort the categories in a meaningful order
    real_categories = [
        f"real {size} {freq} {morph}"
        for size in ["short", "long"]
        for freq in ["high", "low"]
        for morph in ["simple", "complex"]
    ]
    pseudo_categories = [
        f"pseudo {size} {morph}"
        for size in ["short", "long"]
        for morph in ["simple", "complex"]
    ]
    order = real_categories + pseudo_categories

    # Simplify category labels
    short_order = ["".join(word[0].upper() for word in cat.split()) for cat in order]

    sns.barplot(
        x="Category",
        y="value",
        hue="variable",
        order=order,
        data=melted,
        ax=ax,
    )

    if ax:
        ax.set_title("Average Error Rates by Word Category", fontsize=14)
        ax.set_xlabel("Error Category", fontsize=13)
        ax.set_ylabel("Average Error Count", fontsize=13)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(short_order)
        ax.legend(title="Category")
        ax.grid(True)

    else:
        plt.title("Average Error Rates by Word Category", fontsize=14)
        plt.xlabel("Error Category", fontsize=13)
        plt.ylabel("Average Error Count", fontsize=13)
        plt.xticks(range(len(order)), short_order)
        plt.legend(title="Category")
        plt.grid(True)
        plt.tight_layout()


# Function to plot Average Edit Distance by Position
def plot_errors_by_position(df, ax=None):
    """Plot average edit distance by relative position within each sequence.

    Parameters:
        df (pd.DataFrame): Data containing 'Lexicality', 'Sequence Length',
            and 'Error Indices'.
        ax (matplotlib.axes.Axes, optional): Axes object to draw the plot onto.
            If None, a new figure and axes are created.

    Returns:
        None
    """
    data_by_lexicality = []

    # Iterate through rows grouped by Lexicality
    for lexicality, group_df in df.groupby("Lexicality"):
        totals = {}
        errors = {}

        for _, row in group_df.iterrows():
            length = row["Sequence Length"]

            # Count total occurrences and errors by normalized position
            for index in range(1, length + 1):
                normalized = (index - 1) / (length - 1)
                totals[normalized] = totals.get(normalized, 0) + 1

            for index in row["Error Indices"]:
                normalized = (index - 1) / (length - 1)
                errors[normalized] = errors.get(normalized, 0) + 1

        # Create data entries for the current lexicality
        data_by_lexicality.extend(
            [
                {
                    "Position": index,
                    "Error Rate": errors.get(index, 0) / total,
                    "Lexicality": lexicality,
                }
                for index, total in totals.items()
            ]
        )

    # Convert to DataFrame
    plot_df = pd.DataFrame(data_by_lexicality)

    # Plot using Seaborn with hue for Lexicality
    sns.lineplot(
        x="Position",
        y="Error Rate",
        hue="Lexicality",
        data=plot_df,
        marker="o",
        markersize=8,
        linewidth=3,
        ax=ax,
    )

    # Set axis labels and grid
    if ax:
        ax.set_title("Average Edit Distance by Relative Position", fontsize=14)
        ax.set_xlabel("Relative Position", fontsize=13)
        ax.set_ylabel("Average Edit Distance", fontsize=13)
        ax.grid(True)
    else:
        plt.title("Average Edit Distance by Relative Position", fontsize=14)
        plt.xlabel("Relative Position", fontsize=13)
        plt.ylabel("Average Edit Distance", fontsize=13)
        plt.grid(True)
        plt.tight_layout()


def plot_errors_by_sonority(df, ax=None):
    """Plot average edit distance grouped by sonority.

    Parameters:
        df (pd.DataFrame): Data containing 'Sonority', 'Type', and 'Edit Distance'.
        ax (matplotlib.axes.Axes, optional): Axes object to draw the plot onto.
            If None, a new figure and axes are created.

    Returns:
        None
    """
    data = df.copy()
    grouped_df = (
        data.groupby(["Sonority", "Type"], observed=True)["Edit Distance"]
        .mean()
        .reset_index()
    )

    sns.lineplot(
        data=grouped_df,
        x="Sonority",
        y="Edit Distance",
        hue="Type",
        marker="o",
        markersize=8,
        linewidth=3,
        ax=ax,
    )

    if ax is not None:
        ax.set_title("Average Edit Distance by Sonority", fontsize=14)
        ax.set_xlabel("Sonority Gradient", fontsize=13)
        ax.set_ylabel("Average Edit Distance", fontsize=13)
        ax.legend(title="CCV or VCC")
        ax.grid(True)

    else:
        plt.title("Average Edit Distance by Sonority", fontsize=14)
        plt.xlabel("Sonority Gradient", fontsize=13)
        plt.ylabel("Average Edit Distance", fontsize=13)
        plt.legend(title="Order")
        plt.grid(True)
        plt.tight_layout()


# TODO Average across folds
def error_plots(
    test_df: pd.DataFrame,
    sonority_df: pd.DataFrame,
    model_name: str,
    train_name: str,
    checkpoint: str,
) -> None:
    """Create multiple error plots (length, category, position, sonority) in a single figure.

    Parameters:
        test_df (pd.DataFrame): Data containing the necessary columns for length,
            category, and position plots.
        sonority_df (pd.DataFrame): Data containing the necessary columns for the sonority plot.
        model_name (str): Name of the model used (e.g., "RNN", "LSTM").
        train_name (str): Name of the training configuration.
        checkpoint (str): Identifier for the current checkpoint/epoch.

    Returns:
        None
    """
    m, h, l, v, d, t, s = [p[1:] for p in model_name.split("_")[1:]]
    b, r, f, ss = [p[1:] for p in train_name.split("_")]
    m = "LSTM" if m[0] == "S" else "RNN"
    title = f"{m}: E={checkpoint} H={h}, L={l}, D={d}, TF={t}, LR={r} V={v} F={f}"
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    axes = axes.flatten()
    plot_errors_by_length(test_df, axes[0])
    plot_errors_by_category(test_df, axes[1])
    plot_errors_by_position(test_df, axes[2])
    plot_errors_by_sonority(sonority_df, axes[3])
    fig.suptitle(title, fontsize=16, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    figures_dir = get_figures_dir() / f"{model_name}~{train_name}"
    figures_dir.mkdir(exist_ok=True)
    plt.savefig(figures_dir / f"{checkpoint}.png", dpi=300, bbox_inches="tight")
    plt.close()


def regression_plots(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    model_name: str,
    train_name: str,
    checkpoint: str,
) -> None:
    """Perform regression analysis on test data and plot feature importance and correlation matrix.

    Parameters:
        test_df (pd.DataFrame): Data containing features ('Lexicality', 'Zipf Frequency',
            'Morphology', 'Sequence Length', 'Bigram Frequency') and target ('Edit Distance').
        model_name (str): Name of the model (used in output/logging).
        train_name (str): Training configuration name (used in output/logging).
        checkpoint (str): Identifier for the current checkpoint/epoch.

    Returns:
        None
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    real_df = test_df[test_df["Lexicality"] == "real"].copy()
    pseudo_df = test_df[test_df["Lexicality"] == "pseudo"].copy()

    for i, df in enumerate([real_df, pseudo_df]):
        print("\n")
        categorical_features = ["Morphology"]
        continuous_features = ["Sequence Length", "Bigram Frequency"]
        if i == 0:
            continuous_features.append("Zipf Frequency")

        X = df[categorical_features + continuous_features]
        y = df["Edit Distance"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(),
                    categorical_features,
                ),  # One-hot encode
                (
                    "num",
                    StandardScaler(),
                    continuous_features,
                ),  # Standardize
            ],
            remainder="drop",
        )

        # Linear Regression
        pipeline = Pipeline(
            [("preprocessor", preprocessor), ("regressor", LinearRegression())]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R-squared: {r2:.4f}")

        # Feature importance
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        feature_names = np.delete(feature_names, 1)
        name_map = {}
        for fn in feature_names:
            if "Sequence Length" in fn:
                name_map[fn] = "Length"
            elif "Bigram Frequency" in fn:
                name_map[fn] = "Bigram"
            elif "Zipf Frequency" in fn:
                name_map[fn] = "Frequency"
            elif "Morphology" in fn:
                name_map[fn] = "Morphology"
        feature_names = [name_map[fn] for fn in feature_names]
        coefficients = pipeline.named_steps["regressor"].coef_
        coefficients = np.delete(coefficients, 1)
        feature_importance = pd.DataFrame(
            {"Feature": feature_names, "Coefficient": coefficients}
        ).sort_values(by="Coefficient", ascending=False)
        print(feature_importance)

        # Correlation matrix
        X_train_processed = pipeline.named_steps["preprocessor"].transform(X_train)
        X_train_processed = np.delete(X_train_processed, 1, axis=1)
        X_train_processed_df = pd.DataFrame(
            X_train_processed,
            index=X_train.index,
            columns=feature_names,
        )
        corr_matrix = X_train_processed_df.corr()
        print(corr_matrix)

        # Plotting
        feature_importance_sorted = feature_importance.reindex(
            feature_importance["Coefficient"].abs().sort_values(ascending=True).index
        )
        sns.barplot(
            x="Coefficient",
            y="Feature",
            hue="Feature",
            data=feature_importance_sorted,
            ax=axes[0, i],
            orient="h",
            palette="Blues",
            legend=False,
        )
        axes[0, i].axvline(0, color="black", linestyle="--", linewidth=1)
        axes[0, i].set_title(f"Feature Importance ({'Real' if i == 0 else 'Pseudo'})")
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            ax=axes[1, i],
            cmap="Blues",
            cbar=False,
        )
        axes[1, i].set_title("Feature Correlation Matrix")

    # Parse model parameters for title
    m, h, l, v, d, t, s = [p[1:] for p in model_name.split("_")[1:]]
    b, r, f, ss = [p[1:] for p in train_name.split("_")]
    m = "LSTM" if m[0] == "S" else "RNN"
    title = f"{m}: E={checkpoint} H={h}, L={l}, D={d}, TF={t}, LR={r} V={v} F={f}"
    fig.suptitle(title, fontsize=16, y=0.95)

    figures_dir = get_figures_dir() / f"{model_name}~{train_name}"
    figures_dir.mkdir(exist_ok=True)
    plt.savefig(figures_dir / f"{checkpoint}_reg.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("\n")


# Plot the confusion matrix for the test data
def confusion_matrix(confusions: dict, model_name: str, epoch: str) -> None:
    # TODO Daniel docstring
    df = pd.DataFrame.from_dict(confusions, orient="index")

    # TODO get_phoneme_to_id
    phonemes = None

    # Normalize the confusion matrix
    # df = np.log1p(df) # log scale
    # df = df.div(df.sum(axis=1), axis=0) # row normal
    # df = (df - df.min().min()) / (df.max().max() - df.min().min()) # min max
    df = (df - df.mean()) / df.std()  # z score

    # Create a confusion matrix
    plt.figure(figsize=(8, 7))

    # Plot the heatmap with enhanced aesthetics
    heatmap = sns.heatmap(
        df,
        annot=False,
        cmap="Blues",
        square=True,
        cbar_kws={"label": "Counts"},
        xticklabels=True,
        yticklabels=True,
    )

    # Adjust the colorbar
    cbar = heatmap.collections[0].colorbar
    cbar.ax.set_aspect(10)  # Larger values make the colorbar narrower

    # Display X-ticks on the top
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position("top")

    # Set axis labels and title
    plt.title("Confusion Matrix", fontsize=14, pad=20)
    plt.xlabel("Prediction", fontsize=10)
    plt.ylabel("Ground Truth", fontsize=10)
    plt.xticks(fontsize=5, rotation=90)
    plt.yticks(fontsize=5, rotation=0)
    plt.tight_layout()

    MODEL_FIGURES_DIR = get_figures_dir() / model_name
    MODEL_FIGURES_DIR.mkdir(exist_ok=True)

    filename = f"confusion{epoch}.png"
    plt.savefig(MODEL_FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
