import math
import math
import pathlib
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*set_ticklabels\(\) should only be used with a fixed number of ticks.*",
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .paths import get_figures_dir

sns.set_palette("colorblind")


# Plot the training and validation loss curves
def training_curves(train_losses: list, valid_losses: list, model: str, n_epochs: int):
    # Extract parameters from the model name
    h, r, d, t, l, m, f = [p[1:] for p in model.split("_")]
    m = "RNN" if m[0] == "n" else "LSTM"

    plt.figure(figsize=(12, 6))
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
    plt.savefig(filename, dpi=300)  # , bbox_inches="tight")
    plt.close()


def set_edge_ticks(ax, tick_fontsize=22, x_decimal_places=2, y_decimal_places=2):
    # Force the figure to render so we get the final limits.
    plt.draw()
    # Retrieve current x- and y-axis limits.
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # round x limits
    x_min = int(round(x_min))
    x_max = int(round(x_max))

    # Set x-axis ticks to exactly the minimum and maximum values.
    ax.set_xticks([x_min, x_max])
    ax.set_xticklabels(
        [f"{x_min:.{x_decimal_places}f}", f"{x_max:.{x_decimal_places}f}"],
        fontsize=tick_fontsize,
    )

    # For the y-axis, lower limit will be set a bit below zero for clarity.
    new_y_min = -0.05 * y_max  # Adjust the factor as needed.
    ax.set_ylim(new_y_min, y_max)
    ax.set_yticks([0, y_max])
    ax.set_yticklabels(["0", f"{y_max:.{y_decimal_places}f}"], fontsize=tick_fontsize)

    # use linspace to get a range of values between the min and max
    x_lines = np.linspace(x_min, x_max, 6)
    y_lines = np.linspace(new_y_min, y_max, 6)

    # Draw grid lines
    for x in x_lines:
        ax.axvline(x, color="gray", linewidth=0.5, linestyle="--")
    for y in y_lines:
        ax.axhline(y, color="gray", linewidth=0.5, linestyle="--")


# Function to plot Edit Distance by Length
def plot_length_errors(df, checkpoint: str, dir: pathlib.Path):
    """Plot average edit distance by sequence length.

    Parameters:
        df (pd.DataFrame): Data containing 'Sequence Length', 'Lexicality',
            'Morphology', and 'Edit Distance'.
        ax (matplotlib.axes.Axes, optional): Axes object to draw the plot onto.
            If None, a new figure and axes are created.

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
    plt.figure(figsize=(11, 6))
    ax = sns.lineplot(
        data=grouped_df,
        x="Sequence Length",
        y="Edit Distance",
        hue="Lexicality",
        style="Morphology",
        marker="o",
        markersize=8,
        linewidth=3,
    )
    plt.xlabel("Sequence Length", fontsize=24, labelpad=-10)
    plt.ylabel("Edit Distance", fontsize=24, labelpad=-40)
    plt.xlabel("Sequence Length", fontsize=24, labelpad=-10)
    plt.ylabel("Edit Distance", fontsize=24, labelpad=-40)
    handles, labels = ax.get_legend_handles_labels()
    filtered_handles = []
    filtered_labels = []
    for h, l in zip(handles, labels):
        if l not in ["Lexicality", "Morphology"]:
            filtered_handles.append(h)
            filtered_labels.append(l)
    leg = plt.legend(
        filtered_handles,
        filtered_labels,
        title="Lexicality & Morphology",
        fontsize=22,
        title_fontsize=22,
        fontsize=22,
        title_fontsize=22,
        ncol=2,
    )
    plt.setp(leg.get_title(), multialignment="left")
    set_edge_ticks(ax, tick_fontsize=22)
    plt.savefig(dir / f"{checkpoint}~len_errors.png", dpi=300)
    plt.setp(leg.get_title(), multialignment="left")
    set_edge_ticks(ax, tick_fontsize=22)
    plt.savefig(dir / f"{checkpoint}~len_errors.png", dpi=300)
    plt.close()


# Function to plot Average Edit Distance by Position
def plot_position_errors(df, checkpoint: str, dir: pathlib.Path):
    """Plot average edit distance by relative position within each sequence.

    Parameters:
        df (pd.DataFrame): Data containing 'Lexicality', 'Sequence Length',
            and 'Error Indices'.
        ax (matplotlib.axes.Axes, optional): Axes object to draw the plot onto.
            If None, a new figure and axes are created.

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
    plot_df = pd.DataFrame(data_by_lexicality)
    plt.figure(figsize=(11, 6))
    ax = sns.lineplot(
    ax = sns.lineplot(
        x="Position",
        y="Error Rate",
        hue="Lexicality",
        data=plot_df,
        marker="o",
        markersize=8,
        linewidth=3,
    )
    plt.xlabel("Relative Position", fontsize=24, labelpad=-10)
    plt.ylabel("Error Rate", fontsize=24, labelpad=-40)
    plt.xlabel("Relative Position", fontsize=24, labelpad=-10)
    plt.ylabel("Error Rate", fontsize=24, labelpad=-40)
    plt.legend(title="Lexicality", fontsize=24, title_fontsize=24)
    set_edge_ticks(ax, tick_fontsize=22, x_decimal_places=1, y_decimal_places=2)
    set_edge_ticks(ax, tick_fontsize=22, x_decimal_places=1, y_decimal_places=2)
    plt.savefig(dir / f"{checkpoint}~pos_errors.png", dpi=300)  # , bbox_inches="tight")
    plt.close()


def plot_sonority_errors(df, checkpoint: str, dir: pathlib.Path):
    """Plot average edit distance grouped by sonority.

    Parameters:
        df (pd.DataFrame): Data containing 'Sonority', 'Type', and 'Edit Distance'.
        ax (matplotlib.axes.Axes, optional): Axes object to draw the plot onto.
            If None, a new figure and axes are created.

    """
    data = df.copy()
    grouped_df = (
        data.groupby(["Sonority", "Type"], observed=True)["Edit Distance"]
        .mean()
        .reset_index()
    )
    plt.figure(figsize=(11, 6))
    ax = sns.lineplot(
    ax = sns.lineplot(
        data=grouped_df,
        x="Sonority",
        y="Edit Distance",
        hue="Type",
        marker="o",
        markersize=8,
        linewidth=3,
    )
    plt.xlabel("Sonority Gradient", fontsize=24, labelpad=-10)
    plt.ylabel("Edit Distance", fontsize=24, labelpad=-35)
    plt.xlabel("Sonority Gradient", fontsize=24, labelpad=-10)
    plt.ylabel("Edit Distance", fontsize=24, labelpad=-35)
    plt.legend(title="CCV or VCC", fontsize=24, title_fontsize=24)
    set_edge_ticks(ax, tick_fontsize=22, x_decimal_places=0, y_decimal_places=2)
    set_edge_ticks(ax, tick_fontsize=22, x_decimal_places=0, y_decimal_places=2)
    plt.savefig(dir / f"{checkpoint}~son_errors.png", dpi=300)  # , bbox_inches="tight")
    plt.close()


def plot_category_errors(df, checkpoint: str, dir: pathlib.Path):
    """Plot average errors (deletions, insertions, substitutions) by word category.

    Parameters:
        df (pd.DataFrame): Data containing 'Lexicality', 'Size', 'Frequency',
            'Morphology', 'Deletions', 'Insertions', and 'Substitutions'.
        ax (matplotlib.axes.Axes, optional): Axes object to draw the plot onto.
            If None, a new figure and axes are created.

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
    short_order = ["".join(word[0].upper() for word in cat.split()) for cat in order]

    plt.figure(figsize=(11, 6))
    sns.barplot(
        x="Category",
        y="value",
        hue="variable",
        order=order,
        data=melted,
    )
    plt.xlabel("Error Category", fontsize=14, labelpad=10)
    plt.ylabel("Average Error Count", fontsize=14, labelpad=10)
    plt.xticks(range(len(order)), short_order)
    plt.legend(title="Category", fontsize=12, title_fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.savefig(dir / f"{checkpoint}~cat_errors.png", dpi=300)  # , bbox_inches="tight")
    plt.close()


def regression_plots(
    df: pd.DataFrame,
    checkpoint: str,
    filepath: pathlib.Path,
    plot_num: int,
) -> None:
    """Perform regression analysis on test data, plot feature importance and correlation matrix.

    Parameters:
        df (pd.DataFrame): Dataframe containing features ('Lexicality', 'Zipf Frequency',
            'Morphology', 'Sequence Length', 'Bigram Frequency') and target ('Edit Distance').
        model_name (str): Name of the model (used in output/logging).
        train_name (str): Training configuration name (used in output/logging).
        checkpoint (str): Identifier for the current checkpoint/epoch.
        filepath (pathlib.Path): Directory path where plots will be saved.
    """
    # Copy the DataFrame and adjust Zipf Frequency for 'pseudo' lexicality.
    df = df.copy()
    df.loc[df["Lexicality"] == "pseudo", "Zipf Frequency"] = 0

    # Define features
    categorical_features = (
        ["Morphology", "Lexicality"] if plot_num == 1 else ["Morphology"]
    )
    continuous_features = (
        ["Sequence Length"] if plot_num == 1 else ["Sequence Length", "Zipf Frequency"]
    )

    X = df[categorical_features + continuous_features]
    y = df["Edit Distance"]

    # Split data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocessing: one-hot encode (dropping first) and standardize.
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first"), categorical_features),
            ("num", StandardScaler(), continuous_features),
        ]
    )

    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())]
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r_value = np.corrcoef(y_test, y_pred)[0, 1]
    print(f"\nMSE: {mse:.4f}, r: {r_value:.4f}\n")

    # --- Feature Importance Plot ---

    # Get raw feature names and simplify them using a dictionary mapping.
    raw_feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    mapping = {
        "Sequence Length": "Length",
        "Bigram Frequency": "Bigram",
        "Zipf Frequency": "Frequency",
        "Morphology": "Morphology",
        "Lexicality": "Lexicality",
    }
    mapped_feature_names = [
        next((v for k, v in mapping.items() if k in fn), fn) for fn in raw_feature_names
    ]

    # Retrieve coefficients and prepare the feature importance DataFrame
    coefficients = pipeline.named_steps["regressor"].coef_
    feature_importance = pd.DataFrame(
        {
            "Feature": mapped_feature_names,
            "Coefficient": coefficients,
        }
    )
    # Use absolute values for bar lengths
    feature_importance["Importance"] = feature_importance["Coefficient"].abs()
    # Create a new column for sign
    feature_importance["Sign"] = feature_importance["Coefficient"].apply(
        lambda x: "Positive" if x >= 0 else "Negative"
    )
    # Sort so that the smallest importance is at the top
    feature_importance_sorted = feature_importance.sort_values(
        by="Importance", ascending=True
    )

    plt.rcParams.update({"font.size": 24})
    fig1, ax1 = plt.subplots(figsize=(11, 6))
    custom_palette = {"Positive": "#5bc25f", "Negative": "#f54e4e"}
    sns.barplot(
        x="Importance",
        y="Feature",
        hue="Sign",
        data=feature_importance_sorted,
        orient="h",
        palette=custom_palette,
        ax=ax1,
    )
    ax1.axvline(0, color="black", linestyle="--", linewidth=1)
    ax1.set_xlabel("Feature Importance")
    ax1.set_ylabel("")
    ax1.legend(title=None)

    ax = plt.gca()
    # xticks = ax.get_xticks()
    # new_xticklabels = [
    #     f"{tick:.2f}" if (i == 0 or i == len(xticks) - 2) else ""
    #     for i, tick in enumerate(xticks)
    # ]
    # ax.set_xticklabels(new_xticklabels, fontsize=22)

    xticks = ax.get_xticks()
    new_xticklabels = [
        (
            f"{tick:.0f}"
            if i == 0
            else (format(tick, ".3g") if i == len(xticks) - 2 else "")
        )
        for i, tick in enumerate(xticks)
    ]
    ax.set_xticklabels(new_xticklabels, fontsize=22)
    filename = f"{checkpoint}~fimport{plot_num}.png"
    fig1.savefig(filepath / filename, dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # --- Correlation Matrix Plot ---

    # Compute the correlation matrix on the processed training data.
    X_train_processed = pipeline.named_steps["preprocessor"].transform(X_train)
    X_train_processed_df = pd.DataFrame(
        X_train_processed, columns=mapped_feature_names, index=X_train.index
    )
    corr_matrix = X_train_processed_df.corr(method="pearson")

    # Create a separate figure for the correlation matrix.
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        ax=ax2,
        cbar=False,
    )
    ax2.tick_params(axis="both", labelsize=16)
    filename = f"{checkpoint}~cmatrix{plot_num}.png"
    fig2.savefig(filepath / filename, dpi=300)  # , bbox_inches="tight")
    plt.close(fig2)


# Plot the confusion matrix for the test data
# def confusion_matrix(confusions: dict, model_name: str, epoch: str) -> None:
#     # TODO Daniel docstring

#     # Initialize the confusion matrix
#     # confusions = {}
#     # for t in phoneme_stats.keys():
#     #     confusions[t] = {p: 0 for p in phoneme_stats.keys()}

#     # Tabulate confusion between prediction and target
#     # if len(target) == len(prediction):
#     #     for t, p in zip(target, prediction):
#     #         confusions[t][p] += 1
#     df = pd.DataFrame.from_dict(confusions, orient="index")

#     # TODO get_phoneme_to_id
#     phonemes = None

#     # Normalize the confusion matrix
#     # df = np.log1p(df) # log scale
#     # df = df.div(df.sum(axis=1), axis=0) # row normal
#     # df = (df - df.min().min()) / (df.max().max() - df.min().min()) # min max
#     df = (df - df.mean()) / df.std()  # z score

#     # Create a confusion matrix
#     plt.figure(figsize=(8, 7))

#     # Plot the heatmap with enhanced aesthetics
#     heatmap = sns.heatmap(
#         df,
#         annot=False,
#         cmap="Blues",
#         square=True,
#         cbar_kws={"label": "Counts"},
#         xticklabels=True,
#         yticklabels=True,
#     )

#     # Adjust the colorbar
#     cbar = heatmap.collections[0].colorbar
#     cbar.ax.set_aspect(10)  # Larger values make the colorbar narrower

#     # Display X-ticks on the top
#     plt.gca().xaxis.tick_top()
#     plt.gca().xaxis.set_label_position("top")

#     # Set axis labels and title
#     plt.title("Confusion Matrix", fontsize=14, pad=20)
#     plt.xlabel("Prediction", fontsize=10)
#     plt.ylabel("Ground Truth", fontsize=10)
#     plt.xticks(fontsize=5, rotation=90)
#     plt.yticks(fontsize=5, rotation=0)
#     plt.tight_layout()

#     MODEL_FIGURES_DIR = get_figures_dir() / model_name
#     MODEL_FIGURES_DIR.mkdir(exist_ok=True)

#     filename = f"confusion{epoch}.png"
#     plt.savefig(MODEL_FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
#     plt.close()


# Function to plot Frequency vs Edit Distance
def plot_errors_by_frequency(df, dir: pathlib.Path, ax=None):
    """Plot average edit distance by word frequency.

    Parameters:
        df (pd.DataFrame): Data containing 'Lexicality', 'Zipf Frequency', 'Size',
            'Morphology', and 'Edit Distance'.
        ax (matplotlib.axes.Axes, optional): Axes object to draw the plot onto.
            If None, a new figure and axes are created.

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
    )
    plt.title("Average Edit Distance by Word Frequency")
    plt.xlabel("Zipf Frequency")
    plt.ylabel("Average Edit Distance")
    plt.legend(title="Size & Morphology")
    plt.grid(True)
    plt.close()
