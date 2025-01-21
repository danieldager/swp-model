import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
from Levenshtein import editops

from .paths import get_figures_dir

sns.set_palette("colorblind")


# Plot the training and validation loss curves
def training_curves(train_losses: list, valid_losses: list, model: str, n_epochs: int):
    # TODO Daniel docstring
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
    # TODO Daniel docstring
    data = df.copy()
    grouped_df = (
        data.groupby(["Length", "Lexicality", "Morphology"], observed=True)[
            "Edit Distance"
        ]
        .mean()
        .reset_index()
    )

    sns.lineplot(
        data=grouped_df,
        x="Length",
        y="Edit Distance",
        hue="Lexicality",
        style="Morphology",
        marker="o",
        markersize=8,
        ax=ax,
    )

    if ax:
        ax.set_title("Edit Distance by Length")
        ax.set_xlabel("Word Length")
        ax.set_ylabel("Average Edit Distance")
        ax.legend(title="Lexicality & Morphology")
        ax.grid(True)

    else:
        plt.title("Edit Distance by Length")
        plt.xlabel("Word Length")
        plt.ylabel("Average Edit Distance")
        plt.legend(title="Lexicality & Morphology")
        plt.grid(True)
        plt.tight_layout()


# Function to plot Frequency vs Edit Distance
def plot_errors_by_frequency(df, ax=None):
    # TODO Daniel docstring
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
    # TODO Daniel docstring
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
        ax.set_title("Average Error Rates by Word Category")
        ax.set_xlabel("Category")
        ax.set_ylabel("Average Error Count")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(short_order)
        ax.legend(title="Category")
        ax.grid(True)

    else:
        plt.title("Average Error Rates by Word Category")
        plt.xlabel("Category")
        plt.ylabel("Average Error Count")
        plt.xticks(range(len(order)), short_order)
        plt.legend(title="Category")
        plt.grid(True)
        plt.tight_layout()


# Function to plot Average Edit Distance by Position
def plot_errors_by_position(df, ax=None):
    # TODO Daniel docstring
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
        ax=ax,
    )

    # Set axis labels and grid
    if ax:
        ax.set_title("Average Edit Distance by Relative Position")
        ax.set_xlabel("Relative Position")
        ax.set_ylabel("Error Rate")
        ax.grid(True)
    else:
        plt.title("Average Edit Distance by Relative Position")
        plt.xlabel("Relative Position")
        plt.ylabel("Error Rate")
        plt.grid(True)
        plt.tight_layout()


def plot_errors_by_sonority(df, ax=None):
    # TODO Daniel docstring
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
        ax=ax,
    )

    if ax is not None:
        ax.set_title("Average Edit Distance by Sonority")
        ax.set_xlabel("Sonority")
        ax.set_ylabel("Order")
        ax.legend(title="CCV or VCC")
        ax.grid(True)

    else:
        plt.title("Average Edit Distance by Sonority")
        plt.xlabel("Sonority")
        plt.ylabel("Average Edit Distance")
        plt.legend(title="Order")
        plt.grid(True)
        plt.tight_layout()


def calculate_errors(phonemes: list, prediction: list, phoneme_to_id: dict) -> dict:
    phonemes = [phoneme_to_id[p] for p in phonemes]
    prediction = [phoneme_to_id[p] for p in prediction]

    # Compute errors
    errors = editops(phonemes, prediction)
    counts = Counter(op for op, _, _ in errors)

    # Identify mismatched indices
    error_indices = [
        i + 1 for i, (j, k) in enumerate(zip(phonemes, prediction)) if j != k
    ]

    return {
        "Edit Distance": len(errors),
        "Insertions": counts["insert"],
        "Deletions": counts["delete"],
        "Substitutions": counts["replace"],
        "Sequence Length": len(phonemes),
        "Error Indices": error_indices,
    }


# Function to combine all plots into one figure
def create_error_plots(
    test_df: pd.DataFrame,
    sonority_df: pd.DataFrame,
    model_name: str,
    training_name: str,
    epoch: str,
    checkpoint: str = None,
) -> None:
    # TODO Daniel docstring

    # Parse model parameters for title
    m, h, l, v, d, t, s = [p[1:] for p in model_name.split("_")[1:]]
    b, r, f = [p[1:] for p in training_name.split("_")]
    m = "LSTM" if m[0] == "S" else "RNN"

    if checkpoint is not None:
        epoch = f"{epoch}_{checkpoint}"
    title = f"{m}: E={epoch} H={h}, L={l}, D={d}, TF={t}, LR={r} V={v} F={f}"

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    # Flatten axes for easy indexing
    axes = axes.flatten()

    plot_errors_by_length(test_df, axes[0])
    plot_errors_by_category(test_df, axes[1])
    plot_errors_by_position(test_df, axes[2])
    plot_errors_by_sonority(sonority_df, axes[3])

    fig.suptitle(title, fontsize=16, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    dir_name = f"{model_name}~{training_name}"
    results_figures_dir = get_figures_dir() / dir_name
    results_figures_dir.mkdir(exist_ok=True)

    filename = f"errors_{epoch}"
    if checkpoint is not None:
        filename = f"{filename}_{checkpoint}"
    plt.savefig(results_figures_dir / f"{filename}.png", dpi=300, bbox_inches="tight")


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
