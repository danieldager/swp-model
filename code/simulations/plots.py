from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

""" PATHS """
FILE_DIR = Path(__file__).resolve()
ROOT_DIR = FILE_DIR.parent.parent.parent
FIGURES_DIR = ROOT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# Plot the training and validation loss curves
def training_curves(
    train_losses: list, valid_losses: list, model: str, num_epochs: int
):
    # Extract parameters from the model name
    h, l, d, r = [p[1:] for p in model.split("_")[1:]]

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(1, num_epochs + 1), y=train_losses, label="Training")
    sns.lineplot(x=range(1, num_epochs + 1), y=valid_losses, label="Validation")
    plt.title(f"Model: Hidden={h}, Layers={l}, Dropout={d}, LR={r}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.legend()
    plt.tight_layout()

    MODEL_FIGURES_DIR = FIGURES_DIR / model
    MODEL_FIGURES_DIR.mkdir(exist_ok=True)
    filename = MODEL_FIGURES_DIR / "training_curves.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")


# Plot the edit operations and distance for each test category
def error_plots(
    df: pd.DataFrame, model: str, epoch: Optional[int | str] = None
) -> None:
    sns.set_palette("colorblind")
    fig, (length_ax, freq_ax, errors_ax) = plt.subplots(3, 1, figsize=(12, 16))
    # fig.suptitle(f"{model} (Epoch {epoch})" if epoch else model, fontsize=16, y=0.9)

    # Figure 1: Edit Distance by Length
    data = df.copy()

    # Group and calculate average edit distance
    data["Lex_Morphology"] = data["Lexicality"] + "-" + data["Morphology"]
    grouped_length_df = (
        data.groupby(["Length", "Lex_Morphology"], observed=True)["Edit Distance"]
        .mean()
        .reset_index()
    )

    sns.lineplot(
        data=grouped_length_df,
        x="Length",
        y="Edit Distance",
        hue="Lex_Morphology",
        markers=True,
        marker="o",
        markersize=8,
        ax=length_ax,
    )
    length_ax.set_title("Edit Distance by Length")
    length_ax.set_xlabel("Word Length")
    length_ax.set_ylabel("Average Edit Distance")
    length_ax.legend(title="Lexicality & Morphology")
    length_ax.grid(True)

    # Figure 2: Frequency vs Edit Distance
    data = df[df["Lexicality"] == "real"].copy()

    # Group and calculate average edit distance
    data["Size_Morphology"] = data["Size"] + "-" + data["Morphology"]
    data["Zipf Bin"] = pd.cut(
        data["Zipf Frequency"], bins=[1, 2, 3, 4, 5, 6, 7], right=False
    )

    grouped_df = (
        data.groupby(["Zipf Bin", "Size_Morphology"], observed=True)["Edit Distance"]
        .mean()
        .reset_index()
    )
    grouped_df["Zipf Bin"] = grouped_df["Zipf Bin"].astype(str)

    sns.lineplot(
        data=grouped_df,
        x="Zipf Bin",
        y="Edit Distance",
        hue="Size_Morphology",
        marker="o",
        markersize=8,
        ax=freq_ax,
    )
    freq_ax.set_title("Edit Distance by Frequency")
    freq_ax.set_xlabel("Zipf Frequency")
    freq_ax.set_ylabel("Average Edit Distance")
    freq_ax.legend(title="Size & Morphology")
    freq_ax.grid(True)

    # Function to calculate average operations and total distance
    def calc_averages(group):
        return pd.Series(
            {
                "Deletions": group["Deletions"].mean(),
                "Insertions": group["Insertions"].mean(),
                "Substitutions": group["Substitutions"].mean(),
                "Edit Distance": group["Edit Distance"].mean(),
            }
        )

    # Figure 3: Errors by Test Category
    data = df.copy()

    # Group by categories and calculate averages
    data["Category"] = data.apply(
        lambda row: (
            f"pseudo {row['Size']} {row['Morphology']}"
            if row["Lexicality"] == "pseudo"
            else f"real {row['Size']} {row['Frequency']} {row['Morphology']}"
        ),
        axis=1,
    )
    grouped = data.groupby("Category").apply(calc_averages).reset_index()

    # Sort the categories in a meaningful order
    order = []
    # Add all real categories first
    for size in ["short", "long"]:
        for freq in ["high", "low"]:
            for morph in ["simple", "complex"]:
                order.append(f"real {size} {freq} {morph}")

    # Add all pseudo categories
    for size in ["short", "long"]:
        for morph in ["simple", "complex"]:
            order.append(f"pseudo {size} {morph}")

    # Melt the DataFrame for easier plotting
    melted = pd.melt(
        grouped,
        id_vars=["Category"],
        value_vars=["Deletions", "Insertions", "Substitutions", "Edit Distance"],
    )

    sns.barplot(
        x="Category", y="value", hue="variable", data=melted, order=order, ax=errors_ax
    )
    # Convert categories to a single letter for better readability
    order = ["".join([cat[:1].capitalize() for cat in cats.split()]) for cats in order]
    errors_ax.set_title("Errors by Category")
    errors_ax.set_xlabel("Category (Lexicality Size Frequency Morphology)")
    errors_ax.set_ylabel("Average Error Count")
    errors_ax.set_xticks(range(len(order)))
    errors_ax.set_xticklabels(order)
    errors_ax.legend(title="Error Type")
    errors_ax.grid(True)

    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)  # Adjust the vertical spacing between subplots

    MODEL_FIGURES_DIR = FIGURES_DIR / model
    MODEL_FIGURES_DIR.mkdir(exist_ok=True)
    filename = f"errors{epoch}.png" if epoch else "errors.png"
    plt.savefig(MODEL_FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


# Plot the confusion matrix for the test data
def confusion_matrix(
    confusions: dict, model: str, epoch: Optional[int | str] = None
) -> None:
    df = pd.DataFrame.from_dict(confusions, orient="index")

    # Separate the ALPAbet vowels and consonants

    # Normalize the confusion matrix
    # df = np.log1p(df) # log scale
    # df = df.div(df.sum(axis=1), axis=0) # row normal
    # df = (df - df.min().min()) / (df.max().max() - df.min().min()) # min max
    df = (df - df.mean()) / df.std()  # z score

    # Create a confusion matrix
    plt.figure(figsize=(8, 7))

    # Plot the heatmap with enhanced aesthetics
    sns.heatmap(
        df,
        annot=False,
        cmap="Blues",
        square=True,
        cbar_kws={"label": "Counts", "shrink": 0.2},
        xticklabels=True,
        yticklabels=True,
    )

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

    MODEL_FIGURES_DIR = FIGURES_DIR / model
    MODEL_FIGURES_DIR.mkdir(exist_ok=True)
    filename = f"confusion{epoch}.png" if epoch else "confusion.png"
    plt.savefig(MODEL_FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
