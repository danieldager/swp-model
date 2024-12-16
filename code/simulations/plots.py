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

sns.set_palette("colorblind")


# Plot the training and validation loss curves
def training_curves(train_losses: list, valid_losses: list, model: str, n_epochs: int):
    # Extract parameters from the model name
    h, l, d, t, r = [p[1:] for p in model.split("_")[1:]]

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(1, n_epochs + 1), y=train_losses, label="Training")
    sns.lineplot(x=range(1, n_epochs + 1), y=valid_losses, label="Validation")
    plt.title(f"Model: H={h}, L={l}, D={d}, TF={t}, LR={r}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.legend()
    plt.tight_layout()

    MODEL_FIGURES_DIR = FIGURES_DIR / model
    MODEL_FIGURES_DIR.mkdir(exist_ok=True)
    filename = MODEL_FIGURES_DIR / "training.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


# Function to plot Edit Distance by Length
def plot_errors_by_length(ax, df):
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
        hue="Lexicality",  # Line color by Lexicality
        style="Morphology",  # Line type by Morphology
        marker="o",
        markersize=8,
        ax=ax,
    )

    ax.set_title("Edit Distance by Length")
    ax.set_xlabel("Word Length")
    ax.set_ylabel("Average Edit Distance")
    ax.legend(title="Lexicality & Morphology")
    ax.grid(True)


# Function to plot Frequency vs Edit Distance
def plot_errors_by_frequency(ax, df):
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
        hue="Size",  # Line color by Size
        style="Morphology",  # Line type by Morphology
        marker="o",
        markersize=8,
        ax=ax,
    )

    ax.set_title("Edit Distance by Frequency")
    ax.set_xlabel("Zipf Frequency")
    ax.set_ylabel("Average Edit Distance")
    ax.legend(title="Size & Morphology")
    ax.grid(True)


# Function to plot Errors by Test Category
def plot_errors_by_category(ax, df):
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
    order = []

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
    ax.set_title("Errors by Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Average Error Count")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(short_order)
    ax.grid(True)


# Function to plot Error Rate by Position
def plot_errors_by_position(ax, df):
    totals = {}
    errors = {}

    for _, row in df.iterrows():
        length = row["Sequence Length"]

        for index in range(1, length + 1):
            normalized = (index - 1) / (length - 1)
            totals[normalized] = totals.get(normalized, 0) + 1

        for index in row["Error Indices"]:
            normalized = (index - 1) / (length - 1)
            errors[normalized] = errors.get(normalized, 0) + 1

    data = [
        {"Position": index, "Error Rate": errors.get(index, 0) / total}
        for index, total in totals.items()
    ]
    plot_df = pd.DataFrame(data)

    sns.lineplot(
        x="Position",
        y="Error Rate",
        data=plot_df,
        marker="o",
        markersize=8,
        ax=ax,
    )

    ax.set_title("Error Rate by Relative Position")
    ax.set_xlabel("Relative Position")
    ax.set_ylabel("Error Rate")
    ax.grid(True)


# Function to combine all plots into one figure
def error_plots(df: pd.DataFrame, model: str, epoch: str) -> None:
    # Parse model parameters for title
    e, h, l, d, t, r = [p[1:] for p in model.split("_")]
    title = f"Model: E={epoch} H={h}, L={l}, D={d}, TF={t}, LR={r}"

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    # Flatten axes for easy indexing
    axes = axes.flatten()

    plot_errors_by_length(axes[0], df)
    plot_errors_by_frequency(axes[1], df)
    plot_errors_by_category(axes[2], df)
    plot_errors_by_position(axes[3], df)

    fig.suptitle(title, fontsize=16, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    MODEL_FIGURES_DIR = FIGURES_DIR / model
    MODEL_FIGURES_DIR.mkdir(exist_ok=True)

    filename = f"errors{epoch}.png" if epoch else "errors.png"
    plt.savefig(MODEL_FIGURES_DIR / filename, dpi=300, bbox_inches="tight")


# Plot the confusion matrix for the test data
def confusion_matrix(confusions: dict, model: str, epoch: str) -> None:
    df = pd.DataFrame.from_dict(confusions, orient="index")

    # Separate the ALPAbet vowels and consonants
    phonemes = [
        "AA0",
        "AA1",
        "AA2",
        "AE0",
        "AE1",
        "AE2",
        "AH0",
        "AH1",
        "AH2",
        "AO0",
        "AO1",
        "AO2",
        "AW0",
        "AW1",
        "AW2",
        "AY0",
        "AY1",
        "AY2",
        "B",
        "CH",
        "D",
        "DH",
        "EH0",
        "EH1",
        "EH2",
        "ER0",
        "ER1",
        "ER2",
        "EY0",
        "EY1",
        "EY2",
        "F",
        "G",
        "HH",
        "IH0",
        "IH1",
        "IH2",
        "IY0",
        "IY1",
        "IY2",
        "JH",
        "K",
        "L",
        "M",
        "N",
        "NG",
        "OW0",
        "OW1",
        "OW2",
        "OY0",
        "OY1",
        "OY2",
        "P",
        "R",
        "S",
        "SH",
        "T",
        "TH",
        "UH0",
        "UH1",
        "UH2",
        "UW",
        "UW0",
        "UW1",
        "UW2",
        "V",
        "W",
        "Y",
        "Z",
        "ZH",
    ]

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

    MODEL_FIGURES_DIR = FIGURES_DIR / model
    MODEL_FIGURES_DIR.mkdir(exist_ok=True)

    filename = f"confusion{epoch}.png"
    plt.savefig(MODEL_FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
