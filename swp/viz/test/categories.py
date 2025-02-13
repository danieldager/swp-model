import pathlib
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*set_ticklabels\(\) should only be used with a fixed number of ticks.*",
)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_palette("colorblind")


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
