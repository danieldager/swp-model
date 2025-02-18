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


def plot_category_errors(df, dir: pathlib.Path):
    """Plot average errors (deletions, insertions, substitutions) by word category.

    Parameters:
        df (pd.DataFrame): Data containing 'Lexicality', 'Size', 'Frequency',
            'Morphology', 'Deletions', 'Insertions', and 'Substitutions'.
        ax (matplotlib.axes.Axes, optional): Axes object to draw the plot onto.
            If None, a new figure and axes are created.

    """
    data = df.copy()
    grouped = (
        data.groupby("Condition")[["Deletions", "Insertions", "Substitutions"]]
        .mean()
        .reset_index()
    )
    melted = pd.melt(
        grouped,
        id_vars=["Condition"],
        value_vars=["Deletions", "Insertions", "Substitutions"],
    )

    plt.figure(figsize=(11, 6))
    sns.barplot(
        x="Condition",
        y="value",
        hue="variable",
        # order=order,
        data=melted,
    )
    plt.xlabel("Error Condition", fontsize=14, labelpad=10)
    plt.ylabel("Average Error Count", fontsize=14, labelpad=10)
    # plt.xticks(range(len(order)), short_order)
    plt.legend(title="Condition", fontsize=12, title_fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.savefig(dir / f"errors_con.png", dpi=300)  # , bbox_inches="tight")
    plt.close()
