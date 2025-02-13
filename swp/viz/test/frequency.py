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
