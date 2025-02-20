import pathlib
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*set_ticklabels\(\) should only be used with a fixed number of ticks.*",
)

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_palette("colorblind")


def plot_frequency_errors(df, dir: pathlib.Path):
    """Plot average edit distance by word frequency.

    Parameters:
        df (pd.DataFrame): Data containing 'Lexicality', 'Zipf Frequency', 'Size',
            'Morphology', and 'Edit Distance'.
    """
    # Filter for only real word predictions.
    data = df.copy()
    data = data[data["Lexicality"] == "real"]

    # Bin the Zipf Frequency.
    data["Zipf Bin"] = pd.cut(
        data["Zipf Frequency"], bins=[1, 2, 3, 4, 5, 6, 7], right=False
    )

    # Group by the Zipf Bin only (ignoring Size and Morphology) and compute the average Edit Distance.
    grouped_df = (
        data.groupby(["Zipf Bin"], observed=True)["Edit Distance"].mean().reset_index()
    )
    # Convert the bins to string for plotting
    grouped_df["Zipf Bin"] = grouped_df["Zipf Bin"].astype(str)

    # Create a dummy control DataFrame with a value of 0 for each Zipf Bin.
    dummy_df = grouped_df.copy()
    dummy_df["Edit Distance"] = 0

    # Plotting
    plt.figure(figsize=(11, 6))
    ax = plt.gca()

    # Plot the ablated data (from df) in red.
    sns.lineplot(
        data=grouped_df,
        x="Zipf Bin",
        y="Edit Distance",
        # style="Size",
        marker="o",
        markersize=8,
        linewidth=3,
        color="red",
        label="ablated",
        ax=ax,
    )
    # Plot the dummy control (all zeros) in blue.
    sns.lineplot(
        data=dummy_df,
        x="Zipf Bin",
        y="Edit Distance",
        marker="o",
        markersize=8,
        linewidth=3,
        color="blue",
        label="control",
        ax=ax,
    )
    plt.xlabel("Zipf Frequency", fontsize=24, labelpad=10)
    plt.ylabel("Edit Distance", fontsize=24, labelpad=-35)
    plt.legend(fontsize=22, title_fontsize=26)

    # Manually adjust y-axis ticks
    yticks = ax.get_yticks()
    y_min, y_max = 0, yticks[-1]
    new_ticks = np.linspace(y_min, y_max, 7)
    ax.set_yticks(new_ticks)

    # Manually set the tick labels
    tick_labels = ["" for _ in new_ticks]
    tick_labels[0] = "0"
    tick_labels[-1] = f"{new_ticks[-1]:.2f}"
    while tick_labels[-1][-1] in {"0", "."}:
        tick_labels[-1] = tick_labels[-1][:-1]
    ax.set_yticklabels(tick_labels, fontsize=22)

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid(True)
    plt.savefig(dir / "errors_frq.png", dpi=300, bbox_inches="tight")
    plt.close()
