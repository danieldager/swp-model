import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*set_ticklabels\(\) should only be used with a fixed number of ticks.*",
)
sns.set_palette("colorblind")


def fi_scatter(fi_df: pd.DataFrame, model_dir: Path):
    plt.rcParams.update({"font.size": 18})

    fig, ax = plt.subplots(figsize=(8, 8))

    sns.scatterplot(
        data=fi_df,
        x="fi_len",
        y="fi_frq",
        hue="layer_name",
        palette={"encoder": "blue", "decoder": "red"},
        alpha=0.9,
        s=50,
    )
    ax.set_xlabel("Length", fontsize=24, labelpad=10)
    ax.set_ylabel("Frequency", fontsize=24)
    ax.grid(True)

    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    plt.tight_layout()
    plt.savefig(model_dir / f"scatter_fi.png", dpi=300)
    plt.close(fig)


def scatter_plot(
    results_df: pd.DataFrame,
    x,
    y,
    xlabel,
    ylabel,
    filename,
    model_dir,
    log_scale=False,
):
    """
    Produce and save a scatter plot for the specified x and y columns.

    Parameters:
        results_df (DataFrame): DataFrame containing the data.
        x (str): Name of the x column.
        y (str): Name of the y column.
        xlabel (str): Label for the x axis.
        ylabel (str): Label for the y axis.
        filename (str): File name to save the plot.
        model_dir (Path): Directory where to save the plot.
        log_scale (bool): If True, set both axes to a combined linear/logarithmic scale.
                          The region from 0 to 0.01 is linear (with 0 shown as the lower tick),
                          and above that the scale is logarithmic. The lower and upper ticks are
                          forced to be 0 and 100 respectively, with intermediate ticks as whole numbers.
    """
    # Set text elements to font size 18
    plt.rcParams.update({"font.size": 18})

    fig, ax = plt.subplots(figsize=(8, 8))

    results_df = results_df.copy()
    results_df[x] = 1 - results_df[x]
    results_df[y] = 1 - results_df[y]
    results_df["distance"] = (results_df[y] - results_df[x]) / np.sqrt(2)

    ### Scatter plot
    sns.scatterplot(
        data=results_df,
        x=x,
        y=y,
        hue="layer_name",
        palette={"encoder": "blue", "decoder": "red"},
        alpha=0.9,
        s=50,
        ax=ax,
    )
    ax.set_xlabel(xlabel, fontsize=24, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=24)

    ax.grid(True)
    if log_scale:
        ax.set_xscale("symlog", linthresh=0.001)
        ax.set_yscale("symlog", linthresh=0.001)
        ax.set_xlim(-1e-4, 1)
        ax.set_ylim(-1e-4, 1)

        ticks = ["0", "0.1", "1", "10", "100"]

        log_grid = [i * 1e-4 for i in [0, 2.5, 5, 7.5]] + [
            i * 10 ** -(3 - j) for j in range(0, 4) for i in range(1, 10)
        ]
        for x in log_grid:
            ax.axhline(x, linestyle="--", color="k", alpha=0.1)
            ax.axvline(x, linestyle="--", color="k", alpha=0.1)
    else:
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        lin_grid = [i * 1e-1 for i in range(11)]
        ticks = ["0", "0", "20", "40", "60", "80", "100"]

        for x in lin_grid:
            ax.axhline(x, linestyle="--", color="k", alpha=0.1)
            ax.axvline(x, linestyle="--", color="k", alpha=0.1)

    ax.plot([-0.001, 1], [-0.001, 1], color="grey", linestyle="--", linewidth=1)
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)

    # Draw a diagonal reference line
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    plt.savefig(model_dir / f"scatter_{filename}.png", dpi=300)

    # Create a legend figure
    ax.legend(title="Layer")
    handles, labels = ax.get_legend_handles_labels()
    figLegend = plt.figure(figsize=(2, 2))
    figLegend.legend(handles, labels, loc="center", title="Layer")
    figLegend.canvas.draw()
    plt.axis("off")
    # figLegend.savefig(model_dir / "scatter_legend.png", dpi=300, bbox_inches="tight")

    plt.close(fig)
    plt.close(figLegend)

    ### Histogram
    common_bins = np.linspace(
        results_df["distance"].min(), results_df["distance"].max(), 21
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    for ax, layer, color in zip(axes, ["encoder", "decoder"], ["blue", "red"]):
        subset = results_df[results_df["layer_name"] == layer]
        ax.hist(subset["distance"], bins=common_bins, color=color)
        ax.grid(True)
        xabs_max = abs(max(ax.get_xlim(), key=abs))
        ax.set_xlim(xmin=-xabs_max, xmax=xabs_max)
        ax.axvline(0, color="grey", linestyle="--", linewidth=2)

    xlabel = fig.supxlabel("Distance from Diagonal", fontsize=24)
    xlabel.set_position((0.54, 0.05))
    axes[0].set_ylabel("# of Neurons", fontsize=24, labelpad=10)
    plt.tight_layout()
    plt.savefig(model_dir / f"histo_{filename}.png", dpi=300)
    plt.close()
