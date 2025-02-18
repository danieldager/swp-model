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
from scipy.interpolate import pchip_interpolate

from ...utils.plots import set_edge_ticks

sns.set_palette("colorblind")


# Function to plot Average Edit Distance by Position
def plot_position_errors(df, dir: pathlib.Path):
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
        x="Position",
        y="Error Rate",
        hue="Lexicality",
        data=plot_df,
        marker="o",
        markersize=8,
        linewidth=3,
        palette={"real": "red", "pseudo": "blue"},
    )
    plt.xlabel("Relative Position", fontsize=24, labelpad=-10)
    plt.ylabel("Error Rate", fontsize=24, labelpad=-40)
    plt.legend(title="Lexicality", fontsize=24, title_fontsize=24)
    # set_edge_ticks(ax, tick_fontsize=22, x_decimal_places=1, y_decimal_places=2)
    plt.savefig(dir / f"pos_errors.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_position_smoothened_errors(df, dir: pathlib.Path, multi: bool = False):
    """Plot smoothened average edit distance by relative position within each sequence.
    Also does a subplot for every length.

    Parameters:
        df (pd.DataFrame): Data containing 'Lexicality', 'Sequence Length',
            and 'Error Indices'.
        ax (matplotlib.axes.Axes, optional): Axes object to draw the plot onto.
            If None, a new figure and axes are created.

    """
    data_by_lexicality = []
    # Iterate through rows grouped by Lexicality
    num_points = 100
    x = np.linspace(0, 1, num_points, endpoint=True)
    for lexicality, group_df in df.groupby("Lexicality"):
        y = np.zeros(num_points)
        for length, length_df in group_df.groupby("Sequence Length"):
            errors = {index / (length - 1): 0 for index in range(length)}
            for _, row in length_df.iterrows():
                length = row["Sequence Length"]

                # Count total occurrences and errors by normalized position

                for index in row["Error Indices"]:
                    normalized = (index - 1) / (length - 1)
                    errors[normalized] = errors.get(normalized, 0) + 1
            indices = np.array([index for index in errors])
            rates = np.array([error / len(length_df) for error in errors.values()])
            curr_y = pchip_interpolate(xi=indices, yi=rates, x=x)
            if multi:
                data_by_lexicality.extend(
                    [
                        {
                            "Position": x[i],
                            "Error Rate": curr_y[i],
                            "Lexicality": lexicality,
                            "Length": length,
                            "Smooth": "smooth",
                        }
                        for i in range(len(y))
                    ]
                )
                data_by_lexicality.extend(
                    [
                        {
                            "Position": indices[i],
                            "Error Rate": rates[i],
                            "Lexicality": lexicality,
                            "Length": length,
                            "Smooth": "raw",
                        }
                        for i in range(length)
                    ]
                )
            y += len(length_df) * curr_y
        y /= len(group_df)

        # Create data entries for the current lexicality
        data_by_lexicality.extend(
            [
                {
                    "Position": x[i],
                    "Error Rate": y[i],
                    "Lexicality": lexicality,
                    "Length": "all",
                    "Smooth": "smooth",
                }
                for i in range(len(y))
            ]
        )
    plot_df = pd.DataFrame(data_by_lexicality)
    if multi:
        f, axs = plt.subplots(8, figsize=(11, 48))
        for i, (length, plot_subdf) in enumerate(plot_df.groupby("Length")):
            ax = sns.lineplot(
                x="Position",
                y="Error Rate",
                hue="Lexicality",
                style="Smooth",
                data=plot_subdf,
                markersize=8,
                linewidth=3,
                ax=axs[i],
                palette={"real": "red", "pseudo": "blue"},
            )
            ax.set_xlabel("Relative Position", fontsize=24, labelpad=-10)
            ax.set_ylabel("Error Rate", fontsize=24, labelpad=-40)
            ax.legend(
                title=f"Lexicality {length}",
                fontsize=24,
                title_fontsize=24,
                bbox_to_anchor=(1.05, 1),
            )
            set_edge_ticks(ax, tick_fontsize=22, x_decimal_places=1, y_decimal_places=2)
    else:
        plt.figure(figsize=(11, 6))
        ax = sns.lineplot(
            x="Position",
            y="Error Rate",
            hue="Lexicality",
            style="Smooth",
            data=plot_df,
            markersize=8,
            linewidth=3,
            palette={"real": "red", "pseudo": "blue"},
        )
        ax.set_xlabel("Relative Position", fontsize=24, labelpad=-10)
        ax.set_ylabel("Error Rate", fontsize=24, labelpad=-40)
        # ax.legend(
        #     title=f"Lexicality",
        #     fontsize=24,
        #     title_fontsize=24,
        #     bbox_to_anchor=(1.05, 1),
        # )
        # get rid of legend
        ax.get_legend().remove()
        set_edge_ticks(ax, tick_fontsize=22, x_decimal_places=1, y_decimal_places=2)
    plt.savefig(
        dir / f"errors_pos{'_len' if multi else ''}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_position_errors_bins(
    df, dir: pathlib.Path, num_bins=3, improve_normalize: bool = True
):
    """Plot binned average edit distance by relative position within each sequence.
    Setting `improve_normalize` to True does so that a word can contribut at most once
    to a bin, and contribute if at least one errors should be attributed to that bin.

    Parameters:
        df (pd.DataFrame): Data containing 'Lexicality', 'Sequence Length',
            and 'Error Indices'.
        ax (matplotlib.axes.Axes, optional): Axes object to draw the plot onto.
            If None, a new figure and axes are created.

    """
    data_by_lexicality = []
    for lexicality, group_df in df.groupby("Lexicality"):
        bins = {i / num_bins: 0 for i in range(num_bins)}
        for _, row in group_df.iterrows():
            length = row["Sequence Length"]

            # Count total occurrences and errors by normalized position
            added = set()
            for index in row["Error Indices"]:
                normalized = (index - 1) / (length - 1)
                for i in range(num_bins):
                    if normalized == 1.0:
                        curr_bin = (num_bins - 1) / num_bins
                        break
                    elif i / num_bins <= normalized:
                        curr_bin = i / num_bins
                    else:
                        break
                if improve_normalize:
                    if curr_bin not in added:
                        bins[curr_bin] += 1
                        added.add(curr_bin)
                else:
                    bins[curr_bin] += 1
        # Create data entries for the current lexicality
        data_by_lexicality.extend(
            [
                {
                    "Position": curr_bin,
                    "Error Rate": bins[curr_bin] / len(group_df),
                    "Lexicality": lexicality,
                }
                for curr_bin in bins
            ]
        )
    plot_df = pd.DataFrame(data_by_lexicality)
    plt.figure(figsize=(11, 6))
    ax = sns.barplot(
        x="Position",
        y="Error Rate",
        hue="Lexicality",
        data=plot_df,
        linewidth=3,
        palette={"real": "red", "pseudo": "blue"},
    )
    ax.set_xlabel(" ", fontsize=24, labelpad=-10)
    ax.set_ylabel("Error Rate", fontsize=24, labelpad=-40)
    ax.legend(
        title="Lexicality", fontsize=24, title_fontsize=24, bbox_to_anchor=(1.05, 1)
    )
    set_edge_ticks(
        ax, tick_fontsize=22, x_decimal_places=1, y_decimal_places=2, bins=True
    )
    plt.savefig(
        dir / f"errors_pos_{num_bins}b.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
