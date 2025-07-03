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

sns.set_palette("colorblind")


def plot_length_errors(
    df: pd.DataFrame,
    dir: pathlib.Path | None = None,
    var: str = "ci",
):
    """Plot average edit distance by sequence length.

    Parameters:
        df (pd.DataFrame): Data containing 'Length', 'Lexicality',
            'Morphology', and 'Edit_Distance'.\
        dir (pathlib.Path): Directory to save the plot.
    """
    data = df.copy()
    # plt.figure(figsize=(11, 6))
    plt.figure(figsize=(7, 6))
    ax = sns.lineplot(
        data=data,
        x="Length",
        y="Edit_Distance",
        hue="Lexicality",
        style="Morphology",
        marker="o",
        markersize=8,
        linewidth=3,
        palette={"real": "red", "pseudo": "blue"},
        errorbar=None,
    )
    plt.xlabel("Length", fontsize=24, labelpad=-15)
    plt.ylabel("Edit Distance", fontsize=24, labelpad=-15)
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
        # title="Lexicality & Morphology",
        # title_fontsize=22,
        fontsize=20,
        ncol=2,
    )
    plt.setp(leg.get_title(), multialignment="left")
    ax.set_xticks([3, 4, 5, 6, 7, 8, 9])
    ax.set_xticklabels(["3", "", "", "", "", "", "9"], fontsize=22)

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
    ax.grid(True)

    if dir:
        plt.savefig(dir / f"errors_len.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
