import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*set_ticklabels\(\) should only be used with a fixed number of ticks.*",
)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette("colorblind")


def plot_sonority_errors(
    df: pd.DataFrame,
    path: pathlib.Path | None = None,
    var: str = "se",
) -> None:
    """Plot average edit distance grouped by sonority.

    Parameters:
        df (pd.DataFrame): Data containing 'Sonority', 'Type', and 'Edit_Distance'.
        path (pathlib.Path): Directory to save the plot.
        var (str): Error bar type, default is 'se' (standard error).
    """
    plt.figure(figsize=(11, 6))
    ax = sns.lineplot(
        data=df,
        x="Sonority",
        y="Edit_Distance",
        hue="Type",
        marker="o",
        markersize=8,
        linewidth=3,
        errorbar=var,
    )
    plt.xlabel("Sonority Gradient", fontsize=24, labelpad=-10)
    plt.ylabel("Edit Distance", fontsize=24, labelpad=-15)
    plt.legend(title="CCV or VCC", fontsize=24, title_fontsize=24)
    ax.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    ax.set_xticklabels(["-5", "", "", "", "", "", "", "", "", "", "5"], fontsize=22)

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

    if path:
        plt.savefig(path / "errors_son.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
