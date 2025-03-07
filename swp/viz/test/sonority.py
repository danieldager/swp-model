import pathlib
import warnings

import numpy as np

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*set_ticklabels\(\) should only be used with a fixed number of ticks.*",
)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette("colorblind")


def plot_sonority_errors(df, dir: pathlib.Path):
    """Plot average edit distance grouped by sonority.

    Parameters:
        df (pd.DataFrame): Data containing 'Sonority', 'Type', and 'Edit Distance'.
        ax (matplotlib.axes.Axes, optional): Axes object to draw the plot onto.
            If None, a new figure and axes are created.

    """
    data = df.copy()
    grouped_df = (
        data.groupby(["Sonority", "Type"], observed=True)["Edit Distance"]
        .mean()
        .reset_index()
    )
    plt.figure(figsize=(11, 6))
    ax = sns.lineplot(
        data=grouped_df,
        x="Sonority",
        y="Edit Distance",
        hue="Type",
        marker="o",
        markersize=8,
        linewidth=3,
    )
    plt.xlabel("Sonority Gradient", fontsize=24, labelpad=-10)
    plt.ylabel("Edit Distance", fontsize=24, labelpad=-35)
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
    plt.savefig(dir / "errors_son.png", dpi=300, bbox_inches="tight")
    plt.close()
