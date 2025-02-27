import pathlib
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*set_ticklabels\(\) should only be used with a fixed number of ticks.*",
)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_palette("colorblind")


def plot_length_errors(df, dir: pathlib.Path):
    """Plot average edit distance by sequence length.

    Parameters:
        df (pd.DataFrame): Data containing 'Sequence Length', 'Lexicality',
            'Morphology', and 'Edit Distance'.
    """
    data = df.copy()
    grouped_df = (
        data.groupby(
            [
                "Sequence Length",
                "Lexicality",
                "Morphology",
            ],
            observed=True,
        )["Edit Distance"]
        .mean()
        .reset_index()
    )
    plt.figure(figsize=(11, 6))
    ax = sns.lineplot(
        data=grouped_df,
        x="Sequence Length",
        y="Edit Distance",
        hue="Lexicality",
        style="Morphology",
        marker="o",
        markersize=8,
        linewidth=3,
        palette={"real": "red", "pseudo": "blue"},
    )
    plt.xlabel("Sequence Length", fontsize=24, labelpad=-10)
    plt.ylabel("Edit Distance", fontsize=24, labelpad=-5)
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
        title="Lexicality & Morphology",
        fontsize=22,
        title_fontsize=22,
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
    plt.savefig(dir / f"errors_len.png", dpi=300, bbox_inches="tight")
    plt.close()
