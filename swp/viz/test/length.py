import pathlib
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*set_ticklabels\(\) should only be used with a fixed number of ticks.*",
)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette("colorblind")


def plot_length_errors(df, checkpoint: str, dir: pathlib.Path):
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
    )
    plt.xlabel("Sequence Length", fontsize=24, labelpad=-10)
    plt.ylabel("Edit Distance", fontsize=24, labelpad=-35)
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
    y_ticks = ax.get_yticks()
    y_tick_labels = ["0" if tick == 0 else "" for tick in y_ticks]
    y_tick_labels[-1] = f"{y_ticks[-1]:.2f}"
    ax.set_yticklabels(y_tick_labels, fontsize=22)
    ymin, _ = ax.get_ylim()
    ax.set_ylim(ymin, y_ticks[-1])
    ax.grid(True)
    plt.savefig(dir / f"{checkpoint}~len_errors.png", dpi=300, bbox_inches="tight")
    plt.close()
