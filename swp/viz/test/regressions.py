import pathlib
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*set_ticklabels\(\) should only be used with a fixed number of ticks.*",
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*Attempting to set identical low and high xlims*",
)

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

from ...utils.metrics import calc_importance

warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

sns.set_palette("colorblind")

### TODO what to do with this ?
# ### Correlation Matrix ###

# X_train_processed = pipeline.named_steps["preprocessor"].transform(X_train)
# X_train_processed_df = pd.DataFrame(
#     X_train_processed, columns=mapped_feature_names, index=X_train.index
# )
# corr_matrix = X_train_processed_df.corr(method="pearson")

# fig2, ax2 = plt.subplots(figsize=(8, 8))
# sns.heatmap(
#     corr_matrix,
#     annot=True,
#     fmt=".2f",
#     cmap="Blues",
#     ax=ax2,
#     cbar=False,
# )
# ax2.tick_params(axis="both", labelsize=16)
# filename = f"errors_matrix{plot_num}.png"
# fig2.savefig(str((filepath / filename).absolute()), dpi=300, bbox_inches="tight")
# plt.close(fig2)


def regression_plots(
    dfs: pd.DataFrame | list[pd.DataFrame],
    path: pathlib.Path | None = None,
    mode: str = "both",
    var: str = "ci",
) -> None:
    """Perform regression analysis on test data, plot feature importance.

    Parameters:
        df (pd.DataFrame): Dataframe containing features ('Lexicality', 'Zipf Frequency',
            'Morphology', 'Sequence Length', 'Bigram Frequency') and target ('Edit Distance').
        path (pathlib.Path): Directory path where plots will be saved.
        mode (int): Flag to indicate which subset of the data to use.
        var (str): Type of error bar to use. Defaults to "se".
    """
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
    elif isinstance(dfs, list) and not all(isinstance(df, pd.DataFrame) for df in dfs):
        raise ValueError("All elements in the list must be DataFrames.")

    data = {
        "Feature": [],
        "Importance": [],
    }

    for df in dfs:
        pipeline, *fi_list = calc_importance(df, mode=mode)

        # Get raw feature names and simplify them using a dictionary mapping.
        features = pipeline.named_steps["preprocessor"].get_feature_names_out()
        mapping = {
            "Length": "Len.",
            "Lexicality": "Lex.",
            "Morphology": "Mor.",
            "Zipf Frequency": "Frq.",
        }
        features = [
            next((v for k, v in mapping.items() if k in fn), fn) for fn in features
        ]

        coefs = pipeline.named_steps["regressor"].coef_
        for i, fi in enumerate(fi_list):
            data["Feature"].append(features[i])
            data["Importance"].append(np.sign(coefs[i]) * np.abs(fi))

    fi_df = pd.DataFrame(data)

    fi_df["AbsImp"] = fi_df["Importance"].abs()
    fi_sorted = fi_df.sort_values(by="AbsImp", ascending=True)

    plt.rcParams.update({"font.size": 24})
    fig1, ax1 = plt.subplots(figsize=(11, 6))
    # bar plot with error bars
    sns.barplot(
        data=fi_sorted,
        x="Importance",
        y="Feature",
        orient="h",
        ax=ax1,
        errorbar=var,
    )
    ax1.grid(axis="x", alpha=0.6)
    ax1.axvline(0, color="black", linestyle="--", linewidth=1)

    # Force x-axis symmetric around 0
    max_imp = fi_sorted["AbsImp"].max()
    ax1.set_xlim(-max_imp, max_imp)
    ax1.set_xlabel("Feature Importance", labelpad=-5)
    ax1.set_ylabel("")
    ax1.legend([], [], frameon=False)
    ticks = np.linspace(-max_imp - 0.005, max_imp + 0.005, 5)
    ax1.set_xticks(ticks)
    labels = ["" for _ in ticks]
    labels[0] = f"{-ticks[0]:.2f}"
    labels[-1] = f"{ticks[-1]:.2f}"
    ax1.set_xticklabels(labels, fontsize=22)

    # Save the plot
    if path:
        fig_path = path / f"errors_import_{mode}.png"
        fig1.savefig(fig_path, dpi=300, bbox_inches="tight")  # type: ignore
        plt.close(fig1)
    else:
        plt.show()


def activation_plots(
    df: pd.DataFrame,
    path: pathlib.Path | None = None,
) -> None:
    """Plot histograms of feature importance for each neuron's activations.

    Parameters:
        df (pd.DataFrame): Dataframe containing feature importance values for each neuron.
        path (pathlib.Path): Directory path where plots will be saved.
    """

    factors = df.columns[1:]
    for factor in factors:
        plt.figure(figsize=(11, 6))
        bins = np.linspace(df[factor].min(), df[factor].max(), 21)
        ax = sns.histplot(df[factor], bins=bins, color="blue")  # type: ignore
        xlow, xhigh = ax.get_xlim()
        xabs_max = max(abs(xlow), abs(xhigh))
        ax.set_xlim(-xabs_max, xabs_max)
        ax.axvline(0, color="grey", linestyle="--", linewidth=2)
        ax.set_ylabel("# of Neurons", fontsize=24, labelpad=10)
        ax.set_xlabel("Feature Importance", fontsize=24)
        ax.tick_params(axis="both", labelsize=20)
        ax.grid(True)
        plt.tight_layout()

        # TODO: Investigate this bug, so strange
        # factor = str(factor).lower()
        # print(factor, type(factor))
        if path:
            fig_path = path / f"histo_{factor}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    for x, y in combinations(factors, 2):
        plt.figure(figsize=(11, 6))
        ax = plt.gca()

        # Plot the scatter data
        sns.scatterplot(data=df, x=x, y=y, alpha=0.9, s=50, ax=ax)
        ax.set_xlabel(x, fontsize=24, labelpad=10)
        ax.set_ylabel(y, fontsize=24, labelpad=10)
        ax.grid(True)

        # Determine the symmetric limit so that (0,0) centered
        max_abs = max(
            abs(df[x].min()),
            abs(df[x].max()),
            abs(df[y].min()),
            abs(df[y].max()),
        )
        # Add a 5% margin.
        margin = 0.05 * max_abs
        limit = max_abs + margin

        if False:
            # Use symlog to support negative values.
            # Here, we set a linear threshold relative to the limit (or use a small default).
            linthresh = limit * 0.001 if limit != 0 else 1e-3
            ax.set_xscale("symlog", linthresh=linthresh)
            ax.set_yscale("symlog", linthresh=linthresh)

        # Set symmetric limits for both axes.
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.axvline(0, color="grey", linestyle=":", linewidth=1)
        ax.axhline(0, color="grey", linestyle=":", linewidth=1)
        ax.set_aspect("equal", adjustable="box")

        if path:
            fig_path = path / f"scatter_{x}_{y}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
