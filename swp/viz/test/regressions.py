import pathlib
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*set_ticklabels\(\) should only be used with a fixed number of ticks.*",
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


def regression_plots(
    df: pd.DataFrame, filepath: pathlib.Path, mode: str = "real"
) -> None:
    """Perform regression analysis on test data, plot feature importance and correlation matrix.

    In addition to the original pipeline, this function now fits two regression models
    using interaction terms (Length * Lexicality * Morphology):
      - A linear regression on Edit Distance.
      - A logistic regression on Error Rate (a binary outcome derived from Edit Distance).

    It then prints the p-values for the three main effects and (if any) significant interactions.

    Parameters:
        df (pd.DataFrame): Dataframe containing features ('Lexicality', 'Zipf Frequency',
            'Morphology', 'Sequence Length', 'Bigram Frequency') and target ('Edit Distance').
        filepath (pathlib.Path): Directory path where plots will be saved.
        plot_num (int): Flag to indicate which subset of the data to use.
    """
    # Copy the DataFrame and adjust Zipf Frequency for 'pseudo' lexicality.
    df = df.copy()

    pipeline, *fi_list = calc_importance(df, mode=mode)

    # Get raw feature names and simplify them using a dictionary mapping.
    raw_feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    mapping = {
        "Length": "Len.",
        "Lexicality": "Lex.",
        "Morphology": "Mor.",
        "Zipf Frequency": "Frq.",
    }
    mapped_feature_names = [
        next((v for k, v in mapping.items() if k in fn), fn) for fn in raw_feature_names
    ]

    # Retrieve coefficients and prepare the feature importance DataFrame.
    coefficients = pipeline.named_steps["regressor"].coef_
    # print(coefficients)

    # apply the signs of coefficients to the feature importance values
    for i, fi in enumerate(fi_list):
        fi_list[i] = np.sign(coefficients[i]) * np.abs(fi)

    feature_importance = pd.DataFrame(
        {
            "Feature": mapped_feature_names,
            "Importance": fi_list,
        }
    )
    # Sort by absolute magnitude.
    feature_importance["AbsCoef"] = feature_importance["Importance"].abs()
    feature_importance_sorted = feature_importance.sort_values(
        by="AbsCoef", ascending=True
    )

    # Plot the "PlotCoefficient"
    plt.rcParams.update({"font.size": 24})
    fig1, ax1 = plt.subplots(figsize=(11, 6))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=feature_importance_sorted,
        orient="h",
        ax=ax1,
    )
    ax1.grid(axis="x", alpha=0.6)
    ax1.axvline(0, color="black", linestyle="--", linewidth=1)

    # Force x-axis symmetric around 0 based on the maximum absolute coefficient.
    max_coef = feature_importance_sorted["AbsCoef"].max()
    ax1.set_xlim(-max_coef, max_coef)
    ax1.set_xlabel("Feature Importance", labelpad=-5)
    ax1.set_ylabel("")
    ax1.legend([], [], frameon=False)
    ticks = np.linspace(-max_coef - 0.005, max_coef + 0.005, 5)
    ax1.set_xticks(ticks)
    labels = ["" for _ in ticks]
    labels[0] = f"{-ticks[0]:.2f}"
    labels[-1] = f"{ticks[-1]:.2f}"
    ax1.set_xticklabels(labels, fontsize=22)

    # Save the plot
    filename = f"errors_import_{mode}.png"
    fig1.savefig(str((filepath / filename).absolute()), dpi=300, bbox_inches="tight")
    plt.close(fig1)

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


def activation_plots(results_df: pd.DataFrame, figures_dir: pathlib.Path) -> None:
    """Plot histograms of feature importance for each neuron."""

    factors = results_df.columns[1:]
    for factor in factors:
        plt.figure(figsize=(11, 6))
        bins = np.linspace(results_df[factor].min(), results_df[factor].max(), 21)
        ax = sns.histplot(results_df[factor], bins=bins, color="blue")  # type: ignore
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
        fig_path = figures_dir / f"histo_{factor}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()

    for x, y in combinations(factors, 2):
        plt.figure(figsize=(11, 6))
        ax = plt.gca()

        # Plot the scatter data
        sns.scatterplot(data=results_df, x=x, y=y, alpha=0.9, s=50, ax=ax)
        ax.set_xlabel(x, fontsize=24, labelpad=10)
        ax.set_ylabel(y, fontsize=24, labelpad=10)
        ax.grid(True)

        # Determine the symmetric limit so that (0,0) is at the center.
        max_abs = max(
            abs(results_df[x].min()),
            abs(results_df[x].max()),
            abs(results_df[y].min()),
            abs(results_df[y].max()),
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

        # Add dotted lines at x=0 and y=0.
        ax.axvline(0, color="grey", linestyle=":", linewidth=1)
        ax.axhline(0, color="grey", linestyle=":", linewidth=1)

        # Force an equal aspect ratio.
        ax.set_aspect("equal", adjustable="box")

        # Save the figure.
        fig_path = figures_dir / f"scatter_{x}_{y}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
