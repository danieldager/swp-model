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
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sns.set_palette("colorblind")


def regression_plots(
    df: pd.DataFrame,
    checkpoint: str,
    filepath: pathlib.Path,
    plot_num: int,
) -> None:
    """Perform regression analysis on test data, plot feature importance and correlation matrix.

    Parameters:
        df (pd.DataFrame): Dataframe containing features ('Lexicality', 'Zipf Frequency',
            'Morphology', 'Sequence Length', 'Bigram Frequency') and target ('Edit Distance').
        model_name (str): Name of the model (used in output/logging).
        train_name (str): Training configuration name (used in output/logging).
        checkpoint (str): Identifier for the current checkpoint/epoch.
        filepath (pathlib.Path): Directory path where plots will be saved.
    """
    # Copy the DataFrame and adjust Zipf Frequency for 'pseudo' lexicality.
    df = df.copy()

    if plot_num == 2:
        df = df[df["Lexicality"] == "real"]

    # Define features
    categorical_features = (
        ["Morphology", "Lexicality"] if plot_num == 1 else ["Morphology"]
    )
    continuous_features = (
        ["Sequence Length"] if plot_num == 1 else ["Sequence Length", "Zipf Frequency"]
    )

    X = df[categorical_features + continuous_features]
    y = df["Edit Distance"]

    # Split data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocessing: one-hot encode (dropping first) and standardize.
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first"), categorical_features),
            ("num", StandardScaler(), continuous_features),
        ]
    )

    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())]
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r_value = np.corrcoef(y_test, y_pred)[0, 1]
    print(f"\nMSE: {mse:.4f}, r: {r_value:.4f}\n")

    # --- Feature Importance Plot ---

    # Get raw feature names and simplify them using a dictionary mapping.
    raw_feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    mapping = {
        "Sequence Length": "Length",
        "Bigram Frequency": "Bigram",
        "Zipf Frequency": "Frequency",
        "Morphology": "Morphology",
        "Lexicality": "Lexicality",
    }
    mapped_feature_names = [
        next((v for k, v in mapping.items() if k in fn), fn) for fn in raw_feature_names
    ]

    # Retrieve coefficients and prepare the feature importance DataFrame
    coefficients = pipeline.named_steps["regressor"].coef_
    feature_importance = pd.DataFrame(
        {
            "Feature": mapped_feature_names,
            "Coefficient": coefficients,
        }
    )
    # Use absolute values for bar lengths
    feature_importance["Importance"] = feature_importance["Coefficient"].abs()
    # Create a new column for sign
    feature_importance["Sign"] = feature_importance["Coefficient"].apply(
        lambda x: "Positive" if x >= 0 else "Negative"
    )
    # Sort so that the smallest importance is at the top
    feature_importance_sorted = feature_importance.sort_values(
        by="Importance", ascending=True
    )

    plt.rcParams.update({"font.size": 24})
    fig1, ax1 = plt.subplots(figsize=(11, 6))
    custom_palette = {"Positive": "#5bc25f", "Negative": "#f54e4e"}
    sns.barplot(
        x="Importance",
        y="Feature",
        hue="Sign",
        data=feature_importance_sorted,
        orient="h",
        palette=custom_palette,
        ax=ax1,
    )
    ax1.axvline(0, color="black", linestyle="--", linewidth=1)
    ax1.set_xlabel("Feature Importance")
    ax1.set_ylabel("")
    ax1.legend(title=None)

    ax = plt.gca()
    # xticks = ax.get_xticks()
    # new_xticklabels = [
    #     f"{tick:.2f}" if (i == 0 or i == len(xticks) - 2) else ""
    #     for i, tick in enumerate(xticks)
    # ]
    # ax.set_xticklabels(new_xticklabels, fontsize=22)

    xticks = ax.get_xticks()
    new_xticklabels = [
        (
            f"{tick:.0f}"
            if i == 0
            else (format(tick, ".3g") if i == len(xticks) - 2 else "")
        )
        for i, tick in enumerate(xticks)
    ]
    ax.set_xticklabels(new_xticklabels, fontsize=22)
    filename = f"{checkpoint}~fimport{plot_num}.png"
    fig1.savefig(str((filepath / filename).absolute()), dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # --- Correlation Matrix Plot ---

    # Compute the correlation matrix on the processed training data.
    X_train_processed = pipeline.named_steps["preprocessor"].transform(X_train)
    X_train_processed_df = pd.DataFrame(
        X_train_processed, columns=mapped_feature_names, index=X_train.index
    )
    corr_matrix = X_train_processed_df.corr(method="pearson")

    # Create a separate figure for the correlation matrix.
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        ax=ax2,
        cbar=False,
    )
    ax2.tick_params(axis="both", labelsize=16)
    filename = f"{checkpoint}~cmatrix{plot_num}.png"
    fig2.savefig(str((filepath / filename).absolute()), dpi=300, bbox_inches="tight")
    plt.close(fig2)
