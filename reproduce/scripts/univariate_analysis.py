#!/usr/bin/env python3
"""
Generate univariate feature importance analysis.
Analyzes how well individual features predict neuron activations.
"""

import argparse
import os
from typing import List, Optional, Tuple

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from utils import CONVERTERS, get_default_paths, load_model, setup_environment

from swp.datasets.phonemes import get_phoneme_testloader
from swp.test.repetition import test
from swp.utils.embeddings import create_embeddings_LSTM_hook


def clean_wfe_data(
    wfe_df: pd.DataFrame, categorical_features: List[str]
) -> pd.DataFrame:
    """Clean WFE dataset by handling NaN values and ensuring uniform data types."""
    wfe_df = wfe_df.copy()

    # Handle NaN values in Zipf_Frequency (set to 0)
    wfe_df["Zipf_Frequency"] = wfe_df["Zipf_Frequency"].fillna(0)

    # Convert categorical features to strings to ensure uniform data types for OneHotEncoder
    for feature in categorical_features:
        if feature in wfe_df.columns:
            wfe_df[feature] = wfe_df[feature].astype(str)

    return wfe_df


def extract_embeddings(
    model, device, wfe_df: pd.DataFrame, batch_size: int
) -> np.ndarray:
    """Extract embeddings from the model at final sequence positions."""
    # Create hook to extract embeddings
    buffer = {"Hidden": [], "Cell": [], "Out": []}
    hook = create_embeddings_LSTM_hook(buffer, store_out=True)
    hook_handle = model.encoder.recurrent.register_forward_hook(hook)

    # Evaluate on the WFE Dataset
    wfe_loader = get_phoneme_testloader(batch_size=batch_size, dataset_df=wfe_df)
    test(
        model=model,
        device=device,
        test_df=wfe_df,
        test_loader=wfe_loader,
    )

    # Extract embeddings at final positions
    lengths = wfe_df["Length"].to_numpy()
    emb = np.vstack(buffer["Out"])
    embeddings = emb[np.arange(len(wfe_df)), lengths, :]

    hook_handle.remove()

    return embeddings


def calculate_feature_importance(
    X: pd.DataFrame,
    embeddings: np.ndarray,
    continuous_features: List[str],
    categorical_features: List[str],
    hidden_size: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Calculate permutation feature importance for each neuron."""
    features = continuous_features + categorical_features

    # Set up preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("con", StandardScaler(), continuous_features),
            ("cat", OneHotEncoder(drop="first"), categorical_features),
        ]
    )
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("regressor", LinearRegression())]
    )

    # Store results
    data = {"Neuron": [], "Feature": [], "Importance": [], "Spearman": []}

    # Analyze each neuron
    for idx in range(hidden_size):
        print(f"Calculating importance for neuron {idx+1}...  ", end="\r")

        y = embeddings[:, idx]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )

        # Fit pipeline
        pipeline.fit(X_train, y_train)

        # Calculate permutation importance
        result = permutation_importance(
            pipeline,
            X_test,
            y_test,
            n_repeats=100,
            random_state=seed,
        )

        importances = result.importances_mean  # type: ignore

        # Calculate Spearman correlation for overall model performance
        y_pred = pipeline.predict(X_test)
        rho, _ = spearmanr(y_test, y_pred)

        # Store results for each feature
        for i, feature in enumerate(features):
            data["Neuron"].append(idx)
            data["Feature"].append(feature)
            data["Importance"].append(importances[i])
            data["Spearman"].append(rho)

    print()  # New line after progress

    return pd.DataFrame(data)


def find_best_n_clusters(
    df: pd.DataFrame, max_clusters: int = 10, seed: int = 42
) -> Tuple[int, np.ndarray]:
    """Find the optimal number of clusters using KMeans and silhouette method."""
    best_num_clusters = 0
    best_silhouette_score = -1

    for num_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=num_clusters, random_state=seed)
        labels = kmeans.fit_predict(df)
        silhouette_avg = silhouette_score(df, labels)

        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_num_clusters = num_clusters

    kmeans = KMeans(n_clusters=best_num_clusters, random_state=seed)
    clusters = kmeans.fit_predict(df)

    return best_num_clusters, clusters


def create_stacked_importance_plot(
    df: pd.DataFrame,
    output_path: str,
    gap_size: int = 5,
    figsize: Tuple[float, float] = (22, 6),
) -> None:
    """Create a stacked bar plot of feature importance by neuron clusters."""

    # Prepare data for clustering and visualization
    copy = df.copy()
    copy["Importance"] = np.sign(copy["Importance"]) * np.log1p(
        np.abs(copy["Importance"])
    )

    # Cluster neurons by their feature importance patterns
    pivot = df.pivot_table(index="Neuron", columns="Feature", values="Importance")
    best_num_clusters, clusters = find_best_n_clusters(pivot, max_clusters=8)
    print(f"Optimal number of clusters: {best_num_clusters}")

    pivot["Cluster"] = clusters
    pivot = pivot.reset_index()
    copy = copy.merge(pivot[["Neuron", "Cluster"]], on="Neuron")

    # Prepare data for stacked plotting
    df_m = copy.copy()

    # Remove negative importance values by clipping at zero
    df_m["Importance"] = df_m["Importance"].clip(lower=0)

    # Compute cumulative sums per neuron for stacking
    df_m = df_m.sort_values(["Neuron", "Importance"], ascending=[True, False])
    df_m["cum_importance"] = df_m.groupby("Neuron")["Importance"].cumsum()
    df_m["bottom"] = df_m["cum_importance"] - df_m["Importance"]

    # Compute total importance per neuron
    neuron_totals = (
        df_m.groupby("Neuron")["Importance"]
        .sum()
        .reset_index()
        .rename(columns={"Importance": "total_importance"})
    )

    # Get unique neuron-level info and merge totals
    neurons = df_m[["Neuron", "Cluster"]].drop_duplicates().reset_index(drop=True)
    neurons = neurons.merge(neuron_totals, on="Neuron", how="left")

    # Compute total importance per cluster and sort clusters descending
    cluster_totals = neurons.groupby("Cluster")["total_importance"].sum().reset_index()
    cluster_totals = cluster_totals.sort_values(
        "total_importance", ascending=False
    ).reset_index(drop=True)

    # Create mapping from original cluster to new cluster order
    cluster_order_map = {
        old: new for new, old in enumerate(cluster_totals["Cluster"], start=1)
    }
    neurons["new_cluster"] = neurons["Cluster"].map(cluster_order_map)

    # Sort neurons by new cluster order and total importance
    neurons = neurons.sort_values(
        ["new_cluster", "total_importance"], ascending=[True, False]
    ).reset_index(drop=True)
    neurons["within_cluster_order"] = neurons.groupby("new_cluster").cumcount()

    # Define custom x-axis with gaps between clusters
    offset_map = {}
    current_offset = 0
    for cl in sorted(neurons["new_cluster"].unique()):
        offset_map[cl] = current_offset
        count_in_cluster = neurons[neurons["new_cluster"] == cl].shape[0]
        current_offset += count_in_cluster + gap_size

    neurons["unit_index"] = neurons["within_cluster_order"] + neurons[
        "new_cluster"
    ].map(offset_map)

    # Merge custom unit_index back into the melted DataFrame
    df_m = df_m.merge(neurons[["Neuron", "unit_index"]], on="Neuron", how="left")

    # Create the plot
    plt.figure(figsize=figsize)
    ax = plt.gca()

    # Use Seaborn's default "deep" palette for features
    features = sorted(df_m["Feature"].unique())
    palette = dict(zip(features, sns.color_palette("deep", len(features))))

    # Draw each segment as a bar segment
    for _, row in df_m.iterrows():
        ax.bar(
            row["unit_index"],
            row["Importance"],
            bottom=row["bottom"],
            width=1,
            color=palette.get(row["Feature"], "gray"),
        )

    ax.set_xlabel("Neuron (with clusters sorted by total FI & gaps between clusters)")
    ax.set_ylabel("Stacked Feature Importance")
    ax.set_title("Stacked Feature Importance per Neuron")
    plt.xticks([])  # Hide x-tick labels

    # Create legend
    handles = [
        mlines.Line2D(
            [],
            [],
            color=palette[feat],
            marker="s",
            linestyle="None",
            markersize=10,
            label=feat,
        )
        for feat in features
    ]
    plt.legend(handles=handles, title="Feature", loc="upper right")

    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_feature_importance_analysis(
    model_name: str,
    weights_path: str,
    batch_size: int,
    hidden_size: int,
    data_dir: str,
    figures_dir: str | None,
    continuous_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    regenerate: bool = False,
    seed: int = 42,
):
    """Generate complete univariate feature importance analysis."""
    # Setup
    device = setup_environment(seed)
    model = load_model(model_name, weights_path, device)

    # Get default paths
    paths = get_default_paths()

    # Default features if not specified
    if continuous_features is None:
        continuous_features = ["Length", "Zipf_Frequency"]
    if categorical_features is None:
        categorical_features = ["Lexicality", "Morphology"]

    features = continuous_features + categorical_features

    # Check for cached results
    results_path = os.path.join(data_dir, "univariate_importance.csv")

    if os.path.exists(results_path) and not regenerate:
        print(f"Loading cached feature importance results from {results_path}")
        df = pd.read_csv(results_path)
    else:
        print("Generating new feature importance analysis...")

        # Load and clean WFE dataset
        wfe_df = pd.read_csv(paths["datasets"]["wfe"], converters=CONVERTERS)
        wfe_df = clean_wfe_data(wfe_df, categorical_features)

        # Extract embeddings
        print("Extracting embeddings...")
        embeddings = extract_embeddings(model, device, wfe_df, batch_size)

        # Calculate feature importance
        print("Calculating feature importance...")
        X = wfe_df[features].copy()
        df = calculate_feature_importance(
            X, embeddings, continuous_features, categorical_features, hidden_size, seed
        )

        # Save results
        os.makedirs(data_dir, exist_ok=True)
        df.to_csv(results_path, index=False)
        print("Feature importance analysis completed!")

    # Create visualization
    if figures_dir:
        print("Creating stacked importance plot...")
        plot_path = os.path.join(figures_dir, "stacked_feature_importance.png")
        os.makedirs(figures_dir, exist_ok=True)
        create_stacked_importance_plot(df, plot_path)
        print(f"Plot saved to: {plot_path}")

    return df


def main():
    """Main function to run feature importance analysis."""
    parser = argparse.ArgumentParser(
        description="Generate univariate feature importance analysis"
    )
    parser.add_argument(
        "--model_name",
        default="Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1",
        help="Model name",
    )
    parser.add_argument(
        "--weights_path",
        help="Path to model weights",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=128,
        help="Hidden size of the model",
    )
    parser.add_argument(
        "--data_dir",
        help="Directory to save cached data",
    )
    parser.add_argument(
        "--figures_dir",
        help="Directory to save figures",
    )
    parser.add_argument(
        "--continuous_features",
        nargs="+",
        default=["Length", "Zipf_Frequency"],
        help="List of continuous features",
    )
    parser.add_argument(
        "--categorical_features",
        nargs="+",
        default=["Lexicality", "Morphology"],
        help="List of categorical features",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Force regeneration of cached results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots",
    )

    args = parser.parse_args()

    # Get default paths if not provided
    paths = get_default_paths()

    weights_path = args.weights_path or paths["weights"]["1024"]
    data_dir = args.data_dir or paths["data"]
    figures_dir = args.figures_dir or paths["figures"]

    # Run analysis
    df = generate_feature_importance_analysis(
        model_name=args.model_name,
        weights_path=weights_path,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        data_dir=data_dir,
        figures_dir=None if args.no_plot else figures_dir,
        continuous_features=args.continuous_features,
        categorical_features=args.categorical_features,
        regenerate=args.regenerate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
