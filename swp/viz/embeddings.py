import pathlib

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import PowerNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from mlem_minimal import feature_distances, mlem, representation_distances
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score


def dissim_matrix(
    df: pd.DataFrame,
    dataset: str,
    num_layers: int,
    path: pathlib.Path,
    metric: str = "euclidean",
) -> None:
    """Compute the dissimilarity matrix for a DataFrame of embeddings"""
    for layer in range(1, num_layers + 1):
        h_emb = np.array(df[f"H{layer}"].to_list())
        c_emb = np.array(df[f"C{layer}"].to_list())
        hc_emb = np.concatenate([h_emb, c_emb], axis=1)

        # for name, emb in zip(["H", "C", "HC"], [h_emb, c_emb, hc_emb]):
        for name, emb in zip(["H"], [h_emb]):
            dis = pdist(emb, metric)  # type: ignore
            lnk = linkage(emb, method="ward")

            if dataset == "phoneme":
                figsize = (20, 12)
            else:
                figsize = (40, 24)
            plt.figure(figsize=figsize)

            if dataset == "evaluation":
                wspace = 0.1
            else:
                wspace = 0.05
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4], wspace=wspace)

            ax_den = plt.subplot(gs[0])
            den = dendrogram(lnk, orientation="left")
            ax_den.set_xticks([])
            ax_den.set_yticks([])
            order = den["leaves"]

            if dataset == "phoneme":
                full = True
                labels = [v[0] for v in df["No_Stress"].values[order]]
            elif dataset == "bigram":
                full = False
                labels = ["-".join(v) for v in df["No_Stress"].values[order]]
            elif dataset == "evaluation":
                full = False
                labels = [word for word in df["Word"].values[order]]

            ax_mat = plt.subplot(gs[1])
            mat = squareform(dis)[order, :][:, order][::-1, :]
            # vmin = mat.min()
            # vmax = mat.max()
            vmin = np.percentile(mat, 1)
            vmax = np.percentile(mat, 99)
            norm = TwoSlopeNorm(np.median(mat), vmin, mat.max())  # type: ignore
            # norm = PowerNorm(gamma=5, vmin=mat.min(), vmax=mat.max())
            im = ax_mat.imshow(mat, aspect="auto", cmap="RdBu_r", norm=norm)
            ax_mat.set_xticks(np.arange(len(labels)))
            ax_mat.set_yticks(np.arange(len(labels)))

            if full:
                ax_mat.set_xticklabels(labels)
                ax_mat.set_yticklabels(labels[::-1])

            else:
                ax_mat.set_xticklabels([])
                y_labels = [
                    label if i % 10 == 0 else "" for i, label in enumerate(labels[::-1])
                ]
                ax_mat.set_yticklabels(y_labels)

            plt.colorbar(im, ax=ax_mat)
            # plt.tight_layout()
            plt.savefig(path / f"{dataset}_dmatrix_{name}{layer}_{metric}.png", dpi=300)
            plt.close()


def custom_dmatrix(
    df: pd.DataFrame,
    metric: str = "euclidean",
    key: str = "H1",
) -> None:
    custom_order = {
        # "C": 0,
        # "V": 1,
        "CC": 2,
        "CV": 3,
        "VC": 4,
        "VV": 5,
        "CCC": 6,
        "CCV": 7,
        "CVC": 8,
        "CVV": 9,
        "VCC": 10,
        "VCV": 11,
        "VVC": 12,
        "VVV": 13,
    }
    df["Order"] = df["Type"].map(custom_order)
    df = df.sort_values("Order")
    order = df.index.tolist()
    labels = df["Type"].tolist()

    emb = np.array(df[key].tolist())
    dis = pdist(emb, metric)  # type: ignore

    # Reverse the labels for matrix, 1 label per group
    reversed_labels = labels[::-1]
    y_labels = []
    prev_type = None
    for label in reversed_labels:
        y_labels.append("" if label == prev_type else label)
        prev_type = label

    # Reorder the square dissimilarity matrix using the sorted order.
    # The [::-1, :] reverses the row order to match the reversed y-axis labels.
    matrix = squareform(dis)[order, :][:, order][::-1, :]

    # Set up normalization based on the percentiles of the matrix
    vmin = np.percentile(matrix, 1)
    vmax = np.percentile(matrix, 99)
    norm = TwoSlopeNorm(np.median(matrix), vmin, vmax)  # type: ignore

    fig, ax = plt.subplots(figsize=(20, 16))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(y_labels[::-1])
    ax.set_yticklabels(y_labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)

    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=18)
    # cbar.set_label(f"{metric.title()} Distance", fontsize=18)

    plt.show()


# TODO: Danny rewrite
def pca_mds(
    df: pd.DataFrame,
    dataset: str,
    num_layers: int,
    path: pathlib.Path | None,
    last_token: bool = False,
    features_list: list[str] = [],
    plot_mds: bool = True,
    plot_txt: bool = False,
    df_subset: pd.DataFrame | None = None,
) -> None:
    """Compute the PCA and MDS for a DataFrame of embedding vectors"""
    # df["First Token"] = df["No_Stress"].apply(lambda x: x[0])
    # df["Second Token"] = df["No_Stress"].apply(lambda x: x[1])
    # df["Last Token"] = df["No_Stress"].apply(lambda x: x[-1])

    # df["First Type"] = df["First Token"].apply(
    #     lambda x: "Vowel" if x[0] in "AEIOUY" else "Consonant"
    # )
    # df["Second Type"] = df["Second Token"].apply(
    #     lambda x: "Vowel" if x[0] in "AEIOUY" else "Consonant"
    # )
    # df["Last Type"] = df["Last Token"].apply(
    #     lambda x: "Vowel" if x[0] in "AEIOUY" else "Consonant"
    # )

    for layer in range(1, num_layers + 1):
        h_emb = np.array(df[f"H{layer}"].to_list())
        c_emb = np.array(df[f"C{layer}"].to_list())
        hc_emb = np.concatenate([h_emb, c_emb], axis=1)

        if dataset == "phoneme":
            features = []
            figsize = (6, 6)
            labels = [v[0] for v in df["No_Stress"].values]
        elif dataset == "bigram":
            figsize = (12, 12)
            if last_token:
                features = ["First Type", "Type"]
            else:
                features = ["Place", "Height"]
            labels = ["-".join(v) for v in df["No_Stress"].values]
        elif dataset == "evaluation":
            figsize = (12, 12)
            if last_token:
                features = ["First Type", "Second Type"]
                # features = ["Length", "First Type"]
            else:
                features = ["Length", "Lexicality"]
            labels = [word for word in df["Word"].values]
        elif dataset == "trigram":
            figsize = (12, 12)
            # figsize = (18, 18)
            # figsize = (36, 36)
            if last_token:
                features = ["First Type", "Second Type"]
            else:
                features = ["Type", "Length"]

            if df_subset is None:
                labels = ["-".join(v) for v in df["No_Stress"].values]
            else:
                labels = ["-".join(v) for v in df_subset["No_Stress"].values]

        if features_list:
            features = features_list

        # for name, emb in zip(["H", "C", "HC"], [h_emb, c_emb, hc_emb]):
        for name, emb in zip(["H"], [h_emb]):
            pca = PCA(n_components=2)
            if df_subset is None:
                pca_results = pca.fit_transform(emb)
            else:
                print("here")
                pca.fit(emb)
                sub_emb = np.array(df_subset[f"H{layer}"].to_list())  # type: ignore
                pca_results = pca.transform(sub_emb)
                df = df_subset

            if plot_mds:
                mds = MDS(n_components=2, random_state=42)
                mds_results = mds.fit_transform(emb)
                zipped = zip(["pca", "mds"], [pca_results, mds_results])

            else:
                zipped = zip(["pca"], [pca_results])

            for mtd, results in zipped:
                plt.figure(figsize=figsize)
                ax = plt.gca()

                feature1 = features[0]
                classes1 = sorted(df[feature1].unique())
                custom_colors = ["orange", "red", "black"]
                cmap = {
                    val: custom_colors[i % len(custom_colors)]
                    for i, val in enumerate(classes1)
                }
                # palette = plt.cm.get_cmap("viridis", len(classes1))
                # palette = plt.cm.get_cmap("tab10", len(classes1))
                # cmap = {val: palette(i) for i, val in enumerate(classes1)}

                if plot_txt:
                    plt.scatter(results[:, 0], results[:, 1], alpha=0)
                    # Add colored text labels
                    for i, txt in enumerate(labels):
                        txt_clr = cmap[df.iloc[i][feature1]]
                        plt.text(
                            results[i, 0],
                            results[i, 1],
                            txt,
                            fontsize=9,
                            ha="center",
                            va="center",
                            color=txt_clr,
                        )

                    # Legend for feature1 (colors)
                    color_handles = [
                        plt.Line2D(  # type: ignore
                            [0], [0], marker="o", color="w", markerfacecolor=cmap[val]
                        )
                        for val in classes1
                    ]
                    ncol = (len(classes1) // 5) + 1
                    ax.legend(
                        color_handles,
                        classes1,
                        title=feature1,
                        loc="upper left",
                        ncol=ncol,
                    )

                else:
                    feature2 = features[1]
                    classes2 = sorted(df[feature2].unique())
                    mrks = ["$1$", "$2$", "$3$", "$4$", "$5$", "$6$", "$7$", "$8$"]
                    # mrks = ["o", "s", "x", "D", "^", "v", ">", "<", "p", "*", "h"]
                    mmap = {val: mrks[i % len(mrks)] for i, val in enumerate(classes2)}

                    grouped = df.groupby(features).indices
                    for key, idx in grouped.items():
                        pts = results[idx, :]
                        clr = cmap[key[0]]  # type: ignore
                        mrk = mmap[key[1]]  # type: ignore
                        plt.scatter(pts[:, 0], pts[:, 1], color=clr, marker=mrk)  # type: ignore

                    color_handles = [
                        plt.Line2D(  # type: ignore
                            [0],
                            [0],
                            marker="o",
                            color="w",
                            markerfacecolor=cmap[val],
                        )
                        for val in classes1
                    ]
                    ncol = (len(classes1) // 5) + 1
                    legend1 = ax.legend(
                        color_handles,
                        classes1,
                        title=feature1,
                        loc="upper left",
                        ncol=ncol,
                    )
                    ax.add_artist(legend1)

                    marker_handles = [
                        plt.Line2D(  # type: ignore
                            [0],
                            [0],
                            marker=mmap[val],
                            color="k",
                            linestyle="",
                            markersize=6,
                        )
                        for val in classes2
                    ]
                    ncol = (len(classes2) // 5) + 1
                    ax.legend(
                        marker_handles,
                        classes2,
                        title=feature2,
                        loc="upper right",
                        ncol=ncol,
                    )

                plt.xticks([])
                plt.yticks([])
                # plt.tight_layout()
                if path:
                    filename = f"{dataset}_{mtd}_{name}{layer}{'_last' if last_token else ''}.png"
                    plt.savefig(path / filename, dpi=300)  # type: ignore
                    plt.close()
                else:
                    plt.show()


def custom_pca(
    pca: PCA,
    df: pd.DataFrame,
    features: list[str],
    plot_txt: bool = False,
    cell: bool = False,
) -> None:
    """Compute the PCA and MDS for a DataFrame of embedding vectors"""

    labels = ["-".join(v) for v in df["No_Stress"].values]

    if not cell:
        emb = np.array(df[f"H1"].to_list())
    else:
        emb = np.array(df[f"C1"].to_list())

    proj = pca.transform(emb)

    # get the projection of a zero vector
    zero_vector = np.zeros(emb.shape[1])
    proj_zero = pca.transform(zero_vector.reshape(1, -1))

    plt.figure(figsize=(18, 18))
    ax = plt.gca()

    f1 = features[0]
    v1 = sorted(df[f1].unique())
    # colors = ["blue", "red", "black"]
    # cmap = {val: colors[i % len(colors)] for i, val in enumerate(v1)}
    palette = plt.cm.get_cmap("viridis", len(v1))  # type: ignore
    # palette = plt.cm.get_cmap("tab10", len(v1))
    cmap = {val: palette(i) for i, val in enumerate(v1)}

    if plot_txt:
        plt.scatter(proj[:, 0], proj[:, 1], alpha=0)

        # Add colored text labels
        for i, txt in enumerate(labels):
            txt_clr = cmap[df.iloc[i][f1]]
            plt.text(
                proj[i, 0],
                proj[i, 1],
                txt,
                fontsize=9,
                ha="center",
                va="center",
                color=txt_clr,
            )

        # Legend for first feature
        color_handles = [
            plt.Line2D(  # type: ignore
                [0], [0], marker="o", color="w", markerfacecolor=cmap[val]
            )
            for val in v1
        ]
        ncol = (len(v1) // 5) + 1
        ax.legend(
            color_handles,
            v1,
            title=f1,
            loc="upper left",
            ncol=ncol,
        )

    else:
        f2 = features[1]
        v2 = sorted(df[f2].unique())
        # mrks = ["$C$", "$V$"]
        # mrks = ["$1$", "$2$", "$3$", "$4$", "$5$", "$6$", "$7$", "$8$"]
        mrks = ["o", "x", "D", "s", "^", "v", ">", "<", "p", "*", "h"]
        mmap = {val: mrks[i % len(mrks)] for i, val in enumerate(v2)}

        grouped = df.groupby(features).indices
        for key, idx in grouped.items():
            pts = proj[idx, :]
            clr = cmap[key[0]]  # type: ignore
            mrk = mmap[key[1]]  # type: ignore
            plt.scatter(pts[:, 0], pts[:, 1], color=clr, marker=mrk)  # type: ignore

        plt.scatter(
            proj_zero[0, 0],
            proj_zero[0, 1],
            color="black",
            marker="x",  # type: ignore
            s=300,
            # label="Zero Vector",
        )

        color_handles = [
            plt.Line2D(  # type: ignore
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=cmap[val],
            )
            for val in v1
        ]
        ncol = (len(v1) // 5) + 1
        legend1 = ax.legend(
            color_handles,
            v1,
            title=f1,
            loc="upper right",
            ncol=ncol,
        )
        ax.add_artist(legend1)

        marker_handles = [
            plt.Line2D(  # type: ignore
                [0],
                [0],
                marker=mmap[val],
                color="k",
                linestyle="",
                markersize=6,
            )
            for val in v2
        ]
        ncol = (len(v2) // 5) + 1
        ax.legend(
            marker_handles,
            v2,
            title=f2,
            loc="upper left",
            ncol=ncol,
        )

    plt.xticks([])
    plt.yticks([])
    plt.show()


def pca_types(pca: PCA, df: pd.DataFrame, key: str = "H1") -> None:
    """
    Compute PCA on embedding vectors and plot each point with a custom marker
    based on the 'Type' column in df. The 'Type' string indicates the phoneme
    structure (e.g., "V", "VC", "VCV", etc.).

    For 1 phoneme: Draw a marker (an "x" in this example) with a single color.
    For 2 phonemes: Draw a square split into two vertical halves.
    For 3 or more: Draw a circle divided into equal wedges.
    """
    # Convert the embedding column to a numpy array and project with PCA.
    emb = np.array(df[key].tolist())
    proj = pca.transform(emb)

    custom_palette = {
        "V": "#fc0000",
        "VV": "#fc0000",
        "VVV": "#fc0000",
        "VC": "#fc8b00",
        "VCC": "#fc8b00",
        "VCV": "#f4fc00",
        "VVC": "#32fc00",
        "C": "#0400fc",
        "CC": "#0400fc",
        "CCC": "#0400fc",
        "CV": "#9300fc",
        "CVV": "#9300fc",
        "CVC": "#00e7fc",
        "CCV": "#00fc43",
    }

    unique_types = sorted(df["Type"].unique())
    type_palette = plt.cm.get_cmap("tab10", len(unique_types))  # type: ignore
    type_cmap = {val: type_palette(i) for i, val in enumerate(unique_types)}

    fig, ax = plt.subplots(figsize=(18, 18))

    # Loop over each row to draw the custom marker.
    for i, row in df.iterrows():
        x, y = proj[i, 0], proj[i, 1]  # type: ignore
        phoneme_type = row["Type"]  # e.g., "V", "VC", "VCV", etc.
        length_val = str(row["Length"])
        text_color = custom_palette.get(phoneme_type, "black")

        ax.scatter(
            x, y, marker="o", color=type_cmap[phoneme_type], s=100, zorder=3, alpha=0  # type: ignore
        )
        ax.text(
            x, y, length_val, color=text_color, ha="center", va="center", fontsize=12
        )

    ax.set_aspect("equal")
    plt.xticks([])
    plt.yticks([])
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=tp,
            # markerfacecolor=type_cmap[tp],
            markerfacecolor=custom_palette.get(tp, "black"),
            markersize=10,
        )
        for tp in unique_types
    ]

    ax.legend(handles=legend_elements, loc="lower right")
    plt.show()


def find_best_n_clusters(
    df: pd.DataFrame,
    min_clusters: int = 2,
    max_clusters: int = 10,
    seed: int = 0,
) -> tuple[int, np.ndarray]:
    """Finds the optimal number of clusters for the points in `df` using KMeans and the silhouette method.

    Parameters
    ----------
    df : pandas dataframe
        Dataframe of points to cluster
    max_clusters : int, default=6
        Maximum number of clusters
    seed : int, default=0
        Random seed for reproducibility

    Returns
    -------
    best_num_clusters: int
        Optimal number of clusters
    clusters: array
        Cluster labels for each point
    """

    best_num_clusters = 0
    best_silhouette_score = -1
    for num_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=num_clusters, random_state=seed)
        labels = kmeans.fit_predict(df)
        silhouette_avg = silhouette_score(df, labels)

        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_num_clusters = num_clusters

    kmeans = KMeans(n_clusters=best_num_clusters, random_state=seed)
    clusters = kmeans.fit_predict(df)

    return best_num_clusters, clusters


# TODO: Danny rewrite
def mlem_importance(
    df: pd.DataFrame,
    dataset: str,
    num_layers: int,
    path: pathlib.Path,
    metric: str = "euclidean",
) -> None:
    """Use metric learning to derive feature importances on embeddings"""

    drops = ["Phonemes", "No_Stress", "Prediction", "H1", "C1"]

    if dataset == "phoneme":
        drops += ["Included", "Dipthong"]
    elif dataset == "evaluation":
        drops += ["Word", "Condition", "Size", "Frequency", "Length"]

    features = df.drop(columns=drops)
    feat_dists = feature_distances(features, verbose=0)

    num_layers = 1
    for layer in range(1, num_layers + 1):
        h_emb = np.array(df[f"H{layer}"].to_list())
        c_emb = np.array(df[f"C{layer}"].to_list())
        hc_emb = np.concatenate([h_emb, c_emb], axis=1)

        # for name, emb in zip(["H", "C", "HC"], [h_emb, c_emb, hc_emb]):
        for name, emb in zip(["H"], [h_emb]):
            repr_dists = representation_distances(emb, metric=metric, verbose=0)
            # plt.figure(figsize=(8, 6))
            # plt.imshow(repr_dists, cmap="viridis")
            # plt.colorbar(label=f"{metric.title()} Distance")
            # plt.xlabel("Stimulus Index")
            # plt.ylabel("Stimulus Index")
            # plt.show()

            results = mlem(
                repr_dists,
                feat_dists,
                features_df=features,
                outer_folds=2,
                inner_folds=3,
                n_permutations=50,
                random_state=0,
                verbose=0,
                n_jobs=-2,
                scale=True,
            )

            plt.figure(figsize=(8, 6))
            plt.bar(results["feature"], results["importance"], yerr=results["std"])
            # plt.xlabel("Features")
            plt.ylabel("Importance")
            plt.title(f"Feature Importance {name}{layer}")
            # plt.xticks(rotation=45)

            # Highlight significant features
            # TODO: does this work ?
            for i, is_significant in enumerate(results["significant"]):
                color = "green" if is_significant else "red"
                plt.text(
                    i,
                    results["importance"].iloc[i] + results["std"].iloc[i] + 0.01,
                    "*" if is_significant else "ns",
                    ha="center",
                    color=color,
                )
            plt.tight_layout()
            plt.savefig(path / f"{dataset}_mlem_{name}{layer}_{metric}.png", dpi=300)
            plt.close()


def mlem_univariate(
    df: pd.DataFrame,
    emb: np.ndarray,
    dataset: str,
    h_type: str = "H",
    metric: str = "euclidean",
    path: pathlib.Path | None = None,
):  # -> None:
    """Use metric learning to derive feature importances on embeddings

    Parameters:
        df (pd.DataFrame): DataFrame containing the features and embeddings.
        emb (np.ndarray): Embedding vectors extracted from the model.
        dataset (str): Name of the dataset (e.g., "phoneme", "bigram", "evaluation").
        path (pathlib.Path): Directory to save the plots.
        metric (str): Distance metric to use for representation distances.
    """
    drops = [
        "Word",
        "Condition",
        # "Lexicality",
        "Size",
        # "Morphology",
        "Frequency",
        # "Length",
        # "Zipf_Frequency",
        "Phonemes",
        "No_Stress",
        "Part of Speech",
        "Vowel Count",
        "Consonant Count",
        "Prediction",
        "Edit_Distance",
        "Insertions",
        "Deletions",
        "Substitutions",
        "Error_Indices",
    ]
    features = df.drop(columns=drops)
    feat_dists = feature_distances(features, verbose=0)

    batch_size = emb.shape[0]
    num_neurons = emb.shape[-1]

    if h_type == "H":
        # use lengths to get final time step
        lengths = df["Length"].to_numpy()
        emb = emb[np.arange(batch_size), lengths, :]
    elif h_type != "C":
        raise ValueError(f"Unknown hidden type: {h_type}")

    data = {"Neuron": [], "Feature": [], "Importance": [], "Spearman": []}

    for idx in range(num_neurons):
        print(f"Neuron {idx + 1}/{num_neurons}  ", end="\r")
        repr_dists = representation_distances(
            emb[:, idx : idx + 1], metric=metric, verbose=0
        )
        results = mlem(
            repr_dists,
            feat_dists,
            features_df=features,
            outer_folds=2,
            inner_folds=3,
            n_permutations=50,
            random_state=0,
            verbose=0,
            n_jobs=-2,
            scale=True,
        )

        for row in results.itertuples():
            data["Neuron"].append(idx)
            data["Feature"].append(row.feature)
            data["Importance"].append(row.importance)
            spearman = (row.test_score_0 + row.test_score_1) / 2  # type: ignore
            data["Spearman"].append(spearman)

    results_df = pd.DataFrame(data)
    return results_df

    # if path:
    #     plt.savefig(path / f"{dataset}_mlem_{h_type}_{metric}.png", dpi=300)
    #     plt.close()
    # else:
    #     plt.show()


if __name__ == "__main__":
    # Create a sample dataframe
    data = {
        "unit": [
            1,
            1,
            2,
            2,
            3,
            3,
        ],  # Units to be clustered
        "Feature": [
            "Noun",
            "Verb",
            "Noun",
            "Verb",
            "Noun",
            "Verb",
        ],  # Features (coordinates)
        "Importance": [0.1, 0.5, 0.05, 0.6, 0.4, 0.3],  # Feature importances
        "Spearman": [
            0.5,
            0.5,
            0.7,
            0.7,
            0.9,
            0.9,
        ],  # Encoding Spearman correlation (per unit)
    }
    df = pd.DataFrame(data)
    # Also try with "normalized" Feature Importance (uncomment below)
    # Normalize by the encoding performance of each unit, since they can have very different performance
    # df['Importance'] /= df.Spearman

    # Reshape the dataframe to have units as rows and Feature Importances as columns (= sklearn features)
    df_reshaped = df.pivot_table(index="unit", columns="Feature", values="Importance")

    best_num_clusters, clusters = find_best_n_clusters(df_reshaped, max_clusters=2)
    print(f"Optimal number of clusters: {best_num_clusters}")

    # Add the cluster labels to the original dataframe
    df_reshaped["cluster"] = clusters
    df_reshaped = df_reshaped.reset_index()
    df = df.merge(df_reshaped[["unit", "cluster"]])
    print(df)
