import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import PowerNorm, TwoSlopeNorm
from mlem_minimal import feature_distances, mlem, representation_distances
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import MDS


def dissim_matrix(
    df: pd.DataFrame,
    dataset: str,
    num_layers: int,
    figures_dir: pathlib.Path,
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
                labels = [v[0] for v in df["No Stress"].values[order]]
            elif dataset == "bigram":
                full = False
                labels = ["-".join(v) for v in df["No Stress"].values[order]]
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
            plt.savefig(
                figures_dir / f"{dataset}_dmatrix_{name}{layer}_{metric}.png", dpi=300
            )
            plt.close()


def mlem_importance(
    df: pd.DataFrame,
    dataset: str,
    num_layers: int,
    figures_dir: pathlib.Path,
    metric: str = "euclidean",
) -> None:
    """Use metric learning to derive feature importances on embeddings"""

    drops = ["Phonemes", "No Stress", "Prediction", "H1", "C1"]

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
            plt.savefig(
                figures_dir / f"{dataset}_mlem_{name}{layer}_{metric}.png", dpi=300
            )
            plt.close()


def pca_mds(
    df: pd.DataFrame,
    dataset: str,
    num_layers: int,
    figures_dir: pathlib.Path,
    last_token: bool = False,
) -> None:
    """Compute the PCA and MDS for a DataFrame of embedding vectors"""
    df["Last Token"] = df["No Stress"].apply(lambda x: x[-1])

    for layer in range(1, num_layers + 1):
        h_emb = np.array(df[f"H{layer}"].to_list())
        c_emb = np.array(df[f"C{layer}"].to_list())
        hc_emb = np.concatenate([h_emb, c_emb], axis=1)
        if dataset == "phoneme":
            features = []
            figsize = (6, 6)
            labels = [v[0] for v in df["No Stress"].values]
        elif dataset == "bigram":
            figsize = (12, 12)
            if last_token:
                features = ["Last Token", "Type"]
            else:
                features = ["Place", "Height"]
            labels = ["-".join(v) for v in df["No Stress"].values]
        elif dataset == "evaluation":
            figsize = (12, 12)
            if last_token:
                features = ["Last Token", "Length"]
            else:
                features = ["Length", "Lexicality"]
            labels = [word for word in df["Word"].values]

        # for name, emb in zip(["H", "C", "HC"], [h_emb, c_emb, hc_emb]):
        for name, emb in zip(["H"], [h_emb]):
            pca = PCA(n_components=2)
            pca_results = pca.fit_transform(emb)

            mds = MDS(n_components=2, random_state=42)
            mds_results = mds.fit_transform(emb)

            for mtd, results in zip(["pca", "mds"], [pca_results, mds_results]):
                plt.figure(figsize=figsize)
                ax = plt.gca()

                if features:
                    feature1 = features[0]
                    classes1 = sorted(df[feature1].unique())
                    palette = plt.cm.get_cmap("viridis", len(classes1))
                    cmap = {val: palette(i) for i, val in enumerate(classes1)}

                    feature2 = features[1]
                    classes2 = sorted(df[feature2].unique())
                    mrks = ["o", "x", "s", "D", "^", "v", ">", "<", "p", "*", "h"]
                    mmap = {val: mrks[i % len(mrks)] for i, val in enumerate(classes2)}

                    grouped = df.groupby(features).indices
                    for key, idx in grouped.items():
                        pts = results[idx, :]
                        clr = cmap[key[0]]  # type: ignore
                        mrk = mmap[key[1]]  # type: ignore
                        plt.scatter(pts[:, 0], pts[:, 1], color=clr, marker=mrk)

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

                else:
                    plt.scatter(results[:, 0], results[:, 1], alpha=0)
                    for i, txt in enumerate(labels):
                        plt.text(
                            results[i, 0],
                            results[i, 1],
                            txt,
                            fontsize=10,
                            ha="center",
                            va="center",
                        )

                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(
                    figures_dir
                    / f"{dataset}_{mtd}_{name}{layer}{'_last' if last_token else ''}.png",
                    dpi=300,
                )
                plt.close()
