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
    figures_dir: pathlib.Path | None,
    last_token: bool = False,
    features_list: list[str] = [],
    plot_mds: bool = True,
    plot_txt: bool = False,
    df_subset: pd.DataFrame | None = None,
) -> None:
    """Compute the PCA and MDS for a DataFrame of embedding vectors"""
    # df["First Token"] = df["No Stress"].apply(lambda x: x[0])
    # df["Second Token"] = df["No Stress"].apply(lambda x: x[1])
    # df["Last Token"] = df["No Stress"].apply(lambda x: x[-1])

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
            labels = [v[0] for v in df["No Stress"].values]
        elif dataset == "bigram":
            figsize = (12, 12)
            if last_token:
                features = ["First Type", "Type"]
            else:
                features = ["Place", "Height"]
            labels = ["-".join(v) for v in df["No Stress"].values]
        elif dataset == "evaluation":
            figsize = (12, 12)
            if last_token:
                features = ["First Type", "Second Type"]
                # features = ["Length", "First Type"]
            else:
                features = ["Length", "Lexicality"]
            labels = [word for word in df["Word"].values]
        elif dataset == "full_trigram":
            figsize = (12, 12)
            # figsize = (18, 18)
            # figsize = (36, 36)
            if last_token:
                features = ["First Type", "Second Type"]
            else:
                features = ["Type", "Length"]

            if df_subset is None:
                labels = ["-".join(v) for v in df["No Stress"].values]
            else:
                labels = ["-".join(v) for v in df_subset["No Stress"].values]

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

                plt.xticks([])
                plt.yticks([])
                # plt.tight_layout()
                if figures_dir:
                    filename = f"{dataset}_{mtd}_{name}{layer}{'_last' if last_token else ''}.png"
                    plt.savefig(figures_dir / filename, dpi=300)  # type: ignore
                    plt.close()
                else:
                    plt.show()


def custom_pca(
    df1: pd.DataFrame,
    dataset: str,
    num_layers: int,
    figures_dir: pathlib.Path | None,
    last_token: bool = False,
    features_list: list[str] = [],
    plot_txt: bool = False,
    df2: pd.DataFrame | None = None,
) -> None:
    """Compute the PCA and MDS for a DataFrame of embedding vectors"""

    emb = np.array(df1[f"H1"].to_list())
    # c_emb = np.array(df[f"C1"].to_list())
    # hc_emb = np.concatenate([h_emb, c_emb], axis=1)

    figsize = (12, 12)
    # figsize = (18, 18)
    # figsize = (36, 36)

    if df2 is None:
        labels = ["-".join(v) for v in df1["No Stress"].values]
    else:
        labels = ["-".join(v) for v in df2["No Stress"].values]

    if features_list:
        features = features_list

        pca = PCA(n_components=2)
        if df2 is None:
            pca_results = pca.fit_transform(emb)
        else:
            print("here")
            pca.fit(emb)
            sub_emb = np.array(df_subset[f"H{layer}"].to_list())  # type: ignore
            pca_results = pca.transform(sub_emb)
            df = df2
  
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

            plt.xticks([])
            plt.yticks([])
            plt.show()
