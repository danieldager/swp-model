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
from sklearn.decomposition import PCA
from sklearn.manifold import MDS


def norm_by_length(df: pd.DataFrame, key: str = "H1") -> None:

    # calculate the norm of the embedding vector
    df["Norm"] = df[key].apply(lambda x: np.linalg.norm(np.array(x), ord=2))  # type: ignore
    pass


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


# def custom_dmatrix(
#     df: pd.DataFrame,
#     metric: str = "euclidean",
#     key: str = "H1",
# ) -> None:

#     df = df.sort_values("Type")

#     emb = np.array(df[key].to_list())
#     dis = pdist(emb, metric)  # type: ignore
#     lnk = linkage(emb, method="ward")

#     figsize = (40, 24)
#     plt.figure(figsize=figsize)
#     gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4], wspace=0.1)

#     # dendrogram
#     ax_den = plt.subplot(gs[0])
#     # den = dendrogram(lnk, orientation="left")
#     # ax_den.set_xticks([])
#     # ax_den.set_yticks([])
#     # order = den["leaves"]

#     ax_den.axis("off")
#     order = list(range(len(df)))
#     # labels = [word for word in df["Type"].values[order]]
#     labels = df["Type"].tolist()

#     # dissimilarity matrix
#     ax_mat = plt.subplot(gs[1])
#     matrix = squareform(dis)[order, :][:, order][::-1, :]
#     vmin = np.percentile(matrix, 1)
#     vmax = np.percentile(matrix, 99)
#     norm = TwoSlopeNorm(np.median(matrix), vmin, matrix.max())  # type: ignore
#     im = ax_mat.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm)
#     ax_mat.set_xticks(np.arange(len(labels)))
#     ax_mat.set_yticks(np.arange(len(labels)))
#     ax_mat.set_xticklabels([])
#     # y_labels = [label if i % 10 == 0 else "" for i, label in enumerate(labels[::-1])]
#     reversed_labels = labels[::-1]
#     y_labels = []
#     prev_type = None
#     for label in reversed_labels:
#         if label != prev_type:
#             y_labels.append(label)
#         else:
#             y_labels.append("")
#         prev_type = label
#     ax_mat.set_yticklabels(y_labels)
#     plt.colorbar(im, ax=ax_mat)
#     plt.show()


def custom_dmatrix(
    df: pd.DataFrame,
    metric: str = "euclidean",
    key: str = "H1",
) -> None:
    custom_order = {
        "C": 0,
        "V": 1,
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
    vmin = np.percentile(matrix, 20)
    vmax = np.percentile(matrix, 80)
    norm = TwoSlopeNorm(np.median(matrix), vmin, vmax)  # type: ignore

    fig, ax = plt.subplots(figsize=(20, 14))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(y_labels[::-1])
    ax.set_yticklabels(y_labels)
    plt.colorbar(im, ax=ax)
    plt.show()


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
        elif dataset == "trigram":
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
    pca: PCA,
    df: pd.DataFrame,
    features: list[str],
    plot_txt: bool = False,
    cell: bool = False,
) -> None:
    """Compute the PCA and MDS for a DataFrame of embedding vectors"""

    labels = ["-".join(v) for v in df["No Stress"].values]

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
    palette = plt.cm.get_cmap("viridis", len(v1))
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
            plt.scatter(pts[:, 0], pts[:, 1], color=clr, marker=mrk)

        plt.scatter(
            proj_zero[0, 0],
            proj_zero[0, 1],
            color="black",
            marker="x",
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


# def pca_types(
#     pca: PCA, df: pd.DataFrame, key: str = "H1", wedges: bool = False
# ) -> None:
#     """
#     Compute PCA on embedding vectors and plot each point with a custom marker
#     based on the 'Type' column in df. The 'Type' string indicates the phoneme
#     structure (e.g., "V", "VC", "VCV", etc.).

#     For 1 phoneme: Draw a marker (an "x" in this example) with a single color.
#     For 2 phonemes: Draw a square split into two vertical halves.
#     For 3 or more: Draw a circle divided into equal wedges.
#     """
#     # Convert the embedding column to a numpy array and project with PCA.
#     emb = np.array(df[key].tolist())
#     proj = pca.transform(emb)

#     # Define a color mapping for phoneme types.
#     # You can change these colors as needed.
#     color_map = {"V": "blue", "C": "red"}

#     unique_types = sorted(df["Type"].unique())
#     type_palette = plt.cm.get_cmap("tab10", len(unique_types))
#     type_cmap = {val: type_palette(i) for i, val in enumerate(unique_types)}

#     fig, ax = plt.subplots(figsize=(18, 18))

#     # Loop over each row to draw the custom marker.
#     for i, row in df.iterrows():
#         x, y = proj[i, 0], proj[i, 1]  # type: ignore
#         phoneme_type = row["Type"]  # e.g., "V", "VC", "VCV", etc.
#         length_val = str(row["Length"])
#         text_color = type_cmap.get(row["Type"], "black")

#         if not wedges:
#             # ax.scatter(x, y, marker="o", color=type_cmap[phoneme_type], s=100, zorder=3)
#             plt.scatter(x, y, alpha=0)
#             ax.text(
#                 x,
#                 y,
#                 length_val,
#                 color=text_color,
#                 ha="center",
#                 va="center",
#                 fontsize=12,
#             )

#             continue

#         if len(phoneme_type) == 1:
#             # For a single phoneme, plot an "x" marker (or a filled circle)
#             # Here we choose an "x" marker.
#             ax.scatter(
#                 x,
#                 y,
#                 marker="x",
#                 color=color_map.get(phoneme_type, "gray"),
#                 s=100,
#                 zorder=3,
#             )

#         elif len(phoneme_type) == 2:
#             # For two phonemes, draw a square divided into 2 vertical halves.
#             size = 0.2  # adjust size as needed
#             # Left half rectangle.
#             left_rect = patches.Rectangle(
#                 (x - size / 2, y - size / 2),
#                 width=size / 2,
#                 height=size,
#                 facecolor=color_map.get(phoneme_type[0], "gray"),
#                 lw=0.0,
#                 zorder=2,
#             )
#             # Right half rectangle.
#             right_rect = patches.Rectangle(
#                 (x, y - size / 2),
#                 width=size / 2,
#                 height=size,
#                 facecolor=color_map.get(phoneme_type[1], "gray"),
#                 lw=0.0,
#                 zorder=2,
#             )
#             ax.add_patch(left_rect)
#             ax.add_patch(right_rect)

#         else:
#             # For three or more phonemes, draw a circle divided into equal wedges.
#             num_segments = len(phoneme_type)
#             angle = 360 / num_segments
#             radius = 0.05  # adjust as needed for the tri-circles
#             for j, phon in enumerate(phoneme_type):
#                 center_angle = 90 - j * angle
#                 theta1 = center_angle - angle / 2
#                 theta2 = center_angle + angle / 2
#                 wedge = patches.Wedge(
#                     (x, y),
#                     radius,
#                     theta1,
#                     theta2,
#                     facecolor=color_map.get(phon, "gray"),
#                     edgecolor="none",
#                     zorder=2,
#                 )
#                 ax.add_patch(wedge)

#     ax.set_aspect("equal")
#     plt.xticks([])
#     plt.yticks([])
#     if not wedges:
#         legend_elements = [
#             Line2D(
#                 [0],
#                 [0],
#                 marker="o",
#                 color="w",
#                 label=tp,
#                 markerfacecolor=type_cmap[tp],
#                 markersize=10,
#             )
#             for tp in unique_types
#         ]
#     else:
#         legend_elements = [
#             Line2D(
#                 [0],
#                 [0],
#                 marker="o",
#                 color="w",
#                 label="Vowel",
#                 markerfacecolor="blue",
#                 markersize=10,
#             ),
#             Line2D(
#                 [0],
#                 [0],
#                 marker="o",
#                 color="w",
#                 label="Consonant",
#                 markerfacecolor="red",
#                 markersize=10,
#             ),
#         ]
#     ax.legend(handles=legend_elements, loc="lower right")
#     plt.show()


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
    type_palette = plt.cm.get_cmap("tab10", len(unique_types))
    type_cmap = {val: type_palette(i) for i, val in enumerate(unique_types)}

    fig, ax = plt.subplots(figsize=(18, 18))

    # Loop over each row to draw the custom marker.
    for i, row in df.iterrows():
        x, y = proj[i, 0], proj[i, 1]  # type: ignore
        phoneme_type = row["Type"]  # e.g., "V", "VC", "VCV", etc.
        length_val = str(row["Length"])
        text_color = custom_palette.get(phoneme_type, "black")

        ax.scatter(
            x, y, marker="o", color=type_cmap[phoneme_type], s=100, zorder=3, alpha=0
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
