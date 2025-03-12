import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import PowerNorm, TwoSlopeNorm
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import MDS


def dissimilarity_matrix(
    df: pd.DataFrame, num_layers: int, figures_dir: pathlib.Path
) -> None:
    """Compute the dissimilarity matrix for a DataFrame of embeddings"""
    for layer in range(1, num_layers + 1):
        h_emb = np.array(df[f"H{layer}"].to_list())
        c_emb = np.array(df[f"C{layer}"].to_list())
        hc_emb = np.concatenate([h_emb, c_emb], axis=1)

        for name, emb in zip(["H", "C", "HC"], [h_emb, c_emb, hc_emb]):
            dis = pdist(emb, metric="euclidean")
            lnk = linkage(emb, method="ward")

            plt.figure(figsize=(19, 12))
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4], wspace=0.05)

            ax_den = plt.subplot(gs[0])
            den = dendrogram(lnk, orientation="left")
            ax_den.set_xticks([])
            ax_den.set_yticks([])
            order = den["leaves"]
            labels = [v[0] for v in df["No Stress"].values[order]]

            ax_mat = plt.subplot(gs[1])
            mat = squareform(dis)[order, :][:, order][::-1, :]
            norm = TwoSlopeNorm(np.median(mat), mat.min() + 3, mat.max())  # type: ignore
            # norm = PowerNorm(gamma=5, vmin=mat.min(), vmax=mat.max())
            im = ax_mat.imshow(mat, aspect="auto", cmap="RdBu_r", norm=norm)
            ax_mat.set_xticks(np.arange(len(labels)))
            ax_mat.set_yticks(np.arange(len(labels)))
            ax_mat.set_xticklabels(labels)
            ax_mat.set_yticklabels(labels[::-1])
            plt.colorbar(im, ax=ax_mat)
            plt.tight_layout()
            plt.savefig(figures_dir / f"dissimilarity_{name}{layer}.png")
            plt.close()


def embeddings_PCA_MDS(
    df: pd.DataFrame, num_layers: int, figures_dir: pathlib.Path
) -> None:
    """Compute the PCA and MDS for a DataFrame of embedding vectors"""
    for layer in range(1, num_layers + 1):
        h_emb = np.array(df[f"H{layer}"].to_list())
        c_emb = np.array(df[f"C{layer}"].to_list())
        hc_emb = np.concatenate([h_emb, c_emb], axis=1)
        labels = [v[0] for v in df["No Stress"].values]

        for name, emb in zip(["H", "C", "HC"], [h_emb, c_emb, hc_emb]):
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(emb)
            plt.figure(figsize=(6, 6))
            plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0)
            for i, txt in enumerate(labels):
                plt.text(
                    pca_result[i, 0],
                    pca_result[i, 1],
                    txt,
                    fontsize=10,
                    ha="center",
                    va="center",
                )
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(figures_dir / f"PCA_{name}{layer}.png")
            plt.close()

            mds = MDS(n_components=2, dissimilarity="euclidean", random_state=42)
            mds_result = mds.fit_transform(emb)
            plt.figure(figsize=(6, 6))
            plt.scatter(mds_result[:, 0], mds_result[:, 1], alpha=0)
            for i, txt in enumerate(labels):
                plt.text(
                    mds_result[i, 0],
                    mds_result[i, 1],
                    txt,
                    fontsize=10,
                    ha="center",
                    va="center",
                )
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(figures_dir / f"MDS_{name}{layer}.png")
            plt.close()
