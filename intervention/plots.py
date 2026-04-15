#%%
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional
from states_extract import StateExtractor, StatesDataset
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
from swp.utils.setup import seed_everything, set_device
from swp.utils.models import get_model
from swp.datasets.phonemes import get_phoneme_to_id
from ast import literal_eval
from sklearn.decomposition import PCA
from typing import Optional
#%%

seed_everything(42)
device = set_device()

model_name = "Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1"
weights_path = "../reproduce/weights/1024_75.pth"

#  Setup
model = get_model(model_name)
model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
model.to(device)
model.eval()
phoneme_to_id = get_phoneme_to_id()
id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}
converters = {"Word": str, "Phonemes": literal_eval, "No_Stress": literal_eval}

extractor = StateExtractor(model, phoneme_to_id, device)
# Load phoneme categories
phoneme_data = pd.read_csv("datasets/phonemes.csv")
vowels = phoneme_data["Phoneme"][phoneme_data["Type"] == "V"].tolist()
consonants = phoneme_data["Phoneme"][phoneme_data["Type"] == "C"].tolist()

train_ds = StatesDataset.load("states_ds/train_states_with_hc") # train dataset with 30k real words
test_ds = StatesDataset.load("states_ds/wfe_states") # 1200 words real and pseudo words
# %%

def norms(embeddings: np.ndarray) -> np.ndarray:
    """L2 norm of each row. Input: (N, D) → Output: (N,)"""
    return np.linalg.norm(embeddings, axis=-1)

def norms_boxplot(dataset, embed_type: str = "h", path: Optional[str] = None):
    """Boxplot of embedding norms grouped by position."""
    all_norms = norms(dataset.get_embeddings(embed_type))
    df = dataset.metadata.copy()
    df["norm"] = all_norms
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x="position", y="norm", data=df, ax=ax)
    title = f"{embed_type} Norms by Position"
    ax.set_title(title or f"{embed_type} Norms by Position")
    plt.tight_layout()
    if path:
        plt.savefig(path)

# %%
def pca_2d(dataset, embed_type: str = "delta_state",column_name: str = "position", max_pos: int = 9, path: Optional[str] = None):
    """PCA 2D scatter plot colored by position or phoneme type."""
    pos_mask = dataset.metadata["position"] < max_pos
    embeddings = dataset.get_embeddings(embed_type)[pos_mask]
    pca = PCA(n_components=2)
    pca_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    if column_name == "position":
        scatter = plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=dataset.metadata.loc[pos_mask, "position"], cmap="winter", alpha=0.7)
        plt.colorbar(scatter, label="Position")
    elif column_name == "phoneme_type":
        type_colors = {"V": "orange", "C": "blue", "EOS": "gray"}
        colors = dataset.metadata.loc[pos_mask, "phoneme_type"].map(type_colors)
        scatter = plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=colors, alpha=0.7)
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8) 
               for color in type_colors.values()]
        plt.legend(handles=handles, labels=type_colors.keys(), title="Phoneme Type")
    
    plt.xlabel("PC1 variance: {:.2f}%".format(pca.explained_variance_ratio_[0] * 100))
    plt.ylabel("PC2 variance: {:.2f}%".format(pca.explained_variance_ratio_[1] * 100))
    plt.title(f"PCA of {embed_type} colored by {column_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if path:
        plt.savefig(path)

# %%
# norms plots
norms_boxplot(train_ds, embed_type="h", path="plots/important/norms_boxplot_h.png")
norms_boxplot(train_ds, embed_type="delta_h", path="plots/important/norms_boxplot_delta_h.png")
norms_boxplot(train_ds, embed_type="c", path="plots/important/norms_boxplot_c.png")
norms_boxplot(train_ds, embed_type="delta_c", path="plots/important/norms_boxplot_delta_c.png")
norms_boxplot(train_ds, embed_type="state", path="plots/important/norms_boxplot_state.png")
norms_boxplot(train_ds, embed_type="delta_state", path="plots/important/norms_boxplot_delta_state.png")

# %%
# pca plots
for column in ["position", "phoneme_type"]:
    for embed in ["delta_state", "state", "delta_h", "h", "delta_c", "c"]:
        pca_2d(train_ds, embed_type=embed, column_name=column, path=f"plots/important/pca_{embed}_{column}.png")

# %%
