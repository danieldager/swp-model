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
import plotly.graph_objects as go
import plotly.colors as pc

#  Setup
seed_everything(42)
device = set_device()
model_name = "Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1"
weights_path = "../reproduce/weights/1024_75.pth"

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
# load datasets (StatesDataset)
train_ds = StatesDataset.load("states_ds/train_states") # train dataset with 30k real words
test_ds = StatesDataset.load("states_ds/test_states") # 1200 words real and pseudo words


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

def pca_2d_by_column(df, embeddings, column_name, state_name):
    pca = PCA(n_components=2)
    # standardized_data = StandardScaler().fit_transform(embeddings)
    state_pca = pca.fit_transform(embeddings)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=state_pca[:,0], y=-state_pca[:,1], hue=df[column_name], alpha=0.7)
    plt.title(f"Word Embeddings in PCA Space - {state_name} - {column_name}")
    plt.xlabel("PC1 variance: {:.2f}%".format(pca.explained_variance_ratio_[0] * 100))
    plt.ylabel("PC2 variance: {:.2f}%".format(pca.explained_variance_ratio_[1] * 100))
    plt.legend(title=column_name)
    plt.tight_layout()
    plt.savefig(f"plots/pca_test_words/pca_{state_name}_{column_name}.png")


def pca3d_by_type(meta, embeddings,path):
    pca_3d = PCA(n_components=3)
    delta_h_3d = pca_3d.fit_transform(embeddings)

    # Prepare metadata
    # meta = dataset.metadata.copy()
    meta['pca_1'] = delta_h_3d[:, 0]
    meta['pca_2'] = delta_h_3d[:, 1]
    meta['pca_3'] = delta_h_3d[:, 2]

    # Filter positions < 9
    meta = meta[meta['position'] < 9].reset_index(drop=True)

    # Create traces for each coloring scheme
    # 1. By Position (with better color differentiation)
    positions = sorted(meta['position'].unique())
    colors_pos = {pos: i for i, pos in enumerate(positions)}
    meta['color_pos'] = meta['position'].map(colors_pos)

    trace_pos = go.Scatter3d(
        x=meta['pca_1'], y=meta['pca_2'], z=meta['pca_3'],
        mode='markers',
        marker=dict(size=4, color=meta['color_pos'], colorscale='Viridis', 
                    showscale=True, colorbar=dict(title="Position", tickvals=list(colors_pos.values()), 
                                                ticktext=[str(p) for p in positions])),
        text=[f"Pos: {p}" for p in meta['position']],
        name='By Position',
        visible=True
    )

    # 2. By Type
    type_colors_map = {'V': 0, 'C': 1, 'EOS': 2}
    meta['color_type'] = meta['phoneme_type'].map(type_colors_map)

    trace_type = go.Scatter3d(
        x=meta['pca_1'], y=meta['pca_2'], z=meta['pca_3'],
        mode='markers',
        marker=dict(size=4, color=meta['color_type'], 
                    colorscale=[[0, '#FF9500'], [0.5, '#0033CC'], [1, '#999999']],
                    showscale=True, colorbar=dict(title="Type", tickvals=[0, 0.5, 1],
                                                ticktext=['V', 'C', 'EOS'])),
        text=[f"Type: {t}" for t in meta['phoneme_type']],
        name='By Type',
        visible=False
    )

    # 3. By Phoneme Identity (with discrete colors for each phoneme)
    phonemes = sorted(meta['phoneme'].unique())
    phoneme_map = {p: i for i, p in enumerate(phonemes)}
    meta['color_phoneme'] = meta['phoneme'].map(phoneme_map)

    # Use a colorscale with good separation
    trace_phoneme = go.Scatter3d(
        x=meta['pca_1'], y=meta['pca_2'], z=meta['pca_3'],
        mode='markers',
        marker=dict(
            size=4, 
            color=meta['color_phoneme'], 
            colorscale='Rainbow',
            showscale=True, 
            colorbar=dict(
                title="Phoneme",
                tickvals=list(range(0, len(phonemes), max(1, len(phonemes)//10))),
                ticktext=[phonemes[i] for i in range(0, len(phonemes), max(1, len(phonemes)//10))]
            )
        ),
        text=[f"Phoneme: {p}" for p in meta['phoneme']],
        name='By Phoneme',
        visible=False
    )

    # Create figure with all traces
    fig = go.Figure(data=[trace_pos, trace_type, trace_phoneme])

    # Add dropdown buttons
    buttons = [
        dict(label="By Position",
            method="update",
            args=[{"visible": [True, False, False]},
                {"title": "PCA 3D — Colored by Position (pos < 9)"}]),
        dict(label="By Type",
            method="update",
            args=[{"visible": [False, True, False]},
                {"title": "PCA 3D — Colored by Phoneme Type (pos < 9)"}]),
        dict(label="By Phoneme Identity",
            method="update",
            args=[{"visible": [False, False, True]},
                {"title": "PCA 3D — Colored by Phoneme (pos < 9)"}]),
    ]

    fig.update_layout(
        updatemenus=[dict(active=0, buttons=buttons, x=0.1, y=0.9)],
        title="PCA 3D — Colored by Position (pos < 9)",
        scene=dict(
            xaxis_title=f"PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})",
            yaxis_title=f"PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})",
            zaxis_title=f"PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})",
        ),
        height=800,
        hovermode='closest',
    )


    # Save as HTML
    fig.write_html(path)

def pca3d_by_position(dataset, embeddings, path):
    # Compute PCA
    pca_3d = PCA(n_components=3)
    delta_h_3d = pca_3d.fit_transform(embeddings)

    # Prepare metadata
    meta = dataset.metadata.copy()
    meta['pca_1'] = delta_h_3d[:, 0]
    meta['pca_2'] = delta_h_3d[:, 1]
    meta['pca_3'] = delta_h_3d[:, 2]

    # Filter positions < 9
    meta = meta[meta['position'] < 9].reset_index(drop=True)

    # Get phoneme lists
    all_phonemes = sorted(meta['phoneme'].unique())
    consonants = sorted(meta[meta['phoneme_type'] == 'C']['phoneme'].unique())
    vowels = sorted(meta[meta['phoneme_type'] == 'V']['phoneme'].unique())
    positions = sorted(meta['position'].unique())

    # Create traces for each combination of phoneme/category and position coloring
    traces = []
    trace_visibility = []

    # Define phoneme groups
    phoneme_groups = {
        'All': all_phonemes,
        'Consonants': consonants,
        'Vowels': vowels,
    }
    phoneme_groups.update({p: [p] for p in all_phonemes})  # add individual phonemes

    # Position color mapping
    colors_pos = {pos: i for i, pos in enumerate(positions)}

    for group_name, phoneme_list in phoneme_groups.items():
        # Filter data for this group
        group_mask = meta['phoneme'].isin(phoneme_list)
        group_meta = meta[group_mask].copy()
        
        if len(group_meta) == 0:
            continue
        
        # Color by position
        group_meta['color_pos'] = group_meta['position'].map(colors_pos)
        
        trace = go.Scatter3d(
            x=group_meta['pca_1'], 
            y=group_meta['pca_2'], 
            z=group_meta['pca_3'],
            mode='markers',
            marker=dict(
                size=5, 
                color=group_meta['color_pos'], 
                colorscale='Viridis',
                showscale=True, 
                colorbar=dict(
                    title="Position",
                    tickvals=list(colors_pos.values()), 
                    ticktext=[str(p) for p in positions]
                )
            ),
            text=[f"Pos: {p}, Phoneme: {ph}" for p, ph in zip(group_meta['position'], group_meta['phoneme'])],
            name=f'{group_name}',
            visible=False
        )
        traces.append(trace)

    # Create figure with all traces
    fig = go.Figure(data=traces)

    # Create dropdown buttons for phoneme/category selection
    buttons = []
    for i, group_name in enumerate(phoneme_groups.keys()):
        if i < len(traces):  # Only add if trace exists
            visibility = [False] * len(traces)
            visibility[i] = True
            
            buttons.append(
                dict(
                    label=group_name,
                    method="update",
                    args=[
                        {"visible": visibility},
                        {"title": f"PCA 3D — {group_name} colored by Position (pos < 9)"}
                    ]
                )
            )

    # Set first valid button as active
    fig.data[0].visible = True

    fig.update_layout(
        updatemenus=[dict(active=0, buttons=buttons, x=0.05, y=0.95, xanchor='left', yanchor='top')],
        title="PCA 3D — All phonemes colored by Position (pos < 9)",
        scene=dict(
            xaxis_title=f"PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})",
            yaxis_title=f"PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})",
            zaxis_title=f"PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})",
        ),
        height=800,
        hovermode='closest',
        font=dict(size=10),
    )

    # Save as HTML
    fig.write_html(path)


if __name__ == "__main__":
    # norms plots
    for embed in ["h", "delta_h", "c", "delta_c", "state", "delta_state"]:
         norms_boxplot(train_ds, embed_type=embed, path=f"plots/norms/norms_boxplot_{embed}.png")

    # pca plots for train 
    for column in ["position", "phoneme_type"]:
        for embed in ["delta_state", "state", "delta_h", "h", "delta_c", "c"]:
            pca_2d(train_ds, embed_type=embed, column_name=column, path=f"plots/pca_train_general/pca_{embed}_{column}.png")
    
    # pca plots for test
    for column in ["position", "phoneme_type"]:
        for embed in ["delta_state", "state", "delta_h", "h", "delta_c", "c"]:
            pca_2d(test_ds, embed_type=embed, column_name=column, path=f"plots/pca_test_general/pca_{embed}_{column}.png")
    # color based on wfe features
    words_mask = test_ds.get_mask(phoneme="<EOS>")
    words_state = test_ds.state[words_mask]
    words_h = test_ds.h[words_mask]
    words_c = test_ds.c[words_mask]

    df = test_ds.metadata[words_mask].copy()
    # merge with wfe (all columns) by matching word
    wfe = pd.read_csv("datasets/wfe_with_repetition.csv", converters=converters)
    df = df.merge(wfe, left_on="word", right_on="Word", how="left")

    for f in ["Lexicality", "Size", "Morphology", "Length", "can_repeat"]:
        pca_2d_by_column(df, words_state, f, "state")
        pca_2d_by_column(df, words_h, f, "h")
        pca_2d_by_column(df, words_c, f, "c")

