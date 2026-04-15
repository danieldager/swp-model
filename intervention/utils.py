import plotly.graph_objects as go
from sklearn.decomposition import PCA
import plotly.colors as pc

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