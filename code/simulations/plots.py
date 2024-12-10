import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Optional

""" PATHS """
FILE_DIR = Path(__file__).resolve()
ROOT_DIR = FILE_DIR.parent.parent.parent
FIGURES_DIR = ROOT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

sns.set_palette("colorblind")

# Plot the training and validation loss curves
def training_curves(train_losses: list, valid_losses: list, model: str, n_epochs: int):
    # Extract parameters from the model name
    h, l, d, t, r = [p[1:] for p in model.split('_')[1:]]

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(1, n_epochs + 1), y=train_losses, label='Training')
    sns.lineplot(x=range(1, n_epochs + 1), y=valid_losses, label='Validation')
    plt.title(f'Model: H={h}, L={l}, D={d}, TF={t}, LR={r}')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.tight_layout()

    MODEL_FIGURES_DIR = FIGURES_DIR / model
    MODEL_FIGURES_DIR.mkdir(exist_ok=True)
    filename = MODEL_FIGURES_DIR / 'training_curves.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Plot the edit operations and distance for each test category
def error_plots(df: pd.DataFrame, model: str, epoch: Optional[int] = None) -> None:
    fig, (length_ax, freq_ax, errors_ax) = plt.subplots(3, 1, figsize=(12, 16))
    # fig.suptitle(f"{model} (Epoch {epoch})" if epoch else model, fontsize=16, y=0.9)

    """ Figure 1: Edit Distance by Length """
    data = df.copy()

    # Calculate mean edit distance and variance
    data['Lex_Morphology'] = data['Lexicality'] + '-' + data['Morphology']
    
    grouped_df = data.groupby(
        ['Length', 'Lex_Morphology'], observed=True)['Edit Distance'].mean().reset_index()
    
    grouped_df['Edit Distance Std'] = data.groupby(
        ['Length', 'Lex_Morphology'], observed=True)['Edit Distance'].std().reset_index()['Edit Distance']
    
    sns.lineplot(
        data=grouped_df,
        x='Length',
        y='Edit Distance',
        hue='Lex_Morphology',
        markers=True,
        marker='o',
        markersize=8,
        ax=length_ax
    )
    # length_ax.errorbar(
    #     x=grouped_df['Length'], 
    #     y=grouped_df['Edit Distance'], 
    #     yerr=grouped_df['Edit Distance Std'],
    #     # fmt='none',  # No line markers
    #     ecolor='black',  # Color of error bars
    #     capsize=3,  # Cap width
    #     alpha=0.5,  # Transparency
    #     zorder=0  # Ensure error bars are behind the line plot
    # )
    length_ax.set_title('Edit Distance by Length')
    length_ax.set_xlabel('Word Length')
    length_ax.set_ylabel('Average Edit Distance')
    length_ax.legend(title='Lexicality & Morphology')
    length_ax.grid(True)
    
    """ Figure 2: Frequency vs Edit Distance """
    data = df[df['Lexicality'] == 'real'].copy()

    # Calculate average edit distance and variance
    data['Size_Morphology'] = data['Size'] + '-' + data['Morphology']
    data['Zipf Bin'] = pd.cut(data['Zipf Frequency'], bins=[1, 2, 3, 4, 5, 6, 7], right=False)

    grouped_df = data.groupby(
        ['Zipf Bin', 'Size_Morphology'], observed=True)['Edit Distance'].mean().reset_index()
    grouped_df['Zipf Bin'] = grouped_df['Zipf Bin'].astype(str)

    grouped_df['Edit Distance Std'] = data.groupby(
        ['Zipf Bin', 'Size_Morphology'], observed=True)['Edit Distance'].std().reset_index()['Edit Distance']

    sns.lineplot(
        data=grouped_df,
        x='Zipf Bin',
        y='Edit Distance',
        hue='Size_Morphology',
        marker='o',
        markersize=8,
        ax=freq_ax
    )
    # freq_ax.errorbar(
    #     x=grouped_df['Zipf Bin'],
    #     y=grouped_df['Edit Distance'],
    #     yerr=grouped_df['Edit Distance Std'],
    #     fmt='none',
    #     ecolor='black',
    #     capsize=3,
    #     alpha=0.5,
    #     zorder=0
    # )
    freq_ax.set_title('Edit Distance by Frequency')
    freq_ax.set_xlabel('Zipf Frequency')
    freq_ax.set_ylabel('Average Edit Distance')
    freq_ax.legend(title='Size & Morphology')
    freq_ax.grid(True)

    """ Figure 3: Errors by Test Category """
    data = df.copy()

    # Function to calculate averages and variance
    def calc_mean_and_std(group):
        return pd.Series({
            'Deletions': group['Deletions'].mean(),
            'Deletions Std': group['Deletions'].std(),
            'Insertions': group['Insertions'].mean(),
            'Insertions Std': group['Insertions'].std(),
            'Substitutions': group['Substitutions'].mean(),
            'Substitutions Std': group['Substitutions'].std(),
            # 'Edit Distance': group['Edit Distance'].mean(),
            # 'Edit Distance Std': group['Edit Distance'].std()
        })

    # Calculate averages and variance for each category
    data['Category'] = data.apply(lambda row: 
        f"pseudo {row['Size']} {row['Morphology']}" if row['Lexicality'] == 'pseudo' 
        else f"real {row['Size']} {row['Frequency']} {row['Morphology']}", axis=1)
    grouped = data.groupby('Category').apply(calc_mean_and_std).reset_index()

    # Sort the categories in a meaningful order
    order = []
    # Add all real categories first
    for size in ['short', 'long']:
        for freq in ['high', 'low']:
            for morph in ['simple', 'complex']:
                order.append(f'real {size} {freq} {morph}')

    # Add all pseudo categories
    for size in ['short', 'long']:
        for morph in ['simple', 'complex']:
            order.append(f'pseudo {size} {morph}')

    # Melt the DataFrame for easier plotting
    melted = pd.melt(grouped, id_vars=['Category'], 
                     value_vars=['Deletions', 'Insertions',
                                 'Substitutions']) #, 'Edit Distance'])
    sns.barplot(
        x='Category',
        y='value',
        hue='variable',
        data=melted,
        order=order,
        ax=errors_ax
    )
    # # Add error bars
    # for i, cat in enumerate(order):
    #     for j, err_type in enumerate(['Deletions', 'Insertions', 'Substitutions', 'Edit Distance']):
    #         y = grouped.loc[grouped['Category'] == cat, err_type].values[0]
    #         yerr = grouped.loc[grouped['Category'] == cat, f'{err_type} Std'].values[0]
    #         errors_ax.errorbar(
    #             x=i + j * 0.2 - 0.3,
    #             y=y,
    #             yerr=yerr,
    #             fmt='none',
    #             ecolor='black',
    #             capsize=3,
    #             alpha=0.5,
    #             # zorder=0
    #         )

    # Convert categories to a single letter for better readability
    order = ["".join([cat[:1].capitalize() for cat in cats.split()]) for cats in order]
    errors_ax.set_title('Errors by Category')
    errors_ax.set_xlabel('Category (Lexicality Size Frequency Morphology)')
    errors_ax.set_ylabel('Average Error Count')
    errors_ax.set_xticks(range(len(order)))
    errors_ax.set_xticklabels(order)
    errors_ax.legend(title='Error Type')
    errors_ax.grid(True)
    
    plt.subplots_adjust(hspace=0.3)  # Adjust the vertical spacing between subplots

    MODEL_FIGURES_DIR = FIGURES_DIR / model
    MODEL_FIGURES_DIR.mkdir(exist_ok=True)
    filename = f'errors{epoch}.png' if epoch else 'errors.png'
    plt.savefig(MODEL_FIGURES_DIR / filename, dpi=300, bbox_inches='tight')
    # plt.close()


# Plot the confusion matrix for the test data
def confusion_matrix(confusions: dict, model: str, epoch: str) -> None:
    df = pd.DataFrame.from_dict(confusions, orient='index')

    # Separate the ALPAbet vowels and consonants
    phonemes = [
        'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
        'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 
        'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
        'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 
        'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 
        'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 
        'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
    ]

    # Normalize the confusion matrix
    # df = np.log1p(df) # log scale
    # df = df.div(df.sum(axis=1), axis=0) # row normal
    # df = (df - df.min().min()) / (df.max().max() - df.min().min()) # min max
    df = (df - df.mean()) / df.std() # z score

    # Create a confusion matrix
    plt.figure(figsize=(8, 7))

    # Plot the heatmap with enhanced aesthetics
    heatmap = sns.heatmap(df, annot=False, cmap='Blues', square=True, 
                cbar_kws={'label': 'Counts'},
                xticklabels=True, yticklabels=True)
    
    # Adjust the colorbar
    cbar = heatmap.collections[0].colorbar
    cbar.ax.set_aspect(10)  # Larger values make the colorbar narrower

    # Display X-ticks on the top
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')

    # Set axis labels and title
    plt.title('Confusion Matrix', fontsize=14, pad=20)
    plt.xlabel('Prediction', fontsize=10)
    plt.ylabel('Ground Truth', fontsize=10)
    plt.xticks(fontsize=5, rotation=90)
    plt.yticks(fontsize=5, rotation=0)
    plt.tight_layout()
    
    MODEL_FIGURES_DIR = FIGURES_DIR / model
    MODEL_FIGURES_DIR.mkdir(exist_ok=True)
    filename = f'confusion{epoch}.png'
    plt.savefig(MODEL_FIGURES_DIR / filename, dpi=300, bbox_inches='tight')
    # plt.close()

def primacy_recency(df: pd.DataFrame, model: str, epoch: Optional[int] = None) -> None:
    df = df.copy()
    # # Combining plots
    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # sns.barplot(x='Category', y='Value', data=df, ax=axes[0])
    # sns.boxplot(x='Category', y='Value', data=df, ax=axes[1])

    counts = {}
    data = []

    for _, row in df.iterrows():
        length = row['Sequence Length']
        if length not in counts:
            counts[length] = {}
        
        for index in row['Error Indices']:
            if index not in counts[length]:
                counts[length][index] = 0
            counts[length][index] += 1
    
    for length, indices in counts.items():
        indices = {i: indices.get(i, 0) for i in range(1, length + 1)}
        
        for idx, count in indices.items():
            data.append({"Length": length, "Index": idx, "Count": count})

    plot_df = pd.DataFrame(data)

    # # Calculate the max counts for each length
    # max_counts = plot_df.groupby("Length")["Count"].transform("max")

    # # Normalize the counts between 0 and 1
    # plot_df["Count"] = plot_df["Count"] / max_counts

    # # Y offset for each sequence length
    # plot_df["Offset"] = plot_df.apply(
    #     lambda row: row["Count"] + (row["Length"] - min(plot_df["Length"])) * 1.2, axis=1)

    palette = [
        "#4E79A7",  # Soft blue
        "#F28E2B",  # Warm orange
        "#76B7B2",  # Muted teal
        "#E15759",  # Light red
        "#59A14F",  # Fresh green
        "#EDC948",  # Soft yellow
        "#FF9DA7",  # Light coral pink
        "#9C755F",  # Soft brown
        "#BAB0AC"   # Muted gray
    ]

    # Plotting
    plt.figure(figsize=(8, 8))
    sns.lineplot(
        data=plot_df, 
        x="Index", 
        y="Count", 
        hue="Length", 
        marker="o",
        linewidth=1.5,  # Thinner lines
        # alpha=0.6,    # Reduced opacity
        palette=palette
    )
    plt.title("Serial Position Curve")
    plt.xlabel("Error Index")
    plt.ylabel("Normalized Error Count")
    plt.legend(title="Sequence Length")
    plt.grid(True)
    plt.tight_layout()

    MODEL_FIGURES_DIR = FIGURES_DIR / model
    MODEL_FIGURES_DIR.mkdir(exist_ok=True)
    filename = f'position{epoch}.png'
    plt.savefig(MODEL_FIGURES_DIR / filename, dpi=300, bbox_inches='tight')
    # plt.close()
