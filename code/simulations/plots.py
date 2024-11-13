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

# Plot the training and validation loss curves
def training_curves(train_losses: list, valid_losses: list, model: str, num_epochs: int):
    # Extract parameters from the model name
    h, l, d, r = [p[1:] for p in model.split('_')[1:]]

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(1, num_epochs + 1), y=train_losses, label='Training')
    sns.lineplot(x=range(1, num_epochs + 1), y=valid_losses, label='Validation')
    plt.title(f'Model: Hidden={h}, Layers={l}, Dropout={d}, LR={r}')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.tight_layout()

    MODEL_FIGURES_DIR = FIGURES_DIR / model
    MODEL_FIGURES_DIR.mkdir(exist_ok=True)
    filename = MODEL_FIGURES_DIR / 'training_curves.png'
    plt.savefig(filename, dpi= 300, bbox_inches='tight')

# Plot the edit operations and distance for each test category
def errors_bar_chart(df: pd.DataFrame, model: str, epoch: Optional[int] = None) -> None:
    df = df.copy()
    sns.set_palette("colorblind")

    # Function to calculate average operations and total distance
    def calc_averages(group):
        return pd.Series({
            'Deletions': group['Deletions'].mean(),
            'Insertions': group['Insertions'].mean(),
            'Substitutions': group['Substitutions'].mean(),
            'Edit Distance': group['Edit Distance'].mean()
        })

    # Group by categories and calculate averages
    df['Category'] = df.apply(lambda row: 
        f"pseudo {row['Size']} {row['Morphology']}" if row['Lexicality'] == 'pseudo' 
        else f"real {row['Size']} {row['Frequency']} {row['Morphology']}", axis=1)
    grouped = df.groupby('Category').apply(calc_averages).reset_index()

    # Sort the categories in a meaningful order
    order = []
    for size in ['short', 'long']:
        # Add pseudo categories for each morphology type
        for morphology in df[df['Lexicality'] == 'pseudo']['Morphology'].unique():
            order.append(f'pseudo {size} {morphology}')
        # Then add real categories
        for morphology in df[df['Lexicality'] == 'real']['Morphology'].unique():
            for freq in df['Frequency'].unique():
                order.append(f'real {size} {freq} {morphology}')

    # Filter category order to only include categories that exist in the data
    order = [cat for cat in order if cat in grouped['Category'].unique()]

    # Melt the DataFrame for easier plotting
    melted = pd.melt(grouped, id_vars=['Category'], 
                     value_vars=['Deletions', 'Insertions',
                                 'Substitutions', 'Edit Distance'])

    # Create the plot
    plt.figure(figsize=(14, 7))
    sns.barplot(x='Category', y='value', hue='variable', data=melted, order=order)

    plt.title('Errors by Test Category')
    plt.xlabel('Test Category')
    plt.ylabel('Average Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Error Type')
    plt.tight_layout()

    MODEL_FIGURES_DIR = FIGURES_DIR / model
    MODEL_FIGURES_DIR.mkdir(exist_ok=True)
    filename = f'errors{epoch}.png' if epoch else 'errors.png'
    plt.savefig(MODEL_FIGURES_DIR / filename, dpi= 300, bbox_inches='tight')
    plt.close()

# Plot the edit distance by word length and frequency
def parametric_plots(df: pd.DataFrame, model: str, epoch: Optional[int] = None) -> None:
    df = df.copy()
    sns.set_palette("colorblind")

    # Create figure with two subplots
    fig, (length_ax, freq_ax) = plt.subplots(2, 1, figsize=(14, 7))
    
    # First subplot: Length vs Edit Distance
    df['Lex_Morphology'] = df['Lexicality'] + '-' + df['Morphology']
    grouped_length_df = df.groupby(
        ['Length', 'Lex_Morphology'], observed=True)['Edit Distance'].mean().reset_index()
    
    sns.lineplot(
        data=grouped_length_df,
        x='Length',
        y='Edit Distance',
        hue='Lex_Morphology',
        markers=True,
        marker='o',
        markersize=8,
        ax=length_ax  # Specify which axis to use
    )

    length_ax.set_title('Edit Distance by Length')
    length_ax.set_xlabel('Word Length')
    length_ax.set_ylabel('Average Edit Distance')
    length_ax.legend(title='Lexicality & Morphology')
    length_ax.grid(True)
    
    # Second subplot: Frequency vs Edit Distance
    real_words = df[df['Lexicality'] == 'real'].copy()    
    real_words['Size_Morphology'] = real_words['Size'] + '-' + real_words['Morphology']
    real_words['Zipf Bin'] = pd.cut(
        real_words['Zipf Frequency'], bins=[1, 2, 3, 4, 5, 6, 7], right=False)
    
    grouped_df = real_words.groupby(
        ['Zipf Bin', 'Size_Morphology'], observed=True)['Edit Distance'].mean().reset_index()
    grouped_df['Zipf Bin'] = grouped_df['Zipf Bin'].astype(str)

    sns.scatterplot(
        data=grouped_df,
        x='Zipf Bin',
        y='Edit Distance',
        hue='Size_Morphology',
        s=100,
        ax=freq_ax
    )

    freq_ax.set_title('Edit Distance by Frequency')
    freq_ax.set_xlabel('Zipf Frequency')
    freq_ax.set_ylabel('Average Edit Distance')
    freq_ax.legend(title='Size & Morphology')
    freq_ax.grid(True)
    plt.tight_layout()

    MODEL_FIGURES_DIR = FIGURES_DIR / model
    MODEL_FIGURES_DIR.mkdir(exist_ok=True)
    filename = f'parametric{epoch}.png' if epoch else 'parametric.png'
    plt.savefig(MODEL_FIGURES_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()

def confusion_matrix():
    pass
