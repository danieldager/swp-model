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
    # Create a directory for the model's figures
    MODEL_FIGURES_DIR = FIGURES_DIR / model
    MODEL_FIGURES_DIR.mkdir(exist_ok=True)

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
    save_path = MODEL_FIGURES_DIR / 'training_curves.png'
    plt.savefig(save_path, dpi= 300, bbox_inches='tight')

# Plot the edit operations and distance for each test category
def errors_bar_chart(df: pd.DataFrame, model_name: str, epoch: Optional[int] = None) -> None:
    # Create a directory for the model's figures
    MODEL_FIGURES_DIR = FIGURES_DIR / model_name
    MODEL_FIGURES_DIR.mkdir(exist_ok=True)
    
    # Function to calculate average operations and total distance
    def calc_averages(group):
        return pd.Series({
            'Avg Deletions': group['Deletions'].mean(),
            'Avg Insertions': group['Insertions'].mean(),
            'Avg Substitutions': group['Substitutions'].mean(),
            'Avg Edit Distance': group['Edit Distance'].mean()
        })
    
    # Copy the df to avoid modifying the original
    df = df.copy()

    # Create a category column
    df['Category'] = df.apply(lambda row: 
        f"pseudo {row['Size']} {row['Morphology']}" if row['Lexicality'] == 'pseudo' 
        else f"real {row['Size']} {row['Frequency']} {row['Morphology']}", axis=1)

    # Group by the new category and calculate averages
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
                     value_vars=['Avg Deletions', 'Avg Insertions',
                                 'Avg Substitutions', 'Avg Edit Distance'])

    # Create the plot
    plt.figure(figsize=(14, 7))
    sns.barplot(x='Category', y='value', hue='variable', data=melted, order=order)

    plt.title('Error Counts and Edit Distance by Test Category')
    plt.xlabel('Test Category')
    plt.ylabel('Average Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Error Type')
    plt.tight_layout()

    if epoch: file = MODEL_FIGURES_DIR / f'errors{epoch}.png'
    else: file = MODEL_FIGURES_DIR / 'errors.png'
    plt.savefig(file, dpi= 300, bbox_inches='tight')
    plt.close()

# Plot the edit operations and distance for each test category
def parametric_plots(df: pd.DataFrame, model: str, epoch: Optional[int] = None, num_bins: int = 10) -> None:
    """
    Create parametric plots showing edit distance versus word length and
    average edit distance versus word frequency.
    """
    # Create a directory for the model's figures
    MODEL_FIGURES_DIR = FIGURES_DIR / model
    MODEL_FIGURES_DIR.mkdir(exist_ok=True)
    
    # Prepare data
    df = df.copy()
    if 'Split' in df.columns:
        df['Dataset'] = df['Split'].map(
            {'train': 'Training', 'test': 'Test'},
            na_value='Test'
        )
    else:
        df['Dataset'] = 'Test'
    
    # Create figure
    fig, (length_ax, freq_ax) = plt.subplots(2, 1, figsize=(14, 7))
    
    # Plot 1: Length vs Edit Distance
    sns.lineplot(
        data=df,
        x='Length',
        y='Edit Distance',
        hue='Lexicality',
        style='Morphology',
        dashes=[(None, None), (2, 2)],  # solid for simple, dashed for complex
        markers=['o', 's'],  # circle for real, square for pseudo
        estimator='mean',
        errorbar=None,
        ax=length_ax
    )
    
    length_ax.set(
        title='Edit Distance vs. Word Length',
        xlabel='Word Length',
        ylabel='Average Edit Distance'
    )
    
    # Plot 2: Average Edit Distance vs Zipf Frequency (binned)
    real_words = df[df['Lexicality'] == 'real'].copy()
    real_words.loc[:, 'Frequency Bin'] = pd.cut(real_words['Zipf Frequency'], bins=num_bins)
    
    # Group by bins and calculate mean edit distance
    binned_avg = real_words.groupby(
        ['Frequency Bin', 'Morphology', 'Size'], observed=True)['Edit Distance'].mean().reset_index()

    # Format interval endpoints to rounded strings for better display
    def format_interval(interval): return f"[{interval.left:.2f}, {interval.right:.2f})"
    binned_avg['Frequency Bin'] = binned_avg['Frequency Bin'].apply(lambda x: format_interval(x))

    # Plot the average edit distance per bin
    sns.lineplot(
        data=binned_avg,
        x='Frequency Bin',
        y='Edit Distance',
        hue='Morphology',
        style='Size',
        marker='o',
        ax=freq_ax
    )

    # Remove legend titles for a cleaner look
    if freq_ax.get_legend() is not None:
        freq_ax.get_legend().set_title(None)
    
    freq_ax.set(
        title='Average Edit Distance vs. Zipf Frequency (Binned)',
        xlabel='Zipf Frequency Bin',
        ylabel='Average Edit Distance'
    )
    freq_ax.set_xticks(range(len(binned_avg)))
    freq_ax.set_xticklabels(binned_avg['Frequency Bin'].astype(str), rotation=45, ha='right')
    
    # Adjust layout and save
    plt.tight_layout()
    filename = f'parametric{epoch}.png' if epoch else 'parametric.png'
    plt.savefig(MODEL_FIGURES_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()

def confusion_matrix():
    pass
