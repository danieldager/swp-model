import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

""" PATHS """
FILE_DIR = Path(__file__).resolve()
ROOT_DIR = FILE_DIR.parent.parent.parent
FIGURES_DIR = ROOT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

def training_curves(train_losses: list, valid_losses: list, model: str, num_epochs: int):
    # Create a directory for the model's figures
    MODEL_FIGURES_DIR = FIGURES_DIR / model
    MODEL_FIGURES_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(1, num_epochs + 1), y=train_losses, label='Training')
    sns.lineplot(x=range(1, num_epochs + 1), y=valid_losses, label='Validation')
    plt.title(f'Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.tight_layout()
    save_path = MODEL_FIGURES_DIR / 'training_curves.png'
    plt.savefig(save_path, dpi= 300, bbox_inches='tight')


# Plot the operations and total distance for each word category
def errors_bar_chart(df: pd.DataFrame, model_name: str, epoch: int):
    # Create a copy of the DataFrame
    df = df.copy()

    # Create a directory for the model's figures
    if epoch:
        MODEL_FIGS = FIGURES_DIR / model_name
        MODEL_FIGS.mkdir(exist_ok=True)
    
    # Function to calculate average operations and total distance
    def calc_averages(group):
        return pd.Series({
            'Avg Deletions': group['Deletions'].mean(),
            'Avg Insertions': group['Insertions'].mean(),
            'Avg Substitutions': group['Substitutions'].mean(),
            'Avg Edit Distance': group['Edit Distance'].mean()
        })

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

    if epoch: file = MODEL_FIGS / f'errors{epoch}.png'
    else: file = MODEL_FIGS / 'errors.png'
    plt.savefig(file, dpi= 300, bbox_inches='tight')


def parametric_plots(df: pd.DataFrame, model_name: str, epoch: int):
    if epoch:
        MODEL_FIGS = FIGURES_DIR / model_name
        MODEL_FIGS.mkdir(exist_ok=True)

    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Create figure with two subplots stacked vertically
    # Increased height (15) and reduced width (8) for vertical stacking
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7))
    
    # Check if Split column exists, if not assume it's all test data
    if 'Split' in df.columns:
        df['Dataset'] = df['Split'].map({'train': 'Training', 'test': 'Test'})
    else:
        df['Dataset'] = 'Test'  # Assume it's all test data
    
    """ LENGTH VS EDIT DISTANCE """
    # Calculate mean edit distance grouped by length, lexicality, complexity, and dataset
    length_grouped = df.groupby(
        ['Length', 'Lexicality', 'Morphology', 'Dataset']
        )['Edit Distance'].mean().reset_index()
    
    # Use seaborn color palette
    colors = {'real simple': sns.color_palette()[0], 
              'real complex': sns.color_palette()[1], 
              'pseudo simple': sns.color_palette()[2],
              'pseudo complex': sns.color_palette()[3],
    }
    
    # Plot for all categories
    for dataset in length_grouped['Dataset'].unique():
        for lexicality in length_grouped['Lexicality'].unique():
            for morphology in length_grouped['Morphology'].unique():
                
                mask = ((length_grouped['Dataset'] == dataset) & 
                       (length_grouped['Lexicality'] == lexicality) & 
                       (length_grouped['Morphology'] == morphology))
                linestyle = '-' if dataset == 'Training' else '--'
                color = colors[morphology] if lexicality == 'real' else colors['pseudo']
                marker = 'o' if lexicality == 'real' else 's'
                
                ax1.plot(length_grouped[mask]['Length'], 
                         length_grouped[mask]['Edit Distance'],
                         color=color,
                         marker=marker,
                         linestyle=linestyle,
                         label=f"{lexicality} {morphology} ({dataset})")
    
    ax1.set_xlabel('Word Length')
    ax1.set_ylabel('Average Edit Distance')
    ax1.set_title('Edit Distance vs. Word Length')
    ax1.legend(loc='best')
    
    """ FREQUENCY VS EDIT DISTANCE """    
    # Filter for real words and group by frequency
    real_df = df[df['Lexicality'] == 'real']
    freq_grouped = real_df.groupby(['Zipf Frequency', 'Morphology', 'Dataset'])['Edit Distance'].mean().reset_index()
    
    for dataset in freq_grouped['Dataset'].unique():
        for complexity in freq_grouped['Morphology'].unique():
            
            mask = ((freq_grouped['Dataset'] == dataset) & 
                    (freq_grouped['Morphology'] == complexity))
            linestyle = '-' if dataset == 'Training' else '--'
            
            ax2.plot(freq_grouped[mask]['Zipf Frequency'],
                     freq_grouped[mask]['Edit Distance'],
                     marker='o',
                     linestyle=linestyle,
                     color=colors[complexity],
                     label=f'{complexity} ({dataset})')
    
    ax2.set_xlabel('Zipf Frequency')
    ax2.set_ylabel('Average Edit Distance')
    ax2.set_title('Edit Distance vs. Zipf Frequency')
    ax2.legend(loc='best')
    plt.tight_layout()

    if epoch: file = MODEL_FIGS / f'parametric{epoch}.png'
    else: file = FIGURES_DIR / f'{model_name}_parametric_plots.png'
    plt.savefig(file, dpi= 300, bbox_inches='tight')

def confusion_matrix():
    pass
