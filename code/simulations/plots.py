import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

""" PATHS """
FILE_DIR = Path(__file__).resolve()
ROOT_DIR = FILE_DIR.parent.parent.parent
FIGURES_DIR = ROOT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

def training_curves(train_losses:list, valid_losses:list, model:str, num_epochs:int):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(1, num_epochs + 1), y=train_losses, label='Training')
    sns.lineplot(x=range(1, num_epochs + 1), y=valid_losses, label='Validation')
    plt.title(f'Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{model}_loss.png", dpi= 300, bbox_inches='tight')


# Plot the operations and total distance for each word category
def levenshtein_bar_graph(df: pd.DataFrame, model_name: str):
    
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
        f"pseudo {row['Size']}" if row['Lexicality'] == 'pseudo' else
        f"real {row['Size']} {row['Frequency']} {row['Morph Complexity']}", axis=1)

    # Group by the new category and calculate averages
    grouped = df.groupby('Category').apply(calc_averages).reset_index()

    # Sort the categories in a meaningful order
    order = []
    for size in ['short', 'long']:
        # Add pseudo categories first
        order.append(f'pseudo {size}')
        # Then add real categories
        for complexity in df['Morph Complexity'].unique():
            for freq in df['Frequency'].unique():
                order.append(f'real {size} {freq} {complexity}')
    
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
    plt.savefig(FIGURES_DIR / f'{model_name}_errors.png', dpi= 300, bbox_inches='tight')


def parametric_plots(df: pd.DataFrame, model_name: str):
    plt.style.use('seaborn')  # For better-looking plots
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Create dataset split column
    df['Dataset'] = df['Split'].map({'train': 'Training', 'test': 'Test'})
    
    # Plot 1: Length vs Edit Distance
    # --------------------------------
    
    # Calculate mean edit distance grouped by length, lexicality, complexity, and dataset
    length_grouped = df.groupby(['Length', 'Lexicality', 'Morph Complexity', 'Dataset'])['Edit Distance'].mean().reset_index()
    
    # Plot for real words
    real_data = length_grouped[length_grouped['Lexicality'] == 'real']
    for dataset in ['Training', 'Test']:
        for complexity in real_data['Morph Complexity'].unique():
            mask = (real_data['Dataset'] == dataset) & (real_data['Morph Complexity'] == complexity)
            linestyle = '-' if dataset == 'Training' else '--'
            ax1.plot(real_data[mask]['Length'], 
                    real_data[mask]['Edit Distance'],
                    linestyle=linestyle,
                    marker='o',
                    label=f'Real {complexity} ({dataset})')
    
    # Plot for pseudo words
    pseudo_data = length_grouped[length_grouped['Lexicality'] == 'pseudo']
    for dataset in ['Training', 'Test']:
        mask = pseudo_data['Dataset'] == dataset
        linestyle = '-' if dataset == 'Training' else '--'
        ax1.plot(pseudo_data[mask]['Length'],
                pseudo_data[mask]['Edit Distance'],
                linestyle=linestyle,
                marker='s',  # Different marker for pseudo
                label=f'Pseudo ({dataset})')
    
    ax1.set_xlabel('Word Length')
    ax1.set_ylabel('Average Edit Distance')
    ax1.set_title('Edit Distance vs. Word Length')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Zipf Frequency vs Edit Distance (real words only)
    # ------------------------------------------------------
    
    # Filter for real words and group by frequency
    real_df = df[df['Lexicality'] == 'real']
    freq_grouped = real_df.groupby(['Zipf Frequency', 'Morph Complexity', 'Dataset'])['Edit Distance'].mean().reset_index()
    
    for dataset in ['Training', 'Test']:
        for complexity in freq_grouped['Morph Complexity'].unique():
            mask = (freq_grouped['Dataset'] == dataset) & (freq_grouped['Morph Complexity'] == complexity)
            linestyle = '-' if dataset == 'Training' else '--'
            ax2.plot(freq_grouped[mask]['Zipf Frequency'],
                    freq_grouped[mask]['Edit Distance'],
                    linestyle=linestyle,
                    marker='o',
                    label=f'{complexity} ({dataset})')
    
    ax2.set_xlabel('Zipf Frequency')
    ax2.set_ylabel('Average Edit Distance')
    ax2.set_title('Edit Distance vs. Zipf Frequency\n(Real Words Only)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'{model_name}_parametric_plots.png', 
                bbox_inches='tight', 
                dpi=300)
    # plt.show()
    plt.close()


def confusion_matrix():
    pass
