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

def confusion_matrix():
    pass
