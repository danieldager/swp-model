import os
import torch
import random
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import spacy
from g2p_en import G2p
from morphemes import Morphemes
from wordfreq import zipf_frequency, iter_wordlist, word_frequency

import nltk
# nltk.download('averaged_perceptron_tagger_eng')

import time
from functools import wraps

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        name = func.__name__
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        total = end - start
        buf = 25 - len(name)
        print(f'{name}: {" "*buf} {total:.2f} seconds')
        return result
    return timeit_wrapper

# NOTE: run in terminal: python -m spacy download en
g2p = G2p()
nlp = spacy.load('en_core_web_sm')
mrp = Morphemes("../../data/morphemes_data")

# Seed everything for reproducibility
def seed_everything(seed=42) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

# Process the hand-made evaluation datasets
@timeit
def process_dataset(directory: str, is_real=True) -> pd.DataFrame:
    data = []
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            name_parts = file.split('.')[0].split('_')
            df = pd.read_csv(os.path.join(directory, file))
            
            df['Lexicality'] = name_parts[1]
            df['Morph Complexity'] = name_parts[-1]

            if is_real: df['Frequency'] = name_parts[2]
            data.append(df)
    
    data = pd.concat(data, join="outer", ignore_index=True)
    return data

# Get morphological data for a word
def get_morphological_data(word: str) -> list:
    parse = mrp.parse(word)

    if parse["status"] == "NOT_FOUND":
        return [], [], [], [], 1, "0-1-0"
    
    tree = parse["tree"]

    prefixes = []
    roots = []
    root_freqs = []
    suffixes = []

    for node in tree:
        if node["type"] == "prefix":
            prefixes.append(node["text"])
        
        elif "children" in node:
            for child in node["children"]:
                if child["type"] == "root":
                    roots.append(child["text"])
                    root_freqs.append(zipf_frequency(child["text"], 'en'))
        else:
            suffixes.append(node["text"])

    count = parse["morpheme_count"]
    structure = f"{len(prefixes)}-{len(roots)}-{len(suffixes)}"

    return prefixes, roots, root_freqs, suffixes, count, structure

# Add frequency, part of speech, phonemes, and morphology to the dataset
@timeit
def clean_and_enrich_data(df: pd.DataFrame, is_real=True) -> pd.DataFrame:
    
    # Drop rows with no word value
    df = df.dropna(subset=['word'])
    
    # Rename columns
    df = df.rename(columns={
        'word': 'Word',
        'PoS': 'Part of Speech',
        'num letters': 'Length',
    })

    # Add Zipf Frequency and Part of Speech columns
    if is_real:
        df = df.drop(columns=['Number', 'percentile freq', 'morph structure'])
        df['Zipf Frequency'] = df['Word'].apply(lambda x: zipf_frequency(x, 'en'))
        df['Part of Speech'] = df['Word'].apply(lambda x: nlp(x)[0].pos_)
    
    # Add Phonemes column
    df["Phonemes"] = df["Word"].apply(g2p)

    # NOTE: slow, so commented out for debugging
    # Add Morphological data
    # columns = ["Prefixes", "Roots", "Frequencies", "Suffixes", "Morpheme Count", "Morphology"]
    # df[columns] = df['Word'].apply(lambda word: pd.Series(get_morphological_data(word)))
    
    return df

# Combine and reformat the real and pseudo word datasets
@timeit
def get_evaluation_data():
    
    # Process real words
    real_words = process_dataset('../../data/eval_dataset_real')
    real_words = clean_and_enrich_data(real_words)

    # Process pseudo words
    pseudo_words = process_dataset('../../data/eval_dataset_pseudo', is_real=False)
    pseudo_words = clean_and_enrich_data(pseudo_words, is_real=False)

    # Combine datasets
    data = pd.concat([real_words, pseudo_words], join="outer", ignore_index=True)

    # Rearrange columns
    columns = ["Word", "Length", "Frequency", "Zipf Frequency", 
            "Morph Complexity", "Lexicality", "Part of Speech", "Phonemes"]
    data = data.reindex(columns=columns)

    # Isolate words and their phonemes
    real_words = real_words[['Word', 'Phonemes']]
    pseudo_words = pseudo_words[['Word', 'Phonemes']]
    
    return data, real_words, pseudo_words

# Sample words according to their frequency in the language
@timeit
def sample_words(word_count: int, language='en', max_words=100000) -> list:
    word_list = []
    total_freq = 0
    cumulative_probs = []

    for i, word in enumerate(iter_wordlist(language)):
        if i >= max_words: break
        
        word_list.append(word)
        freq = word_frequency(word, language)
        
        total_freq += freq
        cumulative_probs.append(total_freq)

    # Cumulative list of each word's frequency, sums to 1
    cumulative_probs = np.array(cumulative_probs) / total_freq
    
    # Generate random values between 0 and 1
    random_values = np.random.random(word_count)
    
    # Find the index of the first value greater than random value
    indices = np.searchsorted(cumulative_probs, random_values)

    # Get the word corresponding to each index
    sampled_words = [word_list[i] for i in indices]

    # Get phonemes for each word
    phonemes = [g2p(word) for word in sampled_words]
    
    return sampled_words, phonemes

# Plot the operations and total distance for each word category
def levenshtein_bar_graph(df: pd.DataFrame, model_name: str):
    
    # Function to calculate average operations and total distance
    def calc_averages(group):
        return pd.Series({
            'Avg Insertions': group['Insertions'].mean(),
            'Avg Deletions': group['Deletions'].mean(),
            'Avg Substitutions': group['Substitutions'].mean(),
            'Avg Total Distance': group['Levenshtein'].mean()
        })

    # Create a category column
    df['Category'] = df.apply(lambda row: 
        'Pseudo' if row['Lexicality'] == 'pseudo' else
        f"Real {row['Morph Complexity']} {row['Frequency']}", axis=1)

    # Group by the new category and calculate averages
    grouped = df.groupby('Category').apply(calc_averages).reset_index()

    # Melt the DataFrame for easier plotting
    melted = pd.melt(grouped, id_vars=['Category'], 
                     value_vars=['Avg Insertions', 'Avg Deletions', 
                                 'Avg Substitutions', 'Avg Total Distance'])

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Category', y='value', hue='variable', data=melted)

    plt.title('Average Levenshtein Operations and Total Distance by Word Category')
    plt.xlabel('Word Category')
    plt.ylabel('Average Count')
    plt.xticks(rotation=70, ha='right')
    plt.legend(title='Operation Type')
    plt.tight_layout()
    plt.savefig(f'../../figures/{model_name}_bar_graph.png')
    plt.show()

    # Drop the category column
    df = df.drop(columns=['Category'])