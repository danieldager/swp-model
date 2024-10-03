import os
import torch
import random
import numpy as np
import pandas as pd
from wordfreq import iter_wordlist, word_frequency


def seed_everything(seed=42):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


def get_all_eval_data():
    data = []
    
    folder = "../../data/eval_real_dataset"
    files = os.listdir(folder)
    for file in files:
        if file.endswith('.csv'):
            name = file.split('.')[0]
            _, lexicality, frequency, _, morphology = name.split('_')
            df = pd.read_csv(folder + "/" + file)

            # Add lexicality, morph complexity, and frequency columns
            df['Frequency'] = frequency
            df['Lexicality'] = lexicality
            df['Morph Complexity'] = morphology

            data.append(df)

    folder = "../../data/eval_pseudo_dataset"
    files = os.listdir(folder)
    for file in files:
        if file.endswith('.csv'):
            name = file.split('.')[0]
            _, lexicality, _, morphology = name.split('_')
            df = pd.read_csv(folder + "/" + file)

            df['Lexicality'] = lexicality
            df['Morph Complexity'] = morphology

            data.append(df)

    # Combine all dataframes into one
    df = pd.concat(data, join="outer", ignore_index=True)
    
    return df


# NOTE: review how this works
def sample_words(word_count, language='en', max_words=100000):
    words = []
    total_freq = 0
    cumulative_probs = []

    for i, word in enumerate(iter_wordlist(language)):
        if i >= max_words: break
        words.append(word)
        freq = word_frequency(word, language)
        total_freq += freq
        cumulative_probs.append(total_freq)
        
    cumulative_probs = np.array(cumulative_probs) / total_freq
    random_values = np.random.random(word_count)
    indices = np.searchsorted(cumulative_probs, random_values)
    
    return [words[i] for i in indices]


# class WordSampler:
#     def __init__(self, word_count, language='en', max_words=100000):
#         self.words = []
#         self.word_count = word_count
#         self.language = language
#         self.cumulative_probs = []
        
#         total_freq = 0
#         for i, word in enumerate(iter_wordlist(language)):
#             if i >= max_words:
#                 break
#             freq = word_frequency(word, language)
#             total_freq += freq
#             self.words.append(word)
#             self.cumulative_probs.append(total_freq)
        
#         self.cumulative_probs = np.array(self.cumulative_probs) / total_freq

#     def sample(self):
#         random_values = np.random.random(self.word_count)
#         indices = np.searchsorted(self.cumulative_probs, random_values)        
#         return [self.words[i] for i in indices]


if __name__ == "__main__":
    # seed_everything()
    # words = sample_words(50)
    get_all_eval_data()
    
