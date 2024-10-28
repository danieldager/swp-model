import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import spacy
from g2p_en import G2p
from morphemes import Morphemes
from wordfreq import zipf_frequency, iter_wordlist, word_frequency

import nltk
# nltk.download('averaged_perceptron_tagger_eng')

import time
from functools import wraps

""" PATHS """
FILE_DIR = Path(__file__).resolve()
ROOT_DIR = FILE_DIR.parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
TEST_DATA_REAL = DATA_DIR / "test_dataset_real"
TEST_DATA_PSEUDO = DATA_DIR / "test_dataset_pseudo"

""" SEEDING """
# Seed everything for reproducibility
def seed_everything(seed=42) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

""" DEVICE """
def set_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"Using CUDA device: {device_name}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device

""" PERFORMANCE """
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

class Timer:
    def __init__(self):
        self.times = defaultdict(float)
        self.counts = defaultdict(int)
        
    def start(self):
        self._start_time = time.time()
        
    def stop(self, name):
        elapsed = time.time() - self._start_time
        self.times[name] += elapsed
        self.counts[name] += 1
        
    def summary(self):
        print("\nTiming Summary:")
        print("-" * 60)
        print(f"{'Operation':<30} {'Total Time (s)':<15} {'Avg Time (s)':<15}")
        print("-" * 60)
        for name in self.times:
            total = self.times[name]
            avg = total / self.counts[name]
            print(f"{name:<30} {total:>13.3f}s {avg:>13.3f}s")


""" TEST SET PROCESSING """
# NOTE: python -m spacy download en_core_web_lg
g2p = G2p()
nlp = spacy.load('en_core_web_lg')
mrp = Morphemes(str(DATA_DIR / "morphemes_data"))

# Process the hand-made test datasets
def process_dataset(directory: Path, real=False) -> pd.DataFrame:
    data = []
    for file in directory.glob('*.csv'):
        name_parts = file.stem.split('_')
        df = pd.read_csv(file)
        df['Lexicality'] = name_parts[1]
        df['Morph Complexity'] = name_parts[-1]
        if real: df['Frequency'] = name_parts[2]
        data.append(df)

    data = pd.concat(data, join="outer")
    return data

# Get morphological data for a word
def get_morphological_data(word: str) -> list:
    parse = mrp.parse(word)

    if parse["status"] == "NOT_FOUND":
        return None, None, None, None, None, None
    
    tree = parse["tree"]
    prefixes, roots, root_freqs, suffixes = [], [], [], []    

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
def clean_and_enrich_data(df: pd.DataFrame, real=False) -> pd.DataFrame:
    
    # Drop rows with no word value
    df = df.dropna(subset=['word'])
    
    # Rename columns
    df = df.rename(columns={
        'word': 'Word',
        'PoS': 'Part of Speech',
        'num letters': 'Length',
    })

    # Add Zipf Frequency and Part of Speech columns
    if real:
        df = df.drop(columns=['Number', 'percentile freq', 'morph structure'])
        df['Zipf Frequency'] = df['Word'].apply(lambda x: zipf_frequency(x, 'en'))
        df['Part of Speech'] = df['Word'].apply(lambda x: nlp(x)[0].pos_)
    
    # Add Phonemes column
    df["Phonemes"] = df["Word"].apply(g2p)

    # NOTE: Very slow
    # Add Morphological data
    # columns = ["Prefixes", "Roots", "Frequencies", "Suffixes", "Morpheme Count", "Morphology"]
    # df[columns] = df['Word'].apply(lambda word: pd.Series(get_morphological_data(word)))
    
    return df

# Combine and reformat the real and pseudo word datasets
def get_test_data():
    # Process real words
    real_words = process_dataset(TEST_DATA_REAL, real=True)
    real_words = clean_and_enrich_data(real_words, real=True)

    # Process pseudo words
    pseudo_words = process_dataset(TEST_DATA_PSEUDO)
    pseudo_words = clean_and_enrich_data(pseudo_words)

    # Combine datasets
    dataframe = pd.concat([real_words, pseudo_words], join="outer") #, ignore_index=True)

    # Rearrange columns
    columns = [
        "Word", "Length", "Frequency", "Zipf Frequency", 
        "Morph Complexity", "Lexicality", "Part of Speech", "Phonemes"
    ]
    dataframe = dataframe.reindex(columns=columns)

    # Isolate words and their phonemes
    real_words = real_words[['Word', 'Phonemes']]
    pseudo_words = pseudo_words[['Word', 'Phonemes']]
    
    return dataframe, real_words, pseudo_words

""" WORD SAMPLING """
# Sample words for training and validation datasets
# @timeit
def sample_words(word_count: int, language='en', split=0.9, freq_threshold=0.5) -> list:    
    word_list = []
    freq_list = []
    total_freq = 0

    for i, word in enumerate(iter_wordlist(language)):
        # Limit the number of words
        if i >= word_count: break
        freq = word_frequency(word, language)
        word_list.append(word)
        freq_list.append(freq)
        total_freq += freq

    # Normalize frequencies
    freq_array = np.array(freq_list) / total_freq

    # Sort words by frequency (low to high)
    sorted_indices = np.argsort(freq_array)
    sorted_words = [word_list[i] for i in sorted_indices]
    sorted_freqs = freq_array[sorted_indices]

    # print(f"sorted indices: {sorted_indices[:10]}")
    # print(f"sorted words: {sorted_words[:10]}")
    # print(f"sorted freqs: {sorted_freqs[:10]}")

    # Determine the index that separates low and high frequency words
    lf_index = np.searchsorted(np.cumsum(sorted_freqs), freq_threshold)

    # print(f"low frequency index: {lf_index}")

    # Sample training words
    train_count = int(word_count * split)
    probs = freq_array / freq_array.sum()
    indices = np.random.choice(len(sorted_words), train_count, replace=False, p=probs)
    train_words = [sorted_words[i] for i in indices]

    start = time.perf_counter()

    # NOTE: Why is this so much slower than the previous step?
    # Sample validation words from low frequency words
    valid_count = word_count - train_count
    candidates = [w for i, w in enumerate(sorted_words) if i < lf_index and w not in train_words]
    valid_words = random.sample(candidates, min(valid_count, len(candidates)))

    # print(f"Time to sample validation words: {time.perf_counter() - start:.2f} seconds")
    # print(f"valid_words samples: {valid_words[:10]}")

    # NOTE: Slow and probably unnecessary
    # If we don't have enough low frequency words, sample from remaining words
    # if len(valid_words) < valid_count:
    #     remaining_words = [w for w in sorted_words if w not in train_words and w not in valid_words]
    #     valid_words.extend(random.sample(remaining_words, valid_count - len(valid_words)))

    # print(f'train_words: {len(train_words)}, valid_words: {len(valid_words)}')

    # Get phonemes for each word
    train_phonemes = [g2p(word) for word in train_words]
    valid_phonemes = [g2p(word) for word in valid_words]

    # print(f"Time to get phonemes: {time.perf_counter() - start:.2f} seconds")
    
    return train_phonemes, valid_phonemes
