import os
import time
import random
from pathlib import Path
from functools import wraps
from collections import defaultdict

import nltk
import numpy as np
import pandas as pd
import spacy
import torch
import torch.version
import torch.backends.cudnn

from g2p_en import G2p
import spacy, spacy.cli
from morphemes import Morphemes
from wordfreq import iter_wordlist, word_frequency, zipf_frequency

# nltk.download('averaged_perceptron_tagger_eng')

""" PATHS """
FILE_DIR = Path(__file__).resolve()
ROOT_DIR = FILE_DIR.parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
TEST_DATA_REAL = DATA_DIR / "test_dataset_real"
TEST_DATA_PSEUDO = DATA_DIR / "test_dataset_pseudo"

""" SEEDING """
def seed_everything(seed=42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        cuda_version = list(map(int, torch.version.cuda.split(".")))
        if cuda_version[0] == 10:
            if cuda_version[1] == 1:
                os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            if cuda_version[1] > 1:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
                    ":4096:8"  # setting CUBLAS_WORKSPACE_CONFIG=:16:8 also works
                )
        elif cuda_version[0] > 10:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
                ":4096:8"  # setting CUBLAS_WORKSPACE_CONFIG=:16:8 also works
            )
        else:
            raise RuntimeError(
                f"CUDA version {torch.version.cuda} might be too old to support deterministic behavior"
            )


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
        print(f"{'Operation':<30} {'Total (s)':<15}")
        print("-" * 60)
        for name in self.times:
            print(f"{name:<30} {self.times[name]:>13.3f}s")


""" TEST DATA PROCESSING """
g2p = G2p()
mrp = Morphemes(str(DATA_DIR / "morphemes_data"))

if not spacy.util.is_package("en_core_web_lg"):
    spacy.cli.download("en_core_web_lg")
nlp = spacy.load('en_core_web_lg')

# Process the hand-made test datasets
def process_dataset(directory: Path, real=False) -> pd.DataFrame:
    data = []
    for file in directory.glob("*.csv"):
        name_parts = file.stem.split("_")
        df = pd.read_csv(file)
        df["Lexicality"] = name_parts[1]
        df["Morphology"] = name_parts[-1]
        if real:
            df["Size"] = name_parts[3]
            df["Frequency"] = name_parts[2]
        else:
            df["Size"] = name_parts[2]
        data.append(df)

    data = pd.concat(data, join="outer")
    return data


# Get morphological data for a word
def get_morphological_data(word: str):
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
                    root_freqs.append(zipf_frequency(child["text"], "en"))
        else:
            suffixes.append(node["text"])

    count = parse["morpheme_count"]
    structure = f"{len(prefixes)}-{len(roots)}-{len(suffixes)}"

    return prefixes, roots, root_freqs, suffixes, count, structure


# Add frequency, part of speech, phonemes, and morphology to the dataset
def clean_and_enrich_data(df: pd.DataFrame, real=False) -> pd.DataFrame:

    # Drop rows with no word value
    df = df.dropna(subset=["word"])

    # Rename columns
    df = df.rename(
        columns={
            "word": "Word",
            "PoS": "Part of Speech",
            "num letters": "Length",
        }
    )

    # Add Zipf Frequency and Part of Speech columns
    if real:
        df = df.drop(columns=["Number", "percentile freq", "morph structure"])
        df["Zipf Frequency"] = df["Word"].apply(lambda x: zipf_frequency(x, "en"))
        df["Part of Speech"] = df["Word"].apply(lambda x: nlp(x)[0].pos_)

    # Add Phonemes column
    df["Phonemes"] = df["Word"].apply(g2p)

    # NOTE: Very slow
    # Add Morphological data
    # columns = ["Prefixes", "Roots", "Frequencies", "Suffixes", "Morpheme Count", "Structure"]
    # df[columns] = df['Word'].apply(lambda word: pd.Series(get_morphological_data(word)))

    return df


# Combine and reformat the real and pseudo word datasets
def get_test_data() -> tuple:
    # Process real words
    real_words = process_dataset(TEST_DATA_REAL, real=True)
    real_words = clean_and_enrich_data(real_words, real=True)

    # Process pseudo words
    pseudo_words = process_dataset(TEST_DATA_PSEUDO)
    pseudo_words = clean_and_enrich_data(pseudo_words)

    # Combine datasets
    dataframe = pd.concat(
        [real_words, pseudo_words], join="outer"
    )  # , ignore_index=True)

    # Rearrange columns
    columns = [
        "Word",
        "Size",
        "Length",
        "Frequency",
        "Zipf Frequency",
        "Morphology",
        "Lexicality",
        "Part of Speech",
        "Phonemes",
    ]
    dataframe = dataframe.reindex(columns=columns)

    # Isolate words and their phonemes
    real_words = real_words[["Word", "Phonemes"]]
    pseudo_words = pseudo_words[["Word", "Phonemes"]]

    return dataframe, real_words["Word"].tolist()


""" WORD SAMPLING """

def sample_words(test_data, word_count=50000, split=0.9, freq_th=0.95) -> list:    
    word_list = []
    freq_list = []
    total_freq = 0
    
    test_words = test_data["Word"].tolist()
    for i, word in enumerate(iter_wordlist("en")):
        # Limit the number of words
        if i >= 30000:
            break
        # Skip any non-alphabetic words
        if not word.isalpha():
            continue
        # Skip any words in the test set
        if word in test_words:
            continue
        # Skip any words that don't have vowels
        if not any(char in "aeiou" for char in word):
            continue

        freq = word_frequency(word, "en")
        word_list.append(word)
        freq_list.append(freq)
        total_freq += freq

    # Normalize frequencies
    freq_array = np.array(freq_list) / total_freq

    # Sort words by frequency (low to high)
    sorted_indices = np.argsort(freq_array)
    sorted_freqs = freq_array[sorted_indices]
    sorted_words = [word_list[i] for i in sorted_indices]

    # Sample training words
    train_count = int(word_count * split)
    train_words = np.random.choice(sorted_words, train_count, p=sorted_freqs)

    # Sample validation words from low frequency words
    valid_count = word_count - train_count

    # Determine the index that separates low frequency words
    lf_index = np.searchsorted(np.cumsum(sorted_freqs), freq_th)
    
    # Sample validation words from low frequency candidate words
    candidates = [
        w for i, w in enumerate(sorted_words) if i < lf_index and w not in train_words
    ]
    valid_words = random.sample(candidates, min(valid_count, len(candidates)))

    # Get phonemes for each word
    train_phonemes = [g2p(word) for word in train_words]
    valid_phonemes = [g2p(word) for word in valid_words]

    # start = time.perf_counter()
    # print(f"{time.perf_counter() - start:.2f} seconds")
    return train_phonemes, valid_phonemes

def phoneme_statistics(phonemes: list):
    # Get the counts for each phoneme
    phoneme_stats = defaultdict(int)
    for word in phonemes:
        for phoneme in word:
            phoneme_stats[phoneme] += 1

    # Sort descending by count
    phoneme_stats = dict(sorted(phoneme_stats.items(), key=lambda x: x[1], reverse=True))
    phoneme_stats["<STOP>"] = 0 # Add stop token

    # Get the bigram counts for each phoneme pair
    bigram_stats = defaultdict(int)
    for word in phonemes:
        for i in range(len(word) - 1):
            bigram = " ".join(word[i:i+2])
            bigram_stats[bigram] += 1

    # trigram_stats = defaultdict(int)
    # for sequence in phonemes:
    #     for i in range(len(sequence) - 2):
    #         trigram = " ".join(sequence[i:i+3])
    #         trigram_stats[trigram] += 1

    return phoneme_stats, bigram_stats

""" CUSTOM LOSS FUNCTION """
def alignment_loss(output, target, criterion, penalty):
    """
    Args:
        outputs: (output_len, vocab_size) tensor of logits
        targets: (target_len) tensor of target indices
    """
    output_len = output.size(0)
    target_len = target.size(0)
    
    # Initialize score matrix
    M = torch.zeros(output_len + 1, target_len + 1, device=output.device)
    
    # Initialize first row and column (penalty for skips)
    for i in range(output_len + 1): M[i, 0] = i * penalty
    for j in range(target_len + 1): M[0, j] = j * penalty
    
    # Fill matrix
    for i in range(1, output_len + 1):
        for j in range(1, target_len + 1):
            
            # Calculate match score using cross entropy
            score = criterion(
                output[i-1].unsqueeze(0), 
                target[j-1].unsqueeze(0)
            )
            
            # Take minimum of three possible operations:
            M[i, j] = torch.min(
                torch.stack([
                    M[i-1, j-1] + score,  # match/mismatch
                    M[i-1, j] + penalty,  # skip in output
                    M[i, j-1] + penalty   # skip in target
                ])
            )

    # print("M[x,y]", M[output_len, target_len])
    # print("M", M)

    return M[output_len, target_len]

# Decoder forward pass using alignment loss ^^^
def alignment_forward(self, x, hidden, stop_token, target_len):
    outputs = []
    
    # Set a limit for pred length
    max_length = target_len + 10
    
    # Forward pass loop 
    for _ in range(max_length):
        output, hidden = self.rnn(x, hidden)
        
        # Generate output logits
        logits = self.fc(output)
        outputs.append(logits)

        # Check for stop token
        if torch.argmax(output) == stop_token: print("STOP"); break
        
        # Pass output (not logits) to rnn 
        x = output
    
    # Return logits (pred_len, vocab_size)
    outputs = torch.stack(outputs, dim=0)
    return outputs


if __name__ == "__main__":
    sample_words([])
