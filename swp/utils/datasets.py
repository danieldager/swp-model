import random
from ast import literal_eval
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
import spacy.cli
from g2p_en import G2p
from morphemes import Morphemes
from wordfreq import iter_wordlist, word_frequency, zipf_frequency

from .paths import get_dataset_dir


def process_dataset(directory: Path, real=False) -> pd.DataFrame:
    r"""Process the hand-made test datasets"""
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


def get_morphological_data(word: str):
    r"""Get morphological data for a word"""
    mrp = Morphemes(
        str(get_dataset_dir() / "morphemes_data")
    )  # TODO check that the path is ok
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


def clean_and_enrich_data(df: pd.DataFrame, real=False) -> pd.DataFrame:
    r"""Add frequency, part of speech, phonemes, and morphology to the dataset"""
    g2p = G2p()
    if not spacy.util.is_package("en_core_web_lg"):
        spacy.cli.download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

    # Rename columns
    df = df.rename(
        columns={
            "word": "Word",
            "PoS": "Part of Speech",
            "num letters": "Length",
        }
    )

    # Drop rows with no word value
    df = df.dropna(subset=["Word"])

    # Add Zipf Frequency and Part of Speech columns
    if real:
        df = df.drop(
            columns=["Number", "percentile freq", "morph structure"], errors="ignore"
        )
        df["Zipf Frequency"] = df["Word"].apply(lambda x: zipf_frequency(x, "en"))
        df["Part of Speech"] = df["Word"].apply(lambda x: nlp(x)[0].pos_)

    # Add Phonemes column
    df["Phonemes"] = df["Word"].apply(g2p)

    # NOTE: Very slow
    # Add Morphological data
    # columns = ["Prefixes", "Roots", "Frequencies", "Suffixes", "Morpheme Count", "Structure"]
    # df[columns] = df['Word'].apply(lambda word: pd.Series(get_morphological_data(word)))

    return df


def create_test_data() -> pd.DataFrame:
    r"""Combine and reformat the real and pseudo word datasets"""
    # Process real words
    handmade_real_path = get_dataset_dir() / "handmade" / "test_dataset_real"
    csv_real_path = get_dataset_dir() / "dataframe" / "real_test.csv"
    csv_real_path.parent.mkdir(parents=True, exist_ok=True)
    real_words = process_dataset(handmade_real_path, real=True)
    real_words = clean_and_enrich_data(real_words, real=True)
    real_words.to_csv(csv_real_path)

    # Process pseudo words
    handmade_pseudo_path = get_dataset_dir() / "handmade" / "test_dataset_pseudo"
    csv_pseudo_path = get_dataset_dir() / "dataframe" / "pseudo_test.csv"
    pseudo_words = process_dataset(handmade_pseudo_path)
    pseudo_words = clean_and_enrich_data(pseudo_words)
    pseudo_words.to_csv(csv_pseudo_path)

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

    csv_complete_path = get_dataset_dir() / "dataframe" / "complete_test.csv"
    dataframe.to_csv(csv_complete_path)

    return dataframe


def get_test_data(force_recreate: bool = False) -> pd.DataFrame:
    csv_test_path = get_dataset_dir() / "dataframe" / "complete_test.csv"
    if csv_test_path.exists() and not force_recreate:
        dataframe = pd.read_csv(csv_test_path)
        dataframe["Phonemes"] = dataframe["Phonemes"].apply(literal_eval)
    else:
        dataframe = create_test_data()
    return dataframe


def create_train_data(test_data: pd.DataFrame) -> pd.DataFrame:
    word_list = []
    freq_list = []
    test_words = set(test_data["Word"])
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
        if not any(char in "aeiouy" for char in word):
            continue

        word_list.append(word)
        freq_list.append(word_frequency(word, "en"))

    dataframe = pd.DataFrame({"Word": word_list, "Frequency": freq_list})
    dataframe = clean_and_enrich_data(dataframe, real=True)

    csv_train_path = get_dataset_dir() / "dataframe" / "complete_train.csv"
    dataframe.to_csv(csv_train_path)

    return dataframe


def get_train_data(force_recreate: bool = False) -> pd.DataFrame:
    csv_train_path = get_dataset_dir() / "dataframe" / "complete_train.csv"
    if csv_train_path.exists() and not force_recreate:
        dataframe = pd.read_csv(csv_train_path)
        dataframe["Phonemes"] = dataframe["Phonemes"].apply(literal_eval)
    else:
        test_df = get_test_data(force_recreate)
        dataframe = create_train_data(test_df)
    return dataframe


def create_split(
    train_data: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:

    generator = np.random.default_rng(seed=seed)

    train_split = train_data.sample(frac=0.9, random_state=generator)
    valid_split = train_data.drop(train_split.index)

    csv_train_split_path = get_dataset_dir() / "dataframe" / "train_split.csv"
    train_split.to_csv(csv_train_split_path)
    csv_valid_split_path = get_dataset_dir() / "dataframe" / "train_split.csv"
    valid_split.to_csv(csv_valid_split_path)

    return train_split, valid_split


def get_training_split(force_recreate: bool = False) -> pd.DataFrame:
    csv_train_split_path = get_dataset_dir() / "dataframe" / "train_split.csv"
    if csv_train_split_path.exists() and not force_recreate:
        dataframe = pd.read_csv(csv_train_split_path)
        dataframe["Phonemes"] = dataframe["Phonemes"].apply(literal_eval)
    else:
        train_df = get_train_data(force_recreate)
        dataframe, _ = create_split(train_df)
    return dataframe


def get_val_split(force_recreate: bool = False) -> pd.DataFrame:
    csv_valid_split_path = get_dataset_dir() / "dataframe" / "train_split.csv"
    if csv_valid_split_path.exists() and not force_recreate:
        dataframe = pd.read_csv(csv_valid_split_path)
        dataframe["Phonemes"] = dataframe["Phonemes"].apply(literal_eval)
    else:
        train_df = get_train_data(force_recreate)
        _, dataframe = create_split(train_df)
    return dataframe


def sample_words(
    word_count=50000, split=0.9, freq_th=0.95
) -> tuple[list[list[str]], list[list[str]]]:
    g2p = G2p()

    train_df = get_train_data()

    freq_list = train_df["Frequency"].tolist()
    word_list = train_df["Word"].tolist()
    total_freq = train_df["Frequency"].sum()

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
    phoneme_stats = dict(
        sorted(phoneme_stats.items(), key=lambda x: x[1], reverse=True)
    )
    phoneme_stats["<STOP>"] = 0  # Add stop token

    # Get the bigram counts for each phoneme pair
    bigram_stats = defaultdict(int)
    for word in phonemes:
        for i in range(len(word) - 1):
            bigram = " ".join(word[i : i + 2])
            bigram_stats[bigram] += 1

    # trigram_stats = defaultdict(0)
    # for sequence in phonemes:
    #     for i in range(len(sequence) - 2):
    #         trigram = " ".join(sequence[i:i+3])
    #         trigram_stats[trigram] += 1

    return phoneme_stats, bigram_stats


def get_word_to_freq(word_data: pd.DataFrame) -> dict[str, float]:
    words = word_data["Word"]
    freqs = word_data["Zipf Frequency"]
    word_to_freq = {}
    for word, freq in zip(words, freqs):
        word_to_freq[word] = freq
    return word_to_freq
