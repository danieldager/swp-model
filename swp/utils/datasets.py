import json
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

from .paths import get_dataframe_dir, get_dataset_dir, get_folds_dir


def process_dataset(directory: Path, real=False) -> pd.DataFrame:
    r"""Process the hand-made test datasets located in `directory`.
    Set `real` to ̀`True` to process real words instead of pseudo words."""
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
    r"""Get morphological data for a `word`"""
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


def clean_and_enrich_data(
    df: pd.DataFrame, real: bool = False, morph: bool = False
) -> pd.DataFrame:
    r"""Clean column names and add phonemes to the dataset contained in `df`.
    Set `real` to `True` for extended data (Zipf freq, part of speech).
    Set `morph` to `True` to add morphological data (might be consequently slower).
    """
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

    # Add Phonemes column with no stress
    def remove_stress(phonemes):
        return [p[:-1] if p[-1].isdigit() else p for p in phonemes]

    df["No Stress"] = df["Phonemes"].apply(remove_stress)

    # NOTE: Very slow
    # Add Morphological data
    if morph:
        columns = [
            "Prefixes",
            "Roots",
            "Frequencies",
            "Suffixes",
            "Morpheme Count",
            "Structure",
        ]
        df[columns] = df["Word"].apply(
            lambda word: pd.Series(get_morphological_data(word))
        )

    return df


def create_test_data() -> pd.DataFrame:
    r"""Combine and reformat the real and pseudo word datasets"""
    # Process real words
    handmade_real_path = get_dataset_dir() / "handmade" / "test_dataset_real"
    csv_real_path = get_dataframe_dir() / "real_test.csv"
    real_words = process_dataset(handmade_real_path, real=True)
    real_words = clean_and_enrich_data(real_words, real=True)
    real_words.to_csv(csv_real_path)

    # Process pseudo words
    handmade_pseudo_path = get_dataset_dir() / "handmade" / "test_dataset_pseudo"
    csv_pseudo_path = get_dataframe_dir() / "pseudo_test.csv"
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
        "No Stress",
    ]
    dataframe = dataframe.reindex(columns=columns)

    csv_complete_path = get_dataframe_dir() / "complete_test.csv"
    dataframe.to_csv(csv_complete_path)

    return dataframe


def get_test_data(force_recreate: bool = False) -> pd.DataFrame:
    r"""Return dataframe of aggregated test data.
    Set `force_recreate` to `True` to enforce recomputation of the data."""
    csv_test_path = get_dataframe_dir() / "complete_test.csv"
    if csv_test_path.exists() and not force_recreate:
        dataframe = pd.read_csv(csv_test_path, index_col=0, converters={"Word": str})
        dataframe["Phonemes"] = dataframe["Phonemes"].apply(literal_eval)
        dataframe["No Stress"] = dataframe["No Stress"].apply(literal_eval)
    else:
        dataframe = create_test_data()
    return dataframe


def create_train_data(num_unique_words: int = 50000) -> pd.DataFrame:
    r"""Create a training dataset with `num_unique_words` words selected from the most frequent english words.

    Data is enrichened with word Frequency, Zipf Frequency, Part of Speech and Phonemes
    """
    word_list = []
    freq_list = []
    count = 0
    for word in iter_wordlist("en"):
        # Skip any non-alphabetic words
        if not word.isalpha():
            continue
        # Skip any words that don't have vowels
        if not any(char in "aeiouy" for char in word):
            continue

        word_list.append(word)
        freq_list.append(word_frequency(word, "en"))
        count += 1
        if count == num_unique_words:
            break

    if count != num_unique_words:
        raise RuntimeError(
            f"Extracted {count}/{num_unique_words} words before exhausting vocabulary."
        )

    dataframe = pd.DataFrame({"Word": word_list, "Frequency": freq_list})
    dataframe = clean_and_enrich_data(dataframe, real=True)

    csv_train_path = get_dataframe_dir() / "complete_train.csv"
    dataframe.to_csv(csv_train_path)

    return dataframe


def get_train_data(force_recreate: bool = False) -> pd.DataFrame:
    r"""Get saved training dataset if it exists, create it otherwise.

    Use `force_recreate` to recreate the training set from scratch"""
    csv_train_path = get_dataframe_dir() / "complete_train.csv"
    if csv_train_path.exists() and not force_recreate:
        dataframe = pd.read_csv(csv_train_path, index_col=0, converters={"Word": str})
        dataframe["Phonemes"] = dataframe["Phonemes"].apply(literal_eval)
        dataframe["No Stress"] = dataframe["No Stress"].apply(literal_eval)
    else:
        dataframe = create_train_data()
    return dataframe


def create_folds(
    train_data: pd.DataFrame,
    num_folds: int = 5,
    generator: np.random.Generator | None = None,
) -> None:
    r"""Create `num_folds` equilibrated folds from `train_data`.

    Training folds differ of at most 1 sample in size, same for validation folds.

    Randomness of splits can be controlled through the `generator` argument.
    If left as `None`, a generator is deterministically seeded and used.
    """
    # TODO check that folds are balanced
    if generator is None:
        generator = np.random.default_rng(seed=42)
    folds_dir = get_folds_dir()
    dataset_len = len(train_data.index)

    fold_ids = np.array([i % num_folds for i in range(dataset_len)])
    generator.shuffle(fold_ids)

    for i in range(num_folds):
        mask: np.ndarray = fold_ids == i

        ith_train_fold = train_data[np.logical_not(mask)].reset_index(drop=True)
        ith_valid_fold = train_data[mask].reset_index(drop=True)

        csv_ith_train_fold_path = folds_dir / f"train_fold_{i}.csv"
        csv_ith_valid_fold_path = folds_dir / f"valid_fold_{i}.csv"

        ith_train_fold.to_csv(csv_ith_train_fold_path)
        ith_valid_fold.to_csv(csv_ith_valid_fold_path)


def get_train_fold(fold_id: int, force_recreate: bool = False) -> pd.DataFrame:
    r"""Get saved training fold number `fold_id` if it exists, recreate all folds otherwise.

    Use `force_recreate` to recreate training set and folds from scratch"""
    csv_train_fold_path = get_folds_dir() / f"train_fold_{fold_id}.csv"
    if force_recreate or not csv_train_fold_path.exists():
        train_df = get_train_data(force_recreate)
        create_folds(train_df)
    dataframe = pd.read_csv(csv_train_fold_path, index_col=0, converters={"Word": str})
    dataframe["Phonemes"] = dataframe["Phonemes"].apply(literal_eval)
    dataframe["No Stress"] = dataframe["No Stress"].apply(literal_eval)
    return dataframe


def get_valid_fold(fold_id: int, force_recreate: bool = False) -> pd.DataFrame:
    r"""Get saved validation fold number `fold_id` if it exists, recreate all folds otherwise.

    Use `force_recreate` to recreate training set and folds from scratch"""
    csv_valid_fold_path = get_folds_dir() / f"valid_fold_{fold_id}.csv"
    if force_recreate or not csv_valid_fold_path.exists():
        train_df = get_train_data(force_recreate)
        create_folds(train_df)
    dataframe = pd.read_csv(csv_valid_fold_path, index_col=0, converters={"Word": str})
    dataframe["Phonemes"] = dataframe["Phonemes"].apply(literal_eval)
    dataframe["No Stress"] = dataframe["No Stress"].apply(literal_eval)
    return dataframe


def create_epoch(
    fold_id: int,
    train_split: pd.DataFrame,
    epoch_size: int = 100000000,
    generator: np.random.Generator | None = None,
) -> np.ndarray:
    r"""Samples `epoch_size` samples from the training split `train_split`.
    Saves the generated ids in a `.npy` file depeding on ̀`fold_id`.

    Generated epoch contains at least each sample once, and is sampled according to word frequency.

    Randomness of sampling can be controlled through `generator`.
    If left `None`, is instantiated in a deterministic way.
    """
    if generator is None:
        generator = np.random.default_rng(seed=42)
    array_epoch_path = get_folds_dir() / f"epoch_fold_{fold_id}.npy"
    indices = train_split.index.to_numpy()
    weights = train_split["Frequency"].to_numpy()
    normalized_weights = weights / np.sum(weights)
    # Generate epoch_size samples
    weighted_samples = generator.choice(indices, epoch_size, p=normalized_weights)
    # For each class, remove first occurence as it will be enforced later
    filtered_samples = weighted_samples[pd.Series(weighted_samples).duplicated()]
    # If still too long, truncate further
    truncated_samples = filtered_samples[: epoch_size - len(indices)]
    # Enforce first occurence
    samples = np.concatenate([indices, truncated_samples])
    generator.shuffle(samples)
    np.save(array_epoch_path, samples)
    return samples


def get_epoch_numpy(
    fold_id: int, force_recreate: bool = False, epoch_size: int = 10**8
) -> np.ndarray:
    r"""Get saved training fold `fold_id` epoch ids as numpy array if they exist, create them otherwise.

    Use `force_recreate` to recreate training set and folds from scratch"""
    array_epoch_path = get_folds_dir() / f"epoch_fold_{fold_id}.npy"
    if array_epoch_path.exists() and not force_recreate:
        indices = np.load(array_epoch_path)
    else:
        train_fold = get_train_fold(fold_id, force_recreate)
        indices = create_epoch(fold_id, train_fold, epoch_size)
    return indices


def get_epoch(fold_id: int, force_recreate: bool = False) -> pd.DataFrame:
    r"""Get saved training fold `fold_id` epoch dataframe if epoch ids exist, create them otherwise.

    Use `force_recreate` to recreate training set and folds from scratch"""
    array_epoch_path = get_folds_dir() / f"epoch_fold_{fold_id}.npy"
    train_fold = get_train_fold(fold_id, force_recreate)
    if array_epoch_path.exists() and not force_recreate:
        indices = np.load(array_epoch_path)
    else:
        indices = create_epoch(fold_id, train_fold)
    return train_fold.iloc[indices]


def phoneme_statistics(phonemes: list):
    # TODO docstring
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
    r"""Return a dictionary mapping words to their frequency from the data inside `word_data`."""
    words = word_data["Word"]
    freqs = word_data["Frequency"]
    word_to_freq = {}
    for word, freq in zip(words, freqs):
        word_to_freq[word] = freq
    return word_to_freq


def create_phoneme_to_id(train_data: pd.DataFrame) -> dict[str, int]:
    r"""Creates a dictionary mapping every phonemes present in `train_data` to ids for tokenization.
    Extra tokens are `<SOS>`, `<EOS>` and `<PAD>`."""
    phoneme_dict_path = get_dataset_dir() / "phonemes_to_id.json"
    phonemes = train_data["Phonemes"]
    phonemes_unique = set().union(*phonemes)
    sorted_phonemes = sorted(list(phonemes_unique))
    all_tokens = ["<PAD>", "<SOS>", "<EOS>"] + sorted_phonemes
    phoneme_to_id = {token: i for i, token in enumerate(all_tokens)}
    with phoneme_dict_path.open("w") as f:
        json.dump(phoneme_to_id, f)
    return phoneme_to_id


def get_phoneme_to_id(force_recreate: bool = False) -> dict[str, int]:
    r"""Get saved validation phoneme to id dictionary if it exists, recreate it otherwise.

    Use `force_recreate` to recreate training set from scratch"""
    phoneme_dict_path = get_dataset_dir() / "phonemes_to_id.json"
    if phoneme_dict_path.exists() and not force_recreate:
        with phoneme_dict_path.open("r") as f:
            phoneme_dict = json.load(f)
    else:
        train_data = get_train_data(force_recreate)
        phoneme_dict = create_phoneme_to_id(train_data)
    return phoneme_dict
