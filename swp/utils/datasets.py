import json
from ast import literal_eval
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
import spacy.cli
from g2p_en import G2p
from Levenshtein import editops
from morphemes import Morphemes
from nltk.corpus import cmudict
from wordfreq import word_frequency, zipf_frequency

from .abbreviations import abbreviations_en
from .paths import (
    get_dataframe_dir,
    get_folds_dir,
    get_handmade_dir,
    get_morphemes_dir,
    get_stimuli_dir,
)


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
    # TODO check that the path is ok
    mrp = Morphemes(str(get_morphemes_dir()))
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
    # if not spacy.util.is_package("en_core_web_lg"):
    #     spacy.cli.download("en_core_web_lg")
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
    else:
        df["Zipf Frequency"] = 0.0

    # Add Phonemes column
    cmu = cmudict.dict()
    g2p = G2p()

    def phonemize(word: str) -> list[str]:
        if word in cmu:
            return cmu[word][0]
        else:
            return g2p(word)

    df["Phonemes"] = df["Word"].apply(phonemize)

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


def enrich_for_plotting(df: pd.DataFrame, include_stress: bool) -> pd.DataFrame:
    """
    Calculate error and bigram statistics for each row in the DataFrame and append them as new columns.

    Args:
        df (pd.DataFrame): The input DataFrame with columns for phonemes, predictions, and optionally stress.
        phoneme_to_id (dict): A mapping of phonemes to their corresponding IDs.
        include_stress (bool): Whether to use the "Phonemes" column or "No Stress" column.

    Returns:
        pd.DataFrame: The DataFrame with additional columns for error statistics.
    """
    phoneme_to_id = get_phoneme_to_id(include_stress)
    phoneme_key = "Phonemes" if include_stress else "No Stress"
    # df[phoneme_key] = df[phoneme_key].apply(literal_eval)
    # df["Prediction"] = df["Prediction"].apply(literal_eval)
    df = df[df[phoneme_key].apply(len) > 1].copy()

    # Initialize lists to store results
    edit_distances = []
    insertions = []
    deletions = []
    substitutions = []
    sequence_lengths = []
    error_indices = []
    bigram_frequency = []

    stress = "sw" if include_stress else "sn"
    stats_dir = get_stimuli_dir() / "statistics"
    bigram_stats_df = pd.read_csv(stats_dir / f"bigram_stats_{stress}.csv")
    bigram_to_freq = dict(
        zip(bigram_stats_df["Bigram"], bigram_stats_df["Normalized Frequency"])
    )

    for _, row in df.iterrows():
        # Compute average bigram frequency for the sequence
        phonemes = row[phoneme_key]
        bigrams = [" ".join(phonemes[i : i + 2]) for i in range(len(phonemes) - 1)]
        bigram_freqs = [bigram_to_freq.get(bigram, 0) for bigram in bigrams]
        avg_bigram_freq = sum(bigram_freqs) / len(bigram_freqs)

        # Tally edit operations and identify error indices
        phonemes = [phoneme_to_id[p] for p in phonemes]
        prediction = [phoneme_to_id[p] for p in row["Prediction"]]
        errors = editops(phonemes, prediction)
        counts = Counter(op for op, _, _ in errors)
        mismatched_indices = [
            i + 1 for i, (j, k) in enumerate(zip(phonemes, prediction)) if j != k
        ]

        # Append results to the respective lists
        edit_distances.append(len(errors))
        insertions.append(counts["insert"])
        deletions.append(counts["delete"])
        substitutions.append(counts["replace"])
        sequence_lengths.append(len(phonemes))
        error_indices.append(mismatched_indices)
        bigram_frequency.append(avg_bigram_freq)

    # Add results as new columns to the DataFrame
    df["Edit Distance"] = edit_distances
    df["Insertions"] = insertions
    df["Deletions"] = deletions
    df["Substitutions"] = substitutions
    df["Sequence Length"] = sequence_lengths
    df["Error Indices"] = error_indices
    df["Bigram Frequency"] = bigram_frequency

    return df


def enrich_for_ablations(df: pd.DataFrame) -> pd.DataFrame:
    r"""Enrich training data with word size and morphological complexity."""
    df = df.copy()
    mrp = Morphemes(str(get_morphemes_dir()))

    if "Size" not in df.columns:
        df["Size"] = df["Word"].apply(lambda x: "short" if len(x) <= 6 else "long")
    else:
        missing_ids = df["Size"].isna()
        df.loc[missing_ids, "Size"] = df.loc[missing_ids, "Word"].apply(
            lambda x: "short" if len(x) <= 6 else "long"
        )

    # TODO make a set with complex and simple words for lookup
    # if not found, then use the morphemes API
    if "Morphology" not in df.columns:
        df["Morphology"] = df["Word"].apply(
            lambda x: "simple" if mrp.parse(x)["morpheme_count"] <= 1 else "complex"
        )
    else:
        missing_ids = df["Morphology"].isna()
        df.loc[missing_ids, "Morphology"] = df.loc[missing_ids, "Word"].apply(
            lambda x: "simple" if mrp.parse(x)["morpheme_count"] <= 1 else "complex"
        )

    return df


def classify_error_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'Primacy Error' and 'Recency Error' columns to the DataFrame.

    For each row:
      - 'Primacy Error' is 1 if any index in 'Error Indices' is less than or equal to half
        of 'Sequence Length'; otherwise 0.
      - 'Recency Error' is 1 if any index in 'Error Indices' is greater than half
        of 'Sequence Length'; otherwise 0.
    """
    df = df.copy()

    def classify(row):
        threshold = row["Sequence Length"] // 2
        return pd.Series(
            {
                "Primacy Error": int(
                    any(idx <= threshold for idx in row["Error Indices"])
                ),
                "Recency Error": int(
                    any(idx > threshold for idx in row["Error Indices"])
                ),
            }
        )

    df.loc[:, ["Primacy Error", "Recency Error"]] = df.apply(classify, axis=1)
    return df


def get_test_data() -> pd.DataFrame:
    r"""Return dataframe of aggregated test data.
    Set `force_recreate` to `True` to enforce recomputation of the data."""
    csv_test_path = get_handmade_dir() / "test_equalized.csv"
    if csv_test_path.exists():
        dataframe = pd.read_csv(
            csv_test_path,
            # index_col=0,
            converters={
                "Word": str,
                "Phonemes": literal_eval,
                "No Stress": literal_eval,
            },
        )
    else:
        raise FileNotFoundError("User does not have the evaluation dataset.")

    return dataframe


def get_curated_words() -> list[tuple[str, float]]:
    r"""Returns a list of curated english words sorted by descending frequency"""
    # TODO add support for language maybe ?
    cmu = cmudict.dict()
    curated = []
    for k in cmu:
        freq = word_frequency(k, "en")
        if (
            k.isalpha()
            and freq != 0
            and (k in {"a", "i"}) ** (len(k) == 1)  # reverse implication
            and k not in abbreviations_en
        ):
            curated.append((k, freq))
    curated.sort(key=lambda x: x[1], reverse=True)
    return curated


def create_train_data(num_unique_words: int = 50000) -> pd.DataFrame:
    r"""Create a training dataset with `num_unique_words` words selected from the most frequent english words.

    Data is enrichened with word Frequency, Zipf Frequency, Part of Speech and Phonemes
    """
    curated = get_curated_words()
    if num_unique_words > len(curated):
        raise ValueError(
            f"Not enough unique words to pick from, only {len(curated)} words available"
        )
    selected = curated[:num_unique_words]
    word_tuple, freq_tuple = zip(*selected)

    dataframe = pd.DataFrame({"Word": list(word_tuple), "Frequency": list(freq_tuple)})
    dataframe = clean_and_enrich_data(dataframe, real=True)
    csv_train_path = get_dataframe_dir() / "complete_train.csv"
    dataframe.to_csv(csv_train_path)

    # ablation_train = dataframe.sample(frac=0.1)
    # ablation_train = enrich_for_ablations(ablation_train)
    # ablation_train.to_csv(get_dataframe_dir() / "ablation_train.csv")

    return dataframe


def get_train_data(force_recreate: bool = False) -> pd.DataFrame:
    r"""Get saved training dataset if it exists, create it otherwise.

    Use `force_recreate` to recreate the training set from scratch"""
    csv_train_path = get_dataframe_dir() / "complete_train.csv"
    if csv_train_path.exists() and not force_recreate:
        dataframe = pd.read_csv(
            csv_train_path,
            index_col=0,
            converters={
                "Word": str,
                "Phonemes": literal_eval,
                "No Stress": literal_eval,
            },
        )
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


def get_train_fold(fold_id: int | None, force_recreate: bool = False) -> pd.DataFrame:
    r"""Get saved training fold number `fold_id` if it exists, recreate all folds otherwise.
    If `fold_id` is None, return the complete training set.

    Use `force_recreate` to recreate training set and folds from scratch"""
    train_df = None
    csv_train_fold_path = get_folds_dir() / f"train_fold_{fold_id}.csv"
    if force_recreate or not csv_train_fold_path.exists():
        train_df = get_train_data(force_recreate)
        create_folds(train_df)
    if fold_id is None:
        dataframe = get_train_data(force_recreate) if train_df is None else train_df
    else:
        dataframe = pd.read_csv(
            csv_train_fold_path,
            index_col=0,
            converters={
                "Word": str,
                "Phonemes": literal_eval,
                "No Stress": literal_eval,
            },
        )
    return dataframe


def get_valid_fold(fold_id: int | None, force_recreate: bool = False) -> pd.DataFrame:
    r"""Get saved validation fold number `fold_id` if it exists, recreate all folds otherwise.
    If `fold_id` is None, return the complete training set.

    Use `force_recreate` to recreate training set and folds from scratch"""
    train_df = None
    csv_valid_fold_path = get_folds_dir() / f"valid_fold_{fold_id}.csv"
    if force_recreate or not csv_valid_fold_path.exists():
        train_df = get_train_data(force_recreate)
        create_folds(train_df)
    if fold_id is None:
        dataframe = get_train_data(force_recreate) if train_df is None else train_df
    else:
        dataframe = pd.read_csv(
            csv_valid_fold_path,
            index_col=0,
            converters={
                "Word": str,
                "Phonemes": literal_eval,
                "No Stress": literal_eval,
            },
        )
    return dataframe


def create_epoch(
    fold_id: int | None,
    train_data: pd.DataFrame,
    epoch_size: int = 10**6,
    generator: np.random.Generator | None = None,
) -> np.ndarray:
    r"""Samples `epoch_size` samples from the training split `train_data`.
    Saves the generated ids in a `.npy` file depeding on ̀`fold_id`.
    If `fold_id` is None, consider it is a complete set and save it under a specific name.

    Generated epoch contains at least each sample once, and is sampled according to word frequency.

    Randomness of sampling can be controlled through `generator`.
    If left `None`, is instantiated in a deterministic way.
    """
    if generator is None:
        generator = np.random.default_rng(seed=42)
    array_epoch_path = (
        get_folds_dir()
        / f"epoch_{'complete' if fold_id is None else f'fold_{fold_id}'}.npy"
    )
    indices = train_data.index.to_numpy()
    weights = train_data["Frequency"].to_numpy()
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
    fold_id: int | None, force_recreate: bool = False, epoch_size: int = 10**8
) -> np.ndarray:
    r"""Get saved training fold `fold_id` epoch ids as numpy array if they exist, create them otherwise.

    Use `force_recreate` to recreate training set and folds from scratch"""
    array_epoch_path = (
        get_folds_dir()
        / f"epoch_{'complete' if fold_id is None else f'fold_{fold_id}'}.npy"
    )
    if array_epoch_path.exists() and not force_recreate:
        indices = np.load(array_epoch_path)
    else:
        train_fold = get_train_fold(fold_id, force_recreate)
        indices = create_epoch(fold_id, train_fold, epoch_size)
    return indices


def get_epoch(fold_id: int | None, force_recreate: bool = False) -> pd.DataFrame:
    r"""Get saved training fold `fold_id` epoch dataframe if epoch ids exist, create them otherwise.

    Use `force_recreate` to recreate training set and folds from scratch"""
    array_epoch_path = (
        get_folds_dir()
        / f"epoch_{'complete' if fold_id is None else f'fold_{fold_id}'}.npy"
    )
    train_fold = get_train_fold(fold_id, force_recreate)
    if array_epoch_path.exists() and not force_recreate:
        indices = np.load(array_epoch_path)
    else:
        indices = create_epoch(fold_id, train_fold)
    return train_fold.iloc[indices]


def get_phoneme_statistics(train_df: pd.DataFrame):
    # TODO Daniel docstring

    # iterate over the rows of the dataframe
    for phoneme_key in ["Phonemes", "No Stress"]:
        phoneme_stats = defaultdict(int)
        bigram_stats = defaultdict(int)
        trigram_stats = defaultdict(int)

        for _, row in train_df.iterrows():
            phonemes = row[phoneme_key]
            for phoneme in phonemes:
                phoneme_stats[phoneme] += 1 * row["Frequency"]

            for i in range(len(phonemes) - 1):
                bigram = " ".join(phonemes[i : i + 2])
                bigram_stats[bigram] += 1 * row["Frequency"]

            for i in range(len(phonemes) - 2):
                trigram = " ".join(phonemes[i : i + 3])
                trigram_stats[trigram] += 1 * row["Frequency"]

        # Normalize frequencies
        phoneme_total = sum(phoneme_stats.values())
        bigram_total = sum(bigram_stats.values())
        trigram_total = sum(trigram_stats.values())

        # Convert to dataframes with normalized frequencies
        phoneme_df = pd.DataFrame(
            [(k, v, v / phoneme_total) for k, v in phoneme_stats.items()],
            columns=["Phoneme", "Frequency", "Normalized Frequency"],
        )
        bigram_df = pd.DataFrame(
            [(k, v, v / bigram_total) for k, v in bigram_stats.items()],
            columns=["Bigram", "Frequency", "Normalized Frequency"],
        )
        trigram_df = pd.DataFrame(
            [(k, v, v / trigram_total) for k, v in trigram_stats.items()],
            columns=["Trigram", "Frequency", "Normalized Frequency"],
        )

        statistics_dir = get_stimuli_dir() / "statistics"
        statistics_dir.mkdir(exist_ok=True, parents=True)

        if phoneme_key == "Phonemes":
            phoneme_df.to_csv(statistics_dir / "phoneme_stats_sw.csv")
            bigram_df.to_csv(statistics_dir / "bigram_stats_sw.csv")
            trigram_df.to_csv(statistics_dir / "trigram_stats_sw.csv")
        else:
            phoneme_df.to_csv(statistics_dir / "phoneme_stats_sn.csv")
            bigram_df.to_csv(statistics_dir / "bigram_stats_sn.csv")
            trigram_df.to_csv(statistics_dir / "trigram_stats_sn.csv")


def get_word_to_freq(word_data: pd.DataFrame) -> dict[str, float]:
    r"""Return a dictionary mapping words to their frequency from the data inside `word_data`."""
    words = word_data["Word"]
    freqs = word_data["Frequency"]
    word_to_freq = {}
    for word, freq in zip(words, freqs):
        word_to_freq[word] = freq
    return word_to_freq


def create_phoneme_to_id(
    train_data: pd.DataFrame, include_stress: bool = False
) -> dict[str, int]:
    r"""Creates a dictionary mapping every phonemes present in `train_data` to ids for tokenization.
    Extra tokens are `<SOS>`, `<EOS>` and `<PAD>`."""
    phonemes = train_data["Phonemes"]
    phonemes_unique = set().union(*phonemes)
    sorted_phonemes = sorted(list(phonemes_unique))
    all_tokens = ["<PAD>", "<SOS>", "<EOS>"] + sorted_phonemes

    phoneme_to_id_sw = {}
    phoneme_to_id_sn = {}
    no_stress_index = 0
    for i, token in enumerate(all_tokens):
        phoneme_to_id_sw[token] = i
        if token[-1].isdigit():
            no_stress_token = token[:-1]
        else:
            no_stress_token = token
        if no_stress_token not in phoneme_to_id_sn:
            phoneme_to_id_sn[no_stress_token] = no_stress_index
            no_stress_index += 1

    phoneme_dict_path = get_stimuli_dir() / "phonemes_to_id_sw.json"
    with phoneme_dict_path.open("w") as f:
        json.dump(phoneme_to_id_sw, f, indent=4)

    phoneme_dict_path = get_stimuli_dir() / "phonemes_to_id_sn.json"
    with phoneme_dict_path.open("w") as f:
        json.dump(phoneme_to_id_sn, f, indent=4)

    if include_stress:
        return phoneme_to_id_sw
    else:
        return phoneme_to_id_sn


def get_phoneme_to_id(
    include_stress: bool = False,
    force_recreate: bool = False,
) -> dict[str, int]:
    r"""Get saved validation phoneme to id dictionary if it exists, recreate it otherwise.

    Use `force_recreate` to recreate training set from scratch"""
    phoneme_dict_path = f'phonemes_to_id{"_sw" if include_stress else "_sn"}.json'
    phoneme_dict_path = get_stimuli_dir() / phoneme_dict_path
    if phoneme_dict_path.exists() and not force_recreate:
        with phoneme_dict_path.open("r") as f:
            phoneme_dict = json.load(f)
    else:
        train_data = get_train_data(force_recreate)
        phoneme_dict = create_phoneme_to_id(train_data, include_stress)
    return phoneme_dict
