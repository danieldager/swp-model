from ast import literal_eval
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nested import nested_tensor
from torch.utils.data import DataLoader, Dataset

from ..utils.datasets import (
    get_epoch_numpy,
    get_evaluation_dataset,
    get_phoneme_to_id,
    get_train_dataset,
    get_train_fold,
    get_valid_fold,
)
from ..utils.paths import get_dataframe_dir, get_handmade_dir


class PhonemeTrainDataset(Dataset):
    r"""Dataset class to handle phonemes dataset with folds.

    Training fold is used if ̀`train` is set to ̀`True`, validation otherwise.

    Args :
        `fold_id` : fold number to load classes from, if `None`, returns the complete training set
        `train` : return training split if set to `True`, validation split otherwise, useless when `fold_id` is None
        `phoneme_to_id` : dict mapping phonemes to int for tokenization
        `include_stress` : if set to `True`, the phonemes will include stress

    Attributes :
        `fold_id` : index of loaded fold
        `train` : bool indicating if it is training split
        `data_df` : DataFrame containing all the fold data
        `epoch_ids` : ids to use through one epoch to access the data in `data_df`
        `phoneme_to_id` : dict mapping phonemes to int for tokenization
    """

    # is map-style dataset
    def __init__(
        self,
        fold_id: int | None,
        train: bool,
        phoneme_to_id: dict[str, int],
        include_stress: bool = False,
    ):
        self.fold_id = fold_id
        self.train = train
        if self.train:
            self.data_df = get_train_fold(fold_id)
            self.epoch_ids = get_epoch_numpy(fold_id=fold_id, epoch_size=int(1e6))
        else:
            self.data_df = get_valid_fold(self.fold_id)
            self.epoch_ids = np.arange(len(self.data_df))
        self.phoneme_to_id = phoneme_to_id
        self.phoneme_key = "Phonemes" if include_stress else "No_Stress"

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        phonemes: list[str] = self.data_df.iloc[self.epoch_ids[index]][
            self.phoneme_key
        ].copy()
        phonemes.append("<EOS>")
        tokenized = torch.tensor(
            [self.phoneme_to_id[phoneme] for phoneme in phonemes], dtype=torch.long
        )
        return tokenized, tokenized.clone()

    def __len__(self) -> int:
        return len(self.epoch_ids)


def phoneme_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]], pad_value: int):
    r"""A collate function that get tensors of different length from `batch`, then
    batch them together by extending them to the max length, filling with `pad_value`"""
    data, target = tuple(zip(*batch))
    nt_data = nested_tensor(list(data), dtype=torch.long)
    nt_target = nested_tensor(list(target), dtype=torch.long)
    padded_data = nt_data.to_padded_tensor(padding=pad_value)
    padded_target = nt_target.to_padded_tensor(padding=pad_value)
    return padded_data, padded_target


def get_phoneme_trainloader(
    fold_id: int | None,
    train: bool,
    batch_size: int,
    generator: torch.Generator | None = None,
    include_stress: bool = False,
) -> DataLoader:
    r"""Return a dataloader containing the phoneme training data corresponding to the `fold_id` fold, batched in size `batch_size`.
    Shuffling is controlled by `generator`. If `generator` is None, it is deterministically instantiated.

    If `include_stress` is set to `True`, phonemes will include stress.

    Return the corresponding training data if `train` is set to `True`.
    Return the validation data otherwise.
    If `fold_id` is None, returns the complete training set, independently of `train` value.
    """
    phoneme_to_id = get_phoneme_to_id()
    phoneme_set = PhonemeTrainDataset(
        fold_id=fold_id,
        train=train,
        phoneme_to_id=phoneme_to_id,
        include_stress=include_stress,
    )
    if generator is None:
        generator = torch.Generator().manual_seed(42)
    phoneme_loader = DataLoader(
        phoneme_set,
        batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=lambda batch: phoneme_collate_fn(
            batch, pad_value=phoneme_to_id["<PAD>"]
        ),
    )
    return phoneme_loader


class PhonemeTestDataset(Dataset):
    r"""Dataset class to handle phonemes dataset with folds.

    Training fold is used if ̀`train` is set to ̀`True`, validation otherwise.

    Args :
        `phoneme_to_id` : dict mapping phonemes to int for tokenization
        `include_stress` : if set to `True`, the phonemes will include stress
        `override_data_df` : an optionnal `pandas.DataFrame` that can be passed to override the original test data

    Attributes :
        `fold_id` : index of loaded fold
        `train` : bool indicating if it is training split
        `data_df` : DataFrame containing all the fold data
        `epoch_ids` : ids to use through one epoch to access the data in `data_df`
        `phoneme_to_id` : dict mapping phonemes to int for tokenization
    """

    # is map-style dataset
    def __init__(
        self,
        phoneme_to_id: dict[str, int],
        include_stress: bool = False,
        dataset_df: pd.DataFrame | None = None,
    ):
        if dataset_df is None:
            self.data_df = get_evaluation_dataset()
        else:
            self.data_df = dataset_df
        self.phoneme_to_id = phoneme_to_id
        self.phoneme_key = "Phonemes" if include_stress else "No_Stress"

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        phonemes: list[str] = self.data_df.iloc[index][self.phoneme_key].copy()
        phonemes.append("<EOS>")
        tokenized = torch.Tensor([self.phoneme_to_id[phoneme] for phoneme in phonemes])
        return tokenized, tokenized.clone()

    def __len__(self) -> int:
        return len(self.data_df)


def get_phoneme_testloader(
    batch_size: int,
    include_stress: bool = False,
    dataset_df: pd.DataFrame | None = None,
) -> DataLoader:
    r"""Return a dataloader containing the phoneme test data batched in size `batch_size`.
    If `include_stress` is set to `True`, phonemes will include stress.
    Pass a dataframe as `override_data_df` to override the test data used.
    """
    phoneme_to_id = get_phoneme_to_id(include_stress)
    phoneme_set = PhonemeTestDataset(
        phoneme_to_id=phoneme_to_id,
        include_stress=include_stress,
        dataset_df=dataset_df,
    )
    phoneme_loader = DataLoader(
        phoneme_set,
        batch_size,
        collate_fn=lambda batch: phoneme_collate_fn(
            batch, pad_value=phoneme_to_id["<PAD>"]
        ),
    )
    return phoneme_loader


### Datasets ###

# fmt: off
vowels = [
    "AH0", "OY0", "AA0", "AY0", "ER0", "AO0", "UW0", "IH0", 
    "EH0", "UH0", "IY0", "EY0", "OW0", "AE0", "AW0"
]  # fmt: on
plosives = ["P", "T", "K", "B", "D", "G"]
fricatives = ["F", "TH", "S", "SH", "Z", "ZH", "V", "DH", "HH"]
affricates = ["CH", "JH"]
nasals = ["M", "N", "NG"]
liquids = ["L", "R"]
glides = ["W", "Y"]
consonants = plosives + fricatives + affricates + nasals + liquids + glides
converters = {"Word": str, "Phonemes": literal_eval, "No_Stress": literal_eval}


def get_handmade_dataset(name: str) -> pd.DataFrame:
    path = get_handmade_dir() / f"{name}.csv"
    return pd.read_csv(path, index_col=0, converters=converters)


def son_score(c):
    if c in plosives:
        return 0
    elif c in fricatives:
        return 1
    elif c in affricates:
        return 2
    elif c in nasals:
        return 3
    elif c in liquids:
        return 4
    elif c in glides:
        return 5
    else:
        raise ValueError(
            f"Phoneme {c} was not recognized in any of the consonant classes"
        )


def get_sonority_dataset(include_stress: bool = False) -> pd.DataFrame:
    path = get_dataframe_dir() / f"sonority.csv"

    # check if the sonority.csv exists
    if path.exists():
        return pd.read_csv(path, index_col=0, converters=converters)

    else:
        words1 = []
        words2 = []
        scores = []
        types = []

        for c1 in consonants:
            for c2 in consonants:
                if c1 != c2:
                    score = son_score(c2) - son_score(c1)
                    for v in vowels:
                        # CCV
                        words1.append([c1, c2, v])
                        words2.append([c1, c2, v[:-1]])
                        scores.append(score)
                        types.append("CCV")

                        # VCC
                        words1.append([v, c1, c2])
                        words2.append([v[:-1], c1, c2])
                        scores.append(-score)
                        types.append("VCC")

        # create sonority dataset
        sonority_dataset = pd.DataFrame(
            {
                "Phonemes": words1,
                "No_Stress": words2,
                "Sonority": scores,
                "Type": types,
            }
        )
        sonority_dataset["Length"] = 3

        train_df = get_train_dataset()
        train_3 = train_df[train_df["No_Stress"].apply(lambda x: len(x)) == 3]
        train3_set = set()
        for phonemes in train_3["No_Stress"]:
            train3_set.add(tuple(phonemes))  # only tuples are hashable
        filtered_sonority_dataset = sonority_dataset[
            sonority_dataset["No_Stress"].apply(lambda x: tuple(x) not in train3_set)
        ]
        sonority_dataset.to_csv(path)
        return filtered_sonority_dataset


def get_phoneme_dataset() -> pd.DataFrame:
    path = get_dataframe_dir() / f"phonemes.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, converters=converters)

    else:
        data = {
            "Phonemes": [],
            "No_Stress": [],
            "Type": [],
            "Included": [],
            "Length": [],
        }
        train_df = get_train_dataset()
        train1 = train_df[train_df["No_Stress"].apply(lambda x: len(x)) == 1]
        train1_set = {tuple(phonemes) for phonemes in train1["No_Stress"]}

        phonemes = vowels + consonants
        for p in phonemes:
            data["Phonemes"].append([p])
            no_stress = p[:-1] if p in vowels else p
            data["No_Stress"].append([no_stress])
            data["Type"].append("V" if p in vowels else "C")
            data["Included"].append(True if (no_stress,) in train1_set else False)
            data["Length"].append(1)

        phoneme_df = pd.DataFrame(data)
        phoneme_df.to_csv(path)
        return phoneme_df


def get_bigram_dataset() -> pd.DataFrame:
    path = get_dataframe_dir() / f"bigrams.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, converters=converters)

    else:
        data = {
            "Phonemes": [],
            "No_Stress": [],
            "Type": [],
            "Consonant": [],
            "Vowel": [],
            "Included": [],
        }
        train_df = get_train_dataset()
        train2 = train_df[train_df["No_Stress"].apply(lambda x: len(x)) == 2]
        train2_set = {tuple(phonemes) for phonemes in train2["No_Stress"]}

        for c in consonants:
            for v in vowels:
                # CV
                data["Phonemes"].append([c, v])
                data["No_Stress"].append([c, v[:-1]])
                data["Type"].append("CV")
                data["Consonant"].append(c)
                data["Vowel"].append(v[:-1])
                data["Included"].append(True if (c, v[:-1]) in train2_set else False)

                # VC
                data["Phonemes"].append([v, c])
                data["No_Stress"].append([v[:-1], c])
                data["Type"].append("VC")
                data["Consonant"].append(c)
                data["Vowel"].append(v[:-1])
                data["Included"].append(True if (v[:-1], c) in train2_set else False)

        bigram_df = pd.DataFrame(data)
        bigram_df.to_csv(path)
        return bigram_df


def get_trigram_dataset() -> pd.DataFrame:
    path = get_dataframe_dir() / f"trigrams.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, converters=converters)

    else:
        print("Generating trigram dataset...")
        df = {
            "Phonemes": [],
            "No_Stress": [],
            "Type": [],
            "Real": [],
            "Length": [],
            "First": [],
            "Second": [],
            "Third": [],
        }

        train_df = get_train_dataset()
        train_df = train_df[train_df["No_Stress"].apply(lambda x: len(x)) <= 3]
        train_set = {tuple(phonemes) for phonemes in train_df["No_Stress"]}

        phonemes = vowels + consonants
        for p1 in phonemes:
            df["Phonemes"].append([p1])
            t1 = "V" if p1 in vowels else "C"
            df["Type"].append(t1)
            n1 = p1[:-1] if p1 in vowels else p1
            df["No_Stress"].append([n1])
            df["Real"].append(True if (n1,) in train_set else False)
            df["Length"].append(1)
            df["First"].append(t1)
            df["Second"].append(None)
            df["Third"].append(None)

            for p2 in phonemes:
                df["Phonemes"].append([p1, p2])
                t2 = "V" if p2 in vowels else "C"
                df["Type"].append(t1 + t2)
                n2 = p2[:-1] if p2 in vowels else p2
                df["No_Stress"].append([n1, n2])
                df["Real"].append(True if (n1, n1) in train_set else False)
                df["Length"].append(2)
                df["First"].append(t1)
                df["Second"].append(t2)
                df["Third"].append(None)

                for p3 in phonemes:
                    df["Phonemes"].append([p1, p2, p3])
                    t3 = "V" if p3 in vowels else "C"
                    df["Type"].append(t1 + t2 + t3)
                    n3 = p3[:-1] if p3 in vowels else p3
                    df["No_Stress"].append([n1, n2, n3])
                    df["Real"].append(True if (n1, n2, n3) in train_set else False)
                    df["Length"].append(3)
                    df["First"].append(t1)
                    df["Second"].append(t2)
                    df["Third"].append(t3)

        trigram_df = pd.DataFrame(df)
        trigram_df.to_csv(path)
        return trigram_df
