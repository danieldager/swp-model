from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nested import nested_tensor
from torch.utils.data import DataLoader, Dataset

from ..utils.datasets import (
    get_epoch_numpy,
    get_phoneme_to_id,
    get_test_data,
    get_train_fold,
    get_valid_fold,
)


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
        if self.train or self.fold_id is None:
            self.data_df = get_train_fold(self.fold_id)
            self.epoch_ids = get_epoch_numpy(fold_id=self.fold_id, epoch_size=int(1e6))
        else:
            self.data_df = get_valid_fold(self.fold_id)
            self.epoch_ids = np.arange(len(self.data_df))
        self.phoneme_to_id = phoneme_to_id
        if include_stress:
            self.phoneme_key = "Phonemes"
        else:
            self.phoneme_key = "No Stress"

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
    # TODO Robin docstring
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
        override_data_df: pd.DataFrame | None = None,
    ):
        if override_data_df is None:
            self.data_df = get_test_data()
        else:
            self.data_df = override_data_df
        self.epoch_ids = np.arange(len(self.data_df))
        self.phoneme_to_id = phoneme_to_id
        if include_stress:
            self.phoneme_key = "Phonemes"
        else:
            self.phoneme_key = "No Stress"

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        phonemes: list[str] = self.data_df.iloc[self.epoch_ids[index]][
            self.phoneme_key
        ].copy()
        phonemes.append("<EOS>")
        tokenized = torch.Tensor([self.phoneme_to_id[phoneme] for phoneme in phonemes])
        return tokenized, tokenized.clone()

    def __len__(self) -> int:
        return len(self.epoch_ids)


def get_phoneme_testloader(
    batch_size: int,
    include_stress: bool = False,
    override_data_df: pd.DataFrame | None = None,
) -> DataLoader:
    r"""Return a dataloader containing the phoneme test data batched in size `batch_size`.
    If `include_stress` is set to `True`, phonemes will include stress.
    Pass a dataframe as `override_data_df` to override the test data used.
    """
    phoneme_to_id = get_phoneme_to_id(include_stress=include_stress)
    phoneme_set = PhonemeTestDataset(
        phoneme_to_id=phoneme_to_id,
        include_stress=include_stress,
        override_data_df=override_data_df,
    )
    phoneme_loader = DataLoader(
        phoneme_set,
        batch_size,
        collate_fn=lambda batch: phoneme_collate_fn(
            batch, pad_value=phoneme_to_id["<PAD>"]
        ),
    )
    return phoneme_loader


def get_sonority_dataset(include_stress: bool = False) -> pd.DataFrame:
    # TODO Daniel docstring, save dataset to file

    vowels = [
        "AH0",
        "OY0",
        "AA0",
        "AY0",
        "ER0",
        "AO0",
        "UW0",
        "IH0",
        "EH0",
        "UH0",
        "IY0",
        "EY0",
        "OW0",
        "AE0",
        "AW0",
    ]

    if not include_stress:
        vowels = [vowel[:-1] for vowel in vowels]

    plosives = ["P", "T", "K", "B", "D", "G"]
    fricatives = ["F", "TH", "S", "SH", "Z", "ZH", "V", "DH", "HH"]
    affricates = ["CH", "JH"]
    nasals = ["M", "N", "NG"]
    liquids = ["L", "R"]
    glides = ["W", "Y"]

    consonants = plosives + fricatives + affricates + nasals + liquids + glides

    def get_sonority(c):
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

    def get_sonority_score(c1, c2):
        return get_sonority(c2) - get_sonority(c1)

    ccv = {}
    vcc = {}
    for c1 in consonants:
        for c2 in consonants:
            if c1 != c2:
                score = get_sonority_score(c1, c2)
                for v in vowels:
                    ccv[(c1, c2, v)] = score
                    vcc[(v, c1, c2)] = score

    # create test dataframe for sonority plotting
    ccv_df = pd.DataFrame(
        # [(repr(list(phonemes)), score) for phonemes, score in ccv.items()],
        [(list(phonemes), score) for phonemes, score in ccv.items()],
        columns=["Phonemes" if include_stress else "No Stress", "Sonority"],
    )
    vcc_df = pd.DataFrame(
        # [(repr(list(phonemes)), score) for phonemes, score in vcc.items()],
        [(list(phonemes), score) for phonemes, score in vcc.items()],
        columns=["Phonemes" if include_stress else "No Stress", "Sonority"],
    )
    ccv_df["Type"] = "CCV"
    vcc_df["Type"] = "VCC"
    sonority_dataset = pd.concat([ccv_df, vcc_df])

    return sonority_dataset
