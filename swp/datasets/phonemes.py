from typing import Any

import numpy as np
import torch
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
        `fold_id` : fold number to load classes from
        `train` : return training split if set to `True`, validation split otherwise
        `phoneme_to_id` : dict mapping phonemes to int for tokenization
        `pad_to_length` : length up to which the dataset should pad

    Attributes :
        `fold_id` : index of loaded fold
        `train` : bool indicating if it is training split
        `data_df` : DataFrame containing all the fold data
        `epoch_ids` : ids to use through one epoch to access the data in `data_df`
        `pad_to_length` : length up to which the dataset is padding
        `phoneme_to_id` : dict mapping phonemes to int for tokenization
    """

    # is map-style dataset
    def __init__(
        self,
        fold_id: int,
        train: bool,
        phoneme_to_id: dict[str, int],
        pad_to_length: int = 0,
    ):
        self.fold_id = fold_id
        self.train = train
        if self.train:
            self.data_df = get_train_fold(self.fold_id)
            self.epoch_ids = get_epoch_numpy(fold_id=self.fold_id, epoch_size=int(1e6))
        else:
            self.data_df = get_valid_fold(self.fold_id)
            self.epoch_ids = np.arange(len(self.data_df))
        self.phoneme_to_id = phoneme_to_id
        self.pad_length = pad_to_length

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        phonemes: list[str] = self.data_df.iloc[self.epoch_ids[index]]["Phonemes"]
        phonemes.append("<EOS>")
        if self.pad_length > 0:
            phonemes.extend(["<PAD>" for _ in range(self.pad_length - len(phonemes))])
        tokenized = torch.tensor(
            [self.phoneme_to_id[phoneme] for phoneme in phonemes], dtype=torch.long
        )
        return tokenized, tokenized.clone()

    def __len__(self) -> int:
        return len(self.epoch_ids)


def get_phoneme_trainloader(
    fold_id: int,
    train: bool,
    batch_size: int,
    pad_to_length: int,
    generator: torch.Generator | None = None,
) -> DataLoader:
    r"""Return a dataloader containing the phoneme training data corresponding to the `fold_id` fold, batched in size `batch_size`.
    Shuffling is controlled by `generator`. If `generator` is None, it is deterministically instantiated.

    Return the corresponding training data if `train` is set to `True`.
    Return the validation data otherwise.
    """
    phoneme_set = PhonemeTrainDataset(
        fold_id=fold_id,
        train=train,
        phoneme_to_id=get_phoneme_to_id(),
        pad_to_length=pad_to_length,
    )
    if generator is None:
        generator = torch.Generator().manual_seed(42)
    phoneme_loader = DataLoader(
        phoneme_set, batch_size, shuffle=True, generator=generator
    )
    return phoneme_loader


class PhonemeTestDataset(Dataset):
    r"""Dataset class to handle phonemes dataset with folds.

    Training fold is used if ̀`train` is set to ̀`True`, validation otherwise.

    Args :
        `phoneme_to_id` : dict mapping phonemes to int for tokenization
        `pad_to_length` : length up to which the dataset should pad

    Attributes :
        `fold_id` : index of loaded fold
        `train` : bool indicating if it is training split
        `data_df` : DataFrame containing all the fold data
        `epoch_ids` : ids to use through one epoch to access the data in `data_df`
        `pad_to_length` : length up to which the dataset is padding
        `phoneme_to_id` : dict mapping phonemes to int for tokenization
    """

    # is map-style dataset
    def __init__(
        self,
        phoneme_to_id: dict[str, int],
        pad_to_length: int,
    ):
        self.data_df = get_test_data()
        self.epoch_ids = np.arange(len(self.data_df))
        self.phoneme_to_id = phoneme_to_id
        self.pad_length = pad_to_length

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        phonemes: list[str] = self.data_df.iloc[self.epoch_ids[index]]["Phonemes"]
        phonemes.append("<EOS>")
        phonemes.extend(["<PAD>" for _ in range(self.pad_length - len(phonemes))])
        tokenized = torch.Tensor([self.phoneme_to_id[phoneme] for phoneme in phonemes])
        return tokenized, tokenized.clone()

    def __len__(self) -> int:
        return len(self.epoch_ids)


def get_phoneme_testloader(batch_size: int, pad_to_length: int) -> DataLoader:
    r"""Return a dataloader containing the phoneme test data batched in size `batch_size`."""
    phoneme_set = PhonemeTestDataset(
        phoneme_to_id=get_phoneme_to_id(),
        pad_to_length=pad_to_length,
    )
    phoneme_loader = DataLoader(phoneme_set, batch_size)
    return phoneme_loader
