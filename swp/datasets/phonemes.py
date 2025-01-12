import json
from itertools import chain
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..utils.datasets import (
    get_epoch_numpy,
    get_test_data,
    get_training_fold,
    get_val_fold,
    phoneme_statistics,
    sample_words,
)
from ..utils.paths import get_dataset_dir


class CustomDataset(Dataset):
    def __init__(self, phonemes):
        # Convert phoneme sequences to tensors
        self.data = [torch.tensor(seq) for seq in phonemes]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return inputs and targets
        return self.data[idx], self.data[idx].clone()


class PhonemeDataset(Dataset):
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
        pad_to_length: int,
    ):
        self.fold_id = fold_id
        self.train = train
        if self.train:
            self.data_df = get_training_fold(self.fold_id)
            self.epoch_ids = get_epoch_numpy(self.fold_id)
        else:
            self.data_df = get_val_fold(self.fold_id)
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


class Phonemes:
    def __init__(self) -> None:
        self.test_data = get_test_data()

        phonemes_dir = get_dataset_dir() / "phonemes"
        # Cache for train and validation phonemes
        # TODO add checks, better nouns, better format
        train_cache = phonemes_dir / "train_phonemes.json"
        valid_cache = phonemes_dir / "valid_phonemes.json"
        stats_cache = phonemes_dir / "phoneme_stats.json"
        bigram_cache = phonemes_dir / "bigram_stats.json"

        # If cache dir exists, load phonemes and stats
        if phonemes_dir.exists():
            with train_cache.open("r") as f:
                self.train_phonemes = json.load(f)
            with valid_cache.open("r") as f:
                self.valid_phonemes = json.load(f)
            with stats_cache.open("r") as f:
                self.phoneme_stats = json.load(f)
            with bigram_cache.open("r") as f:
                self.bigram_stats = json.load(f)

        # Otherwise, generate and save phonemes to cache
        else:
            phonemes_dir.mkdir(parents=True, exist_ok=True)
            self.train_phonemes, self.valid_phonemes = sample_words()
            self.phoneme_stats, self.bigram_stats = phoneme_statistics(
                self.train_phonemes
            )
            with train_cache.open("w") as f:
                json.dump(self.train_phonemes, f)
            with valid_cache.open("w") as f:
                json.dump(self.valid_phonemes, f)
            with stats_cache.open("w") as f:
                json.dump(self.phoneme_stats, f)
            with bigram_cache.open("w") as f:
                json.dump(self.bigram_stats, f)

        # Add stop token to phoneme sequences
        train_phonemes = [seq + ["<STOP>"] for seq in self.train_phonemes]
        valid_phonemes = [seq + ["<STOP>"] for seq in self.valid_phonemes]
        test_phonemes = [seq + ["<STOP>"] for seq in self.test_data["Phonemes"]]

        # Flatten and deduplicate lists of phonemes
        all_phonemes = list(
            set(chain(*train_phonemes, *valid_phonemes, *test_phonemes))
        )

        # Create phoneme to index map
        phone_to_index = {p: i + 1 for i, p in enumerate(all_phonemes)}

        # # Add start token to beginning of index map
        phone_to_index["<SOS>"] = 0

        # Create index to phoneme map
        index_to_phone = {i: p for p, i in phone_to_index.items()}

        # Get vocab size
        vocab_size = len(phone_to_index)

        # Encode phonemes to indices
        train_encoded = [[phone_to_index[p] for p in w] for w in train_phonemes]
        valid_encoded = [[phone_to_index[p] for p in w] for w in valid_phonemes]
        test_encoded = [[phone_to_index[p] for p in w] for w in test_phonemes]

        # Inputs are same as targets because of AE architecture
        train_dataset = CustomDataset(train_encoded)
        valid_dataset = CustomDataset(valid_encoded)
        test_dataset = CustomDataset(test_encoded)

        # Create dataloaders
        self.train_dataloader = DataLoader(train_dataset, shuffle=True)
        self.valid_dataloader = DataLoader(valid_dataset, shuffle=False)
        self.test_dataloader = DataLoader(test_dataset)

        # Save attributes
        self.vocab_size = vocab_size
        self.phone_to_index = phone_to_index
        self.index_to_phone = index_to_phone
