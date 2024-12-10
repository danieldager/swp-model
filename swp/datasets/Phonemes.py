import json
from itertools import chain
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..utils.legacy_utils import get_test_data, phoneme_statistics, sample_words

CUR_DIR = Path(__file__).resolve()
CACHE_DIR = CUR_DIR.parent / "cache"


class CustomDataset(Dataset):
    def __init__(self, phonemes):
        # Convert phoneme sequences to tensors
        self.data = [torch.tensor(seq) for seq in phonemes]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return inputs and targets
        return self.data[idx], self.data[idx].clone()


class Phonemes:
    def __init__(self) -> None:
        self.test_data = get_test_data()

        # Cache for train and validation phonemes
        train_cache = CACHE_DIR / "train_phonemes.json"
        valid_cache = CACHE_DIR / "valid_phonemes.json"
        stats_cache = CACHE_DIR / "phoneme_stats.json"
        bigram_cache = CACHE_DIR / "bigram_stats.json"

        # If cache dir exists, load phonemes and stats
        if CACHE_DIR.exists():
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
            CACHE_DIR.mkdir(exist_ok=True)
            self.train_phonemes, self.valid_phonemes = sample_words(self.test_data)
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
