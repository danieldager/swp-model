import json
from pathlib import Path
from itertools import chain

import torch
from torch.utils.data import Dataset, DataLoader

from utils import sample_words, get_test_data

""" 
"phoneme tensors" are one-hot tensors of a list of phonemes for a single word
"grapheme tensors" are 1D image tensors of a 64x64 image of a single word
"""

CUR_DIR = Path(__file__).resolve()
CACHE_DIR = CUR_DIR.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

class CustomDataset(Dataset):
    def __init__(self, phonemes):
        # Convert phoneme sequences to tensors
        self.data = [torch.tensor(seq) for seq in phonemes]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return inputs and targets
        return self.data[idx], self.data[idx].clone()

class Phonemes():
    def __init__(self, word_count: int = 50000, savepath=None):
        # Convert savepath to Path object if provided
        self.savepath = Path(savepath) if savepath else None

        # Get test phonemes
        self.test_data, self.real_words = get_test_data()

        # Cache for train and validation phonemes
        train_cache = CACHE_DIR / "train_phonemes.json"
        valid_cache = CACHE_DIR / "valid_phonemes.json"

        # Load phonemes from cache if available
        if train_cache.exists() and valid_cache.exists():
            with train_cache.open('r') as f: self.train_phonemes = json.load(f)
            with valid_cache.open('r') as f: self.valid_phonemes = json.load(f)

        # Otherwise, generate and save phonemes to cache
        else:
            self.train_phonemes, self.valid_phonemes = sample_words(word_count, self.real_words)
            with train_cache.open('w') as f: json.dump(self.train_phonemes, f)
            with valid_cache.open('w') as f: json.dump(self.valid_phonemes, f)

        # Add stop token to phoneme sequences
        train_phonemes = [seq + ["<STOP>"] for seq in self.train_phonemes]
        valid_phonemes = [seq + ["<STOP>"] for seq in self.valid_phonemes]
        test_phonemes = [seq + ["<STOP>"] for seq in self.test_data['Phonemes']]

        # NOTE: Deduplicate the train phonemes !!

        # Flatten and deduplicate lists of phonemes
        all_phonemes = list(set(chain(*train_phonemes, *valid_phonemes, *test_phonemes)))

        # Create phoneme to index map
        phone_to_index = {p: i+1 for i, p in enumerate(all_phonemes)}

        # Add padding token to beginning of index map
        phone_to_index["<PAD>"] = 0

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
        self.index_to_phone = index_to_phone
