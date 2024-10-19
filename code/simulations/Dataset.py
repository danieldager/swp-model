import numpy as np
from itertools import chain
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

from utils import sample_words, get_test_data

""" 
"phoneme tensors" are one-hot tensors of a list of phonemes for a single word
"grapheme tensors" are 1D image tensors of a 64x64 image of a single word
"""

class Dataset():
    def __init__(self, word_count: int, savepath=None):
        # Generate training data and fetch test data
        self.train_phonemes, self.valid_phonemes = sample_words(word_count)
        self.test_data, self.real_words, self.pseudo_words = get_test_data()
        
        # Path for saving images
        self.savepath = savepath

    def phoneme_dataloaders(self) -> tuple:
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

        # Get vocab size and max sequence length
        vocab_size = len(phone_to_index)
        max_length = max(
            max(len(word) for word in train_phonemes), 
            max(len(word) for word in valid_phonemes),
            max(len(word) for word in test_phonemes)
        ) + 1

        # Encode phonemes to indices
        train_encoded = [[phone_to_index[p] for p in w] for w in train_phonemes]
        valid_encoded = [[phone_to_index[p] for p in w] for w in valid_phonemes]
        test_encoded = [[phone_to_index[p] for p in w] for w in test_phonemes]
        train_inputs = torch.tensor(train_encoded)
        valid_inputs = torch.tensor(valid_encoded)
        test_inputs = torch.tensor(test_encoded)

        # Inputs are same as targets because of AE architecture
        train_dataset = TensorDataset(train_inputs, train_inputs.clone())
        valid_dataset = TensorDataset(valid_inputs, valid_inputs.clone())
        test_dataset = TensorDataset(test_inputs, test_inputs.clone())

        # Create dataloaders
        train_dataloader = DataLoader(train_dataset, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, shuffle=False)
        test_dataloader = DataLoader(test_dataset) 

        return (train_dataloader, valid_dataloader, test_dataloader,
                max_length, vocab_size, index_to_phone)
    
    # refactored from @author: aakash
    # NOTE: Missing image transformations and perturbations 
    def text_to_grapheme(
        self, words: list=["text"], savepath=None, index=1, 
        fontname='Arial', W = 64, H = 64, size=10, spacing=0,
        xshift=0, yshift=-3, upper=False, invert=False, mirror=False, show=None
    ) -> list:

        tensors = []
        for word in words:
            if upper: word = word.upper()
            if invert: word = word[::-1]
            
            img = Image.new("L", (W,H), color=10)
            fnt = ImageFont.truetype(fontname+'.ttf', size)
            draw = ImageDraw.Draw(img)

            # Starting word anchor
            w = sum([(fnt.getbbox(l)[2] - fnt.getbbox(l)[0]) for l in word])
            h = sum([(fnt.getbbox(l)[3] - fnt.getbbox(l)[1]) for l in word]) / len(word)
            w = w + spacing * (len(word) - 1)
            h_anchor = (W - w) / 2
            v_anchor = (H - h) / 2

            x, y = (xshift + h_anchor, yshift + v_anchor)
            
            # Draw the word letter by letter
            for l in word:
                draw.text((x,y), l, font=fnt, fill="white")
                letter_w = fnt.getbbox(l)[2] - fnt.getbbox(l)[0]
                x += letter_w + spacing

            if x > (W + spacing + 2) or (xshift + h_anchor) < -1:
                raise ValueError(f"Text width is bigger than image. Failed on size:{size}")
            
            if savepath:
                img.save(f"{savepath}/{word}.jpg")

            # Convert images to tensors
            img_np = np.array(img)
            img_tensor = torch.from_numpy(img_np)
            tensors.append(img_tensor)
        
        return tensors

    # NOTE: This function needs to be checked, missing image transformations
    def get_image_train_data(self):
        grapheme_tensors = self.text_to_grapheme(self.words, self.savepath)
        grapheme_dataset = TensorDataset(*grapheme_tensors)
        grapheme_dataloader = DataLoader(grapheme_dataset, batch_size=self.batch_size, 
                                         shuffle=True, drop_last=True)

        return grapheme_dataloader

    def get_image_valid_data(self):
        pass
