

import numpy as np
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

from utils import timeit, sample_words, get_evaluation_data

""" 
"phoneme tensors" are one-hot tensors of a list of phonemes for a single word
"grapheme tensors" are 1D image tensors of a 64x64 image of a single word
"""

class DataGenerator():
    def __init__(self, word_count=500, batch_size=10, savepath=None):
        self.savepath, self.batch_size = savepath, batch_size

        # Generate training data and fetch evaluation data
        self.train_words, self.train_phonemes = sample_words(word_count)
        self.eval_data, self.real_words, self.pseudo_words = get_evaluation_data()

    @timeit
    def get_phoneme_dataloaders(self):
        # Add stop token to phoneme sequences
        self.train_phonemes = [word + ["<STOP>"] for word in self.train_phonemes]

        # Collect all phonemes and count their occurrences
        phoneme_counter = Counter(p for word in self.train_phonemes for p in word)
        phoneme_counter.update(p for word in self.eval_data['Phonemes'] for p in word)

        # Create phoneme to index map, sorting by frequency
        sorted_phonemes = sorted(phoneme_counter, key=phoneme_counter.get, reverse=True)
        phoneme_to_index = {p: i+1 for i, p in enumerate(sorted_phonemes)}
        
        # Add padding token to beginning of index map
        phoneme_to_index["<PAD>"] = 0

        # Create index to phoneme map
        index_to_phoneme = {i: p for p, i in phoneme_to_index.items()}

        # Get vocab size and max sequence length
        vocab_size = len(phoneme_to_index)
        max_length = max(
            max(len(word) for word in self.train_phonemes), 
            max(len(word) for word in self.eval_data['Phonemes'])
        ) + 1

        # NOTE: certain phonemes in eval data may not be in train data
        # Encode phonemes to indices, add stop token at the end of each sequence
        train_encoded = [[phoneme_to_index[p] for p in w] for w in self.train_phonemes]
        eval_encoded = [[phoneme_to_index[p] for p in w] for w in self.eval_data['Phonemes']]

        # Left pad sequences to the same length
        train_padded = [[0] * (max_length - len(seq)) + seq for seq in train_encoded]
        eval_padded = [[0] * (max_length - len(seq)) + seq for seq in eval_encoded]

        # NOTE: an embedding layer removes the need for one-hot vectors
        # phoneme_tensors = torch.nn.functional.one_hot(padded_sequences, num_classes=vocab_size)

        # Inputs are same as targets because of AE architecture
        train_inputs = torch.tensor(train_padded)
        eval_inputs = torch.tensor(eval_padded)
        train_targets = train_inputs.clone()
        eval_targets = eval_inputs.clone()

        # Create dataloaders
        train_dataset = TensorDataset(train_inputs, train_targets)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                        shuffle=True, drop_last=True)
        
        eval_dataset = TensorDataset(eval_inputs, eval_targets)
        eval_dataloader = DataLoader(eval_dataset, batch_size=1)

        return train_dataloader, eval_dataloader, max_length, vocab_size, index_to_phoneme
    
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

    def get_image_eval_data(self):
        pass

if __name__ == "__main__":
    gen = DataGenerator(word_count=50, batch_size=10)
    phoneme_dataloader = gen.generate_phonemes()