import numpy as np
from collections import defaultdict

from g2p_en import G2p
from PIL import Image, ImageDraw, ImageFont
from wordfreq import iter_wordlist, word_frequency

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

""" Notes on naming:

"phoneme tensors" are one-hot tensors of a list of phonemes for a single word
"grapheme tensors" are 1D image tensors of a 64x64 image of a single word

"""

""" Sample words based on frequency """

class WordSampler:
    def __init__(self, word_count, language='en', max_words=100000):
        self.words = []
        self.word_count = word_count
        self.language = language
        self.cumulative_probs = []
        
        total_freq = 0
        for i, word in enumerate(iter_wordlist(language)):
            if i >= max_words:
                break
            freq = word_frequency(word, language)
            total_freq += freq
            self.words.append(word)
            self.cumulative_probs.append(total_freq)
        
        self.cumulative_probs = np.array(self.cumulative_probs) / total_freq

    def sample(self):
        random_values = np.random.random(self.word_count)
        indices = np.searchsorted(self.cumulative_probs, random_values)        
        return [self.words[i] for i in indices]
    

def text_to_phoneme(words: list, g2p: G2p):

    # Get list of phonemes for each word
    phonemes = [g2p(word) for word in words]

    # Create a dictionary to map phonemes to indices
    phoneme_to_index = defaultdict(lambda: len(phoneme_to_index) + 1)
    encoded_phonemes = [[phoneme_to_index[p] for p in phoneme] for phoneme in phonemes]

    # Pad sequences to the same length
    encoded_phonemes = [torch.tensor(lst) for lst in encoded_phonemes]
    padded_sequences = pad_sequence(encoded_phonemes, batch_first=True, padding_value=0)

    # Create one-hot encodings for each phoneme
    vocab_size = len(phoneme_to_index) + 1
    phoneme_tensors = torch.nn.functional.one_hot(padded_sequences, num_classes=vocab_size).float()

    return phoneme_tensors

def text_to_grapheme(
        words: list=["text"], savepath=None, index=1, mirror=False,
        fontname='Arial', W = 64, H = 64, size=10, spacing=0,
        xshift=0, yshift=-3, upper=False, invert=False, show=None
    ):

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

class DataGenerator():
    def __init__(self, word_count=500, batch_size=10, savepath=None):
        self.word_count = word_count
        self.batch_size = batch_size
        self.savepath = savepath
        
        self.sampler = WordSampler(word_count)
        self.words = self.sampler.sample()
        self.g2p = G2p()

    def text_to_phoneme(self, words: list, g2p: G2p):

        # Get list of phonemes for each word
        phonemes = [g2p(word) for word in words]

        # Create a dictionary to map phonemes to indices
        phoneme_to_index = defaultdict(lambda: len(phoneme_to_index) + 1)
        encoded_phonemes = [[phoneme_to_index[p] for p in phoneme] for phoneme in phonemes]

        # Pad sequences to the same length
        encoded_phonemes = [torch.tensor(lst) for lst in encoded_phonemes]
        padded_sequences = pad_sequence(encoded_phonemes, batch_first=True, padding_value=0)

        # Create one-hot encodings for each phoneme
        vocab_size = len(phoneme_to_index) + 1
        phoneme_tensors = torch.nn.functional.one_hot(padded_sequences, num_classes=vocab_size)

        return phoneme_tensors

    def text_to_grapheme(
        self, words: list=["text"], savepath=None, index=1, 
        fontname='Arial', W = 64, H = 64, size=10, spacing=0,
        xshift=0, yshift=-3, upper=False, invert=False, mirror=False, show=None
    ):

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

    def generate_phonemes(self):
        phoneme_tensors = self.text_to_phoneme(self.words, self.g2p)

        print(phoneme_tensors.shape)
        seq_length = phoneme_tensors.shape[1]
        vocab_size = phoneme_tensors.shape[2]

        inputs = phoneme_tensors
        targets = phoneme_tensors.clone()

        phoneme_dataset = TensorDataset(inputs, targets)
        phoneme_dataloader = DataLoader(phoneme_dataset, batch_size=self.batch_size, 
                                        shuffle=True, drop_last=True)

        return phoneme_dataloader, seq_length, vocab_size

    def generate_graphemes(self):
        grapheme_tensors = self.text_to_grapheme(self.words, self.savepath)
        grapheme_dataset = TensorDataset(*grapheme_tensors)
        grapheme_dataloader = DataLoader(grapheme_dataset, batch_size=self.batch_size, 
                                         shuffle=True, drop_last=True)

        return grapheme_dataloader


if __name__ == "__main__":
    gen = DataGenerator(word_count=50, batch_size=10)
    phoneme_dataloader = gen.generate_phonemes()