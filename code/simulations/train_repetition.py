import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from DataGenerator import DataGenerator
from AudioEncoder import AudioEncoder
from Decoder import Decoder

from utils import seed_everything, timeit
from Levenshtein import distance

# Set random seed for reproducibility
seed_everything()

# For GPU-accelerated training on apple silicon
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print(torch.ones(1, device=mps_device))
else: print ("MPS device not found.")

def train_repetition(
        word_count      = 20000,
        num_epochs      = 10, 
        batch_size      = 30, 
        hidden_size     = 256,
        dropout         = 0.0,
        num_layers      = 1,
        learning_rate   = 1e-3,
        grid_search     = 1,
        plot_train      = False,
        plot_eval       = False,
    ):
    
    """ LOAD DATA """
    # Initialize data generator
    gen = DataGenerator(word_count, batch_size)

    # Get dataloaders, vocab size, sequence length, and index-phoneme map
    (train_dl, eval_dl, seq_length, 
     vocab_size, index_to_phoneme) = gen.get_phoneme_dataloaders()

    """ INITIALIZE MODEL """
    for _ in range(grid_search):
        
        # Randomly sample hyperparameters
        if grid_search > 1:
            num_epochs      = random.choice([10, 20, 30])
            batch_size      = random.choice([30, 60, 90])
            hidden_size     = random.choice([128, 256, 512])
            dropout         = random.choice([0.0, 0.1, 0.2])
            num_layers      = random.choice([1, 2, 3])
            learning_rate   = random.choice([1e-3, 1e-4, 1e-5])

        # Initialize models, loss function, optimizer
        encoder = AudioEncoder(
            input_size=vocab_size, hidden_size=hidden_size, batch_size=batch_size,
            num_layers=num_layers, dropout=dropout
        )

        decoder = Decoder(
            hidden_size=hidden_size, output_size=vocab_size, batch_size=batch_size,
            num_layers=num_layers, dropout=dropout
        )

        loss_fn = nn.CrossEntropyLoss() # try focal loss for class imbalance
        parameters = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = optim.Adam(parameters, lr=learning_rate)


        """ TRAINING LOOP """
        for epoch in range(num_epochs):
            for inputs, targets in train_dl:
                # Zero gradients from previous step
                optimizer.zero_grad()

                # Encoder forward pass
                # NOTE: the 2nd dimension is seq_length and not batch_size?
                encoder_hidden = encoder(inputs)

                # Initialize decoder input
                decoder_input = torch.zeros(batch_size, seq_length, hidden_size)
                
                # Decoder forward pass
                decoder_output = decoder(decoder_input, encoder_hidden)

                # One hot encode targets
                targets = F.one_hot(targets, num_classes=vocab_size).float()

                # Compute loss and backpropagate
                loss = loss_fn(decoder_output, targets)
                loss.backward()
                optimizer.step()
        
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()/targets.size(1)}")


        """ EVALUATION """
        # Evaluation loop
        lev_distances = []
        model_phonemes = []
        avg_lev_distance = 0

        for inputs, targets in eval_dl:

            # Encoder forward pass
            encoder_hidden = encoder(inputs)

            # Initialize decoder input
            decoder_input = torch.zeros(1, seq_length, hidden_size)
            
            # Decoder forward pass
            decoder_output = decoder(decoder_input, encoder_hidden)

            # Argmax to get predicted phonemes
            phonemes = torch.argmax(decoder_output, dim=-1)
            phonemes = phonemes.tolist()[0]

            # Levenshtein distance with targets
            lev_distance = distance(phonemes, targets.tolist()[0])
            lev_distances.append(lev_distance)
            avg_lev_distance += lev_distance

            # Convert indices to phonemes
            phonemes = [index_to_phoneme[i] for i in phonemes]

            # Remove left padding
            # NOTE: only remove padding at beginning
            phonemes = [p for p in phonemes if p != '<PAD>']

            model_phonemes.append(phonemes)

        print(f"Average Levenshtein distance: {avg_lev_distance/len(eval_dl)}")

    gen.eval_data['Model_Phonemes'] = model_phonemes
    gen.eval_data['Levenshtein'] = lev_distances

    return gen.eval_data
        



