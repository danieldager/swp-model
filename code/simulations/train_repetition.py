import time, random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import seaborn as sns
import matplotlib.pyplot as plt
from Levenshtein import distance, editops

from DataGenerator import DataGenerator
from AudioEncoder import AudioEncoder
from Decoder import Decoder

from utils import seed_everything, levenshtein_bar_graph
from utils import remove_left_padding, print_examples

# Set random seed for reproducibility
seed_everything()

# For GPU-accelerated training on apple silicon
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print(torch.ones(1, device=mps_device))
else: print ("MPS device not found.")


def train_repetition(
        G: DataGenerator,
        word_count      = 50000,
        num_epochs      = 10, 
        batch_size      = 30, 
        hidden_size     = 64,
        dropout         = 0.1,
        num_layers      = 2,
        learning_rate   = 1e-4,
        grid_search     = 1,
        plot_train      = True,
        plot_test      = True,
    ):
    

    """ LOAD DATA """
    # Initialize data generator
    # G = DataGenerator(word_count, batch_size)

    # Get dataloaders, vocab size, sequence length, and index-phoneme map
    (train_dl, valid_dl, test_dl, seq_length, vocab_size, index_to_phoneme
    ) = G.phoneme_dataloaders()


    """ INITIALIZE MODEL """
    for model in range(grid_search):
        
        # Randomly sample hyperparameters
        if grid_search > 1:
            num_epochs      = random.choice([10, 15, 20])
            batch_size      = random.choice([30, 60, 90])
            hidden_size     = random.choice([1, 2, 4, 8, 16])
            dropout         = random.choice([0.0, 0.1, 0.2])
            num_layers      = random.choice([1, 2, 3])
            learning_rate   = random.choice([1e-3, 1e-4, 1e-5])

        # Create model name
        date = time.strftime("%d-%m_")
        params = f'{num_epochs}_{batch_size}_{hidden_size}'
        params += f'_{dropout}_{num_layers}_{learning_rate}'
        model_name = date + params

        # Initialize models, loss function, optimizer
        encoder = AudioEncoder(
            input_size=vocab_size, hidden_size=hidden_size, batch_size=batch_size,
            num_layers=num_layers, dropout=dropout
        )

        decoder = Decoder(
            hidden_size=hidden_size, output_size=vocab_size, batch_size=batch_size,
            num_layers=num_layers, dropout=dropout
        )

        criterion = nn.CrossEntropyLoss() # try focal loss for class imbalance
        parameters = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = optim.Adam(parameters, lr=learning_rate)


        """ TRAINING LOOP """
        train_losses = []
        valid_losses = []

        for epoch in range(num_epochs):
            encoder.train()
            decoder.train()
            train_loss = 0

            for inputs, targets in train_dl:

                # Zero gradients from previous step
                optimizer.zero_grad()

                # Encoder forward pass
                encoder_hidden = encoder(inputs)

                # Decoder forward pass
                decoder_input = torch.zeros(batch_size, seq_length, hidden_size)
                decoder_output = decoder(decoder_input, encoder_hidden)

                # Reshape for CrossEntropyLoss (batch_size * seq_length)
                outputs = decoder_output.view(-1, vocab_size)  
                targets = targets.view(-1)       
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                loss.backward()
                # print(any(param.grad is not None and torch.sum(param.grad != 0) > 0 for param in parameters))

                # print(encoder.embedding.weight[0, 0])    
                optimizer.step()
                # print(encoder.embedding.weight[0, 0])

                # print(torch.isnan(loss).any(), torch.isinf(loss).any())
                train_loss += loss.item()
        
            # Calculate average training loss
            train_loss /= len(train_dl)
            train_losses.append(train_loss)
 

            """ VALIDATION LOOP """
            encoder.eval()
            decoder.eval()
            valid_loss = 0

            with torch.no_grad():
                for inputs, targets in valid_dl:
                    encoder_hidden = encoder(inputs)
                    decoder_input = torch.zeros(inputs.size(0), seq_length, hidden_size).to(inputs.device)
                    decoder_output = decoder(decoder_input, encoder_hidden)

                    outputs = decoder_output.view(-1, vocab_size)
                    targets = targets.view(-1)
                    loss = criterion(outputs, targets)

                    valid_loss += loss.item()

            valid_loss /= len(valid_dl)
            valid_losses.append(valid_loss)

            print(f"Epoch {epoch+1}: Training Loss: {train_loss:.4f} Validation Loss: {valid_loss:.4f}")

        """ PRINTING OUTPUTS """
        # Print examples after the final epoch
        n = 5
        print(f"\n--- {n} Examples from Final Training Loop ---")
        print_examples(train_dl, encoder, decoder, index_to_phoneme, n=n)

        print(f"\n--- {n} Examples from Final Validation Loop ---")
        print_examples(valid_dl, encoder, decoder, index_to_phoneme, n=n)

        """ PLOTTING LOSS """
        if plot_train:
            plt.figure(figsize=(10, 6))
            sns.lineplot(x=range(1, num_epochs + 1), y=train_losses, label='Training Loss')
            sns.lineplot(x=range(1, num_epochs + 1), y=valid_losses, label='Validation Loss')
            plt.title(f'Training and Validation Loss Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Cross Entropy Loss')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"../../figures/{model_name}_loss.png")
            plt.show()


        """ TESTING LOOP """
        lev_distances = []
        model_phonemes = []
        avg_lev_distance = 0
        insertions, deletions, substitutions = [], [], []

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            for inputs, targets in test_dl:
                insertion, deletion, substitution = 0, 0, 0

                # Encoder forward pass
                encoder_hidden = encoder(inputs)

                # Initialize decoder input
                decoder_input = torch.zeros(1, seq_length, hidden_size)
                
                # Decoder forward pass
                decoder_output = decoder(decoder_input, encoder_hidden)

                # Argmax to get predicted phonemes, remove padding
                phonemes = torch.argmax(decoder_output, dim=-1)
                phonemes = remove_left_padding(phonemes.tolist()[0])

                # Remove left padding from targets
                targets = remove_left_padding(targets.tolist()[0])

                # # Levenshtein distance with targets
                # lev_distance = distance(phonemes, targets)
                # lev_distances.append(lev_distance)
                # avg_lev_distance += lev_distance

                # Calculate Levenshtein operations
                ops = editops(phonemes, targets)
                for op, _, _ in ops:
                    if op == 'insert': insertion += 1
                    elif op == 'delete': deletion += 1
                    elif op == 'replace': substitution += 1
            
                lev_distances.append(len(ops))
                insertions.append(insertion)
                deletions.append(deletion)
                substitutions.append(substitution)
                
                # Convert indices to phonemes
                phonemes = [index_to_phoneme[i] for i in phonemes]

                model_phonemes.append(phonemes)

            print(f"Average Levenshtein distance: {avg_lev_distance/len(test_dl)}")

        G.test_data['Model_Phonemes'] = model_phonemes
        G.test_data['Levenshtein'] = lev_distances
        G.test_data['Insertions'] = insertions
        G.test_data['Deletions'] = deletions
        G.test_data['Substitutions'] = substitutions

    # Plot levenshtein distances, insertions, deletions, substitutions
    if plot_test: 
        levenshtein_bar_graph(G.test_data, model_name)
        G.test_data.drop(columns=['Category'])

    return G.test_data


if __name__ == "__main__":
    data = train_repetition()