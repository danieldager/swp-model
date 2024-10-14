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

# Set random seed for reproducibility
seed_everything()

# For GPU-accelerated training on apple silicon
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print(torch.ones(1, device=mps_device))
else: print ("MPS device not found.")


def train_repetition(
        word_count      = 50000,
        num_epochs      = 10, 
        batch_size      = 30, 
        hidden_size     = 64,
        dropout         = 0.1,
        num_layers      = 2,
        learning_rate   = 1e-4,
        grid_search     = 1,
        plot_train      = True,
        plot_eval       = True,
    ):
    

    """ LOAD DATA """
    # Initialize data generator
    gen = DataGenerator(word_count, batch_size)

    # Get dataloaders, vocab size, sequence length, and index-phoneme map
    (train_dl, eval_dl, seq_length, 
     vocab_size, index_to_phoneme) = gen.get_phoneme_dataloaders()


    """ INITIALIZE MODEL """
    for model in range(grid_search):
        
        # Randomly sample hyperparameters
        if grid_search > 1:
            num_epochs      = random.choice([10, 15, 20])
            batch_size      = random.choice([30, 60, 90])
            hidden_size     = random.choice([64, 128, 256])
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
        epoch_losses = []
        for epoch in range(num_epochs):
            encoder.train()
            decoder.train()
            total_loss = 0
            batch_losses = []

            for inputs, targets in train_dl:

                # Zero gradients from previous step
                optimizer.zero_grad()

                # Encoder forward pass
                encoder_hidden = encoder(inputs)

                decoder_input = torch.zeros(batch_size, seq_length, hidden_size)
                decoder_output = decoder(decoder_input, encoder_hidden)

                # Reshape outputs and targets for CrossEntropyLoss
                outputs = decoder_output.view(-1, vocab_size)  # Reshape to (batch_size * seq_length, vocab_size)
                targets = targets.view(-1)                     # Reshape to (batch_size * seq_length)
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                loss.backward()
                # print(any(param.grad is not None and torch.sum(param.grad != 0) > 0 for param in parameters))

                # print(encoder.embedding.weight[0, 0])    
                optimizer.step()
                # print(encoder.embedding.weight[0, 0])

                # print(torch.isnan(loss).any(), torch.isinf(loss).any())
                total_loss += loss.item()
                batch_losses.append(loss.item())
        
            # Print epoch loss
            avg_loss = total_loss / len(train_dl)
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch+1}: {avg_loss}")

        # Plot training loss
        if plot_train:
            plt.figure(figsize=(10, 6))
            sns.lineplot(x=range(1, num_epochs + 1), y=epoch_losses)
            plt.title(f'Model {model+1} Training Loss Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Cross Entropy Loss')
            plt.tight_layout()
            plt.savefig(f"../../figures/{model_name}_training_loss.png")
            plt.show()


        """ EVALUATION """
        # Evaluation loop
        lev_distances = []
        model_phonemes = []
        avg_lev_distance = 0
        insertions, deletions, substitutions = [], [], []

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            for inputs, targets in eval_dl:
                insertion, deletion, substitution = 0, 0, 0

                # Encoder forward pass
                encoder_hidden = encoder(inputs)

                # Initialize decoder input
                decoder_input = torch.zeros(1, seq_length, hidden_size)
                
                # Decoder forward pass
                decoder_output = decoder(decoder_input, encoder_hidden)

                # Argmax to get predicted phonemes
                phonemes = torch.argmax(decoder_output, dim=-1)
                phonemes = phonemes.tolist()[0]

                # Remove left padding from phonemes
                # NOTE: only remove padding at beginning
                for i in range(seq_length):
                    if phonemes[i] != 0:
                        phonemes = phonemes[i:]
                        break

                # Remove left padding from targets
                targets = [p for p in targets.tolist()[0] if p != 0]

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

            print(f"Average Levenshtein distance: {avg_lev_distance/len(eval_dl)}")

    gen.eval_data['Model_Phonemes'] = model_phonemes
    gen.eval_data['Levenshtein'] = lev_distances
    gen.eval_data['Insertions'] = insertions
    gen.eval_data['Deletions'] = deletions
    gen.eval_data['Substitutions'] = substitutions

    # Plot levenshtein distances, insertions, deletions, substitutions
    if plot_eval: levenshtein_bar_graph(gen.eval_data, model_name)

    return gen.eval_data


if __name__ == "__main__":
    eval_data = train_repetition()