import os, sys, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

import seaborn as sns
import matplotlib.pyplot as plt
from Levenshtein import editops

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / "models"))

from DataGen import DataGen
from EncoderRNN import EncoderRNN
from DecoderRNN import DecoderRNN

from utils import seed_everything, levenshtein_bar_graph

# Set random seed for reproducibility
seed_everything()

""" SET DEVICE """
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    print(f"Using CUDA device: {device_name}")

elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")

else:
    device = torch.device("cpu")
    print("Using CPU device")

""" TRAIN AND TEST PHONEME MODELS """
def train_repetition(
        D: DataGen,
        num_epochs      = 3, 
        hidden_size     = 8,
        num_layers      = 1,
        dropout         = 0.0,
        learning_rate   = 1e-3,
        grid_search     = 1,
        plot_train      = True,
        plot_test       = True,
    ):
    
    """ LOAD DATA """
    # Initialize data generator
    # D = DataGen()

    # Get dataloaders, vocab size, and index-phoneme map
    train_dl, valid_dl, test_dl, vocab_size, index_to_phoneme = D.dataloaders()

    """ INITIALIZE MODEL """
    for n in range(grid_search):
        
        # Randomly sample hyperparameters
        if grid_search > 1:
            num_epochs      = random.choice([5, 10])
            hidden_size     = random.choice([1, 2, 4, 8])
            num_layers      = random.choice([1, 2])
            dropout         = random.choice([0.0, 0.1, 0.2])
            learning_rate   = random.choice([1e-1, 5e-2, 1e-2, 5e-3, 1e-1])
            # early_stopping

            print(f"Training model {n+1}/{grid_search} with hyperparameters:")

        # Create model name
        model = f'{num_epochs}_{hidden_size}_{num_layers}_{dropout}_{learning_rate}'
        print(model)

        # Initialize models, loss function, optimizer
        encoder = EncoderRNN(
            input_size=vocab_size, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout
        ).to(device)

        decoder = DecoderRNN(
            hidden_size=hidden_size, output_size=vocab_size,
            num_layers=num_layers, dropout=dropout
        ).to(device)

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
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Zero gradients from previous step
                optimizer.zero_grad()

                # Encoder forward pass
                encoder_hidden = encoder(inputs)

                # Decoder forward pass
                # NOTE: Include start token
                decoder_input = torch.zeros(1, inputs.shape[1], hidden_size, device=device)
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
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    # Forward pass
                    encoder_hidden = encoder(inputs)
                    decoder_input = torch.zeros(1, inputs.shape[1], hidden_size, device=device)
                    decoder_output = decoder(decoder_input, encoder_hidden)

                    # Calculate loss
                    outputs = decoder_output.view(-1, vocab_size)
                    targets = targets.view(-1)
                    loss = criterion(outputs, targets)
                    valid_loss += loss.item()

            valid_loss /= len(valid_dl)
            valid_losses.append(valid_loss)

            print(f"Epoch {epoch+1}: T Loss: {train_loss:.4f} V Loss: {valid_loss:.4f}")

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
            
            # Create figures directory if it doesn't exist
            figures_dir = PROJECT_ROOT / "figures"
            figures_dir.mkdir(exist_ok=True)
            plt.savefig(figures_dir / f"{model}_loss.png")
            plt.show()

        """ SAVE MODEL """
        # Create weights directory if it doesn't exist
        weights_dir = Path("weights")
        weights_dir.mkdir(exist_ok=True)
        
        # Save encoder and decoder weights
        torch.save(encoder.state_dict(), weights_dir / f"encoder_{model}.pth")
        torch.save(decoder.state_dict(), weights_dir / f"decoder_{model}.pth")

        """ TESTING LOOP """
        predictions = []
        deletions = []
        insertions = []
        substitutions = []
        edit_distance = []

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            for inputs, targets in test_dl:
                inputs = inputs.to(device)
                targets = targets.to(device)
                insertion, deletion, substitution = 0, 0, 0

                # Foward pass
                encoder_hidden = encoder(inputs)
                decoder_input = torch.zeros(1, inputs.shape[1], hidden_size, device=device)
                decoder_output = decoder(decoder_input, encoder_hidden)

                # Argmax to get predicted phonemes
                prediction = torch.argmax(decoder_output, dim=-1)
                prediction = prediction.squeeze().cpu().tolist()
                targets = targets.squeeze().cpu().tolist()

                # Tabulate errors
                ops = editops(prediction, targets)
                for op, _, _ in ops:
                    if op == 'insert': insertion += 1
                    elif op == 'delete': deletion += 1
                    elif op == 'replace': substitution += 1
            
                deletions.append(deletion)
                insertions.append(insertion)
                substitutions.append(substitution)
                edit_distance.append(len(ops))
                
                # Convert predicted indices to phonemes
                predictions.append([index_to_phoneme[i] for i in prediction][:-1])

        # Update test dataframe with predictions and errors
        D.test_data['Prediction'] = predictions
        D.test_data['Deletions'] = deletions
        D.test_data['Insertions'] = insertions
        D.test_data['Substitutions'] = substitutions
        D.test_data['Edit Distance'] = edit_distance

    # Plot levenshtein distances, insertions, deletions, substitutions
    if plot_test: 
        levenshtein_bar_graph(D.test_data, model)
        D.test_data.drop(columns=['Category'])

    return D.test_data

if __name__ == "__main__":
    D = DataGen()
    data = train_repetition(D, grid_search=20)