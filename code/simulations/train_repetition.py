import sys, time, random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

import seaborn as sns
import matplotlib.pyplot as plt
from Levenshtein import editops

""" PATHS """
CUR_DIR = Path(__file__)
MODELS_DIR = CUR_DIR.parent.parent / "models"
WEIGHTS_DIR = MODELS_DIR / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)
sys.path.append(str(MODELS_DIR))

from DataGen import DataGen
from EncoderRNN import EncoderRNN
from DecoderRNN import DecoderRNN

from utils import seed_everything, levenshtein_bar_graph

# Set random seed for reproducibility
seed_everything()

""" DEVICE """
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

""" TIMER """
class Timer:
    def __init__(self):
        self.times = defaultdict(float)
        self.counts = defaultdict(int)
        
    def start(self):
        self._start_time = time.time()
        
    def stop(self, name):
        elapsed = time.time() - self._start_time
        self.times[name] += elapsed
        self.counts[name] += 1
        
    def summary(self):
        print("\nTiming Summary:")
        print("-" * 60)
        print(f"{'Operation':<30} {'Total Time (s)':<15} {'Avg Time (s)':<15}")
        print("-" * 60)
        for name in self.times:
            total = self.times[name]
            avg = total / self.counts[name]
            print(f"{name:<30} {total:>13.3f}s {avg:>13.3f}s")


""" TRAIN AND TEST PHONEME MODELS """
def train_repetition(
        D: DataGen,
        num_epochs      = 3, 
        hidden_size     = 4,
        num_layers      = 1,
        dropout         = 0.0,
        learning_rate   = 5e-3,
        grid_search     = 1,
        plot_train      = True,
        plot_test       = True,
    ):
    
    timer = Timer()
    
    """ LOAD DATA """
    train_dl, valid_dl, test_dl, vocab_size, index_to_phoneme = D.dataloaders()

    """ INITIALIZE MODEL """
    for n in range(grid_search):
        epoch_times = []
        
        # Randomly sample hyperparameters
        if grid_search > 1:
            num_epochs      = random.choice([5, 10])
            hidden_size     = random.choice([1, 2, 4, 8])
            num_layers     = random.choice([1, 2])
            dropout        = random.choice([0.0, 0.1, 0.2])
            learning_rate  = random.choice([1e-1, 5e-2, 1e-2, 5e-3, 1e-1])

            print(f"Training model {n+1}/{grid_search} with hyperparameters:")

        # Create model name
        model = f'{num_epochs}_{hidden_size}_{num_layers}_{dropout}_{learning_rate}'
        print(model)

        encoder = EncoderRNN(
            input_size=vocab_size, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout
        ).to(device)

        decoder = DecoderRNN(
            hidden_size=hidden_size, output_size=vocab_size,
            num_layers=num_layers, dropout=dropout
        ).to(device)

        # if CUDA, use DataParallel
        if torch.cuda.device_count() > 1:
            encoder = nn.DataParallel(encoder).to(device)
            decoder = nn.DataParallel(decoder).to(device)

        criterion = nn.CrossEntropyLoss()
        parameters = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = optim.Adam(parameters, lr=learning_rate)

        """ TRAINING LOOP """
        train_losses = []
        valid_losses = []

        for epoch in range(num_epochs):
            epoch_start = time.time()
            encoder.train()
            decoder.train()
            train_loss = 0

            timer.start()
            for inputs, targets in train_dl:
                inputs = inputs.to(device)
                targets = targets.to(device)

                # check all details about inputs
                print(inputs.shape)
                print(inputs.device)
                print(inputs)

                optimizer.zero_grad()

                # Forward passes
                timer.start()
                encoder_hidden = encoder(inputs)
                print(encoder_hidden.shape)
                print(encoder_hidden.device)

                timer.stop("Encoder Forward Pass")

                timer.start()
                decoder_input = torch.zeros(1, inputs.shape[1], hidden_size, device=device)

                print(decoder_input.shape)
                print(decoder_input.device)

                decoder_output = decoder(decoder_input, encoder_hidden)
                timer.stop("Decoder Forward Pass")

                # Loss computation and backward pass
                timer.start()
                outputs = decoder_output.view(-1, vocab_size)  
                targets = targets.view(-1)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                timer.stop("Backward Pass")

                train_loss += loss.item()
        
            train_loss /= len(train_dl)
            train_losses.append(train_loss)
 
            """ VALIDATION LOOP """
            timer.start()
            encoder.eval()
            decoder.eval()
            valid_loss = 0

            with torch.no_grad():
                for inputs, targets in valid_dl:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    encoder_hidden = encoder(inputs)
                    decoder_input = torch.zeros(1, inputs.shape[1], hidden_size, device=device)
                    decoder_output = decoder(decoder_input, encoder_hidden)

                    outputs = decoder_output.view(-1, vocab_size)
                    targets = targets.view(-1)
                    loss = criterion(outputs, targets)
                    valid_loss += loss.item()

            valid_loss /= len(valid_dl)
            valid_losses.append(valid_loss)
            timer.stop("Validation")

            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            print(f"Epoch {epoch+1}: T Loss: {train_loss:.4f} V Loss: {valid_loss:.4f} Time: {epoch_time:.2f}s")

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
            
            figures_dir = CUR_DIR.parent.parent.parent / "figures"
            figures_dir.mkdir(exist_ok=True)
            plt.savefig(figures_dir / f"{model}_loss.png")
            plt.show()

        """ SAVE MODEL """
        torch.save(encoder.state_dict(), WEIGHTS_DIR / f"encoder_{model}.pth")
        torch.save(decoder.state_dict(), WEIGHTS_DIR / f"decoder_{model}.pth")

        """ TESTING LOOP """
        timer.start()
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

                encoder_hidden = encoder(inputs)
                decoder_input = torch.zeros(1, inputs.shape[1], hidden_size, device=device)
                decoder_output = decoder(decoder_input, encoder_hidden)

                prediction = torch.argmax(decoder_output, dim=-1)
                prediction = prediction.squeeze().cpu().tolist()
                targets = targets.squeeze().cpu().tolist()

                ops = editops(prediction, targets)
                for op, _, _ in ops:
                    if op == 'insert': insertion += 1
                    elif op == 'delete': deletion += 1
                    elif op == 'replace': substitution += 1
            
                deletions.append(deletion)
                insertions.append(insertion)
                substitutions.append(substitution)
                edit_distance.append(len(ops))
                predictions.append([index_to_phoneme[i] for i in prediction][:-1])

        D.test_data['Prediction'] = predictions
        D.test_data['Deletions'] = deletions
        D.test_data['Insertions'] = insertions
        D.test_data['Substitutions'] = substitutions
        D.test_data['Edit Distance'] = edit_distance
        timer.stop("Testing")

        if plot_test:
            levenshtein_bar_graph(D.test_data, model)
            D.test_data.drop(columns=['Category'])

    # Print timing summary
    timer.summary()

    return D.test_data

if __name__ == "__main__":
    D = DataGen()
    data = train_repetition(D, grid_search=10)