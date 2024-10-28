import sys, time
import pandas as pd
import random, argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

import seaborn as sns
import matplotlib.pyplot as plt
from Levenshtein import editops

""" PATHS """
FILE_DIR = Path(__file__).resolve()
MODELS_DIR = FILE_DIR.parent.parent / "models"
WEIGHTS_DIR = MODELS_DIR / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)
sys.path.append(str(MODELS_DIR))

from DataGen import DataGen
from EncoderRNN import EncoderRNN
from DecoderRNN import DecoderRNN

from utils import seed_everything, set_device, Timer
from plots import training_curves

seed_everything()
device = set_device()

""" HYPERPARAMETERS """
def get_random_parameters():
    """Generate random parameters within specified ranges."""
    return {
        'n_epochs' : random.choice([5, 10]),
        'h_size'   : random.choice([1, 2, 4, 8]),
        'n_layers' : random.choice([1, 2]),
        'dropout'  : random.choice([0.0, 0.1, 0.2]),
        'l_rate'   : random.choice([1e-1, 5e-2, 1e-2, 5e-3, 1e-1])
    }

""" COMMAND LINE INTERFACE """
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--grid_search',
                       action='store_true',
                       help='Select parameters from predefined ranges')
    
    parser.add_argument('--n_epochs', 
                       type=int, 
                       default=5,
                       help='Number of training epochs')
    
    parser.add_argument('--h_size', 
                       type=int, 
                       default=4,
                       help='Hidden layer size')
    
    parser.add_argument('--n_layers', 
                       type=int, 
                       default=1,
                       help='Number of hidden layers')
    
    parser.add_argument('--dropout', 
                       type=float, 
                       default=0.1,
                       help='Dropout rate (0.0 to 1.0)')
    
    parser.add_argument('--l_rate', 
                       type=float, 
                       default=0.001,
                       help='Learning rate')
    
    args = parser.parse_args()

    if args.grid_search:
        random_params = get_random_parameters()
        args.n_epochs = random_params['n_epochs']
        args.h_size = random_params['h_size']
        args.n_layers = random_params['n_layers']
        args.dropout = random_params['dropout']
        args.l_rate = random_params['l_rate']
        
    return args

""" TRAINING LOOP """
def train_repetition(D:DataGen, args: dict, plot: bool=False) -> pd.DataFrame:

    """ LOAD DATA """
    train_dl, valid_dl, _, vocab_size, _ = D.dataloaders()

    """ UNPACK PARAMETERS """
    num_epochs     = args['n_epochs']
    hidden_size    = args['h_size']
    num_layers     = args['n_layers']
    dropout        = args['dropout']
    learning_rate  = args['l_rate']

    """ INITIALIZE MODEL """
    model = f'e{num_epochs}_h{hidden_size}_l{num_layers}_d{dropout}_r{learning_rate}'
 
    encoder = EncoderRNN(
        input_size=vocab_size, hidden_size=hidden_size,
        num_layers=num_layers, dropout=dropout
    ).to(device)

    decoder = DecoderRNN(
        hidden_size=hidden_size, output_size=vocab_size,
        num_layers=num_layers, dropout=dropout
    ).to(device)

    # # if CUDA, use DataParallel
    # if torch.cuda.device_count() > 1:
    #     encoder = nn.DataParallel(encoder).to(device)
    #     decoder = nn.DataParallel(decoder).to(device)

    criterion = nn.CrossEntropyLoss()
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(parameters, lr=learning_rate)

    """ TRAINING LOOP """
    train_losses = []
    valid_losses = []
    epoch_times = []
    timer = Timer()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        epoch_start = time.time()
        encoder.train()
        decoder.train()
        train_loss = 0

        timer.start()
        for i, (inputs, targets) in enumerate(train_dl):
            print(f"Batch {i+1}/{len(train_dl)}", end='\r')

            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            # Forward passes
            timer.start()
            encoder_hidden = encoder(inputs)
            timer.stop("Encoder Forward Pass")

            timer.start()
            decoder_input = torch.zeros(1, inputs.shape[1], hidden_size, device=device)
            decoder_output = decoder(decoder_input, encoder_hidden)
            timer.stop("Decoder Forward Pass")

            # Loss computation
            timer.start()
            outputs = decoder_output.view(-1, vocab_size)  
            targets = targets.view(-1)
            loss = criterion(outputs, targets)
            timer.stop("Loss Computation")

            # Backward pass
            timer.start()
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
        log = f"Epoch {epoch+1}: Train: {train_loss:.3f} "
        log += f"Valid: {valid_loss:.3f} Time: {epoch_time:.2f}s"
        print(log)

    """ PLOT LOSS """
    training_curves(train_losses, valid_losses, model, num_epochs)

    """ SAVE MODEL """
    torch.save(encoder.state_dict(), WEIGHTS_DIR / f"encoder_{model}.pth")
    torch.save(decoder.state_dict(), WEIGHTS_DIR / f"decoder_{model}.pth")

    # Print timing summary
    timer.summary()

    return D.test_data

if __name__ == "__main__":
    D = DataGen()

    args = parse_args()
    print(f"Running with parameters:")
    print(f"Num Epochs    : {args.n_epochs}")
    print(f"Hidden Size   : {args.h_size}")
    print(f"Num Layers    : {args.n_layers}")
    print(f"Dropout       : {args.dropout}")
    print(f"Learning Rate : {args.l_rate}")

    data = train_repetition(D, vars(args))