import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

""" PATHS """
FILE_DIR = Path(__file__).resolve()
ROOT_DIR = FILE_DIR.parent.parent.parent

WEIGHTS_DIR = ROOT_DIR / "weights"
DATA_DIR = ROOT_DIR / "data"

WEIGHTS_DIR.mkdir(exist_ok=True)

from Phonemes import Phonemes
from plots import training_curves
from utils import Timer, seed_everything, set_device

# TODO: replace this with comment below
import sys
MODELS_DIR = ROOT_DIR / "code" / "models"
sys.path.append(str(MODELS_DIR))
from EncoderRNN import EncoderRNN
from DecoderRNN import DecoderRNN
# from ..models.DecoderRNN import DecoderRNN
# from ..models.EncoderRNN import EncoderRNN

device = set_device()
penalty = torch.tensor(0.0, device=device)

""" ARGUMENT PARSER """
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_epochs", type=int, default=40, help="Number of epochs")

    parser.add_argument("--h_size", type=int, default=4, help="Hidden size")

    parser.add_argument("--n_layers", type=int, default=1, help="Hidden layers")

    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")

    parser.add_argument("--l_rate", type=float, default=0.001, help="Learning rate")

    parser.add_argument("--tf_ratio", type=float, default=0.0, help="Teacher forcing")

    args = parser.parse_args()
    return args

""" CHECKPOINTING """
def save_checkpoint(filepath, encoder, decoder, epoch, checkpoint=None):
    if checkpoint:
        epoch = f"{epoch}_{checkpoint}"
    encoder_path = filepath / f"encoder{epoch}.pth"
    decoder_path = filepath / f"decoder{epoch}.pth"
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)

""" GRID SEARCH LOGGING """
def grid_search_log(train_losses, valid_losses, model):
    try:
        df = pd.read_csv(DATA_DIR / "grid_search.csv")
    except FileNotFoundError:
        # Create a new DataFrame if the file doesn't exist
        columns = ["model", "h_size", "n_layers", "dropout", "l_rate", "tf_ratio"]
        columns += [f"T{i}" for i in range(1, 41)] + [f"V{i}" for i in range(1, 41)]
        df = pd.DataFrame(columns=columns)

    # Extract parameters from the model name
    h, l, d, r, t = [p[1:] for p in model.split("_")[1:]]
    df.loc[model] = [model, h, l, d, r, t] + train_losses + valid_losses

    # Save the DataFrame to a CSV file
    df.to_csv(DATA_DIR / "grid_search.csv", index=False)

""" TRAINING LOOP """
def train_repetition(P: Phonemes, params: dict) -> pd.DataFrame:
    # Unpack variables
    vocab_size = P.vocab_size
    # phone_to_index = P.phone_to_index
    # stop_token = phone_to_index["<STOP>"]
    train_dataloader = P.train_dataloader
    valid_dataloader = P.valid_dataloader

    # Unpack hyperparameters
    n_epochs = params["n_epochs"]
    h_size = params["h_size"]
    n_layers = params["n_layers"]
    dropout = params["dropout"]
    l_rate = params["l_rate"]
    tf_ratio = params["tf_ratio"]

    print(f"Training model with hyperparameters:")
    print(f"Epochs:    {n_epochs}")
    print(f"Hidden:    {h_size}")
    print(f"Layers:    {n_layers}")
    print(f"Dropout:   {dropout}")
    print(f"Learning:  {l_rate}")
    print(f"Teacher:   {tf_ratio}")

    # Initialize model
    model = f"e{n_epochs}_h{h_size}_l{n_layers}"
    model += f"_d{dropout}_r{l_rate}_t{tf_ratio}"
    MODEL_WEIGHTS_DIR = WEIGHTS_DIR / model
    MODEL_WEIGHTS_DIR.mkdir(exist_ok=True)

    encoder = EncoderRNN(vocab_size, h_size, n_layers, dropout).to(device)
    decoder = DecoderRNN(h_size, vocab_size, n_layers, dropout, encoder.embedding).to(device)

    criterion = nn.CrossEntropyLoss()
    parameters = list(set(encoder.parameters()).union(set(decoder.parameters())))
    optimizer = optim.Adam(parameters, lr=l_rate)

    """ TRAINING LOOP """
    train_losses = []
    valid_losses = []
    epoch_times = []
    timer = Timer()

    for epoch in range(1, n_epochs + 1):
        print(f"Epoch {epoch}")

        train_loss = 0
        encoder.train()
        decoder.train()
        epoch_start = time.time()
        if epoch == 1:
            checkpoint = 1

        timer.start()
        for i, (input, target) in enumerate(train_dataloader):
            print(f"{i+1}/{len(train_dataloader)}", end="\r")

            input = input.to(device)
            target = target.to(device)
            optimizer.zero_grad()

            # Forward passes
            # timer.start()
            encoder_hidden = encoder(input)
            # timer.stop("Encoder Forward Pass")

            # timer.start()
            decoder_input = torch.zeros(1, 1, dtype=torch.int64, device=device)
            decoder_output = decoder(decoder_input, encoder_hidden, target, tf_ratio)
            # timer.stop("Decoder Forward Pass")

            # Loss computation
            # timer.start()
            output = decoder_output.view(-1, vocab_size)
            target = target.view(-1)
            # print("o", output.shape)
            # print("t", target.shape)
            loss = criterion(output, target)       
            # timer.stop("Loss Computation")

            # Backward pass
            timer.start()
            loss.backward()
            optimizer.step()
            timer.stop("Backward Pass")
            train_loss += loss.item()

            if epoch == 1 and i+1 % (len(train_dataloader) // 10) == 0:
                save_checkpoint(MODEL_WEIGHTS_DIR, encoder, decoder, epoch, checkpoint)
                checkpoint += 1

        train_loss /= len(train_dataloader)
        print(f"Train loss: {train_loss:.3f}")
        train_losses.append(train_loss)

        """ VALIDATION LOOP """
        timer.start()
        encoder.eval()
        decoder.eval()
        valid_loss = 0

        with torch.no_grad():
            for input, target in valid_dataloader:
                input = input.to(device)
                target = target.to(device)

                encoder_hidden = encoder(input)
                decoder_input = torch.zeros(1, 1, dtype=torch.int64, device=device)
                decoder_output = decoder(decoder_input, encoder_hidden, target, 0.0)

                output = decoder_output.view(-1, vocab_size)
                target = target.view(-1)
                loss = criterion(output, target)
                valid_loss += loss.item()

        valid_loss /= len(valid_dataloader)
        valid_losses.append(valid_loss)
        timer.stop("Validation")

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        log = f"Epoch {epoch}: Train: {train_loss:.3f} "
        log += f"Valid: {valid_loss:.3f} Time: {epoch_time:.2f}s"
        print(log)

        # Save model weights for every epoch
        save_checkpoint(MODEL_WEIGHTS_DIR, encoder, decoder, epoch)

    # Plot loss curves and create gridsearch log
    training_curves(train_losses, valid_losses, model, n_epochs)
    grid_search_log(train_losses, valid_losses, model)

    # Print timing summary
    timer.summary()

    return model

if __name__ == "__main__":
    seed_everything()
    P = Phonemes()
    args = parse_args()
    parameters = vars(args)
    train_repetition(P, parameters)
