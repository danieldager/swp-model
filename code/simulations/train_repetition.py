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
from utils import Timer, seed_everything, set_device, alignment_loss

# TODO: replace this with comment below
import sys
MODELS_DIR = ROOT_DIR / "code" / "models"
sys.path.append(str(MODELS_DIR))
from EncoderRNN import EncoderRNN
from DecoderRNN import DecoderRNN
# from ..models.DecoderRNN import DecoderRNN
# from ..models.EncoderRNN import EncoderRNN

device = set_device()
penalty = torch.tensor(0.5, device=device)

""" ARGUMENT PARSER """
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n_epochs", type=int, default=30, help="Number of training epochs"
    )

    parser.add_argument("--h_size", type=int, default=4, help="Hidden layer size")

    parser.add_argument(
        "--n_layers", type=int, default=1, help="Number of hidden layers"
    )

    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout rate (0.0 to 1.0)"
    )

    parser.add_argument("--l_rate", type=float, default=0.001, help="Learning rate")

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
        columns = ["model", "hidden_size", "num_layers", "dropout", "learning_rate"]
        columns += [f"T{i}" for i in range(1, 31)] + [f"V{i}" for i in range(1, 31)]
        df = pd.DataFrame(columns=columns)

    # Extract parameters from the model name
    h, l, d, r = [p[1:] for p in model.split("_")[1:]]
    df.loc[model] = [model, h, l, d, r] + train_losses + valid_losses

    # Save the DataFrame to a CSV file
    df.to_csv(DATA_DIR / "grid_search.csv", index=False)


""" TRAINING LOOP """
def train_repetition(P: Phonemes, params: dict) -> pd.DataFrame:
    # Unpack variables
    tf_ratio = 0.5
    vocab_size = P.vocab_size
    phone_to_index = P.phone_to_index
    # stop_token = phone_to_index["<STOP>"]
    train_dataloader = P.train_dataloader
    valid_dataloader = P.valid_dataloader

    # Unpack hyperparameters
    num_epochs = params["n_epochs"]
    hidden_size = params["h_size"]
    num_layers = params["n_layers"]
    dropout = params["dropout"]
    learning_rate = params["l_rate"]

    print(f"Training model with hyperparameters:")
    print(f"Epochs:    {num_epochs}")
    print(f"Hidden:    {hidden_size}")
    print(f"Layers:    {num_layers}")
    print(f"Dropout:   {dropout}")
    print(f"Learning:  {learning_rate}")

    # Initialize model
    model = f"e{num_epochs}_h{hidden_size}_l{num_layers}_d{dropout}_r{learning_rate}"
    MODEL_WEIGHTS_DIR = WEIGHTS_DIR / model
    MODEL_WEIGHTS_DIR.mkdir(exist_ok=True)

    encoder = EncoderRNN(
        input_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    decoder = DecoderRNN(
        hidden_size=hidden_size,
        output_size=vocab_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(parameters, lr=learning_rate)

    """ TRAINING LOOP """
    train_losses = []
    valid_losses = []
    epoch_times = []
    timer = Timer()

    for epoch in range(1, num_epochs + 1):
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

            # print("target", target.shape)

            input = input.to(device)
            target = target.to(device)
            optimizer.zero_grad()

            # Forward passes
            # timer.start()
            embedded, encoder_hidden = encoder(input)
            # timer.stop("Encoder Forward Pass")

            # timer.start()
            decoder_input = torch.zeros(1, hidden_size, device=device)

            # embedded is the same as nn.embedding(target)
            decoder_output = decoder(decoder_input, encoder_hidden, embedded, tf_ratio)
            # print(decoder_output.shape, target.shape)
            # timer.stop("Decoder Forward Pass")

            # Loss computation
            # timer.start()
            # print("before", decoder_output.shape, target.shape)
            output = decoder_output.view(-1, vocab_size)
            target = target.view(-1)
            # print("after", output.shape, target.shape)

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

                embedded, encoder_hidden = encoder(input)
                decoder_input = torch.zeros(1, hidden_size, device=device)
                decoder_output = decoder(decoder_input, encoder_hidden, embedded, 0.0)

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
    training_curves(train_losses, valid_losses, model, num_epochs)
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
