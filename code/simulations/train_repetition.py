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
from test_repetition import test_repetition
from utils import Timer, seed_everything, set_device

# TODO: replace this with comment below
import sys
MODELS_DIR = ROOT_DIR / "code" / "models"
sys.path.append(str(MODELS_DIR))
from EncoderRNN import EncoderRNN
from DecoderRNN import DecoderRNN
from EncoderLSTM import EncoderLSTM
from DecoderLSTM import DecoderLSTM
# from ..models.DecoderRNN import DecoderRNN
# from ..models.EncoderRNN import EncoderRNN

device = set_device()
penalty = torch.tensor(0.0, device=device)

""" ARGUMENT PARSER """
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_epochs", type=int, default=40)

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")

    parser.add_argument("--hidden_size", type=int, default=4, help="Hidden size")

    parser.add_argument("--num_layers", type=int, default=1, help="Hidden layers")

    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")

    parser.add_argument("--tf_ratio", type=float, default=0.0, help="Teacher forcing")
    
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()
    return args

""" CHECKPOINTING """
def save_weights(filepath, encoder, decoder, epoch, checkpoint=None):
    if checkpoint:
        epoch = f"{epoch}_{checkpoint}"
    encoder_path = filepath / f"encoder{epoch}.pth"
    decoder_path = filepath / f"decoder{epoch}.pth"
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)

""" GRID SEARCH LOGGING """
def grid_search_log(train_losses, valid_losses, model, num_epochs):
    try:
        df = pd.read_csv(DATA_DIR / "grid_search.csv")
    except FileNotFoundError:
        # Create a new DataFrame if the file doesn't exist
        print("\nCreating new grid search log")
        columns = ["model", "h_size", "n_layers", "dropout", "tf_ratio", "l_rate",]
        columns += [f"T{i}" for i in range(1, num_epochs + 1)]
        columns += [f"V{i}" for i in range(1, num_epochs + 1)]
        df = pd.DataFrame(columns=columns)

    # Extract parameters from the model name
    h, l, d, t, r = [p[1:] for p in model.split("_")[1:]]
    df.loc[model] = [model, h, l, d, t, r] + train_losses + valid_losses
    print("model", model)

    # Save the DataFrame to a CSV file
    df.to_csv(DATA_DIR / "grid_search.csv", index=False)

""" TRAINING LOOP """
def train_repetition(P: Phonemes, params: dict) -> pd.DataFrame:
    # Unpack variables
    vocab_size = P.vocab_size
    index_to_phone = P.index_to_phone
    train_dataloader = P.train_dataloader
    valid_dataloader = P.valid_dataloader

    # Unpack hyperparameters
    num_epochs    = params["num_epochs"]
    batch_size    = params["batch_size"]
    hidden_size   = params["hidden_size"]
    num_layers    = params["num_layers"]
    dropout       = params["dropout"]
    tf_ratio      = params["tf_ratio"]
    learning_rate = params["learning_rate"]

    print(f"\nTraining model with hyperparameters:")
    print(f"Epochs:    {num_epochs}")
    print(f"Hidden:    {hidden_size}")
    print(f"Layers:    {num_layers}")
    print(f"Dropout:   {dropout}")
    print(f"Teacher:   {tf_ratio}")
    print(f"Learning:  {learning_rate}")

    # Initialize model
    model = f"e{num_epochs}_h{hidden_size}_l{num_layers}"
    model += f"_d{dropout}_t{tf_ratio}_r{learning_rate}"
    MODEL_WEIGHTS_DIR = WEIGHTS_DIR / model
    MODEL_WEIGHTS_DIR.mkdir(exist_ok=True)

    # encoder = EncoderRNN(vocab_size, hidden_size, num_layers, dropout).to(device)
    # decoder = DecoderRNN(hidden_size, vocab_size, num_layers, dropout).to(device)
    encoder = EncoderLSTM(vocab_size, hidden_size, num_layers).to(device)
    decoder = DecoderLSTM(hidden_size, vocab_size, num_layers).to(device)

    criterion = nn.CrossEntropyLoss()
    parameters = list(encoder.parameters()) + list(decoder.parameters())    
    optimizer = optim.Adam(parameters, lr=learning_rate)

    timer = Timer()
    train_losses = []
    valid_losses = []
    epoch_times = []

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch}")

        """ TRAINING LOOP """
        encoder.train()
        decoder.train()
        train_loss = 0
        checkpoint = 1

        for i, (input, target) in enumerate(train_dataloader, 1):
            print(f"{i}/{len(train_dataloader)}", end="\r")

            length = input.shape[1]
            input = input.to(device)
            target = target.to(device)
            optimizer.zero_grad()

            # timer.start()
            encoder_hidden = encoder(input)
            # print("e out", encoder_hidden.shape)
            # timer.stop("Encoder Forward Pass")

            decoder_input = torch.zeros(batch_size, length, hidden_size, device=device) # (B, L, H)
            # print("d input", decoder_input.shape)

            # start_token = torch.zeros(1, batch_size, dtype=torch.int64, device=device)
            # start_token = encoder.embedding(start_token) # (1, B, H)

            # timer.start()
            output = decoder(decoder_input, encoder_hidden) # (B, L, V)
            # print("d out", output.shape)
            # timer.stop("Decoder Forward Pass")

            # Loss computation
            # length = output.shape[1]
            # if length > 7:
            #     p = torch.argmax(output, dim=2)
            #     p = p.squeeze().tolist()
            #     p = [index_to_phone[i] for i in p]
            #     t = target.squeeze().tolist()
            #     t = [index_to_phone[i] for i in t]
            #     print(p)
            #     print(t)  
            
            output = output.view(-1, vocab_size)        # (B * L, V)
            target = target.view(-1)                    # (B * L)
            loss = criterion(output, target)           
            # print("loss fn", output.shape, target.view(-1).shape)
            train_loss += loss.item()
            
            # if length > 7: print(loss.item(), '\n')

            # Backward pass
            timer.start()
            loss.backward()
            optimizer.step()
            timer.stop("Backward Pass")

            if epoch == 1 and i % ((len(train_dataloader) // 10)) == 0:
                save_weights(MODEL_WEIGHTS_DIR, encoder, decoder, epoch, checkpoint)
                print(f"Checkpoint {checkpoint} loss: {(train_loss / i):.3f}")
                checkpoint += 1

        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)
        print(f"Train loss: {train_loss:.3f}")

        """ VALIDATION LOOP """
        encoder.eval()
        decoder.eval()
        valid_loss = 0

        with torch.no_grad():
            for i, (input, target) in enumerate(valid_dataloader, 1):
                print(f"{i+1}/{len(valid_dataloader)}", end="\r")
                
                length = input.shape[1]
                input = input.to(device)
                target = target.to(device)

                # Forward passes
                encoder_hidden = encoder(input)
                start_token = torch.zeros(batch_size, length, hidden_size, device=device)                
                output = decoder(start_token, encoder_hidden)

                # Loss computation
                output = output.view(-1, vocab_size)
                target = target.view(-1)
                loss = criterion(output, target)
                valid_loss += loss.item()

        valid_loss /= len(valid_dataloader)
        valid_losses.append(valid_loss)
        print(f"Valid loss: {valid_loss:.3f}")

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        print(f"Epoch time: {epoch_time // 3600:.0f}h {epoch_time % 3600 // 60:.0f}m")

        # Save model weights for every epoch
        save_weights(MODEL_WEIGHTS_DIR, encoder, decoder, epoch)

    # Plot loss curves and create gridsearch log
    training_curves(train_losses, valid_losses, model, num_epochs)
    grid_search_log(train_losses, valid_losses, model, num_epochs)

    # Print timing summary
    timer.summary()

    return model

if __name__ == "__main__":
    seed_everything()
    P = Phonemes()
    args = parse_args()
    params = vars(args)
    model = train_repetition(P, params)
    # results = test_repetition(P, model)
