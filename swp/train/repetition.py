import time

import torch
import torch.nn as nn
import torch.optim as optim

from ..datasets.phonemes import Phonemes
from ..models.decoders import DecoderLSTM, DecoderRNN
from ..models.encoders import EncoderLSTM, EncoderRNN
from ..plots import training_curves
from ..utils.grid_search import grid_search_log
from ..utils.models import save_weights
from ..utils.paths import get_checkpoint_dir
from ..utils.perf import Timer


def train_repetition(P: Phonemes, params: dict, device):
    # Unpack variables
    vocab_size = P.vocab_size
    index_to_phone = P.index_to_phone
    train_dataloader = P.train_dataloader
    valid_dataloader = P.valid_dataloader

    # Unpack hyperparameters
    num_epochs = params["num_epochs"]
    batch_size = params["batch_size"]
    hidden_size = params["hidden_size"]
    num_layers = params["num_layers"]
    dropout = params["dropout"]
    tf_ratio = params["tf_ratio"]
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
    MODEL_WEIGHTS_DIR = get_checkpoint_dir() / model
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

            decoder_input = torch.zeros(
                batch_size, length, hidden_size, device=device
            )  # (B, L, H)
            # print("d input", decoder_input.shape)

            # start_token = torch.zeros(1, batch_size, dtype=torch.int64, device=device)
            # start_token = encoder.embedding(start_token) # (1, B, H)

            # timer.start()
            output = decoder(decoder_input, encoder_hidden)  # (B, L, V)
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

            output = output.view(-1, vocab_size)  # (B * L, V)
            target = target.view(-1)  # (B * L)
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
                start_token = torch.zeros(
                    batch_size, length, hidden_size, device=device
                )
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
