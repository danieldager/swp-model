import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..models.autoencoder import Unimodel
from ..datasets.phonemes import PhonemeDataset
from ..models.decoders import DecoderLSTM, DecoderRNN
from ..models.encoders import EncoderLSTM, EncoderRNN

from ..utils.perf import Timer
from ..plots import training_curves
from ..utils.models import save_weights
from ..utils.paths import get_weights_dir
from ..utils.datasets import get_phoneme_to_id
from ..utils.grid_search import grid_search_log


def train_repetition(
    fold_id: int,
    model_type: str,
    num_epochs: int,
    batch_size: int,
    hidden_size: int,
    num_layers: int,
    learn_rate: float,
    dropout: float,
    tf_ratio: float,
    device: str | torch.device,
):
    phoneme_to_id = get_phoneme_to_id(force_recreate=False)
    vocab_size = len(phoneme_to_id)

    print(f"\nTraining model with hyperparameters:")
    print(f"Fold:      {fold_id}")
    print(f"Type:      {model_type}")
    print(f"Epochs:    {num_epochs}")
    print(f"Hidden:    {hidden_size}")
    print(f"Layers:    {num_layers}")
    print(f"Learn:     {learn_rate}")
    print(f"Dropout:   {dropout}")
    print(f"Teacher:   {tf_ratio}")

    # Initialize model
    model_name = f"h{hidden_size}_r{learn_rate}_d{dropout}_t{tf_ratio}"
    model_name += f"_l{num_layers}_{model_type}_f{fold_id}"
    model_weights_path = get_weights_dir() / model_name
    model_weights_path.mkdir(exist_ok=True)

    if model_type == "rnn":
        encoder = EncoderRNN(
            vocab_size,
            hidden_size,
            num_layers,
            dropout,
        ).to(device)
        decoder = DecoderRNN(
            hidden_size,
            vocab_size,
            num_layers,
            dropout,
            tf_ratio,
        ).to(device)

    elif model_type == "lstm":
        encoder = EncoderLSTM(
            vocab_size,
            hidden_size,
            num_layers,
            dropout,
        ).to(device)
        decoder = DecoderLSTM(
            hidden_size,
            vocab_size,
            num_layers,
            dropout,
            tf_ratio,
        ).to(device)

    else:
        raise ValueError(f"Invalid model type: {model_type}")

    start = torch.zeros(batch_size, 1, dtype=int, device=device)
    model = Unimodel(encoder, decoder, start)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    timer = Timer()
    train_losses = []
    valid_losses = []
    epoch_times = []

    errors = []
    error_count = 0

    # NOTE what's the difference between get_epoch and get_epoch_numpy ?
    train_dataset = PhonemeDataset(fold_id, train=True, phoneme_to_id=phoneme_to_id)
    valid_dataset = PhonemeDataset(fold_id, train=False, phoneme_to_id=phoneme_to_id)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch}")

        """ TRAINING LOOP """
        model.train()
        train_loss = 0
        checkpoint = 1

        for i, (input, target) in enumerate(train_dataloader, 1):
            # print(f"{i}/{len(train_dataloader)}", end="\r")
            timer.start()

            if i == 100:
                print("100th iteration")

            if i == 1000:
                print("1000th iteration")

            if i == 10000:
                print("10000th iteration")

            if i == 100000:
                print("100000th iteration")

            # Forward pass
            input = input.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output, _ = model(input, target)

            # Loss computation
            output = output.view(-1, vocab_size)
            target = target.view(-1)
            loss = criterion(output, target)
            train_loss += loss.item()

            if epoch == num_epochs:
                p = torch.argmax(output, dim=1)
                p = p.squeeze().tolist()
                t = target.squeeze().tolist()
                if p != t:
                    error_count += 1
                    # p = [index_to_phone[i] for i in p]
                    # t = [index_to_phone[i] for i in t]
                    # errors.append((p, t))

            # Backward pass
            timer.start()
            loss.backward()
            optimizer.step()
            timer.stop("Train step")

            # Save model weights for every 1/10 of first epoch
            if (
                epoch == 1
                and checkpoint != 10
                and i % ((len(train_dataloader) // 10)) == 0
            ):
                save_weights(model_weights_path, model, epoch, checkpoint)
                print(f"Checkpoint {checkpoint}: {(train_loss / i):.3f}", flush=True)
                checkpoint += 1

        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)
        print(f"Train loss: {train_loss:.3f}", flush=True)

        """ VALIDATION LOOP """
        model.eval()
        valid_loss = 0

        with torch.no_grad():
            for i, (input, target) in enumerate(valid_dataloader, 1):
                # print(f"{i+1}/{len(valid_dataloader)}", end="\r")

                # Forward pass
                input = input.to(device)
                target = target.to(device)
                output, _ = model(input, target)

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
        save_weights(model_weights_path, model, epoch)

    # Plot loss curves and create gridsearch log
    training_curves(train_losses, valid_losses, model_name, num_epochs)
    grid_search_log(train_losses, valid_losses, model_name, num_epochs)

    # Print timing summary
    timer.summary()

    # Print error summary
    print(f"\nError rate: {error_count / len(train_dataloader):.2f}")
    # for p, t in errors:
    #     print(p)
    #     print(t, "\n")

    return model_name
