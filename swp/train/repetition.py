import time

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..models.autoencoder import Bimodel, Unimodel
from ..utils.datasets import get_phoneme_to_id
from ..utils.grid_search import grid_search_log
from ..utils.models import save_weights


def train(
    model: Unimodel | Bimodel,
    model_name: str,
    train_name: str,
    criterion: nn.Module,
    optimizer: Optimizer,
    phoneme_to_id: dict[str, int],
    train_loader: DataLoader,
    valid_loader: DataLoader,
    num_epochs: int,
    device: str | torch.device,
    verbose: bool = False,
) -> None:
    r"""Trains the `model` over `num_epoch` epochs with the data contained in the `train_loader`,
    the `criterion` loss and the `optimizer` weight update method.

    Set `verbose` to `True` to print intermediate logs.

    Training performances and validation performances (evaluated over `valid_loader`)
    are saved in the end.

    Checkpointing happens 10 times during the first epoch, then once after each epoch.
    """

    if isinstance(model, Unimodel) and not model.is_auditory:
        raise ValueError("Auditory model required")
    if isinstance(model, Bimodel):
        model.to_audio()
    model.to(device)
    model.train()

    train_losses = []
    valid_losses = []
    train_errors = []
    valid_errors = []
    epoch_times = []

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        if verbose:
            print(f"\nEpoch {epoch}")

        ### TRAINING LOOP ###
        model.train()
        train_loss = 0
        train_error = 0
        checkpoint = 1

        for i, (data, target) in enumerate(train_loader, 1):
            if verbose:
                print(f"{i}/{len(train_loader)}", end="\r")

            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()

            # Forward pass
            output = model(data, target)

            # Loss computation
            loss = criterion(output, target)
            train_loss += loss.item()

            # Error computation
            preds = torch.argmax(output[0], dim=-1)
            mask = target != phoneme_to_id["<PAD>"]
            train_error += torch.any((preds != target) * mask, dim=1).sum().item()

            # Backward pass
            loss.backward()
            optimizer.step()

            if epoch == 1 and checkpoint != 10 and i % ((len(train_loader) // 10)) == 0:
                save_weights(model_name, train_name, model, epoch, checkpoint)
                if verbose:
                    print(f"Checkpoint {checkpoint}: {(train_loss / i):.3f}")
                checkpoint += 1

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        train_errors.append(train_error)
        if verbose:
            if train_loss >= 0.001:
                print(f"Train Loss: {train_loss:.3f}")
            else:
                print(f"Train Loss: {train_loss:.2e}")

        ### VALIDATION LOOP ###
        model.eval()
        valid_loss = 0
        valid_error = 0

        with torch.no_grad():
            for i, (data, target) in enumerate(valid_loader, 1):
                if verbose:
                    print(f"{i+1}/{len(valid_loader)}", end="\r")

                data = data.to(device)
                target = target.to(device)

                # Forward pass
                output = model(data, target)

                # Loss computation
                loss = criterion(output, target)
                valid_loss += loss.item()

                # Error computation
                preds = torch.argmax(output[0], dim=-1)
                mask = target != phoneme_to_id["<PAD>"]
                valid_error += torch.any((preds != target) * mask, dim=1).sum().item()

        valid_loss /= len(valid_loader)
        valid_losses.append(valid_loss)
        valid_errors.append(valid_error)
        if verbose:
            if valid_loss >= 0.001:
                print(f"Valid Loss: {valid_loss:.3f}")
            else:
                print(f"Valid Loss: {valid_loss:.2e}")

        ### POST TRAIN/VALID ###
        save_weights(model_name, train_name, model=model, epoch=epoch)
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        if verbose:
            print(f"Train Errors: {train_error}")
            print(f"Valid Errors: {valid_error}")
            h = epoch_time // 3600
            m = epoch_time % 3600 // 60
            s = epoch_time % 3600 % 60
            print(f"Epoch Time: {h:.0f}h {m:.0f}m {s:.0f}s")

    grid_search_log(
        train_losses,
        valid_losses,
        train_errors,
        valid_errors,
        model_name,
        train_name,
        num_epochs,
    )
