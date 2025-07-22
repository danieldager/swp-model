import time

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..models.autoencoder import Bimodel, Unimodel
from ..utils.earlystop import SlurmHandler
from ..utils.grid_search import grid_search_log
from ..utils.models import save_training_checkpoint, save_weights


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
    sig_handler: SlurmHandler | None = None,
    from_epoch: int = 0,
) -> None:
    r"""Trains the `model` up to epoch `num_epoch` from epoch `from_epoch` with the data contained in the `train_loader`,
    the `criterion` loss and the `optimizer` weight update method.

    Set `verbose` to `True` to print intermediate logs.

    Training performances and validation performances (evaluated over `valid_loader`)
    are saved in the end.

    Checkpointing happens 10 times during the first epoch, then once after each epoch.

    Provide a `sig_handler` in order to trigger premature training saving and exit when a signal is sent to the process.
    The premature saving and exit can only take place at the end of an epoch.
    """

    if isinstance(model, Unimodel) and not model.is_visual:
        raise ValueError(
            "The model to train is not made to be trained with visual data"
        )
    if isinstance(model, Bimodel):
        model.to_visual()
    model.to(device)
    model.train()

    train_losses = []
    valid_losses = []
    train_errors = []
    valid_errors = []
    epoch_times = []

    for epoch in range(from_epoch + 1, num_epochs + 1):
        epoch_start = time.time()
        if verbose:
            print(f"\nEpoch {epoch}")

        ### TRAINING LOOP ###
        model.train()
        train_loss = 0
        train_error = 0
        checkpoint = 1

        ### Max size batch to allocate memory and avoid fragmentation ###
        batch_size = train_loader.batch_size
        if batch_size is not None:
            data_buffer = torch.zeros((batch_size, 3, 224, 224), device=device)
            target_buffer = torch.zeros(
                (batch_size, train_loader.dataset.max_len),  # type: ignore
                dtype=torch.long,
                device=device,
            )
        else:
            data_buffer = torch.zeros((3, 224, 224), device=device)
            target_buffer = torch.zeros((train_loader.dataset.max_len), dtype=torch.long, device=device)  # type: ignore

        for i, (data, target) in enumerate(train_loader, 1):
            if verbose:
                print(f"{i}/{len(train_loader)}", end="\n")

            batch_len = len(data)
            seq_len = target.size(-1)
            data_dev = data_buffer[:batch_len].copy_(data)
            target_dev = target_buffer[:batch_len, :seq_len].copy_(target)
            optimizer.zero_grad()

            # Forward pass
            output = model(data_dev, target_dev)

            # Loss computation
            loss = criterion(output, target_dev)
            train_loss += loss.detach().cpu().numpy()

            # Error computation
            preds = torch.argmax(output[0], dim=-1)
            mask = target_dev != phoneme_to_id["<PAD>"]
            train_error += (
                torch.any((preds != target_dev) * mask, dim=1)
                .detach()
                .cpu()
                .numpy()
                .sum()
            )
            # TODO add error computation for visual prediction

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
                    print(f"{i+1}/{len(valid_loader)}", end="\n")

                data = data.to(device)
                target = target.to(device)

                # Forward pass
                output = model(data, target)

                # Loss computation
                loss = criterion(output, target)
                valid_loss += loss.detach().cpu().numpy()

                # Error computation
                preds = torch.argmax(output[0], dim=-1)
                mask = target != phoneme_to_id["<PAD>"]
                valid_error += (
                    torch.any((preds != target) * mask, dim=1)
                    .detach()
                    .cpu()
                    .numpy()
                    .sum()
                )
                # TODO add error computation for visual loss

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
            s = epoch_time % 60
            print(f"Epoch Time: {h:.0f}h {m:.0f}m {s:.0f}s")

        if sig_handler is not None and sig_handler.stop_signal and epoch != num_epochs:
            sig_handler.ask_requeue()
            save_training_checkpoint(
                model_name=model_name,
                train_name=train_name,
                optimizer=optimizer,
                epoch=epoch,
                train_loader=train_loader,
                valid_loader=valid_loader,
            )
            break

    grid_search_log(
        train_losses,
        valid_losses,
        train_errors,
        valid_errors,
        model_name,
        train_name,
        num_epochs,
        append=sig_handler is not None,
    )
