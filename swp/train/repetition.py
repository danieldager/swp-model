import time

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from swp.utils.models import save_weights

from ..models.autoencoder import Bimodel, Unimodel
from ..utils.grid_search import grid_search_log
from ..utils.perf import Timer


def train(
    train_loader: DataLoader,
    valid_loader: DataLoader,
    model: Unimodel | Bimodel,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: str | torch.device,
    model_name: str,
    training_name: str,
    num_epochs: int,
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
        raise ValueError(
            "The model to train is not made to be trained with auditory data"
        )
    if isinstance(model, Bimodel):
        model.to_audio()
    model.to(device)

    timer = Timer()
    train_losses = []
    valid_losses = []
    epoch_times = []

    errors = []
    error_count = 0

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        if verbose:
            print(f"\nEpoch {epoch}")

        # training loop
        model.train()
        train_loss = 0
        checkpoint = 1

        for i, (data, target) in enumerate(train_loader, 1):
            if verbose:
                print(f"{i}/{len(train_loader)}", end="\r")
            timer.start()

            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()

            # Forward pass
            output = model(data, target)  # throw away object prediction

            # Loss computation
            loss = criterion(output, target)
            train_loss += loss.item()

            # TODO reconsider this
            # if epoch == num_epochs:
            #     p = torch.argmax(output, dim=1)
            #     p = p.squeeze().tolist()
            #     t = target.squeeze().tolist()
            #     if p != t:
            #         error_count += 1
            #         # p = [index_to_phone[i] for i in p]
            #         # t = [index_to_phone[i] for i in t]
            #         # errors.append((p, t))

            # Backward pass
            timer.start()
            loss.backward()
            optimizer.step()
            timer.stop("Train step")

            if epoch == 1 and checkpoint != 10 and i % ((len(train_loader) // 10)) == 0:
                save_weights(
                    model_name, training_name, model, epoch, checkpoint=checkpoint
                )
                if verbose:
                    print(f"Checkpoint {checkpoint}: {(train_loss / i):.3f}")
                checkpoint += 1

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        if verbose:
            print(f"Train loss: {train_loss:.3f}")

        # VALIDATION LOOP
        model.eval()
        valid_loss = 0

        with torch.no_grad():
            for i, (data, target) in enumerate(valid_loader, 1):
                if verbose:
                    print(f"{i+1}/{len(valid_loader)}", end="\r")

                data = data.to(device)
                target = target.to(device)

                # Forward passes

                output = model(data, target)

                # Loss computation
                loss = criterion(output, target)
                valid_loss += loss.item()

        valid_loss /= len(valid_loader)
        valid_losses.append(valid_loss)
        if verbose:
            print(f"Valid loss: {valid_loss:.3f}")

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        if verbose:
            print(
                f"Epoch time: {epoch_time // 3600:.0f}h {epoch_time % 3600 // 60:.0f}m {epoch_time % 3600:.0f}s"
            )

        # Save model weights for every epoch
        save_weights(model_name, training_name, model=model, epoch=epoch)

    # Plot loss curves and create gridsearch log
    # training_curves(train_losses, valid_losses, model_name, num_epochs)
    grid_search_log(train_losses, valid_losses, model_name, training_name, num_epochs)

    # Print timing summary
    timer.summary()

    # Print error summary
    if verbose:
        print(f"\nError rate: {error_count / len(train_loader):.2f}")
        # for p, t in errors:
        #     print(p)
        #     print(t, "\n")
