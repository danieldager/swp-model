import time
from textwrap import indent

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader

from ..models.autoencoder import Bimodel, Unimodel
from ..models.metrics import TaskErrormeter
from ..utils.earlystop import SlurmHandler
from ..utils.grid_search import grid_search_log
from ..utils.metrics import compute_preds
from ..utils.models import save_training_checkpoint, save_weights


def train(
    model: Unimodel | Bimodel,
    model_name: str,
    train_name: str,
    criterion: nn.Module,
    optimizer: Optimizer,
    train_loader: StatefulDataLoader,
    valid_loader: StatefulDataLoader,
    num_epochs: int,
    device: str | torch.device,
    train_errormeter: TaskErrormeter,
    valid_errormeter: TaskErrormeter,
    verbose: bool = False,
    sig_handler: SlurmHandler | None = None,
    resume_from: tuple[int, str, int, float, float] | None = None,
    optimize_memory: bool = False,
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
    middle_broke = False
    if resume_from is not None:
        last_epoch, where, last_batch, train_loss, valid_loss = resume_from
        if where == "end":
            where = None
        else:
            last_epoch -= 1
    else:
        last_epoch = 0
        where = None
        last_batch = 0
        train_loss = 0
        valid_loss = 0
        train_errormeter.reset()
        valid_errormeter.reset()

    if isinstance(model, Unimodel) and not model.is_visual:
        raise ValueError(
            "The model to train is not made to be trained with visual data"
        )
    if isinstance(model, Bimodel):
        model.to_visual()
    model.to(device)
    optimizer.load_state_dict(
        optimizer.state_dict()
    )  # trick to ensure optimizer is on right device

    train_losses = []
    valid_losses = []
    train_errors = []
    valid_errors = []
    epoch_times = []

    for epoch in range(last_epoch + 1, num_epochs + 1):
        epoch_start = time.time()
        if verbose:
            print(f"\nEpoch {epoch}")

        if where == "train":
            train_loss, train_error, middle_broke = train_epoch(
                model=model,
                model_name=model_name,
                train_name=train_name,
                criterion=criterion,
                optimizer=optimizer,
                train_loader=train_loader,
                valid_loader=valid_loader,
                train_errormeter=train_errormeter,
                valid_errormeter=valid_errormeter,
                epoch=epoch,
                device=device,
                verbose=verbose,
                sig_handler=sig_handler,
                last_batch=last_batch,
                initial_loss=train_loss,
                optimize_memory=optimize_memory,
            )
            where = None
            last_batch = 0
        elif where is None:
            train_loss, train_error, middle_broke = train_epoch(
                model=model,
                model_name=model_name,
                train_name=train_name,
                criterion=criterion,
                optimizer=optimizer,
                train_loader=train_loader,
                valid_loader=valid_loader,
                train_errormeter=train_errormeter,
                valid_errormeter=valid_errormeter,
                epoch=epoch,
                device=device,
                verbose=verbose,
                sig_handler=sig_handler,
                optimize_memory=optimize_memory,
            )
        else:
            train_error = train_errormeter.get_errors()
        if middle_broke:
            break
        train_losses.append(train_loss)
        train_errors.append(train_error)

        if where == "valid":
            valid_loss, valid_error, middle_broke = valid_epoch(
                model=model,
                model_name=model_name,
                train_name=train_name,
                criterion=criterion,
                optimizer=optimizer,
                train_loader=train_loader,
                valid_loader=valid_loader,
                train_errormeter=train_errormeter,
                valid_errormeter=valid_errormeter,
                epoch=epoch,
                device=device,
                verbose=verbose,
                train_loss=train_loss,
                sig_handler=sig_handler,
                last_batch=last_batch,
                initial_loss=valid_loss,
                optimize_memory=optimize_memory,
            )
            where = None
            last_batch = 0
        elif where is None:
            valid_loss, valid_error, middle_broke = valid_epoch(
                model=model,
                model_name=model_name,
                train_name=train_name,
                criterion=criterion,
                optimizer=optimizer,
                train_loader=train_loader,
                valid_loader=valid_loader,
                train_errormeter=train_errormeter,
                valid_errormeter=valid_errormeter,
                epoch=epoch,
                device=device,
                verbose=verbose,
                train_loss=train_loss,
                sig_handler=sig_handler,
                optimize_memory=optimize_memory,
            )
        else:
            valid_error = valid_errormeter.get_errors()
        if middle_broke:
            break
        valid_losses.append(valid_loss)
        valid_errors.append(valid_error)

        ### POST TRAIN/VALID ###
        save_weights(model_name, train_name, model=model, epoch=epoch)
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        if verbose:
            print(f"Train Errors:\n{indent(train_errormeter.summary(), prefix='    ')}")
            print(f"Valid Errors:\n{indent(valid_errormeter.summary(), prefix='    ')}")
            h = epoch_time // 3600
            m = epoch_time % 3600 // 60
            s = epoch_time % 60
            print(f"Epoch Time: {h:.0f}h {m:.0f}m {s:.0f}s")

        if sig_handler is not None and sig_handler.stop_signal and epoch != num_epochs:
            sig_handler.ask_requeue()
            save_training_checkpoint(
                model_name=model_name,
                train_name=train_name,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                where="end",
                train_loader=train_loader,
                valid_loader=valid_loader,
                train_errormeter=train_errormeter,
                valid_errormeter=valid_errormeter,
            )
            if verbose:
                print("Training checkpoint succesfully created")
            break

        valid_loss = 0
        valid_errormeter.reset()
        train_errormeter.reset()

    if not middle_broke:
        grid_search_log(
            train_losses,
            valid_losses,
            train_errors,
            valid_errors,
            model_name,
            train_name,
            epoch,
            append=sig_handler is not None,
            from_epoch=last_epoch,
        )


def train_epoch(
    model: Bimodel | Unimodel,
    model_name: str,
    train_name: str,
    criterion: nn.Module,
    optimizer: Optimizer,
    train_loader: StatefulDataLoader,
    valid_loader: StatefulDataLoader,
    train_errormeter: TaskErrormeter,
    valid_errormeter: TaskErrormeter,
    epoch: int,
    device: str | torch.device,
    verbose: bool,
    sig_handler: SlurmHandler | None = None,
    last_batch: int = 0,
    initial_loss: float = 0,
    optimize_memory: bool = False,
) -> tuple[float, tuple[int, ...], bool]:
    # TODO docstring
    model.train()
    checkpoint = 1

    train_loss = initial_loss

    ### Max size batch to allocate memory and avoid fragmentation ###
    if optimize_memory:
        raise NotImplementedError("Memory optimization not implemented")
        # batch_size = train_loader.batch_size
        # if batch_size is not None:
        #     data_buffer = torch.zeros((batch_size, 3, 224, 224), device=device)
        #     target_buffer = torch.zeros(
        #         (batch_size, train_loader.dataset.max_len),  # type: ignore
        #         dtype=torch.long,
        #         device=device,
        #     )
        # else:
        #     data_buffer = torch.zeros((3, 224, 224), device=device)
        #     target_buffer = torch.zeros((train_loader.dataset.max_len), dtype=torch.long, device=device)  # type: ignore

    for i, tensordict in enumerate(train_loader, 1 + last_batch):
        if verbose:
            print(f"{i}/{len(train_loader)}", end="\n")

        if optimize_memory:
            raise NotImplementedError("Memory optimization not implemented")
            # batch_len = len(data)
            # seq_len = target.size(-1)
            # data_dev = data_buffer[:batch_len].copy_(data)
            # target_dev = target_buffer[:batch_len, :seq_len].copy_(target)
        tensordict = tensordict.to(device)

        optimizer.zero_grad()

        # Forward pass
        tensordict = model(tensordict)

        # Loss computation
        tensordict = criterion(tensordict)
        loss = tensordict["loss"]
        train_loss += loss.detach().cpu().numpy()

        # Error computation
        tensordict = compute_preds(tensordict)
        train_errormeter.accumulate(tensordict)

        # Backward pass
        loss.backward()
        optimizer.step()

        if epoch == 1 and checkpoint != 10 and i % ((len(train_loader) // 10)) == 0:
            save_weights(model_name, train_name, model, epoch, checkpoint)
            if verbose:
                print(f"Checkpoint {checkpoint}: {(train_loss / i):.3f}")
            checkpoint += 1
        if sig_handler is not None and sig_handler.stop_signal:
            sig_handler.ask_requeue()
            save_training_checkpoint(
                model_name=model_name,
                train_name=train_name,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                where="train",
                train_loader=train_loader,
                valid_loader=valid_loader,
                train_errormeter=train_errormeter,
                valid_errormeter=valid_errormeter,
                last_batch=i,
                train_loss=train_loss,
            )
            if verbose:
                print("Training checkpoint succesfully created")
            return 0, (), True

    train_loss /= len(train_loader)

    if verbose:
        if train_loss >= 0.001:
            print(f"Train Loss: {train_loss:.3f}")
        else:
            print(f"Train Loss: {train_loss:.2e}")

    return train_loss, train_errormeter.get_errors(), False


def valid_epoch(
    model: Bimodel | Unimodel,
    model_name: str,
    train_name: str,
    criterion: nn.Module,
    optimizer: Optimizer,
    train_loader: StatefulDataLoader,
    valid_loader: StatefulDataLoader,
    train_errormeter: TaskErrormeter,
    valid_errormeter: TaskErrormeter,
    epoch: int,
    device: str | torch.device,
    verbose: bool,
    train_loss: float,
    sig_handler: SlurmHandler | None = None,
    last_batch: int = 0,
    initial_loss: float = 0,
    optimize_memory: bool = False,
) -> tuple[float, tuple[int, ...], bool]:
    # TODO docstring
    model.eval()

    valid_loss = initial_loss

    with torch.no_grad():
        for i, tensordict in enumerate(valid_loader, 1 + last_batch):
            if verbose:
                print(f"{i}/{len(valid_loader)}", end="\n")

            if optimize_memory:
                raise NotImplementedError("Memory optimization not implemented")
                # batch_len = len(data)
                # seq_len = target.size(-1)
                # data_dev = data_buffer[:batch_len].copy_(data)
                # target_dev = target_buffer[:batch_len, :seq_len].copy_(target)
            tensordict.to(device)

            # Forward pass
            tensordict = model(tensordict)

            # Loss computation
            tensordict = criterion(tensordict)
            valid_loss += tensordict["loss"].detach().cpu().numpy()

            # Error computation
            tensordict = compute_preds(tensordict)
            valid_errormeter.accumulate(tensordict)
            if sig_handler is not None and sig_handler.stop_signal:
                sig_handler.ask_requeue()
                save_training_checkpoint(
                    model_name=model_name,
                    train_name=train_name,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    where="valid",
                    last_batch=i,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    train_errormeter=train_errormeter,
                    valid_errormeter=valid_errormeter,
                    train_loss=train_loss,
                    valid_loss=valid_loss,
                )
                if verbose:
                    print("Training checkpoint succesfully created")
                return 0, (), True

    valid_loss /= len(valid_loader)

    if verbose:
        if valid_loss >= 0.001:
            print(f"Valid Loss: {valid_loss:.3f}")
        else:
            print(f"Valid Loss: {valid_loss:.2e}")

    return valid_loss, valid_errormeter.get_errors(), False
