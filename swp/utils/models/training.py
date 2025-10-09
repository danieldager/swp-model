import random

import numpy as np
import torch
from torch.optim import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader

from ...models.autoencoder import Bimodel, Unimodel
from ...models.metrics import ErrorMeter
from ..paths import get_checkpoint_dir


def save_training_checkpoint(
    model_name: str,
    train_name: str,
    model: Bimodel | Unimodel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    where: str,
    train_loader: StatefulDataLoader,
    valid_loader: StatefulDataLoader,
    train_errormeter: ErrorMeter,
    valid_errormeter: ErrorMeter,
    last_batch: int = 0,
    train_loss: float = 0,
    valid_loss: float = 0,
):
    r"""Allows to save all the training state (except the model weights) to resume
    a fully reproducible training later. Make sure to call `save_weights` as well."""
    ckpt_path = get_checkpoint_dir() / f"{model_name}~{train_name}.ckpt"
    checkpoint = {
        "epoch": epoch,
        "where": where,
        "last_batch": last_batch,
        "model": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "random_state": random.getstate(),
        "numpy_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "trainloader_state_dict": train_loader.state_dict(),
        "validloader_state_dict": valid_loader.state_dict(),
        "train_errormeter": train_errormeter.state_dict(),
        "valid_errormeter": valid_errormeter.state_dict(),
        "train_loss": train_loss,
        "valid_loss": valid_loss,
    }
    torch.save(checkpoint, ckpt_path)


def load_last_training_checkpoint(
    model: Unimodel | Bimodel,
    optimizer: Optimizer,
    model_name: str,
    train_name: str,
    train_loader: StatefulDataLoader,
    valid_loader: StatefulDataLoader,
    train_errormeter: ErrorMeter,
    valid_errormeter: ErrorMeter,
    device: torch.device = torch.device("cpu"),
) -> tuple[int, str, int, float, float] | None:
    r"""Restores the state of the saved training of the corresponding model, including model weights.
    Returns four generators, respectively for :
    - the training set
    - the training dataloader
    - the validation set
    - the validation dataloader
    as well as an int containing the last achieved epoch.
    If no saved training is found, does nothing, returns `None` instead of generators, and 0 as the last achieved epoch.
    """
    ckpt_path = get_checkpoint_dir() / f"{model_name}~{train_name}.ckpt"
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        model.bind()

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        random.setstate(checkpoint["random_state"])
        np.random.set_state(checkpoint["numpy_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and checkpoint["torch_cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state"])

        train_loader.load_state_dict(checkpoint["trainloader_state_dict"])
        valid_loader.load_state_dict(checkpoint["validloader_state_dict"])

        train_errormeter.load_state_dict(checkpoint["train_errormeter"])
        valid_errormeter.load_state_dict(checkpoint["valid_errormeter"])

        return (
            checkpoint["epoch"],
            checkpoint["where"],
            checkpoint["last_batch"],
            checkpoint["train_loss"],
            checkpoint["valid_loss"],
        )
    else:
        return None
