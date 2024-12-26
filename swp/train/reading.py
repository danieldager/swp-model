import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train(
    train_loader: DataLoader,
    valid_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: str | torch.device,
    model_name: str,
    num_epochs: int,
    verbose: bool,
):
    # TODO docstring
    pass
