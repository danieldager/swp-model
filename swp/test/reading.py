import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def test(
    test_loader: DataLoader,
    model: nn.Module,
    device: str | torch.device,
    model_name: str,
    verbose: bool,
):
    # TODO docstring
    pass
