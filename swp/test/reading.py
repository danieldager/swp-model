import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.autoencoder import Bimodel, Unimodel


def test(
    test_loader: DataLoader,
    model: Unimodel | Bimodel,
    device: str | torch.device,
    model_name: str,
    verbose: bool,
):
    # TODO docstring
    if isinstance(model, Unimodel) and not model.is_visual:
        raise ValueError("The model to train is not made to be tested with visual data")
    if isinstance(model, Bimodel):
        model.to_visual()
    # TODO code
