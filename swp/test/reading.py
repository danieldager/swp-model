import torch
import torch.nn as nn
from torchdata.stateful_dataloader import StatefulDataLoader

from ..models.autoencoder import Bimodel, Unimodel


def test(
    test_loader: StatefulDataLoader,
    model: Unimodel | Bimodel,
    device: str | torch.device,
    model_name: str,
    verbose: bool,
):
    # TODO Robin docstring
    if isinstance(model, Unimodel) and not model.is_visual:
        raise ValueError("The model to train is not made to be tested with visual data")
    if isinstance(model, Bimodel):
        model.to_visual()
    # TODO code
