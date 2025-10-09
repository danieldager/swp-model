import torch

from ...models.autoencoder import Bimodel, Unimodel
from ..paths import get_weights_dir


def save_weights(
    model_name: str,
    train_name: str,
    model: Unimodel | Bimodel,
    epoch: int,
    checkpoint: int | None = None,
) -> None:
    r"""Save weights of a model for a given training procedure."""
    save_dir = get_weights_dir() / model_name / train_name
    save_dir.mkdir(exist_ok=True, parents=True)
    epoch_str = f"{epoch}"
    if checkpoint is not None:
        epoch_str = f"{epoch_str}_{checkpoint}"
    model_path = save_dir / f"{epoch_str}.pth"
    torch.save(model.state_dict(), model_path)


def load_weights(
    model: Unimodel | Bimodel,
    model_name: str,
    train_name: str,
    checkpoint: str,
    device: torch.device,
) -> None:
    r"""Load the weights of a model for a given training procedure at a specific
    epoch and potential checkpoint.
    """
    save_dir = get_weights_dir() / model_name / train_name
    model_path = save_dir / f"{checkpoint}.pth"
    model.to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.bind()
