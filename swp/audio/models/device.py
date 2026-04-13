from __future__ import annotations

import torch


def select_device(device: str | None) -> torch.device:
    """Select a torch device.

    Args:
        device: explicit device string ('cpu', 'cuda', 'mps', etc.),
                or None to auto-select CUDA > MPS > CPU.

    Returns:
        torch.device
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")