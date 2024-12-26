import os
import random

import numpy as np
import torch
import torch.backends.cudnn
import torch.version


def seed_everything(seed=42) -> None:
    r"""Seeds Python random module, numpy and torch.
    Also enables CUDA determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        if torch.version.cuda is None:
            raise RuntimeError("Cuda marked as available but no version is provided")
        else:
            cuda_version = list(map(int, torch.version.cuda.split(".")))
            if cuda_version[0] == 10:
                if cuda_version[1] == 1:
                    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
                if cuda_version[1] > 1:
                    os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
                        ":4096:8"  # setting CUBLAS_WORKSPACE_CONFIG=:16:8 also works
                    )
            elif cuda_version[0] > 10:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
                    ":4096:8"  # setting CUBLAS_WORKSPACE_CONFIG=:16:8 also works
                )
            else:
                raise RuntimeError(
                    f"CUDA version {torch.version.cuda} might be too old to support deterministic behavior"
                )


def set_device() -> torch.device:
    r"""Select available device, with priority order CUDA, then MPS and finally CPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"Using CUDA device: {device_name}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device
