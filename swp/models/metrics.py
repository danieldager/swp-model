import torch

from ..utils.datasets import get_phoneme_to_id


def free_gen_errors(preds: torch.Tensor, target: torch.Tensor) -> int:
    r"""
    Returns the number of errors in the predicted phonemes while accounting
    for overgeneration
    """

    return int(torch.any(preds != target, dim=1).sum().item())


def classic_errors(preds: torch.Tensor, target: torch.Tensor) -> int:
    r"""
    Returns the number of errors in the predicted phonemes, truncate by
    the length of the target phonemes
    """

    mask = target != get_phoneme_to_id()["<PAD>"]
    return int(torch.any((preds != target) * mask, dim=1).sum().item())
