from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


def create_embeddings_LSTM_hook(
    embeddings: dict[str, list[np.ndarray]],
    is_batched: bool = True,
    store_out: bool = False,
    num_layers: int = 1,
) -> Callable[
    [
        nn.Module,
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
    ],
    None,
]:
    """
    Creates a hook function to capture the final hidden state of an LSTM layer.

    Args:
        embeddings (dict[str, list[np.ndarray]]): Dictionary to store the embeddings.
        is_batched (bool): Whether the input is batched.
        store_out (bool): Whether to store the "output" tensor of the LSTM.
        num_layers (int): Number of layers in the LSTM.

    Returns:
        Callable: A hook function that captures the final hidden state and output of the LSTM.
    """
    MAX_PAD = 20

    def embeddings_LSTM_hook(
        module: nn.Module,
        inputs: tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        output: tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
    ):
        """Hook function to capture the final hidden state."""
        out, (h, c) = output  # h, c (L, B, H)

        if store_out and isinstance(out, PackedSequence):
            out, _ = pad_packed_sequence(out, batch_first=True)

        if not is_batched:
            out = out.unsqueeze(0)
            h = h.unsqueeze(1)
            c = c.unsqueeze(1)

        # Pad output to allow for concatenation
        if store_out:
            out = out.detach().cpu().numpy()
            B, T, H = out.shape
            padded_out = np.zeros((B, MAX_PAD, H))
            padded_out[:, :T, :] = out
            embeddings["Out"].append(padded_out)

        h = h.squeeze(0)
        h = h.detach().cpu().numpy()
        embeddings["Hidden"].append(h)

        c = c.squeeze(0)
        c = c.detach().cpu().numpy()
        embeddings["Cell"].append(c)

    return embeddings_LSTM_hook
