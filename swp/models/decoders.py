from random import random
from typing import Any

import torch
import torch.nn as nn


class PhonemeDecoder(nn.Module):
    r"""Parent class for phoneme encoders.

    Over forward pass, use `inp` as the starting phoneme embedding for phoneme generation.
    Phoneme are generated one by one and fedback to a recurrent subnetwork taking `hidden_data`as initial hidden states.
    Recurrent subnetwork generate phoneme embeddings that are decoded to be outputed,
    then re-embedded and passed through a dropout layer before being fedback.
    Generated sequence length is matching the length of `target`.

    Args :
        `vocab_size` : number of phonemes
        `hidden_size` : phoneme embedding dimensions
        `num_layers` : number of layers in the recurrent subnetwork
        `dropout` : dropout rate
        `tf_ratio` : teacher forcing rate
        `generator` : generator used to control teacher forcing. If `None`, then a generator is initialized deterministically.

    Attributes:
        `vocab_size` : number of phonemes
        `hidden_size` : phoneme embedding dimensions
        `num_layers` : number of layers in the recurrent subnetwork
        `droprate` : dropout rate
        `tf_ratio` : teacher forcing rate
        `embedding` : embedding layer
        `dropout` : dropout layer
        `recurrent` : recurrent subnetwork
        `generator` : generator used to control teacher forcing
        `expected_hidden_shape` : size or tuple of sizes required by the recurrent subnetwork hidden data. Batch dimension(s) contain `-1`
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        tf_ratio: float,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.droprate = dropout
        self.tf_ratio = tf_ratio

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.dropout = nn.Dropout(self.droprate)
        self.recurrent: nn.RNNBase
        self.expected_hidden_shape: torch.Size | tuple[torch.Size, ...]
        if generator is not None:
            self.generator = generator
        else:
            self.generator = torch.Generator().manual_seed(42)

    def forward(
        self,
        inp: torch.Tensor,
        hidden_state: Any,
        target: torch.Tensor,
    ) -> torch.Tensor:
        length = target.size(1)
        logits = []

        for i in range(length):

            # Start token
            if i == 0:
                curr = self.embedding(inp)

            # Teacher forcing
            elif self.tf_ratio > torch.rand(1, generator=self.generator):
                curr = self.embedding(target[:, i].unsqueeze(1))
                curr = self.dropout(curr)

            # No teacher forcing
            else:
                # if self.training:
                #     probs = F.softmax(phoneme_pred, dim=2)
                #     curr = probs @ self.embedding.weight
                curr = self.embedding(phoneme_pred.argmax(dim=2))
                curr = self.dropout(curr)

            # Forward pass
            embed_pred, hidden_state = self.recurrent(curr, hidden_state)

            # Compute logits
            phoneme_pred = embed_pred @ self.embedding.weight.T
            logits.append(phoneme_pred)

        output = torch.cat(phoneme_pred, dim=1)

        return output


class DecoderLSTM(PhonemeDecoder):
    r"""A vocal decoder based on LSTM recurrent networks, see `torch.nn.LSTM`.
    LSTM has `batch_first = True`.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_layers: int,
        dropout: float,
        tf_ratio: float,
    ):
        super(DecoderLSTM, self).__init__(
            vocab_size, hidden_size, num_layers, dropout, tf_ratio
        )
        self.recurrent = nn.LSTM(
            self.hidden_size, self.hidden_size, self.num_layers, batch_first=True
        )

        hidden_shape = torch.Size(
            (
                self.recurrent.num_layers * (2 if self.recurrent.bidirectional else 1),
                -1,
                (
                    self.recurrent.proj_size
                    if self.recurrent.proj_size > 0
                    else self.recurrent.hidden_size
                ),
            )
        )
        cell_shape = torch.Size(
            (
                self.recurrent.num_layers * (2 if self.recurrent.bidirectional else 1),
                -1,
                self.recurrent.hidden_size,
            )
        )
        self.expected_hidden_shape = (hidden_shape, cell_shape)

    def forward(
        self,
        inp: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(inp, hidden_state, target)


class DecoderRNN(PhonemeDecoder):
    r"""A vocal decoder based on RNN recurrent networks, see `torch.nn.RNN`.
    RNN has `batch_first = True`.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_layers: int,
        dropout: float,
        tf_ratio: float,
    ):
        super(DecoderRNN, self).__init__(
            vocab_size, hidden_size, num_layers, dropout, tf_ratio
        )
        self.recurrent = nn.RNN(
            self.hidden_size, self.hidden_size, self.num_layers, batch_first=True
        )
        hidden_shape = torch.Size(
            (
                self.recurrent.num_layers * (2 if self.bidirectional else 1),
                -1,
                self.recurrent.hidden_size,
            )
        )
        self.expected_hidden_shape = hidden_shape

    def forward(
        self,
        inp: torch.Tensor,
        hidden_state: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(inp, hidden_state, target)
