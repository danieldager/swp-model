"""The frozen SWP repeat model: a phoneme autoencoder, and ``get_model`` to build it.

Vendored from the parent SWP project so this package stands alone. Only the *auditory*
path is kept — every model the intervention pipeline touches is a ``Ua_*`` unimodel, and
the visual half (``CorNetEncoder``, ``VisualEncoder``, ``Bimodel``) pulled in the CORnet
submodule and torchvision for code that never ran here. Module and attribute names are
unchanged, so the original checkpoints load as-is.

Driving a built model — encoding a word, decoding from a state — lives next door in
``repeat_model``; this file only defines the architecture.
"""
from __future__ import annotations

from typing import Any, TypedDict

import torch
import torch.nn as nn

from intervention.paths import get_phoneme_to_id


# --------------------------------------------------------------------------- #
# Encoders
# --------------------------------------------------------------------------- #
class PhonemeEncoder(nn.Module):
    r"""Parent class for phoneme encoders.

    Passes the data through an embedding layer, then a dropout layer and finally
    a recurrent subnetwork.

    Args :
        `vocab_size` : number of phonemes
        `hidden_size` : phoneme embedding dimensions
        `num_layers` : number of layers in the recurrent subnetwork
        `dropout` : dropout rate

    Methods :
        `to_unroll` : sets the recurrent subnetwork to process input phonemes one by one
        `to_chain` : sets the recurrent subnetwork to process input phonemes in one single pass
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.droprate = dropout

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.recurrent: nn.RNNBase
        self.dropout = nn.Dropout(self.droprate)
        self.unrolling = False

    def forward(self, inp: torch.Tensor):
        if self.unrolling:
            out = self.unrolled_forward(inp)
        else:
            out = self.chained_forward(inp)
        return out

    def chained_forward(self, inp: torch.Tensor):
        out = self.dropout(self.embedding(inp))

        # TODO: make this an argument, it is only necessary
        # when we want LSTM hooks to return the correct (h, c)
        # when the embedding dataset is of variable lengths

        # packing the input to avoid padding
        # pad_idx = get_phoneme_to_id()["<PAD>"]
        # lengths = (inp != pad_idx).sum(dim=-1).cpu()
        # packed = pack_padded_sequence(
        #     out, lengths, batch_first=True, enforce_sorted=False
        # )
        # out = packed

        _, hidden = self.recurrent(out)
        return hidden

    def unrolled_forward(self, inp: torch.Tensor):
        embedded = self.embedding(inp)
        dropped = self.dropout(embedded)
        hidden = None
        for i in range(dropped.shape[-2]):
            rec_input = dropped[..., i : i + 1, :]
            if hidden is None:
                _, hidden = self.recurrent(rec_input)
            else:
                _, hidden = self.recurrent(rec_input, hidden)
        if hidden is None:
            raise ValueError("Time dimension is of length 0")
        return hidden

    def to_unroll(self):
        self.unrolling = True

    def to_chain(self):
        self.unrolling = False


class EncoderRNN(PhonemeEncoder):
    r"""An auditory encoder based on RNN recurrent networks, see `torch.nn.RNN`.
    RNN has `batch_first = True`.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super(EncoderRNN, self).__init__(vocab_size, hidden_size, num_layers, dropout)
        self.recurrent = nn.RNN(
            self.hidden_size, self.hidden_size, self.num_layers, batch_first=True
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return super().forward(inp)


class EncoderLSTM(PhonemeEncoder):
    r"""An auditory encoder based on LSTM recurrent networks, see `torch.nn.LSTM`.
    LSTM has `batch_first = True`.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super(EncoderLSTM, self).__init__(vocab_size, hidden_size, num_layers, dropout)
        self.recurrent = nn.LSTM(
            self.hidden_size, self.hidden_size, self.num_layers, batch_first=True
        )

    def forward(self, inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return super().forward(inp)


# --------------------------------------------------------------------------- #
# Decoders
# --------------------------------------------------------------------------- #
class PhonemeDecoder(nn.Module):
    r"""Parent class for phoneme decoders.

    Over forward pass, use `inp` as the starting phoneme embedding for phoneme generation.
    Phoneme are generated one by one and fedback to a recurrent subnetwork taking `hidden_data` as initial hidden states.
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
        self.eos_idx = get_phoneme_to_id()["<EOS>"]
        self.pad_idx = get_phoneme_to_id()["<PAD>"]

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
        start_tokens: torch.Tensor,
        hidden_state: Any,
        target: torch.Tensor,
    ) -> torch.Tensor:

        logits = []
        length = target.size(1)

        for i in range(length):

            # Start tokens
            if i == 0:
                curr = self.embedding(start_tokens)

            # Teacher forcing
            elif self.training and (
                self.tf_ratio > torch.rand([], generator=self.generator)
            ):
                curr = self.embedding(target[:, i].unsqueeze(1))
                curr = self.dropout(curr)

            # No teacher forcing
            else:
                curr = self.embedding(phoneme_pred.argmax(dim=-1))
                curr = self.dropout(curr)

            # Forward pass
            embed_pred, hidden_state = self.recurrent(curr, hidden_state)

            # Compute logits
            phoneme_pred = embed_pred @ self.embedding.weight.T
            logits.append(phoneme_pred)

        output = torch.cat(logits, dim=1)

        # Turn all tokens after first <EOS> into <PAD>
        if not self.training:
            preds = output.argmax(dim=-1)
            eos_mask = (preds == self.eos_idx).int()
            first_eos = eos_mask.argmax(dim=1, keepdim=True)
            mask = torch.arange(preds.shape[-1], device=preds.device)
            if len(preds.shape) > 1:
                mask = mask.unsqueeze(0)
            mask = mask > first_eos
            pad_one_hot = torch.zeros(self.vocab_size, device=preds.device)
            pad_one_hot[self.pad_idx] = 1
            output[mask] = pad_one_hot.expand_as(output[mask])

        return output


class DecoderLSTM(PhonemeDecoder):
    r"""A vocal decoder based on LSTM recurrent networks, see `torch.nn.LSTM`.
    LSTM has `batch_first = True`.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
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
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        tf_ratio: float,
    ):
        super(DecoderRNN, self).__init__(
            vocab_size,
            hidden_size,
            num_layers,
            dropout,
            tf_ratio,
        )
        self.recurrent = nn.RNN(
            self.hidden_size, self.hidden_size, self.num_layers, batch_first=True
        )
        hidden_shape = torch.Size(
            (
                self.recurrent.num_layers * (2 if self.recurrent.bidirectional else 1),
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


# --------------------------------------------------------------------------- #
# Autoencoder
# --------------------------------------------------------------------------- #
class Unimodel(nn.Module):
    r"""A Module interfacing an auditory encoder with a vocal decoder.

    The encoder and decoder share one embedding table (see `bind`), which is why
    `additive_intervention` can seed a delta from `model.encoder.embedding`.

    Args :
        `encoder` : instantiated auditory encoder
        `decoder` : instantiated vocal decoder
        `start_token_id` : id of the token passed to the decoder to start decoding
    """

    def __init__(
        self,
        encoder: PhonemeEncoder,
        decoder: PhonemeDecoder,
        start_token_id: int,
    ):
        super(Unimodel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.is_auditory = True
        self.is_visual = False
        self.start_token_id = start_token_id
        self.bind()

    def forward(
        self, inp: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        hidden = self.encoder(inp)
        start = (
            torch.Tensor([self.start_token_id])
            .repeat((inp.size(0), 1))
            .to(inp.device, dtype=torch.int)
        )
        phoneme_prediction = self.decoder(start, hidden, target)
        return phoneme_prediction, None

    def bind(self):
        self.decoder.embedding = self.encoder.embedding

    def to_unroll(self):
        self.encoder.to_unroll()

    def to_chain(self):
        self.encoder.to_chain()


# --------------------------------------------------------------------------- #
# Construction from a model name
# --------------------------------------------------------------------------- #
class ModelArgs(TypedDict):
    r"""Values required to build a model, parsed out of a `model_name`.

    `model_class` : `"Ua"` for a unimodel with an auditory encoder (the only one supported here)
    `recur_type` : `"LSTM"` or `"RNN"`
    `hidden_size` : hidden size of the network
    `num_layers` : number of recurrent layers
    `vocab_size` : size of the vocabulary
    `droprate` : dropout ratio
    `tf_ratio` : teacher forcing ratio
    `start_token_id` : id of the token to use as first input for decoding
    """

    model_class: str
    recur_type: str
    hidden_size: int
    num_layers: int
    vocab_size: int
    droprate: float
    tf_ratio: float
    start_token_id: int


def get_model_args(model_name: str) -> ModelArgs:
    r"""Parse a `model_name` such as ``Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1`` into the
    arguments needed to build it. See `ModelArgs`."""
    main_name = model_name.split("__")[0]  # a `__` suffix carried the visual encoder args
    name_split = main_name.split("_")
    str_args = {arg[0]: arg[1:] for arg in name_split[2:]}
    return ModelArgs(
        {
            "model_class": name_split[0],
            "recur_type": name_split[1],
            "hidden_size": int(str_args["h"]),
            "num_layers": int(str_args["l"]),
            "vocab_size": int(str_args["v"]),
            "droprate": float(str_args["d"]),
            "tf_ratio": float(str_args["t"]),
            "start_token_id": int(str_args["s"]),
        }
    )


def get_model(model_name: str) -> Unimodel:
    r"""Create the model corresponding to `model_name`.

    Only auditory unimodels (`Ua_*`) are supported: the visual encoders of the parent
    SWP project are not vendored here.
    """
    model_args = get_model_args(model_name)
    recur_type = model_args["recur_type"].upper()
    if recur_type == "LSTM":
        encoder_class: type[PhonemeEncoder] = EncoderLSTM
        decoder_class: type[PhonemeDecoder] = DecoderLSTM
    elif recur_type == "RNN":
        encoder_class, decoder_class = EncoderRNN, DecoderRNN
    else:
        raise NotImplementedError(
            f"Recurrent type {recur_type} is not currently supported"
        )

    if model_args["model_class"] != "Ua":
        raise ValueError(
            f"Only auditory unimodels ('Ua_*') are supported, got {model_args['model_class']!r} "
            f"from {model_name!r}. Visual and bimodal encoders live in the SWP project."
        )

    encoder = encoder_class(
        vocab_size=model_args["vocab_size"],
        hidden_size=model_args["hidden_size"],
        num_layers=model_args["num_layers"],
        dropout=model_args["droprate"],
    )
    decoder = decoder_class(
        vocab_size=model_args["vocab_size"],
        hidden_size=model_args["hidden_size"],
        num_layers=model_args["num_layers"],
        dropout=model_args["droprate"],
        tf_ratio=model_args["tf_ratio"],
    )
    return Unimodel(
        encoder=encoder, decoder=decoder, start_token_id=model_args["start_token_id"]
    )
