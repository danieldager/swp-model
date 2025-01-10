import torch
import torch.nn as nn

from ..utils.models import can_reshape_magic, reshape_magic
from .decoders import PhonemeDecoder
from .encoders import PhonemeEncoder, VisualEncoder


class Unimodel(nn.Module):
    r"""A Module interfacing either an auditory or a visual encoder with a vocal decoder.

    Over forward pass, returns both the `phoneme_prediction` of the decoder, and
    the added `object_pred` from the visual encoder, defaulting to `None` for auditory encoders.

    Args :
        `encoder` : instantiated encoder
        `decoder` : instantiated decoder
        `start_tensor` : tensor to be passed to the decoder to start decoding

    Methods :
        `bind` : allows binding the embedding layers of the auditory encoder and vocal decoder

    Attributes:
        `encoder` : encoder part of the model
        `decoder` : decoder part of the model
        `is_auditory` : `True` if encoder part is a `PhonemeEncoder`
        `is_visual` : `True` if encoder part is a `VisualEncoder`
        `start_tensor` : tensor passed to the decoder at the beginning of decoding
    """

    def __init__(
        self,
        encoder: PhonemeEncoder | VisualEncoder,
        decoder: PhonemeDecoder,
        start_tensor: torch.Tensor,
    ):
        super(Unimodel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.is_auditory = isinstance(self.encoder, PhonemeEncoder)
        if isinstance(self.encoder, VisualEncoder):
            self.is_visual = True
            if not can_reshape_magic(
                self.encoder.hidden_shape, self.decoder.expected_hidden_shape
            ):
                raise ValueError(
                    f"Visual encoder outputs cannot be reshaped as auditory decoder hidden state"
                )
        self.start_tensor = start_tensor
        self.bind()

    def forward(
        self, inp: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, None | torch.Tensor]:
        object_pred = None
        if isinstance(self.encoder, PhonemeEncoder):
            hidden = self.encoder(inp)
        else:
            object_pred, toreshape_hidden = self.encoder(inp)
            hidden = reshape_magic(
                toreshape_hidden,
                self.encoder.hidden_shape,
                self.decoder.expected_hidden_shape,
            )
        start = self.start_tensor.repeat((inp.size(0), 1, 1)).to(
            inp.device, dtype=torch.int
        )
        phoneme_prediction = self.decoder(start, hidden, target)
        return phoneme_prediction, object_pred

    def bind(self):
        if isinstance(self.encoder, PhonemeEncoder):
            self.decoder.embedding = self.encoder.embedding


class Bimodel(nn.Module):
    r"""A Module interfacing both auditory and visual encoders with a vocal decoder.

    Args :
        `audit_encoder` : instantiated auditory encoder
        `visual_encoder` : instantiated visual encoder
        `decoder` : instantiated vocal decoder
        `start_tensor` : tensor to be passed to the decoder to start decoding

    Methods :
        `bind` : allows binding the embedding layers of the auditory encoder and vocal decoder
        `to_audio` : switch the model in audio input mode
        `to_visual` : switch the model in visual input mode

    Attributes:
        `audit_encoder` : auditory encoder part of the model
        `audit_encoder` : visual encoder part of the model
        `decoder` : decoder part of the model
        `mode` : current mode of execution of the model
        `start_tensor` : tensor passed to the decoder at the beginning of decoding
    """

    def __init__(
        self,
        audit_encoder: PhonemeDecoder,
        visual_encoder: VisualEncoder,
        decoder: PhonemeDecoder,
        start_tensor: torch.Tensor,
    ):
        super(Bimodel, self).__init__()
        self.audit_encoder = audit_encoder
        self.visual_encoder = visual_encoder
        if not can_reshape_magic(
            self.visual_encoder.hidden_shape, self.decoder.expected_hidden_shape
        ):
            raise ValueError(
                f"Visual encoder outputs cannot be reshaped as auditory decoder hidden state"
            )
        self.decoder = decoder
        self.start_tensor = start_tensor
        self.bind()
        self.mode = "audio"

    def forward(
        self, inp: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, None | torch.Tensor]:
        # TODO find logic for multimodal input
        object_pred = None
        if self.mode == "audio":
            hidden = self.audit_encoder(inp)
        elif self.mode == "visual":
            object_pred, toreshape_hidden = self.visual_encoder(inp)
            hidden = reshape_magic(
                toreshape_hidden,
                self.visual_encoder.hidden_shape,
                self.decoder.expected_hidden_shape,
            )
        else:
            raise ValueError(
                f"Model is made for modes audio and visual, current mode {self.mode} is not recognized"
            )
        start = self.start_tensor.repeat((inp.size(0), 1, 1)).to(
            inp.device, dtype=torch.int
        )
        out = self.decoder(start, hidden, target)
        return out, object_pred

    def bind(self):
        self.decoder.embedding = self.audit_encoder.embedding

    def to_audio(self):
        self.mode = "audio"

    def to_visual(self):
        self.mode = "visual"
