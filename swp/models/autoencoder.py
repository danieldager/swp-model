import torch
import torch.nn as nn

from ..utils.reshape import Reshaper
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
        `to_unroll` : sets the encoder to process input phonemes one by one
        `to_chain` : sets the encoder to process input phonemes in one single pass

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
        start_token_id: int,
    ):
        super(Unimodel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.is_auditory = isinstance(self.encoder, PhonemeEncoder)
        if isinstance(self.encoder, VisualEncoder):
            self.is_visual = True
            self.reshaper = Reshaper(
                self.encoder.hidden_shape, self.decoder.expected_hidden_shape
            )
        self.start_token_id = start_token_id
        self.bind()

    def forward(
        self, inp: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, None | torch.Tensor]:
        object_pred = None
        if isinstance(self.encoder, PhonemeEncoder):
            hidden = self.encoder(inp)
        else:
            object_pred, toreshape_hidden = self.encoder(inp)
            hidden = self.reshaper(to_reshape=toreshape_hidden)
        start = (
            torch.Tensor([self.start_token_id])
            .repeat((inp.size(0), 1))
            .to(inp.device, dtype=torch.int)
        )
        phoneme_prediction = self.decoder(start, hidden, target)
        return phoneme_prediction, object_pred

    def bind(self):
        if isinstance(self.encoder, PhonemeEncoder):
            self.decoder.embedding = self.encoder.embedding

    def to_unroll(self):
        self.encoder.to_unroll()

    def to_chain(self):
        self.encoder.to_chain()


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
        `to_unroll` : sets the auditory encoder to process input phonemes one by one
        `to_chain` : sets the auditory encoder to process input phonemes in one single pass

    Attributes:
        `audit_encoder` : auditory encoder part of the model
        `audit_encoder` : visual encoder part of the model
        `decoder` : decoder part of the model
        `mode` : current mode of execution of the model
        `start_tensor` : tensor passed to the decoder at the beginning of decoding
    """

    def __init__(
        self,
        audit_encoder: PhonemeEncoder,
        visual_encoder: VisualEncoder,
        decoder: PhonemeDecoder,
        start_token_id: int,
    ):
        super(Bimodel, self).__init__()
        self.audit_encoder = audit_encoder
        self.visual_encoder = visual_encoder
        self.reshaper = Reshaper(
            self.visual_encoder.hidden_shape, self.decoder.expected_hidden_shape
        )
        self.decoder = decoder
        self.start_token_id = start_token_id
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
            hidden = self.reshaper(to_reshape=toreshape_hidden)
        else:
            raise ValueError(
                f"Model is made for modes audio and visual, current mode {self.mode} is not recognized"
            )
        start = (
            torch.Tensor([self.start_token_id])
            .repeat((inp.size(0), 1))
            .to(inp.device, dtype=torch.int)
        )
        out = self.decoder(start, hidden, target)
        return out, object_pred

    def bind(self):
        self.decoder.embedding = self.audit_encoder.embedding

    def to_audio(self):
        self.mode = "audio"

    def to_visual(self):
        self.mode = "visual"

    def to_unroll(self):
        self.audit_encoder.to_unroll()

    def to_chain(self):
        self.audit_encoder.to_chain()
