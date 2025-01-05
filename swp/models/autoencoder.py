import torch
import torch.nn as nn

from .decoders import PhonemeDecoder
from .encoders import PhonemeEncoder, VisualEncoder


class Unimodel(nn.Module):
    r"""A Module interfacing either an auditory or a visual encoder with a vocal decoder.

    Over forward pass, returns both the `phoneme_prediction` of the decoder, and
    the added `object_pred` from the visual encoder, defaulting to `None` for auditory encoders.

    Args :
        `encoder` : instantiated encoder
        `decoder` : instantiated decoder

    Methods :
        `bind` : allows binding the embedding layers of the auditory encoder and vocal decoder

    Attributes:
        `encoder` : encoder part of the model
        `decoder` : decoder part of the model
        `is_auditory` : `True` if encoder part is a `PhonemeEncoder`
        `is_visual` : `True` if encoder part is a `VisualEncoder`
    """

    def __init__(
        self,
        encoder: PhonemeEncoder | VisualEncoder,
        decoder: PhonemeDecoder,
    ):
        super(Unimodel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.is_auditory = isinstance(self.encoder, PhonemeEncoder)
        self.is_visual = isinstance(self.encoder, VisualEncoder)
        self.bind()

    def forward(self, inp: torch.Tensor) -> tuple[torch.Tensor, None | torch.Tensor]:
        object_pred = None
        if isinstance(self.encoder, PhonemeEncoder):
            hidden = self.encoder(inp)
        else:
            object_pred, hidden = self.encoder(inp)
            # TODO maybe put reshaping there ?
        phoneme_prediction = self.decoder(hidden)
        return phoneme_prediction, object_pred

    def bind(self):
        if isinstance(self.encoder, PhonemeEncoder):
            # TODO proper code, after classes implementation
            # get encoder embedding layer
            # replace decoder embedding layer
            pass


class Bimodel(nn.Module):
    r"""A Module interfacing both auditory and visual encoders with a vocal decoder.

    Args :
        `audit_encoder` : instantiated auditory encoder
        `visual_encoder` : instantiated visual encoder
        `decoder` : instantiated vocal decoder

    Methods :
        `bind` : allows binding the embedding layers of the auditory encoder and vocal decoder
    """

    # TODO implement further functionalities

    def __init__(
        self,
        audit_encoder: PhonemeDecoder,
        visual_encoder: VisualEncoder,
        decoder: PhonemeDecoder,
    ):
        super(Bimodel, self).__init__()
        self.audit_encoder = audit_encoder
        self.visual_decoder = visual_encoder
        self.decoder = decoder
        self.bind()

    def forward(self, inp):
        # TODO find logic for multimodal input
        hidden = self.encoder(inp)
        out = self.decoder(hidden)
        return out

    def bind(self):
        # TODO check if there is a better way to load shared weights
        # some code to share wieghts
        pass
