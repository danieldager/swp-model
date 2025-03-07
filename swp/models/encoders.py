from typing import Callable, Type, Union

import torch
import torch.fx
import torch.nn as nn
from torch.utils.model_zoo import load_url
from torchvision.models.feature_extraction import create_feature_extractor

from .cornet_r import HASH as HASH_R
from .cornet_r import CORnet_R
from .cornet_rt import HASH as HASH_RT
from .cornet_rt import CORnet_RT
from .cornet_s import HASH as HASH_S
from .cornet_s import CORnet_S
from .cornet_z import HASH as HASH_Z
from .cornet_z import CORnet_Z


class PhonemeEncoder(nn.Module):
    r"""Parent class for phoneme encoders.

    Passes the data through an embedding layer, then a dropout layer and finally
    a recurrent subnetwork.

    Args :
        `vocab_size` : number of phonemes
        `hidden_size` : phoneme embedding dimensions
        `num_layers` : number of layers in the recurrent subnetwork
        `dropout` : dropout rate

    Attributes:
        `vocab_size` : number of phonemes
        `hidden_size` : phoneme embedding dimensions
        `num_layers` : number of layers in the recurrent subnetwork
        `droprate` : dropout rate
        `embedding` : embedding layer
        `dropout` : dropout layer
        `recurrent` : recurrent subnetwork
    """

    # TODO add unroll doc

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
        embedded = self.embedding(inp)
        dropped = self.dropout(embedded)
        _, hidden = self.recurrent(dropped)
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


def cornet_loader(
    model_letter: str,
    pretrained: bool = True,
    map_location=None,
) -> nn.Module:
    r"""A function to load CORNet models.
    `model_letter` is expected to be among `R`, `RT`, `S` or `Z`, defining the CORNet submodel to load.
    `pretrained` : weither or not to load the pretrained weights
    `map_location` : see `torch.load`
    """

    model_code = model_letter.upper()
    model_class: Union[Type[nn.Module], Callable[[], nn.Module]]
    if model_code == "R":
        model_class = CORnet_R
        model_hash = HASH_R
    elif model_code == "RT":
        model_class = CORnet_RT
        model_hash = HASH_RT
    elif model_code == "S":
        model_class = CORnet_S
        model_hash = HASH_S
    elif model_code == "Z":
        model_class = CORnet_Z
        model_hash = HASH_Z
    else:
        raise ValueError(
            f"CORnet model letter(s) {model_letter} not recognized. Use R, RT, S or Z."
        )
    model = model_class()
    if pretrained:
        url = f"https://s3.amazonaws.com/cornet-models/cornet_{model_letter.lower()}-{model_hash}.pth"
        ckpt_data = load_url(url, map_location=map_location)
        state_dict = ckpt_data["state_dict"]
        new_state_dict = {k.removeprefix("module."): v for (k, v) in state_dict.items()}
        model.load_state_dict(new_state_dict)
    return model


class VisualEncoder(nn.Module):
    r"""Parent class for visual encoders.
    `cnn` is expected to have two outputs :
    - `"output"` that provides the CNN output
    - `"neural_code"` that provides the tensor to pass to `to_hidden` to generate the encoding

    Over forward pass, returns both the `class_prediction` output of `cnn`, and the added `hidden` encoding
    generated by feeding `to_hidden` the neural code from `cnn`.

    Args :
        `cnn` : the CNN that generates object classification and neural code
        `to_hidden` : the network that convert neural code to proper encoding
    Attributes :
        `cnn` : CNN part of the model
        `to_hidden` : subnetwork converting neural code to proper encoding
        `hidden_shape` : `torch.Size` object representing the output shape of `to_hidden`. Dimension containing `-1` is expected to be batch dimension.
    """

    def __init__(self, cnn: torch.fx.GraphModule, to_hidden: nn.Module) -> None:
        super().__init__()
        self.cnn = cnn
        self.to_hidden = to_hidden
        self.hidden_shape: torch.Size

    def forward(self, inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cnn_outs = self.cnn(inp)
        image_preds = cnn_outs["output"]
        hidden = self.to_hidden(cnn_outs["neural_code"])
        return image_preds, hidden

    def to_unroll(self):
        pass

    def to_chain(self):
        pass


class CorNetEncoder(VisualEncoder):
    r"""A visual encoder based on CORNet convolutional networks.

    Args :
        `hidden_size` : the size of the encoding to generate
        `cornet_model` : the CORNet model code of the model to use for the encoder
    Attributes :
        `cnn` : CNN part of the model
        `to_hidden` : subnetwork converting neural code to proper encoding
        `hidden_shape` : `torch.Size` object representing the output shape of `to_hidden`. The dimension containing `-1` is expected to be batch dimension.
    """

    def __init__(self, hidden_size: int, cornet_model: str):
        self.cnn_model = cornet_model
        self.hidden_size = hidden_size
        cornet = cornet_loader(cornet_model)

        return_nodes = {
            "decoder.flatten.view": "neural_code",
            "decoder.linear": "output",
        }

        cnn = create_feature_extractor(cornet, return_nodes=return_nodes)
        linear = nn.Linear(cnn.decoder.linear.in_features, hidden_size)
        self.hidden_shape = torch.Size((-1, hidden_size))
        super(CorNetEncoder, self).__init__(cnn=cnn, to_hidden=linear)
