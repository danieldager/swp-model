from typing import Callable, Type, Union

import cornet
import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url
from torchvision.models.feature_extraction import create_feature_extractor


def cornet_loader(
    model_letter: str,
    pretrained: bool = True,
    map_location=None,
) -> nn.Module:
    if model_letter.upper() not in {"R", "RT", "S", "Z"}:
        raise ValueError(
            f"CORnet model letter(s) {model_letter} not recognized. Use R, RT, S or Z."
        )
    model_class: Union[Type[nn.Module], Callable[[], nn.Module]] = getattr(
        cornet, f"CORnet_{model_letter.upper()}"
    )
    model = model_class()
    if pretrained:
        url = f"https://s3.amazonaws.com/cornet-models/cornet_{model_letter.lower()}-{getattr(cornet, f'HASH_{model_letter.upper()}')}.pth"
        ckpt_data = load_url(url, map_location=map_location)
        state_dict = ckpt_data["state_dict"]
        new_state_dict = {k.removeprefix("module."): v for (k, v) in state_dict.items()}
        model.load_state_dict(new_state_dict)
    return model


class EncoderCNN(nn.Module):
    def __init__(self, hidden_size, cornet_model):
        super(EncoderCNN, self).__init__()
        cornet = cornet_loader(cornet_model)

        return_nodes = {
            "decoder.flatten.view": "neural_code",
            "decoder.linear": "output",
        }

        self.cnn = create_feature_extractor(cornet, return_nodes=return_nodes)
        self.to_hidden = nn.Linear(self.cnn.decoder.linear.in_features, hidden_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cnn_outs = self.cnn(x)
        image_class_prediction = cnn_outs["output"]
        hidden = self.to_hidden(cnn_outs["neural_code"])

        return image_class_prediction, hidden
