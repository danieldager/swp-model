from typing import Callable, Type, Union

import cornet
import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url
from torchvision.models.feature_extraction import create_feature_extractor


class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(EncoderLSTM, self).__init__()
        self.input_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)  # (B, L, H)
        # print("\ne embedded", embedded.shape)
        _, (hidden, _) = self.lstm(embedded)  # (N, B, H)
        # print("e hidden", hidden.shape)

        return hidden


""" NOTE:
should we batch the data by sequence length ?
does an embedding layer make the model less interpretable ? 
does it make the model more or less bio-realistic ?
"""


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout):
        super(EncoderRNN, self).__init__()
        self.input_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        if num_layers == 1:
            dropout = 0.0
        self.rnn = nn.RNN(
            hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True
        )

    def forward(self, x):  # (B, L)
        # Original implementation
        print("\nx", x.shape)
        # embedded = self.dropout(self.embedding(x))
        embedded = self.embedding(x)  # (B, L, H)
        print("embedded", embedded.shape)
        # print("embedded+1", embedded.shape)
        _, hidden = self.rnn(embedded)  # (Layers, B, H)
        print("hidden", hidden.shape)

        # Permuted implementation
        # x = x.permute(1, 0)                        # (length, B)
        # embedded = self.dropout(self.embedding(x)) # (length, B, H)
        # _, hidden = self.rnn(embedded)             # (layers, B, H)

        # # Embedding layer
        # x = self.embedding(x)           # (B, length, H)
        # embedded = x.permute(1, 0, 2)   # (length, B, H)

        # # RNN layer
        # x = self.dropout(embedded)      # (length, B, H)
        # _, hidden = self.rnn(x)         # (layers, B, H)

        # inputs = targets -> embedded = target_embedded
        return hidden


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
