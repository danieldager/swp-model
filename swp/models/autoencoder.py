import torch.nn as nn


class Unimodel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super(Unimodel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bind()

    def forward(self, input):
        hidden = self.encoder(input)
        out = self.decoder(hidden)
        return out

    def bind(self):
        # TODO check if there is a better way to load shared weights
        # some code to share wieghts
        pass


class Bimodel(nn.Module):
    def __init__(
        self, audit_encoder: nn.Module, visual_encoder: nn.Module, decoder: nn.Module
    ):
        super(Bimodel, self).__init__()
        self.audit_encoder = audit_encoder
        self.visual_decoder = visual_encoder
        self.decoder = decoder
        self.bind()

    def forward(self, input):
        hidden = self.encoder(input)
        out = self.decoder(hidden)
        return out

    def bind(self):
        # TODO check if there is a better way to load shared weights
        # some code to share wieghts
        pass
