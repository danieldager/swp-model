import torch
import torch.nn as nn
from random import random
from torch.nn import functional as F


class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_layers, dropout, shared_embedding):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = vocab_size
        self.num_layers = num_layers
        self.droupout = dropout

        self.embedding = shared_embedding
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x, hidden, cell, target, tf_ratio):
        length = target.size(1)
        logits = []

        for i in range(length):

            # First iteration
            if i == 0:
                x = self.embedding(x)

            # Teacher forcing
            elif tf_ratio > random():
                x = self.embedding(target[:, i].unsqueeze(1))

            # No teacher forcing
            else:
                x = self.embedding(output.argmax(dim=2))

            output, (hidden, cell) = self.lstm(x, (hidden, cell))

            # Use embedding weights to get logits
            output = output @ self.embedding.weight.T
            logits.append(output)

        logits = torch.cat(logits, dim=1)  # (B, L, V)

        return logits
