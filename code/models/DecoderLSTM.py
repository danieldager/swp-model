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
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x, hidden, cell, target, tf_ratio):
        length = target.size(1)
        logits = []

        for i in range(length):

            # Start token
            if i == 0:
                x = self.embedding(x)

            # Teacher forcing
            elif tf_ratio > random():
                x = self.embedding(target[:, i].unsqueeze(1))

            # No teacher forcing
            else:
                # if self.training:
                #     probs = F.softmax(output, dim=2)
                #     x = probs @ self.embedding.weight
                # else:
                #     x = self.embedding(output.argmax(dim=2))
                x = self.embedding(output.argmax(dim=2))

            # Forward pass
            x = self.dropout(x)
            output, (hidden, cell) = self.lstm(x, (hidden, cell))

            # Compute logits
            output = output @ self.embedding.weight.T
            logits.append(output)

        outputs = torch.cat(logits, dim=1)

        return outputs
