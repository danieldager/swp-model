import torch
import torch.nn as nn
from random import random

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout)
        if num_layers == 1: dropout = 0.0
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, target, tf_ratio):
        # Computes all time steps at once
        if x.shape[0] > 1: 
            outputs, _ = self.rnn(x, hidden)   # (L, B, H)
            outputs = self.linear(outputs)     # (L, B, V)

        # Loop for each timestep in target 
        else: 
            outputs = []
            for i in range(target.shape[0]):

                # For first timestep,       x = start  (1, B, H)
                if i == 0: x = x

                # If teacher forcing,       x = target (1, B, H)
                elif random() < tf_ratio: 
                    x = target[:, i:i+1, :]
        
                # No teacher forcing,       x = output (1, B, H)
                else: x = self.dropout(output)

                # Generate outputs          (1, B, H), (layers, B, H)
                output, hidden = self.rnn(x, hidden)

                # Generate logits           (1, B, V)
                logits = self.linear(output)
                outputs.append(logits)

            # Create output tensor          (B, L, V)
            outputs = torch.concat(outputs, dim=1)
        
        return outputs
