import torch
import torch.nn as nn
from random import random

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_layers, dropout):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = vocab_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout)
        if num_layers == 1: dropout = 0.0
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        # Original implementation
        # if x.shape[0] > 1:
        print("x", x.shape)
        output, _ = self.rnn(x, hidden)    # (B, L, H)
        print("output", output.shape)
        logits = self.linear(output)       # (B, L, V)
        print("logits", logits.shape)

        # # Computes all time steps at once
        # elif x.shape[0] > 1: 
        #     output, _ = self.rnn(x, hidden)   # (L, B, H)
        #     logits = self.linear(output)      # (L, B, V)

        # Loop for each timestep in target 
        # else: 
        #     logits = []
        #     for i in range(target.shape[0]):

        #         # For first timestep       (1, B, H) x = start  // (B, 1, H)
        #         if i == 0: x = x

        #         # If teacher forcing       (1, B, H) x = target // (B, 1, H)
        #         elif random() < tf_ratio: 
        #             x = target[:, i:i+1, :]
        
        #         # No teacher forcing       (1, B, H) x = output // (B, 1, H)
        #         else: x = self.dropout(output)

        #         print("x", x.shape)
        #         # Generate outputs         (1, B, H), (layers, B, H)
        #         output, hidden = self.rnn(x, hidden) 
        #         print("output", output.shape)
        #         print("hidden", hidden.shape)

        #         # Generate logits          (1, B, V)
        #         logits.append(self.linear(output))

        #     # Create output tensor         (B, L, V)
        #     logits = torch.concat(logits, dim=1)
        #     print("logits", logits.shape)
        
        return logits
