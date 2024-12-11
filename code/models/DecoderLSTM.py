import torch
import torch.nn as nn

class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_layers):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = vocab_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, length):
        cell = torch.zeros_like(hidden)             # (N, B, H)
        
        # All time steps at once
        # output, _ = self.lstm(x, (hidden, cell))  # (B, L, H)
        # logits = self.linear(output)              # (B, L, V)

        # Get output for the first time step
        # print("x", x.shape)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        # print("output", output.shape)

        # Get logits for the first time step, initialize list
        logits = [self.linear(output)]

        for i in range(1, length):
            x = output                  # (B, L, H)
            # print("x", x.shape)
            output, (hidden, cell) = self.lstm(x, (hidden, cell))
            # print("output", output.shape)
            logits.append(self.linear(output))

            # x = output[:, i:i+1, :] # slice index to get (B, 1, H)
        
        logits = torch.cat(logits, dim=1)           # (B, L, V)
        # print("logits", logits.shape, '\n')
        
        return logits