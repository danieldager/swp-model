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

    def forward(self, x, hidden):
        cell = torch.zeros_like(hidden)
        output, _ = self.lstm(x, (hidden, cell))  # (B, L, H)
        # print("d output", output.shape)
        logits = self.linear(output)              # (B, L, V)
        # print("d logits", logits.shape)
        
        return logits