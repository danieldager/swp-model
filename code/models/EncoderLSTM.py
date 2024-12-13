import torch.nn as nn


class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout, shared_embedding):
        super(EncoderLSTM, self).__init__()
        self.input_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = shared_embedding
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))  # (B, L, H)
        _, (hidden, cell) = self.lstm(embedded)  # (N, B, H)

        return hidden, cell
