import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout, shared_embedding):
        super(EncoderRNN, self).__init__()
        self.input_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = shared_embedding
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):  # (B, L)
        embedded = self.embedding(x)  # (B, L, H)
        _, hidden = self.rnn(embedded)  # (N, B, H)

        return hidden
