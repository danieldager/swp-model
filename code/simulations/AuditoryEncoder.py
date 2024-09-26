import torch.nn as nn

class AuditoryEncoder(nn.Module):
    def __init__(
            self, input_size, hidden_size,
            num_layers=1, dropout=0.1
            ):
        super(AuditoryEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.encoder = nn.RNN(hidden_size, hidden_size, num_layers, dropout=dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.encoder(embedded)
        return hidden

