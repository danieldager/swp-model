import torch
import torch.nn as nn
import torch.optim as optim

class AudioEncoder(nn.Module):
    def __init__(
            self, input_size, hidden_size, batch_size,
            num_layers=1, dropout=0.1
        ):

        super(AudioEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout

        # does an embedding layer make the model less interpretable ? 
        # does it make the model more or less bio-realistic ?
        # if we use an embedding layer, we don't need one-hot vectors
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.encoder = nn.RNN(hidden_size, hidden_size, num_layers, dropout=dropout)

    def forward(self, x):
        # should we parameterize the hidden state ?
        # hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

        # if we use an embedding layer, we don't need one-hot vectors
        x = torch.argmax(x, dim=-1)

        embedded = self.embedding(x)
        _, hidden = self.encoder(embedded)
        return hidden
