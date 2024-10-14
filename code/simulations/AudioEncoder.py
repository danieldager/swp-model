import torch.nn as nn

""" NOTE:
should we batch the data by sequence length ?
does an embedding layer make the model less interpretable ? 
does it make the model more or less bio-realistic ?
"""

class AudioEncoder(nn.Module):
    def __init__(
            self, input_size, hidden_size, batch_size, num_layers=1, dropout=0.1
    ):
        super(AudioEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size + 1, hidden_size)
        self.encoder = nn.RNN(hidden_size, hidden_size, num_layers, dropout=dropout)

        # NOTE: if we use an embedding layer, we don't need one-hot vectors
        # NOTE: input size = vocab size, + 1 because of num_embeddings

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        _, hidden = self.encoder(embedded)

        # NOTE: should we parameterize the hidden state ?
        # hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        # NOTE: if we use an embedding layer, we don't need one-hot vectors
        # x = torch.argmax(x, dim=-1)
        
        return hidden
