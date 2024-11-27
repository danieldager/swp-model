import torch.nn as nn

""" NOTE:
should we batch the data by sequence length ?
does an embedding layer make the model less interpretable ? 
does it make the model more or less bio-realistic ?
"""

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # input_size + 1 = vocab size, because the # of embeddings
        self.embedding = nn.Embedding(input_size + 1, hidden_size)
        self.dropout = nn.Dropout(dropout)
        if num_layers == 1: dropout = 0
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, dropout=dropout)

    def forward(self, x):
        x = x.squeeze()
        # print("x", x)
        embedded = self.dropout(self.embedding(x))
        # print("embedded", embedded)
        _, hidden = self.rnn(embedded)
        
        return embedded, hidden
