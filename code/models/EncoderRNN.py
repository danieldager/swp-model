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
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        if num_layers == 1: dropout = 0.0
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, dropout=dropout)

    def forward(self, x):
        # print("\ne x", x.shape)
        embedded = self.dropout(self.embedding(x))
        embedded = embedded.permute(1, 0, 2)
        # print("e embedded", embedded.shape)
        o, hidden = self.rnn(embedded)
        # print("e o", o.shape)
        # print("e hidden", hidden.shape)
        
        return hidden
