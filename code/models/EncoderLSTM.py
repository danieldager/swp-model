import torch.nn as nn

""" NOTE:
should we batch the data by sequence length ?
does an embedding layer make the model less interpretable ? 
does it make the model more or less bio-realistic ?
"""

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers, dropout):
        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size + 1, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        if num_layers == 1: dropout = 0
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded)
        
        return hidden
