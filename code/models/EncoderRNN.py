import torch.nn as nn

""" NOTE:
should we batch the data by sequence length ?
does an embedding layer make the model less interpretable ? 
does it make the model more or less bio-realistic ?
"""

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout):
        super(EncoderRNN, self).__init__()
        self.input_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        if num_layers == 1: dropout = 0.0
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, dropout=dropout)

    def forward(self, x):               # (B, length)
        x = self.embedding(x)           # (B, length, H) 
        embedded = x.permute(1, 0, 2)   # (length, B, H)
        x = self.dropout(embedded)
        _, hidden = self.rnn(x)         # (layers, B, H)

        # inputs = targets -> embedded = target_embedded
        return hidden, embedded
