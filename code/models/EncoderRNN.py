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
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True)

    def forward(self, x):               # (B, L)
        # Original implementation
        print("\nx", x.shape)
        # embedded = self.dropout(self.embedding(x))
        embedded = self.embedding(x)                 # (B, L, H)
        print("embedded", embedded.shape)
        # print("embedded+1", embedded.shape)
        _, hidden = self.rnn(embedded)             # (Layers, B, H)
        print("hidden", hidden.shape)

        # Permuted implementation
        # x = x.permute(1, 0)                        # (length, B)
        # embedded = self.dropout(self.embedding(x)) # (length, B, H)
        # _, hidden = self.rnn(embedded)             # (layers, B, H)

        # # Embedding layer
        # x = self.embedding(x)           # (B, length, H) 
        # embedded = x.permute(1, 0, 2)   # (length, B, H)
        
        # # RNN layer
        # x = self.dropout(embedded)      # (length, B, H)
        # _, hidden = self.rnn(x)         # (layers, B, H)

        # inputs = targets -> embedded = target_embedded
        return hidden
