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
        
        # vocab size = input_size + 1 because of num_embeddings
        self.embedding = nn.Embedding(input_size + 1, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, dropout=dropout)


    def forward(self, x):
        # three tied embedding layers
        embedded = self.dropout(self.embedding(x))
        _, hidden = self.rnn(embedded)

        # NOTE: should we parameterize the hidden state ?
        # hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        
        return hidden
