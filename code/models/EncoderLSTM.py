import torch.nn as nn

class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(EncoderLSTM, self).__init__()
        self.input_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)            # (B, L, H)
        _, (hidden, _) = self.lstm(embedded)    # (N, B, H)
        
        return hidden
