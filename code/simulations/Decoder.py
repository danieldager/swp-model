import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(
            self, hidden_size, output_size, batch_size,
            num_layers=1, dropout=0.1
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(hidden_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # embedded = self.embedding(x)
        output, hidden = self.rnn(x, hidden)
        output = self.fc(output)

        # Two connected embedding layers ? 
        
        # NOTE: Don't need softmax with CrossEntropyLoss
        # output = F.softmax(output, dim=-1)

        # NOTE: do we want to include a learned start token ?
        # x = torch.zeros(self.batch_size, 1, self.hidden_size)

        return output
