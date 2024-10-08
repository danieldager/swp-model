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
        self.dropout = dropout

        self.decoder = nn.RNN(hidden_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # NOTE: do we want to include a learned start token ?
        # x = torch.zeros(self.batch_size, 1, self.hidden_size)
        output, _ = self.decoder(x, hidden)
        output = self.fc(output)
        output = F.softmax(output, dim=-1)

        return output
