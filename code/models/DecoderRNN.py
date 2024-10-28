import torch.nn as nn

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        if num_layers == 1: dropout = 0
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # hidden_embedded = self.embedding(hidden)

        # NOTE: pass decoder outputs to rnn
        output, hidden = self.rnn(x, hidden)
        output = self.fc(output)

        # Two connected embedding layers ? 
        
        # NOTE: Don't need softmax with CrossEntropyLoss
        # output = F.softmax(output, dim=-1)

        # NOTE: do we want to include a learned start token ?
        # x = torch.zeros(self.batch_size, 1, self.hidden_size)

        # If we want to use teacher forcing, we need to iterate through the target sequence
        # Initialize decoder hidden state as encoder's final hidden state
        # decoder_hidden = encoder_hidden
        # for t in range(targets.size(1)):  # for each time step
        #     # Decoder forward pass (at each time step)
        #     decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

        #     # Compute loss (comparing decoder output with the true target at this time step)
        #     loss += loss_fn(decoder_output.squeeze(1), targets[:, t])

        #     # Optionally use teacher forcing (use the true target as the next input)
        #     teacher_force = random.random() < TEACHER_FORCING_RATIO
        #     decoder_input = targets[:, t].unsqueeze(1) if teacher_force else decoder_output.argmax(dim=2)

        return output
