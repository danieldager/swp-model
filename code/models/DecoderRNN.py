import torch
import torch.nn as nn

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout, tf_ratio, embedding):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embedding = embedding
        self.dropout = nn.Dropout(dropout)
        if num_layers == 1: dropout = 0.0
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, target, tf_ratio):
        outputs = []
        # print("\nd x", x.shape)
        # print("d hidden", hidden.shape)
        # print("d target", target.shape)
        # print(target)

        # Forward pass loop for each token in target
        for i in range(target.shape[1]):
            
            # Initial time step
            if i == 0:
                x = self.dropout(self.embedding(x))
                # print("\nd1 x", x.shape)

            # Teacher forcing
            elif torch.rand(1).item() < tf_ratio:
                x = target[:, i].unsqueeze(0)
                # print("\nd2 x", x.shape)
                x = self.dropout(self.embedding(x))
                # print("d2 x", x.shape)
            
            # No teacher forcing
            else: 
                x = self.dropout(output)
                # print("\nd3 x", x.shape)
            
            # Generate outputs
            output, hidden = self.rnn(x, hidden)
            # print("\nd output", output.shape)
            # print("d hidden", hidden.shape)

            # Generate logits
            logits = self.fc(output)
            outputs.append(logits)
        
        # Return logits (seq_length, vocab_size)
        # print("\nd outputs", len(outputs))
        # print(outputs[0].shape)
        outputs = torch.stack(outputs, dim=0)
        # print(outputs.shape)

        return outputs
