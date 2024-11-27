import torch
import torch.nn as nn

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, embedded, tf_ratio:float=0.0):
        outputs = []
        # print("embedded", embedded)

        # Forward pass loop for each token in target
        for i in range(embedded.shape[0]):
            output, hidden = self.rnn(x, hidden)
            
            # Generate output logits
            logits = self.fc(output)
            outputs.append(logits)

            # Teacher forcing
            if torch.rand(1).item() < tf_ratio:
                x = embedded[i, :].unsqueeze(0).float()

            else:
                # print("output", output.shape) 
                x = output
            
            # print("x", x)
        
        # Return logits (pred_len, vocab_size)
        outputs = torch.stack(outputs, dim=0)
        return outputs
