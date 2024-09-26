import torch
import torch.nn as nn
import torch.optim as optim

from DataGenerator import DataGenerator
from AuditoryEncoder import AuditoryEncoder
from Decoder import Decoder

# Assume we have a dataset loader providing input sequences and target sequences
# inputs -> [batch_size, max_seq_len] (indices representing phonemes)
# targets -> [batch_size, max_seq_len] (target sequences of the same shape)

# Initialize model, loss function, optimizer
encoder = AuditoryEncoder(input_size=VOCAB_SIZE, hidden_size=HIDDEN_SIZE)
decoder = Decoder(hidden_size=HIDDEN_SIZE, output_size=VOCAB_SIZE)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    for inputs, targets in data_loader:
        # Zero gradients from previous step
        optimizer.zero_grad()

        # Encoder forward pass
        encoder_hidden = encoder(inputs)  # encoder_hidden is [num_layers, batch_size, hidden_size]

        # Initialize decoder input (often a start-of-sequence token)
        decoder_input = torch.zeros(batch_size, 1, dtype=torch.long)  # <SOS> token or similar
        
        # Initialize decoder hidden state as encoder's final hidden state
        decoder_hidden = encoder_hidden
        
        loss = 0

        # Iterate through each time step in the target sequence
        for t in range(targets.size(1)):  # for each time step
            # Decoder forward pass (at each time step)
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

            # Compute loss (comparing decoder output with the true target at this time step)
            loss += loss_fn(decoder_output.squeeze(1), targets[:, t])

            # Optionally use teacher forcing (use the true target as the next input)
            teacher_force = random.random() < TEACHER_FORCING_RATIO
            decoder_input = targets[:, t].unsqueeze(1) if teacher_force else decoder_output.argmax(dim=2)

        # Backpropagation
        loss.backward()
        
        # Update model parameters
        optimizer.step()

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item()/targets.size(1)}")