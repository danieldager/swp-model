import torch
import torch.nn as nn
import torch.optim as optim

from DataGenerator import DataGenerator
from AudioEncoder import AudioEncoder
from Decoder import Decoder

gen = DataGenerator(word_count=10000, batch_size=30)
data, SEQ_LENGTH, VOCAB_SIZE = gen.generate_phonemes()

# Hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 30
HIDDEN_SIZE = 256
DROPOUT = 0.0
NUM_LAYERS = 1
LEARNING_RATE = 1e-3
TEACHER_FORCING_RATIO = 0.5

# Initialize models, loss function, optimizer
encoder = AudioEncoder(
    input_size=VOCAB_SIZE, hidden_size=HIDDEN_SIZE, batch_size=BATCH_SIZE,
    num_layers=NUM_LAYERS, dropout=DROPOUT
)

decoder = Decoder(
    hidden_size=HIDDEN_SIZE, output_size=VOCAB_SIZE, batch_size=BATCH_SIZE,
    num_layers=NUM_LAYERS, dropout=DROPOUT
)

# Test Models
x = torch.zeros(BATCH_SIZE, SEQ_LENGTH, VOCAB_SIZE, dtype=torch.int)
print(f"Input: {x.shape}")

hidden = encoder(x)
print(f"Encoder hidden: {hidden.shape}")

start = torch.zeros(BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)
output = decoder(start, hidden)
print(f"Decoder output: {output.shape}")


loss_fn = nn.CrossEntropyLoss() # might want to try focal loss to deal with class imbalance
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    for inputs, targets in data:

        # Zero gradients from previous step
        optimizer.zero_grad()

        # Encoder forward pass
        encoder_hidden = encoder(inputs) # [num_layers, seq_length (not batch_size?), hidden_size]

        # Initialize decoder input
        decoder_input = torch.zeros(BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)
        
        # Decoder forward pass
        decoder_output = decoder(decoder_input, encoder_hidden)

        # Compute loss and backpropagate
        loss = loss_fn(decoder_output.squeeze(1), targets.float())
        loss.backward()
        optimizer.step()

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

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item()/targets.size(1)}")