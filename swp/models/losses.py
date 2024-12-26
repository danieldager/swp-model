import torch


def alignment_loss(output, target, criterion, penalty):
    r"""
    Args:
        outputs: (output_len, vocab_size) tensor of logits
        targets: (target_len) tensor of target indices
    """
    # TODO more precise docstring
    output_len = output.size(0)
    target_len = target.size(0)

    # Initialize score matrix
    M = torch.zeros(output_len + 1, target_len + 1, device=output.device)

    # Initialize first row and column (penalty for skips)
    for i in range(output_len + 1):
        M[i, 0] = i * penalty
    for j in range(target_len + 1):
        M[0, j] = j * penalty

    # Fill matrix
    for i in range(1, output_len + 1):
        for j in range(1, target_len + 1):

            # Calculate match score using cross entropy
            score = criterion(output[i - 1].unsqueeze(0), target[j - 1].unsqueeze(0))

            # Take minimum of three possible operations:
            M[i, j] = torch.min(
                torch.stack(
                    [
                        M[i - 1, j - 1] + score,  # match/mismatch
                        M[i - 1, j] + penalty,  # skip in output
                        M[i, j - 1] + penalty,  # skip in target
                    ]
                )
            )

    # print("M[x,y]", M[output_len, target_len])
    # print("M", M)

    return M[output_len, target_len]


# Decoder forward pass using alignment loss ^^^
def alignment_forward(self, x, hidden, stop_token, target_len):
    # TODO docstring

    outputs = []

    # Set a limit for pred length
    max_length = target_len + 10

    # Forward pass loop
    for _ in range(max_length):
        output, hidden = self.rnn(x, hidden)

        # Generate output logits
        logits = self.fc(output)
        outputs.append(logits)

        # Check for stop token
        if torch.argmax(output) == stop_token:
            print("STOP")
            break

        # Pass output (not logits) to rnn
        x = output

    # Return logits (pred_len, vocab_size)
    outputs = torch.stack(outputs, dim=0)
    return outputs
