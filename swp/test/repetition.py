import torch
import torch.nn as nn

from ..datasets.phonemes import Phonemes
from ..models.decoders import DecoderLSTM, DecoderRNN
from ..models.encoders import EncoderLSTM, EncoderRNN
from ..plots import confusion_matrix, error_plots
from ..utils.models import load_weigths
from ..utils.paths import get_weights_dir
from .core import calculate_errors


def test_repetition(P: Phonemes, model: str, device) -> list:
    print(f"\nTesting model: {model}")
    # Unpack parameters from model name
    e, h, l, d, t, r = [p[1:] for p in model.split("_")]
    print(f"Parameters: e={e} h={h} l={l} d={d} t={t} l={r}")
    n_epochs, h_size, n_layers, dropout = int(e), int(h), int(l), float(d)

    # Unpack variables from Phonemes class
    test_data = P.test_data
    vocab_size = P.vocab_size
    index_to_phone = P.index_to_phone
    test_dataloader = P.test_dataloader

    # Sort index_to_phone alphabetically
    index_to_phone = {
        i: p for i, p in sorted(index_to_phone.items(), key=lambda x: x[1])
    }

    # Add checkpoints for proper iteration
    checkpoints = [f"1_{i}" for i in range(1, 11)]
    epochs = checkpoints + list(range(1, n_epochs + 1))

    # For testing only the final epoch
    epochs = ["30"]

    dataframes = []
    for epoch in epochs:

        """LOAD MODEL"""
        model_weights_dir = get_weights_dir() / model

        embedding = nn.Embedding(vocab_size, h_size)

        encoder = EncoderLSTM(
            vocab_size,
            h_size,
            n_layers,
            dropout,
            embedding,
        ).to(device)

        decoder = DecoderLSTM(
            h_size,
            vocab_size,
            n_layers,
            dropout,
            embedding,
        ).to(device)

        load_weigths(model_weights_dir, embedding, encoder, decoder, epoch, device)

        """ TESTING LOOP """
        outputs = []
        test_data = test_data.copy()

        deletions, insertions, substitutions = [], [], []
        error_indices, sequence_lengths = [], []
        edit_distances = []

        # Initialize the confusion matrix
        confusions = {}
        for t in index_to_phone.values():
            confusions[t] = {p: 0 for p in index_to_phone.values()}

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            for inputs, target in test_dataloader:
                inputs = inputs.to(device)
                target = target.to(device)

                hidden, cell = encoder(inputs)
                start = torch.zeros(1, 1, dtype=int, device=device)
                output = decoder(start, hidden, cell, target, 0)

                output = torch.argmax(output, dim=2)
                output = output.squeeze().cpu().tolist()
                target = target.squeeze().cpu().tolist()

                # Calculate errors TODO: Refactor this
                errors = calculate_errors(output, target)
                deletions.append(errors["dels"])
                insertions.append(errors["inss"])
                substitutions.append(errors["subs"])
                edit_distances.append(errors["total"])
                error_indices.append(errors["indices"])
                sequence_lengths.append(errors["length"])

                # Convert indices to phonemes
                output = [index_to_phone[i] for i in output]
                target = [index_to_phone[i] for i in target]

                # Tabulate confusion between output and target
                if len(target) == len(output):
                    for t, p in zip(target, output):
                        confusions[t][p] += 1

                outputs.append(output[:-1])

        test_data["Prediction"] = outputs
        test_data["Deletions"] = deletions
        test_data["Insertions"] = insertions
        test_data["Substitutions"] = substitutions
        test_data["Edit Distance"] = edit_distances
        test_data["Error Indices"] = error_indices
        test_data["Sequence Length"] = sequence_lengths

        error_plots(test_data, model, epoch)
        confusion_matrix(confusions, model, epoch)

        dataframes.append(test_data)

    return dataframes
