import torch

from ..datasets.phonemes import Phonemes
from ..models.decoders import DecoderLSTM, DecoderRNN
from ..models.encoders import EncoderLSTM, EncoderRNN
from ..plots import confusion_matrix, error_plots, primacy_recency
from ..utils.models import load_weigths
from ..utils.paths import get_checkpoint_dir
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
    phoneme_stats = P.phoneme_stats
    index_to_phone = P.index_to_phone
    test_dataloader = P.test_dataloader

    # Sort index_to_phone alphabetically
    index_to_phone = {
        i: p for i, p in sorted(index_to_phone.items(), key=lambda x: x[1])
    }

    # Add checkpoints for proper iteration
    checkpoints = [f"1_{i}" for i in range(1, 11)]
    epochs = checkpoints + list(range(2, n_epochs + 1))

    # For testing only the final epoch
    epochs = ["1_1", epochs[-1]]

    dataframes = []
    for epoch in epochs:
        # print(f"Epoch {epoch+1}/{epochs}", end="\r")

        """LOAD MODEL"""
        MODEL_WEIGHTS_DIR = get_checkpoint_dir() / model
        encoder = EncoderRNN(vocab_size, h_size, n_layers, dropout).to(device)
        decoder = DecoderRNN(h_size, vocab_size, n_layers, dropout).to(device)

        encoder = EncoderLSTM(vocab_size, h_size, n_layers).to(device)
        decoder = DecoderLSTM(h_size, vocab_size, n_layers).to(device)
        load_weigths(MODEL_WEIGHTS_DIR, encoder, decoder, epoch, device)

        """ TESTING LOOP """
        predictions = []
        test_data = test_data.copy()

        deletions, insertions, substitutions = [], [], []
        error_indices, sequence_lengths = [], []
        edit_distances = []

        # Initialize the confusion matrix
        confusions = {}
        for t in phoneme_stats.keys():
            confusions[t] = {p: 0 for p in phoneme_stats.keys()}
            confusions[t]["OY0"] = 0
        confusions["OY0"] = {p: 0 for p in phoneme_stats.keys()}
        confusions["OY0"]["OY0"] = 0
        # TODO: Fix this hardcoding

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            for inputs, target in test_dataloader:
                inputs = inputs.to(device)
                target = target.to(device)

                encoder_hidden = encoder(inputs)
                decoder_inputs = torch.zeros(1, inputs.shape[1], h_size, device=device)
                decoder_output = decoder(decoder_inputs, encoder_hidden)

                prediction = torch.argmax(decoder_output, dim=-1)
                prediction = prediction.squeeze().cpu().tolist()
                target = target.squeeze().cpu().tolist()

                # Calculate errors
                errors = calculate_errors(prediction, target)
                deletions.append(errors["dels"])
                insertions.append(errors["inss"])
                substitutions.append(errors["subs"])
                edit_distances.append(errors["total"])
                error_indices.append(errors["indices"])
                sequence_lengths.append(errors["length"])

                # Convert indices to phonemes
                prediction = [index_to_phone[i] for i in prediction]
                target = [index_to_phone[i] for i in target]

                # Tabulate confusion between prediction and target
                if len(target) == len(prediction):
                    for t, p in zip(target, prediction):
                        confusions[t][p] += 1

                predictions.append(prediction[:-1])

        test_data["Prediction"] = predictions
        test_data["Deletions"] = deletions
        test_data["Insertions"] = insertions
        test_data["Substitutions"] = substitutions
        test_data["Edit Distance"] = edit_distances
        test_data["Error Indices"] = error_indices
        test_data["Sequence Length"] = sequence_lengths

        error_plots(test_data, model, epoch)
        primacy_recency(test_data, model, epoch)
        confusion_matrix(confusions, model, epoch)

        dataframes.append(test_data)

    return dataframes
