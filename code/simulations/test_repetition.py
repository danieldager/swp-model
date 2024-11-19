import argparse
from pathlib import Path

import pandas as pd
import torch
from Levenshtein import editops

""" PATHS """
FILE_DIR = Path(__file__).resolve()
ROOT_DIR = FILE_DIR.parent.parent.parent

WEIGHTS_DIR = ROOT_DIR / "weights"
DATA_DIR = ROOT_DIR / "data"

WEIGHTS_DIR.mkdir(exist_ok=True)

from Phonemes import Phonemes
from plots import confusion_matrix, error_plots
from utils import seed_everything, set_device

from ..models.DecoderRNN import DecoderRNN
from ..models.EncoderRNN import EncoderRNN

device = set_device()

""" ARGUMENT PARSING """


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    args = parser.parse_args()
    print(f"Testing model: {args.name}")
    return args


""" TESTING LOOP """


def test_repetition(P: Phonemes, model: str) -> list:
    # Unpack parameters from model name
    e, h, l, d = [p[1:] for p in model.split("_")[:-1]]
    num_epochs, hidden_size, num_layers, dropout = int(e), int(h), int(l), float(d)

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
    epochs = checkpoints + list(range(2, num_epochs + 1))

    dataframes = []
    for epoch in epochs:
        print(f"Testing epoch {epoch}...")

        """ LOAD MODEL """
        MODEL_WEIGHTS_DIR = WEIGHTS_DIR / model
        encoder_path = MODEL_WEIGHTS_DIR / f"encoder{epoch}.pth"
        decoder_path = MODEL_WEIGHTS_DIR / f"decoder{epoch}.pth"
        encoder = EncoderRNN(vocab_size, hidden_size, num_layers, dropout).to(device)
        decoder = DecoderRNN(hidden_size, vocab_size, num_layers, dropout).to(device)
        encoder.load_state_dict(
            torch.load(encoder_path, map_location=device, weights_only=True)
        )
        decoder.load_state_dict(
            torch.load(decoder_path, map_location=device, weights_only=True)
        )

        """ TESTING LOOP """
        deletions = []
        insertions = []
        substitutions = []
        edit_distance = []
        predictions = []
        test_data = test_data.copy()

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
                insertion, deletion, substitution = 0, 0, 0

                encoder_hidden = encoder(inputs)
                decoder_input = torch.zeros(
                    1, inputs.shape[1], hidden_size, device=device
                )
                decoder_output = decoder(decoder_input, encoder_hidden)

                prediction = torch.argmax(decoder_output, dim=-1)
                prediction = prediction.squeeze().cpu().tolist()
                target = target.squeeze().cpu().tolist()

                # Tabulate errors by type and calculate edit distance
                ops = editops(prediction, target)
                for op, _, _ in ops:
                    if op == "insert":
                        insertion += 1
                    elif op == "delete":
                        deletion += 1
                    elif op == "replace":
                        substitution += 1
                deletions.append(deletion)
                insertions.append(insertion)
                substitutions.append(substitution)
                edit_distance.append(len(ops))

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
        test_data["Edit Distance"] = edit_distance

        error_plots(test_data, model, epoch)
        confusion_matrix(confusions, model, epoch)
        dataframes.append(test_data)

    return dataframes


if __name__ == "__main__":
    seed_everything()
    P = Phonemes()
    args = parse_args()
    dfrs = test_repetition(P, args.name)
