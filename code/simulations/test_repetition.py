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
from utils import seed_everything, set_device
from plots import confusion_matrix, error_plots, primacy_recency

# TODO: replace this with comment below
import sys
MODELS_DIR = ROOT_DIR / "code" / "models"
sys.path.append(str(MODELS_DIR))
from EncoderRNN import EncoderRNN
from DecoderRNN import DecoderRNN
from EncoderLSTM import EncoderLSTM
from DecoderLSTM import DecoderLSTM
# from ..models.DecoderRNN import DecoderRNN
# from ..models.EncoderRNN import EncoderRNN

device = set_device()

""" ARGUMENT PARSING """
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    args = parser.parse_args()
    print(f"Testing model: {args.name}")
    return args

""" ERROR CALCULATION """
def calculate_errors(prediction: list, target: list) -> dict:
    errors = {
        "inss": 0, "dels": 0, "subs": 0, "total": 0, "length": len(target),
        "indices": [i+1 for i, (p, t) in enumerate(zip(prediction, target)) if p != t],
    }

    # Tabulate errors by type
    ops = editops(prediction, target)
    for op, _, _ in ops:
        if op == "insert": errors['inss'] += 1
        elif op == "delete": errors['dels'] += 1
        elif op == "replace": errors['subs'] += 1
    errors['total'] = len(ops)

    return errors

""" TESTING LOOP """
def test_repetition(P: Phonemes, model: str) -> list:
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

        """ LOAD MODEL """
        MODEL_WEIGHTS_DIR = WEIGHTS_DIR / model
        encoder_path = MODEL_WEIGHTS_DIR / f"encoder{epoch}.pth"
        decoder_path = MODEL_WEIGHTS_DIR / f"decoder{epoch}.pth"
        # encoder = EncoderRNN(vocab_size, h_size, n_layers, dropout).to(device)
        # decoder = DecoderRNN(h_size, vocab_size, n_layers, dropout).to(device)
        encoder = EncoderLSTM(vocab_size, h_size, n_layers).to(device)
        decoder = DecoderLSTM(h_size, vocab_size, n_layers).to(device)
        
        encoder.load_state_dict(
            torch.load(encoder_path, map_location=device, weights_only=True)
        )
        decoder.load_state_dict(
            torch.load(decoder_path, map_location=device, weights_only=True)
        )

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
            confusions[t]['OY0'] = 0
        confusions['OY0'] = {p: 0 for p in phoneme_stats.keys()}
        confusions['OY0']['OY0'] = 0
        # TODO: Fix this hardcoding

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            for inputs, target in test_dataloader:
                inputs = inputs.to(device)
                target = target.to(device)

                encoder_hidden, _ = encoder(inputs)
                start_token = torch.zeros(1, 1, vocab_size, device=device)
                decoder_output = decoder(start_token, encoder_hidden, target, 0.0)

                prediction = torch.argmax(decoder_output, dim=-1)
                prediction = prediction.squeeze().cpu().tolist()
                target = target.squeeze().cpu().tolist()

                # Calculate errors
                errors = calculate_errors(prediction, target)
                deletions.append(errors['dels'])
                insertions.append(errors['inss'])
                substitutions.append(errors['subs'])
                edit_distances.append(errors['total'])
                error_indices.append(errors['indices'])
                sequence_lengths.append(errors['length'])
                
                # Convert indices to phonemes
                prediction = [index_to_phone[i] for i in prediction]
                target = [index_to_phone[i] for i in target]

                # Tabulate confusion between prediction and target
                if len(target) == len(prediction):
                    for t, p in zip(target, prediction):
                        confusions[t][p] += 1

                predictions.append(prediction[:-1])

        test_data["Prediction"]      = predictions
        test_data["Deletions"]       = deletions
        test_data["Insertions"]      = insertions
        test_data["Substitutions"]   = substitutions
        test_data["Edit Distance"]   = edit_distances
        test_data["Error Indices"]   = error_indices
        test_data["Sequence Length"] = sequence_lengths

        error_plots(test_data, model, epoch)
        primacy_recency(test_data, model, epoch)
        confusion_matrix(confusions, model, epoch)

        dataframes.append(test_data)

    return dataframes

if __name__ == "__main__":
    seed_everything()
    P = Phonemes()
    args = parse_args()
    dfrs = test_repetition(P, args.name)
