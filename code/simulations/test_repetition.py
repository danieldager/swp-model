import argparse
from pathlib import Path

import pandas as pd
import torch
from Levenshtein import editops

""" PATHS """
FILE_DIR = Path(__file__).resolve()
ROOT_DIR = FILE_DIR.parent.parent.parent

WEIGHTS_DIR = ROOT_DIR / "weights_other"
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
        "inss": 0, "dels": 0, "subs": 0, "total": 0,
        "h_count": 0, "m_count": 0, "t_count": 0
    }

    # Tabulate errors by type
    ops = editops(prediction, target)
    for op, _, _ in ops:
        if op == "insert": errors['inss'] += 1
        elif op == "delete": errors['dels'] += 1
        elif op == "replace": errors['subs'] += 1
    errors['total'] = len(ops)
    
    # Get indices of errors
    indices = [i+1 for i, (p, t) in enumerate(zip(prediction, target)) if p != t]

    # Calculate the boundaries primacy/recency effect
    if len(target) <= 5: head, tail = 1, len(target)
    elif len(target) <= 7: head, tail = 2, len(target) - 1
    else: head, tail = 3, len(target) - 2

    # Count errors in each segment
    for index in indices:
        if index <= head: errors['h_count'] += 1
        elif index <= tail: errors['m_count'] += 1
        else: errors['t_count'] += 1
    
    return errors

""" TESTING LOOP """
def test_repetition(P: Phonemes, model: str) -> list:
    # Unpack parameters from model name
    e, h, l, d = [p[1:] for p in model.split("_")[:-1]]
    num_epochs, hidden_size, num_layers, dropout = int(e), int(h), int(l), float(d)

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
    epochs = checkpoints + list(range(2, num_epochs + 1))
    
    # For testing only the final epoch
    epochs = ["1_1", 30]

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
        predictions = []
        test_data = test_data.copy()

        deletions, insertions, substitutions = [], [], []
        start_errors, middle_errors, end_errors = [], [], []
        edit_distances = []

        # Initialize the confusion matrix
        confusions = {}
        for t in phoneme_stats.keys():
            confusions[t] = {p: 0 for p in phoneme_stats.keys()}

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            for inputs, target in test_dataloader:
                inputs = inputs.to(device)
                target = target.to(device)

                encoder_hidden = encoder(inputs)
                decoder_input = torch.zeros(
                    1, inputs.shape[1], hidden_size, device=device
                )
                decoder_output = decoder(decoder_input, encoder_hidden)

                prediction = torch.argmax(decoder_output, dim=-1)
                prediction = prediction.squeeze().cpu().tolist()
                target = target.squeeze().cpu().tolist()

                # Calculate errors
                errors = calculate_errors(prediction, target)
                deletions.append(errors['dels'])
                insertions.append(errors['inss'])
                substitutions.append(errors['subs'])
                edit_distances.append(errors['total'])
                start_errors.append(errors['h_count'])
                middle_errors.append(errors['m_count'])
                end_errors.append(errors['t_count'])
                
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
        test_data["Start Errors"] = start_errors
        test_data["Middle Errors"] = middle_errors
        test_data["End Errors"] = end_errors

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
