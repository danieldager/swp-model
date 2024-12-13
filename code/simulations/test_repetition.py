import sys
import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from Levenshtein import editops

""" PATHS """
FILE_DIR = Path(__file__).resolve()
ROOT_DIR = FILE_DIR.parent.parent.parent

WEIGHTS_DIR = ROOT_DIR / "weights"
DATA_DIR = ROOT_DIR / "data"

WEIGHTS_DIR.mkdir(exist_ok=True)

from Phonemes import Phonemes
from utils import seed_everything, set_device
from plots import confusion_matrix, error_plots

MODELS_DIR = ROOT_DIR / "code" / "models"
sys.path.append(str(MODELS_DIR))
from EncoderLSTM import EncoderLSTM
from DecoderLSTM import DecoderLSTM

device = set_device()

""" ARGUMENT PARSING """


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    args = parser.parse_args()
    print(f"Testing model: {args.name}")
    return args


""" ERROR CALCULATION """


def calculate_errors(output: list, target: list) -> dict:
    errors = {
        "inss": 0,
        "dels": 0,
        "subs": 0,
        "total": 0,
        "length": len(target),
        "indices": [i + 1 for i, (p, t) in enumerate(zip(output, target)) if p != t],
    }

    # Tabulate errors by type
    ops = editops(output, target)
    for op, _, _ in ops:
        if op == "insert":
            errors["inss"] += 1
        elif op == "delete":
            errors["dels"] += 1
        elif op == "replace":
            errors["subs"] += 1
    errors["total"] = len(ops)

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
        MODEL_WEIGHTS_DIR = WEIGHTS_DIR / model
        embedding_path = MODEL_WEIGHTS_DIR / f"embedding{epoch}.pth"
        encoder_path = MODEL_WEIGHTS_DIR / f"encoder{epoch}.pth"
        decoder_path = MODEL_WEIGHTS_DIR / f"decoder{epoch}.pth"

        shared_embedding = nn.Embedding(vocab_size, h_size)
        encoder = EncoderLSTM(
            vocab_size, h_size, n_layers, dropout, shared_embedding
        ).to(device)
        decoder = DecoderLSTM(
            h_size, vocab_size, n_layers, dropout, shared_embedding
        ).to(device)

        encoder.load_state_dict(
            torch.load(encoder_path, map_location=device, weights_only=True)
        )
        decoder.load_state_dict(
            torch.load(decoder_path, map_location=device, weights_only=True)
        )
        shared_embedding.load_state_dict(
            torch.load(embedding_path, map_location=device, weights_only=True)
        )

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
        # primacy_recency(test_data, model, epoch)
        confusion_matrix(confusions, model, epoch)

        dataframes.append(test_data)

    return dataframes


if __name__ == "__main__":
    seed_everything()
    P = Phonemes()
    args = parse_args()
    dfrs = test_repetition(P, args.name)
