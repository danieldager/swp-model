import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import argparse

import torch

from swp.datasets.phonemes import Phonemes
from swp.train.repetition import train_repetition
from swp.utils.setup import seed_everything, set_device

""" ARGUMENT PARSER """


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size",
    )

    parser.add_argument(
        "--hidden_size",
        type=int,
        default=4,
        help="Hidden size",
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="Hidden layers",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate",
    )

    parser.add_argument(
        "--tf_ratio",
        type=float,
        default=0.0,
        help="Teacher forcing",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    device = set_device()
    seed_everything()
    P = Phonemes()
    args = parse_args()
    params = vars(args)
    model = train_repetition(P, params, device)
