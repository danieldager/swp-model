import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import argparse

from swp.train.repetition import train_repetition
from swp.utils.setup import seed_everything, set_device

""" ARGUMENT PARSER """


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold_id",
        type=int,
        default=4,
        help="Evaluation fold id (out of 5)",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="rnn",
        help="Model type (e.g. rnn, lstm)",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (fixed to 1 for repetition)",
    )

    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="Hidden size of recurrent layers",
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="Number of hidden recurrent layers",
    )

    parser.add_argument(
        "--learn_rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate",
    )

    parser.add_argument(
        "--tf_ratio",
        type=float,
        default=0.2,
        help="Teacher forcing ratio",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    seed_everything()
    device = set_device()
    args = parse_args()
    params = vars(args)
    print(params)
    model = train_repetition(*params.values(), device)
