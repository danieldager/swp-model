import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import argparse

from swp.datasets.Phonemes import Phonemes
from swp.test.repetition import test_repetition
from swp.utils.legacy_utils import seed_everything, set_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    args = parser.parse_args()
    print(f"Testing model: {args.name}")
    return args


if __name__ == "__main__":
    device = set_device()
    seed_everything()
    P = Phonemes()
    args = parse_args()
    dfrs = test_repetition(P, args.name, device)
