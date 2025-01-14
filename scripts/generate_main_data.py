import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import argparse

import numpy as np

from swp.utils.datasets import (
    create_epoch,
    create_folds,
    create_test_data,
    create_train_data,
    get_phoneme_to_id,
    get_train_fold,
)
from swp.utils.setup import seed_everything

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=50000)
    parser.add_argument("--epoch_size", type=int, default=100000000)
    parser.add_argument("--num_folds", type=int, default=5)

    args = parser.parse_args()

    seed_everything()
    create_test_data()
    train_df = create_train_data(args.vocab_size)
    generator = np.random.default_rng(seed=3407)
    create_folds(train_df, num_folds=args.num_folds, generator=generator)
    for fold_id in range(args.num_folds):
        fold_train_df = get_train_fold(fold_id)
        create_epoch(fold_id, fold_train_df, args.epoch_size, generator)
    get_phoneme_to_id()
