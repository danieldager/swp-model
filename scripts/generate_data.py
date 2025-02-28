import argparse
import os
import sys

import numpy as np

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from swp.utils.datasets import (
    create_epoch,
    create_folds,
    create_phoneme_to_id,
    create_train_data,
    get_phoneme_statistics,
    get_train_fold,
)
from swp.utils.setup import seed_everything

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--epoch_size", type=int, default=1000000)
    parser.add_argument("--num_folds", type=int, default=5)
    args = parser.parse_args()

    print("\nGenerating test data...")
    seed_everything()

    print("\nGenerating train data...")
    train_df = create_train_data(args.vocab_size)
    generator = np.random.default_rng(seed=3407)
    print("train_df len: ", len(train_df))

    print("\nGenerating folds...")
    create_folds(train_df, num_folds=args.num_folds, generator=generator)

    print("\nGenerating epochs...")
    for fold_id in range(args.num_folds):
        fold_train_df = get_train_fold(fold_id)
        create_epoch(fold_id, fold_train_df, args.epoch_size, generator)
    create_epoch(None, train_df, epoch_size=args.epoch_size, generator=generator)

    print("\nComputing phoneme statistics...\n")
    get_phoneme_statistics(train_df)

    print("\nComputing phoneme tokens...\n")
    create_phoneme_to_id(train_df, False)
