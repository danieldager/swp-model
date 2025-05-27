import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import argparse

from swp.datasets.graphemes.image_gen import create_gen_arg_dict
from swp.datasets.graphemes.testdata_gen import create_test_dataset
from swp.datasets.graphemes.traindata_gen import create_train_dataset
from swp.utils.datasets import create_epoch, get_evaluation_dataset, get_train_dataset
from swp.utils.paths import get_graphemes_dir
from swp.utils.setup import seed_everything

if __name__ == "__main__":

    query = "(Word.str.len() < 11) & (Word.str.len() > 1)"
    filtered_train_df = get_train_dataset(query=query)
    train_words = filtered_train_df["Word"].tolist()
    filtered_test_df = get_evaluation_dataset(query)
    test_words = filtered_test_df["Word"].tolist()
    all_words = train_words + test_words
    graphemes_dir = get_graphemes_dir()

    create_gen_arg_dict(graphemes_dir, all_words)
    create_train_dataset(graphemes_dir, train_words, 100, seed=42)
    create_test_dataset(graphemes_dir, test_words)

    create_epoch(None, filtered_train_df, epoch_size=10**6)
