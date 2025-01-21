import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import argparse
import numpy as np
import pandas as pd
from ast import literal_eval

from swp.datasets.phonemes import get_phoneme_to_id
from swp.utils.plots import calculate_errors, create_error_plots

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--training_name", type=str, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--checkpoint", type=int, default=None)

    args = parser.parse_args()

    dir_name = f"{args.model_name}~{args.training_name}"
    file_name = f"{args.epoch}"
    if args.checkpoint is not None:
        file_name = f"{args.epoch}_{args.checkpoint}"

    # TODO this needs to take "include stress" into account
    phoneme_to_id = get_phoneme_to_id()

    test_df = pd.read_csv(f"{parent}/results/test/{dir_name}/{file_name}.csv")
    test_df["Phonemes"] = test_df["Phonemes"].apply(literal_eval)
    test_df["Prediction"] = test_df["Prediction"].apply(literal_eval)
    test_df = test_df.assign(
        **test_df.apply(
            lambda row: calculate_errors(
                row["Phonemes"], row["Prediction"], phoneme_to_id
            ),
            result_type="expand",
            axis=1,
        )
    )

    sonority_df = pd.read_csv(f"{parent}/results/test/{dir_name}/{file_name}~ssp.csv")
    sonority_df["Phonemes"] = sonority_df["Phonemes"].apply(literal_eval)
    sonority_df["Prediction"] = sonority_df["Prediction"].apply(literal_eval)
    sonority_df = sonority_df.assign(
        **sonority_df.apply(
            lambda row: calculate_errors(
                row["Phonemes"], row["Prediction"], phoneme_to_id
            ),
            result_type="expand",
            axis=1,
        )
    )

    create_error_plots(
        test_df,
        sonority_df,
        args.model_name,
        args.training_name,
        args.epoch,
        args.checkpoint,
    )
