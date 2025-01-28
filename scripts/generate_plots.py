import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import argparse
import numpy as np
import pandas as pd
from ast import literal_eval

from swp.utils.paths import get_gridsearch_test_dir
from swp.datasets.phonemes import get_phoneme_to_id
from swp.utils.plots import calculate_errors, create_error_plots

results_test_dir = get_gridsearch_test_dir()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--include_stress", action="store_true")
    args = parser.parse_args()

    # TODO this needs to take "include stress" into account
    phoneme_to_id = get_phoneme_to_id(include_stress=args.include_stress)
    results_model_dir = results_test_dir / f"{args.model_name}~{args.train_name}"

    if args.checkpoint is None:
        checkpoints = set(
            [f.stem.split("~")[0] for f in results_model_dir.glob("*.csv")]
        )
    else:
        checkpoints = [args.checkpoint]

    for checkpoint in checkpoints:
        test_df = pd.read_csv(results_model_dir / f"{checkpoint}.csv")
        if args.include_stress:
            test_df["Phonemes"] = test_df["Phonemes"].apply(literal_eval)
        else:
            test_df["No Stress"] = test_df["No Stress"].apply(literal_eval)
        test_df["Prediction"] = test_df["Prediction"].apply(literal_eval)
        test_df = test_df.assign(
            **test_df.apply(
                lambda row: calculate_errors(
                    row["Phonemes" if args.include_stress else "No Stress"],
                    row["Prediction"],
                    phoneme_to_id,
                ),
                result_type="expand",
                axis=1,
            )
        )

        sonority_df = pd.read_csv(results_model_dir / f"{checkpoint}~ssp.csv")
        if args.include_stress:
            sonority_df["Phonemes"] = sonority_df["Phonemes"].apply(literal_eval)
        else:
            sonority_df["No Stress"] = sonority_df["No Stress"].apply(literal_eval)
        sonority_df["Prediction"] = sonority_df["Prediction"].apply(literal_eval)
        sonority_df = sonority_df.assign(
            **sonority_df.apply(
                lambda row: calculate_errors(
                    row["Phonemes" if args.include_stress else "No Stress"],
                    row["Prediction"],
                    phoneme_to_id,
                ),
                result_type="expand",
                axis=1,
            )
        )

        create_error_plots(
            test_df,
            sonority_df,
            args.model_name,
            args.train_name,
            checkpoint,
        )
