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
from swp.utils.plots import enrich_for_plotting, error_plots, regression_plots

results_test_dir = get_gridsearch_test_dir()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--include_stress", action="store_true")
    args = parser.parse_args()

    phoneme_to_id = get_phoneme_to_id(include_stress=args.include_stress)
    model_dir = results_test_dir / f"{args.model_name}~{args.train_name}"

    if args.checkpoint is None:
        checkpoints = set([f.stem.split("~")[0] for f in model_dir.glob("*.csv")])
    else:
        checkpoints = [args.checkpoint]

    for checkpoint in checkpoints:
        test_df = pd.read_csv(model_dir / f"{checkpoint}.csv")
        test_df = enrich_for_plotting(test_df, phoneme_to_id, args.include_stress)

        ssp_df = pd.read_csv(model_dir / f"{checkpoint}~ssp.csv")
        ssp_df = enrich_for_plotting(ssp_df, phoneme_to_id, args.include_stress)

        train_df = pd.read_csv(model_dir / f"{checkpoint}~train.csv")
        train_df = enrich_for_plotting(train_df, phoneme_to_id, args.include_stress)

        error_plots(
            test_df,
            ssp_df,
            args.model_name,
            args.train_name,
            checkpoint,
        )

        regression_plots(
            test_df,
            train_df,
            args.model_name,
            args.train_name,
            checkpoint,
        )
