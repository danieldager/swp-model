import argparse
import os
import sys
from ast import literal_eval

import pandas as pd

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from swp.utils.datasets import enrich_for_plotting
from swp.utils.paths import get_figures_dir, get_test_dir
from swp.viz.test import (
    plot_category_errors,
    plot_length_errors,
    plot_position_errors,
    plot_sonority_errors,
    regression_plots,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--include_stress", action="store_true")
    args = parser.parse_args()

    results_dir = get_test_dir() / f"{args.model_name}~{args.train_name}"
    figures_dir = get_figures_dir() / f"{args.model_name}~{args.train_name}"
    figures_dir.mkdir(exist_ok=True)

    if args.checkpoint is None:
        checkpoints = set([f.stem.split("~")[0] for f in results_dir.glob("*.csv")])
    else:
        checkpoints = [args.checkpoint]

    phoneme_key = "Phonemes" if args.include_stress else "No Stress"

    for checkpoint in checkpoints:
        test_df = pd.read_csv(results_dir / f"{checkpoint}.csv")
        test_df[phoneme_key] = test_df[phoneme_key].apply(literal_eval)
        test_df["Prediction"] = test_df["Prediction"].apply(literal_eval)
        test_df = enrich_for_plotting(test_df, args.include_stress)

        ssp_df = pd.read_csv(results_dir / f"{checkpoint}~ssp.csv")
        ssp_df[phoneme_key] = ssp_df[phoneme_key].apply(literal_eval)
        ssp_df["Prediction"] = ssp_df["Prediction"].apply(literal_eval)
        ssp_df = enrich_for_plotting(ssp_df, args.include_stress)

        train_df = pd.read_csv(results_dir / f"{checkpoint}~train.csv")
        train_df[phoneme_key] = train_df[phoneme_key].apply(literal_eval)
        train_df["Prediction"] = train_df["Prediction"].apply(literal_eval)
        train_df = enrich_for_plotting(train_df, args.include_stress)

        plot_length_errors(test_df, checkpoint, figures_dir)
        plot_position_errors(test_df, checkpoint, figures_dir)
        plot_sonority_errors(ssp_df, checkpoint, figures_dir)
        plot_category_errors(test_df, checkpoint, figures_dir)

        regression_plots(
            test_df,
            args.model_name,
            args.train_name,
            checkpoint,
            figures_dir,
        )
