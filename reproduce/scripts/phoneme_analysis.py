#!/usr/bin/env python3
"""
Generate phoneme embeddings analysis plots (Figure 3A-B).
Includes dendrogram/distance matrix and feature importance analysis.
"""

import argparse
import os
import pathlib

from utils import (
    ensure_output_dir,
    get_default_paths,
    load_model,
    load_or_generate_embeddings,
    parse_figsize,
    setup_environment,
)

from swp.viz.embeddings import dendro_dmatrix, mlem_importance, mlem_phonemes


def generate_phoneme_analysis(
    model_name: str,
    weights_path: str,
    batch_size: int,
    data_dir: str,
    figures_dir: str,
    figsize: tuple = (5, 2.5),
    regenerate: bool = False,
    seed: int = 42,
):
    """Generate phoneme embeddings analysis plots."""
    # Setup
    device = setup_environment(seed)
    model = load_model(model_name, weights_path, device)

    # Get default paths
    paths = get_default_paths()

    # Load/generate phoneme embeddings
    embeddings_path = os.path.join(data_dir, "phonemes_embeddings.npy")
    embeddings, phonemes_df = load_or_generate_embeddings(
        embeddings_path=embeddings_path,
        dataset_path=paths["datasets"]["phonemes"],
        model=model,
        device=device,
        batch_size=batch_size,
        position=None,  # Use first position for phonemes
        regenerate=regenerate,
    )

    # Create output directory
    output_path = ensure_output_dir(os.path.join(figures_dir, "phonemes"))

    # Generate dendrogram and distance matrix
    print("Generating dendrogram and distance matrix...")
    dendro_dmatrix(phonemes_df, embeddings, path=output_path)

    # Separate phonemes into vowels and consonants for feature importance
    print("Analyzing vowel features...")
    v_df = phonemes_df.query("Type == 'V' and Diphthong == False")
    v_drops = [
        "Phoneme",
        "Type",
        "Place",
        "Manner",
        "Voiced",
        "No_Stress",
        "Length",
        "Prediction",
    ]
    v_df = v_df.drop(columns=v_drops)
    v_emb = embeddings[v_df.index]

    print("Analyzing consonant features...")
    c_df = phonemes_df.query("Type == 'C'")
    c_drops = [
        "Phoneme",
        "Type",
        "Height",
        "Backness",
        "Diphthong",
        "No_Stress",
        "Length",
        "Prediction",
    ]
    c_df = c_df.drop(columns=c_drops)
    c_emb = embeddings[c_df.index]

    # Calculate feature importances and plot
    print("Calculating feature importances...")
    v_results = mlem_importance(v_df, v_emb)
    c_results = mlem_importance(c_df, c_emb)
    mlem_phonemes(v_results, c_results, path=output_path, figsize=figsize)

    print(f"Plots saved to: {output_path}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Generate phoneme embeddings analysis plots (Figure 3A-B)"
    )

    # Model parameters
    parser.add_argument(
        "--model_name", default="Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1", help="Model name"
    )
    parser.add_argument("--weights_path", help="Path to model weights file")
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="Batch size for evaluation"
    )

    # I/O parameters
    parser.add_argument("--data_dir", help="Directory for cached data files")
    parser.add_argument("--figures_dir", help="Directory for output figures")

    # Plot parameters
    parser.add_argument(
        "--figsize",
        default="5,2.5",
        help='Figure size for feature importance plot as "width,height"',
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate embeddings and analyses even if cached files exist",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Get default paths if not provided
    paths = get_default_paths()

    weights_path = args.weights_path or paths["weights"]["1024"]
    data_dir = args.data_dir or paths["data"]
    figures_dir = args.figures_dir or paths["figures"]

    # Parse figsize
    figsize = parse_figsize(args.figsize)

    # Generate plots
    generate_phoneme_analysis(
        model_name=args.model_name,
        weights_path=weights_path,
        batch_size=args.batch_size,
        data_dir=data_dir,
        figures_dir=figures_dir,
        figsize=figsize,
        regenerate=args.regenerate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
