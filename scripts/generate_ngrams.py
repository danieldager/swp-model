import math
import os
import sys

import numpy as np

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import argparse

from swp.datasets.phonemes import get_ngram_dataset


def round_to_second_digit_floor(n):
    """Round down to the floor of the second significant digit.

    Examples:
        179344 -> 170000
        2345 -> 2300
        42 -> 42 (no change for numbers < 100)
    """
    if n < 100:
        return n

    magnitude = 10 ** math.floor(math.log10(n))
    second_digit_multiplier = 10 ** (math.floor(math.log10(n)) - 1)
    return math.floor(n / second_digit_multiplier) * second_digit_multiplier


def calculate_sample_size(n, num_phonemes=39, max_samples=500000, base_n=3):
    """Calculate appropriate sample size for n-grams using logspace scaling.

    Args:
        n: Length of n-grams
        num_phonemes: Number of unique phonemes in the vocabulary
        max_samples: Maximum sample size for n=10 (default 500,000)
        base_n: Base n-gram length with full coverage (default 3)

    Returns:
        int: Sample size to use for this n-gram length, rounded to second digit
    """
    if n <= base_n:
        # For base_n and below, return all permutations
        return num_phonemes**n

    # Calculate base samples (full permutations at base_n)
    base_samples = num_phonemes**base_n

    # Create a logspace from base_samples to max_samples
    max_n = 10
    # Create 8 points (for n=3 to n=10)
    samples = np.logspace(
        np.log10(base_samples), np.log10(max_samples), max_n - base_n + 1
    )

    # Get the sample size for the requested n
    sample_size = int(samples[n - base_n])

    # Round down to floor of second significant digit
    return round_to_second_digit_floor(sample_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n", type=int, required=True, help="Length of ngrams to generate"
    )
    parser.add_argument(
        "--max_samples", type=int, default=500000, help="Maximum sample size for n=10"
    )
    parser.add_argument(
        "--base_n",
        type=int,
        default=3,
        help="Base n-gram length with full permutation coverage",
    )
    args = parser.parse_args()

    # Calculate appropriate sample size for this n-gram length
    sample_size = calculate_sample_size(
        args.n, max_samples=args.max_samples, base_n=args.base_n
    )

    print(f"Generating {args.n}grams dataset with sample size: {sample_size}...")
    dataset = get_ngram_dataset(args.n, limit=sample_size)
    print(f"Generated dataset with {len(dataset)} entries.")
