"""
Shared utilities for reproduce scripts.
"""

import os
import pathlib
import sys
import warnings
from ast import literal_eval
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

# Add parent directory to path for swp imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from swp.datasets.phonemes import get_phoneme_testloader
from swp.test.repetition import test
from swp.utils.datasets import enrich_for_plotting
from swp.utils.embeddings import create_embeddings_LSTM_hook
from swp.utils.models import get_model
from swp.utils.setup import seed_everything, set_device

# Standard converters for CSV files
CONVERTERS = {
    "Word": str,
    "Phonemes": literal_eval,
    "No_Stress": literal_eval,
}


def setup_environment(seed: int = 42) -> torch.device:
    """Set up the environment with random seed and device."""
    seed_everything(seed)
    device = set_device()

    # Suppress warnings for nested tensors
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="The PyTorch API of nested tensors is in prototype stage",
    )

    return device


def load_model(model_name: str, weights_path: str, device: torch.device) -> Any:
    """Load a model with the specified weights."""
    model = get_model(model_name)
    model.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    model.to(device)
    return model


def load_or_generate_dataset_results(
    data_path: str,
    dataset_path: str,
    model: Any,
    device: torch.device,
    batch_size: int,
    enrich: bool = True,
    regenerate: bool = False,
) -> pd.DataFrame:
    """Load cached results or generate new ones.

    Set regenerate=True to ignore cache and recompute.
    """
    if os.path.exists(data_path) and not regenerate:
        print(f"Loading cached results from {data_path}")
        return pd.read_csv(data_path, converters=CONVERTERS, index_col=0)

    print(f"Generating new results for {dataset_path}")
    df = pd.read_csv(dataset_path, converters=CONVERTERS, index_col=0)
    loader = get_phoneme_testloader(batch_size=batch_size, dataset_df=df)

    results, _ = test(
        model=model,
        device=device,
        test_df=df,
        test_loader=loader,
    )

    if enrich:
        results = enrich_for_plotting(results)

    # Save results
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    results.to_csv(data_path)

    return results


def load_or_generate_embeddings(
    embeddings_path: str,
    dataset_path: str,
    model: Any,
    device: torch.device,
    batch_size: int,
    position: Optional[int] = None,
    regenerate: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load cached embeddings or generate new ones."""
    df = pd.read_csv(dataset_path, converters=CONVERTERS)

    if os.path.exists(embeddings_path) and not regenerate:
        print(f"Loading cached embeddings from {embeddings_path}")
        embeddings = np.load(embeddings_path)
        return embeddings, df

    print(f"Generating new embeddings for {dataset_path}")

    # Create hook to extract embeddings
    buffer = {"Hidden": [], "Cell": [], "Out": []}
    hook = create_embeddings_LSTM_hook(buffer, store_out=True)
    hook_handle = model.encoder.recurrent.register_forward_hook(hook)

    # Generate embeddings
    loader = get_phoneme_testloader(batch_size=batch_size, dataset_df=df)
    test(
        model=model,
        device=device,
        test_df=df,
        test_loader=loader,
    )

    # Extract embeddings based on position
    if position is not None:
        # For specific position (like final position for WFE)
        lengths = df["Length"].to_numpy()
        embeddings = np.vstack(buffer["Out"])
        embeddings = embeddings[np.arange(len(df)), lengths, :]
    else:
        # For first position (like phonemes)
        embeddings = buffer["Out"][0][np.arange(len(df)), 0, :]

    hook_handle.remove()

    # Save embeddings
    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
    np.save(embeddings_path, embeddings)

    return embeddings, df


def ensure_output_dir(output_dir: str) -> pathlib.Path:
    """Create output directory if it doesn't exist."""
    path = pathlib.Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_default_paths() -> Dict[str, Any]:
    """Get default file paths relative to the reproduce directory."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    return {
        "datasets": {
            "wfe": os.path.join(base_dir, "datasets", "wfe.csv"),
            "ssp": os.path.join(base_dir, "datasets", "ssp.csv"),
            "phonemes": os.path.join(base_dir, "datasets", "phonemes.csv"),
        },
        "weights": {
            "1024": os.path.join(base_dir, "weights", "1024_75.pth"),
            "2048": os.path.join(base_dir, "weights", "2048_75.pth"),
        },
        "data": os.path.join(base_dir, "data"),
        "figures": os.path.join(base_dir, "figures"),
    }


def parse_figsize(figsize_str: str) -> Tuple[float, float]:
    """Parse figsize string like '7,6' into tuple."""
    try:
        width, height = map(float, figsize_str.split(","))
        return (width, height)
    except ValueError:
        raise ValueError(
            f"Invalid figsize format: {figsize_str}. Expected format: 'width,height'"
        )
        raise ValueError(
            f"Invalid figsize format: {figsize_str}. Expected format: 'width,height'"
