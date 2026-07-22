from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
from typing import Dict
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
for path in (ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from intervention.states_extract import StatesDataset
from swp.utils.datasets import get_phoneme_to_id

VALID_EMBEDDING_INITS = {
    "pretrained",
    "delta_c_mean",
    "delta_c_median",
    "delta_h_mean",
    "delta_h_median",
    "delta_state_mean",
    "delta_state_median",
}


def save_phoneme_mean_median_embeddings(
    output_path: Path,
    states_dataset,
    phoneme_to_id: Dict[str, int],
) -> None:
    output_path = Path(output_path)
    tokens = [token for token, _ in sorted(phoneme_to_id.items(), key=lambda item: item[1])]
    num_tokens = len(tokens)

    if states_dataset.delta_c is None or states_dataset.delta_h is None:
        raise ValueError("StatesDataset is missing delta_h or delta_c arrays")

    hidden_dim = states_dataset.delta_c.shape[1]
    delta_h_mean = np.zeros((num_tokens, hidden_dim), dtype=np.float32)
    delta_h_median = np.zeros((num_tokens, hidden_dim), dtype=np.float32)
    delta_c_mean = np.zeros((num_tokens, hidden_dim), dtype=np.float32)
    delta_c_median = np.zeros((num_tokens, hidden_dim), dtype=np.float32)

    if states_dataset.delta_state is None:
        states_dataset.delta_state = np.concatenate([states_dataset.delta_h, states_dataset.delta_c], axis=1)

    state_dim = states_dataset.delta_state.shape[1]
    delta_state_mean = np.zeros((num_tokens, state_dim), dtype=np.float32)
    delta_state_median = np.zeros((num_tokens, state_dim), dtype=np.float32)
    counts = np.zeros((num_tokens,), dtype=np.int32)

    for token in tokens:
        token_idx = phoneme_to_id[token]
        mask = states_dataset.metadata["phoneme"] == token
        counts[token_idx] = int(mask.sum())
        if counts[token_idx] == 0:
            continue

        delta_h_values = states_dataset.delta_h[mask.values]
        delta_c_values = states_dataset.delta_c[mask.values]

        delta_h_mean[token_idx] = np.mean(delta_h_values, axis=0)
        delta_h_median[token_idx] = np.median(delta_h_values, axis=0)
        delta_c_mean[token_idx] = np.mean(delta_c_values, axis=0)
        delta_c_median[token_idx] = np.median(delta_c_values, axis=0)

        delta_state_values = states_dataset.delta_state[mask.values]
        delta_state_mean[token_idx] = np.mean(delta_state_values, axis=0)
        delta_state_median[token_idx] = np.median(delta_state_values, axis=0)

    np.savez_compressed(
        output_path,
        phonemes=np.array(tokens, dtype=object),
        counts=counts,
        delta_h_mean=delta_h_mean,
        delta_h_median=delta_h_median,
        delta_c_mean=delta_c_mean,
        delta_c_median=delta_c_median,
        delta_state_mean=delta_state_mean,
        delta_state_median=delta_state_median,
    )


def load_token_embedding_from_stats(
    embedding_init: str,
    stats_path: Path,
    phoneme_to_id: Dict[str, int],
    repeat_model_embedding: nn.Embedding | None = None,
) -> nn.Embedding:
    embedding_init = embedding_init.strip().lower()
    if embedding_init not in VALID_EMBEDDING_INITS:
        raise ValueError(
            f"Invalid embedding_init={embedding_init}. "
            f"Expected one of {sorted(VALID_EMBEDDING_INITS)}."
        )
    if embedding_init == "pretrained":
        raise ValueError("Use repeat_model.encoder.embedding for pretrained init.")

    if not stats_path.exists():
        raise FileNotFoundError(f"Saved stats file not found: {stats_path}")

    data = np.load(stats_path, allow_pickle=True)
    if embedding_init not in data.files:
        raise KeyError(
            f"Embedding statistics key '{embedding_init}' not found in {stats_path}."
        )

    weights = data[embedding_init].astype(np.float32)
    counts = data["counts"] if "counts" in data.files else None

    if weights.shape[0] != len(phoneme_to_id):
        raise ValueError(
            f"Saved weights length {weights.shape[0]} does not match vocab size {len(phoneme_to_id)}"
        )

    def _learn_linear_projection(
        source_weights: np.ndarray,
        target_weights: np.ndarray,
        counts: np.ndarray,
    ) -> np.ndarray:
        valid = np.where(counts > 0)[0]
        if valid.size == 0:
            raise ValueError("Cannot learn projection: no tokens with nonzero counts available.")

        X = source_weights[valid]
        Y = target_weights[valid]
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("Source and target weights must be 2D arrays.")

        solution, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        return solution

    if repeat_model_embedding is not None and counts is not None:
        source_dim = repeat_model_embedding.embedding_dim
        target_dim = weights.shape[1]
        if source_dim != target_dim:
            source_weights = repeat_model_embedding.weight.data.detach().cpu().numpy()
            try:
                projection = _learn_linear_projection(source_weights, weights, counts)
            except ValueError:
                projection = None

            for token in ["<PAD>", "<SOS>", "<EOS>"]:
                token_idx = phoneme_to_id[token]
                if counts[token_idx] == 0:
                    fallback = (
                        repeat_model_embedding.weight.data[token_idx]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    if projection is not None:
                        weights[token_idx] = fallback @ projection
                    else:
                        if source_dim == target_dim:
                            weights[token_idx] = fallback
                        elif source_dim * 2 == target_dim:
                            weights[token_idx] = np.concatenate([fallback, fallback], axis=0)
                        elif source_dim < target_dim:
                            pad = np.zeros((target_dim - source_dim,), dtype=fallback.dtype)
                            weights[token_idx] = np.concatenate([fallback, pad], axis=0)
                        else:
                            weights[token_idx] = fallback[:target_dim]

    embedding = nn.Embedding(len(phoneme_to_id), weights.shape[1])
    with torch.no_grad():
        embedding.weight.data.copy_(torch.from_numpy(weights))
    return embedding

if __name__ == "__main__":
    states_path = Path("states_ds/train_states")
    output_path = Path("states_ds/phoneme_state_embeddings.npz")
    states_ds = StatesDataset.load(str(states_path))
    phoneme_to_id = get_phoneme_to_id()
    save_phoneme_mean_median_embeddings(output_path, states_ds, phoneme_to_id)
    print(f"Saved phoneme mean/median embeddings to {output_path}")
