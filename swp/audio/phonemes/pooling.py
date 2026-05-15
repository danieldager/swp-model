from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch

from swp.audio.phonemes.boundaries import OPTIONAL_PASSTHROUGH, SILENCE_LABELS


def tokens_for_phoneme(
    start_s: float,
    end_s: float,
    n_tokens: int,
    token_rate_hz: float,
    method: Literal["center", "start"] = "center",
) -> np.ndarray:
    """Boolean mask over [0, n_tokens) for tokens falling inside [start_s, end_s).

    center (default):
        include token i  if  start_s <= (i + 0.5) / token_rate_hz < end_s
    start:
        include token i  if  start_s <= i / token_rate_hz < end_s

    Returns a bool numpy array of shape (n_tokens,).
    """
    indices = np.arange(n_tokens, dtype=np.float64)
    token_times = (indices + 0.5) / token_rate_hz if method == "center" else indices / token_rate_hz
    return (token_times >= start_s) & (token_times < end_s)


def pool_phoneme(
    activation: torch.Tensor,  # [D, L] float32
    mask: np.ndarray,           # [L] bool
) -> torch.Tensor:
    """Mean over columns of [D, L] selected by mask. Returns [D] float32.

    Returns a NaN vector if mask is entirely False (zero-token phoneme).
    """
    if not mask.any():
        return torch.full((activation.shape[0],), float("nan"), dtype=torch.float32)
    torch_mask = torch.from_numpy(mask)
    return activation[:, torch_mask].float().mean(dim=1)


def build_phoneme_embeddings(
    run_dir: Path,
    boundaries_df: pd.DataFrame,
    layers: list[str],
    token_selection: Literal["center", "start"] = "center",
    max_items: int | None = None,
) -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
    """Pool hidden states over phoneme intervals for all items and layers.

    Iterates over items in manifest order. For each item, loads the saved [D, L]
    activation tensors and mean-pools the columns whose token times fall within
    each phoneme boundary interval.

    Args:
        run_dir:         extraction run directory (must contain manifest.json and
                         activations/ sub-directory)
        boundaries_df:   loaded boundary DataFrame from load_boundaries()
        layers:          layer names to embed (must all be present in run manifest)
        token_selection: "center" (default) or "start" — token assignment rule
        max_items:       stop after first N items from manifest (for smoke tests)

    Returns:
        embeddings:  dict[layer_name → Tensor [n_phonemes_total, D]] float32 CPU
        meta_df:     DataFrame with one row per phoneme, row-aligned with tensors
    """
    with open(run_dir / "manifest.json") as f:
        manifest = json.load(f)
    run_params = manifest["run_params"]
    token_rate_hz = float(run_params["token_rate_hz"])
    hop_s = 1.0 / token_rate_hz

    # Validate requested layers
    available = set(run_params["layers"])
    bad = [la for la in layers if la not in available]
    if bad:
        raise ValueError(
            f"Layers not found in run manifest: {bad}\nAvailable: {sorted(available)}"
        )

    # Validate item coverage
    run_items = set(manifest["items"].keys())
    boundary_items = set(boundaries_df["item_id"].unique())
    missing_in_run = boundary_items - run_items
    if missing_in_run:
        raise ValueError(
            f"{len(missing_in_run)} item_id(s) in boundary CSV but absent from run:\n"
            + ", ".join(sorted(missing_in_run))
        )
    skipped = run_items - boundary_items
    if skipped:
        print(f"  Note: {len(skipped)} run items have no boundary data — skipping.")

    # Per-item max phoneme_index for phoneme_position_from_end
    item_max_idx = (
        boundaries_df.groupby("item_id")["phoneme_index"].max().to_dict()
    )

    item_ids = [iid for iid in manifest["items"] if iid in boundary_items]
    if max_items is not None:
        item_ids = item_ids[:max_items]

    meta_rows: list[dict] = []
    layer_vecs: dict[str, list[torch.Tensor]] = {la: [] for la in layers}

    for item_id in item_ids:
        item_bounds = (
            boundaries_df[boundaries_df["item_id"] == item_id]
            .sort_values("phoneme_index")
            .reset_index(drop=True)
        )

        # Load activation tensors for this item
        layer_paths = manifest["items"][item_id]
        acts: dict[str, torch.Tensor] = {
            la: torch.load(
                run_dir / layer_paths[la], map_location="cpu", weights_only=True
            )
            for la in layers
        }
        n_tokens: int = acts[layers[0]].shape[1]
        represented_s: float = n_tokens * hop_s
        max_phoneme_idx: int = item_max_idx[item_id]

        for _, row in item_bounds.iterrows():
            phoneme = str(row["phoneme"])
            ph_idx = int(row["phoneme_index"])
            start_s = float(row["start_s"])
            end_s_raw = float(row["end_s"])
            dur_s = end_s_raw - float(row["start_s"])

            # Clamp end to represented duration (floor rounding may drop trailing samples)
            if end_s_raw > represented_s + hop_s:
                print(
                    f"  Warning: {item_id} phoneme '{phoneme}' "
                    f"end={end_s_raw:.4f}s > represented={represented_s:.4f}s — clamping."
                )
            end_s = min(end_s_raw, represented_s)

            mask = tokens_for_phoneme(start_s, end_s, n_tokens, token_rate_hz, token_selection)
            n_tok = int(mask.sum())
            silent = phoneme in SILENCE_LABELS
            valid = n_tok > 0

            if not valid:
                print(
                    f"  Warning: {item_id} phoneme '{phoneme}' (idx {ph_idx}) "
                    f"has 0 tokens in [{start_s:.4f}, {end_s:.4f}) — NaN embedding."
                )

            nonzero = np.flatnonzero(mask)
            tok_start = int(nonzero[0]) if valid else -1
            tok_end = int(nonzero[-1]) if valid else -1

            meta: dict = {
                "item_id": item_id,
                "phoneme": phoneme,
                "phoneme_index": ph_idx,
                "phoneme_position_from_start": ph_idx,
                "phoneme_position_from_end": max_phoneme_idx - ph_idx,
                "start_s": start_s,
                "end_s": end_s_raw,
                "duration_s": dur_s,
                "token_start_idx": tok_start,
                "token_end_idx": tok_end,
                "n_tokens": n_tok,
                "valid_embedding": valid,
                "is_silence": silent,
            }
            for col in OPTIONAL_PASSTHROUGH:
                if col in item_bounds.columns:
                    meta[col] = row[col]
            meta_rows.append(meta)

            for la in layers:
                layer_vecs[la].append(pool_phoneme(acts[la], mask))

    if not meta_rows:
        raise RuntimeError(
            "No phoneme rows were processed. "
            "Check that item_ids in the boundary CSV match those in the run manifest."
        )

    meta_df = pd.DataFrame(meta_rows)
    embeddings: dict[str, torch.Tensor] = {
        la: torch.stack(layer_vecs[la])  # [n_phonemes, D]
        for la in layers
    }
    return embeddings, meta_df
