"""Canonical xarray export for audio codec activations.

Converts per-item per-layer .pt activation tensors from a Step 1 extraction
run into a canonical xarray.Dataset with dimensions (trials, time, neurons).

Design rules (Phase A)
----------------------
- Activations are stored raw, without normalisation, z-scoring, or resampling.
- Padding is applied only at the end of the time axis.
- Padded positions carry NaN in `activations` and False in `valid_time`.
- Missing .pt files and n_neurons mismatches cause hard failures.
- Only 'encoder_out' and 'decoder_in' are eligible; 'decoder_out' is rejected.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr


# Layers containing latent representations — eligible for xarray export
LATENT_LAYERS: frozenset[str] = frozenset({"encoder_out", "decoder_in"})

# Waveform layer — rejected by this module
WAVEFORM_LAYER = "decoder_out"

# Columns loaded from the dataset CSV
_META_COLS = [
    "item_id",
    "condition",
    "lexicality",
    "length_bin",
    "morphology",
    "frequency_bin",
    "speaker",
    "duration_s",
]

# String columns where NaN is replaced with "NA"
_STR_META_COLS = [
    "condition",
    "lexicality",
    "length_bin",
    "morphology",
    "frequency_bin",
    "speaker",
]


# ── Manifest ──────────────────────────────────────────────────────────────────


def load_manifest(run_dir: Path) -> dict:
    """Load and validate manifest.json from a run directory."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found in {run_dir}. "
            "Expected a Step 1 extraction run directory."
        )
    with open(manifest_path) as f:
        manifest = json.load(f)
    required_keys = {"run_id", "run_params", "items"}
    missing = required_keys - manifest.keys()
    if missing:
        raise ValueError(f"manifest.json is missing required keys: {missing}")
    return manifest


# ── Metadata ──────────────────────────────────────────────────────────────────


def resolve_dataset_csv(manifest: dict, repo_root: Path) -> Path:
    """Resolve the dataset CSV path recorded in the manifest."""
    dataset_rel = manifest["run_params"]["dataset_path"]
    dataset_abs = (repo_root / dataset_rel).resolve()
    if not dataset_abs.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found: {dataset_abs}\n"
            f"(recorded in manifest as '{dataset_rel}')"
        )
    return dataset_abs


def load_trial_metadata(manifest: dict, repo_root: Path) -> pd.DataFrame:
    """Load trial-level metadata from the dataset CSV.

    Rows are ordered to match manifest.items iteration order.
    NaN values in string columns are replaced with 'NA'.

    Returns a DataFrame with columns: item_id, condition, lexicality,
    length_bin, morphology, frequency_bin, speaker, duration_s.
    """
    csv_path = resolve_dataset_csv(manifest, repo_root)
    df = pd.read_csv(csv_path)

    missing_cols = [c for c in _META_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset CSV is missing required columns: {missing_cols}\n"
            f"Found columns: {list(df.columns)}"
        )

    df = df[_META_COLS].copy()

    # Replace NaN in string/categorical columns with explicit "NA"
    for col in _STR_META_COLS:
        df[col] = df[col].fillna("NA").astype(str)

    if df["item_id"].duplicated().any():
        raise ValueError("Dataset CSV contains duplicate item_id values.")
    df = df.set_index("item_id")

    # Reorder to match manifest.items order
    item_ids = list(manifest["items"].keys())
    missing_items = [iid for iid in item_ids if iid not in df.index]
    if missing_items:
        raise ValueError(
            f"{len(missing_items)} item(s) from manifest not found in dataset CSV: "
            f"{missing_items[:5]}{'...' if len(missing_items) > 5 else ''}"
        )

    return df.loc[item_ids].reset_index()


# ── Activation loading ────────────────────────────────────────────────────────


def load_activation_pt(pt_path: Path) -> torch.Tensor:
    """Load a single activation .pt file and return a float32 tensor.

    Raises:
        FileNotFoundError if the file does not exist.
    """
    if not pt_path.exists():
        raise FileNotFoundError(f"Activation file not found: {pt_path}")
    return torch.load(pt_path, map_location="cpu").float()


def validate_and_load_layer(
    manifest: dict,
    run_dir: Path,
    layer: str,
) -> tuple[list[str], list[torch.Tensor]]:
    """Load all activation tensors for one layer, failing hard on any issue.

    Args:
        manifest: loaded manifest dict
        run_dir:  extraction run directory
        layer:    layer name (must be in LATENT_LAYERS)

    Returns:
        item_ids: list of item IDs in manifest order
        tensors:  list of float32 tensors, shape [C, T] each

    Raises:
        ValueError         if layer is 'decoder_out' or not in LATENT_LAYERS
        FileNotFoundError  if any .pt file is missing
        ValueError         if any tensor has a mismatched n_neurons (C dimension)
        ValueError         if any tensor is not 2D
    """
    if layer == WAVEFORM_LAYER:
        raise ValueError(
            f"Layer '{layer}' is a reconstructed waveform (shape [1, T_audio]) "
            "and is not eligible for the canonical xarray export. "
            "Only 'encoder_out' and 'decoder_in' are supported in Phase A."
        )
    if layer not in LATENT_LAYERS:
        raise ValueError(
            f"Unknown layer '{layer}'. "
            f"Eligible layers for xarray export: {sorted(LATENT_LAYERS)}."
        )

    items = manifest["items"]
    item_ids: list[str] = []
    tensors: list[torch.Tensor] = []
    reference_n_neurons: int | None = None
    n_items = len(items)

    for i, (item_id, layer_paths) in enumerate(items.items()):
        if layer not in layer_paths:
            raise FileNotFoundError(
                f"Layer '{layer}' not found in manifest for item '{item_id}'. "
                f"Available layers for this item: {list(layer_paths.keys())}"
            )

        pt_path = run_dir / layer_paths[layer]
        t = load_activation_pt(pt_path)  # raises FileNotFoundError if missing

        if t.dim() != 2:
            raise ValueError(
                f"Expected 2D tensor [C, T] for item '{item_id}', layer '{layer}', "
                f"but got {t.dim()}D tensor with shape {tuple(t.shape)}."
            )

        n_neurons = int(t.shape[0])
        if reference_n_neurons is None:
            reference_n_neurons = n_neurons
        elif n_neurons != reference_n_neurons:
            raise ValueError(
                f"n_neurons mismatch: item '{item_id}', layer '{layer}' has "
                f"{n_neurons} neurons, but the first item has {reference_n_neurons}. "
                "All items must have the same number of neurons."
            )

        item_ids.append(item_id)
        tensors.append(t)

        if (i + 1) % 30 == 0 or i == 0 or i == n_items - 1:
            print(f"  [xarray_builder] Loaded {i + 1}/{n_items}  item={item_id}")

    return item_ids, tensors


# ── Array construction ────────────────────────────────────────────────────────


def build_padded_array(
    tensors: list[torch.Tensor],
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack [C, T_i] tensors into a padded (n_trials, T_max, n_neurons) array.

    Each tensor [C, T_i] is transposed to [T_i, C] before stacking.
    Padded positions (t >= T_i) are set to NaN.

    Returns:
        padded:       ndarray shape (n_trials, T_max, n_neurons), float
        frame_counts: ndarray shape (n_trials,), int64 — valid frames per trial
    """
    frame_counts = np.array([t.shape[1] for t in tensors], dtype=np.int64)
    T_max = int(frame_counts.max())
    n_trials = len(tensors)
    n_neurons = int(tensors[0].shape[0])

    padded = np.full((n_trials, T_max, n_neurons), fill_value=np.nan, dtype=dtype)
    for i, t in enumerate(tensors):
        T_i = int(t.shape[1])
        padded[i, :T_i, :] = t.numpy().T  # [C, T] → [T, C]

    return padded, frame_counts


def build_valid_time(frame_counts: np.ndarray, T_max: int) -> np.ndarray:
    """Build a boolean valid_time mask of shape (n_trials, T_max).

    valid_time[i, t] = True iff t < frame_counts[i] (valid, non-padded).
    """
    t_idx = np.arange(T_max, dtype=np.int64)[np.newaxis, :]  # (1, T_max)
    return (t_idx < frame_counts[:, np.newaxis]).astype(bool)  # (n_trials, T_max)


def _compute_frame_rate_hz(
    frame_counts: np.ndarray, duration_s: np.ndarray
) -> float:
    """Estimate frame rate in Hz as the median of (n_frames / duration_s)."""
    rates = frame_counts.astype(float) / duration_s.astype(float)
    return float(np.median(rates))


# ── Dataset construction ──────────────────────────────────────────────────────


def build_xarray_dataset(
    padded: np.ndarray,
    valid_time: np.ndarray,
    frame_counts: np.ndarray,
    meta_df: pd.DataFrame,
    run_id: str,
    layer_name: str,
    manifest: dict,
    run_dir: Path,
) -> xr.Dataset:
    """Build the canonical xarray.Dataset from padded activations and metadata.

    Dimensions:
        trials  — one per stimulus item
        time    — seconds from word onset (empirical frame rate)
        neurons — one per codec channel

    The time axis is computed as np.arange(T_max) / frame_rate_hz where
    frame_rate_hz is the median of (n_frames / duration_s) across all trials.
    """
    n_trials, T_max, n_neurons = padded.shape
    assert valid_time.shape == (n_trials, T_max), \
        f"valid_time shape {valid_time.shape} does not match expected ({n_trials}, {T_max})"
    assert len(meta_df) == n_trials, \
        f"meta_df has {len(meta_df)} rows but expected {n_trials}"

    duration_s_values = meta_df["duration_s"].values.astype(float)
    frame_rate_hz = _compute_frame_rate_hz(frame_counts, duration_s_values)
    time_s = np.arange(T_max, dtype=np.float64) / frame_rate_hz

    ds = xr.Dataset(
        {
            "activations": xr.DataArray(
                padded,
                dims=["trials", "time", "neurons"],
                attrs={"long_name": "raw activations (NaN at padded positions)"},
            ),
            "valid_time": xr.DataArray(
                valid_time,
                dims=["trials", "time"],
                attrs={"long_name": "True where time frame is valid (not padded)"},
            ),
        },
        coords={
            "trial_id": ("trials", np.arange(n_trials, dtype=np.int64)),
            "item_id": ("trials", np.array(meta_df["item_id"].values.tolist(), dtype=object)),
            "time": ("time", time_s),
            "neuron": ("neurons", np.arange(n_neurons, dtype=np.int64)),
        },
    )

    # Trial-level string metadata as data variables
    for col in _STR_META_COLS:
        ds[col] = xr.DataArray(
            np.array(meta_df[col].values.tolist(), dtype=object),
            dims=["trials"],
            attrs={"long_name": col},
        )

    ds["duration_s"] = xr.DataArray(
        meta_df["duration_s"].values.astype(np.float64),
        dims=["trials"],
        attrs={"long_name": "audio duration in seconds"},
    )
    ds["n_valid_frames"] = xr.DataArray(
        frame_counts,
        dims=["trials"],
        attrs={"long_name": "number of valid (non-padded) time frames per trial"},
    )

    run_params = manifest["run_params"]
    ds.attrs = {
        "run_id": run_id,
        "model_name": run_params["model_name"],
        "layer_name": layer_name,
        "sample_rate": int(run_params["sample_rate"]),
        "dataset_path": run_params["dataset_path"],
        "manifest_path": str((run_dir / "manifest.json").resolve()),
        "source_activation_dir": str((run_dir / "activations").resolve()),
        "n_trials": n_trials,
        "n_neurons": n_neurons,
        "max_time": float(time_s[-1]) if T_max > 0 else 0.0,
        "padding_strategy": "end_padding_nan",
        "time_axis_description": (
            f"seconds from word onset; empirical frame rate = {frame_rate_hz:.4f} Hz "
            f"(median of n_frames/duration_s across {n_trials} trials)"
        ),
        "frame_rate_hz": float(frame_rate_hz),
    }

    return ds


# ── QC ────────────────────────────────────────────────────────────────────────


def compute_qc(
    manifest: dict,
    item_ids_loaded: list[str],
    tensors: list[torch.Tensor],
    padded: np.ndarray,
    valid_time: np.ndarray,
    layer_name: str,
    meta_df: pd.DataFrame,
) -> dict:
    """Compute a QC report dict for a given layer export.

    Note: n_missing_activation_files and n_shape_mismatches are always 0
    because validate_and_load_layer fails hard on those conditions.
    """
    frame_counts = np.array([t.shape[1] for t in tensors], dtype=np.int64)
    T_max = int(padded.shape[1])

    invalid_mask = ~valid_time  # (n_trials, T_max)
    total_trial_time_cells = int(padded.shape[0]) * T_max
    n_padded_cells = int(np.sum(invalid_mask))
    padding_fraction = float(n_padded_cells / total_trial_time_cells) if total_trial_time_cells > 0 else 0.0

    # NaN inside valid region (should never happen for clean activations)
    valid_expanded = valid_time[:, :, np.newaxis]           # (n_trials, T_max, 1)
    has_nan_inside = bool(np.any(np.isnan(padded) & valid_expanded))

    # Non-NaN outside valid region (should never happen by construction)
    invalid_expanded = invalid_mask[:, :, np.newaxis]       # (n_trials, T_max, 1)
    has_non_nan_outside = bool(np.any(~np.isnan(padded) & invalid_expanded))

    meta_item_ids = set(meta_df["item_id"].values)
    metadata_merge_success = set(item_ids_loaded).issubset(meta_item_ids)

    return {
        "run_id": manifest["run_id"],
        "layer": layer_name,
        "n_trials_loaded": len(item_ids_loaded),
        "n_trials_expected": len(manifest["items"]),
        "n_neurons": int(tensors[0].shape[0]) if tensors else 0,
        "min_frames": int(frame_counts.min()),
        "max_frames": int(frame_counts.max()),
        "mean_frames": float(frame_counts.mean()),
        "median_frames": float(np.median(frame_counts)),
        "n_missing_activation_files": 0,
        "n_shape_mismatches": 0,
        "padding_fraction": padding_fraction,
        "has_nan_inside_valid_region": has_nan_inside,
        "has_non_nan_outside_valid_region": has_non_nan_outside,
        "metadata_merge_success": metadata_merge_success,
    }


# ── High-level export ─────────────────────────────────────────────────────────


def export_layer(
    run_dir: Path,
    layer: str,
    repo_root: Path,
    output_dir: Path | None = None,
    dtype: np.dtype = np.float32,
    overwrite: bool = False,
) -> tuple[xr.Dataset, dict]:
    """Export one layer to a .nc file and return (dataset, qc_report).

    Args:
        run_dir:    path to the extraction run directory
        layer:      layer name (must be in LATENT_LAYERS)
        repo_root:  repo root, used to resolve dataset_path from manifest
        output_dir: where to write outputs (default: run_dir/xarray/)
        dtype:      numpy dtype for the padded activations (default float32)
        overwrite:  if False, raise FileExistsError if output already exists

    Returns:
        ds:  the constructed xarray.Dataset
        qc:  QC report dict

    Raises:
        ValueError       if layer is 'decoder_out' or not in LATENT_LAYERS
        FileExistsError  if nc_path already exists and overwrite=False
        FileNotFoundError if any activation .pt file is missing
        ValueError       if n_neurons is inconsistent across items
    """
    run_dir = run_dir.resolve()
    manifest = load_manifest(run_dir)
    run_id = manifest["run_id"]

    if output_dir is None:
        output_dir = run_dir / "xarray"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nc_path = output_dir / f"{layer}.nc"
    qc_path = output_dir / f"{layer}_qc.json"

    if nc_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {nc_path}\n"
            "Pass overwrite=True (or --overwrite in CLI) to replace it."
        )

    n_items = len(manifest["items"])
    print(f"[xarray_builder] Run: {run_id}  Layer: {layer}")
    print(f"[xarray_builder] Loading {n_items} activation tensors...")

    item_ids, tensors = validate_and_load_layer(manifest, run_dir, layer)

    print(f"[xarray_builder] Building padded array...")
    padded, frame_counts = build_padded_array(tensors, dtype=dtype)
    valid_time_arr = build_valid_time(frame_counts, int(padded.shape[1]))

    print(f"[xarray_builder] Loading trial metadata...")
    meta_df = load_trial_metadata(manifest, repo_root)

    print(f"[xarray_builder] Building xarray.Dataset...")
    ds = build_xarray_dataset(
        padded, valid_time_arr, frame_counts, meta_df,
        run_id=run_id,
        layer_name=layer,
        manifest=manifest,
        run_dir=run_dir,
    )

    qc = compute_qc(manifest, item_ids, tensors, padded, valid_time_arr, layer, meta_df)

    print(f"[xarray_builder] Writing {nc_path}...")
    ds.to_netcdf(nc_path, engine="h5netcdf")
    print(f"[xarray_builder] Written: {nc_path}")

    qc_path.write_text(json.dumps(qc, indent=2))
    print(f"[xarray_builder] Written: {qc_path}")

    return ds, qc