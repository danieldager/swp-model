"""Temporal binning: xarray activation Dataset → prebinned y array.

Converts a canonical xarray.Dataset (trials, time, neurons) produced by
xarray_builder.py into a dense numpy array (n_trials, n_bins, n_neurons)
with no NaN.

Strategy
--------
For each trial, only the valid (non-padded) frames are used. They are split
into n_bins contiguous groups using np.array_split, which handles unequal
group sizes gracefully (floor/ceil), with no interpolation or resampling.
The activation in each bin is the mean across the frames in that bin.

Time axis returned
------------------
bin_centers : ndarray shape (n_bins,)
    Centre of each bin in relative time [0, 1].
    Example for n_bins=10: [0.05, 0.15, ..., 0.95]

This array should be passed as `times` to AudioUnivariatEncoder (with
prebinned=True), together with window_size = 1/n_bins and step_size = 1/n_bins.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def build_binned_y(
    ds: xr.Dataset,
    n_bins: int = 10,
) -> np.ndarray:
    """Convert xarray activations to a prebinned dense array.

    Args:
        ds:      xarray.Dataset produced by xarray_builder.export_layer().
                 Must have data variables 'activations' (trials, time, neurons)
                 and 'valid_time' (trials, time).
        n_bins:  Number of relative-time bins. Must be <= min(n_valid_frames).

    Returns:
        y_binned: ndarray shape (n_trials, n_bins, n_neurons), float32.
                  No NaN: each entry is the mean of valid frames in that bin.

    Raises:
        ValueError  if n_bins > min(n_valid_frames) across all trials.
        ValueError  if y_binned contains NaN after construction (sanity check).
    """
    activations = ds["activations"].values   # (n_trials, T_max, n_neurons)
    valid_time = ds["valid_time"].values     # (n_trials, T_max) bool

    n_trials, T_max, n_neurons = activations.shape

    # Check feasibility
    n_valid_per_trial = valid_time.sum(axis=1).astype(np.int64)  # (n_trials,)
    min_valid = int(n_valid_per_trial.min())
    if n_bins > min_valid:
        raise ValueError(
            f"n_bins={n_bins} > min(n_valid_frames)={min_valid}. "
            "Reduce n_bins or check that the xarray was built correctly."
        )

    y_binned = np.empty((n_trials, n_bins, n_neurons), dtype=np.float32)

    for i in range(n_trials):
        valid_indices = np.where(valid_time[i])[0]  # frame indices that are valid
        bins = np.array_split(valid_indices, n_bins)  # n_bins groups, unequal OK
        for b, bin_idx in enumerate(bins):
            y_binned[i, b, :] = activations[i, bin_idx, :].mean(axis=0)

    if np.isnan(y_binned).any():
        raise ValueError(
            "y_binned contains NaN after construction. "
            "This should not happen — check that activations have no NaN "
            "inside the valid region (see qc_check.json)."
        )

    return y_binned


def bin_centers(n_bins: int) -> np.ndarray:
    """Return relative-time bin centres for n_bins bins over [0, 1].

    Example: n_bins=10 → [0.05, 0.15, ..., 0.95]
    """
    return np.array([b / n_bins + 1 / (2 * n_bins) for b in range(n_bins)])