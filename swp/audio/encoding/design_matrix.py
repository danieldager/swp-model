"""Design matrix construction for univariate encoding analyses.

Builds X (n_trials, n_features) from trial-level metadata stored in an
xarray.Dataset produced by xarray_builder.py.

Encoding scheme
---------------
Effect coding: −1 / +1 for binary categorical factors.
No intercept column is added; Ridge uses fit_intercept=True by default.
X is not standardised (consistent with encoder.py / univariate_encoder.py).

Analysis sets
-------------
all_items:   lexicality, length_bin, morphology  (n=180 for male subset)
words_only:  frequency_bin, length_bin, morphology  (n=90, words only)

Usage
-----
    X, feature_names, trial_mask = build_design_matrix(ds, analysis_set="all_items")
    # X: (n_selected_trials, n_features), float32
    # feature_names: list of str
    # trial_mask: bool array (n_trials,) — True for selected trials
"""

from __future__ import annotations

import numpy as np
import xarray as xr

# Effect coding maps: label → value
_CODING: dict[str, dict[str, float]] = {
    "lexicality":    {"nonword": -1.0, "word":    +1.0},
    "length_bin":    {"short":   -1.0, "long":    +1.0},
    "morphology":    {"simple":  -1.0, "complex": +1.0},
    "frequency_bin": {"low":     -1.0, "high":    +1.0},
}

_ANALYSIS_FEATURES: dict[str, list[str]] = {
    "all_items":  ["lexicality", "length_bin", "morphology"],
    "words_only": ["frequency_bin", "length_bin", "morphology"],
}

ENCODING_SCHEME = "effect_-1_+1"


def build_design_matrix(
    ds: xr.Dataset,
    analysis_set: str,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build X (n_selected_trials, n_features) for the given analysis set.

    Args:
        ds:            xarray.Dataset from xarray_builder.export_layer().
                       Must contain data variables for metadata columns.
        analysis_set:  'all_items' or 'words_only'.

    Returns:
        X:             float32 ndarray, shape (n_selected_trials, n_features).
                       Effect-coded, no NaN.
        feature_names: list of str, length n_features.
        trial_mask:    bool ndarray, shape (n_trials,).
                       True for trials included in this analysis set.

    Raises:
        ValueError  if analysis_set is unknown.
        ValueError  if any label in the data is not in the coding map.
        ValueError  if X contains NaN after construction.
    """
    if analysis_set not in _ANALYSIS_FEATURES:
        raise ValueError(
            f"Unknown analysis_set '{analysis_set}'. "
            f"Valid options: {list(_ANALYSIS_FEATURES.keys())}."
        )

    feature_names = _ANALYSIS_FEATURES[analysis_set]
    n_trials = int(ds.sizes["trials"])

    # Build trial mask
    if analysis_set == "words_only":
        lex_values = np.array(ds["lexicality"].values)
        trial_mask = lex_values == "word"
    else:
        trial_mask = np.ones(n_trials, dtype=bool)

    selected_idx = np.where(trial_mask)[0]
    n_selected = len(selected_idx)

    if n_selected == 0:
        raise ValueError(
            f"analysis_set='{analysis_set}' produced 0 selected trials. "
            "Check that the xarray contains the expected metadata."
        )

    X = np.empty((n_selected, len(feature_names)), dtype=np.float32)

    for j, feat in enumerate(feature_names):
        raw_values = np.array(ds[feat].values)  # (n_trials,) object array
        coding_map = _CODING[feat]

        unknown = set(raw_values[selected_idx]) - set(coding_map.keys())
        if unknown:
            raise ValueError(
                f"Feature '{feat}' contains unexpected labels: {unknown}. "
                f"Expected: {set(coding_map.keys())}."
            )

        X[:, j] = np.array(
            [coding_map[v] for v in raw_values[selected_idx]], dtype=np.float32
        )

    if np.isnan(X).any():
        raise ValueError("Design matrix X contains NaN after construction.")

    return X, feature_names, trial_mask