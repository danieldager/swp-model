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

Filtering
---------
An optional trial_filter dict further restricts the trial set after the
analysis_set mask. E.g. {"length_bin": "short"} selects only short trials.
Any feature that becomes constant after filtering is automatically dropped
from X with a warning. Dropped features are returned in dropped_features.

Speaker covariate
-----------------
When speaker_covariate=True, 'speaker' is appended as the last feature column
(MALE=-1 / FEMALE=+1), treated as a nuisance regressor.

Usage
-----
    X, feature_names, trial_mask, dropped = build_design_matrix(
        ds, analysis_set="all_items",
        trial_filter={"length_bin": "short"},
    )
    # X: (n_selected_trials, n_features), float32
    # feature_names: list of str (active features, after dropping constants)
    # trial_mask: bool array (n_trials,) — True for selected trials
    # dropped: list of dicts describing any dropped constant features
"""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr

# Effect coding maps: label → value
_CODING: dict[str, dict[str, float]] = {
    "lexicality":    {"nonword": -1.0, "word":    +1.0},
    "length_bin":    {"short":   -1.0, "long":    +1.0},
    "morphology":    {"simple":  -1.0, "complex": +1.0},
    "frequency_bin": {"low":     -1.0, "high":    +1.0},
    "speaker":       {"MALE":    -1.0, "FEMALE":  +1.0},
}

_ANALYSIS_FEATURES: dict[str, list[str]] = {
    "all_items":  ["lexicality", "length_bin", "morphology"],
    "words_only": ["frequency_bin", "length_bin", "morphology"],
}

ENCODING_SCHEME = "effect_-1_+1"


def build_design_matrix(
    ds: xr.Dataset,
    analysis_set: str,
    trial_filter: dict[str, str] | None = None,
    speaker_covariate: bool = False,
) -> tuple[np.ndarray, list[str], np.ndarray, list[dict]]:
    """Build X (n_selected_trials, n_features) for the given analysis set.

    Args:
        ds:               xarray.Dataset from xarray_builder.export_layer().
                          Must contain data variables for metadata columns.
        analysis_set:     'all_items' or 'words_only'.
        trial_filter:     optional {column: value} dict applied on top of the
                          analysis_set mask. E.g. {"length_bin": "short"}.
        speaker_covariate: if True, append 'speaker' as the last feature column
                          (MALE=-1 / FEMALE=+1), treated as a nuisance regressor.

    Returns:
        X:               float32 ndarray, shape (n_selected, n_features). No NaN.
        feature_names:   list of str — active features after dropping constants.
        trial_mask:      bool ndarray, shape (n_trials,). True for selected trials.
        dropped_features: list of dicts for features dropped as constant:
                          [{"feature", "reason", "constant_value", "original_label"}].

    Raises:
        ValueError  if analysis_set is unknown.
        ValueError  if any label in the data is not in the coding map.
        ValueError  if trial_filter + analysis_set produces 0 selected trials.
        ValueError  if all features are constant after filtering.
        ValueError  if X contains NaN after construction.
    """
    if analysis_set not in _ANALYSIS_FEATURES:
        raise ValueError(
            f"Unknown analysis_set '{analysis_set}'. "
            f"Valid options: {list(_ANALYSIS_FEATURES.keys())}."
        )

    feature_names: list[str] = list(_ANALYSIS_FEATURES[analysis_set])
    if speaker_covariate:
        feature_names = feature_names + ["speaker"]

    n_trials = int(ds.sizes["trials"])

    # ── Base trial mask (analysis_set) ────────────────────────────────────────
    if analysis_set == "words_only":
        lex_values = np.array(ds["lexicality"].values)
        trial_mask = lex_values == "word"
    else:
        trial_mask = np.ones(n_trials, dtype=bool)

    # ── Additional trial filter ───────────────────────────────────────────────
    if trial_filter:
        for col, val in trial_filter.items():
            col_values = np.array(ds[col].values)
            trial_mask = trial_mask & (col_values == val)

    selected_idx = np.where(trial_mask)[0]
    n_selected = len(selected_idx)

    if n_selected == 0:
        raise ValueError(
            f"analysis_set='{analysis_set}' with trial_filter={trial_filter!r} "
            "produced 0 selected trials. Check filter values match the data."
        )

    # ── Build X ───────────────────────────────────────────────────────────────
    X = np.empty((n_selected, len(feature_names)), dtype=np.float32)

    for j, feat in enumerate(feature_names):
        raw_values = np.array(ds[feat].values)
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

    # ── Detect and drop constant columns ─────────────────────────────────────
    dropped_features: list[dict] = []
    keep_cols: list[int] = []

    for j, feat in enumerate(feature_names):
        col_vals = X[:, j]
        if len(np.unique(col_vals)) == 1:
            constant_val = float(col_vals[0])
            inv_coding = {v: k for k, v in _CODING[feat].items()}
            original_label = inv_coding.get(constant_val, str(constant_val))
            dropped_features.append({
                "feature": feat,
                "reason": "constant_after_filter",
                "constant_value": constant_val,
                "original_label": original_label,
            })
            msg = (
                f"[design_matrix] WARNING: dropping constant feature '{feat}' "
                f"(all {n_selected} selected trials have label '{original_label}' "
                f"→ coded {constant_val:+.1f}). "
                "Expected for length-filtered analysis views."
            )
            warnings.warn(msg, stacklevel=2)
            print(msg)
        else:
            keep_cols.append(j)

    if dropped_features:
        X = X[:, keep_cols]
        feature_names = [feature_names[j] for j in keep_cols]

    if X.shape[1] == 0:
        raise ValueError(
            "All features are constant after filtering — no predictors remain. "
            f"trial_filter={trial_filter!r}, analysis_set='{analysis_set}'."
        )

    return X, feature_names, trial_mask, dropped_features