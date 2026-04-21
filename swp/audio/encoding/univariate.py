"""Orchestration for univariate Ridge encoding of audio activations.

Main entry point: run_univariate_encoding().

Pipeline
--------
1. Load xarray.Dataset from .nc file.
2. Temporal binning: (n_trials, T_max, n_neurons) → (n_trials, n_bins, n_neurons).
3. Design matrix: (n_selected, n_features) for the chosen analysis set.
4. Per-neuron Ridge regression with nested 5×5 CV (parallelisable with joblib).
5. Collect scores + weights → two DataFrames → CSV.
6. Write config.json and qc_check.json.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from swp.audio.encoding.temporal_binning import build_binned_y, bin_centers
from swp.audio.encoding.design_matrix import build_design_matrix, ENCODING_SCHEME
from swp.audio.encoding.univariate_encoder import AudioUnivariateEncoder


_DEFAULT_ALPHAS: list[float] = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


def _process_neuron(
    encoder: AudioUnivariateEncoder,
    X: np.ndarray,
    y_binned_selected: np.ndarray,
    neuron_idx: int,
    compute_fi: bool,
) -> dict:
    """Fit and predict for a single neuron. Joblib-serialisable."""
    y_neuron = y_binned_selected[:, :, neuron_idx]   # (n_selected, n_bins)
    return encoder.fit_predict(X, y_neuron, electrode=str(neuron_idx), compute_fi=compute_fi)


def run_univariate_encoding(
    xarray_path: Path,
    analysis_set: str,
    n_bins: int = 10,
    metrics: tuple[str, ...] = ("r2",),
    compute_fi: bool = False,
    n_jobs: int = 1,
    n_outer_splits: int = 5,
    n_inner_splits: int = 5,
    alphas: list[float] | None = None,
    random_state: int = 0,
    max_neurons: int | None = None,
    output_dir: Path | None = None,
    overwrite: bool = False,
) -> dict:
    """Run univariate Ridge encoding on prebinned audio activations.

    Args:
        xarray_path:   Path to .nc file produced by xarray_builder.export_layer().
        analysis_set:  'all_items' or 'words_only'.
        n_bins:        Number of temporal bins (default 10).
        metrics:       Scoring metrics — 'r2', 'spearman', 'corr'.
        compute_fi:    Compute permutation feature importance (slow).
        n_jobs:        Parallel workers for neuron loop.
        n_outer_splits: Outer CV folds.
        n_inner_splits: Inner CV folds for alpha selection.
        alphas:        Ridge alpha search space.
        random_state:  RNG seed.
        max_neurons:   If set, process only the first N neurons.
        output_dir:    Directory for CSV + JSON outputs; None → no files written.
        overwrite:     Overwrite existing outputs if True.

    Returns:
        dict with keys: 'scores_df', 'weights_df', 'config', 'qc_check'.
    """
    xarray_path = Path(xarray_path)
    if not xarray_path.exists():
        raise FileNotFoundError(f"xarray file not found: {xarray_path}")

    if alphas is None:
        alphas = _DEFAULT_ALPHAS

    # ── Load xarray ───────────────────────────────────────────────────────────

    ds = xr.open_dataset(xarray_path, engine="h5netcdf")
    run_id      = ds.attrs["run_id"]
    model_name  = ds.attrs["model_name"]
    layer       = ds.attrs["layer_name"]
    n_trials_total  = int(ds.sizes["trials"])
    n_neurons_total = int(ds.sizes["neurons"])

    # ── Validate output dir before doing work ─────────────────────────────────

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        scores_path  = output_dir / "scores.csv"
        weights_path = output_dir / "weights.csv"
        config_path  = output_dir / "config.json"
        qc_path      = output_dir / "qc_check.json"
        if not overwrite and scores_path.exists():
            raise FileExistsError(
                f"Output already exists: {scores_path}. "
                "Pass overwrite=True or --overwrite to replace."
            )

    # ── Temporal binning ──────────────────────────────────────────────────────

    y_binned = build_binned_y(ds, n_bins=n_bins)   # (n_trials, n_bins, n_neurons)
    times = bin_centers(n_bins)                     # (n_bins,) relative [0, 1]

    # ── Design matrix ─────────────────────────────────────────────────────────

    X, feature_names, trial_mask = build_design_matrix(ds, analysis_set)
    y_binned_selected = y_binned[trial_mask]        # (n_selected, n_bins, n_neurons)
    n_selected = X.shape[0]

    # ── Neuron subset ─────────────────────────────────────────────────────────

    n_neurons_process = (
        n_neurons_total if max_neurons is None else min(max_neurons, n_neurons_total)
    )
    neuron_indices = list(range(n_neurons_process))

    # ── Encoder (stateless configuration object) ──────────────────────────────

    encoder = AudioUnivariateEncoder(
        alphas=alphas,
        times=times,
        feature_names=feature_names,
        prebinned=True,
        n_outer_splits=n_outer_splits,
        n_inner_splits=n_inner_splits,
        metrics=list(metrics),
        random_state=random_state,
        n_jobs=1,           # parallelism is at neuron level below
        standardize_target=True,
        verbose=False,
    )

    # ── Main loop: one Ridge model per neuron ─────────────────────────────────

    results_list = Parallel(n_jobs=n_jobs)(
        delayed(_process_neuron)(encoder, X, y_binned_selected, n, compute_fi)
        for n in tqdm(neuron_indices, desc="Encoding neurons", unit="neuron")
    )

    # ── Collect into DataFrames ───────────────────────────────────────────────

    scores_rows: list[dict] = []
    weights_rows: list[dict] = []

    common = dict(
        run_id=run_id, model_name=model_name, layer=layer, analysis_set=analysis_set
    )

    for n_idx, res in enumerate(results_list):
        scores = res["scores"]        # {metric: {median, std, all_folds}}
        weight_mean = res["weights"]["mean"]           # (n_bins, n_features)
        weight_std  = np.std(res["weights"]["all_folds"], axis=0)  # (n_bins, n_features)

        for m in metrics:
            for b in range(n_bins):
                scores_rows.append({
                    **common,
                    "neuron":       n_idx,
                    "time_bin":     b,
                    "metric":       m,
                    "score_median": float(scores[m]["median"][b]),
                    "score_std":    float(scores[m]["std"][b]),
                })

        for b in range(n_bins):
            for j, feat in enumerate(feature_names):
                weights_rows.append({
                    **common,
                    "neuron":      n_idx,
                    "time_bin":    b,
                    "feature":     feat,
                    "weight_mean": float(weight_mean[b, j]),
                    "weight_std":  float(weight_std[b, j]),
                })

    scores_df  = pd.DataFrame(scores_rows)
    weights_df = pd.DataFrame(weights_rows)

    # ── QC check ──────────────────────────────────────────────────────────────

    n_valid_per_trial = ds["valid_time"].values.sum(axis=1)
    qc_check = {
        "n_trials_total":      n_trials_total,
        "n_trials_selected":   int(n_selected),
        "n_neurons_total":     n_neurons_total,
        "n_neurons_processed": n_neurons_process,
        "n_bins":              n_bins,
        "min_valid_frames":    int(n_valid_per_trial.min()),
        "nan_in_scores":       bool(scores_df["score_median"].isna().any()),
        "nan_in_weights":      bool(weights_df["weight_mean"].isna().any()),
        "timestamp":           datetime.now().isoformat(timespec="seconds"),
    }

    # ── Config ────────────────────────────────────────────────────────────────

    config = {
        "xarray_path":         str(xarray_path),
        "analysis_set":        analysis_set,
        "n_bins":              n_bins,
        "metrics":             list(metrics),
        "compute_fi":          compute_fi,
        "n_jobs":              n_jobs,
        "n_outer_splits":      n_outer_splits,
        "n_inner_splits":      n_inner_splits,
        "alphas":              list(alphas),
        "random_state":        random_state,
        "max_neurons":         max_neurons,
        "encoding_scheme":     ENCODING_SCHEME,
        "feature_names":       feature_names,
        "run_id":              run_id,
        "model_name":          model_name,
        "layer":               layer,
        "n_trials_total":      n_trials_total,
        "n_trials_selected":   int(n_selected),
        "n_neurons_processed": n_neurons_process,
        "n_features":          len(feature_names),
    }

    # ── Write outputs ─────────────────────────────────────────────────────────

    if output_dir is not None:
        scores_df.to_csv(scores_path, index=False)
        weights_df.to_csv(weights_path, index=False)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        with open(qc_path, "w") as f:
            json.dump(qc_check, f, indent=2)

    ds.close()

    return {
        "scores_df":  scores_df,
        "weights_df": weights_df,
        "config":     config,
        "qc_check":   qc_check,
    }