"""Audio-specific univariate encoder for prebinned activations.

Adapted from Intracranial/encoder.py (Louis Hayot / ENS-LSCP).
Source: /Users/louishayot/MVA/ENS-LSCP/Intracranial/encoder.py

Changes relative to the original
---------------------------------
- Added `prebinned` parameter to __init__ (default False, backward compatible).
- When prebinned=True:
    * n_windows = y.shape[1] (one window per column, already a bin)
    * win_centers = times  (passed directly from caller, e.g. bin_centers array)
    * y_win = y[:, w]  (no averaging, column is already aggregated)
    * The window_size / step_size parameters are stored but unused internally
      when prebinned=True.
- Multi-electrode wrappers (fit_electrodes, predict_electrodes,
  fit_predict_electrodes) depend on FeatureBuilder / epoch xarray and are
  NOT used in the audio pipeline. They are retained verbatim for
  methodological traceability.

Core contract (unchanged)
--------------------------
X : ndarray (n_trials, n_features)   — no NaN, effect-coded
y : ndarray (n_trials, n_bins)       — no NaN (prebinned) or (n_trials, n_timepoints)
times : ndarray (n_bins,) or (n_timepoints,) — relative bin centres or seconds

All downstream logic (Ridge, nested KFold 5×5, metrics, permutation
importance) is identical to the original.
"""

import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from sklearn.inspection import permutation_importance


class AudioUnivariateEncoder:
    """
    Time-resolved Ridge regression encoder for prebinned audio activations.

    Predicts activation from stimulus features using Ridge regression with nested
    cross-validation. Supports both prebinned y (one column per temporal bin) and
    the original sliding-window mode.

    Parameters
    ----------
    alphas : array-like
        Ridge regularization parameters to search.
    times : array-like
        Time axis. In prebinned mode: bin centres (e.g. [0.05, 0.15, ..., 0.95]).
        In windowed mode: epoch time samples in seconds.
    feature_names : list of str
        Names of stimulus features.
    window_size : float, default=0.2
        Window duration in seconds. Unused when prebinned=True.
    step_size : float, default=0.05
        Window step in seconds. Unused when prebinned=True.
    prebinned : bool, default=True
        If True, each column of y is treated as a pre-averaged temporal bin;
        no internal averaging is performed.
    n_outer_splits : int, default=5
    n_inner_splits : int, default=5
    metrics : tuple of str, default=('r2',)
        'r2', 'spearman', 'corr'.
    random_state : int, default=0
    n_jobs : int, default=1
    standardize_target : bool, default=True
        Apply RobustScaler to y globally before CV.
    verbose : bool, default=False
    """

    def __init__(
        self,
        alphas,
        times,
        feature_names,
        window_size=0.2,
        step_size=0.05,
        prebinned=True,
        n_outer_splits=5,
        n_inner_splits=5,
        metrics=("r2",),
        random_state=0,
        n_jobs=1,
        standardize_target=True,
        verbose=False,
    ):
        self.alphas = np.asarray(alphas)
        self.times = np.asarray(times)
        self.feature_names = feature_names
        self.window_size = window_size
        self.step_size = step_size
        self.prebinned = prebinned
        self.n_outer_splits = n_outer_splits
        self.n_inner_splits = n_inner_splits
        self.metrics = list(metrics)
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.standardize_target = standardize_target
        self.verbose = verbose

        if prebinned:
            # times is the bin_centers array; n_windows = len(times)
            self.win_centers = self.times
            self.win_starts = self.times  # kept for interface compatibility
            self.win_ends = self.times    # kept for interface compatibility
            self.n_windows = len(self.times)
        else:
            # Original sliding-window logic from encoder.py
            self.win_starts = np.arange(
                times[0], times[-1] - window_size + step_size, step_size
            )
            self.win_ends = self.win_starts + window_size
            self.win_centers = self.win_starts + window_size / 2
            self.n_windows = len(self.win_starts)

    # ------------------------------------------------------------------
    # Helpers (identical to encoder.py)
    # ------------------------------------------------------------------

    def _fit_inner_cv(self, X_train, y_train):
        inner_cv = KFold(
            n_splits=self.n_inner_splits, shuffle=True, random_state=self.random_state
        )
        gscv = GridSearchCV(
            Ridge(),
            param_grid={"alpha": self.alphas},
            cv=inner_cv,
            scoring="r2",
            n_jobs=1,
        )
        gscv.fit(X_train, y_train)
        return gscv.best_params_["alpha"]

    @staticmethod
    def _fit_ridge(X_train, y_train, alpha):
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        return model.coef_.copy(), model

    @staticmethod
    def _evaluate_metric(y_true, y_pred, metric):
        if metric == "r2":
            return r2_score(y_true, y_pred)
        elif metric == "spearman":
            return spearmanr(y_true, y_pred).correlation
        elif metric == "corr":
            return np.corrcoef(y_true, y_pred)[0, 1]
        else:
            raise ValueError(f"Unknown metric '{metric}'")

    def _get_y_win(self, y: np.ndarray, w: int) -> np.ndarray:
        """Extract the target vector for window w.

        In prebinned mode: return the w-th column directly.
        In windowed mode: average columns in [win_starts[w], win_ends[w]).
        """
        if self.prebinned:
            return y[:, w]
        else:
            t0, t1 = self.win_starts[w], self.win_ends[w]
            idx = (self.times >= t0) & (self.times < t1)
            return y[:, idx].mean(axis=1)

    # ------------------------------------------------------------------
    # Core methods (single neuron / electrode)
    # ------------------------------------------------------------------

    def fit(self, X, y, electrode=None):
        """Fit models for all time windows using nested cross-validation.

        Args:
            X : (n_trials, n_features) — effect-coded, no NaN
            y : (n_trials, n_bins) in prebinned mode — no NaN
            electrode : optional name for logging

        Returns:
            results dict with models, alphas, weights, times, X, y
        """
        if self.standardize_target:
            y = RobustScaler().fit_transform(y)

        n_features = X.shape[1]
        outer_cv = KFold(
            n_splits=self.n_outer_splits, shuffle=True, random_state=self.random_state
        )

        fold_models = np.empty((self.n_outer_splits, self.n_windows), dtype=object)
        fold_alphas = np.zeros((self.n_outer_splits, self.n_windows))
        fold_weights = np.zeros((self.n_outer_splits, self.n_windows, n_features))

        desc = f"Fitting {electrode}" if electrode else "Fitting"
        for w in tqdm(range(self.n_windows), total=self.n_windows, desc=desc, leave=False):
            y_win = self._get_y_win(y, w)

            for f, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
                X_tr = X[train_idx]
                y_tr = y_win[train_idx]

                alpha = self._fit_inner_cv(X_tr, y_tr)
                coef, model = self._fit_ridge(X_tr, y_tr, alpha)

                fold_models[f, w] = model
                fold_alphas[f, w] = alpha
                fold_weights[f, w] = coef

        return {
            "models": fold_models,
            "alphas": {
                "median": np.median(fold_alphas, axis=0),
                "all_folds": fold_alphas,
            },
            "weights": {
                "mean": np.mean(fold_weights, axis=0),
                "all_folds": fold_weights,
            },
            "times": self.win_centers,
            "X": X,
            "y": y,
        }

    def predict(self, results):
        """Evaluate fitted models on held-out folds.

        Args:
            results : output from fit()

        Returns:
            Updated results dict with 'scores' added.
        """
        X = results["X"]
        y = results["y"]
        models = results["models"]

        n_outer_splits, n_windows = models.shape
        outer_cv = KFold(
            n_splits=self.n_outer_splits, shuffle=True, random_state=self.random_state
        )
        fold_scores = {m: np.zeros((n_outer_splits, n_windows)) for m in self.metrics}

        for w in range(n_windows):
            y_win = self._get_y_win(y, w)

            for f, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
                X_te = X[test_idx]
                y_te = y_win[test_idx]

                model = models[f, w]
                if model is not None:
                    y_pred = model.predict(X_te)
                    for m in self.metrics:
                        fold_scores[m][f, w] = self._evaluate_metric(y_te, y_pred, m)

        results["scores"] = {
            m: {
                "median": np.median(fold_scores[m], axis=0),
                "std": np.std(fold_scores[m], axis=0),
                "all_folds": fold_scores[m],
            }
            for m in self.metrics
        }
        return results

    def compute_permutation_importance(
        self, results, electrode=None, n_repeats=10, random_state=None
    ):
        """Compute permutation feature importance on held-out folds.

        Args:
            results    : output from fit()
            electrode  : optional name for logging
            n_repeats  : number of permutation shuffles per feature
            random_state : seed (uses self.random_state if None)

        Returns:
            fi_dict with 'mean', 'std', 'all_folds'
        """
        if random_state is None:
            random_state = self.random_state

        models = results["models"]
        X = results["X"]
        y = results["y"]

        n_outer_splits, n_windows = models.shape
        n_features = X.shape[1]
        fold_fi = np.zeros((n_outer_splits, n_windows, n_features))

        outer_cv = KFold(
            n_splits=self.n_outer_splits, shuffle=True, random_state=self.random_state
        )

        desc = f"FI {electrode}" if electrode else "Computing FI"
        for w in tqdm(range(n_windows), total=n_windows, desc=desc, leave=False):
            y_win = self._get_y_win(y, w)

            for f, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
                X_te = X[test_idx]
                y_te = y_win[test_idx]

                model = models[f, w]
                if model is not None:
                    fi = permutation_importance(
                        model,
                        X_te,
                        y_te,
                        n_repeats=n_repeats,
                        random_state=random_state,
                    ).importances_mean
                    fold_fi[f, w] = fi

        return {
            "mean": np.mean(fold_fi, axis=0),
            "std": np.std(fold_fi, axis=0),
            "all_folds": fold_fi,
        }

    def fit_predict(self, X, y, electrode=None, compute_fi=False):
        """Combined fit and predict for a single neuron / electrode.

        Args:
            X          : (n_trials, n_features)
            y          : (n_trials, n_bins) in prebinned mode
            electrode  : optional name
            compute_fi : whether to compute permutation importance

        Returns:
            results dict
        """
        results = self.fit(X, y, electrode=electrode)
        results = self.predict(results)
        if compute_fi:
            results["fi"] = self.compute_permutation_importance(
                results, electrode=electrode
            )
        return results

    # ------------------------------------------------------------------
    # Multi-electrode wrappers (retained from encoder.py, not used in
    # the audio pipeline but kept for methodological traceability)
    # ------------------------------------------------------------------

    def fit_predict_electrodes(
        self, epochs, feature_builder, electrodes="all", features_to_use=None, compute_fi=False
    ):
        """Process multiple electrodes in parallel (Intracranial API).

        Not used in the audio pipeline. Retained for compatibility.
        """
        if electrodes == "all":
            electrodes = epochs.ch_name.values
        electrodes = list(electrodes)

        def process_electrode(elec):
            X, y, _ = feature_builder.prepare_XY(epochs, electrode=elec)
            results = self.fit_predict(X, y, electrode=elec, compute_fi=compute_fi)
            return elec, results

        with tqdm(total=len(electrodes), desc="Processing electrodes") as pbar:
            parallel = Parallel(n_jobs=self.n_jobs)
            tasks = [delayed(process_electrode)(elec) for elec in electrodes]
            results_list = parallel(tasks)
            pbar.update(len(electrodes))

        return dict(results_list)