"""Audio-specific univariate encoder for prebinned activations.

Adapted from Intracranial/encoder.py (Louis Hayot / ENS-LSCP).
Source: /Users/louishayot/MVA/ENS-LSCP/Intracranial/encoder.py

Changes relative to the original
---------------------------------
- Class renamed: Encoder → AudioUnivariateEncoder.                 # AUDIO CHANGE 1
- Added `prebinned` parameter to __init__ (default True).          # AUDIO CHANGE 3
  When prebinned=True:
    * n_windows = len(times) (one window per bin centre)
    * win_centers = win_starts = win_ends = times
    * y_win = y[:, w]  (no averaging; column is already aggregated)
    * window_size / step_size are stored but unused when prebinned=True.
- Added _get_y_win(self, y, w): dispatches to y[:, w] (prebinned) or # AUDIO CHANGE 5
  _window_target(y, t0, t1) (windowed). _window_target is unchanged.
- fit / predict / compute_permutation_importance: loop changed from   # AUDIO CHANGE 6
  bounds-based to index-based using _get_y_win.
- fit_predict: added fi_n_repeats parameter (default 10).            # AUDIO CHANGE 7

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
from sklearn.linear_model import Ridge  # replace with FracRidge if installed
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from sklearn.inspection import permutation_importance


class AudioUnivariateEncoder:  # AUDIO CHANGE 1: class name
    """
    Time-resolved Ridge regression encoder for intracranial EEG data.

    Predicts neural activity from stimulus features using Ridge regression with nested
    cross-validation. Operates on sliding time windows to capture temporal dynamics of
    encoding. Supports single-electrode and multi-electrode workflows with automatic
    parallelization.

    The encoding workflow:
    1. Divides the epoch time axis into overlapping windows (window_size, step_size)
    2. For each window, averages neural activity to get one value per trial
    3. Fits Ridge regression: X (stimulus features) → y (neural activity)
    4. Uses nested CV: inner loop selects alpha, outer loop evaluates performance
    5. Returns models, coefficients, and performance metrics for each window

    Key features:
    - Nested cross-validation prevents overfitting in hyperparameter selection
    - Time-resolved: captures how encoding strength evolves over time
    - Multiple metrics: R², Spearman correlation, Pearson correlation
    - Feature importance via permutation testing
    - Parallel processing across electrodes

    Parameters
    ----------
    alphas : array-like
        Ridge regression regularization parameters to search.
        Common: np.logspace(-3, 3, 7) or np.logspace(-2, 4, 10).
    times : array-like
        Time points (in seconds) corresponding to epoch samples.
        Shape: (n_timepoints,). Typically from epochs.time.values.
    feature_names : list of str
        Names of stimulus features for interpretation and plotting.
    window_size : float, default=0.2
        Duration of sliding window in seconds. Larger windows smooth temporal dynamics.
    step_size : float, default=0.05
        Step between consecutive windows in seconds. Smaller steps = finer temporal resolution.
    n_outer_splits : int, default=5
        Number of folds in outer cross-validation (performance evaluation).
    n_inner_splits : int, default=5
        Number of folds in inner cross-validation (alpha selection).
    metrics : tuple of str, default=('r2',)
        Metrics to compute. Options: 'r2', 'spearman', 'corr'.
    random_state : int, default=0
        Random seed for reproducible CV splits.
    n_jobs : int, default=1
        Number of parallel jobs for fit_electrodes() and related methods.
        -1 uses all cores. For single electrode, keep at 1.
    standardize_target : bool, default=True
        Whether to apply RobustScaler to neural targets (y) before fitting.
    save_path : str, optional
        Path to save results (not currently implemented).
    verbose : bool, default=False
        Whether to print progress messages.

    Attributes
    ----------
    win_starts : ndarray
        Start times of each sliding window.
    win_centers : ndarray
        Center times of each sliding window (used for plotting).
    n_windows : int
        Total number of time windows.

    Examples
    --------
    **Single electrode encoding (time-resolved target):**

    >>> import numpy as np
    >>> from encoder import Encoder
    >>> from feature_builder import FeatureBuilder
    >>>
    >>> # Prepare data for one electrode
    >>> feature_builder = FeatureBuilder(analysis='encoder', experiment='houston')
    >>> X, y, feature_names = feature_builder.prepare_XY(epochs, electrode='LAH1')
    >>> # X: (n_trials, n_features) - e.g., (200, 5) for 5 stimulus features
    >>> # y: (n_trials, n_timepoints) - e.g., (200, 500) for 500 time samples
    >>>
    >>> # Configure encoder
    >>> encoder = Encoder(
    ...     alphas=np.logspace(-3, 3, 7),
    ...     times=epochs.time.values,
    ...     feature_names=feature_names,
    ...     window_size=0.2,  # 200ms windows
    ...     step_size=0.05,   # 50ms steps
    ...     metrics=('r2', 'spearman'),
    ...     n_jobs=1
    ... )
    >>>
    >>> # Fit and evaluate
    >>> results = encoder.fit_predict(X, y, electrode='LAH1')
    >>>
    >>> # Access results
    >>> print(results['scores']['r2']['median'])  # R² over time
    >>> print(results['weights']['mean'])  # Feature weights (n_windows, n_features)
    >>> print(results['times'])  # Window centers for plotting

    **Encoding without time dimension (static target):**

    >>> # Example static neural target: one value per trial (no time axis)
    >>> y_static = y.mean(axis=1)  # shape: (n_trials,)
    >>>
    >>> encoder_static = Encoder(
    ...     alphas=np.logspace(-3, 3, 7),
    ...     times=None,  # no sliding windows; single static window
    ...     feature_names=feature_names,
    ...     metrics=('r2', 'spearman'),
    ... )
    >>> static_results = encoder_static.fit_predict(X, y_static, electrode='LAH1')
    >>> print(static_results['scores']['r2']['median'])  # shape: (1,)

    **Multiple electrodes with parallelization:**

    >>> # Process all electrodes in parallel
    >>> all_results = encoder.fit_predict_electrodes(
    ...     epochs=epochs,
    ...     feature_builder=feature_builder,
    ...     electrodes='all',  # or list of electrode names
    ...     compute_fi=True,   # include permutation importance
    ... )
    >>>
    >>> # Results dict keyed by electrode name
    >>> lah1_results = all_results['LAH1']
    >>> print(lah1_results['scores']['spearman']['median'])
    >>> print(lah1_results['fi']['mean'])  # Feature importance (n_windows, n_features)

    **Incremental workflow (fit, then predict separately):**

    >>> # Fit models (stores models, X, y)
    >>> fit_results = encoder.fit(X, y, electrode='LAH1')
    >>>
    >>> # Evaluate performance
    >>> eval_results = encoder.predict(fit_results)
    >>> print(eval_results['scores']['r2']['median'])
    >>>
    >>> # Compute feature importance
    >>> fi_results = encoder.compute_permutation_importance(
    ...     fit_results,
    ...     electrode='LAH1',
    ...     n_repeats=10
    ... )
    >>> print(fi_results['mean'])  # (n_windows, n_features)

    **Visualization with Plotter:**

    >>> from plotter import Plotter
    >>>
    >>> plotter = Plotter(epochs)
    >>>
    >>> # Plot encoding performance + FI with current panel helper API
    >>> plot_configs = [
    ...     Plotter.glass_brain_panel(height=1),
    ...     Plotter.encoding_fi_panel(
    ...         results=all_results,
    ...         feature_names=feature_names,
    ...         height=3,
    ...         metric='spearman',
    ...     ),
    ... ]
    >>>
    >>> plotter.single_electrode(
    ...     electrode='LAH1',
    ...     trials=[],
    ...     plot_configs=plot_configs,
    ...     results_dict=all_results,
    ...     save_dir='encoding_plots'
    ... )

    Notes
    -----
    - **Nested CV rationale**: Inner CV selects optimal alpha without leaking test data,
      outer CV provides unbiased performance estimates.
    - **Time window aggregation**: Each window's target is the mean neural activity across
      time samples in that window, reducing temporal variance.
    - **Feature importance**: Permutation importance shuffles one feature at a time and
      measures performance drop. Higher values = more important feature.
    - **Parallelization strategy**: Set n_jobs=1 for single electrode (to avoid nested
      parallelism conflicts), n_jobs=-1 for fit_electrodes() to parallelize across electrodes.
    - **Memory considerations**: Storing all fold models can be memory-intensive for many
      electrodes. Results dict contains (n_folds × n_windows) model objects per electrode.
    - **Alternative to Ridge**: Replace with FracRidgeCV for fractional ridge regression
      which can handle multicollinearity better.

    See Also
    --------
    Decoder : Time-resolved classification for SEEG
    Searchlight : Whole-brain localization of encoding/decoding
    FeatureBuilder : Prepare stimulus features and neural targets
    """

    def __init__(
        self,
        alphas,
        times,
        feature_names,
        window_size=0.2,
        step_size=0.05,
        prebinned: bool = True,  # AUDIO CHANGE 3: prebinned param
        n_outer_splits=5,
        n_inner_splits=5,
        metrics=("r2",),
        random_state=0,
        n_jobs=1,
        standardize_target=True,
        save_path=None,
        verbose=False,
    ):
        self.alphas = np.asarray(alphas)
        self.times = None if times is None else np.array(times)
        self.feature_names = feature_names
        self.window_size = window_size
        self.step_size = step_size
        self.prebinned = prebinned  # AUDIO CHANGE 3: store prebinned
        self.n_outer_splits = n_outer_splits
        self.n_inner_splits = n_inner_splits
        self.metrics = list(metrics)
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.standardize_target = standardize_target
        self.save_path = save_path
        self.verbose = verbose

        # AUDIO CHANGE 4: prebinned init branch
        if prebinned:
            self.win_centers = self.times
            self.win_starts  = self.times
            self.win_ends    = self.times
        # different time dimension handling on whether it iEEG, model embeddings, or fMRI
        elif self.times is None:
            self.win_starts = np.array([0.0])
            self.win_ends = np.array([1.0])
            self.win_centers = np.array([0.0])
        else:
            self.win_starts = np.arange(self.times[0], self.times[-1] - window_size + step_size, step_size)
            self.win_ends = self.win_starts + window_size
            self.win_centers = self.win_starts + window_size / 2
        self.n_windows = len(self.win_starts)
    # -----------------------------------------
    # Helper methods
    # -----------------------------------------
    def _window_target(self, y, t0=None, t1=None):
        # static target: y shape (n_trials,)
        if y.ndim == 1:
            return y
        # time-resolved target: y shape (n_trials, n_timepoints)
        if self.times is None:
            return y.mean(axis=1)
        idx = (self.times >= t0) & (self.times < t1)
        return y[:, idx].mean(axis=1)

    def _get_y_win(self, y, w):  # AUDIO CHANGE 5: new method
        if self.prebinned:
            return y[:, w]
        return self._window_target(y, self.win_starts[w], self.win_ends[w])

    def _fit_inner_cv(self, X_train, y_train):
        inner_cv = KFold(n_splits=self.n_inner_splits, shuffle=True, random_state=self.random_state)
        gscv = GridSearchCV(Ridge(), param_grid={"alpha": self.alphas}, cv=inner_cv, scoring="r2", n_jobs=1)
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
            raise ValueError(f"Unknown metric {metric}")

    def _run_fold(self, X_train, y_train, X_test):
        alpha = self._fit_inner_cv(X_train, y_train)
        coef, model = self._fit_ridge(X_train, y_train, alpha)
        y_pred = model.predict(X_test)
        return y_pred, alpha, coef, model

    # =========================================
    # CORE FUNCTIONS (single electrode)
    # =========================================

    def fit(self, X, y, electrode=None):
        """
        Fit models for all time windows using nested cross-validation.

        Args:
            X: Feature matrix (n_trials, n_features)
            y: Target matrix (n_trials, n_timepoints)
            electrode: Optional electrode name for logging

        Returns:
            results: dict with models, alphas, weights, times, X, y
        """
        if self.standardize_target:
            y = RobustScaler().fit_transform(y)

        n_trials, n_features = X.shape[0], X.shape[1]
        outer_cv = KFold(n_splits=self.n_outer_splits, shuffle=True, random_state=self.random_state)

        # Storage
        fold_models = np.empty((self.n_outer_splits, self.n_windows), dtype=object)
        fold_alphas = np.zeros((self.n_outer_splits, self.n_windows))
        fold_weights = np.zeros((self.n_outer_splits, self.n_windows, n_features))

        desc = f"Fitting {electrode}" if electrode else "Fitting"
        for w in tqdm(range(self.n_windows), total=self.n_windows, desc=desc, leave=False):  # AUDIO CHANGE 6: index loop
            # idx = (self.times >= t0) & (self.times < t1)
            # y_win = y[:, idx].mean(axis=1)
            y_win = self._get_y_win(y, w)  # AUDIO CHANGE 6

            for f, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr = y_win[train_idx]

                alpha = self._fit_inner_cv(X_tr, y_tr)
                coef, model = self._fit_ridge(X_tr, y_tr, alpha)

                fold_models[f, w] = model
                fold_alphas[f, w] = alpha
                fold_weights[f, w] = coef

        results = {
            "models": fold_models,
            "alphas": {"median": np.median(fold_alphas, axis=0), "all_folds": fold_alphas},
            "weights": {"mean": np.mean(fold_weights, axis=0), "all_folds": fold_weights},
            "times": self.win_centers,
            "X": X,
            "y": y,
        }

        return results

    def predict(self, results):
        """
        Evaluate fitted models using cross-validation.

        Args:
            results: Output from fit() containing models, X, y

        Returns:
            Updated results dict with 'scores' added
        """
        X = results['X']
        y = results['y']
        models = results['models']

        n_outer_splits, n_windows = models.shape
        outer_cv = KFold(n_splits=self.n_outer_splits, shuffle=True, random_state=self.random_state)
        fold_scores = {m: np.zeros((n_outer_splits, n_windows)) for m in self.metrics}

        for w in range(n_windows):  # AUDIO CHANGE 6: index loop
            y_win = self._get_y_win(y, w)  # AUDIO CHANGE 6

            for f, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
                X_te = X[test_idx]
                y_te = y_win[test_idx]

                model = models[f, w]
                if model is not None:
                    y_pred = model.predict(X_te)
                    for m in self.metrics:
                        fold_scores[m][f, w] = self._evaluate_metric(y_te, y_pred, m)

        # Aggregate
        summary = {m: {"median": np.median(fold_scores[m], axis=0),
                       "std": np.std(fold_scores[m], axis=0),
                       "all_folds": fold_scores[m]} for m in self.metrics}

        results["scores"] = summary
        return results

    def compute_permutation_importance(self, results, electrode=None, n_repeats=10, random_state=None):
        """
        Compute permutation feature importance for a single electrode.

        Args:
            results: Output from fit() containing models, X, y
            electrode: Optional electrode name for logging
            n_repeats: Number of permutation repeats
            random_state: Random seed (uses self.random_state if None)

        Returns:
            fi_dict: dict with 'mean', 'std', 'all_folds'
        """
        if random_state is None:
            random_state = self.random_state

        models = results['models']
        X = results['X']
        y = results['y']

        n_outer_splits, n_windows = models.shape
        n_features = X.shape[1]
        fold_fi = np.zeros((n_outer_splits, n_windows, n_features))

        outer_cv = KFold(n_splits=self.n_outer_splits, shuffle=True, random_state=self.random_state)

        desc = f"FI {electrode}" if electrode else "Computing FI"
        for w in tqdm(range(n_windows), total=n_windows, desc=desc, leave=False):  # AUDIO CHANGE 6: index loop
            y_win = self._get_y_win(y, w)  # AUDIO CHANGE 6

            for f, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
                X_te = X[test_idx]
                y_te = y_win[test_idx]

                model = models[f, w]
                if model is not None:
                    fi = permutation_importance(
                        model, X_te, y_te,
                        n_repeats=n_repeats,
                        random_state=random_state
                    ).importances_mean
                    fold_fi[f, w] = fi

        fi_dict = {
            "mean": np.mean(fold_fi, axis=0),
            "std": np.std(fold_fi, axis=0),
            "all_folds": fold_fi
        }

        return fi_dict

    # =========================================
    # WRAPPER FUNCTIONS (multiple electrodes)
    # =========================================

    def fit_electrodes(self, epochs, feature_builder, electrodes='all', features_to_use=None):
        """
        Fit models for multiple electrodes in parallel.

        Args:
            epochs: Epoch data
            feature_builder: FeatureBuilder instance
            electrodes: List of electrode names, or 'all' for all electrodes
            features_to_use: Optional feature subset

        Returns:
            all_results: dict keyed by electrode name
        """
        if electrodes == 'all':
            electrodes = epochs.ch_name.values
        electrodes = list(electrodes)

        def process_electrode(elec):
            X, y, _ = feature_builder.prepare_XY(epochs, electrode=elec)
            results = self.fit(X, y, electrode=elec)
            return elec, results

        with tqdm(total=len(electrodes), desc="Fitting electrodes") as pbar:
            parallel = Parallel(n_jobs=self.n_jobs)
            tasks = [delayed(process_electrode)(elec) for elec in electrodes]
            results_list = parallel(tasks)
            pbar.update(len(electrodes))

        all_results = dict(results_list)
        return all_results

    def predict_electrodes(self, all_results, electrodes='all'):
        """
        Evaluate fitted models for multiple electrodes in parallel.

        Args:
            all_results: Dict of fit results keyed by electrode
            electrodes: List of electrode names, or 'all' for all results

        Returns:
            Updated all_results with scores added
        """
        if electrodes == 'all':
            electrodes = list(all_results.keys())
        electrodes = list(electrodes)

        def process_electrode(elec):
            results = self.predict(all_results[elec])
            return elec, results

        with tqdm(total=len(electrodes), desc="Predicting electrodes") as pbar:
            parallel = Parallel(n_jobs=self.n_jobs)
            tasks = [delayed(process_electrode)(elec) for elec in electrodes]
            results_list = parallel(tasks)
            pbar.update(len(electrodes))

        # Update in place
        for elec, results in results_list:
            all_results[elec] = results

        return all_results

    def compute_permutation_importance_electrodes(self, all_results, electrodes='all',
                                                   n_repeats=10, random_state=None):
        """
        Compute permutation importance for multiple electrodes in parallel.

        Args:
            all_results: Dict of fit results keyed by electrode
            electrodes: List of electrode names, or 'all' for all results
            n_repeats: Number of permutation repeats
            random_state: Random seed

        Returns:
            Updated all_results with 'fi' added
        """
        if electrodes == 'all':
            electrodes = list(all_results.keys())
        electrodes = list(electrodes)

        if random_state is None:
            random_state = self.random_state

        def process_electrode(elec):
            fi_results = self.compute_permutation_importance(
                all_results[elec],
                electrode=elec,
                n_repeats=n_repeats,
                random_state=random_state
            )
            return elec, fi_results

        with tqdm(total=len(electrodes), desc="Computing FI") as pbar:
            parallel = Parallel(n_jobs=self.n_jobs)
            tasks = [delayed(process_electrode)(elec) for elec in electrodes]
            fi_list = parallel(tasks)
            pbar.update(len(electrodes))

        # Update in place
        for elec, fi_results in fi_list:
            all_results[elec]['fi'] = fi_results

        return all_results

    # =========================================
    # CONVENIENCE FUNCTIONS
    # =========================================

    def fit_predict(self, X, y, electrode=None, compute_fi=False, fi_n_repeats=10):  # AUDIO CHANGE 7: fi_n_repeats
        """
        Combined fit and predict for single electrode (backward compatible).

        Args:
            X: Feature matrix
            y: Target matrix
            electrode: Optional electrode name
            compute_fi: Whether to compute permutation importance
            fi_n_repeats: Number of permutation repeats (audio addition)

        Returns:
            results dict
        """
        results = self.fit(X, y, electrode=electrode)
        results = self.predict(results)

        if compute_fi:
            fi_results = self.compute_permutation_importance(results, electrode=electrode, n_repeats=fi_n_repeats)  # AUDIO CHANGE 7
            results["fi"] = fi_results

        return results

    def fit_predict_electrodes(self, epochs, feature_builder, electrodes='all',
                               features_to_use=None, compute_fi=False):
        """
        Combined fit and predict for multiple electrodes (backward compatible).

        Args:
            epochs: Epoch data
            feature_builder: FeatureBuilder instance
            electrodes: List of electrode names, or 'all'
            features_to_use: Optional feature subset
            compute_fi: Whether to compute permutation importance

        Returns:
            all_results: dict keyed by electrode name
        """
        if electrodes == 'all':
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

        all_results = dict(results_list)
        return all_results