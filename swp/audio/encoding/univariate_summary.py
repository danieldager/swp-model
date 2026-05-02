"""Summary statistics over univariate encoding results.

Reads outputs from one or more run_univariate_encoding() output directories
and produces aggregated CSVs. No statistical tests, no figures, no clustering.

Four base outputs (always produced)
------------------------------------
A. score_summary_by_time.csv
B. weight_summary_by_feature_time.csv
C. top_units_by_feature.csv
D. global_feature_ranking.csv

Three FI outputs (produced only when feature_importance.csv is present)
------------------------------------------------------------------------
E. fi_summary_by_feature_time.csv
F. global_fi_ranking.csv
G. top_units_by_fi.csv

Main entry point: summarize()
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ── I/O ───────────────────────────────────────────────────────────────────────


def load_run_dir(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, str]:
    """Load scores.csv, weights.csv, config.json from a run output directory.

    Also derives analysis_view from the directory name (e.g. 'all_items_short_only').
    This is the intended naming tag that distinguishes filter / speaker-covariate
    variants that share the same analysis_set value in config.json.

    Raises FileNotFoundError if any expected file is absent.
    Returns (scores_df, weights_df, config_dict, analysis_view).
    """
    run_dir = Path(run_dir)
    scores_path  = run_dir / "scores.csv"
    weights_path = run_dir / "weights.csv"
    config_path  = run_dir / "config.json"

    for p in (scores_path, weights_path, config_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Expected file not found: {p}\n"
                "Make sure this is a valid run_univariate_encoding output directory."
            )

    scores  = pd.read_csv(scores_path)
    weights = pd.read_csv(weights_path)
    with open(config_path) as f:
        config = json.load(f)

    analysis_view = run_dir.name   # e.g. "all_items_short_only_speakerctrl"
    return scores, weights, config, analysis_view


# ── Compute A ─────────────────────────────────────────────────────────────────


def compute_score_summary_by_time(scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate score_median across neurons per (run, layer, analysis_set, metric, time_bin).

    Returns one row per group with: mean, median, std, q25, q75, frac_positive.
    """
    group_keys = [
        "run_id", "analysis_view", "model_name", "layer", "analysis_set",
        "metric", "time_bin", "relative_time",
    ]
    tmp = scores.assign(is_positive=(scores["score_median"] > 0).astype(float))
    g     = tmp.groupby(group_keys)["score_median"]
    g_pos = tmp.groupby(group_keys)["is_positive"]

    return pd.DataFrame({
        "mean_score":          g.mean(),
        "median_score":        g.median(),
        "std_score":           g.std(),
        "q25_score":           g.quantile(0.25),
        "q75_score":           g.quantile(0.75),
        "frac_positive_score": g_pos.mean(),
    }).reset_index()


# ── Compute B ─────────────────────────────────────────────────────────────────


def compute_weight_summary_by_feature_time(weights: pd.DataFrame) -> pd.DataFrame:
    """Aggregate weight_mean across neurons per (run, layer, analysis_set, feature, time_bin).

    Returns one row per group with signed and absolute weight stats.
    """
    group_keys = [
        "run_id", "analysis_view", "model_name", "layer", "analysis_set",
        "feature", "time_bin", "relative_time",
    ]
    tmp = weights.assign(
        abs_weight=(weights["weight_mean"].abs()),
        is_positive=(weights["weight_mean"] > 0).astype(float),
    )
    g     = tmp.groupby(group_keys)["weight_mean"]
    g_abs = tmp.groupby(group_keys)["abs_weight"]
    g_pos = tmp.groupby(group_keys)["is_positive"]

    return pd.DataFrame({
        "mean_weight":          g.mean(),
        "median_weight":        g.median(),
        "mean_abs_weight":      g_abs.mean(),
        "median_abs_weight":    g_abs.median(),
        "std_weight":           g.std(),
        "q25_weight":           g.quantile(0.25),
        "q75_weight":           g.quantile(0.75),
        "frac_positive_weight": g_pos.mean(),
    }).reset_index()


# ── Compute C ─────────────────────────────────────────────────────────────────


def compute_top_units_by_feature(weights: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    """For each feature, return top-k neurons ranked by max |weight| across time bins.

    For each (run, layer, analysis_set, feature, neuron): find the single time bin
    that carries the highest |weight|. Then keep the top-k neurons per feature group.

    Columns:
        run_id, model_name, layer, analysis_set, feature, neuron,
        max_abs_weight, time_bin_at_max, relative_time_at_max, signed_weight_at_max
    """
    identity_keys      = ["run_id", "analysis_view", "model_name", "layer", "analysis_set"]
    feature_neuron_keys = identity_keys + ["feature", "neuron"]

    tmp = weights.assign(abs_weight=weights["weight_mean"].abs())

    # Per (identity, feature, neuron): row with maximum |weight|
    idx_at_max = tmp.groupby(feature_neuron_keys)["abs_weight"].idxmax()
    peak = (
        tmp.loc[idx_at_max]
        .rename(columns={
            "abs_weight":    "max_abs_weight",
            "time_bin":      "time_bin_at_max",
            "relative_time": "relative_time_at_max",
            "weight_mean":   "signed_weight_at_max",
        })[identity_keys + ["feature", "neuron",
                             "max_abs_weight", "time_bin_at_max",
                             "relative_time_at_max", "signed_weight_at_max"]]
    )

    # Keep top-k within each (identity, feature)
    rank_keys = identity_keys + ["feature"]
    result = (
        peak
        .sort_values(rank_keys + ["max_abs_weight"],
                     ascending=[True] * len(rank_keys) + [False])
        .groupby(rank_keys, sort=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    return result


# ── Compute C (weights) ───────────────────────────────────────────────────────


def compute_global_feature_ranking(weights: pd.DataFrame) -> pd.DataFrame:
    """Global summary per feature: mean |weight|, max |weight|, mean signed weight, rank.

    Rank is computed within each (run_id, model_name, layer, analysis_set) group,
    ascending from the feature with the highest mean absolute weight (rank 1).
    """
    identity_keys = ["run_id", "analysis_view", "model_name", "layer", "analysis_set"]
    group_keys    = identity_keys + ["feature"]

    tmp   = weights.assign(abs_weight=weights["weight_mean"].abs())
    g     = tmp.groupby(group_keys)["weight_mean"]
    g_abs = tmp.groupby(group_keys)["abs_weight"]

    df = pd.DataFrame({
        "mean_abs_weight_over_time_and_neurons": g_abs.mean(),
        "max_abs_weight":                        g_abs.max(),
        "mean_signed_weight":                    g.mean(),
    }).reset_index()

    df["rank_by_mean_abs_weight"] = (
        df.groupby(identity_keys)["mean_abs_weight_over_time_and_neurons"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    return (
        df.sort_values(identity_keys + ["rank_by_mean_abs_weight"])
        .reset_index(drop=True)
    )


# ── Compute D (weights) ───────────────────────────────────────────────────────

# (section header kept for symmetry — function below unchanged)


# ── Compute E–G : FI summaries ────────────────────────────────────────────────


def compute_fi_summary_by_feature_time(fi: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fi_mean across neurons per (run, layer, analysis_set, feature, time_bin).

    Returns one row per group with signed and absolute FI stats.
    """
    group_keys = [
        "run_id", "analysis_view", "model_name", "layer", "analysis_set",
        "feature", "time_bin", "relative_time",
    ]
    tmp = fi.assign(
        abs_fi=(fi["fi_mean"].abs()),
        is_positive=(fi["fi_mean"] > 0).astype(float),
    )
    g     = tmp.groupby(group_keys)["fi_mean"]
    g_abs = tmp.groupby(group_keys)["abs_fi"]
    g_pos = tmp.groupby(group_keys)["is_positive"]

    return pd.DataFrame({
        "mean_fi":          g.mean(),
        "median_fi":        g.median(),
        "mean_abs_fi":      g_abs.mean(),
        "median_abs_fi":    g_abs.median(),
        "std_fi":           g.std(),
        "q25_fi":           g.quantile(0.25),
        "q75_fi":           g.quantile(0.75),
        "frac_positive_fi": g_pos.mean(),
    }).reset_index()


def compute_global_fi_ranking(fi: pd.DataFrame) -> pd.DataFrame:
    """Global summary per feature: mean/abs FI over time × neurons, ranked.

    Two ranks:
      rank_by_mean_fi      — descending by mean signed FI (rank 1 = highest)
      rank_by_mean_abs_fi  — descending by mean |FI| (rank 1 = most important regardless of sign)
    """
    identity_keys = ["run_id", "analysis_view", "model_name", "layer", "analysis_set"]
    group_keys    = identity_keys + ["feature"]

    tmp   = fi.assign(abs_fi=fi["fi_mean"].abs())
    g     = tmp.groupby(group_keys)["fi_mean"]
    g_abs = tmp.groupby(group_keys)["abs_fi"]

    df = pd.DataFrame({
        "mean_fi_over_time_and_neurons":      g.mean(),
        "mean_abs_fi_over_time_and_neurons":  g_abs.mean(),
        "median_fi_over_time_and_neurons":    g.median(),
        "max_fi":                             g.max(),
        "mean_signed_fi":                     g.mean(),   # alias of mean_fi; kept for spec compliance
    }).reset_index()

    df["rank_by_mean_fi"] = (
        df.groupby(identity_keys)["mean_fi_over_time_and_neurons"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    df["rank_by_mean_abs_fi"] = (
        df.groupby(identity_keys)["mean_abs_fi_over_time_and_neurons"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    return (
        df.sort_values(identity_keys + ["rank_by_mean_fi"])
        .reset_index(drop=True)
    )


def compute_top_units_by_fi(fi: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    """For each feature, return top-k neurons ranked by max fi_mean across time bins.

    Columns:
        run_id, model_name, layer, analysis_set, feature, neuron,
        max_fi, time_bin_at_max, relative_time_at_max, signed_fi_at_max
    """
    identity_keys    = ["run_id", "analysis_view", "model_name", "layer", "analysis_set"]
    fi_neuron_keys   = identity_keys + ["feature", "neuron"]

    idx_at_max = fi.groupby(fi_neuron_keys)["fi_mean"].idxmax()
    peak = (
        fi.loc[idx_at_max]
        .rename(columns={
            "fi_mean":       "max_fi",
            "time_bin":      "time_bin_at_max",
            "relative_time": "relative_time_at_max",
        })
    )
    # signed_fi_at_max == max_fi (both are fi_mean at the peak time bin)
    peak = peak.assign(signed_fi_at_max=peak["max_fi"])
    peak = peak[
        identity_keys + ["feature", "neuron",
                          "max_fi", "time_bin_at_max",
                          "relative_time_at_max", "signed_fi_at_max"]
    ]

    rank_keys = identity_keys + ["feature"]
    return (
        peak
        .sort_values(rank_keys + ["max_fi"],
                     ascending=[True] * len(rank_keys) + [False])
        .groupby(rank_keys, sort=False)
        .head(top_k)
        .reset_index(drop=True)
    )


# ── Main entry point ──────────────────────────────────────────────────────────


def summarize(
    input_dirs: list[Path],
    output_dir: Path,
    top_k: int = 20,
    overwrite: bool = False,
) -> dict[str, pd.DataFrame | None]:
    """Load all input run directories, compute summaries, write to output_dir.

    Always produces four base CSVs (A–D). Additionally produces three FI CSVs
    (E–G) when feature_importance.csv is found in at least one input directory.
    Inputs missing feature_importance.csv are skipped (with a warning) for FI
    summaries only; their base data (scores, weights) is still included.

    Args:
        input_dirs: List of run_univariate_encoding() output directories.
        output_dir: Directory where CSV files will be written.
        top_k:      Top-k neurons to keep per feature in top_units_*.csv.
        overwrite:  If False and any output file exists, raise FileExistsError.

    Returns:
        dict with keys: 'score_summary', 'weight_summary', 'top_units',
        'feature_ranking', 'fi_summary' (or None), 'fi_ranking' (or None),
        'top_units_fi' (or None).
    """
    output_dir = Path(output_dir)

    # Probe which inputs have FI before loading anything
    fi_present = {Path(d): (Path(d) / "feature_importance.csv").exists()
                  for d in input_dirs}
    any_fi = any(fi_present.values())

    out_paths: dict[str, Path] = {
        "score_summary":   output_dir / "score_summary_by_time.csv",
        "weight_summary":  output_dir / "weight_summary_by_feature_time.csv",
        "top_units":       output_dir / "top_units_by_feature.csv",
        "feature_ranking": output_dir / "global_feature_ranking.csv",
    }
    if any_fi:
        out_paths.update({
            "fi_summary":   output_dir / "fi_summary_by_feature_time.csv",
            "fi_ranking":   output_dir / "global_fi_ranking.csv",
            "top_units_fi": output_dir / "top_units_by_fi.csv",
        })

    if not overwrite:
        existing = [p for p in out_paths.values() if p.exists()]
        if existing:
            raise FileExistsError(
                f"Output files already exist: {[p.name for p in existing]}. "
                "Pass overwrite=True / --overwrite to replace."
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    all_scores:  list[pd.DataFrame] = []
    all_weights: list[pd.DataFrame] = []
    all_fi:      list[pd.DataFrame] = []

    for d in input_dirs:
        d = Path(d)
        scores, weights, config, analysis_view = load_run_dir(d)
        scores["analysis_view"]  = analysis_view
        weights["analysis_view"] = analysis_view
        all_scores.append(scores)
        all_weights.append(weights)

        if fi_present[d]:
            fi_df = pd.read_csv(d / "feature_importance.csv")
            fi_df["analysis_view"] = analysis_view
            all_fi.append(fi_df)
        elif any_fi:
            print(
                f"[summarize_univariate] WARNING: no feature_importance.csv in {d} "
                "— excluded from FI summaries"
            )

    if not any_fi:
        print(
            "[summarize_univariate] No feature_importance.csv found in any input; "
            "skipping FI summaries"
        )

    scores_all  = pd.concat(all_scores,  ignore_index=True)
    weights_all = pd.concat(all_weights, ignore_index=True)

    score_summary   = compute_score_summary_by_time(scores_all)
    weight_summary  = compute_weight_summary_by_feature_time(weights_all)
    top_units       = compute_top_units_by_feature(weights_all, top_k=top_k)
    feature_ranking = compute_global_feature_ranking(weights_all)

    score_summary.to_csv(out_paths["score_summary"],     index=False)
    weight_summary.to_csv(out_paths["weight_summary"],   index=False)
    top_units.to_csv(out_paths["top_units"],             index=False)
    feature_ranking.to_csv(out_paths["feature_ranking"], index=False)

    fi_summary = fi_ranking = top_units_fi = None

    if all_fi:
        fi_all      = pd.concat(all_fi, ignore_index=True)
        fi_summary  = compute_fi_summary_by_feature_time(fi_all)
        fi_ranking  = compute_global_fi_ranking(fi_all)
        top_units_fi = compute_top_units_by_fi(fi_all, top_k=top_k)

        fi_summary.to_csv(out_paths["fi_summary"],     index=False)
        fi_ranking.to_csv(out_paths["fi_ranking"],     index=False)
        top_units_fi.to_csv(out_paths["top_units_fi"], index=False)

    return {
        "score_summary":   score_summary,
        "weight_summary":  weight_summary,
        "top_units":       top_units,
        "feature_ranking": feature_ranking,
        "fi_summary":      fi_summary,
        "fi_ranking":      fi_ranking,
        "top_units_fi":    top_units_fi,
    }