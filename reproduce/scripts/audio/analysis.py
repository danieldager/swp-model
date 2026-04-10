#!/usr/bin/env python3
"""SWP-style statistical analysis of audio reconstruction metrics.

Loads metrics.csv from a Step 3 extraction run, merges paradigm factor columns
from the associated dataset CSV (resolved via manifest.json), then runs:
  - per-factor group summary tables (mean, std, n)
  - univariate OLS for each factor × metric
  - multivariate OLS (all items; words only with frequency_bin)
  - bar plots for each factor × metric combination

The dataset CSV path is read from manifest.json so no extra --dataset argument
is needed — the manifest keeps metrics and metadata provenance linked.

Usage:
    python reproduce/scripts/audio/analysis.py \\
        --results reproduce/data/audio/encodec__7f7d3b97/metrics.csv

    # Custom output dir
    python reproduce/scripts/audio/analysis.py \\
        --results reproduce/data/audio/encodec__7f7d3b97/metrics.csv \\
        --output /tmp/my_figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive — must be set before pyplot import
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

# Resolve repo root: reproduce/scripts/audio/ → 3 levels up → swp-model/
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

METRICS = ["mel_distance", "si_sdr"]

# Factors applied to all rows
FACTORS_ALL = ["lexicality", "length_bin", "morphology"]

# Required columns in the merged DataFrame
REQUIRED_COLS = METRICS + FACTORS_ALL + ["frequency_bin"]


# ── Data loading ──────────────────────────────────────────────────────────────


def load_data(metrics_path: Path, repo_root: Path) -> pd.DataFrame:
    """Load metrics.csv and merge paradigm factor columns via manifest.json.

    Args:
        metrics_path: path to metrics.csv inside a run directory
        repo_root:    repository root for resolving the dataset_path in manifest

    Returns:
        DataFrame with metric columns and factor columns merged on item_id
    """
    run_dir = metrics_path.parent
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found in {run_dir}. "
            "Expected alongside metrics.csv from a Step 3 extraction run."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    dataset_rel = manifest["run_params"]["dataset_path"]
    dataset_abs = (repo_root / dataset_rel).resolve()
    if not dataset_abs.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found: {dataset_abs}\n"
            f"(recorded in manifest as '{dataset_rel}', relative to repo root {repo_root})"
        )

    metrics_df = pd.read_csv(metrics_path)
    paradigm_df = pd.read_csv(dataset_abs)

    factor_cols = ["item_id", *FACTORS_ALL, "frequency_bin"]
    missing = [c for c in factor_cols if c not in paradigm_df.columns]
    if missing:
        raise ValueError(
            f"Dataset CSV is missing required factor columns: {missing}\n"
            f"Found columns: {list(paradigm_df.columns)}"
        )

    df = metrics_df.merge(paradigm_df[factor_cols], on="item_id", how="left")

    missing_metrics = [c for c in METRICS if c not in df.columns]
    if missing_metrics:
        raise ValueError(f"metrics.csv is missing columns: {missing_metrics}")

    unmatched = df["lexicality"].isna().sum()
    if unmatched > 0:
        raise ValueError(
            f"{unmatched} item(s) in metrics.csv did not match any row in dataset CSV. "
            "Check that metrics.csv and the dataset CSV are from the same run."
        )

    return df


# ── Statistics ────────────────────────────────────────────────────────────────


def group_summary(df: pd.DataFrame, factor: str) -> pd.DataFrame:
    """Mean, std, n for each level of factor, for all METRICS columns."""
    return (
        df.groupby(factor, observed=True)[METRICS]
        .agg(["mean", "std", "count"])
        .round(4)
    )


def _ols_block(df: pd.DataFrame, formula: str) -> str:
    """Fit OLS and return a formatted result block."""
    model = smf.ols(formula, data=df).fit()
    coef_df = pd.DataFrame({
        "coef":    model.params.round(5),
        "std_err": model.bse.round(5),
        "t":       model.tvalues.round(3),
        "p":       model.pvalues.round(4),
    })
    lines = [
        f"Formula : {formula}",
        f"N={int(model.nobs)}  R²={model.rsquared:.4f}  "
        f"adj-R²={model.rsquared_adj:.4f}  "
        f"F({model.df_model:.0f},{model.df_resid:.0f})={model.fvalue:.3f}  "
        f"p={model.f_pvalue:.4g}",
        coef_df.to_string(),
        "",
    ]
    return "\n".join(lines)


def build_regression_summary(df: pd.DataFrame, run_id: str) -> str:
    """Run all OLS models and return a single formatted string."""
    words = df[df["frequency_bin"].notna()].copy()
    blocks: list[str] = [
        f"Audio reconstruction analysis — run: {run_id}",
        "=" * 70,
        "",
    ]

    for metric in METRICS:
        blocks.append("=" * 70)
        blocks.append(f"METRIC: {metric}")
        blocks.append("=" * 70)
        blocks.append("")

        blocks.append("── Univariate (all items) ──────────────────────────────────────────")
        for factor in FACTORS_ALL:
            blocks.append(_ols_block(df, f"{metric} ~ C({factor})"))

        blocks.append(f"── Univariate: frequency_bin (words only, n={len(words)}) ──────────")
        blocks.append(_ols_block(words, f"{metric} ~ C(frequency_bin)"))

        blocks.append("── Multivariate (all items) ────────────────────────────────────────")
        blocks.append(_ols_block(
            df,
            f"{metric} ~ C(lexicality) + C(length_bin) + C(morphology)",
        ))

        blocks.append(f"── Multivariate (words only, n={len(words)}) ───────────────────────")
        blocks.append(_ols_block(
            words,
            f"{metric} ~ C(length_bin) + C(frequency_bin) + C(morphology)",
        ))

    return "\n".join(blocks)


# ── Figures ───────────────────────────────────────────────────────────────────


def _bar_plot(
    df: pd.DataFrame,
    factor: str,
    metric: str,
    out_path: Path,
    title: str | None = None,
) -> None:
    """Bar plot (mean ± 95 % CI) of metric by factor levels."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(
        data=df,
        x=factor,
        y=metric,
        ax=ax,
        errorbar="ci",
        capsize=0.12,
        err_kws={"linewidth": 1.5},
    )
    ax.set_title(title or f"{metric} by {factor}", fontsize=11)
    ax.set_xlabel(factor)
    ax.set_ylabel(metric)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_figures(df: pd.DataFrame, out_dir: Path) -> None:
    """Save all bar plots to out_dir."""
    words = df[df["frequency_bin"].notna()]

    for metric in METRICS:
        # One plot per all-item factor
        for factor in FACTORS_ALL:
            path = out_dir / f"{metric}_by_{factor}.png"
            _bar_plot(df, factor, metric, path)
            print(f"[analysis] Written : {path}")

        # frequency_bin — words only
        path = out_dir / f"{metric}_by_frequency_bin.png"
        _bar_plot(
            words,
            "frequency_bin",
            metric,
            path,
            title=f"{metric} by frequency_bin (words only, n={len(words)})",
        )
        print(f"[analysis] Written : {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SWP-style analysis of audio reconstruction metrics."
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to metrics.csv from a Step 3 extraction run",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output directory "
            "(default: reproduce/figures/audio/{run_id}/ relative to repo root)"
        ),
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repo root for resolving dataset path (default: auto-detected)",
    )
    args = parser.parse_args()

    metrics_path = Path(args.results).resolve()
    if not metrics_path.exists():
        print(f"[analysis] ERROR: metrics.csv not found: {metrics_path}", file=sys.stderr)
        sys.exit(1)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else REPO_ROOT
    run_id = metrics_path.parent.name

    if args.output:
        out_dir = Path(args.output).resolve()
    else:
        out_dir = repo_root / "reproduce" / "figures" / "audio" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[analysis] Run       : {run_id}")
    print(f"[analysis] Metrics   : {metrics_path}")
    print(f"[analysis] Repo root : {repo_root}")
    print(f"[analysis] Output    : {out_dir}")

    # Load & merge
    df = load_data(metrics_path, repo_root)
    n_words = (df["lexicality"] == "word").sum()
    n_nw = (df["lexicality"] == "nonword").sum()
    print(f"[analysis] Items     : {len(df)} ({n_words} words, {n_nw} nonwords)")

    # Summary table
    summaries = []
    for factor in [*FACTORS_ALL, "frequency_bin"]:
        sub = df if factor != "frequency_bin" else df[df["frequency_bin"].notna()]
        s = group_summary(sub, factor)
        s.index = pd.MultiIndex.from_tuples(
            [(factor, str(v)) for v in s.index], names=["factor", "level"]
        )
        summaries.append(s)

    summary_df = pd.concat(summaries)
    summary_path = out_dir / "summary_by_factor.csv"
    summary_df.to_csv(summary_path)
    print(f"[analysis] Written : {summary_path}")

    # Regression summary
    reg_text = build_regression_summary(df, run_id)
    reg_path = out_dir / "regression_summary.txt"
    reg_path.write_text(reg_text)
    print(f"[analysis] Written : {reg_path}")

    # Figures
    sns.set_theme(style="whitegrid", palette="colorblind")
    save_figures(df, out_dir)

    print(f"\n[analysis] All outputs written to {out_dir}")


if __name__ == "__main__":
    main()