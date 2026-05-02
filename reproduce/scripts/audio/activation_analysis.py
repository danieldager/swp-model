#!/usr/bin/env python3
"""SWP-style activation-level analysis of EnCodec hidden states.

Loads per-item per-layer .pt activations from a Step 3 extraction run,
computes summary features (no pooling during extraction — summarized here),
then runs SWP-style OLS analyses and saves figures.

Layer types handled
-------------------
2D layers [C, T] — encoder_out, decoder_in:
    n_channels, n_frames, mean_norm, std_norm, max_norm, mean_abs,
    var_over_time, trajectory_length, trajectory_length_per_frame,
    mean_step_distance

    trajectory_length           = sum of L2 step distances in channel space
    trajectory_length_per_frame = trajectory_length / n_frames  (duration-normalized)
    mean_step_distance          = trajectory_length / (n_frames - 1)  (per-step average)

Waveform-like layers [1, T] — decoder_out:
    n_samples, mean_abs, std_value, max_abs, rms

Analyses (on encoder_out and decoder_in only):
    mean_norm and trajectory_length ~ each paradigm factor (OLS),
    plus frame-control regressions distinguishing duration effects from
    genuine per-step latent dynamics:
        trajectory_length ~ C(factor)              [baseline]
        trajectory_length ~ C(factor) + n_frames   [frame-controlled]
        trajectory_length_per_frame ~ C(factor)    [duration-normalized]
        mean_step_distance ~ C(factor)             [per-step normalized]

Usage:
    python reproduce/scripts/audio/activation_analysis.py \\
        --run reproduce/data/audio/encodec__7f7d3b97/

    # Custom output dir
    python reproduce/scripts/audio/activation_analysis.py \\
        --run reproduce/data/audio/encodec__7f7d3b97/ \\
        --output /tmp/act_figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FACTOR_COLS = ["lexicality", "length_bin", "frequency_bin", "morphology"]
FACTORS_ALL = ["lexicality", "length_bin", "morphology"]  # applied to all rows
META_COLS = ["speaker", "duration_s"]

# 2D analysis targets (layer must have shape [C, T] with C > 1)
ANALYSIS_LAYERS = ["encoder_out", "decoder_in"]
ANALYSIS_FEATURES = ["mean_norm", "trajectory_length"]


# ── Feature extraction ────────────────────────────────────────────────────────


def _is_waveform(t: torch.Tensor) -> bool:
    """True if tensor is waveform-like: 1D [T] or 2D [1, T]."""
    return t.dim() == 1 or (t.dim() == 2 and t.shape[0] == 1)


def summarize_2d(t: torch.Tensor) -> dict:
    """Summary features for a 2D feature-map tensor [C, T].

    Frame norms are L2 norms across the channel dimension.
    Trajectory is measured in the C-dimensional channel space over time.
    """
    t = t.float()
    C, T = t.shape

    frame_norms = t.norm(dim=0)        # [T] — L2 norm across channels per frame
    mean_norm = frame_norms.mean().item()
    std_norm = frame_norms.std().item() if T > 1 else 0.0
    max_norm = frame_norms.max().item()
    mean_abs = t.abs().mean().item()
    var_over_time = t.var(dim=1).mean().item()  # per-channel variance, averaged

    if T > 1:
        steps = (t[:, 1:] - t[:, :-1]).norm(dim=0)  # [T-1]
        trajectory_length = steps.sum().item()
        mean_step_distance = steps.mean().item()
    else:
        trajectory_length = 0.0
        mean_step_distance = 0.0

    trajectory_length_per_frame = trajectory_length / max(T, 1)

    return {
        "n_channels": int(C),
        "n_frames": int(T),
        "mean_norm": mean_norm,
        "std_norm": std_norm,
        "max_norm": max_norm,
        "mean_abs": mean_abs,
        "var_over_time": var_over_time,
        "trajectory_length": trajectory_length,
        "trajectory_length_per_frame": trajectory_length_per_frame,
        "mean_step_distance": mean_step_distance,
    }


def summarize_waveform(t: torch.Tensor) -> dict:
    """Summary features for a waveform-like tensor [1, T] or [T]."""
    x = t.squeeze().float()
    T = x.shape[0]
    return {
        "n_samples": int(T),
        "mean_abs": x.abs().mean().item(),
        "std_value": x.std().item(),
        "max_abs": x.abs().max().item(),
        "rms": x.pow(2).mean().sqrt().item(),
    }


def summarize_tensor(t: torch.Tensor) -> dict:
    """Dispatch to the right summarizer based on tensor shape."""
    if _is_waveform(t):
        return summarize_waveform(t)
    return summarize_2d(t)


# ── Data loading ──────────────────────────────────────────────────────────────


def load_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found in {run_dir}. "
            "Expected a Step 3 extraction run directory."
        )
    with open(manifest_path) as f:
        manifest = json.load(f)
    required_keys = {"run_id", "run_params", "items"}
    missing = required_keys - manifest.keys()
    if missing:
        raise ValueError(f"manifest.json is missing keys: {missing}")
    return manifest


def load_paradigm_factors(manifest: dict, repo_root: Path) -> pd.DataFrame:
    """Load factor columns from the dataset CSV recorded in the manifest."""
    dataset_rel = manifest["run_params"]["dataset_path"]
    dataset_abs = (repo_root / dataset_rel).resolve()
    if not dataset_abs.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found: {dataset_abs}\n"
            f"(recorded in manifest as '{dataset_rel}')"
        )
    df = pd.read_csv(dataset_abs)
    needed = ["item_id", *FACTOR_COLS, *META_COLS]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset CSV is missing required columns: {missing}\n"
            f"Found: {list(df.columns)}"
        )
    return df[needed]


def build_activation_summary(
    manifest: dict,
    run_dir: Path,
    paradigm_df: pd.DataFrame,
) -> pd.DataFrame:
    """Load all .pt files, compute features, assemble one row per item per layer."""
    items = manifest["items"]
    rows: list[dict] = []

    n_items = len(items)
    for i, (item_id, layer_paths) in enumerate(items.items()):
        if (i + 1) % 20 == 0 or i == 0 or i == n_items - 1:
            print(f"[activation_analysis] Summarizing {i + 1}/{n_items}  item={item_id}")

        for layer_name, rel_path in layer_paths.items():
            pt_path = run_dir / rel_path
            if not pt_path.exists():
                raise FileNotFoundError(
                    f"Activation file not found: {pt_path}\n"
                    f"(recorded in manifest for item={item_id}, layer={layer_name})"
                )
            t = torch.load(pt_path, map_location="cpu")
            features = summarize_tensor(t)
            row = {"item_id": item_id, "layer": layer_name, **features}
            rows.append(row)

    summary_df = pd.DataFrame(rows)

    # Merge paradigm factor columns
    merged = summary_df.merge(paradigm_df, on="item_id", how="left")
    n_unmatched = merged["lexicality"].isna().sum()
    if n_unmatched > 0:
        raise ValueError(
            f"{n_unmatched} row(s) in activation summary did not match any row "
            "in the dataset CSV. Check that the run and dataset are consistent."
        )

    # Reorder: item_id, layer, factors/meta, then features
    meta = ["item_id", "layer", *FACTOR_COLS, *META_COLS]
    feature_cols = [c for c in merged.columns if c not in meta]
    return merged[meta + feature_cols]


# ── Statistics ────────────────────────────────────────────────────────────────


def _ols_block(df: pd.DataFrame, formula: str) -> str:
    model = smf.ols(formula, data=df).fit()
    coef_df = pd.DataFrame({
        "coef":    model.params.round(5),
        "std_err": model.bse.round(5),
        "t":       model.tvalues.round(3),
        "p":       model.pvalues.round(4),
    })
    return "\n".join([
        f"Formula : {formula}",
        f"N={int(model.nobs)}  R²={model.rsquared:.4f}  "
        f"adj-R²={model.rsquared_adj:.4f}  "
        f"F({model.df_model:.0f},{model.df_resid:.0f})={model.fvalue:.3f}  "
        f"p={model.f_pvalue:.4g}",
        coef_df.to_string(),
        "",
    ])


def build_regression_summary(df: pd.DataFrame, run_id: str) -> str:
    """Run all OLS models on 2D layers and return formatted text."""
    blocks: list[str] = [
        f"Activation-level analysis — run: {run_id}",
        "=" * 70,
        "",
        "Analyses restricted to 2D layers: encoder_out, decoder_in.",
        "",
    ]

    # ── Section 1: basic factor effects (mean_norm, trajectory_length) ─────────
    blocks.append("SECTION 1 — Basic factor effects")
    blocks.append("Features: mean_norm, trajectory_length")
    blocks.append("=" * 70)
    blocks.append("")

    for layer in ANALYSIS_LAYERS:
        sub = df[df["layer"] == layer].copy()
        words = sub[sub["frequency_bin"].notna()].copy()

        blocks.append("=" * 70)
        blocks.append(f"LAYER: {layer}  (n={len(sub)})")
        blocks.append("=" * 70)
        blocks.append("")

        for feature in ANALYSIS_FEATURES:
            blocks.append(f"── Feature: {feature} ──────────────────────────────────────────")

            blocks.append("Univariate (all items):")
            for factor in FACTORS_ALL:
                blocks.append(_ols_block(sub, f"{feature} ~ C({factor})"))

            blocks.append(f"Univariate: frequency_bin (words only, n={len(words)}):")
            blocks.append(_ols_block(words, f"{feature} ~ C(frequency_bin)"))

            blocks.append("Multivariate (all items):")
            blocks.append(_ols_block(
                sub,
                f"{feature} ~ C(lexicality) + C(length_bin) + C(morphology)",
            ))

            blocks.append(f"Multivariate (words only, n={len(words)}):")
            blocks.append(_ols_block(
                words,
                f"{feature} ~ C(length_bin) + C(frequency_bin) + C(morphology)",
            ))

    # ── Section 2: frame-control models ────────────────────────────────────────
    blocks.append("")
    blocks.append("SECTION 2 — Frame-controlled trajectory analysis")
    blocks.append(
        "Purpose: test whether factor effects on trajectory_length persist after\n"
        "         controlling for n_frames (i.e. whether effects are driven by\n"
        "         sequence duration or by genuine per-step latent dynamics).\n"
        "\n"
        "  baseline  : trajectory_length ~ C(factor)\n"
        "  controlled: trajectory_length ~ C(factor) + n_frames\n"
        "  normalized: trajectory_length_per_frame ~ C(factor)\n"
        "  per-step  : mean_step_distance ~ C(factor)"
    )
    blocks.append("=" * 70)
    blocks.append("")

    for layer in ANALYSIS_LAYERS:
        sub = df[df["layer"] == layer].copy()
        words = sub[sub["frequency_bin"].notna()].copy()

        blocks.append("=" * 70)
        blocks.append(f"LAYER: {layer}  (n={len(sub)})")
        blocks.append("=" * 70)
        blocks.append("")

        for factor in [*FACTORS_ALL, "frequency_bin"]:
            is_freq = factor == "frequency_bin"
            data = words if is_freq else sub
            label = f"(words only, n={len(data)})" if is_freq else f"(all, n={len(data)})"
            blocks.append(f"── Factor: {factor}  {label} ──────────────────────")

            blocks.append("baseline (confounded by duration):")
            blocks.append(_ols_block(data, f"trajectory_length ~ C({factor})"))

            blocks.append("frame-controlled (+ n_frames covariate):")
            blocks.append(_ols_block(data, f"trajectory_length ~ C({factor}) + n_frames"))

            blocks.append("duration-normalized (per-frame):")
            blocks.append(_ols_block(data, f"trajectory_length_per_frame ~ C({factor})"))

            blocks.append("per-step (mean_step_distance):")
            blocks.append(_ols_block(data, f"mean_step_distance ~ C({factor})"))

    # ── Section 3: layer comparison table ──────────────────────────────────────
    blocks.append("")
    blocks.append("SECTION 3 — Layer comparison (encoder_out vs decoder_in)")
    blocks.append("=" * 70)
    cmp_features = ["mean_norm", "trajectory_length", "trajectory_length_per_frame",
                    "mean_step_distance"]
    sub_2d = df[df["layer"].isin(ANALYSIS_LAYERS)].copy()
    cmp_table = (
        sub_2d.groupby("layer")[cmp_features]
        .agg(["mean", "std"])
    )
    cmp_table.columns = ["_".join(c) for c in cmp_table.columns]
    blocks.append(cmp_table.round(5).to_string())
    blocks.append("")

    return "\n".join(blocks)


# ── Figures ───────────────────────────────────────────────────────────────────


def _bar_plot(
    df: pd.DataFrame,
    factor: str,
    feature: str,
    out_path: Path,
    title: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(
        data=df,
        x=factor,
        y=feature,
        ax=ax,
        errorbar="ci",
        capsize=0.12,
        err_kws={"linewidth": 1.5},
    )
    ax.set_title(title or f"{feature} by {factor}", fontsize=11)
    ax.set_xlabel(factor)
    ax.set_ylabel(feature)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_figures(df: pd.DataFrame, out_dir: Path) -> None:
    """Save all activation figures."""

    # 1. mean_norm by each factor for encoder_out and decoder_in
    for layer in ANALYSIS_LAYERS:
        sub = df[df["layer"] == layer]
        words = sub[sub["frequency_bin"].notna()]

        for factor in FACTORS_ALL:
            path = out_dir / f"{layer}__mean_norm_by_{factor}.png"
            _bar_plot(sub, factor, "mean_norm", path)
            print(f"[activation_analysis] Written : {path}")

        path = out_dir / f"{layer}__mean_norm_by_frequency_bin.png"
        _bar_plot(
            words, "frequency_bin", "mean_norm", path,
            title=f"{layer} mean_norm by frequency_bin (words only, n={len(words)})",
        )
        print(f"[activation_analysis] Written : {path}")

    # 2. n_frames vs duration_s scatter for 2D layers
    sub_2d = df[df["layer"].isin(ANALYSIS_LAYERS) & df["n_frames"].notna()].copy()
    if "duration_s" in sub_2d.columns and len(sub_2d) > 0:
        fig, ax = plt.subplots(figsize=(5, 4))
        for layer, grp in sub_2d.groupby("layer"):
            ax.scatter(grp["duration_s"], grp["n_frames"], label=layer, alpha=0.6, s=20)
        ax.set_xlabel("duration_s")
        ax.set_ylabel("n_frames")
        ax.set_title("Encoder frames vs audio duration")
        ax.legend()
        plt.tight_layout()
        path = out_dir / "n_frames_vs_duration.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[activation_analysis] Written : {path}")

    # 3. encoder_out vs decoder_in mean_norm distribution
    sub_cmp = df[df["layer"].isin(ANALYSIS_LAYERS)].copy()
    if len(sub_cmp) > 0:
        fig, ax = plt.subplots(figsize=(5, 4))
        for layer, grp in sub_cmp.groupby("layer"):
            ax.hist(grp["mean_norm"], bins=20, alpha=0.6, label=layer, density=True)
        ax.set_xlabel("mean_norm")
        ax.set_ylabel("density")
        ax.set_title("mean_norm distribution by layer")
        ax.legend()
        plt.tight_layout()
        path = out_dir / "mean_norm_distribution_by_layer.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[activation_analysis] Written : {path}")

    # ── Step 6 additions ──────────────────────────────────────────────────────

    # 4. trajectory_length vs n_frames scatter (both layers, colored by length_bin)
    #    Visualises the core confound: does longer = more frames?
    if len(sub_2d) > 0:
        palette = sns.color_palette("colorblind", n_colors=2)
        length_colors = {v: palette[i] for i, v in enumerate(sorted(sub_2d["length_bin"].unique()))}
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=False, sharey=False)
        for ax, layer in zip(axes, ANALYSIS_LAYERS):
            grp = sub_2d[sub_2d["layer"] == layer]
            for lb, c in length_colors.items():
                mask = grp["length_bin"] == lb
                ax.scatter(grp.loc[mask, "n_frames"], grp.loc[mask, "trajectory_length"],
                           color=c, label=lb, alpha=0.6, s=18)
            ax.set_title(layer)
            ax.set_xlabel("n_frames")
            ax.set_ylabel("trajectory_length")
            ax.legend(title="length_bin")
        fig.suptitle("trajectory_length vs n_frames (by length_bin)", fontsize=11)
        plt.tight_layout()
        path = out_dir / "trajectory_length_vs_n_frames.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[activation_analysis] Written : {path}")

    # 5. trajectory_length_per_frame by length_bin (2-panel, one per analysis layer)
    if len(sub_2d) > 0 and "trajectory_length_per_frame" in sub_2d.columns:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, layer in zip(axes, ANALYSIS_LAYERS):
            grp = sub_2d[sub_2d["layer"] == layer]
            sns.barplot(data=grp, x="length_bin", y="trajectory_length_per_frame",
                        ax=ax, errorbar="ci", capsize=0.12, err_kws={"linewidth": 1.5})
            ax.set_title(layer)
            ax.set_xlabel("length_bin")
            ax.set_ylabel("trajectory_length_per_frame")
        fig.suptitle("trajectory_length_per_frame by length_bin\n"
                     "(duration-normalized — tests whether step size changes)", fontsize=10)
        plt.tight_layout()
        path = out_dir / "trajectory_length_per_frame_by_length_bin.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[activation_analysis] Written : {path}")

    # 6. mean_step_distance by length_bin (2-panel)
    if len(sub_2d) > 0 and "mean_step_distance" in sub_2d.columns:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, layer in zip(axes, ANALYSIS_LAYERS):
            grp = sub_2d[sub_2d["layer"] == layer]
            sns.barplot(data=grp, x="length_bin", y="mean_step_distance",
                        ax=ax, errorbar="ci", capsize=0.12, err_kws={"linewidth": 1.5})
            ax.set_title(layer)
            ax.set_xlabel("length_bin")
            ax.set_ylabel("mean_step_distance")
        fig.suptitle("mean_step_distance by length_bin\n"
                     "(per-step average — pure latent dynamics signal)", fontsize=10)
        plt.tight_layout()
        path = out_dir / "mean_step_distance_by_length_bin.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[activation_analysis] Written : {path}")

    # 7. Layer comparison: grouped bars for mean_norm, traj_per_frame, mean_step_distance
    if len(sub_cmp) > 0:
        cmp_features = ["mean_norm", "trajectory_length_per_frame", "mean_step_distance"]
        cmp_features = [f for f in cmp_features if f in sub_cmp.columns]
        cmp_means = sub_cmp.groupby("layer")[cmp_features].mean()
        # Normalise each feature to [0,1] for side-by-side visibility
        cmp_norm = (cmp_means - cmp_means.min()) / (cmp_means.max() - cmp_means.min() + 1e-12)
        x = range(len(cmp_features))
        width = 0.35
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, (layer, row) in enumerate(cmp_norm.iterrows()):
            offset = (i - 0.5) * width
            bars = ax.bar([xi + offset for xi in x], row.values, width, label=layer, alpha=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(cmp_features, rotation=15, ha="right")
        ax.set_ylabel("normalised value (0–1 per feature)")
        ax.set_title("Layer comparison: encoder_out vs decoder_in\n"
                     "(features normalised independently for visual comparison)")
        ax.legend()
        plt.tight_layout()
        path = out_dir / "layer_feature_comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[activation_analysis] Written : {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Activation-level analysis of a Step 3 extraction run."
    )
    parser.add_argument(
        "--run",
        required=True,
        help="Path to run directory (e.g. reproduce/data/audio/encodec__7f7d3b97/)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output directory "
            "(default: reproduce/figures/audio/{run_id}/activations/ relative to repo root)"
        ),
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repo root for resolving dataset path (default: auto-detected)",
    )
    parser.add_argument(
        "--filter",
        nargs="+",
        default=None,
        metavar="KEY=VALUE",
        help=(
            "Restrict trials to those matching all KEY=VALUE pairs before running "
            "OLS regressions and saving figures. "
            "E.g. --filter length_bin=short. "
            "Applied to the assembled activation summary DataFrame."
        ),
    )
    args = parser.parse_args()

    run_dir = Path(args.run).resolve()
    if not run_dir.exists():
        print(f"[activation_analysis] ERROR: run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else REPO_ROOT
    manifest = load_manifest(run_dir)
    run_id = manifest["run_id"]

    if args.output:
        out_dir = Path(args.output).resolve()
    else:
        out_dir = repo_root / "reproduce" / "figures" / "audio" / run_id / "activations"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[activation_analysis] Run       : {run_id}")
    print(f"[activation_analysis] Run dir   : {run_dir}")
    print(f"[activation_analysis] Repo root : {repo_root}")
    print(f"[activation_analysis] Output    : {out_dir}")

    paradigm_df = load_paradigm_factors(manifest, repo_root)
    summary_df = build_activation_summary(manifest, run_dir, paradigm_df)

    # ── Optional trial filter ─────────────────────────────────────────────────
    if args.filter:
        trial_filter: dict[str, str] = {}
        for item in args.filter:
            if "=" not in item:
                print(
                    f"[activation_analysis] ERROR: --filter items must be KEY=VALUE, "
                    f"got: {item!r}",
                    file=sys.stderr,
                )
                sys.exit(1)
            key, val = item.split("=", 1)
            trial_filter[key.strip()] = val.strip()
        for col, val in trial_filter.items():
            before = len(summary_df)
            summary_df = summary_df[summary_df[col] == val].reset_index(drop=True)
            print(
                f"[activation_analysis] --filter {col}={val!r}: "
                f"{before} → {len(summary_df)} rows"
            )

    n_items = summary_df["item_id"].nunique()
    n_layers = summary_df["layer"].nunique()
    print(f"[activation_analysis] Items     : {n_items}")
    print(f"[activation_analysis] Layers    : {n_layers}  {sorted(summary_df['layer'].unique())}")

    # activation_summary.csv
    summary_path = out_dir / "activation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[activation_analysis] Written : {summary_path}")

    # regression summary
    reg_text = build_regression_summary(summary_df, run_id)
    reg_path = out_dir / "activation_regression_summary.txt"
    reg_path.write_text(reg_text)
    print(f"[activation_analysis] Written : {reg_path}")

    # figures
    sns.set_theme(style="whitegrid", palette="colorblind")
    save_figures(summary_df, out_dir)

    print(f"\n[activation_analysis] All outputs written to {out_dir}")


if __name__ == "__main__":
    main()