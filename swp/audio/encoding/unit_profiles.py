"""Unit-level feature importance profiles from codec neural activations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── constants ─────────────────────────────────────────────────────────────────

_EPSILON = 1e-12
_DPI     = 200

_FEAT_COLOR: dict[str, str] = {
    "lexicality":    "#2ca02c",
    "length_bin":    "#1f77b4",
    "morphology":    "#d62728",
    "frequency_bin": "#9467bd",
}

_MODEL_COLOR: dict[str, str] = {
    "encodec": "#1f77b4",
    "dac":     "#ff7f0e",
}

_UNIT_COLS = ["run_id", "model_name", "layer", "analysis_set", "neuron"]

_REQUIRED_FI_COLS = {
    "run_id", "model_name", "layer", "analysis_set",
    "neuron", "time_bin", "relative_time", "feature", "fi_mean", "fi_std",
}


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_fi_run(fi_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Load feature_importance.csv and config.json from a univariate run dir."""
    fi_dir   = Path(fi_dir)
    fi_path  = fi_dir / "feature_importance.csv"
    cfg_path = fi_dir / "config.json"

    if not fi_path.exists():
        raise FileNotFoundError(f"feature_importance.csv not found in {fi_dir}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in {fi_dir}")

    df = pd.read_csv(fi_path)
    missing = _REQUIRED_FI_COLS - set(df.columns)
    if missing:
        raise ValueError(f"feature_importance.csv missing columns: {missing}")

    with open(cfg_path) as fh:
        config = json.load(fh)

    return df, config


# ── profile computation ───────────────────────────────────────────────────────

def compute_unit_feature_profiles(fi_df: pd.DataFrame) -> pd.DataFrame:
    """Long profile: one row per (unit, feature) with temporal statistics."""
    group_cols = _UNIT_COLS + ["feature"]

    unit_avail = (
        fi_df.groupby(_UNIT_COLS)["feature"]
        .apply(lambda s: "|".join(sorted(s.unique())))
        .reset_index()
        .rename(columns={"feature": "available_features"})
    )

    rows = []
    for keys, grp in fi_df.groupby(group_cols, sort=False):
        unit_dict = dict(zip(group_cols, keys))
        order = np.argsort(grp["relative_time"].values)
        vals  = grp["fi_mean"].values[order]
        rt    = grp["relative_time"].values[order]
        tb    = grp["time_bin"].values[order]

        abs_vals    = np.abs(vals)
        max_idx     = int(np.argmax(vals))
        max_abs_idx = int(np.argmax(abs_vals))

        early_mask = rt < 0.33
        mid_mask   = (rt >= 0.33) & (rt < 0.66)
        late_mask  = rt >= 0.66

        rows.append({
            **unit_dict,
            "mean_fi_over_time":           float(np.mean(vals)),
            "mean_abs_fi_over_time":       float(np.mean(abs_vals)),
            "median_fi_over_time":         float(np.median(vals)),
            "max_fi":                      float(vals.max()),
            "min_fi":                      float(vals.min()),
            "max_abs_fi":                  float(abs_vals.max()),
            "signed_fi_at_max_abs":        float(vals[max_abs_idx]),
            "time_bin_at_max_fi":          int(tb[max_idx]),
            "relative_time_at_max_fi":     float(rt[max_idx]),
            "time_bin_at_max_abs_fi":      int(tb[max_abs_idx]),
            "relative_time_at_max_abs_fi": float(rt[max_abs_idx]),
            "frac_positive_time":          float(np.mean(vals > 0)),
            "early_mean_fi": float(vals[early_mask].mean()) if early_mask.any() else np.nan,
            "mid_mean_fi":   float(vals[mid_mask].mean())   if mid_mask.any()   else np.nan,
            "late_mean_fi":  float(vals[late_mask].mean())  if late_mask.any()  else np.nan,
        })

    long_df = pd.DataFrame(rows)
    long_df = long_df.merge(unit_avail, on=_UNIT_COLS, how="left")
    return long_df


def build_unit_profile_wide(profile_long: pd.DataFrame) -> pd.DataFrame:
    """Wide profile: one row per unit, columns = {feature}_{stat}."""
    stat_cols = [
        c for c in profile_long.columns
        if c not in _UNIT_COLS + ["feature", "available_features"]
    ]
    wide = profile_long.pivot(index=_UNIT_COLS, columns="feature", values=stat_cols)
    wide.columns = [f"{feat}_{stat}" for stat, feat in wide.columns]
    wide = wide.reset_index()

    avail = profile_long.groupby(_UNIT_COLS)["available_features"].first().reset_index()
    wide  = wide.merge(avail, on=_UNIT_COLS, how="left")
    return wide


def compute_unit_dominance(profile_long: pd.DataFrame) -> pd.DataFrame:
    """Dominant feature, margins, and cross-feature ratios, one row per unit."""
    rows = []

    for keys, grp in profile_long.groupby(_UNIT_COLS, sort=False):
        unit_dict = dict(zip(_UNIT_COLS, keys))

        feat_data: dict[str, dict[str, float]] = {
            r["feature"]: {
                "mean_fi":     r["mean_fi_over_time"],
                "mean_abs_fi": r["mean_abs_fi_over_time"],
                "max_fi":      r["max_fi"],
            }
            for _, r in grp.iterrows()
        }

        def _get(feat: str, key: str) -> float:
            return feat_data.get(feat, {}).get(key, np.nan)  # type: ignore[return-value]

        def _ratio(num: float, den: float) -> float:
            if np.isnan(num) or np.isnan(den):
                return np.nan
            return num / (den + _EPSILON)

        sorted_mean = sorted(feat_data.items(), key=lambda x: x[1]["mean_fi"],     reverse=True)
        sorted_abs  = sorted(feat_data.items(), key=lambda x: x[1]["mean_abs_fi"], reverse=True)
        sorted_max  = sorted(feat_data.items(), key=lambda x: x[1]["max_fi"],      reverse=True)

        dom_mean_val = sorted_mean[0][1]["mean_fi"]
        sec_mean_val = sorted_mean[1][1]["mean_fi"]    if len(sorted_mean) > 1 else np.nan
        dom_abs_val  = sorted_abs[0][1]["mean_abs_fi"]
        sec_abs_val  = sorted_abs[1][1]["mean_abs_fi"] if len(sorted_abs)  > 1 else np.nan

        l_mean = _get("length_bin", "mean_fi")
        l_abs  = _get("length_bin", "mean_abs_fi")
        l_max  = _get("length_bin", "max_fi")

        other_mean_vals = [v["mean_fi"]     for f, v in feat_data.items() if f != "length_bin"]
        other_abs_vals  = [v["mean_abs_fi"] for f, v in feat_data.items() if f != "length_bin"]
        other_mean     = float(np.mean(other_mean_vals)) if other_mean_vals else np.nan
        other_abs_mean = float(np.mean(other_abs_vals))  if other_abs_vals  else np.nan

        rows.append({
            **unit_dict,
            "available_features":               grp["available_features"].iloc[0],
            "dominant_feature_by_mean_fi":      sorted_mean[0][0],
            "dominant_feature_by_mean_abs_fi":  sorted_abs[0][0],
            "dominant_feature_by_max_fi":       sorted_max[0][0],
            "dominant_mean_fi":             dom_mean_val,
            "second_mean_fi":              sec_mean_val,
            "dominance_margin_mean_fi":     dom_mean_val - sec_mean_val if not np.isnan(sec_mean_val) else np.nan,
            "dominant_mean_abs_fi":         dom_abs_val,
            "second_mean_abs_fi":           sec_abs_val,
            "dominance_margin_mean_abs_fi": dom_abs_val - sec_abs_val  if not np.isnan(sec_abs_val)  else np.nan,
            "length_mean_fi":               l_mean,
            "length_mean_abs_fi":           l_abs,
            "length_max_fi":                l_max,
            "length_to_other_ratio":        _ratio(l_mean, other_mean),
            "length_to_other_abs_ratio":    _ratio(l_abs,  other_abs_mean),
            "lexicality_mean_fi":           _get("lexicality", "mean_fi"),
            "lexicality_mean_abs_fi":       _get("lexicality", "mean_abs_fi"),
            "lexicality_max_fi":            _get("lexicality", "max_fi"),
            "lexicality_to_length_ratio":     _ratio(_get("lexicality", "mean_fi"),     l_mean),
            "lexicality_to_length_abs_ratio": _ratio(_get("lexicality", "mean_abs_fi"), l_abs),
            "frequency_mean_fi":            _get("frequency_bin", "mean_fi"),
            "frequency_mean_abs_fi":        _get("frequency_bin", "mean_abs_fi"),
            "frequency_max_fi":             _get("frequency_bin", "max_fi"),
            "frequency_to_length_ratio":      _ratio(_get("frequency_bin", "mean_fi"),     l_mean),
            "frequency_to_length_abs_ratio":  _ratio(_get("frequency_bin", "mean_abs_fi"), l_abs),
            "morphology_mean_fi":           _get("morphology", "mean_fi"),
            "morphology_mean_abs_fi":       _get("morphology", "mean_abs_fi"),
            "morphology_max_fi":            _get("morphology", "max_fi"),
            "morphology_to_length_ratio":     _ratio(_get("morphology", "mean_fi"),     l_mean),
            "morphology_to_length_abs_ratio": _ratio(_get("morphology", "mean_abs_fi"), l_abs),
        })

    return pd.DataFrame(rows)


def summarize_unit_dominance(unit_dominance: pd.DataFrame) -> pd.DataFrame:
    """Aggregate dominance stats by (model_name, run_id, layer, analysis_set)."""
    group_cols = ["model_name", "run_id", "layer", "analysis_set"]
    rows = []

    for keys, grp in unit_dominance.groupby(group_cols, sort=True):
        g    = dict(zip(group_cols, keys))
        n    = len(grp)
        aset = g["analysis_set"]

        def _frac(feature: str, criterion: str = "dominant_feature_by_mean_fi") -> float:
            return (grp[criterion] == feature).sum() / n

        def _safe(col: str, fn) -> float:  # type: ignore[type-arg]
            s = grp[col].dropna()
            return float(fn(s)) if len(s) else np.nan

        rows.append({
            **g,
            "n_units": n,
            "frac_dominant_length_mean_fi":       _frac("length_bin"),
            "frac_dominant_morphology_mean_fi":   _frac("morphology"),
            "frac_dominant_lexicality_mean_fi":   _frac("lexicality")    if aset == "all_items"  else np.nan,
            "frac_dominant_frequency_mean_fi":    _frac("frequency_bin") if aset == "words_only" else np.nan,
            "frac_dominant_length_mean_abs_fi":     _frac("length_bin",  "dominant_feature_by_mean_abs_fi"),
            "frac_dominant_morphology_mean_abs_fi": _frac("morphology",  "dominant_feature_by_mean_abs_fi"),
            "frac_dominant_lexicality_mean_abs_fi": _frac("lexicality",    "dominant_feature_by_mean_abs_fi") if aset == "all_items"  else np.nan,
            "frac_dominant_frequency_mean_abs_fi":  _frac("frequency_bin", "dominant_feature_by_mean_abs_fi") if aset == "words_only" else np.nan,
            "mean_length_to_other_ratio":       _safe("length_to_other_ratio",     np.mean),
            "median_length_to_other_ratio":     _safe("length_to_other_ratio",     np.median),
            "mean_length_to_other_abs_ratio":   _safe("length_to_other_abs_ratio", np.mean),
            "median_length_to_other_abs_ratio": _safe("length_to_other_abs_ratio", np.median),
            "mean_lexicality_to_length_abs_ratio": _safe("lexicality_to_length_abs_ratio", np.mean)   if aset == "all_items"  else np.nan,
            "max_lexicality_to_length_abs_ratio":  _safe("lexicality_to_length_abs_ratio", np.max)    if aset == "all_items"  else np.nan,
            "mean_frequency_to_length_abs_ratio":  _safe("frequency_to_length_abs_ratio",  np.mean)   if aset == "words_only" else np.nan,
            "max_frequency_to_length_abs_ratio":   _safe("frequency_to_length_abs_ratio",  np.max)    if aset == "words_only" else np.nan,
            "mean_morphology_to_length_abs_ratio": _safe("morphology_to_length_abs_ratio", np.mean),
        })

    return pd.DataFrame(rows)


# ── private plot helpers ───────────────────────────────────────────────────────

def _plot_dominant_feature_counts(unit_dominance: pd.DataFrame, output_path: Path) -> None:
    criterion  = "dominant_feature_by_mean_abs_fi"
    group_cols = ["model_name", "layer", "analysis_set"]

    groups       = list(unit_dominance.groupby(group_cols, sort=True))
    group_labels = [f"{m}\n{l}\n{a}" for (m, l, a), _ in groups]
    n_groups     = len(groups)
    all_features = sorted(_FEAT_COLOR.keys())

    fracs: dict[str, list[float]] = {feat: [] for feat in all_features}
    for (_, _, _), grp in groups:
        n = len(grp)
        for feat in all_features:
            fracs[feat].append((grp[criterion] == feat).sum() / n)

    fig, ax = plt.subplots(figsize=(max(7, n_groups * 1.6), 5))
    x      = np.arange(n_groups)
    bottom = np.zeros(n_groups)

    for feat in all_features:
        vals = np.array(fracs[feat])
        if vals.sum() == 0:
            continue
        ax.bar(x, vals, bottom=bottom, color=_FEAT_COLOR[feat],
               label=feat, edgecolor="white", linewidth=0.5)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=8)
    ax.set_ylabel("Fraction of units")
    ax.set_ylim(0, 1.05)
    ax.set_title("Dominant feature per unit (mean |FI| criterion)", fontsize=11)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI)
    plt.close(fig)


def _plot_scatter_fi(
    profile_wide: pd.DataFrame,
    x_feat: str,
    y_feat: str,
    analysis_set: str,
    output_path: Path,
    title: str,
) -> None:
    sub = profile_wide[profile_wide["analysis_set"] == analysis_set].copy()
    if sub.empty:
        return

    x_col = f"{x_feat}_mean_fi_over_time"
    y_col = f"{y_feat}_mean_fi_over_time"
    if x_col not in sub.columns or y_col not in sub.columns:
        return

    sub = sub.dropna(subset=[x_col, y_col])
    if sub.empty:
        return

    _alpha = {"encodec": 0.5, "dac": 0.3}
    _msize = {"encodec": 4,   "dac": 2}

    models = sorted(sub["model_name"].unique())
    layers = sorted(sub["layer"].unique())
    row_map = {m: i for i, m in enumerate(models)}
    col_map = {l: i for i, l in enumerate(layers)}

    global_min = min(sub[x_col].min(), sub[y_col].min())
    global_max = max(sub[x_col].max(), sub[y_col].max())

    fig, axes = plt.subplots(
        len(models), len(layers),
        figsize=(len(layers) * 4.5, len(models) * 4),
        sharex=True, sharey=True, squeeze=False,
    )

    for model in models:
        for layer in layers:
            ax   = axes[row_map[model]][col_map[layer]]
            mask = (sub["model_name"] == model) & (sub["layer"] == layer)
            grp  = sub[mask]
            if grp.empty:
                ax.set_visible(False)
                continue
            alpha = _alpha.get(model, 0.4)
            ms    = _msize.get(model, 3)
            ax.scatter(grp[x_col].values, grp[y_col].values,
                       alpha=alpha, s=ms, color="#555555")
            ax.plot([global_min, global_max], [global_min, global_max],
                    "k--", linewidth=0.8, alpha=0.5, zorder=0)
            ax.set_title(f"{model} / {layer}", fontsize=9)

    for ri, model in enumerate(models):
        axes[ri][0].set_ylabel(y_feat, fontsize=8)
    for ci, layer in enumerate(layers):
        axes[-1][ci].set_xlabel(x_feat, fontsize=8)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI)
    plt.close(fig)


def _plot_peak_time_by_feature(profile_long: pd.DataFrame, output_path: Path) -> None:
    models = sorted(profile_long["model_name"].unique())
    layers = sorted(profile_long["layer"].unique())

    fig, axes = plt.subplots(
        len(models), len(layers),
        figsize=(len(layers) * 5, len(models) * 4),
        squeeze=False,
    )

    for ri, model in enumerate(models):
        for ci, layer in enumerate(layers):
            ax   = axes[ri][ci]
            mask = (profile_long["model_name"] == model) & (profile_long["layer"] == layer)
            sub  = profile_long[mask]

            feat_labels:   list[str]         = []
            data_per_feat: list[np.ndarray]  = []
            for feat in sorted(sub["feature"].unique()):
                vals = sub[sub["feature"] == feat]["relative_time_at_max_fi"].dropna().values
                if len(vals) > 0:
                    feat_labels.append(feat)
                    data_per_feat.append(vals)

            if not data_per_feat:
                ax.set_visible(False)
                continue

            bp = ax.boxplot(data_per_feat, labels=feat_labels, patch_artist=True, notch=False)
            for patch, feat in zip(bp["boxes"], feat_labels):
                patch.set_facecolor(_FEAT_COLOR.get(feat, "#888"))
                patch.set_alpha(0.7)

            ax.set_ylim(0, 1)
            ax.set_ylabel("Relative time at peak FI", fontsize=8)
            ax.set_title(f"{model} / {layer}", fontsize=9)
            ax.tick_params(axis="x", labelsize=7)

    fig.suptitle("Peak FI time by feature", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI)
    plt.close(fig)


def _plot_top_lexicality_units(
    top_lex_df: pd.DataFrame,
    top_k: int,
    output_path: Path,
) -> None:
    if top_lex_df.empty:
        return

    df = top_lex_df.head(top_k).copy()
    df["unit_label"] = df.apply(
        lambda r: f"{r['model_name'][:3]}/{r['layer'][:3]}/{r['neuron']}", axis=1
    )
    bar_colors = [_MODEL_COLOR.get(r["model_name"], "#888") for _, r in df.iterrows()]

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.55), 5))
    x = np.arange(len(df))
    ax.bar(x, df["lexicality_max_fi"].values, color=bar_colors)
    ax.set_xticks(x)
    ax.set_xticklabels(df["unit_label"].values, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("lexicality max FI")
    ax.set_title(f"Top-{top_k} lexicality-sensitive units (all_items)", fontsize=11)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

    legend_elements = [
        mpatches.Patch(facecolor=c, label=m)
        for m, c in _MODEL_COLOR.items()
        if m in df["model_name"].values
    ]
    if legend_elements:
        ax.legend(handles=legend_elements, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI)
    plt.close(fig)


# ── main entry point ──────────────────────────────────────────────────────────

def build_profiles(
    fi_dirs: list,
    output_dir: str | Path,
    top_k: int = 30,
    overwrite: bool = False,
) -> None:
    """Build unit feature profiles from a list of FI run directories."""
    output_dir = Path(output_dir)

    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"{output_dir} already exists. Use overwrite=True or --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    dfs = []
    for fi_dir in fi_dirs:
        df, _ = load_fi_run(Path(fi_dir))
        print(
            f"[unit_profiles] loaded {Path(fi_dir).name}: "
            f"{df['neuron'].nunique()} neurons, "
            f"analysis_set={df['analysis_set'].iloc[0]}, "
            f"layer={df['layer'].iloc[0]}"
        )
        dfs.append(df)

    fi_all = pd.concat(dfs, ignore_index=True)
    print(f"[unit_profiles] total rows: {len(fi_all)}")

    # 2. Compute
    print("[unit_profiles] computing unit feature profiles...")
    profile_long = compute_unit_feature_profiles(fi_all)

    print("[unit_profiles] building wide profile...")
    profile_wide = build_unit_profile_wide(profile_long)

    print("[unit_profiles] computing unit dominance...")
    unit_dominance = compute_unit_dominance(profile_long)

    print("[unit_profiles] summarising dominance...")
    dominance_summary = summarize_unit_dominance(unit_dominance)

    # 3. Top-k tables — join relative_time from profile_long
    def _get_times(feature: str) -> pd.DataFrame:
        sub = profile_long[profile_long["feature"] == feature][
            _UNIT_COLS + ["relative_time_at_max_fi"]
        ].copy()
        return sub.rename(columns={"relative_time_at_max_fi": f"{feature}_relative_time_at_max_fi"})

    dom = (
        unit_dominance
        .merge(_get_times("lexicality"),    on=_UNIT_COLS, how="left")
        .merge(_get_times("frequency_bin"), on=_UNIT_COLS, how="left")
        .merge(_get_times("morphology"),    on=_UNIT_COLS, how="left")
    )

    def _select(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        return df[[c for c in cols if c in df.columns]]

    top_lex = (
        dom[dom["analysis_set"] == "all_items"]
        .dropna(subset=["lexicality_max_fi"])
        .sort_values(["lexicality_max_fi", "lexicality_mean_fi"], ascending=False)
        .head(top_k)
        .pipe(_select, _UNIT_COLS + [
            "lexicality_mean_fi", "lexicality_mean_abs_fi", "lexicality_max_fi",
            "lexicality_relative_time_at_max_fi",
            "length_mean_fi",
            "lexicality_to_length_ratio", "lexicality_to_length_abs_ratio",
        ])
    )

    top_freq = (
        dom[dom["analysis_set"] == "words_only"]
        .dropna(subset=["frequency_max_fi"])
        .sort_values(["frequency_max_fi", "frequency_mean_fi"], ascending=False)
        .head(top_k)
        .pipe(_select, _UNIT_COLS + [
            "frequency_mean_fi", "frequency_mean_abs_fi", "frequency_max_fi",
            "frequency_bin_relative_time_at_max_fi",
            "length_mean_fi",
            "frequency_to_length_ratio", "frequency_to_length_abs_ratio",
        ])
    )

    top_morph = (
        dom
        .dropna(subset=["morphology_max_fi"])
        .sort_values(["morphology_max_fi", "morphology_mean_fi"], ascending=False)
        .head(top_k)
        .pipe(_select, _UNIT_COLS + [
            "morphology_mean_fi", "morphology_mean_abs_fi", "morphology_max_fi",
            "morphology_relative_time_at_max_fi",
            "length_mean_fi",
            "morphology_to_length_ratio", "morphology_to_length_abs_ratio",
        ])
    )

    # 4. Save CSVs
    csv_map = {
        "unit_fi_profiles_long.csv":  profile_long,
        "unit_fi_profiles_wide.csv":  profile_wide,
        "unit_feature_dominance.csv": unit_dominance,
        "unit_dominance_summary.csv": dominance_summary,
        "top_lexicality_units.csv":   top_lex,
        "top_frequency_units.csv":    top_freq,
        "top_morphology_units.csv":   top_morph,
    }
    for fname, df in csv_map.items():
        df.to_csv(output_dir / fname, index=False)
        print(f"[unit_profiles] saved {fname}: {df.shape}")

    # 5. Plots
    print("[unit_profiles] generating plots...")
    _plot_dominant_feature_counts(unit_dominance, output_dir / "dominant_feature_counts.png")
    _plot_scatter_fi(
        profile_wide, "length_bin", "lexicality", "all_items",
        output_dir / "lexicality_vs_length_units.png",
        "Lexicality vs Length mean FI per unit (all_items)",
    )
    _plot_scatter_fi(
        profile_wide, "length_bin", "frequency_bin", "words_only",
        output_dir / "frequency_vs_length_units.png",
        "Frequency_bin vs Length mean FI per unit (words_only)",
    )
    _plot_peak_time_by_feature(profile_long, output_dir / "peak_time_by_feature.png")
    _plot_top_lexicality_units(top_lex, top_k, output_dir / "lexicality_top_units_bar.png")

    print(f"[unit_profiles] done — outputs in {output_dir}")