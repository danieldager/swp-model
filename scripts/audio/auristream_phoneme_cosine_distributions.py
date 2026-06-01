#!/usr/bin/env python3
"""Cosine similarity and distance distributions for AuriStream phoneme embeddings.

Plots comparable to Nafis's neural cosine similarity distributions.

Cosine modes
------------
  raw      — L2-normalize each embedding vector directly
  centered — subtract per-dimension mean, then L2-normalize  (removes common direction)
  zscore   — z-score each dimension, then L2-normalize  (diagnostic only)

Analysis families
-----------------
  type              — C-C / C-V / V-V (Nafis color convention: blue / orange / green)
  identity_position — same Φ + same pos / same Φ + diff pos / diff Φ  [not yet implemented]

For count histograms, all categories are sampled to the same n so that bar heights
are directly comparable. Cosine distance = 1 − cosine similarity from the same sampled pairs.

Usage (smoke test — Family A, raw + centered):
    python scripts/audio/auristream_phoneme_cosine_distributions.py \\
        --run reproduce/data/audio/auristream__6ee9aeb6 \\
        --layers embedding block_24 block_48_lnf \\
        --analysis-families type \\
        --cosine-modes raw centered \\
        --max-samples-per-category 10000 \\
        --diagnose \\
        --overwrite

Full run:
    python scripts/audio/auristream_phoneme_cosine_distributions.py \\
        --run reproduce/data/audio/auristream__6ee9aeb6 \\
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_ARPABET_VOWELS: frozenset[str] = frozenset({
    "AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY", "IH", "IY",
    "OW", "OY", "UH", "UW",
})

# Nafis-comparable color convention
_TYPE_COLORS = {"C-C": "#1f77b4", "C-V": "#ff7f0e", "V-V": "#2ca02c"}
_TYPE_ALPHA  = 0.6

_DPI  = 150
_BINS = 50

# Layers for focused pre/post-ln_f comparison panels
_LAST_LAYER_TRIO = ("block_47", "block_48", "block_48_lnf")

DEFAULT_LAYERS = [
    "embedding", "block_01", "block_12", "block_24",
    "block_36", "block_47", "block_48_lnf",
]
DEFAULT_COSINE_MODES = ["raw", "centered"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def _phoneme_type(base: str) -> str:
    if base in _ARPABET_VOWELS:
        return "vowel"
    return "consonant" if base.isalpha() else "other"


def _stats(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {k: float("nan") for k in
                ["mean", "median", "std", "q01", "q05", "q25", "q75", "q95", "q99"]}
    return {
        "mean":   float(np.mean(values)),
        "median": float(np.median(values)),
        "std":    float(np.std(values)),
        "q01":    float(np.percentile(values, 1)),
        "q05":    float(np.percentile(values, 5)),
        "q25":    float(np.percentile(values, 25)),
        "q75":    float(np.percentile(values, 75)),
        "q95":    float(np.percentile(values, 95)),
        "q99":    float(np.percentile(values, 99)),
    }


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    pooled = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2.0)
    return float((a.mean() - b.mean()) / (pooled + 1e-12))


# ── Data loading ───────────────────────────────────────────────────────────────

def load_metadata(emb_dir: Path) -> pd.DataFrame:
    """Load metadata, compute phoneme_base and phoneme_type."""
    meta = pd.read_csv(emb_dir / "metadata_phonemes.csv")
    meta["phoneme_base"] = (
        meta["phoneme"].str.replace(r"\d+$", "", regex=True).str.upper()
    )
    meta["phoneme_type"] = meta["phoneme_base"].map(_phoneme_type)
    return meta


def load_layer(
    emb_dir: Path,
    layer: str,
    meta_all: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray] | None:
    """Load embeddings for one layer, filter valid non-silence phonemes.

    Returns (meta_valid, emb_raw: float32 [n, d]) or None if file not found.
    emb_raw is raw (not normalized); call get_emb_modes() to produce mode-specific arrays.
    """
    pt_path = emb_dir / f"embeddings_{layer}.pt"
    if not pt_path.exists():
        print(f"  [warning] embeddings_{layer}.pt not found — skipping")
        return None

    emb_raw_t = torch.load(pt_path, map_location="cpu", weights_only=True).float()
    assert len(meta_all) == emb_raw_t.shape[0], (
        f"Row mismatch: metadata={len(meta_all)}, embeddings={emb_raw_t.shape[0]}"
    )

    keep = (
        meta_all["valid_embedding"].astype(bool)
        & ~meta_all["is_silence"].astype(bool)
        & torch.isfinite(emb_raw_t).all(dim=1).numpy()
    )
    meta_v  = meta_all[keep].reset_index(drop=True)
    emb_raw = emb_raw_t[keep].numpy().astype(np.float32)

    n_v = int((meta_v["phoneme_type"] == "vowel").sum())
    n_c = int((meta_v["phoneme_type"] == "consonant").sum())
    print(f"  {layer}: {len(meta_v)} valid phonemes  (V={n_v}, C={n_c})  "
          f"shape={emb_raw.shape}")
    return meta_v, emb_raw


def get_emb_modes(emb_raw: np.ndarray, modes: list[str]) -> dict[str, np.ndarray]:
    """Compute L2-normalized embeddings for each requested cosine mode.

    raw      — normalize emb_raw directly
    centered — subtract per-dimension mean, then L2-normalize
    zscore   — z-score per dimension, then L2-normalize
    """
    result: dict[str, np.ndarray] = {}
    if "raw" in modes:
        norms = np.linalg.norm(emb_raw, axis=1, keepdims=True)
        result["raw"] = (emb_raw / (norms + 1e-12)).astype(np.float32)
    if "centered" in modes:
        centered = emb_raw - emb_raw.mean(axis=0)
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        result["centered"] = (centered / (norms + 1e-12)).astype(np.float32)
    if "zscore" in modes:
        std = emb_raw.std(axis=0, keepdims=True) + 1e-12
        zscored = (emb_raw - emb_raw.mean(axis=0)) / std
        norms = np.linalg.norm(zscored, axis=1, keepdims=True)
        result["zscore"] = (zscored / (norms + 1e-12)).astype(np.float32)
    return result


# ── Anisotropy / QC diagnostics ────────────────────────────────────────────────

def compute_anisotropy(
    emb_raw: np.ndarray,
    layer: str,
    rng: np.random.Generator,
) -> dict:
    """Compute norm distribution, mean direction, PCA, pairwise cosine, and duplicate check.

    Covers:
      - raw embedding norm distribution (min/median/mean/max/std)
      - global std and per-dimension std
      - isotropy_ratio: ||mean_vector|| / mean(||x_i||)  — 0=isotropic, 1=collapsed
      - cosine similarity of each vector with the layer mean direction
      - PCA explained variance ratio for PC1–PC3
      - pairwise cosine summary on a random sample
      - mean pairwise Euclidean distance on a random sample
      - number of unique rows (at 4 decimal places)
    """
    n, d = emb_raw.shape
    norms = np.linalg.norm(emb_raw, axis=1)
    emb_norm = (emb_raw / (norms[:, None] + 1e-12)).astype(np.float32)

    per_dim_std = emb_raw.std(axis=0)

    # Mean direction
    mean_vec = emb_raw.mean(axis=0)
    mean_vec_norm = float(np.linalg.norm(mean_vec))
    isotropy_ratio = mean_vec_norm / (float(norms.mean()) + 1e-12)
    mean_dir = (mean_vec / (mean_vec_norm + 1e-12)).astype(np.float32)
    cos_with_mean = emb_norm @ mean_dir

    # PCA on raw embeddings
    n_comp = min(10, d, n)
    pca = PCA(n_components=n_comp)
    pca.fit(emb_raw)
    evr = pca.explained_variance_ratio_

    # Pairwise cosine similarity on a random sample
    n_cos = min(500, n)
    cos_idx = rng.choice(n, n_cos, replace=False)
    cos_mat = emb_norm[cos_idx] @ emb_norm[cos_idx].T
    iu = np.triu_indices(n_cos, k=1)
    cos_pairs = cos_mat[iu]

    # Pairwise Euclidean distance on a smaller random sample (memory-efficient)
    n_eu = min(100, n)
    eu_idx = rng.choice(n, n_eu, replace=False)
    eu_mat = emb_raw[eu_idx].astype(np.float64)
    sq_norms = (eu_mat ** 2).sum(axis=1)
    dot = eu_mat @ eu_mat.T
    dist_sq = np.clip(sq_norms[:, None] + sq_norms[None, :] - 2.0 * dot, 0.0, None)
    eu_iu = np.triu_indices(n_eu, k=1)
    eu_dists = np.sqrt(dist_sq[eu_iu])

    # Near-duplicate rows (rounded to 4 decimal places)
    n_unique = int(pd.DataFrame(np.round(emb_raw, 4)).drop_duplicates().shape[0])

    return {
        "layer": layer, "n": n, "d": d,
        # Norm distribution (raw)
        "norm_min":    float(norms.min()),
        "norm_median": float(np.median(norms)),
        "norm_mean":   float(norms.mean()),
        "norm_max":    float(norms.max()),
        "norm_std":    float(norms.std()),
        # Variance
        "global_std":        float(emb_raw.std()),
        "per_dim_std_mean":  float(per_dim_std.mean()),
        "per_dim_std_median":float(np.median(per_dim_std)),
        # Mean direction / isotropy
        "mean_vec_norm":    float(mean_vec_norm),
        "isotropy_ratio":   float(isotropy_ratio),
        # Cosine similarity with mean direction
        "cos_with_mean_mean": float(cos_with_mean.mean()),
        "cos_with_mean_std":  float(cos_with_mean.std()),
        "cos_with_mean_min":  float(cos_with_mean.min()),
        "cos_with_mean_q05":  float(np.percentile(cos_with_mean, 5)),
        "cos_with_mean_q95":  float(np.percentile(cos_with_mean, 95)),
        "cos_with_mean_max":  float(cos_with_mean.max()),
        # PCA
        "pca_pc1_evr":  float(evr[0]),
        "pca_pc2_evr":  float(evr[1]) if len(evr) > 1 else float("nan"),
        "pca_pc3_evr":  float(evr[2]) if len(evr) > 2 else float("nan"),
        "pca_top3_evr": float(sum(evr[:3])),
        # Pairwise cosine (sampled)
        "cos_pairs_mean": float(cos_pairs.mean()),
        "cos_pairs_q05":  float(np.percentile(cos_pairs, 5)),
        "cos_pairs_q95":  float(np.percentile(cos_pairs, 95)),
        # Pairwise Euclidean (sampled)
        "eu_dist_mean":   float(eu_dists.mean()),
        "eu_dist_median": float(np.median(eu_dists)),
        # Unique rows
        "n_unique": n_unique,
    }


def _print_anisotropy_summary(row: dict) -> None:
    """Print a concise diagnostic summary for one layer."""
    print(
        f"    isotropy_ratio  : {row['isotropy_ratio']:.4f}  "
        f"(0=isotropic, 1=fully collapsed)\n"
        f"    cos_with_mean   : mean={row['cos_with_mean_mean']:.4f}  "
        f"std={row['cos_with_mean_std']:.4f}  "
        f"min={row['cos_with_mean_min']:.4f}  "
        f"[q05={row['cos_with_mean_q05']:.4f}, q95={row['cos_with_mean_q95']:.4f}]  "
        f"max={row['cos_with_mean_max']:.4f}\n"
        f"    PCA PC1 EVR     : {row['pca_pc1_evr']:.4f}  "
        f"top-3: {row['pca_top3_evr']:.4f}\n"
        f"    cos_pairs (sampled): mean={row['cos_pairs_mean']:.4f}  "
        f"[{row['cos_pairs_q05']:.4f}, {row['cos_pairs_q95']:.4f}]\n"
        f"    norm            : [{row['norm_min']:.2f}, {row['norm_median']:.2f}, "
        f"{row['norm_max']:.2f}]  std={row['norm_std']:.3f}\n"
        f"    global_std      : {row['global_std']:.4f}  "
        f"per_dim_std_mean={row['per_dim_std_mean']:.4f}\n"
        f"    eu_dist (sampled): mean={row['eu_dist_mean']:.3f}  "
        f"median={row['eu_dist_median']:.3f}\n"
        f"    n_unique        : {row['n_unique']} / {row['n']}"
    )


# ── Family A — phoneme-type cosine distributions ───────────────────────────────

def compute_type_pairs(
    emb_norm: np.ndarray,
    phoneme_type: np.ndarray,
    max_samples: int,
    rng: np.random.Generator,
) -> dict[str, dict]:
    """Compute C-C, C-V, V-V cosine similarity distributions.

    All categories are sampled to the same target n = min(max_samples, min_available)
    so that count histograms are directly comparable.

    Returns dict[category → {"sim": float32 array, "n_available": int, "n_sampled": int}].
    Cosine distance = 1 − sim; derive from the same arrays outside this function.
    """
    c_mask = phoneme_type == "consonant"
    v_mask = phoneme_type == "vowel"
    emb_c  = emb_norm[c_mask]
    emb_v  = emb_norm[v_mask]

    raw: dict[str, np.ndarray] = {}

    if len(emb_c) >= 2:
        sim_cc = emb_c @ emb_c.T
        iu = np.triu_indices(len(emb_c), k=1)
        raw["C-C"] = sim_cc[iu].astype(np.float32)
    else:
        raw["C-C"] = np.array([], dtype=np.float32)

    if len(emb_v) >= 2:
        sim_vv = emb_v @ emb_v.T
        iu = np.triu_indices(len(emb_v), k=1)
        raw["V-V"] = sim_vv[iu].astype(np.float32)
    else:
        raw["V-V"] = np.array([], dtype=np.float32)

    if len(emb_c) >= 1 and len(emb_v) >= 1:
        raw["C-V"] = (emb_c @ emb_v.T).ravel().astype(np.float32)
    else:
        raw["C-V"] = np.array([], dtype=np.float32)

    # Equal sample size across categories
    available = {cat: len(vals) for cat, vals in raw.items() if len(vals) > 0}
    n_sample = min(max_samples, min(available.values())) if available else 0

    results: dict[str, dict] = {}
    for cat, vals in raw.items():
        n_avail = len(vals)
        if n_avail == 0 or n_sample == 0:
            results[cat] = {"sim": np.array([], dtype=np.float32),
                            "n_available": n_avail, "n_sampled": 0}
        elif n_avail > n_sample:
            idx = rng.choice(n_avail, n_sample, replace=False)
            results[cat] = {"sim": vals[idx], "n_available": n_avail,
                            "n_sampled": n_sample}
        else:
            results[cat] = {"sim": vals, "n_available": n_avail,
                            "n_sampled": n_avail}
    return results


def _plot_type_hist(
    pairs: dict[str, dict],
    metric: str,
    output_path: Path,
    layer: str,
    bins: int,
    plot_density: bool,
    mode: str = "raw",
) -> None:
    """Single-layer histogram for one cosine mode × metric combination."""
    x_lim   = (-1, 1) if metric == "similarity" else (0, 2)
    x_label = ("Cosine similarity" if metric == "similarity"
                else "Cosine distance  (1 − cosine similarity)")
    y_label = "Density" if plot_density else "Count"

    fig, ax = plt.subplots(figsize=(7, 5))
    for cat, color in _TYPE_COLORS.items():
        vals = pairs[cat]["sim"]
        if len(vals) == 0:
            continue
        data = vals if metric == "similarity" else (1.0 - vals)
        n_s  = pairs[cat]["n_sampled"]
        ax.hist(data, bins=bins, range=x_lim, color=color, alpha=_TYPE_ALPHA,
                label=f"{cat}  (n={n_s:,})", density=plot_density,
                histtype="stepfilled", edgecolor="none")

    ax.set_xlim(x_lim)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{layer}  —  phoneme-type cosine {metric}  [{mode}]")
    ax.legend(fontsize=9)
    _savefig(fig, output_path)


def make_summary_panel(
    all_layer_data: dict[str, dict | None],
    metric: str,
    output_path: Path,
    bins: int,
    plot_density: bool,
    mode: str = "raw",
) -> None:
    """One subplot per layer, same-scale x-axis for comparability."""
    layers = [la for la, d in all_layer_data.items() if d is not None]
    if not layers:
        return
    n = len(layers)

    x_lim   = (-1, 1) if metric == "similarity" else (0, 2)
    x_label = ("Cosine similarity" if metric == "similarity"
                else "Cosine distance")

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        pairs = all_layer_data[layer]
        for cat, color in _TYPE_COLORS.items():
            vals = pairs[cat]["sim"]
            if len(vals) == 0:
                continue
            data = vals if metric == "similarity" else (1.0 - vals)
            n_s  = pairs[cat]["n_sampled"]
            ax.hist(data, bins=bins, range=x_lim, color=color, alpha=_TYPE_ALPHA,
                    label=f"{cat}  (n={n_s:,})", density=plot_density,
                    histtype="stepfilled", edgecolor="none")
        ax.set_xlim(x_lim)
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_title(layer, fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Density" if plot_density else "Count")
        ax.legend(fontsize=7)

    fig.suptitle(
        f"Phoneme-type cosine {metric}  [{mode}]  —  all layers", fontsize=11
    )
    fig.tight_layout()
    _savefig(fig, output_path)


def _type_stat_rows(layer: str, pairs: dict, mode: str) -> tuple[list, list]:
    """Build rows for cosine_summary_stats.csv and cosine_separation_metrics.csv."""
    stat_rows = []
    for cat, d in pairs.items():
        vals = d["sim"]
        row  = {"layer": layer, "mode": mode, "family": "type", "category": cat,
                "n_available": d["n_available"], "n_sampled": d["n_sampled"]}
        row.update(_stats(vals))
        if len(vals) > 0:
            dist_stats = _stats(1.0 - vals)
            for k, v in dist_stats.items():
                row[f"dist_{k}"] = v
        stat_rows.append(row)

    sep_rows = []
    for a, b in [("C-C", "C-V"), ("V-V", "C-V"), ("C-C", "V-V")]:
        va, vb = pairs[a]["sim"], pairs[b]["sim"]
        if len(va) == 0 or len(vb) == 0:
            continue
        sep_rows.append({
            "layer": layer, "mode": mode, "family": "type",
            "comparison": f"{a} vs {b}",
            "delta_mean_sim":  float(va.mean() - vb.mean()),
            "cohens_d_sim":    _cohens_d(va, vb),
            "delta_mean_dist": float((1-va).mean() - (1-vb).mean()),
            "cohens_d_dist":   _cohens_d(1-va, 1-vb),
        })
    return stat_rows, sep_rows


# ── Anisotropy metrics panel ───────────────────────────────────────────────────

def plot_anisotropy_metrics_across_layers(
    anisotropy_rows: list[dict],
    output_dir: Path,
) -> None:
    """Two compact figures showing anisotropy indicators and norm/PCA stats across layers."""
    if not anisotropy_rows:
        return

    df = pd.DataFrame(anisotropy_rows)
    layers = df["layer"].tolist()
    x = np.arange(len(layers))

    # ── Figure 1: primary anisotropy indicators ────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    bar_kw = dict(color="#2166ac", edgecolor="none", alpha=0.85)

    for ax, col, title in zip(
        axes,
        ["isotropy_ratio", "cos_pairs_mean", "cos_with_mean_mean"],
        [
            "Isotropy ratio\n(‖μ‖ / mean‖xᵢ‖)\n0=isotropic  1=collapsed",
            "Pairwise cosine\nmean (sampled)",
            "Cosine with mean\ndirection (mean)",
        ],
    ):
        ax.bar(x, df[col], **bar_kw)
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=35, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1.07)
        ax.axhline(1.0, color="#666", linewidth=0.8, linestyle="--")

    # Highlight the trio
    for ax in axes:
        for i, la in enumerate(layers):
            if la in _LAST_LAYER_TRIO:
                ax.get_children()[i].set_color("#d6604d")

    fig.suptitle("Raw-cosine anisotropy indicators  (primary: isotropy_ratio, cos_with_mean)", fontsize=10)
    fig.tight_layout()
    _savefig(fig, output_dir / "summary_anisotropy_raw_cosine_all_layers.png")

    # ── Figure 2: norm / PCA / Euclidean (supplementary) ──────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4))
    bar_kw2 = dict(color="#4dac26", edgecolor="none", alpha=0.85)

    for ax, col, title in zip(
        axes2,
        ["pca_pc1_evr", "norm_mean", "eu_dist_mean"],
        [
            "PCA PC1 EVR\n(note: PCA centers data —\nnot primary anisotropy indicator)",
            "Mean embedding norm\n(raw)",
            "Mean pairwise Euclidean\n(sampled, 100 pts)",
        ],
    ):
        vals = df[col].fillna(0)
        ax.bar(x, vals, **bar_kw2)
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=35, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9)

    for ax in axes2:
        for i, la in enumerate(layers):
            if la in _LAST_LAYER_TRIO:
                ax.get_children()[i].set_color("#f4a582")

    fig2.suptitle("Norm / PCA / Euclidean diagnostics  (supplementary)", fontsize=10)
    fig2.tight_layout()
    _savefig(fig2, output_dir / "summary_pca_norm_diagnostics_all_layers.png")

    print("[cosine_dist] Saved anisotropy panel figures (2 files)")


# ── Last-block delta diagnostic ────────────────────────────────────────────────

def compute_last_block_delta(
    emb_dir: Path,
    meta_all: pd.DataFrame,
    output_dir: Path,
    rng: np.random.Generator,
) -> None:
    """Diagnose the block_47 → block_48 residual update delta = block_48 − block_47.

    Determines whether the last Transformer block adds a nearly common vector to all
    phoneme representations, which would explain raw-cosine saturation in block_48.

    Saves:
        last_block_delta_diagnostics.csv  (per-phoneme)
        summary_last_block_delta_diagnostics.png
    """
    print("\n[cosine_dist] Last-block delta diagnostic (block_48 − block_47)")
    r47 = load_layer(emb_dir, "block_47", meta_all)
    r48 = load_layer(emb_dir, "block_48", meta_all)
    if r47 is None:
        print("  [delta] block_47 not found — skipping")
        return
    if r48 is None:
        print("  [delta] block_48 not found — skipping")
        return

    meta_v, emb47 = r47
    _,      emb48 = r48

    delta = emb48 - emb47  # [n, D]
    n = delta.shape[0]

    # Norms
    delta_norms = np.linalg.norm(delta, axis=1)
    b47_norms   = np.linalg.norm(emb47, axis=1)
    b48_norms   = np.linalg.norm(emb48, axis=1)

    # L2-normalize
    delta_n = (delta / (delta_norms[:, None] + 1e-12)).astype(np.float32)
    emb47_n = (emb47 / (b47_norms[:, None]  + 1e-12)).astype(np.float32)
    emb48_n = (emb48 / (b48_norms[:, None]  + 1e-12)).astype(np.float32)

    # Mean delta direction
    mean_delta     = delta.mean(axis=0)
    mean_delta_norm = float(np.linalg.norm(mean_delta))
    isotropy_delta  = mean_delta_norm / (float(delta_norms.mean()) + 1e-12)
    mean_delta_dir  = (mean_delta / (mean_delta_norm + 1e-12)).astype(np.float32)
    cos_delta_mean  = delta_n @ mean_delta_dir       # [n]

    # Cosine of each delta with the corresponding block_47 / block_48 vector
    cos_delta_b47 = (delta_n * emb47_n).sum(axis=1)  # [n]
    cos_delta_b48 = (delta_n * emb48_n).sum(axis=1)  # [n]

    # Pairwise cosine among deltas (sampled)
    n_cos = min(500, n)
    idx   = rng.choice(n, n_cos, replace=False)
    cos_mat = delta_n[idx] @ delta_n[idx].T
    iu = np.triu_indices(n_cos, k=1)
    cos_delta_pairs = cos_mat[iu]

    # Console summary
    print(f"    n phonemes      : {n}")
    print(f"    delta_norm      : mean={delta_norms.mean():.3f}  std={delta_norms.std():.3f}  "
          f"[{delta_norms.min():.3f}, {delta_norms.max():.3f}]")
    print(f"    isotropy_ratio  : {isotropy_delta:.4f}  (0=isotropic, 1=collapsed)")
    print(f"    cos(delta, μ_δ) : mean={cos_delta_mean.mean():.4f}  "
          f"std={cos_delta_mean.std():.4f}")
    print(f"    cos_pairs (sampled): mean={cos_delta_pairs.mean():.4f}  "
          f"[{np.percentile(cos_delta_pairs, 5):.4f}, {np.percentile(cos_delta_pairs, 95):.4f}]")
    print(f"    cos(delta, b47) : mean={cos_delta_b47.mean():.4f}")
    print(f"    cos(delta, b48) : mean={cos_delta_b48.mean():.4f}")

    # Per-phoneme CSV
    rows = []
    for i in range(n):
        rows.append({
            "item_id":              meta_v.iloc[i]["item_id"],
            "phoneme":              meta_v.iloc[i]["phoneme"],
            "phoneme_type":         meta_v.iloc[i]["phoneme_type"],
            "delta_norm":           float(delta_norms[i]),
            "block_47_norm":        float(b47_norms[i]),
            "block_48_norm":        float(b48_norms[i]),
            "cos_delta_with_mean":  float(cos_delta_mean[i]),
            "cos_delta_with_b47":   float(cos_delta_b47[i]),
            "cos_delta_with_b48":   float(cos_delta_b48[i]),
        })
    pd.DataFrame(rows).to_csv(output_dir / "last_block_delta_diagnostics.csv", index=False)

    # Figure: 3-panel
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].hist(delta_norms, bins=40, color="#1f77b4", alpha=0.85, edgecolor="none")
    axes[0].set_xlabel("‖block_48 − block_47‖")
    axes[0].set_title("Delta norm distribution")

    axes[1].hist(cos_delta_mean, bins=40, range=(-1, 1), color="#ff7f0e", alpha=0.85, edgecolor="none")
    axes[1].set_xlabel("cos(δᵢ,  mean δ)")
    axes[1].set_xlim(-1, 1)
    axes[1].set_title("Delta alignment with mean delta\n(1 = all deltas same direction)")

    axes[2].hist(cos_delta_pairs, bins=40, range=(-1, 1), color="#2ca02c", alpha=0.85, edgecolor="none")
    axes[2].set_xlabel("Pairwise cos(δᵢ, δⱼ)")
    axes[2].set_xlim(-1, 1)
    axes[2].set_title("Pairwise cosine among deltas")

    fig.suptitle("block_47 → block_48 residual update diagnostics", fontsize=11)
    fig.tight_layout()
    _savefig(fig, output_dir / "summary_last_block_delta_diagnostics.png")
    print("  [delta] Saved: last_block_delta_diagnostics.csv + summary_last_block_delta_diagnostics.png")


# ── CLI + main ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", required=True)
    p.add_argument("--embeddings-dir", default=None, dest="embeddings_dir")
    p.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS)
    p.add_argument("--include-raw-block48", action="store_true", dest="include_block48")
    p.add_argument(
        "--cosine-modes", nargs="+", default=DEFAULT_COSINE_MODES,
        choices=["raw", "centered", "zscore"],
        dest="cosine_modes",
        help="Normalization modes for cosine computation (default: raw centered).",
    )
    p.add_argument(
        "--diagnose", action="store_true",
        help=(
            "Run anisotropy / QC diagnostics per layer: norm distribution, mean direction, "
            "PCA EVR, pairwise cosine, Euclidean distance, duplicate check. "
            "Saves anisotropy_diagnostics.csv and two diagnostic panel figures."
        ),
    )
    p.add_argument(
        "--focused-last-layers", action="store_true",
        dest="focused_last_layers",
        help=(
            "Also produce summary panels restricted to block_47 / block_48 / block_48_lnf "
            "(summary_last_layers_cosine_*). All-layers panels are always produced."
        ),
    )
    p.add_argument(
        "--delta-diagnostic", action="store_true",
        dest="delta_diagnostic",
        help=(
            "Compute block_48 − block_47 residual delta diagnostics. "
            "Requires both layers to be in --layers. "
            "Saves last_block_delta_diagnostics.csv and summary figure."
        ),
    )
    p.add_argument("--analysis-families", nargs="+",
                   default=["type", "identity_position"],
                   choices=["type", "identity_position"],
                   dest="families")
    p.add_argument("--max-samples-per-category", type=int, default=50000,
                   dest="max_samples")
    p.add_argument("--min-phone-count", type=int, default=3, dest="min_phone_count")
    p.add_argument("--max-position", type=int, default=None, dest="max_position")
    p.add_argument("--random-state", type=int, default=0, dest="random_state")
    p.add_argument("--plot-density", action="store_true", dest="plot_density")
    p.add_argument("--bins", type=int, default=_BINS)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--output", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = (REPO_ROOT / args.run).resolve()
    emb_dir = (Path(args.embeddings_dir).resolve() if args.embeddings_dir
               else run_dir / "phoneme_embeddings")

    with open(run_dir / "manifest.json") as f:
        run_id = json.load(f)["run_id"]

    output_dir = (Path(args.output).resolve() if args.output
                  else REPO_ROOT / "reproduce" / "figures" / "audio"
                       / "auristream_phoneme_cosine" / run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.overwrite and (output_dir / "config.json").exists():
        print(f"[cosine_dist] Output exists: {output_dir}\nPass --overwrite to regenerate.")
        return

    rng = np.random.default_rng(args.random_state)
    rng_diag = np.random.default_rng(args.random_state + 1)

    meta_all = load_metadata(emb_dir)
    print(f"[cosine_dist] Total phonemes in metadata: {len(meta_all)}")

    base_keep = (
        meta_all["valid_embedding"].astype(bool)
        & ~meta_all["is_silence"].astype(bool)
    )
    n_base = int(base_keep.sum())
    n_v = int((meta_all.loc[base_keep, "phoneme_type"] == "vowel").sum())
    n_c = int((meta_all.loc[base_keep, "phoneme_type"] == "consonant").sum())
    n_o = int((meta_all.loc[base_keep, "phoneme_type"] == "other").sum())
    print(f"  Base valid (non-silence): {n_base}  (V={n_v}, C={n_c}, other={n_o})")
    print(f"  Cosine modes: {args.cosine_modes}")

    layers = list(args.layers)
    if args.include_block48 and "block_48" not in layers:
        layers = layers + ["block_48"]

    # Accumulate per-mode results: (layer, mode) → pairs dict
    all_type_pairs: dict[tuple[str, str], dict | None] = {}
    all_stat_rows:  list[dict] = []
    all_sep_rows:   list[dict] = []
    sampling_rows:  list[dict] = []
    anisotropy_rows: list[dict] = []

    # ── Family A ──────────────────────────────────────────────────────────────
    if "type" in args.families:
        type_dir = output_dir / "phoneme_type"
        print("\n[cosine_dist] Family A — phoneme type")

        for layer in layers:
            result = load_layer(emb_dir, layer, meta_all)
            if result is None:
                for mode in args.cosine_modes:
                    all_type_pairs[(layer, mode)] = None
                continue

            meta_v, emb_raw = result

            # ── Anisotropy diagnostics ────────────────────────────────────────
            if args.diagnose:
                print(f"\n  [diagnose] {layer}")
                diag = compute_anisotropy(emb_raw, layer, rng_diag)
                anisotropy_rows.append(diag)
                _print_anisotropy_summary(diag)

            # ── Compute cosine modes and Family A pairs ───────────────────────
            mode_embs = get_emb_modes(emb_raw, args.cosine_modes)

            cv_mask = meta_v["phoneme_type"].isin(["consonant", "vowel"]).values
            if cv_mask.sum() < 4:
                print(f"  [warning] {layer}: too few C/V phonemes — skipping")
                for mode in args.cosine_modes:
                    all_type_pairs[(layer, mode)] = None
                continue

            layer_dir = type_dir / layer
            layer_dir.mkdir(parents=True, exist_ok=True)

            csv_rows_all: list[dict] = []

            for mode, emb_norm in mode_embs.items():
                pairs = compute_type_pairs(
                    emb_norm[cv_mask],
                    meta_v["phoneme_type"].values[cv_mask],
                    args.max_samples,
                    rng,
                )
                all_type_pairs[(layer, mode)] = pairs

                # Sampled values CSV (all modes appended, mode column added)
                for cat, d in pairs.items():
                    for v in d["sim"]:
                        csv_rows_all.append({
                            "mode": mode, "category": cat,
                            "cosine_similarity": float(v),
                            "cosine_distance": float(1.0 - v),
                        })

                # Histograms (mode × metric)
                for metric in ["similarity", "distance"]:
                    fname = f"cosine_{mode}_{metric}_type_{layer}.png"
                    _plot_type_hist(pairs, metric, layer_dir / fname, layer,
                                    args.bins, plot_density=False, mode=mode)
                    if args.plot_density:
                        _plot_type_hist(
                            pairs, metric,
                            layer_dir / fname.replace(".png", "_density.png"),
                            layer, args.bins, plot_density=True, mode=mode,
                        )

                # Stats
                s_rows, sep_rows = _type_stat_rows(layer, pairs, mode)
                all_stat_rows.extend(s_rows)
                all_sep_rows.extend(sep_rows)
                for cat, d in pairs.items():
                    sampling_rows.append({
                        "layer": layer, "mode": mode, "family": "type",
                        "category": cat,
                        "n_available": d["n_available"], "n_sampled": d["n_sampled"],
                    })

            pd.DataFrame(csv_rows_all).to_csv(
                layer_dir / f"cosine_values_type_{layer}.csv", index=False
            )

        # Summary panels (one per mode × metric)
        for mode in args.cosine_modes:
            valid_data = {
                la: all_type_pairs[(la, mode)]
                for la in layers
                if all_type_pairs.get((la, mode)) is not None
            }
            if not valid_data:
                continue
            for metric in ["similarity", "distance"]:
                # All-layers panel
                out = output_dir / f"summary_all_layers_cosine_{mode}_{metric}_type.png"
                make_summary_panel(valid_data, metric, out, args.bins,
                                   plot_density=False, mode=mode)
                print(f"  saved  {out.name}")
                if args.plot_density:
                    make_summary_panel(
                        valid_data, metric,
                        output_dir / f"summary_all_layers_cosine_{mode}_{metric}_type_density.png",
                        args.bins, plot_density=True, mode=mode,
                    )

                # Focused last-layer panel (block_47 / block_48 / block_48_lnf)
                if args.focused_last_layers:
                    trio_data = {
                        la: all_type_pairs[(la, mode)]
                        for la in _LAST_LAYER_TRIO
                        if all_type_pairs.get((la, mode)) is not None
                    }
                    if trio_data:
                        out = output_dir / f"summary_last_layers_cosine_{mode}_{metric}_type.png"
                        make_summary_panel(trio_data, metric, out, args.bins,
                                           plot_density=False, mode=mode)
                        print(f"  saved  {out.name}")

    # ── Last-block delta diagnostic ───────────────────────────────────────────
    if args.delta_diagnostic:
        compute_last_block_delta(emb_dir, meta_all, output_dir, rng)

    # ── Family B — placeholder ─────────────────────────────────────────────────
    if "identity_position" in args.families:
        print("\n[cosine_dist] Family B (identity × position) — not yet implemented.")

    # ── Write outputs ──────────────────────────────────────────────────────────
    cfg = {
        "run_id": run_id, "layers": layers, "families": args.families,
        "cosine_modes": args.cosine_modes,
        "max_samples": args.max_samples, "min_phone_count": args.min_phone_count,
        "max_position": args.max_position, "random_state": args.random_state,
        "bins": args.bins, "diagnose": args.diagnose,
    }
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    pd.DataFrame([{
        "n_total": len(meta_all), "n_valid_base": n_base,
        "n_vowel": n_v, "n_consonant": n_c, "n_other": n_o,
    }]).to_csv(output_dir / "metadata_qc.csv", index=False)

    if sampling_rows:
        pd.DataFrame(sampling_rows).to_csv(
            output_dir / "pair_sampling_summary.csv", index=False
        )
    if all_stat_rows:
        pd.DataFrame(all_stat_rows).to_csv(
            output_dir / "cosine_summary_stats.csv", index=False
        )
    if all_sep_rows:
        pd.DataFrame(all_sep_rows).to_csv(
            output_dir / "cosine_separation_metrics.csv", index=False
        )
    if anisotropy_rows:
        pd.DataFrame(anisotropy_rows).to_csv(
            output_dir / "anisotropy_diagnostics.csv", index=False
        )
        print(f"\n[cosine_dist] Saved anisotropy_diagnostics.csv  ({len(anisotropy_rows)} rows)")
        plot_anisotropy_metrics_across_layers(anisotropy_rows, output_dir)

        # Focused pre/post-ln_f comparison (saved whenever all 3 layers are present)
        _LNF_TRIO = {"block_47", "block_48", "block_48_lnf"}
        analyzed_layers = {r["layer"] for r in anisotropy_rows}
        if _LNF_TRIO <= analyzed_layers:
            trio_rows = [r for r in anisotropy_rows if r["layer"] in _LNF_TRIO]
            trio_rows.sort(key=lambda r: ["block_47", "block_48", "block_48_lnf"].index(r["layer"]))
            trio_path = output_dir / "anisotropy_diagnostics_block47_block48_block48lnf.csv"
            pd.DataFrame(trio_rows).to_csv(trio_path, index=False)
            print(f"[cosine_dist] Saved focused comparison: {trio_path.name}")

    print(f"\n[cosine_dist] Done — {output_dir}")


if __name__ == "__main__":
    main()