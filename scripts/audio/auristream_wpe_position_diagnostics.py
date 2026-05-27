#!/usr/bin/env python3
"""Learned positional embedding (wpe) diagnostics for AuriStream.

Seven diagnostics targeting the norm-vs-position relationship in wpe:
  1. Direct wpe norm table         — is the decrease global or early-position only?
  2. Random initialization control — is the decrease hard-coded by init?
  3. Phoneme-level projection      — does wpe explain the phoneme-position norm effect?
  4. Position shift control        — is the effect absolute-position-table specific?
  5. Shuffled-position control     — does the ordered structure of wpe matter?
  6. Directional geometry          — is wpe only a magnitude signal, or direction-based?
  7. Propagation across layers     — does the wpe signal persist in deeper layers?

Usage (smoke test):
    python scripts/audio/auristream_wpe_position_diagnostics.py \\
        --run reproduce/data/audio/auristream__6ee9aeb6 \\
        --layers embedding block_01 block_24 block_48_lnf \\
        --max-position-to-plot 512 \\
        --random-init-seeds 5 \\
        --overwrite

Usage (full run):
    python scripts/audio/auristream_wpe_position_diagnostics.py \\
        --run reproduce/data/audio/auristream__6ee9aeb6 \\
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_ID  = "TuKoResearch/AuriStream1B_librilight_ckpt500k"
SEQ_LEN   = 4096
N_EMBD    = 1280
WPE_INIT_STD = 0.02   # from _init_weights in modeling_auristream.py

DEFAULT_LAYERS = [
    "embedding", "block_01", "block_12", "block_24",
    "block_36", "block_47", "block_48_lnf",
]

_ARPABET_VOWELS: frozenset[str] = frozenset({
    "AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY", "IH", "IY",
    "OW", "OY", "UH", "UW",
})

_DPI       = 150
_SLIDE_FIG = (10, 5)
_WIDE_FIG  = (12, 4)
_SQ_FIG    = (6, 5)

# Figures copied to slide_figures/ (in order of importance)
_SLIDE_KEYS = [
    "04_trained_wpe_vs_random_init_norm.png",
    "07_mean_wpe_norm_by_phoneme_position.png",
    "09_shift_control_wpe_norm_by_phoneme_position.png",
    "10_position_shuffle_control.png",
    "11_wpe_cosine_distance_by_lag.png",
    "15_layer_norm_wpe_correlation_by_layer.png",
]


# ── Figure / stats helpers ─────────────────────────────────────────────────────

def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def _corr_stats(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {"pearson": float("nan"), "spearman": float("nan"),
                "slope": float("nan"), "r2": float("nan"), "n": int(len(x))}
    r,   p_r   = stats.pearsonr(x, y)
    rho, p_rho = stats.spearmanr(x, y)
    sl, ic, rv, _, _ = stats.linregress(x, y)
    return {
        "pearson": float(r), "p_pearson": float(p_r),
        "spearman": float(rho), "p_spearman": float(p_rho),
        "slope": float(sl), "intercept": float(ic),
        "r2": float(rv**2), "n": int(len(x)),
    }


def _enrich_meta(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    if "phoneme_base" not in meta.columns:
        meta["phoneme_base"] = (
            meta["phoneme"].str.replace(r"\d+$", "", regex=True).str.upper()
        )
    if "phoneme_type" not in meta.columns:
        def _pt(b: str) -> str:
            if b in _ARPABET_VOWELS:
                return "vowel"
            return "consonant" if b.isalpha() else "other"
        meta["phoneme_type"] = meta["phoneme_base"].map(_pt)
    return meta


# ── Diagnostic 1 — Direct wpe norm table ──────────────────────────────────────

def run_diag1(wpe: torch.Tensor, output_dir: Path, max_pos: int) -> dict:
    norms = wpe.norm(dim=-1).numpy()
    positions = np.arange(len(norms))
    s_full = _corr_stats(positions.astype(float), norms)
    s_used = _corr_stats(positions[:max_pos].astype(float), norms[:max_pos])
    deltas = np.diff(norms)
    frac_neg = float((deltas < 0).mean())

    pd.DataFrame({"position": positions, "wpe_norm": norms}).to_csv(
        output_dir / "wpe_global_stats.csv", index=False
    )

    # Fig 01 — full context
    fig, ax = plt.subplots(figsize=_SLIDE_FIG)
    ax.plot(positions, norms, lw=0.6, color="#1f77b4", alpha=0.6)
    roll = pd.Series(norms).rolling(50, center=True).mean().values
    ax.plot(positions, roll, lw=2, color="#d62728", label="50-pos rolling mean")
    ax.set_xlabel("Absolute token position (0…4095)")
    ax.set_ylabel("L2 norm of wpe[p]")
    ax.set_title(f"wpe norm — full context (seq_len={SEQ_LEN})\n"
                 f"Pearson r={s_full['pearson']:.3f},  slope={s_full['slope']:.5f}/pos")
    ax.legend(fontsize=9)
    _savefig(fig, output_dir / "01_wpe_norm_full_context.png")

    # Fig 02 — used range
    pos_u, norm_u = positions[:max_pos], norms[:max_pos]
    fig, ax = plt.subplots(figsize=_SLIDE_FIG)
    ax.plot(pos_u, norm_u, lw=1.5, color="#1f77b4")
    m, b = s_used["slope"], s_used["intercept"]
    ax.plot(pos_u, m * pos_u + b, lw=2, color="#d62728", linestyle="--",
            label=f"OLS fit (slope={m:.5f}, R²={s_used['r2']:.3f})")
    ax.set_xlabel(f"Absolute token position (0…{max_pos-1})")
    ax.set_ylabel("L2 norm of wpe[p]")
    ax.set_title(f"wpe norm — positions used by stimuli (0–{max_pos})\n"
                 f"Pearson r={s_used['pearson']:.3f},  Spearman ρ={s_used['spearman']:.3f}")
    ax.legend(fontsize=9)
    _savefig(fig, output_dir / "02_wpe_norm_used_range.png")

    # Fig 03 — adjacent delta
    fig, ax = plt.subplots(figsize=_SLIDE_FIG)
    ax.plot(positions[1:max_pos], deltas[:max_pos-1], lw=0.6, color="#7f7f7f", alpha=0.7)
    roll_d = pd.Series(deltas[:max_pos-1]).rolling(20, center=True).mean().values
    ax.plot(positions[1:max_pos], roll_d, lw=2, color="#2ca02c", label="20-pos rolling mean")
    ax.axhline(0, color="black", lw=0.8, linestyle=":")
    ax.set_xlabel(f"Absolute token position (0…{max_pos-1})")
    ax.set_ylabel("Δnorm  (norm[p] − norm[p−1])")
    ax.set_title(f"wpe norm adjacent delta\nFraction of negative deltas: {frac_neg:.3f}")
    ax.legend(fontsize=9)
    _savefig(fig, output_dir / "03_wpe_norm_derivative.png")

    return {
        "norm_mean": float(norms.mean()), "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),   "norm_max": float(norms.max()),
        "full_context_corr": s_full,
        "used_range_corr": s_used,
        "fraction_adjacent_deltas_negative": frac_neg,
    }


# ── Diagnostic 2 — Random initialization control ──────────────────────────────

def run_diag2(wpe: torch.Tensor, output_dir: Path, n_seeds: int, max_pos: int) -> dict:
    norms_trained = wpe.norm(dim=-1).numpy()
    positions = np.arange(len(norms_trained))
    sl_tr, _, _, _, _ = stats.linregress(
        positions[:max_pos].astype(float), norms_trained[:max_pos]
    )

    random_norms, random_slopes = [], []
    for seed in range(n_seeds):
        rng_np = np.random.default_rng(seed)
        rand_wpe = torch.tensor(
            rng_np.normal(0.0, WPE_INIT_STD, (SEQ_LEN, N_EMBD)), dtype=torch.float32
        )
        rn = rand_wpe.norm(dim=-1).numpy()
        random_norms.append(rn)
        sl, _, _, _, _ = stats.linregress(positions[:max_pos].astype(float), rn[:max_pos])
        random_slopes.append(float(sl))

    rand_mat  = np.stack(random_norms)
    rand_mean = rand_mat.mean(axis=0)
    rand_std  = rand_mat.std(axis=0)
    z_score   = (sl_tr - np.mean(random_slopes)) / (np.std(random_slopes) + 1e-12)

    pd.DataFrame({
        "seed": list(range(n_seeds)) + ["trained"],
        "slope": random_slopes + [float(sl_tr)],
        "type": ["random"] * n_seeds + ["trained"],
    }).to_csv(output_dir / "wpe_random_init_control.csv", index=False)

    # Fig 04
    fig, axes = plt.subplots(1, 2, figsize=_SLIDE_FIG)
    ax = axes[0]
    ax.plot(positions[:max_pos], norms_trained[:max_pos], lw=2.5,
            color="#1f77b4", label="Trained wpe")
    ax.fill_between(positions[:max_pos],
                    (rand_mean - rand_std)[:max_pos],
                    (rand_mean + rand_std)[:max_pos],
                    color="#d62728", alpha=0.2, label=f"Random init ±1σ (n={n_seeds})")
    ax.plot(positions[:max_pos], rand_mean[:max_pos], lw=1.5,
            color="#d62728", linestyle="--", alpha=0.8)
    ax.set_xlabel(f"Token position (0…{max_pos-1})")
    ax.set_ylabel("L2 norm")
    ax.set_title("Trained vs random-init wpe norm")
    ax.legend(fontsize=8)

    ax = axes[1]
    def _z(v):
        return (v - v.mean()) / (v.std() + 1e-12)
    ax.plot(positions[:max_pos], _z(norms_trained[:max_pos]),
            lw=2.5, color="#1f77b4", label="Trained wpe (z-scored)")
    for rn in random_norms:
        ax.plot(positions[:max_pos], _z(np.asarray(rn[:max_pos])),
                lw=0.5, color="#d62728", alpha=0.25)
    ax.set_xlabel(f"Token position (0…{max_pos-1})")
    ax.set_ylabel("z-scored norm")
    ax.set_title("Z-scored comparison")
    ax.legend(fontsize=8)
    fig.suptitle(f"Trained wpe vs random initialization (n={n_seeds} seeds)", fontsize=11)
    fig.tight_layout()
    _savefig(fig, output_dir / "04_trained_wpe_vs_random_init_norm.png")

    # Fig 05
    fig, ax = plt.subplots(figsize=_SQ_FIG)
    ax.hist(random_slopes, bins=max(8, n_seeds // 2), color="#d62728", alpha=0.7,
            label=f"Random init slopes (n={n_seeds})")
    ax.axvline(sl_tr, color="#1f77b4", lw=2.5,
               label=f"Trained slope = {sl_tr:.5f}")
    ax.set_xlabel("Slope of wpe_norm vs position (used range)")
    ax.set_ylabel("Count")
    ax.set_title(f"Trained slope vs random distribution\nz-score = {z_score:.2f}")
    ax.legend(fontsize=9)
    _savefig(fig, output_dir / "05_trained_slope_vs_random_slope_distribution.png")

    return {
        "n_seeds": n_seeds,
        "trained_slope_used_range": float(sl_tr),
        "random_slope_mean": float(np.mean(random_slopes)),
        "random_slope_std":  float(np.std(random_slopes)),
        "trained_slope_z_score": float(z_score),
    }


# ── Diagnostic 3 — Phoneme-level projection ────────────────────────────────────

def run_diag3(
    wpe: torch.Tensor,
    meta_all: pd.DataFrame,
    keep: np.ndarray,
    emb_embedding: torch.Tensor,
    output_dir: Path,
) -> tuple[dict, pd.DataFrame]:
    meta_v = meta_all[keep].copy().reset_index(drop=True)
    orig_idx = np.where(keep)[0]
    emb_v = emb_embedding[orig_idx].float()

    wpe_np = wpe.numpy()
    embedding_norm   = emb_v.norm(dim=-1).numpy()
    mean_wpe_norm    = np.empty(len(meta_v), dtype=np.float32)
    mean_abs_tok_pos = np.empty(len(meta_v), dtype=np.float32)

    for i in range(len(meta_v)):
        row = meta_v.iloc[i]
        tok_s, tok_e = int(row["token_start_idx"]), int(row["token_end_idx"])
        wpe_slice = wpe_np[tok_s : tok_e + 1]
        mean_wpe_norm[i]    = np.linalg.norm(wpe_slice, axis=-1).mean()
        mean_abs_tok_pos[i] = (tok_s + tok_e) / 2.0

    meta_v["embedding_norm"]    = embedding_norm
    meta_v["mean_wpe_norm"]     = mean_wpe_norm
    meta_v["mean_abs_tok_pos"]  = mean_abs_tok_pos
    meta_v = _enrich_meta(meta_v)
    meta_v.to_csv(output_dir / "phoneme_wpe_projection.csv", index=False)

    ph_idx = meta_v["phoneme_position_from_start"].values.astype(float)
    c_wpe  = _corr_stats(embedding_norm, mean_wpe_norm)
    c_pos  = _corr_stats(embedding_norm, ph_idx)
    c_tok  = _corr_stats(mean_wpe_norm,  ph_idx)
    c_atp  = _corr_stats(embedding_norm, mean_abs_tok_pos)

    def _ols1(x_col: str):
        x = meta_v[x_col].values.astype(float)
        sl, ic, rv, _, _ = stats.linregress(x, embedding_norm)
        return {"slope": float(sl), "intercept": float(ic), "r2": float(rv**2)}

    def _ols_multi(x_cols: list[str]):
        X = np.column_stack([np.ones(len(meta_v))] + [meta_v[c].values for c in x_cols])
        coef, _, _, _ = np.linalg.lstsq(X, embedding_norm, rcond=None)
        yhat = X @ coef
        ss_res = ((embedding_norm - yhat)**2).sum()
        ss_tot = ((embedding_norm - embedding_norm.mean())**2).sum()
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        return {"coef": coef.tolist(), "r2": r2}

    stats_out = {
        "n_phonemes": len(meta_v),
        "corr_embedding_norm_vs_mean_wpe_norm": c_wpe,
        "corr_embedding_norm_vs_phoneme_index": c_pos,
        "corr_mean_wpe_norm_vs_phoneme_index":  c_tok,
        "corr_embedding_norm_vs_mean_abs_tok_pos": c_atp,
        "regression_A_vs_phoneme_idx":     _ols1("phoneme_position_from_start"),
        "regression_B_vs_mean_abs_tok_pos": _ols1("mean_abs_tok_pos"),
        "regression_C_vs_mean_wpe_norm":    _ols1("mean_wpe_norm"),
        "regression_D_vs_wpe_duration_ntokens": _ols_multi(
            ["mean_wpe_norm", "duration_s", "n_tokens"]
        ),
    }

    # Fig 06
    fig, ax = plt.subplots(figsize=_SQ_FIG)
    ax.scatter(mean_wpe_norm, embedding_norm, s=6, alpha=0.35,
               color="#1f77b4", rasterized=True)
    ax.set_xlabel("mean_wpe_norm  (mean ||wpe[p]|| over phoneme tokens)")
    ax.set_ylabel("embedding_norm  (||mean_pool(h0)||)")
    ax.set_title(f"embedding_norm vs mean_wpe_norm\n"
                 f"Pearson r={c_wpe['pearson']:.3f},  Spearman ρ={c_wpe['spearman']:.3f}"
                 f"  (n={len(meta_v)})")
    _savefig(fig, output_dir / "06_embedding_norm_vs_mean_wpe_norm.png")

    def _by_pos(col):
        return (
            meta_v.groupby("phoneme_position_from_start")[col]
            .agg(mean="mean", sem=lambda x: x.std() / np.sqrt(len(x)), n="count")
            .query("n >= 3").reset_index()
        )

    bp_wpe = _by_pos("mean_wpe_norm")
    bp_emb = _by_pos("embedding_norm")

    # Fig 07
    fig, ax = plt.subplots(figsize=_SLIDE_FIG)
    ax.errorbar(bp_wpe["phoneme_position_from_start"], bp_wpe["mean"],
                yerr=bp_wpe["sem"], color="#D55E00", marker="s", ms=5,
                lw=1.8, label="mean_wpe_norm ± SEM")
    ax.set_xlabel("Phoneme position from word start")
    ax.set_ylabel("mean_wpe_norm  (mean ± SEM)")
    ax.set_title(f"Projected wpe norm by phoneme position\n"
                 f"Pearson r(vs phoneme_idx)={c_tok['pearson']:.3f}")
    ax.legend(fontsize=9)
    _savefig(fig, output_dir / "07_mean_wpe_norm_by_phoneme_position.png")

    # Fig 08
    fig, ax = plt.subplots(figsize=_SLIDE_FIG)
    ax.errorbar(bp_emb["phoneme_position_from_start"], bp_emb["mean"],
                yerr=bp_emb["sem"], color="#0072B2", marker="o", ms=5,
                lw=1.8, label="embedding_norm")
    ax.errorbar(bp_wpe["phoneme_position_from_start"], bp_wpe["mean"],
                yerr=bp_wpe["sem"], color="#D55E00", marker="s", ms=5,
                lw=1.8, linestyle="--", label="mean_wpe_norm")
    ax.set_xlabel("Phoneme position from word start")
    ax.set_ylabel("L2 norm  (mean ± SEM)")
    ax.set_title("Embedding norm vs projected wpe norm by phoneme position")
    ax.legend(fontsize=9)
    _savefig(fig, output_dir / "08_embedding_norm_vs_wpe_norm_by_phoneme_position.png")

    return stats_out, meta_v


# ── Diagnostic 4 — Position shift control ─────────────────────────────────────

def run_diag4(
    wpe: torch.Tensor,
    meta_v: pd.DataFrame,
    output_dir: Path,
    shifts: list[int],
) -> dict:
    wpe_np = wpe.numpy()
    n_wpe  = len(wpe_np)
    stats_out = {}
    shift_curves: dict[int, pd.DataFrame] = {}

    for shift in shifts:
        norms_s, pos_s = [], []
        for _, row in meta_v.iterrows():
            tok_s = int(row["token_start_idx"]) + shift
            tok_e = int(row["token_end_idx"])   + shift
            if tok_e >= n_wpe:
                norms_s.append(float("nan"))
            else:
                norms_s.append(float(np.linalg.norm(wpe_np[tok_s : tok_e + 1], axis=-1).mean()))
            pos_s.append(int(row["phoneme_position_from_start"]))

        tmp = pd.DataFrame({"pos": pos_s, "norm": norms_s}).dropna()
        by_pos = (tmp.groupby("pos")["norm"].agg(mean="mean", n="count")
                  .query("n >= 3").reset_index())
        shift_curves[shift] = by_pos
        stats_out[f"shift_{shift}"] = _corr_stats(
            tmp["pos"].values.astype(float), tmp["norm"].values
        )

    colors = plt.cm.viridis(np.linspace(0, 0.85, len(shifts)))
    fig, ax = plt.subplots(figsize=_SLIDE_FIG)
    for (shift, bp), color in zip(shift_curves.items(), colors):
        lbl = f"shift={shift}" + (" (observed)" if shift == 0 else "")
        ax.plot(bp["pos"], bp["mean"], color=color,
                lw=2.5 if shift == 0 else 1.5, marker="o", ms=3, label=lbl)
    ax.set_xlabel("Phoneme position from word start")
    ax.set_ylabel("Mean projected wpe_norm")
    ax.set_title("Position shift control: projected wpe_norm by phoneme position\n"
                 "(shift = additional token offset applied to all absolute positions)")
    ax.legend(fontsize=8, ncol=2)
    _savefig(fig, output_dir / "09_shift_control_wpe_norm_by_phoneme_position.png")

    return stats_out


# ── Diagnostic 5 — Shuffled-position control ──────────────────────────────────

def run_diag5(
    wpe: torch.Tensor,
    meta_v: pd.DataFrame,
    output_dir: Path,
    n_shuffles: int,
    rng: np.random.Generator,
) -> dict:
    norms_all = wpe.norm(dim=-1).numpy()
    n_wpe     = len(norms_all)
    pos_list  = meta_v["phoneme_position_from_start"].values.astype(int)

    # Observed
    obs_norm = [
        float(norms_all[int(r["token_start_idx"]) : int(r["token_end_idx"]) + 1].mean())
        for _, r in meta_v.iterrows()
    ]
    obs_df = pd.DataFrame({"pos": pos_list, "norm": obs_norm})
    obs_curve = (obs_df.groupby("pos")["norm"].mean().reset_index()
                 .rename(columns={"pos": "phoneme_position_from_start"}))
    obs_slope, _, _, _, _ = stats.linregress(
        obs_curve["phoneme_position_from_start"].values.astype(float),
        obs_curve["norm"].values,
    )

    # Null: permute wpe norm table
    null_curves = []
    for _ in range(n_shuffles):
        perm = rng.permutation(n_wpe)
        norms_sh = norms_all[perm]
        sh_norm = [
            float(norms_sh[int(r["token_start_idx"]) : int(r["token_end_idx"]) + 1].mean())
            for _, r in meta_v.iterrows()
        ]
        tmp = pd.DataFrame({"pos": pos_list, "norm": sh_norm})
        null_curves.append(tmp.groupby("pos")["norm"].mean())

    null_df   = pd.concat(null_curves, axis=1)
    null_mean = null_df.mean(axis=1).values
    null_std  = null_df.std(axis=1).values
    null_pos  = null_df.index.values

    fig, ax = plt.subplots(figsize=_SLIDE_FIG)
    ax.fill_between(null_pos, null_mean - null_std, null_mean + null_std,
                    color="#d62728", alpha=0.25,
                    label=f"Shuffled wpe table ±1σ (n={n_shuffles})")
    ax.plot(null_pos, null_mean, color="#d62728", lw=1.5, linestyle="--", alpha=0.8)
    ax.plot(obs_curve["phoneme_position_from_start"], obs_curve["norm"],
            color="#1f77b4", lw=2.5, marker="o", ms=4, label="Observed (ordered wpe)")
    ax.set_xlabel("Phoneme position from word start")
    ax.set_ylabel("Mean projected wpe_norm")
    ax.set_title(f"Position shuffle control  (n={n_shuffles} shuffles)\n"
                 f"Observed slope = {obs_slope:.5f}")
    ax.legend(fontsize=9)
    _savefig(fig, output_dir / "10_position_shuffle_control.png")

    return {
        "n_shuffles": n_shuffles,
        "observed_slope": float(obs_slope),
        "null_slope_mean": float(
            np.mean([stats.linregress(null_df.index.values.astype(float),
                                     null_df.iloc[:, i].values)[0]
                     for i in range(min(50, null_df.shape[1]))])
        ),
    }


# ── Diagnostic 6 — Directional geometry ──────────────────────────────────────

def run_diag6(wpe: torch.Tensor, output_dir: Path, max_pos: int) -> dict:
    wpe_np   = wpe[:max_pos].numpy()
    norms_v  = np.linalg.norm(wpe_np, axis=-1, keepdims=True)
    wpe_unit = wpe_np / (norms_v + 1e-12)
    positions = np.arange(max_pos)

    lags = [l for l in [1, 2, 5, 10, 20, 50, 100, 200, 500] if l < max_pos]
    lag_rows = []
    for lag in lags:
        n = max_pos - lag
        cos_sim  = (wpe_unit[:n] * wpe_unit[lag : lag + n]).sum(axis=-1)
        cos_dist = 1.0 - cos_sim
        lag_rows.append({
            "lag": lag,
            "mean_cos_dist": float(cos_dist.mean()),
            "std_cos_dist":  float(cos_dist.std()),
            "q10": float(np.percentile(cos_dist, 10)),
            "q90": float(np.percentile(cos_dist, 90)),
            "n_pairs": len(cos_dist),
        })
    lag_df = pd.DataFrame(lag_rows)
    lag_df.to_csv(output_dir / "wpe_directional_geometry.csv", index=False)

    # Fig 11
    fig, ax = plt.subplots(figsize=_SLIDE_FIG)
    ax.errorbar(lag_df["lag"], lag_df["mean_cos_dist"], yerr=lag_df["std_cos_dist"],
                marker="o", ms=5, lw=1.8, color="#1f77b4", label="mean ± 1σ")
    ax.fill_between(lag_df["lag"], lag_df["q10"], lag_df["q90"],
                    color="#1f77b4", alpha=0.15, label="10–90th percentile")
    ax.set_xscale("log")
    ax.set_xlabel("Lag (positional distance, log scale)")
    ax.set_ylabel("Cosine distance  (1 − cosine similarity)")
    ax.set_title(f"wpe directional geometry: cosine distance by lag  (positions 0–{max_pos-1})")
    ax.legend(fontsize=9)
    _savefig(fig, output_dir / "11_wpe_cosine_distance_by_lag.png")

    # PCA of raw wpe
    pca = PCA(n_components=2, random_state=0)
    scores = pca.fit_transform(wpe_np)
    ev = pca.explained_variance_ratio_

    # Fig 12
    fig, ax = plt.subplots(figsize=_SQ_FIG)
    sc = ax.scatter(scores[:, 0], scores[:, 1], c=positions,
                    cmap="viridis", s=4, alpha=0.8, rasterized=True)
    plt.colorbar(sc, ax=ax, label="Absolute token position")
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax.set_title(f"PCA of raw wpe vectors  (positions 0–{max_pos-1})")
    _savefig(fig, output_dir / "12_wpe_pca_by_absolute_position.png")

    wpe_norms = norms_v[:, 0]
    r_pc1_pos  = _corr_stats(positions.astype(float), scores[:, 0])
    r_pc1_norm = _corr_stats(wpe_norms, scores[:, 0])

    # Fig 13
    fig, axes = plt.subplots(1, 2, figsize=_SLIDE_FIG)
    axes[0].scatter(positions, scores[:, 0], s=3, alpha=0.6, color="#1f77b4", rasterized=True)
    axes[0].set_xlabel("Absolute token position")
    axes[0].set_ylabel(f"PC1 ({ev[0]*100:.1f}%)")
    axes[0].set_title(f"PC1 vs position  r={r_pc1_pos['pearson']:.3f}")
    axes[1].scatter(wpe_norms, scores[:, 0], s=3, alpha=0.6, color="#d62728", rasterized=True)
    axes[1].set_xlabel("wpe L2 norm")
    axes[1].set_ylabel(f"PC1 ({ev[0]*100:.1f}%)")
    axes[1].set_title(f"PC1 vs norm  r={r_pc1_norm['pearson']:.3f}")
    fig.suptitle(f"wpe PCA geometry  (positions 0–{max_pos-1})", fontsize=11)
    fig.tight_layout()
    _savefig(fig, output_dir / "13_wpe_pca_pc1_vs_position_and_norm.png")

    return {
        "lag_stats": lag_rows,
        "pca_ev": [float(ev[0]), float(ev[1])],
        "r_pc1_vs_position": r_pc1_pos,
        "r_pc1_vs_norm": r_pc1_norm,
    }


# ── Diagnostic 7 — Propagation across layers ──────────────────────────────────

def run_diag7(
    meta_enriched: pd.DataFrame,
    meta_all: pd.DataFrame,
    keep: np.ndarray,
    emb_dir: Path,
    layers: list[str],
    include_block48: bool,
    output_dir: Path,
) -> dict:
    orig_idx   = np.where(keep)[0]
    wpe_norms  = meta_enriched["mean_wpe_norm"].values
    ph_idx     = meta_enriched["phoneme_position_from_start"].values.astype(float)

    all_layers = list(layers)
    if include_block48 and "block_48" not in all_layers:
        all_layers = all_layers + ["block_48"]

    rows = []
    layer_norms: dict[str, np.ndarray] = {}

    for layer in all_layers:
        pt = emb_dir / f"embeddings_{layer}.pt"
        if not pt.exists():
            print(f"  [warning] embeddings_{layer}.pt not found — skipping")
            continue
        emb_full = torch.load(pt, map_location="cpu", weights_only=True).float()
        emb_v    = emb_full[orig_idx]
        lnorms   = emb_v.norm(dim=-1).numpy()
        layer_norms[layer] = lnorms

        c_wpe = _corr_stats(lnorms, wpe_norms)
        c_pos = _corr_stats(lnorms, ph_idx)
        rows.append({
            "layer": layer,
            "is_raw_block48": layer == "block_48",
            "pearson_vs_mean_wpe_norm": c_wpe["pearson"],
            "spearman_vs_mean_wpe_norm": c_wpe["spearman"],
            "pearson_vs_phoneme_idx": c_pos["pearson"],
            "spearman_vs_phoneme_idx": c_pos["spearman"],
            "slope_vs_mean_wpe_norm": c_wpe["slope"],
            "r2_vs_mean_wpe_norm": c_wpe["r2"],
        })

    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(output_dir / "layer_norm_wpe_correlations.csv", index=False)

    if layer_norms:
        # Fig 14 — norm by phoneme position across layers
        colors = plt.cm.tab10(np.linspace(0, 1, len(layer_norms)))
        fig, ax = plt.subplots(figsize=_WIDE_FIG)
        for (layer, lnorms), color in zip(layer_norms.items(), colors):
            by_pos = (
                pd.DataFrame({"pos": ph_idx.astype(int), "norm": lnorms})
                .groupby("pos")["norm"].agg(mean="mean", sem=lambda x: x.std() / np.sqrt(len(x)), n="count")
                .query("n >= 3").reset_index()
            )
            ls = "--" if layer == "block_48" else "-"
            lbl = f"{layer}  (raw, pre-ln_f)" if layer == "block_48" else layer
            ax.errorbar(by_pos["pos"], by_pos["mean"], yerr=by_pos["sem"],
                        color=color, lw=1.8, marker="o", ms=3, linestyle=ls, label=lbl)
        ax.set_xlabel("Phoneme position from word start")
        ax.set_ylabel("Layer norm  (mean ± SEM)")
        ax.set_title("Per-layer phoneme embedding norm by phoneme position")
        ax.legend(fontsize=7, ncol=2)
        _savefig(fig, output_dir / "14_layer_norm_by_phoneme_position_main_layers.png")

        # Fig 15 — correlation bar chart
        fig, ax = plt.subplots(figsize=_SQ_FIG)
        bar_colors = ["#888888" if r else "#1f77b4"
                      for r in corr_df["is_raw_block48"]]
        ax.bar(corr_df["layer"], corr_df["pearson_vs_mean_wpe_norm"],
               color=bar_colors, edgecolor="white", lw=0.5)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xticks(range(len(corr_df)))
        ax.set_xticklabels(corr_df["layer"], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Pearson r  (layer_norm vs mean_wpe_norm)")
        ax.set_title("wpe norm correlation propagation across layers\n"
                     "(grey = raw block_48, pre-ln_f)")
        _savefig(fig, output_dir / "15_layer_norm_wpe_correlation_by_layer.png")

    return {"layer_correlations": rows}


# ── Report ─────────────────────────────────────────────────────────────────────

def write_report(stats_all: dict, output_dir: Path) -> None:
    d1 = stats_all.get("diag1", {})
    d2 = stats_all.get("diag2", {})
    d3 = stats_all.get("diag3", {})
    d7 = stats_all.get("diag7", {})

    slope_trained = d2.get("trained_slope_used_range", float("nan"))
    z_score       = d2.get("trained_slope_z_score", float("nan"))
    r_wpe_emb     = d3.get("corr_embedding_norm_vs_mean_wpe_norm", {}).get("pearson", float("nan"))
    r_pos_emb     = d3.get("corr_embedding_norm_vs_phoneme_index", {}).get("pearson", float("nan"))
    r_pos_wpe     = d3.get("corr_mean_wpe_norm_vs_phoneme_index",  {}).get("pearson", float("nan"))
    frac_neg      = d1.get("fraction_adjacent_deltas_negative", float("nan"))

    def _fmt(v):
        try:
            return f"{float(v):.4f}"
        except Exception:
            return str(v)

    layer_rows = d7.get("layer_correlations", [])
    layer_table = "\n".join(
        f"  {r['layer']:25s}  r_wpe={_fmt(r['pearson_vs_mean_wpe_norm'])}  "
        f"r_pos={_fmt(r['pearson_vs_phoneme_idx'])}"
        for r in layer_rows
    )

    text = f"""# AuriStream wpe Position Diagnostics — Report

## 1. Is the wpe norm decrease architectural / hard-coded by initialization?

- Trained slope (used range 0–{stats_all.get('max_pos', '?')} positions):  {_fmt(slope_trained)}
- Z-score vs {d2.get('n_seeds', '?')} random-init baselines:  {_fmt(z_score)}
- Fraction of adjacent norm deltas < 0 (used range):  {_fmt(frac_neg)}

**Conclusion**: If |z-score| >> 2, the norm decrease is **not** an initialization artifact —
it is learned during training. A flat random baseline with a decreasing trained curve
indicates this is optimization-induced or a consequence of the training setup.

## 2. Does projected wpe explain the phoneme-position norm effect?

Correlations on real phonemes (n={d3.get('n_phonemes', '?')}):
- embedding_norm vs mean_wpe_norm:        Pearson r = {_fmt(r_wpe_emb)}
- embedding_norm vs phoneme_position_idx: Pearson r = {_fmt(r_pos_emb)}
- mean_wpe_norm  vs phoneme_position_idx: Pearson r = {_fmt(r_pos_wpe)}

**Conclusion**: If r(embedding_norm, mean_wpe_norm) is high and r(mean_wpe_norm, phoneme_idx)
is also high, then the embedding-layer norm-by-position curve is largely explained by the
absolute token positions that phonemes occupy — not by an intrinsic phoneme-identity signal.

See figures 07 and 08 for the visual comparison.

## 3. Position shift and shuffle controls

- **Shift control (Fig 09)**: If shifting all token positions by +N changes the projected
  wpe_norm curve strongly, the effect is absolute-position-table driven, not
  an intrinsic property of phoneme boundaries.
- **Shuffle control (Fig 10)**: If permuting the wpe norm table destroys the phoneme-position
  curve, the ordered structure of the learned wpe table matters.

## 4. Directional geometry (wpe beyond magnitude)

- See `wpe_directional_geometry.csv` and Fig 11 for cosine distance by lag.
- If cosine distance increases smoothly with lag, wpe encodes direction-based
  positional geometry (not only magnitude).
- Fig 12–13: PCA of wpe vectors; check whether PC1 is the norm axis or a positional axis.

## 5. Propagation across transformer layers

Layer-by-layer Pearson r (layer_norm vs mean_wpe_norm):

{layer_table if layer_table else "  (no layer data available)"}

**Conclusion**: If r_wpe decreases monotonically with depth, the wpe signal is absorbed /
transformed by the transformer blocks. If it persists in block_48_lnf, interpret with caution —
deeper representations depend on all context, not only positional embeddings.
block_48_lnf is the representation actually passed to coch_head.

## Cautions for Yair

- The `embedding` layer is **not** a pure token representation: h0[p] = wte[token] + wpe[p].
  Norm variation in `embedding` is not a phoneme identity signal.
- Do not interpret the wpe norm decrease as "AuriStream encodes phoneme position".
  The causal chain is: absolute token positions → wpe[p] norms → phoneme-level mean norm.
- block_48 (raw, pre-ln_f) has PC1 dominated by norm (r ≈ 1.000). Use block_48_lnf
  for any geometry-based phoneme analysis.
- These are descriptive, observational results. No formal inferential tests applied.
"""

    (output_dir / "report.md").write_text(text)
    print(f"  Saved report.md")


# ── CLI + main ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", required=True,
                   help="AuriStream extraction run directory")
    p.add_argument("--embeddings-dir", default=None,
                   help="Phoneme embeddings dir (default: {run}/phoneme_embeddings)")
    p.add_argument("--model-id", default=MODEL_ID,
                   help="HuggingFace model ID for AuriStream")
    p.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS,
                   help="Layers to include in Diagnostic 7")
    p.add_argument("--include-raw-block48", action="store_true",
                   help="Include raw block_48 (pre-ln_f) in layer propagation diagnostic")
    p.add_argument("--max-position-to-plot", type=int, default=512, dest="max_pos",
                   help="Maximum absolute position used for zoomed plots")
    p.add_argument("--random-init-seeds", type=int, default=20, dest="n_seeds",
                   help="Number of random-init seeds for Diagnostic 2")
    p.add_argument("--shift-controls", nargs="+", type=int, default=[0, 25, 50, 100, 200],
                   dest="shifts", help="Token position offsets for Diagnostic 4")
    p.add_argument("--n-shuffles", type=int, default=200,
                   help="Number of position-shuffle repetitions for Diagnostic 5")
    p.add_argument("--random-state", type=int, default=0, dest="random_state")
    p.add_argument("--output", default=None,
                   help="Output directory (default: reproduce/figures/audio/"
                        "auristream_wpe_diagnostics/{run_id}/)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir      = (REPO_ROOT / args.run).resolve()
    emb_dir      = Path(args.embeddings_dir).resolve() if args.embeddings_dir \
                   else run_dir / "phoneme_embeddings"

    with open(run_dir / "manifest.json") as f:
        manifest = json.load(f)
    run_id = manifest["run_id"]

    output_dir = Path(args.output).resolve() if args.output \
        else REPO_ROOT / "reproduce" / "figures" / "audio" \
             / "auristream_wpe_diagnostics" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    slide_dir = output_dir / "slide_figures"
    slide_dir.mkdir(exist_ok=True)

    # Check overwrite
    if not args.overwrite and (output_dir / "report.md").exists():
        print(f"[wpe_diag] Output exists: {output_dir}\n"
              "Pass --overwrite to regenerate.")
        return

    # ── Load model (wpe + wte weights only) ───────────────────────────────────
    print(f"[wpe_diag] Loading {args.model_id} …")
    lm = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True)
    lm.eval()
    wpe = lm.transformer.wpe.weight.detach().cpu().float()  # [4096, 1280]
    del lm
    print(f"  wpe shape: {tuple(wpe.shape)}")

    # ── Load metadata + embedding layer ───────────────────────────────────────
    meta_all = pd.read_csv(emb_dir / "metadata_phonemes.csv")
    emb_path = emb_dir / "embeddings_embedding.pt"
    if not emb_path.exists():
        print("[wpe_diag] ERROR: embeddings_embedding.pt not found.")
        return
    emb_embedding = torch.load(emb_path, map_location="cpu", weights_only=True).float()
    assert len(meta_all) == emb_embedding.shape[0]

    # Validity mask (based on embedding layer)
    keep = (
        meta_all["valid_embedding"].astype(bool)
        & ~meta_all["is_silence"].astype(bool)
        & torch.isfinite(emb_embedding).all(dim=1).numpy()
    ).values
    print(f"  {keep.sum()} valid non-silence phonemes / {len(meta_all)} total")

    rng = np.random.default_rng(args.random_state)

    # ── Run diagnostics ────────────────────────────────────────────────────────
    print("[wpe_diag] Diagnostic 1: wpe norm table …")
    d1 = run_diag1(wpe, output_dir, args.max_pos)

    print("[wpe_diag] Diagnostic 2: random init control …")
    d2 = run_diag2(wpe, output_dir, args.n_seeds, args.max_pos)

    print("[wpe_diag] Diagnostic 3: phoneme-level projection …")
    d3, meta_enriched = run_diag3(wpe, meta_all, keep, emb_embedding, output_dir)

    print("[wpe_diag] Diagnostic 4: position shift control …")
    d4 = run_diag4(wpe, meta_enriched, output_dir, args.shifts)

    print("[wpe_diag] Diagnostic 5: shuffled-position control …")
    d5 = run_diag5(wpe, meta_enriched, output_dir, args.n_shuffles, rng)

    print("[wpe_diag] Diagnostic 6: directional geometry …")
    d6 = run_diag6(wpe, output_dir, args.max_pos)

    print("[wpe_diag] Diagnostic 7: propagation across layers …")
    d7 = run_diag7(
        meta_enriched=meta_enriched,
        meta_all=meta_all,
        keep=keep,
        emb_dir=emb_dir,
        layers=args.layers,
        include_block48=args.include_raw_block48,
        output_dir=output_dir,
    )

    # ── Write config + report ─────────────────────────────────────────────────
    stats_all = {
        "run_id": run_id, "max_pos": args.max_pos,
        "diag1": d1, "diag2": d2, "diag3": d3,
        "diag4": d4, "diag5": d5, "diag6": d6, "diag7": d7,
    }
    cfg = {
        "run_id": run_id, "model_id": args.model_id,
        "layers": args.layers, "include_raw_block48": args.include_raw_block48,
        "max_pos": args.max_pos, "n_seeds": args.n_seeds,
        "shifts": args.shifts, "n_shuffles": args.n_shuffles,
        "random_state": args.random_state,
    }
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (output_dir / "all_stats.json").write_text(
        json.dumps(stats_all, indent=2, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    )
    write_report(stats_all, output_dir)

    # ── Copy slide figures ────────────────────────────────────────────────────
    for fname in _SLIDE_KEYS:
        src = output_dir / fname
        if src.exists():
            shutil.copy2(src, slide_dir / fname)
    print(f"  Slide figures copied to {slide_dir}")

    print(f"\n[wpe_diag] Done — {output_dir}")


if __name__ == "__main__":
    main()
