#!/usr/bin/env python3
"""
auristream_state_delta_cosine_diagnostics.py

State/delta C/V cosine similarity diagnostics for AuriStream.

Computes cosine similarity distributions for consonant/vowel (C-C, C-V, V-V) pairs
using AuriStream hidden states and their temporal deltas, as analogues of RNN/LSTM
phoneme-prefix state analyses (h_t, Δh_t = h_t − h_{t−1}).

Analysis units
--------------
    phoneme_last_delta  — PRIMARY: Δh_p = h_{last(p)} − h_{last(p−1)}.
                          Analogue of LSTM Δh_t.
    phoneme_last_h      — SECONDARY: h_{last_token(p)}.
                          Analogue of LSTM h_t after processing phoneme p.
    token_delta         — AuriStream-native control: Δh_t = h_t − h_{t−1}.
    token_h             — AuriStream-native control: h_t for each cochlear token.

Note: AuriStream has no LSTM cell state (c_t). Δh here is an analogue of Δh in
RNN/LSTM analyses, not a true recurrent hidden-state update.

Usage
-----
    python scripts/audio/auristream_state_delta_cosine_diagnostics.py \\
        --run reproduce/data/audio/auristream__6ee9aeb6 \\
        --dataset data/external/paradigm/processed/subset_male.csv \\
        --boundaries data/external/paradigm/processed/phoneme_boundaries_mfa_subset_male.csv \\
        --layers embedding block_01 block_24 block_47 block_48_lnf \\
        --analysis-units phoneme_last_delta phoneme_last_h token_delta token_h \\
        --max-pairs-per-category 50000 \\
        --exclude-intra-item-pairs \\
        --balance-pairs \\
        --overwrite

See docs/auristream_setup.md §13 for scientific context.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Ensure swp package is importable when called from repo root or scripts/audio/
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from swp.audio.phonemes.boundaries import load_boundaries, SILENCE_LABELS
from swp.audio.phonemes.pooling import tokens_for_phoneme

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ARPABET_VOWELS: frozenset[str] = frozenset({
    "AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY", "IH", "IY",
    "OW", "OY", "UH", "UW",
})

_PAIR_COLORS = {"C-C": "steelblue", "C-V": "darkorange", "V-V": "forestgreen"}
_PAIR_TYPES  = ["C-C", "C-V", "V-V"]

# Uppercase-stripped silence bases for robust comparison against stress-stripped phoneme_base.
# SILENCE_LABELS contains mixed-case labels (sp, sil, SIL, …); strip_stress() always returns
# uppercase, so we normalise here to avoid accidentally classifying silence as consonant.
_SILENCE_BASES: frozenset[str] = frozenset(
    re.sub(r"\d+$", "", x).upper() for x in SILENCE_LABELS
)

_DEFAULT_LAYERS        = ["embedding", "block_01", "block_24", "block_47", "block_48_lnf"]
_DEFAULT_UNITS         = ["phoneme_last_delta", "phoneme_last_h", "token_delta", "token_h"]
_PRIMARY_PANEL_LAYERS  = ["block_24", "block_47", "block_48_lnf"]
_DEFAULT_TOKEN_RATE_HZ = 200.0

# ---------------------------------------------------------------------------
# Manifest and activation loading
# ---------------------------------------------------------------------------

def load_manifest(run_dir: Path) -> dict:
    """Load and validate run manifest.json."""
    path = run_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with open(path) as f:
        manifest = json.load(f)
    for key in ("run_params", "items"):
        if key not in manifest:
            raise KeyError(f"Manifest missing required key '{key}': {path}")
    return manifest


def load_layer_activation(
    run_dir: Path,
    manifest: dict,
    item_id: str,
    layer: str,
) -> np.ndarray:
    """Load one item/layer activation. Returns [L, D] float32 numpy array.

    Saved tensors are [D, L]; we transpose here so downstream code works
    with row-per-token convention.
    """
    if item_id not in manifest["items"]:
        raise KeyError(f"Item '{item_id}' not found in manifest.")
    item_paths = manifest["items"][item_id]
    if layer not in item_paths:
        raise KeyError(
            f"Layer '{layer}' not in manifest for item '{item_id}'.\n"
            f"Available layers: {list(item_paths.keys())}"
        )
    pt_path = run_dir / item_paths[layer]
    if not pt_path.exists():
        raise FileNotFoundError(f"Activation file not found: {pt_path}")
    t = torch.load(pt_path, map_location="cpu", weights_only=True)  # [D, L]
    return t.float().numpy().T  # [L, D]

# ---------------------------------------------------------------------------
# Phoneme classification
# ---------------------------------------------------------------------------

def strip_stress(phoneme: str) -> str:
    """Remove trailing ARPAbet stress digit: AH0/AH1/AH2 → AH."""
    return re.sub(r"\d+$", "", phoneme).upper()


def classify_phoneme_type(phoneme_base: str) -> str:
    if phoneme_base in _ARPABET_VOWELS:
        return "vowel"
    if phoneme_base not in _SILENCE_BASES:
        return "consonant"
    return "other"

# ---------------------------------------------------------------------------
# Token-level metadata
# ---------------------------------------------------------------------------

def build_token_metadata_for_item(
    item_id: str,
    boundaries_item_df: pd.DataFrame,
    n_tokens: int,
    token_rate_hz: float,
) -> pd.DataFrame:
    """Build one row per cochlear token for a single item.

    Token-to-phoneme assignment uses center times:
        token_center_s = (token_index + 0.5) / token_rate_hz
    Tokens not covered by any phoneme get phoneme_type='other'.
    """
    ph_arr        = np.array([None] * n_tokens, dtype=object)
    ph_base_arr   = np.array([None] * n_tokens, dtype=object)
    ph_type_arr   = np.full(n_tokens, "other", dtype=object)
    ph_idx_arr    = np.full(n_tokens, -1, dtype=np.int32)
    ps_arr        = np.full(n_tokens, -1, dtype=np.int32)  # position_from_start
    pe_arr        = np.full(n_tokens, -1, dtype=np.int32)  # position_from_end
    n_tok_arr     = np.full(n_tokens, 0, dtype=np.int32)
    within_arr    = np.full(n_tokens, -1, dtype=np.int32)

    sorted_bounds = boundaries_item_df.sort_values("phoneme_index").reset_index(drop=True)
    max_ph_idx    = int(sorted_bounds["phoneme_index"].max()) if len(sorted_bounds) > 0 else 0

    for _, row in sorted_bounds.iterrows():
        ph_raw  = str(row["phoneme"])
        ph_base = strip_stress(ph_raw)
        ph_type = classify_phoneme_type(ph_base)
        ph_idx  = int(row["phoneme_index"])
        start_s = float(row["start_s"])
        end_s   = float(row["end_s"])

        mask = tokens_for_phoneme(start_s, end_s, n_tokens, token_rate_hz, method="center")
        idxs = np.flatnonzero(mask)
        n_tok = len(idxs)

        ph_arr[idxs]      = ph_raw
        ph_base_arr[idxs] = ph_base
        ph_type_arr[idxs] = ph_type
        ph_idx_arr[idxs]  = ph_idx
        ps_arr[idxs]      = ph_idx
        pe_arr[idxs]      = max_ph_idx - ph_idx
        n_tok_arr[idxs]   = n_tok
        within_arr[idxs]  = np.arange(n_tok)

    tok_idx = np.arange(n_tokens)
    time_s  = (tok_idx + 0.5) / token_rate_hz

    rel_pos = np.where(
        n_tok_arr > 1,
        within_arr / np.maximum(n_tok_arr - 1, 1).astype(float),
        np.where(n_tok_arr == 1, 0.5, np.nan),
    )

    return pd.DataFrame({
        "item_id":                      item_id,
        "token_index":                  tok_idx,
        "time_s":                       time_s,
        "phoneme":                      ph_arr,
        "phoneme_base":                 ph_base_arr,
        "phoneme_type":                 ph_type_arr,
        "phoneme_index":                ph_idx_arr,
        "token_index_within_phoneme":   within_arr,
        "n_tokens_in_phoneme":          n_tok_arr,
        "relative_position_in_phoneme": rel_pos,
        "phoneme_position_from_start":  ps_arr,
        "phoneme_position_from_end":    pe_arr,
    })

# ---------------------------------------------------------------------------
# Analysis unit builders
# ---------------------------------------------------------------------------

def build_token_matrix(
    activation: np.ndarray,   # [L, D]
    token_meta: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """token_h: h_t for tokens assigned to vowel/consonant phonemes."""
    mask    = token_meta["phoneme_type"].isin({"vowel", "consonant"}).values
    indices = np.where(mask)[0]
    return activation[indices], token_meta.iloc[indices].reset_index(drop=True)


def build_token_delta_matrix(
    activation: np.ndarray,   # [L, D]
    token_meta: pd.DataFrame,
    mode: str = "all",
) -> tuple[np.ndarray, pd.DataFrame]:
    """token_delta: Δh_t = h_t − h_{t−1}, labelled by token t.

    Keeps only deltas where both t and t−1 are vowel/consonant.
    mode:
        "all"            — all qualifying consecutive pairs
        "boundary"       — only pairs where t and t−1 are in different phonemes
        "within_phoneme" — only pairs where t and t−1 share the same phoneme
    """
    valid    = {"vowel", "consonant"}
    ph_types = token_meta["phoneme_type"].values
    ph_idxs  = token_meta["phoneme_index"].values

    keep = []
    for t in range(1, len(token_meta)):
        if ph_types[t] not in valid or ph_types[t - 1] not in valid:
            continue
        if mode == "boundary" and ph_idxs[t] == ph_idxs[t - 1]:
            continue
        if mode == "within_phoneme" and ph_idxs[t] != ph_idxs[t - 1]:
            continue
        keep.append(t)

    if not keep:
        return np.empty((0, activation.shape[1]), dtype=np.float32), pd.DataFrame()

    keep   = np.array(keep, dtype=np.int64)
    deltas = activation[keep] - activation[keep - 1]
    meta   = token_meta.iloc[keep].reset_index(drop=True).copy()
    meta["delta_mode"] = mode
    return deltas, meta


def build_phoneme_last_matrix(
    activation: np.ndarray,        # [L, D]
    boundaries_item_df: pd.DataFrame,
    n_tokens: int,
    token_rate_hz: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    """phoneme_last_h: hidden state at the last cochlear token of each phoneme.

    This is the AuriStream analogue of an LSTM encoder state h_t computed
    after processing phoneme p_t: h_{last_token(p_t)}.

    Returns only vowel/consonant phonemes. Phonemes with zero assigned tokens
    are skipped with a warning.
    """
    rows_vecs = []
    rows_meta = []

    sorted_bounds = boundaries_item_df.sort_values("phoneme_index").reset_index(drop=True)
    max_ph_idx    = int(sorted_bounds["phoneme_index"].max()) if len(sorted_bounds) > 0 else 0

    for _, row in sorted_bounds.iterrows():
        ph_raw  = str(row["phoneme"])
        ph_base = strip_stress(ph_raw)
        ph_type = classify_phoneme_type(ph_base)

        if ph_type == "other":
            continue

        ph_idx  = int(row["phoneme_index"])
        start_s = float(row["start_s"])
        end_s   = float(row["end_s"])

        mask = tokens_for_phoneme(start_s, end_s, n_tokens, token_rate_hz, method="center")
        idxs = np.flatnonzero(mask)

        if len(idxs) == 0:
            print(
                f"  Warning: phoneme '{ph_raw}' (idx {ph_idx}) has 0 tokens in "
                f"[{start_s:.4f}, {end_s:.4f}) — skipped.",
                file=sys.stderr,
            )
            continue

        last_idx = int(idxs[-1])
        rows_vecs.append(activation[last_idx])
        rows_meta.append({
            "phoneme":                      ph_raw,
            "phoneme_base":                 ph_base,
            "phoneme_type":                 ph_type,
            "phoneme_index":                ph_idx,
            "last_token_index":             last_idx,
            "n_tokens_in_phoneme":          len(idxs),
            "phoneme_position_from_start":  ph_idx,
            "phoneme_position_from_end":    max_ph_idx - ph_idx,
        })

    if not rows_vecs:
        return np.empty((0, activation.shape[1]), dtype=np.float32), pd.DataFrame()

    return np.stack(rows_vecs), pd.DataFrame(rows_meta)


def build_phoneme_last_delta_matrix(
    last_vectors: np.ndarray,   # [P, D]
    phoneme_meta: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """phoneme_last_delta: Δh_p = h_{last(p)} − h_{last(p−1)}, labelled by phoneme p.

    Analogue of LSTM Δh_t = h_t − h_{t−1} between consecutive phoneme states.
    Deltas are only computed between pairs that are consecutive in the filtered
    (non-silence) phoneme sequence within the same item. If a silence phoneme
    separates two speech phonemes, the resulting delta spans across that silence.

    Both p and p−1 must be vowel/consonant (guaranteed by build_phoneme_last_matrix,
    but checked defensively here).
    """
    if len(last_vectors) < 2:
        return np.empty((0, last_vectors.shape[1]), dtype=np.float32), pd.DataFrame()

    valid    = {"vowel", "consonant"}
    ph_types = phoneme_meta["phoneme_type"].values
    keep = [
        p for p in range(1, len(phoneme_meta))
        if ph_types[p] in valid and ph_types[p - 1] in valid
    ]

    if not keep:
        return np.empty((0, last_vectors.shape[1]), dtype=np.float32), pd.DataFrame()

    keep   = np.array(keep, dtype=np.int64)
    deltas = last_vectors[keep] - last_vectors[keep - 1]
    meta   = phoneme_meta.iloc[keep].reset_index(drop=True).copy()
    return deltas, meta

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-8)


def center_and_l2_normalize(vectors: np.ndarray) -> np.ndarray:
    return l2_normalize(vectors - vectors.mean(axis=0))

# ---------------------------------------------------------------------------
# Pair sampling
# ---------------------------------------------------------------------------

def _est_pairs(n_a: int, n_b: int, symmetric: bool) -> int:
    return n_a * (n_a - 1) // 2 if symmetric else n_a * n_b


def sample_pairs_by_category(
    meta_df: pd.DataFrame,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample C-C, C-V, V-V pairs from the full vector set.

    Memory-safe: avoids materialising the full Cartesian product for large sets.
    For n_available ≤ 2M uses full enumeration; otherwise uses oversampling +
    filtering (sampling with replacement; duplicates are possible but rare when
    the pool is large relative to max_pairs_per_category).

    Pairs are sampled once and shared across cosine modes (raw / centered).
    """
    max_pairs     = args.max_pairs_per_category
    excl_intra    = args.exclude_intra_item_pairs
    excl_same_ph  = args.exclude_same_phoneme_pairs
    balance       = args.balance_pairs

    c_idx = np.where(meta_df["phoneme_type"] == "consonant")[0]
    v_idx = np.where(meta_df["phoneme_type"] == "vowel")[0]

    if len(c_idx) == 0:
        raise RuntimeError("No consonant vectors — cannot compute C-C or C-V pairs.")
    if len(v_idx) == 0:
        raise RuntimeError("No vowel vectors — cannot compute V-V or C-V pairs.")

    item_ids    = meta_df["item_id"].values if "item_id" in meta_df.columns else None
    ph_bases    = meta_df["phoneme_base"].values if "phoneme_base" in meta_df.columns else None

    result_parts = []
    n_sampled    = {}

    for pair_type, a_idx, b_idx, symmetric in [
        ("C-C", c_idx, c_idx, True),
        ("C-V", c_idx, v_idx, False),
        ("V-V", v_idx, v_idx, True),
    ]:
        n_a, n_b  = len(a_idx), len(b_idx)
        n_est     = _est_pairs(n_a, n_b, symmetric)
        use_enum  = n_est <= 2_000_000

        if use_enum:
            if symmetric:
                ii, jj = np.triu_indices(n_a, k=1)
                pi, pj = a_idx[ii], a_idx[jj]
            else:
                gi, gj = np.meshgrid(a_idx, b_idx, indexing="ij")
                pi, pj = gi.ravel(), gj.ravel()

            if excl_intra and item_ids is not None:
                keep   = item_ids[pi] != item_ids[pj]
                pi, pj = pi[keep], pj[keep]
            if excl_same_ph and ph_bases is not None:
                keep   = ph_bases[pi] != ph_bases[pj]
                pi, pj = pi[keep], pj[keep]

            n_avail  = len(pi)
            n_sample = min(n_avail, max_pairs)
            if n_sample < n_avail:
                sel    = rng.choice(n_avail, size=n_sample, replace=False)
                pi, pj = pi[sel], pj[sel]

        else:
            # Oversampling: draw pairs, filter, repeat until we have enough.
            # With replace=True, duplicates are possible but rare for large pools.
            n_items   = len(np.unique(item_ids)) if (excl_intra and item_ids is not None) else 1
            same_frac = 1.0 / max(n_items, 1)
            factor    = max(4.0, 1.5 / max(1.0 - same_frac, 0.01))
            batch     = min(int(max_pairs * factor) + 2000, 500_000)

            pi_parts, pj_parts, collected = [], [], 0
            for _ in range(50):  # hard cap on iterations
                if symmetric:
                    ri = rng.integers(0, n_a, size=batch)
                    rj = rng.integers(0, n_a, size=batch)
                    ok = ri != rj
                    ri, rj = ri[ok], rj[ok]
                    # canonical order to avoid (i,j)/(j,i) duplicates
                    swap   = ri > rj
                    ri_new = np.where(swap, rj, ri)
                    rj_new = np.where(swap, ri, rj)
                    ri, rj = ri_new, rj_new
                else:
                    ri = rng.integers(0, n_a, size=batch)
                    rj = rng.integers(0, n_b, size=batch)

                pb = a_idx[ri]
                qb = b_idx[rj]

                if excl_intra and item_ids is not None:
                    ok = item_ids[pb] != item_ids[qb]
                    pb, qb = pb[ok], qb[ok]
                if excl_same_ph and ph_bases is not None:
                    ok = ph_bases[pb] != ph_bases[qb]
                    pb, qb = pb[ok], qb[ok]

                need = max_pairs - collected
                pi_parts.append(pb[:need])
                pj_parts.append(qb[:need])
                collected += min(len(pb), need)
                if collected >= max_pairs:
                    break

            pi      = np.concatenate(pi_parts)[:max_pairs]
            pj      = np.concatenate(pj_parts)[:max_pairs]
            n_avail = n_est   # estimated; exact count not computed
            n_sample = len(pi)

        n_sampled[pair_type] = n_sample

        row_data: dict = {
            "pair_type":                   pair_type,
            "idx_i":                       pi,
            "idx_j":                       pj,
            "n_available_pairs":           n_avail,
            "n_sampled_pairs_pre_balance": n_sample,
            "phoneme_type_i":     meta_df["phoneme_type"].values[pi],
            "phoneme_type_j":     meta_df["phoneme_type"].values[pj],
        }
        for col in ("item_id", "phoneme", "phoneme_base"):
            if col in meta_df.columns:
                row_data[f"{col}_i"] = meta_df[col].values[pi]
                row_data[f"{col}_j"] = meta_df[col].values[pj]
        for col in ("token_index", "phoneme_index"):
            if col in meta_df.columns:
                row_data[f"{col}_i"] = meta_df[col].values[pi]
                row_data[f"{col}_j"] = meta_df[col].values[pj]

        result_parts.append(pd.DataFrame(row_data))

    pairs_df = pd.concat(result_parts, ignore_index=True)

    if balance:
        min_n = min(n_sampled.values())
        parts = [grp.iloc[:min_n] for _, grp in pairs_df.groupby("pair_type", sort=False)]
        pairs_df = pd.concat(parts, ignore_index=True)

    # n_sampled_pairs reflects the final row count per pair_type after any balancing.
    final_counts = pairs_df.groupby("pair_type").size()
    pairs_df["n_sampled_pairs"] = pairs_df["pair_type"].map(final_counts)

    return pairs_df

# ---------------------------------------------------------------------------
# Cosine computation
# ---------------------------------------------------------------------------

def compute_cosines(norm_vecs: np.ndarray, pairs_df: pd.DataFrame) -> np.ndarray:
    return (norm_vecs[pairs_df["idx_i"].values] * norm_vecs[pairs_df["idx_j"].values]).sum(axis=1)

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def _describe(values: np.ndarray) -> dict:
    return {
        "n":      int(len(values)),
        "mean":   float(np.mean(values)),
        "std":    float(np.std(values)),
        "median": float(np.median(values)),
        "q05":    float(np.quantile(values, 0.05)),
        "q25":    float(np.quantile(values, 0.25)),
        "q75":    float(np.quantile(values, 0.75)),
        "q95":    float(np.quantile(values, 0.95)),
    }


def build_cosine_summary(pairs_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mode, pt), grp in pairs_df.groupby(["cosine_mode", "pair_type"]):
        stat = _describe(grp["cosine"].values)
        stat["cosine_mode"]                 = mode
        stat["pair_type"]                   = pt
        stat["n_available_pairs"]           = int(grp["n_available_pairs"].iloc[0])
        stat["n_sampled_pairs_pre_balance"] = int(grp["n_sampled_pairs_pre_balance"].iloc[0])
        stat["n_sampled_pairs"]             = int(grp["n_sampled_pairs"].iloc[0])
        rows.append(stat)
    return pd.DataFrame(rows)


def build_norm_summary(vdf: pd.DataFrame) -> pd.DataFrame:
    """Aggregate norm statistics by phoneme_type and position."""
    rows = []

    for pt, g in vdf.groupby("phoneme_type"):
        s = _describe(g["vector_norm"].values)
        s.update(group_by="phoneme_type", group_value=str(pt))
        rows.append(s)

    for col in ("phoneme_position_from_start", "phoneme_position_from_end"):
        if col not in vdf.columns:
            continue
        sub = vdf[vdf[col] >= 0]
        for pos, g in sub.groupby(col):
            s = _describe(g["vector_norm"].values)
            s.update(group_by=col, group_value=str(int(pos)))
            rows.append(s)
        if len(sub) > 2:
            x = sub[col].values.astype(float)
            y = sub["vector_norm"].values
            r = float(np.corrcoef(x, y)[0, 1])
            rows.append({
                "group_by": f"pearson_norm_vs_{col}", "group_value": "all",
                "n": len(sub), "mean": r,
                "std": np.nan, "median": np.nan,
                "q05": np.nan, "q25": np.nan, "q75": np.nan, "q95": np.nan,
            })

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_distributions(
    pairs_with_cosine: pd.DataFrame,
    output_path: Path,
    title: str,
    density: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for pt in _PAIR_TYPES:
        grp = pairs_with_cosine[pairs_with_cosine["pair_type"] == pt]
        if grp.empty:
            continue
        ax.hist(
            grp["cosine"].values, bins=80, density=density,
            alpha=0.5, color=_PAIR_COLORS[pt],
            label=f"{pt} (n={len(grp):,})",
            histtype="stepfilled",
        )
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density" if density else "Count")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(-1.05, 1.05)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def make_summary_panel(
    panel_data: list[dict],       # each: {layer, cosine_mode, pairs_df}
    panel_layers: list[str],
    analysis_unit: str,
    output_path: Path,
    density: bool = True,
    requested_layers: set[str] | None = None,
    xlim: tuple[float, float] | None = None,
    show_mean_lines: bool = False,
) -> None:
    """Grid panel: rows=layers, cols=cosine_mode (raw, centered).

    requested_layers: set of layer names that were actually requested via CLI.
    Used to distinguish "Not requested" (layer skipped by user) from
    "No valid data" (layer was requested but produced no pairs).

    xlim: x-axis limits (default: (-1.05, 1.05)).
    show_mean_lines: if True, draw a thin vertical dashed line at the per-category mean.
    """
    cosine_modes = ["raw", "centered"]
    n_rows = len(panel_layers)
    n_cols = len(cosine_modes)
    x_lo, x_hi = xlim if xlim is not None else (-1.05, 1.05)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 3.5 * n_rows), squeeze=False
    )
    lookup = {(d["layer"], d["cosine_mode"]): d["pairs_df"] for d in panel_data}

    for r, layer in enumerate(panel_layers):
        for c, mode in enumerate(cosine_modes):
            ax = axes[r][c]
            pdf = lookup.get((layer, mode))
            if pdf is None or pdf.empty:
                if requested_layers is not None and layer not in requested_layers:
                    msg = "Not requested"
                else:
                    msg = "No valid data"
                ax.text(0.5, 0.5, msg, ha="center", va="center",
                        color="gray", fontsize=9, transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                for pt in _PAIR_TYPES:
                    g = pdf[pdf["pair_type"] == pt]
                    if g.empty:
                        continue
                    ax.hist(
                        g["cosine"].values, bins=60, density=density,
                        alpha=0.5, color=_PAIR_COLORS[pt],
                        label=f"{pt} (n={len(g):,})",
                        histtype="stepfilled",
                    )
                    if show_mean_lines:
                        mu = float(g["cosine"].mean())
                        ax.axvline(
                            mu, color=_PAIR_COLORS[pt],
                            linestyle="--", linewidth=0.8, alpha=0.85,
                        )
            ax.set_xlim(x_lo, x_hi)
            ax.set_xlabel("Cosine similarity", fontsize=8)
            ax.set_ylabel("Density" if density else "Count", fontsize=8)
            ax.set_title(f"{layer} | {mode}", fontsize=9)
            if r == 0 and c == 0:
                ax.legend(fontsize=7)

    fig.suptitle(f"State/delta C/V cosine — {analysis_unit}", fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# QC helpers
# ---------------------------------------------------------------------------

def _check_no_nans(vecs: np.ndarray, label: str) -> None:
    if np.any(np.isnan(vecs)):
        raise RuntimeError(
            f"NaN values found in {label} vectors. "
            "Check activation files and boundary alignment."
        )

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "State/delta C/V cosine similarity diagnostics for AuriStream.\n\n"
            "Primary unit   : phoneme_last_delta  (analogue of LSTM Δh_t)\n"
            "Secondary unit : phoneme_last_h      (analogue of LSTM h_t)\n"
            "Native controls: token_delta, token_h\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run", required=True, metavar="DIR",
                   help="AuriStream extraction run directory (contains manifest.json).")
    p.add_argument("--dataset", required=True, metavar="CSV",
                   help="Processed paradigm CSV.")
    p.add_argument("--boundaries", required=True, metavar="CSV",
                   help="Phoneme boundary CSV from MFA alignment.")
    p.add_argument("--layers", nargs="+", default=_DEFAULT_LAYERS,
                   help=f"Layers to analyse. Default: {' '.join(_DEFAULT_LAYERS)}")
    p.add_argument("--analysis-units", nargs="+", default=_DEFAULT_UNITS,
                   dest="analysis_units",
                   help=f"Analysis units. Default: {' '.join(_DEFAULT_UNITS)}")
    p.add_argument("--token-delta-mode", default="all",
                   choices=["all", "boundary", "within_phoneme"],
                   dest="token_delta_mode",
                   help="Token delta mode (default: all).")
    p.add_argument("--max-pairs-per-category", type=int, default=50_000,
                   dest="max_pairs_per_category",
                   help="Max sampled pairs per category (default: 50000).")
    p.add_argument("--random-seed", type=int, default=0, dest="random_seed",
                   help="Random seed for pair sampling (default: 0).")
    p.add_argument("--exclude-intra-item-pairs", default=True,
                   action=argparse.BooleanOptionalAction,
                   dest="exclude_intra_item_pairs",
                   help="Exclude pairs from the same item (default: True).")
    p.add_argument("--exclude-same-phoneme-pairs", default=False,
                   action=argparse.BooleanOptionalAction,
                   dest="exclude_same_phoneme_pairs",
                   help="Exclude pairs sharing the same phoneme base (default: False).")
    p.add_argument("--balance-pairs", default=True,
                   action=argparse.BooleanOptionalAction,
                   dest="balance_pairs",
                   help="Balance pair counts across C-C, C-V, V-V (default: True).")
    p.add_argument("--max-items", type=int, default=None, dest="max_items",
                   help="Stop after N items (smoke test; default: all items).")
    p.add_argument("--output-dir", default=None, dest="output_dir",
                   help=(
                       "Output directory. "
                       "Default: reproduce/figures/audio/auristream_state_delta_cosine/{run_id}/"
                   ))
    p.add_argument("--density", default=True,
                   action=argparse.BooleanOptionalAction,
                   help="Use density on histogram y-axis (default: True).")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing output directory.")
    return p

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()
    rng  = np.random.default_rng(args.random_seed)

    run_dir = Path(args.run)

    # ── Load manifest
    print(f"Loading manifest: {run_dir}")
    manifest   = load_manifest(run_dir)
    run_params = manifest["run_params"]
    token_rate_hz = float(run_params.get("token_rate_hz", _DEFAULT_TOKEN_RATE_HZ))
    run_id        = manifest.get("run_id", run_dir.name)
    print(f"  run_id={run_id}  token_rate_hz={token_rate_hz}")

    # Validate layers
    available_layers = set(run_params.get("layers", []))
    bad = [la for la in args.layers if la not in available_layers]
    if bad:
        raise ValueError(
            f"Layers not in run manifest: {bad}\n"
            f"Available: {sorted(available_layers)}"
        )

    # ── Load boundaries
    # args.dataset is recorded in config.json for provenance but not loaded here;
    # all analysis metadata (item_ids, phoneme intervals) comes from the manifest
    # and boundaries CSV. Merge with the paradigm dataset CSV later if needed for
    # lexicality / length_bin splits.
    print(f"Loading boundaries: {args.boundaries}")
    boundaries_df = load_boundaries(args.boundaries)

    # ── Resolve item list
    run_items  = set(manifest["items"].keys())
    bnd_items  = set(boundaries_df["item_id"].unique())
    items      = sorted(run_items & bnd_items)
    if not items:
        raise RuntimeError(
            "No items common to run manifest and boundary CSV. "
            "Check that item_ids match."
        )
    only_in_run = run_items - bnd_items
    if only_in_run:
        print(f"  Note: {len(only_in_run)} run items have no boundary data — skipped.")

    if args.max_items is not None:
        items = items[:args.max_items]
    print(f"  Processing {len(items)} items across {len(args.layers)} layers.")

    # ── Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = (
            _REPO_ROOT
            / "reproduce" / "figures" / "audio"
            / "auristream_state_delta_cosine" / run_id
        )
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output directory exists: {out_dir}\nUse --overwrite to replace."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Save config
    cfg = vars(args).copy()
    cfg.update(run_id=run_id, token_rate_hz=token_rate_hz, n_items=len(items))
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    # ── Accumulators
    qc_rows: list[dict]                   = []
    panel_by_unit: dict[str, list[dict]]  = {u: [] for u in args.analysis_units}

    # ==========================================================================
    # Main loop: layer × analysis_unit
    # ==========================================================================
    for layer in args.layers:
        print(f"\n{'='*60}\nLayer: {layer}")

        for analysis_unit in args.analysis_units:
            print(f"  Unit: {analysis_unit}")
            tag = f"{layer}_{analysis_unit}"

            all_vecs_list: list[np.ndarray]  = []
            all_meta_list: list[pd.DataFrame] = []

            for item_id in items:
                act = load_layer_activation(run_dir, manifest, item_id, layer)  # [L, D]
                bounds_item = (
                    boundaries_df[boundaries_df["item_id"] == item_id]
                    .sort_values("phoneme_index")
                    .reset_index(drop=True)
                )
                if bounds_item.empty:
                    print(f"    No boundaries for {item_id} — skipped.", file=sys.stderr)
                    continue

                n_tokens = act.shape[0]

                if analysis_unit in ("token_h", "token_delta"):
                    tok_meta = build_token_metadata_for_item(
                        item_id, bounds_item, n_tokens, token_rate_hz
                    )
                    if analysis_unit == "token_h":
                        vecs, meta = build_token_matrix(act, tok_meta)
                    else:
                        vecs, meta = build_token_delta_matrix(
                            act, tok_meta, args.token_delta_mode
                        )

                else:  # phoneme_last_h or phoneme_last_delta
                    lv, pm = build_phoneme_last_matrix(
                        act, bounds_item, n_tokens, token_rate_hz
                    )
                    pm = pm.copy()
                    pm["item_id"] = item_id

                    if analysis_unit == "phoneme_last_h":
                        vecs, meta = lv, pm
                    else:
                        vecs, meta = build_phoneme_last_delta_matrix(lv, pm)
                        if len(vecs) > 0:
                            meta = meta.copy()
                            meta["item_id"] = item_id

                if len(vecs) == 0:
                    continue

                _check_no_nans(vecs, f"{item_id}/{layer}/{analysis_unit}")

                if "item_id" not in meta.columns:
                    meta = meta.copy()
                    meta["item_id"] = item_id

                all_vecs_list.append(vecs)
                all_meta_list.append(meta)

            if not all_vecs_list:
                print(f"    No vectors for {tag} — skipped.", file=sys.stderr)
                continue

            all_vecs = np.concatenate(all_vecs_list, axis=0).astype(np.float64)
            all_meta = pd.concat(all_meta_list, ignore_index=True)
            all_meta["layer"]         = layer
            all_meta["analysis_unit"] = analysis_unit
            all_meta["vector_norm"]   = np.linalg.norm(all_vecs, axis=1)

            n_total = len(all_vecs)
            n_con   = int((all_meta["phoneme_type"] == "consonant").sum())
            n_vow   = int((all_meta["phoneme_type"] == "vowel").sum())
            print(f"    Vectors: {n_total} total | {n_con} consonant | {n_vow} vowel")

            # ── Vector metadata CSV
            all_meta.to_csv(out_dir / f"vector_metadata_{tag}.csv", index=False)

            # ── Norm summary CSV
            build_norm_summary(all_meta).to_csv(
                out_dir / f"norm_summary_{tag}.csv", index=False
            )

            # ── Sample pairs once (shared for raw and centered)
            try:
                pairs_df = sample_pairs_by_category(all_meta, args, rng)
            except RuntimeError as exc:
                print(f"    Skipping {tag}: {exc}", file=sys.stderr)
                continue

            # ── Compute cosines for both modes on the same pairs
            raw_vecs  = l2_normalize(all_vecs)
            cent_vecs = center_and_l2_normalize(all_vecs)

            pairs_raw          = pairs_df.copy()
            pairs_raw["cosine"]      = compute_cosines(raw_vecs, pairs_df)
            pairs_raw["cosine_mode"] = "raw"

            pairs_cen          = pairs_df.copy()
            pairs_cen["cosine"]      = compute_cosines(cent_vecs, pairs_df)
            pairs_cen["cosine_mode"] = "centered"

            pairs_all = pd.concat([pairs_raw, pairs_cen], ignore_index=True)
            pairs_all["layer"]         = layer
            pairs_all["analysis_unit"] = analysis_unit

            # ── Save pair and summary CSVs
            pairs_all.to_csv(out_dir / f"cosine_pairs_{tag}.csv", index=False)
            build_cosine_summary(pairs_all).to_csv(
                out_dir / f"cosine_summary_{tag}.csv", index=False
            )

            # ── Per-mode plots
            for mode, pmode in (("raw", pairs_raw), ("centered", pairs_cen)):
                plot_distributions(
                    pmode,
                    out_dir / f"cosine_distribution_{tag}_{mode}.png",
                    title=f"{layer} | {analysis_unit} | {mode}",
                    density=args.density,
                )

            # ── QC rows
            for mode, pmode in (("raw", pairs_raw), ("centered", pairs_cen)):
                for pt, grp in pmode.groupby("pair_type"):
                    cosines = grp["cosine"].values
                    qc_rows.append({
                        "layer":            layer,
                        "analysis_unit":    analysis_unit,
                        "cosine_mode":      mode,
                        "pair_type":        pt,
                        "n_vectors_total":  n_total,
                        "n_consonants":     n_con,
                        "n_vowels":         n_vow,
                        "n_pairs":          len(grp),
                        "n_nan_cosines":    int(np.isnan(cosines).sum()),
                        "cosine_mean":      float(np.nanmean(cosines)),
                        "cosine_std":       float(np.nanstd(cosines)),
                    })

            # ── Accumulate for summary panels (primary layers only)
            if layer in _PRIMARY_PANEL_LAYERS:
                for mode, pmode in (("raw", pairs_raw), ("centered", pairs_cen)):
                    panel_by_unit[analysis_unit].append({
                        "layer":       layer,
                        "cosine_mode": mode,
                        "pairs_df":    pmode,
                    })

    # ==========================================================================
    # Summary panels
    # ==========================================================================
    requested_layers = set(args.layers)
    _write_panel(
        panel_by_unit, "phoneme_last_delta",
        _PRIMARY_PANEL_LAYERS, out_dir, args.density,
        primary=True, requested_layers=requested_layers,
    )
    for unit in ("phoneme_last_h", "token_delta"):
        if unit in args.analysis_units:
            _write_panel(
                panel_by_unit, unit,
                _PRIMARY_PANEL_LAYERS, out_dir, args.density,
                primary=False, requested_layers=requested_layers,
            )

    # Zoomed panels (x ∈ [-0.5, 0.5] with per-category mean lines)
    _write_panel(
        panel_by_unit, "phoneme_last_delta",
        _PRIMARY_PANEL_LAYERS, out_dir, args.density,
        primary=True, requested_layers=requested_layers,
        xlim=(-0.5, 0.5), show_mean_lines=True, name_suffix="_zoom",
    )
    if "token_delta" in args.analysis_units:
        _write_panel(
            panel_by_unit, "token_delta",
            _PRIMARY_PANEL_LAYERS, out_dir, args.density,
            primary=False, requested_layers=requested_layers,
            xlim=(-0.5, 0.5), show_mean_lines=True, name_suffix="_zoom",
        )

    # ==========================================================================
    # QC summary
    # ==========================================================================
    qc_df = pd.DataFrame(qc_rows)
    qc_df.to_csv(out_dir / "qc_summary.csv", index=False)
    n_nan = int(qc_df["n_nan_cosines"].sum()) if not qc_df.empty else 0
    if n_nan > 0:
        print(f"\nWARNING: {n_nan} NaN cosines found — check qc_summary.csv.", file=sys.stderr)

    print(f"\nDone. Outputs: {out_dir}")


def _write_panel(
    panel_by_unit: dict,
    analysis_unit: str,
    panel_layers: list[str],
    out_dir: Path,
    density: bool,
    primary: bool,
    requested_layers: set[str] | None = None,
    xlim: tuple[float, float] | None = None,
    show_mean_lines: bool = False,
    name_suffix: str = "",
) -> None:
    data = panel_by_unit.get(analysis_unit, [])
    if not data:
        return
    suffix = "_primary" if primary else ""
    path   = out_dir / f"summary_panel_{analysis_unit}{suffix}{name_suffix}.png"
    make_summary_panel(
        data, panel_layers, analysis_unit, path,
        density=density, requested_layers=requested_layers,
        xlim=xlim, show_mean_lines=show_mean_lines,
    )
    print(f"  Panel: {path.name}")


if __name__ == "__main__":
    main()
