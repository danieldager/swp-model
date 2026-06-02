#!/usr/bin/env python3
"""
auristream_phoneme_identity_position_diagnostics.py

Identity × position cosine similarity diagnostics for AuriStream.

Tests whether AuriStream shows onion-like signatures:
  - identity-related structure in direction (cosine similarity of Δh_p vectors)
  - position-related structure in magnitude (||Δh_p||)

Primary analysis unit: phoneme_last_delta — Δh_p = h_{last(p)} − h_{last(p−1)},
where h_{last(p)} is the hidden state at the last cochlear token of phoneme p.
This is the closest AuriStream analogue of an RNN/LSTM prefix-state delta.

Important caveat: AuriStream is a causal Transformer, not an RNN. This diagnostic
tests for onion-like *signatures* (descriptive patterns), not evidence of an onion
mechanism. Δh_p is a structural analogue, not a true recurrent hidden-state update.

Four pair categories
--------------------
  same_phoneme_same_position        same phoneme base AND same position from word start
  same_phoneme_different_position   same phoneme base, different position
  different_phoneme_same_position   different phoneme base, same position
  different_phoneme_different_position

Cosine modes
------------
  centered (primary): subtract layer mean, L2-normalize, compute cosine
  raw:                L2-normalize raw vectors, compute cosine
Same pairs are reused for both modes.

Usage
-----
    python scripts/audio/auristream_phoneme_identity_position_diagnostics.py \\
        --run reproduce/data/audio/auristream__6ee9aeb6 \\
        --dataset data/external/paradigm/processed/subset_male.csv \\
        --boundaries data/external/paradigm/processed/phoneme_boundaries_mfa_subset_male.csv \\
        --layers block_24 block_47 block_48_lnf \\
        --max-pairs-per-category 50000 \\
        --exclude-intra-item-pairs \\
        --balance-pairs \\
        --overwrite

See docs/auristream_setup.md §14 for scientific context.
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

_SILENCE_BASES: frozenset[str] = frozenset(
    re.sub(r"\d+$", "", x).upper() for x in SILENCE_LABELS
)

_ANALYSIS_UNIT = "phoneme_last_delta"

_PAIR_CATEGORIES = [
    "same_phoneme_same_position",
    "same_phoneme_different_position",
    "different_phoneme_same_position",
    "different_phoneme_different_position",
]

_CAT_COLORS = {
    "same_phoneme_same_position":           "royalblue",
    "same_phoneme_different_position":      "cornflowerblue",
    "different_phoneme_same_position":      "darkorange",
    "different_phoneme_different_position": "lightcoral",
}

_CAT_LABELS = {
    "same_phoneme_same_position":           "same-ph, same-pos",
    "same_phoneme_different_position":      "same-ph, diff-pos",
    "different_phoneme_same_position":      "diff-ph, same-pos",
    "different_phoneme_different_position": "diff-ph, diff-pos",
}

_PHONEME_TYPE_COLORS = {
    "consonant": "steelblue",
    "vowel":     "darkorange",
}

_DEFAULT_LAYERS       = ["block_24", "block_47", "block_48_lnf"]
_PRIMARY_PANEL_LAYERS = ["block_24", "block_47", "block_48_lnf"]
_DEFAULT_TOKEN_RATE_HZ = 200.0

_DATASET_COLS = [
    "speaker", "lexicality", "word_length_phonemes", "word_length_tokens",
    "length_bin", "frequency_bin", "morphology",
]

# ---------------------------------------------------------------------------
# Manifest and activation loading
# ---------------------------------------------------------------------------

def load_manifest(run_dir: Path) -> dict:
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

    Saved tensors are [D, L]; transposed here so downstream code is row-per-token.
    """
    if item_id not in manifest["items"]:
        raise KeyError(f"Item '{item_id}' not found in manifest.")
    item_paths = manifest["items"][item_id]
    if layer not in item_paths:
        raise KeyError(f"Layer '{layer}' not in manifest for item '{item_id}'.")
    pt_path = run_dir / item_paths[layer]
    if not pt_path.exists():
        raise FileNotFoundError(f"Activation file not found: {pt_path}")
    t = torch.load(pt_path, map_location="cpu", weights_only=True)  # [D, L]
    return t.float().numpy().T  # [L, D]

# ---------------------------------------------------------------------------
# Phoneme classification
# ---------------------------------------------------------------------------

def strip_stress(phoneme: str) -> str:
    return re.sub(r"\d+$", "", phoneme).upper()


def classify_phoneme_type(phoneme_base: str) -> str:
    if phoneme_base in _ARPABET_VOWELS:
        return "vowel"
    if phoneme_base not in _SILENCE_BASES:
        return "consonant"
    return "other"

# ---------------------------------------------------------------------------
# Core builder: phoneme_last_h and phoneme_last_delta with rich metadata
# ---------------------------------------------------------------------------

def build_phoneme_last_and_delta(
    activation: np.ndarray,           # [L, D]
    boundaries_item_df: pd.DataFrame,
    n_tokens: int,
    token_rate_hz: float,
    item_id: str,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Build phoneme_last_h and phoneme_last_delta with rich metadata.

    Returns
    -------
    h_vecs      : [P, D]     hidden state at last cochlear token of each speech phoneme
    h_meta      : DataFrame  metadata for h_vecs (P rows)
    delta_vecs  : [K, D]     Δh_p = h_p − h_{p−1} for consecutive speech phonemes
    delta_meta  : DataFrame  metadata for delta_vecs (K rows), including prev_* columns
    """
    rows_vecs: list[np.ndarray] = []
    rows_meta: list[dict]       = []

    sorted_bounds = boundaries_item_df.sort_values("phoneme_index").reset_index(drop=True)
    _empty = np.empty((0, activation.shape[1]), dtype=np.float32)

    if sorted_bounds.empty:
        return _empty, pd.DataFrame(), _empty.copy(), pd.DataFrame()

    max_ph_idx = int(sorted_bounds["phoneme_index"].max())

    for _, row in sorted_bounds.iterrows():
        ph_raw  = str(row["phoneme"])
        ph_base = strip_stress(ph_raw)
        ph_type = classify_phoneme_type(ph_base)

        if ph_type == "other":
            continue

        ph_idx  = int(row["phoneme_index"])
        start_s = float(row["start_s"])
        end_s   = float(row["end_s"])
        dur_s   = float(row.get("duration_s", end_s - start_s))

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
            "item_id":                     item_id,
            "phoneme":                     ph_raw,
            "phoneme_base":                ph_base,
            "phoneme_type":                ph_type,
            "phoneme_index":               ph_idx,
            "phoneme_position_from_start": ph_idx,
            "phoneme_position_from_end":   max_ph_idx - ph_idx,
            "last_token_index":            last_idx,
            "n_tokens_in_phoneme":         len(idxs),
            "duration_s":                  dur_s,
            "absolute_token_position":     last_idx,
            "relative_phoneme_position":   ph_idx / max(max_ph_idx, 1),
            "relative_token_position":     last_idx / max(n_tokens - 1, 1),
        })

    if not rows_vecs:
        return _empty, pd.DataFrame(), _empty.copy(), pd.DataFrame()

    h_vecs = np.stack(rows_vecs)   # [P, D]
    h_meta = pd.DataFrame(rows_meta)

    P = len(h_vecs)
    if P < 2:
        return h_vecs, h_meta, _empty, pd.DataFrame()

    # All h phonemes are already vowel/consonant; keep check defensive.
    valid    = {"vowel", "consonant"}
    ph_types = h_meta["phoneme_type"].values
    keep     = [p for p in range(1, P) if ph_types[p] in valid and ph_types[p - 1] in valid]

    if not keep:
        return h_vecs, h_meta, _empty, pd.DataFrame()

    keep_arr   = np.array(keep, dtype=np.int64)
    delta_vecs = h_vecs[keep_arr] - h_vecs[keep_arr - 1]   # [K, D]

    cur  = h_meta.iloc[keep_arr].reset_index(drop=True)
    prev = h_meta.iloc[keep_arr - 1].reset_index(drop=True)

    h_norms      = np.linalg.norm(h_vecs[keep_arr], axis=1)
    prev_h_norms = np.linalg.norm(h_vecs[keep_arr - 1], axis=1)
    delta_norms  = np.linalg.norm(delta_vecs, axis=1)

    delta_meta = pd.DataFrame({
        "item_id":                     cur["item_id"].values,
        "phoneme":                     cur["phoneme"].values,
        "phoneme_base":                cur["phoneme_base"].values,
        "phoneme_type":                cur["phoneme_type"].values,
        "phoneme_index":               cur["phoneme_index"].values,
        "phoneme_position_from_start": cur["phoneme_position_from_start"].values,
        "phoneme_position_from_end":   cur["phoneme_position_from_end"].values,
        "last_token_index":            cur["last_token_index"].values,
        "n_tokens_in_phoneme":         cur["n_tokens_in_phoneme"].values,
        "duration_s":                  cur["duration_s"].values,
        "absolute_token_position":     cur["absolute_token_position"].values,
        "relative_phoneme_position":   cur["relative_phoneme_position"].values,
        "relative_token_position":     cur["relative_token_position"].values,
        "prev_phoneme":                prev["phoneme"].values,
        "prev_phoneme_base":           prev["phoneme_base"].values,
        "prev_phoneme_type":           prev["phoneme_type"].values,
        "prev_phoneme_index":          prev["phoneme_index"].values,
        "prev_last_token_index":       prev["last_token_index"].values,
        "prev_n_tokens_in_phoneme":    prev["n_tokens_in_phoneme"].values,
        "delta_norm":                  delta_norms,
        "h_norm":                      h_norms,
        "prev_h_norm":                 prev_h_norms,
    })

    return h_vecs, h_meta, delta_vecs, delta_meta

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

def sample_pairs_identity_position(
    meta_df: pd.DataFrame,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample identity × position pairs using full upper-triangle enumeration.

    For phoneme_last_delta the vector pool is small (typically a few hundred to
    low thousands), so full enumeration is always feasible.

    Intra-item pairs are excluded by default. Categories with 0 available pairs
    are reported to stderr and omitted from the result rather than failing.
    """
    N        = len(meta_df)
    ph_bases = meta_df["phoneme_base"].values
    positions= meta_df["phoneme_position_from_start"].values
    item_ids = meta_df["item_id"].values
    dn       = meta_df["delta_norm"].values

    ii, jj = np.triu_indices(N, k=1)

    if args.exclude_intra_item_pairs:
        keep   = item_ids[ii] != item_ids[jj]
        ii, jj = ii[keep], jj[keep]

    if len(ii) == 0:
        raise RuntimeError("No cross-item pairs available after intra-item exclusion.")

    same_ph  = ph_bases[ii]  == ph_bases[jj]
    same_pos = positions[ii] == positions[jj]

    cat_arr = np.where(
        same_ph & same_pos,
        "same_phoneme_same_position",
        np.where(
            same_ph & ~same_pos,
            "same_phoneme_different_position",
            np.where(
                ~same_ph & same_pos,
                "different_phoneme_same_position",
                "different_phoneme_different_position",
            ),
        ),
    )

    result_parts: list[pd.DataFrame] = []
    n_sampled: dict[str, int]        = {}

    for cat in _PAIR_CATEGORIES:
        cat_mask = cat_arr == cat
        c_ii     = ii[cat_mask]
        c_jj     = jj[cat_mask]
        n_avail  = int(len(c_ii))
        n_sample = min(n_avail, args.max_pairs_per_category)

        if n_sample == 0:
            print(
                f"    WARNING: category '{cat}' has 0 available pairs — omitted.",
                file=sys.stderr,
            )
            n_sampled[cat] = 0
            continue

        if n_sample < n_avail:
            sel      = rng.choice(n_avail, size=n_sample, replace=False)
            c_ii, c_jj = c_ii[sel], c_jj[sel]

        n_sampled[cat] = n_sample

        result_parts.append(pd.DataFrame({
            "pair_category":                 cat,
            "idx_i":                         c_ii,
            "idx_j":                         c_jj,
            "item_id_i":                     item_ids[c_ii],
            "item_id_j":                     item_ids[c_jj],
            "phoneme_i":                     meta_df["phoneme"].values[c_ii],
            "phoneme_j":                     meta_df["phoneme"].values[c_jj],
            "phoneme_base_i":                ph_bases[c_ii],
            "phoneme_base_j":                ph_bases[c_jj],
            "phoneme_type_i":                meta_df["phoneme_type"].values[c_ii],
            "phoneme_type_j":                meta_df["phoneme_type"].values[c_jj],
            "phoneme_position_from_start_i": positions[c_ii],
            "phoneme_position_from_start_j": positions[c_jj],
            "phoneme_position_from_end_i":   meta_df["phoneme_position_from_end"].values[c_ii],
            "phoneme_position_from_end_j":   meta_df["phoneme_position_from_end"].values[c_jj],
            "delta_norm_i":                  dn[c_ii],
            "delta_norm_j":                  dn[c_jj],
            "n_available_pairs":             n_avail,
            "n_sampled_pairs_pre_balance":   n_sample,
        }))

    if not result_parts:
        raise RuntimeError("All pair categories are empty after filtering — cannot proceed.")

    pairs_df = pd.concat(result_parts, ignore_index=True)

    if args.balance_pairs:
        pos_counts = {k: v for k, v in n_sampled.items() if v > 0}
        if pos_counts:
            min_n  = min(pos_counts.values())
            parts  = [
                grp.iloc[:min_n]
                for _, grp in pairs_df.groupby("pair_category", sort=False)
            ]
            pairs_df = pd.concat(parts, ignore_index=True)

    final_counts = pairs_df.groupby("pair_category").size()
    pairs_df["n_sampled_pairs"] = pairs_df["pair_category"].map(final_counts)

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
    for (mode, cat), grp in pairs_df.groupby(["cosine_mode", "pair_category"]):
        stat = _describe(grp["cosine"].values)
        stat["cosine_mode"]                 = mode
        stat["pair_category"]               = cat
        stat["n_available_pairs"]           = int(grp["n_available_pairs"].iloc[0])
        stat["n_sampled_pairs_pre_balance"] = int(grp["n_sampled_pairs_pre_balance"].iloc[0])
        stat["n_sampled_pairs"]             = int(grp["n_sampled_pairs"].iloc[0])
        rows.append(stat)
    return pd.DataFrame(rows)


def build_magnitude_summary(meta_df: pd.DataFrame) -> pd.DataFrame:
    """delta_norm statistics by phoneme_type and phoneme_position_from_start."""
    rows: list[dict] = []

    s = _describe(meta_df["delta_norm"].values)
    s.update(group_by="overall", group_value="all")
    rows.append(s)

    for pt, g in meta_df.groupby("phoneme_type"):
        s = _describe(g["delta_norm"].values)
        s.update(group_by="phoneme_type", group_value=str(pt))
        rows.append(s)

    for pos, g in meta_df.groupby("phoneme_position_from_start"):
        s = _describe(g["delta_norm"].values)
        s.update(group_by="phoneme_position_from_start", group_value=str(int(pos)))
        rows.append(s)

    if len(meta_df) > 2:
        x = meta_df["phoneme_position_from_start"].values.astype(float)
        y = meta_df["delta_norm"].values
        r = float(np.corrcoef(x, y)[0, 1])
        rows.append({
            "group_by": "pearson_delta_norm_vs_position", "group_value": "all",
            "n": len(meta_df), "mean": r,
            "std": np.nan, "median": np.nan,
            "q05": np.nan, "q25": np.nan, "q75": np.nan, "q95": np.nan,
        })

    for pt, g in meta_df.groupby("phoneme_type"):
        if len(g) > 2:
            x = g["phoneme_position_from_start"].values.astype(float)
            y = g["delta_norm"].values
            r = float(np.corrcoef(x, y)[0, 1])
            rows.append({
                "group_by": f"pearson_delta_norm_vs_position_{pt}", "group_value": pt,
                "n": len(g), "mean": r,
                "std": np.nan, "median": np.nan,
                "q05": np.nan, "q25": np.nan, "q75": np.nan, "q95": np.nan,
            })

    return pd.DataFrame(rows)


def build_pair_composition(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Breakdown of pair counts per category by phoneme_base, position, and phoneme_type."""
    rows: list[dict] = []
    for cat, grp in pairs_df.groupby("pair_category"):
        for pb, g2 in grp.groupby("phoneme_base_i"):
            rows.append({"pair_category": cat, "group_by": "phoneme_base_i",
                         "group_value": str(pb), "n": len(g2)})
        if "phoneme_position_from_start_i" in grp.columns:
            for pos, g2 in grp.groupby("phoneme_position_from_start_i"):
                rows.append({"pair_category": cat, "group_by": "phoneme_position_from_start_i",
                             "group_value": str(int(pos)), "n": len(g2)})
        for pt, g2 in grp.groupby("phoneme_type_i"):
            rows.append({"pair_category": cat, "group_by": "phoneme_type_i",
                         "group_value": str(pt), "n": len(g2)})
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_identity_position_panel(
    panel_data: list[dict],
    panel_layers: list[str],
    cosine_mode: str,
    output_path: Path,
    density: bool = True,
    xlim: tuple[float, float] | None = None,
    show_mean_lines: bool = False,
    requested_layers: set[str] | None = None,
) -> None:
    """Single-column panel: one row per layer, four overlaid category distributions."""
    n_rows    = len(panel_layers)
    x_lo, x_hi = xlim if xlim is not None else (-1.05, 1.05)
    lookup = {d["layer"]: d["pairs_df"] for d in panel_data if d["cosine_mode"] == cosine_mode}

    fig, axes = plt.subplots(n_rows, 1, figsize=(7, 3.5 * n_rows), squeeze=False)

    for r, layer in enumerate(panel_layers):
        ax  = axes[r][0]
        pdf = lookup.get(layer)

        if pdf is None or pdf.empty:
            msg = (
                "Not requested"
                if (requested_layers is not None and layer not in requested_layers)
                else "No valid data"
            )
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    color="gray", fontsize=9, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            for cat in _PAIR_CATEGORIES:
                g = pdf[pdf["pair_category"] == cat]
                if g.empty:
                    continue
                ax.hist(
                    g["cosine"].values, bins=60, density=density,
                    alpha=0.5, color=_CAT_COLORS[cat],
                    label=f"{_CAT_LABELS[cat]} (n={len(g):,})",
                    histtype="stepfilled",
                )
                if show_mean_lines:
                    mu = float(g["cosine"].mean())
                    ax.axvline(mu, color=_CAT_COLORS[cat],
                               linestyle="--", linewidth=0.8, alpha=0.85)
            ax.legend(fontsize=6, loc="upper right")

        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel("Cosine similarity", fontsize=8)
        ax.set_ylabel("Density" if density else "Count", fontsize=8)
        ax.set_title(f"{layer} | {cosine_mode}", fontsize=9)

    fig.suptitle(
        f"Identity × position cosine — phoneme_last_delta | {cosine_mode}",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_magnitude_by_position(
    magnitude_data: list[dict],
    panel_layers: list[str],
    output_path: Path,
    requested_layers: set[str] | None = None,
) -> None:
    """||Δh_p|| vs phoneme_position_from_start, mean ± std per position, colored by phoneme_type."""
    n_rows = len(panel_layers)
    lookup = {d["layer"]: d["meta_df"] for d in magnitude_data}

    fig, axes = plt.subplots(n_rows, 1, figsize=(7, 3.5 * n_rows), squeeze=False)

    for r, layer in enumerate(panel_layers):
        ax  = axes[r][0]
        mdf = lookup.get(layer)

        if mdf is None or mdf.empty:
            msg = (
                "Not requested"
                if (requested_layers is not None and layer not in requested_layers)
                else "No valid data"
            )
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    color="gray", fontsize=9, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            for pt in ("consonant", "vowel"):
                sub = mdf[mdf["phoneme_type"] == pt]
                if sub.empty:
                    continue
                grouped = sub.groupby("phoneme_position_from_start")["delta_norm"]
                means   = grouped.mean()
                stds    = grouped.std().fillna(0.0)
                ax.errorbar(
                    means.index, means.values, yerr=stds.values,
                    fmt="-o", markersize=4, linewidth=1.2,
                    color=_PHONEME_TYPE_COLORS[pt],
                    capsize=3, elinewidth=0.8,
                    label=pt,
                )
            ax.legend(fontsize=7)

        ax.set_xlabel("Phoneme position from word start", fontsize=8)
        ax.set_ylabel("||Δh|| (L2 norm)", fontsize=8)
        ax.set_title(f"{layer} — Δ magnitude by position", fontsize=9)

    fig.suptitle("Delta magnitude by phoneme position — phoneme_last_delta", fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Identity × position cosine similarity diagnostics for AuriStream.\n\n"
            "Primary unit   : phoneme_last_delta  (analogue of LSTM Δh_t)\n"
            "Pair categories: same/different phoneme × same/different position\n"
            "Primary mode   : centered cosine\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run", required=True, metavar="DIR",
                   help="AuriStream extraction run directory (contains manifest.json).")
    p.add_argument("--dataset", required=True, metavar="CSV",
                   help="Processed paradigm CSV (used for optional metadata merge).")
    p.add_argument("--boundaries", required=True, metavar="CSV",
                   help="Phoneme boundary CSV from MFA alignment.")
    p.add_argument("--layers", nargs="+", default=_DEFAULT_LAYERS,
                   help=f"Layers to analyse. Default: {' '.join(_DEFAULT_LAYERS)}")
    p.add_argument("--max-pairs-per-category", type=int, default=50_000,
                   dest="max_pairs_per_category",
                   help="Max sampled pairs per category (default: 50000).")
    p.add_argument("--random-seed", type=int, default=0, dest="random_seed",
                   help="Random seed for pair sampling (default: 0).")
    p.add_argument("--exclude-intra-item-pairs", default=True,
                   action=argparse.BooleanOptionalAction,
                   dest="exclude_intra_item_pairs",
                   help="Exclude pairs from the same item (default: True).")
    p.add_argument("--balance-pairs", default=True,
                   action=argparse.BooleanOptionalAction,
                   dest="balance_pairs",
                   help="Balance all four categories to the smallest count (default: True).")
    p.add_argument("--max-items", type=int, default=None, dest="max_items",
                   help="Stop after N items (smoke test; default: all items).")
    p.add_argument("--output-dir", default=None, dest="output_dir",
                   help=(
                       "Output directory. Default: "
                       "reproduce/figures/audio/auristream_identity_position/{run_id}/"
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

    print(f"Loading manifest: {run_dir}")
    manifest      = load_manifest(run_dir)
    run_params    = manifest["run_params"]
    token_rate_hz = float(run_params.get("token_rate_hz", _DEFAULT_TOKEN_RATE_HZ))
    run_id        = manifest.get("run_id", run_dir.name)
    print(f"  run_id={run_id}  token_rate_hz={token_rate_hz}")

    available_layers = set(run_params.get("layers", []))
    bad = [la for la in args.layers if la not in available_layers]
    if bad:
        raise ValueError(
            f"Layers not in run manifest: {bad}\n"
            f"Available: {sorted(available_layers)}"
        )

    # Optional dataset metadata for item-level columns
    dataset_meta: pd.DataFrame | None = None
    dataset_path = Path(args.dataset)
    if dataset_path.exists():
        try:
            _ds = pd.read_csv(dataset_path)
            avail_cols = [c for c in _DATASET_COLS if c in _ds.columns]
            if "item_id" in _ds.columns and avail_cols:
                dataset_meta = _ds[["item_id"] + avail_cols].drop_duplicates("item_id")
                print(f"  Dataset metadata: merging columns {avail_cols}")
        except Exception as exc:
            print(f"  Warning: could not load dataset CSV ({exc}).", file=sys.stderr)
    else:
        print(f"  Warning: dataset CSV not found: {dataset_path}", file=sys.stderr)

    print(f"Loading boundaries: {args.boundaries}")
    boundaries_df = load_boundaries(args.boundaries)

    run_items = set(manifest["items"].keys())
    bnd_items = set(boundaries_df["item_id"].unique())
    items     = sorted(run_items & bnd_items)
    if not items:
        raise RuntimeError("No items common to run manifest and boundary CSV.")
    only_in_run = run_items - bnd_items
    if only_in_run:
        print(f"  Note: {len(only_in_run)} run items have no boundary data — skipped.")
    if args.max_items is not None:
        items = items[: args.max_items]
    print(f"  Processing {len(items)} items across {len(args.layers)} layers.")

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = (
            _REPO_ROOT
            / "reproduce" / "figures" / "audio"
            / "auristream_identity_position" / run_id
        )
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output directory exists: {out_dir}\nUse --overwrite to replace."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = vars(args).copy()
    cfg.update(run_id=run_id, token_rate_hz=token_rate_hz,
               n_items=len(items), analysis_unit=_ANALYSIS_UNIT)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    qc_rows:        list[dict] = []
    panel_data:     list[dict] = []   # for cosine summary panels
    magnitude_data: list[dict] = []   # for magnitude-by-position panels

    # =========================================================================
    # Main loop: layer by layer
    # =========================================================================
    for layer in args.layers:
        print(f"\n{'='*60}\nLayer: {layer}")

        all_delta_vecs: list[np.ndarray]   = []
        all_delta_meta: list[pd.DataFrame] = []

        for item_id in items:
            try:
                act = load_layer_activation(run_dir, manifest, item_id, layer)
            except (KeyError, FileNotFoundError) as exc:
                print(f"  Skipping {item_id}: {exc}", file=sys.stderr)
                continue

            bounds_item = (
                boundaries_df[boundaries_df["item_id"] == item_id]
                .sort_values("phoneme_index")
                .reset_index(drop=True)
            )
            if bounds_item.empty:
                continue

            n_tokens = act.shape[0]
            _, _, delta_vecs, delta_meta = build_phoneme_last_and_delta(
                act, bounds_item, n_tokens, token_rate_hz, item_id
            )

            if len(delta_vecs) == 0:
                continue

            if np.any(np.isnan(delta_vecs)):
                print(
                    f"  Warning: NaN in delta_vecs for {item_id}/{layer} — skipped.",
                    file=sys.stderr,
                )
                continue

            all_delta_vecs.append(delta_vecs)
            all_delta_meta.append(delta_meta)

        if not all_delta_vecs:
            print(f"  No vectors for {layer}/{_ANALYSIS_UNIT} — skipped.", file=sys.stderr)
            continue

        delta_vecs_all = np.concatenate(all_delta_vecs, axis=0).astype(np.float64)
        delta_meta_all = pd.concat(all_delta_meta, ignore_index=True)
        delta_meta_all["vector_id"]     = np.arange(len(delta_meta_all))
        delta_meta_all["layer"]         = layer
        delta_meta_all["analysis_unit"] = _ANALYSIS_UNIT

        if dataset_meta is not None:
            delta_meta_all = delta_meta_all.merge(dataset_meta, on="item_id", how="left")

        n_total = len(delta_vecs_all)
        n_con   = int((delta_meta_all["phoneme_type"] == "consonant").sum())
        n_vow   = int((delta_meta_all["phoneme_type"] == "vowel").sum())
        print(f"  Vectors: {n_total} total | {n_con} consonant | {n_vow} vowel")

        # Vector metadata CSV
        delta_meta_all.to_csv(
            out_dir / f"vector_metadata_{layer}_{_ANALYSIS_UNIT}.csv", index=False
        )

        # Magnitude summary CSV
        build_magnitude_summary(delta_meta_all).to_csv(
            out_dir / f"magnitude_position_summary_{layer}_{_ANALYSIS_UNIT}.csv", index=False
        )

        if layer in _PRIMARY_PANEL_LAYERS:
            magnitude_data.append({"layer": layer, "meta_df": delta_meta_all})

        # Pair sampling
        try:
            pairs_df = sample_pairs_identity_position(delta_meta_all, args, rng)
        except RuntimeError as exc:
            print(f"  Skipping {layer}: {exc}", file=sys.stderr)
            continue

        # Composition QC (independent of cosine mode)
        build_pair_composition(pairs_df).to_csv(
            out_dir / f"identity_position_pair_composition_{layer}_{_ANALYSIS_UNIT}.csv",
            index=False,
        )

        # Cosines: raw and centered on the SAME pairs
        raw_vecs  = l2_normalize(delta_vecs_all)
        cent_vecs = center_and_l2_normalize(delta_vecs_all)

        pairs_raw             = pairs_df.copy()
        pairs_raw["cosine"]      = compute_cosines(raw_vecs, pairs_df)
        pairs_raw["cosine_mode"] = "raw"

        pairs_cen             = pairs_df.copy()
        pairs_cen["cosine"]      = compute_cosines(cent_vecs, pairs_df)
        pairs_cen["cosine_mode"] = "centered"

        pairs_all = pd.concat([pairs_raw, pairs_cen], ignore_index=True)
        pairs_all["layer"]         = layer
        pairs_all["analysis_unit"] = _ANALYSIS_UNIT

        pairs_all.to_csv(
            out_dir / f"identity_position_pairs_{layer}_{_ANALYSIS_UNIT}.csv", index=False
        )
        build_cosine_summary(pairs_all).to_csv(
            out_dir / f"identity_position_summary_{layer}_{_ANALYSIS_UNIT}.csv", index=False
        )

        # QC rows
        for mode, pmode in (("raw", pairs_raw), ("centered", pairs_cen)):
            for cat, grp in pmode.groupby("pair_category"):
                cosines = grp["cosine"].values
                qc_rows.append({
                    "layer":             layer,
                    "analysis_unit":     _ANALYSIS_UNIT,
                    "cosine_mode":       mode,
                    "pair_category":     cat,
                    "n_vectors_total":   n_total,
                    "n_consonants":      n_con,
                    "n_vowels":          n_vow,
                    "n_available_pairs": int(grp["n_available_pairs"].iloc[0]),
                    "n_sampled_pairs":   int(grp["n_sampled_pairs"].iloc[0]),
                    "n_nan_cosines":     int(np.isnan(cosines).sum()),
                    "cosine_mean":       float(np.nanmean(cosines)),
                    "cosine_std":        float(np.nanstd(cosines)),
                })

        if layer in _PRIMARY_PANEL_LAYERS:
            for mode, pmode in (("raw", pairs_raw), ("centered", pairs_cen)):
                panel_data.append({
                    "layer":       layer,
                    "cosine_mode": mode,
                    "pairs_df":    pmode,
                })

    # =========================================================================
    # Figures
    # =========================================================================
    requested_layers = set(args.layers)
    tag = _ANALYSIS_UNIT

    # 1. Centered cosine panel
    _p = out_dir / f"summary_panel_identity_position_{tag}_centered.png"
    plot_identity_position_panel(
        panel_data, _PRIMARY_PANEL_LAYERS, "centered", _p,
        density=args.density, requested_layers=requested_layers,
    )
    print(f"  Panel: {_p.name}")

    # 2. Raw cosine panel (QC)
    _p = out_dir / f"summary_panel_identity_position_{tag}_raw.png"
    plot_identity_position_panel(
        panel_data, _PRIMARY_PANEL_LAYERS, "raw", _p,
        density=args.density, requested_layers=requested_layers,
    )
    print(f"  Panel: {_p.name}")

    # 3. Centered cosine panel zoomed with mean lines
    _p = out_dir / f"summary_panel_identity_position_{tag}_centered_zoom.png"
    plot_identity_position_panel(
        panel_data, _PRIMARY_PANEL_LAYERS, "centered", _p,
        density=args.density, requested_layers=requested_layers,
        xlim=(-0.5, 0.5), show_mean_lines=True,
    )
    print(f"  Panel: {_p.name}")

    # 4. Magnitude by position
    _p = out_dir / f"magnitude_by_position_{tag}.png"
    plot_magnitude_by_position(
        magnitude_data, _PRIMARY_PANEL_LAYERS, _p,
        requested_layers=requested_layers,
    )
    print(f"  Panel: {_p.name}")

    # =========================================================================
    # QC summary
    # =========================================================================
    qc_df = pd.DataFrame(qc_rows)
    qc_df.to_csv(out_dir / "qc_summary.csv", index=False)

    if not qc_df.empty:
        n_nan = int(qc_df["n_nan_cosines"].sum())
        if n_nan > 0:
            print(f"\nWARNING: {n_nan} NaN cosines — check qc_summary.csv.", file=sys.stderr)
        small = qc_df[qc_df["n_sampled_pairs"] < 100]
        if not small.empty:
            print(
                f"\nWARNING: {len(small)} (layer×mode×category) entries have <100 sampled pairs "
                "— check qc_summary.csv for small categories.",
                file=sys.stderr,
            )

    print(f"\nDone. Outputs: {out_dir}")


if __name__ == "__main__":
    main()
