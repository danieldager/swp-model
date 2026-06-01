#!/usr/bin/env python3
"""Decompose block_47 → block_48 into attention and MLP residual updates.

Architecture (Block.forward, standard path, no cache):
    x = x + attn_scale * attn(norm1(x))   # attn_scale = 1.0 in this checkpoint
    x = x + mlp(norm2(x))
    ⟹ block_48 = block_47 + attn_update + mlp_update

This script uses saved block_47 token-level activations to re-run the last
Transformer block submodules manually — no full re-extraction needed.

Tasks:
  Task 3: verify block_48 ≈ block_47 + attn_update + mlp_update (exact up to float32)
           and block_48_lnf ≈ ln_f(block_48)
  Task 4: token-level anisotropy for attn_update, mlp_update, delta_total
  Task 5: phoneme-pooled component anisotropy (uses token_start/end_idx from metadata)
  Task 6: position-binned component metrics

Key question:
  Does attn_update or mlp_update have higher isotropy_ratio and alignment with delta_total?
  Whichever dominates explains the near-common residual.

Usage:
    python scripts/audio/auristream_last_block_component_diagnostics.py \\
        --run reproduce/data/audio/auristream__6ee9aeb6 \\
        --max-items 3 \\
        --overwrite

Output:
    reproduce/figures/audio/auristream_last_block_components/auristream__6ee9aeb6/
"""

from __future__ import annotations

import argparse
import importlib
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
_DPI = 150

COMPONENT_NAMES = ["attn_update", "mlp_update", "delta_total"]
COMPONENT_COLORS = {
    "attn_update": "#d73027",
    "mlp_update":  "#4575b4",
    "delta_total": "#333333",
    "block_47":    "#91bfdb",
    "block_48":    "#fee090",
    "block_48_lnf":"#fc8d59",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def _phoneme_type(base: str) -> str:
    return "vowel" if base in _ARPABET_VOWELS else ("consonant" if base.isalpha() else "other")


def _anisotropy_metrics(
    emb_raw: np.ndarray,
    rng: np.random.Generator,
    n_cos: int = 400,
    n_eu:  int = 100,
) -> dict:
    """Norm, isotropy, pairwise cosine, PCA PC1 EVR."""
    n, d = emb_raw.shape
    norms   = np.linalg.norm(emb_raw, axis=1)
    emb_n   = (emb_raw / (norms[:, None] + 1e-12)).astype(np.float32)

    mean_vec      = emb_raw.mean(axis=0)
    mean_vec_norm = float(np.linalg.norm(mean_vec))
    isotropy      = mean_vec_norm / (float(norms.mean()) + 1e-12)
    mean_dir      = (mean_vec / (mean_vec_norm + 1e-12)).astype(np.float32)
    cos_mean      = emb_n @ mean_dir

    n_s = min(n_cos, n)
    idx = rng.choice(n, n_s, replace=False)
    cp  = (emb_n[idx] @ emb_n[idx].T)[np.triu_indices(n_s, k=1)]

    n_pca = min(5, d, n)
    pca   = PCA(n_components=n_pca)
    pca.fit(emb_raw)

    return {
        "n":               n,
        "norm_mean":       float(norms.mean()),
        "norm_std":        float(norms.std()),
        "norm_q05":        float(np.percentile(norms, 5)),
        "norm_q95":        float(np.percentile(norms, 95)),
        "isotropy_ratio":  float(isotropy),
        "cos_with_mean_mean": float(cos_mean.mean()),
        "cos_with_mean_std":  float(cos_mean.std()),
        "cos_pairwise_mean":  float(cp.mean()),
        "cos_pairwise_q05":   float(np.percentile(cp, 5)),
        "cos_pairwise_q95":   float(np.percentile(cp, 95)),
        "pca_pc1_evr":     float(pca.explained_variance_ratio_[0]),
    }


def _cross_cosines(
    a_raw: np.ndarray,
    b_raw: np.ndarray,
    n_sample: int = 400,
    rng: np.random.Generator | None = None,
) -> dict:
    """Per-token cos(a_i, b_i) and cos(mean_a, mean_b)."""
    a_n = (a_raw / (np.linalg.norm(a_raw, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
    b_n = (b_raw / (np.linalg.norm(b_raw, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
    per_token = (a_n * b_n).sum(axis=1)

    ma = a_raw.mean(axis=0)
    mb = b_raw.mean(axis=0)
    ma_n = (ma / (np.linalg.norm(ma) + 1e-12)).astype(np.float32)
    mb_n = (mb / (np.linalg.norm(mb) + 1e-12)).astype(np.float32)
    cos_mean_vecs = float(ma_n @ mb_n)

    return {
        "per_token_mean": float(per_token.mean()),
        "per_token_std":  float(per_token.std()),
        "per_token_q05":  float(np.percentile(per_token, 5)),
        "per_token_q95":  float(np.percentile(per_token, 95)),
        "cos_mean_vecs":  cos_mean_vecs,
    }


def _position_binned_metrics(
    components: dict[str, np.ndarray],
    pos_all: np.ndarray,
    rng: np.random.Generator,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Compute component isotropy_ratio and norm by absolute token position bins."""
    bin_edges = np.unique(
        np.percentile(pos_all, np.linspace(0, 100, n_bins + 1)).astype(int)
    )
    if len(bin_edges) < 2:
        return pd.DataFrame()

    rows = []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (pos_all >= lo) & (pos_all <= hi if i == len(bin_edges) - 2 else pos_all < hi)
        if mask.sum() < 2:
            continue
        bin_label = f"{lo}-{hi}"
        row = {"position_bin": bin_label, "n": int(mask.sum())}
        for name, emb in components.items():
            sub = emb[mask]
            norms = np.linalg.norm(sub, axis=1)
            sub_n = (sub / (norms[:, None] + 1e-12)).astype(np.float32)
            mean_v = sub.mean(axis=0)
            iso = float(np.linalg.norm(mean_v) / (norms.mean() + 1e-12))
            mean_dir = (mean_v / (np.linalg.norm(mean_v) + 1e-12)).astype(np.float32)
            row[f"{name}_norm_mean"] = float(norms.mean())
            row[f"{name}_isotropy_ratio"] = iso
            row[f"{name}_cos_with_mean"] = float((sub_n @ mean_dir).mean())
        rows.append(row)
    return pd.DataFrame(rows)


# ── Figures ────────────────────────────────────────────────────────────────────

def _plot_component_norms(tok_data: dict[str, np.ndarray], output_path: Path) -> None:
    names = list(tok_data.keys())
    means = [float(np.linalg.norm(tok_data[n], axis=1).mean()) for n in names]
    stds  = [float(np.linalg.norm(tok_data[n], axis=1).std())  for n in names]
    colors = [COMPONENT_COLORS.get(n, "steelblue") for n in names]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.4), 4))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, edgecolor="none",
           capsize=4, error_kw={"linewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Mean L2 norm  (± std)", fontsize=9)
    ax.set_title("Component norm distribution  (token level)", fontsize=10)
    fig.tight_layout()
    _savefig(fig, output_path)


def _plot_alignment_with_delta(
    tok_data: dict[str, np.ndarray],
    delta_key: str,
    output_path: Path,
) -> None:
    """Per-token cos(component_i, delta_i) and cos(mean_component, mean_delta)."""
    delta_raw = tok_data[delta_key]
    comparisons = [n for n in tok_data if n != delta_key]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: per-token cosine mean
    per_means = []
    per_q05   = []
    per_q95   = []
    for n in comparisons:
        cc = _cross_cosines(tok_data[n], delta_raw)
        per_means.append(cc["per_token_mean"])
        per_q05.append(cc["per_token_q05"])
        per_q95.append(cc["per_token_q95"])

    x = np.arange(len(comparisons))
    colors = [COMPONENT_COLORS.get(n, "steelblue") for n in comparisons]
    axes[0].bar(x, per_means, color=colors, alpha=0.85, edgecolor="none")
    axes[0].errorbar(x, per_means,
                     yerr=[np.array(per_means) - np.array(per_q05),
                           np.array(per_q95) - np.array(per_means)],
                     fmt="none", color="black", capsize=4, linewidth=1.2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(comparisons, rotation=25, ha="right", fontsize=9)
    axes[0].set_ylim(-0.1, 1.05)
    axes[0].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("cos(component_i, delta_total_i)", fontsize=9)
    axes[0].set_title("Per-token cosine with delta_total\n(mean ± [q05, q95])", fontsize=9)

    # Panel 2: mean-vector cosine
    mean_cos = [_cross_cosines(tok_data[n], delta_raw)["cos_mean_vecs"] for n in comparisons]
    axes[1].bar(x, mean_cos, color=colors, alpha=0.85, edgecolor="none")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(comparisons, rotation=25, ha="right", fontsize=9)
    axes[1].set_ylim(-0.1, 1.05)
    axes[1].set_ylabel("cos(mean_component, mean_delta_total)", fontsize=9)
    axes[1].set_title("Mean-vector cosine with mean delta_total", fontsize=9)

    fig.suptitle("Component alignment with total delta  (token level)", fontsize=10)
    fig.tight_layout()
    _savefig(fig, output_path)


def _plot_anisotropy(aniso_df: pd.DataFrame, output_path: Path) -> None:
    components = aniso_df["component"].tolist()
    x = np.arange(len(components))
    colors = [COMPONENT_COLORS.get(c, "steelblue") for c in components]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, title in zip(
        axes,
        ["isotropy_ratio", "cos_with_mean_mean", "cos_pairwise_mean"],
        ["Isotropy ratio\n(0=isotropic, 1=collapsed)", "Cos with component\nmean direction",
         "Pairwise cosine\n(sampled)"],
    ):
        ax.bar(x, aniso_df[col], color=colors, alpha=0.85, edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(components, rotation=25, ha="right", fontsize=9)
        ax.set_ylim(0, 1.07)
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(title, fontsize=9)

    fig.suptitle("Component anisotropy  (token level)", fontsize=10)
    fig.tight_layout()
    _savefig(fig, output_path)


def _plot_position_metrics(pos_df: pd.DataFrame, output_path: Path) -> None:
    bins = pos_df["position_bin"].tolist()
    x = np.arange(len(bins))

    iso_cols = [c for c in pos_df.columns if c.endswith("_isotropy_ratio")]
    if not iso_cols:
        return

    fig, axes = plt.subplots(1, len(iso_cols), figsize=(4 * len(iso_cols), 4))
    if len(iso_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, iso_cols):
        name = col.replace("_isotropy_ratio", "")
        ax.bar(x, pos_df[col], color=COMPONENT_COLORS.get(name, "steelblue"),
               alpha=0.85, edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1.07)
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(f"{name}\nisotropy_ratio", fontsize=9)

    fig.suptitle("Component isotropy by absolute token position", fontsize=10)
    fig.tight_layout()
    _savefig(fig, output_path)


# ── Core computation ───────────────────────────────────────────────────────────

def decompose_last_block(
    last_block,
    ln_f,
    h47_item: torch.Tensor,  # [D, L]
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], float]:
    """Apply last block submodules step-by-step on one item.

    Returns dict of [D, L] tensors and attn_scale value.
    """
    attn_scale = float(getattr(last_block, "attn_scale", 1.0))
    x = h47_item.T.unsqueeze(0).to(device)  # [1, L, D]

    with torch.inference_mode():
        normed1    = last_block.norm1(x)
        attn_raw   = last_block.attn(normed1)        # [1, L, D]  (no return_kv)
        attn_upd   = attn_scale * attn_raw            # [1, L, D]
        after_attn = x + attn_upd                    # [1, L, D]
        normed2    = last_block.norm2(after_attn)
        mlp_upd    = last_block.mlp(normed2)          # [1, L, D]
        block48_r  = after_attn + mlp_upd             # [1, L, D]
        block48lnf_r = ln_f(block48_r)               # [1, L, D]

    def _to_dl(t: torch.Tensor) -> torch.Tensor:
        return t.squeeze(0).T.float().cpu()  # → [D, L]

    return {
        "attn_update":      _to_dl(attn_upd),
        "mlp_update":       _to_dl(mlp_upd),
        "block_48_recomp":  _to_dl(block48_r),
        "block_48_lnf_recomp": _to_dl(block48lnf_r),
    }, attn_scale


def verify_reconstruction(
    block48_recomp: torch.Tensor,     # [D, L]
    block48_saved:  torch.Tensor,     # [D, L]
    label:          str,
) -> dict:
    if block48_recomp.shape != block48_saved.shape:
        return {"label": label, "shape_ok": False,
                "max_abs": None, "mean_abs": None, "rel_err": None, "pass": False}

    diff     = (block48_recomp - block48_saved).abs()
    max_abs  = float(diff.max())
    mean_abs = float(diff.mean())
    denom    = block48_saved.abs().mean().clamp(min=1e-12)
    rel_err  = float(mean_abs / denom)
    passed   = max_abs < 1e-3

    print(f"    [{label}] max_abs={max_abs:.3e}  mean_abs={mean_abs:.3e}  "
          f"rel={rel_err:.3e}  {'✓ PASS' if passed else '✗ FAIL'}")
    return {
        "label":    label,
        "shape_ok": True,
        "max_abs":  max_abs,
        "mean_abs": mean_abs,
        "rel_err":  rel_err,
        "pass":     passed,
    }


# ── Phoneme pooling ────────────────────────────────────────────────────────────

def pool_components_by_phoneme(
    item_id:    str,
    meta:       pd.DataFrame,
    tok_comps:  dict[str, np.ndarray],  # each [L, D]
) -> list[dict]:
    """Mean-pool component vectors over phoneme token intervals for one item."""
    item_meta = meta[meta["item_id"] == item_id]
    rows = []
    L = next(iter(tok_comps.values())).shape[0]

    for _, row in item_meta.iterrows():
        if not bool(row["valid_embedding"]) or bool(row["is_silence"]):
            continue
        tok_s = int(row["token_start_idx"])
        tok_e = int(row["token_end_idx"])
        if tok_s < 0 or tok_e >= L or tok_e < tok_s:
            continue

        ph_row: dict = {
            "item_id": item_id,
            "phoneme": row["phoneme"],
            "phoneme_type": row["phoneme_type"] if "phoneme_type" in row.index else "?",
        }
        for name, arr in tok_comps.items():
            pooled = arr[tok_s : tok_e + 1].mean(axis=0)  # [D]
            ph_row[name] = pooled
        rows.append(ph_row)
    return rows


# ── CLI + main ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", required=True,
                   help="Extraction run directory, e.g. reproduce/data/audio/auristream__6ee9aeb6")
    p.add_argument("--max-items", type=int, default=3, dest="max_items",
                   help="Number of items to process (default: 3 — keep tiny).")
    p.add_argument("--n-pos-bins", type=int, default=5, dest="n_pos_bins",
                   help="Number of absolute token position bins for Task 6 (default: 5).")
    p.add_argument("--random-state", type=int, default=0, dest="random_state")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--output", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = (REPO_ROOT / args.run).resolve()
    emb_dir = run_dir / "phoneme_embeddings"
    rng = np.random.default_rng(args.random_state)

    with open(run_dir / "manifest.json") as f:
        manifest = json.load(f)
    run_id = manifest["run_id"]

    output_dir = (
        Path(args.output).resolve() if args.output
        else REPO_ROOT / "reproduce" / "figures" / "audio"
             / "auristream_last_block_components" / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "last_block_component_verification.json"
    if not args.overwrite and json_path.exists():
        print(f"[comp_diag] Output exists: {output_dir}\nPass --overwrite to regenerate.")
        return

    for required in ("block_47", "block_48", "block_48_lnf"):
        if required not in manifest["run_params"]["layers"]:
            print(f"[comp_diag] {required!r} not in run manifest — cannot proceed.")
            sys.exit(1)

    item_ids = list(manifest["items"].keys())[: args.max_items]
    print(f"[comp_diag] Items : {item_ids}")
    print(f"[comp_diag] Run   : {run_id}")

    # ── Load model ─────────────────────────────────────────────────────────────
    print("\n[comp_diag] Loading AuriStream model …")
    importlib.import_module("swp.audio.models.auristream")
    from swp.audio.models.registry import get_model
    model = get_model("auristream")
    device = model.device

    last_block = model._lm.transformer.h[-1].eval()
    ln_f       = model._lm.transformer.ln_f.eval()
    attn_scale = float(getattr(last_block, "attn_scale", 1.0))
    print(f"[comp_diag] Last block : transformer.h[{len(model._lm.transformer.h) - 1}]")
    print(f"[comp_diag] attn_scale : {attn_scale}")
    print(f"[comp_diag] use_rope   : {getattr(model._lm.config, 'use_rope', 'unknown')}")
    del model   # free all weights except last_block and ln_f

    # ── Per-item decomposition ─────────────────────────────────────────────────
    verifications: list[dict] = []
    # Accumulate token-level tensors
    tok_attn:    list[np.ndarray] = []
    tok_mlp:     list[np.ndarray] = []
    tok_delta:   list[np.ndarray] = []
    tok_b47:     list[np.ndarray] = []
    tok_b48:     list[np.ndarray] = []
    tok_b48lnf:  list[np.ndarray] = []
    tok_pos:     list[np.ndarray] = []

    # Phoneme-level accumulators
    pooled_rows: list[dict] = []
    meta_loaded: pd.DataFrame | None = None
    if (emb_dir / "metadata_phonemes.csv").exists():
        meta_loaded = pd.read_csv(emb_dir / "metadata_phonemes.csv")
        meta_loaded["phoneme_base"] = (
            meta_loaded["phoneme"].str.replace(r"\d+$", "", regex=True).str.upper()
        )
        meta_loaded["phoneme_type"] = meta_loaded["phoneme_base"].map(_phoneme_type)

    for item_id in item_ids:
        paths = manifest["items"][item_id]
        print(f"\n  {item_id}")

        h47     = torch.load(run_dir / paths["block_47"],     map_location="cpu", weights_only=True).float()
        h48     = torch.load(run_dir / paths["block_48"],     map_location="cpu", weights_only=True).float()
        h48_lnf = torch.load(run_dir / paths["block_48_lnf"], map_location="cpu", weights_only=True).float()

        L = h47.shape[1]
        print(f"    D={h47.shape[0]}, L={L}")

        comps, _ = decompose_last_block(last_block, ln_f, h47, device)

        # Reconstruction checks
        v1 = verify_reconstruction(comps["block_48_recomp"],     h48,     "block_48")
        v2 = verify_reconstruction(comps["block_48_lnf_recomp"], h48_lnf, "block_48_lnf")

        # Also verify block_48 = block_47 + attn_update + mlp_update
        delta_recomp = h47 + comps["attn_update"] + comps["mlp_update"]
        v3 = verify_reconstruction(delta_recomp, h48, "block_47+attn+mlp")

        verifications += [v1, v2, v3]

        if not v1["pass"] or not v3["pass"]:
            print("    ✗ Reconstruction failed — stopping interpretation.")
            break

        # Accumulate token-level arrays: transpose to [L, D]
        attn_arr = comps["attn_update"].T.numpy().astype(np.float32)  # [L, D]
        mlp_arr  = comps["mlp_update"].T.numpy().astype(np.float32)
        delta_arr = (h48 - h47).T.numpy().astype(np.float32)
        b47_arr  = h47.T.numpy().astype(np.float32)
        b48_arr  = h48.T.numpy().astype(np.float32)
        b48lnf_arr = h48_lnf.T.numpy().astype(np.float32)

        tok_attn.append(attn_arr)
        tok_mlp.append(mlp_arr)
        tok_delta.append(delta_arr)
        tok_b47.append(b47_arr)
        tok_b48.append(b48_arr)
        tok_b48lnf.append(b48lnf_arr)
        tok_pos.append(np.arange(L, dtype=np.float32))

        # Phoneme pooling (Task 5)
        if meta_loaded is not None:
            ph_rows = pool_components_by_phoneme(
                item_id, meta_loaded,
                {
                    "attn_update": attn_arr,
                    "mlp_update":  mlp_arr,
                    "delta_total": delta_arr,
                    "block_47":    b47_arr,
                    "block_48":    b48_arr,
                },
            )
            pooled_rows.extend(ph_rows)

    # ── Save verification JSON + MD ────────────────────────────────────────────
    all_pass = all(v["pass"] for v in verifications if v["pass"] is not None)
    report = {
        "run_id":    run_id,
        "items":     item_ids,
        "attn_scale": attn_scale,
        "overall_pass": all_pass,
        "verifications": verifications,
    }
    json_path.write_text(json.dumps(report, indent=2))
    print(f"\n[comp_diag] Verification overall PASS: {all_pass}")

    md_lines = [
        f"# Last-block component verification — {run_id}",
        f"\nItems: {', '.join(item_ids)}  |  attn_scale={attn_scale}",
        "",
        "## Reconstruction checks",
        f"Overall PASS: **{all_pass}**",
        "",
        "| label | shape_ok | max_abs | mean_abs | rel_err | pass |",
        "|-------|----------|---------|----------|---------|------|",
    ]
    for v in verifications:
        if v["max_abs"] is None:
            md_lines.append(f"| {v['label']} | {v['shape_ok']} | N/A | N/A | N/A | ✗ |")
        else:
            md_lines.append(
                f"| {v['label']} | {v['shape_ok']} | {v['max_abs']:.2e} | "
                f"{v['mean_abs']:.2e} | {v['rel_err']:.2e} | {'✓' if v['pass'] else '✗'} |"
            )

    if not all_pass:
        md_lines.append("\n> **Reconstruction failed — component interpretation not reliable.**")
        (output_dir / "last_block_component_verification.md").write_text(
            "\n".join(md_lines) + "\n"
        )
        print("[comp_diag] Reconstruction failed — stopping.")
        return

    # ── Concatenate token-level arrays ────────────────────────────────────────
    tok_data: dict[str, np.ndarray] = {
        "attn_update": np.concatenate(tok_attn,   axis=0),
        "mlp_update":  np.concatenate(tok_mlp,    axis=0),
        "delta_total": np.concatenate(tok_delta,  axis=0),
        "block_47":    np.concatenate(tok_b47,    axis=0),
        "block_48":    np.concatenate(tok_b48,    axis=0),
        "block_48_lnf":np.concatenate(tok_b48lnf, axis=0),
    }
    pos_all = np.concatenate(tok_pos, axis=0)
    N = pos_all.shape[0]
    print(f"\n[comp_diag] Total tokens: {N}  (from {len(item_ids)} items)")

    # ── Task 4: token-level anisotropy ────────────────────────────────────────
    print("\n[comp_diag] Task 4 — token-level component anisotropy …")
    aniso_rows = []
    for name in list(tok_data.keys()):
        m = _anisotropy_metrics(tok_data[name], rng)
        aniso_rows.append({"component": name, **m})
        print(f"  {name:20s}: iso={m['isotropy_ratio']:.5f}  "
              f"cos_mean={m['cos_with_mean_mean']:.5f}  "
              f"norm_mean={m['norm_mean']:.3f}  "
              f"pca_pc1={m['pca_pc1_evr']:.4f}")

    aniso_df = pd.DataFrame(aniso_rows)
    aniso_df.to_csv(output_dir / "component_anisotropy_token_level.csv", index=False)
    print("[comp_diag] Saved component_anisotropy_token_level.csv")

    # Cross-component cosines (key comparison)
    print("\n[comp_diag] Cross-component alignment:")
    for name in ["attn_update", "mlp_update", "block_47"]:
        cc = _cross_cosines(tok_data[name], tok_data["delta_total"])
        print(f"  cos({name}, delta_total): "
              f"per_token_mean={cc['per_token_mean']:.5f}  "
              f"cos_mean_vecs={cc['cos_mean_vecs']:.5f}")

    # Relative norm contributions
    norm_attn  = float(np.linalg.norm(tok_data["attn_update"], axis=1).mean())
    norm_mlp   = float(np.linalg.norm(tok_data["mlp_update"],  axis=1).mean())
    norm_delta = float(np.linalg.norm(tok_data["delta_total"], axis=1).mean())
    print(f"\n  Relative norm contributions (mean norms):")
    print(f"    attn_update / delta_total : {norm_attn  / (norm_delta + 1e-12):.4f}")
    print(f"    mlp_update  / delta_total : {norm_mlp   / (norm_delta + 1e-12):.4f}")
    report["norm_attn_mean"]  = norm_attn
    report["norm_mlp_mean"]   = norm_mlp
    report["norm_delta_mean"] = norm_delta
    report["attn_norm_ratio"] = norm_attn / (norm_delta + 1e-12)
    report["mlp_norm_ratio"]  = norm_mlp  / (norm_delta + 1e-12)
    json_path.write_text(json.dumps(report, indent=2))

    # ── Task 5: phoneme-pooled level ──────────────────────────────────────────
    if pooled_rows:
        print(f"\n[comp_diag] Task 5 — phoneme-pooled anisotropy ({len(pooled_rows)} phonemes) …")
        pooled_aniso_rows = []
        comp_names_pooled = ["attn_update", "mlp_update", "delta_total", "block_47", "block_48"]
        for name in comp_names_pooled:
            arr = np.stack([r[name] for r in pooled_rows if name in r], axis=0)
            m = _anisotropy_metrics(arr, rng)
            pooled_aniso_rows.append({"component": name, **m})
            print(f"  {name:20s}: iso={m['isotropy_ratio']:.5f}  "
                  f"cos_mean={m['cos_with_mean_mean']:.5f}  "
                  f"norm_mean={m['norm_mean']:.3f}")

        pd.DataFrame(pooled_aniso_rows).to_csv(
            output_dir / "component_anisotropy_pooled_level.csv", index=False
        )
        print("[comp_diag] Saved component_anisotropy_pooled_level.csv")

    # ── Task 6: position-binned metrics ───────────────────────────────────────
    print("\n[comp_diag] Task 6 — position-binned component metrics …")
    pos_df = _position_binned_metrics(
        {k: tok_data[k] for k in COMPONENT_NAMES},
        pos_all, rng, n_bins=args.n_pos_bins,
    )
    if not pos_df.empty:
        pos_df.to_csv(output_dir / "component_by_absolute_position.csv", index=False)
        print(f"  Saved component_by_absolute_position.csv  ({len(pos_df)} bins)")
        print(pos_df[["position_bin", "n",
                       "attn_update_isotropy_ratio",
                       "mlp_update_isotropy_ratio",
                       "delta_total_isotropy_ratio"]].to_string(index=False))

    # ── Figures ────────────────────────────────────────────────────────────────
    print("\n[comp_diag] Writing figures …")

    _plot_component_norms(
        {k: tok_data[k] for k in ["attn_update", "mlp_update", "delta_total",
                                   "block_47", "block_48"]},
        output_dir / "summary_component_norms.png",
    )

    _plot_alignment_with_delta(
        {k: tok_data[k] for k in ["attn_update", "mlp_update", "block_47", "delta_total"]},
        "delta_total",
        output_dir / "summary_component_alignment_with_delta.png",
    )

    _plot_anisotropy(
        aniso_df[aniso_df["component"].isin(["attn_update", "mlp_update", "delta_total"])].reset_index(drop=True),
        output_dir / "summary_component_anisotropy.png",
    )

    if not pos_df.empty:
        _plot_position_metrics(pos_df, output_dir / "summary_component_by_absolute_position.png")
        print("  saved summary_component_by_absolute_position.png")

    print("  saved summary_component_norms.png")
    print("  saved summary_component_alignment_with_delta.png")
    print("  saved summary_component_anisotropy.png")

    # ── Update MD report with component summary ────────────────────────────────
    md_lines += [
        "",
        "## Component anisotropy (token level)",
        "",
        "| component | n | isotropy_ratio | cos_with_mean | cos_pairwise | norm_mean | pca_pc1_evr |",
        "|-----------|---|---------------|---------------|--------------|-----------|-------------|",
    ]
    for _, r in aniso_df.iterrows():
        md_lines.append(
            f"| {r['component']} | {r['n']} | {r['isotropy_ratio']:.5f} | "
            f"{r['cos_with_mean_mean']:.5f} | {r['cos_pairwise_mean']:.5f} | "
            f"{r['norm_mean']:.3f} | {r['pca_pc1_evr']:.4f} |"
        )

    md_lines += [
        "",
        "## Relative norm contributions (mean)",
        f"- attn_update mean norm : {norm_attn:.3f}",
        f"- mlp_update mean norm  : {norm_mlp:.3f}",
        f"- delta_total mean norm : {norm_delta:.3f}",
        f"- attn / delta : {norm_attn / (norm_delta + 1e-12):.4f}",
        f"- mlp  / delta : {norm_mlp  / (norm_delta + 1e-12):.4f}",
        "",
        "## Interpretation",
        "- **MLP dominates**: if mlp_update has high isotropy_ratio and "
        "high alignment with delta_total. Near-common update from MLP → "
        "output-preparation hypothesis for coch_head.",
        "- **Attention dominates**: if attn_update has high isotropy_ratio and "
        "high alignment with delta_total → context-dependent explanation more plausible.",
        "- **Both aligned**: quantify by relative norm and per-token cosine.",
        "",
        "> PCA PC1 EVR is supplementary: PCA centers internally; "
        "high isotropy_ratio can exist with modest PC1 EVR.",
    ]

    (output_dir / "last_block_component_verification.md").write_text(
        "\n".join(md_lines) + "\n"
    )
    json_path.write_text(json.dumps(report, indent=2))
    print(f"\n[comp_diag] Done — {output_dir}")


if __name__ == "__main__":
    main()
