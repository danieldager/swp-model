#!/usr/bin/env python3
"""Decompose any AuriStream Transformer block transition into attention and MLP residual updates.

Architecture (Block.forward, standard path):
    x = x + attn_scale * attn(norm1(x))   # attn_scale = 1.0 in AuriStream-1B
    x = x + mlp(norm2(x))
    ⟹ output = input + attn_update + mlp_update

The script uses SAVED token-level activations — no full re-extraction needed.

Examples:

  Last block (block_47 → block_48):
    python scripts/audio/auristream_block_component_diagnostics.py \\
        --run reproduce/data/audio/auristream__6ee9aeb6 \\
        --block-index 47 \\
        --input-layer block_47 \\
        --output-layer block_48 \\
        --max-items 30 \\
        --overwrite

  First block (embedding → block_01):
    python scripts/audio/auristream_block_component_diagnostics.py \\
        --run reproduce/data/audio/auristream__6ee9aeb6 \\
        --block-index 0 \\
        --input-layer embedding \\
        --output-layer block_01 \\
        --max-items 30 \\
        --overwrite

Output per transition:
    reproduce/figures/audio/auristream_block_components/{run_id}/{input_layer}_to_{output_layer}/
        block_component_verification.json
        block_component_verification.md
        component_anisotropy_token_level.csv
        component_anisotropy_pooled_level.csv    (when phoneme metadata available)
        component_by_absolute_position.csv
        summary_component_norms.png
        summary_component_alignment_with_delta.png
        summary_component_anisotropy.png
        summary_component_by_absolute_position.png   (optional)
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

_BASE_COLORS = {
    "attn_update": "#d73027",
    "mlp_update":  "#4575b4",
    "delta_total": "#333333",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def _phoneme_type(base: str) -> str:
    return "vowel" if base in _ARPABET_VOWELS else ("consonant" if base.isalpha() else "other")


def _component_color(name: str, input_layer: str, output_layer: str) -> str:
    if name in _BASE_COLORS:
        return _BASE_COLORS[name]
    if name == input_layer:
        return "#91bfdb"
    if name == output_layer:
        return "#fee090"
    return "steelblue"


# ── Anisotropy / cross-cosine metrics ──────────────────────────────────────────

def _anisotropy_metrics(
    emb_raw: np.ndarray,
    rng: np.random.Generator,
    n_cos: int = 400,
) -> dict:
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
        "n":                  n,
        "norm_mean":          float(norms.mean()),
        "norm_std":           float(norms.std()),
        "norm_q05":           float(np.percentile(norms, 5)),
        "norm_q95":           float(np.percentile(norms, 95)),
        "isotropy_ratio":     float(isotropy),
        "cos_with_mean_mean": float(cos_mean.mean()),
        "cos_with_mean_std":  float(cos_mean.std()),
        "cos_pairwise_mean":  float(cp.mean()),
        "cos_pairwise_q05":   float(np.percentile(cp, 5)),
        "cos_pairwise_q95":   float(np.percentile(cp, 95)),
        "pca_pc1_evr":        float(pca.explained_variance_ratio_[0]),
    }


def _cross_cosines(a_raw: np.ndarray, b_raw: np.ndarray) -> dict:
    a_n = (a_raw / (np.linalg.norm(a_raw, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
    b_n = (b_raw / (np.linalg.norm(b_raw, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
    per_token = (a_n * b_n).sum(axis=1)
    ma  = (a_raw.mean(axis=0) / (np.linalg.norm(a_raw.mean(axis=0)) + 1e-12)).astype(np.float32)
    mb  = (b_raw.mean(axis=0) / (np.linalg.norm(b_raw.mean(axis=0)) + 1e-12)).astype(np.float32)
    return {
        "per_token_mean": float(per_token.mean()),
        "per_token_std":  float(per_token.std()),
        "per_token_q05":  float(np.percentile(per_token, 5)),
        "per_token_q95":  float(np.percentile(per_token, 95)),
        "cos_mean_vecs":  float(ma @ mb),
    }


def _position_binned_metrics(
    components: dict[str, np.ndarray],
    pos_all: np.ndarray,
    rng: np.random.Generator,
    n_bins: int = 5,
) -> pd.DataFrame:
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
        row: dict = {"position_bin": f"{lo}-{hi}", "n": int(mask.sum())}
        for name, emb in components.items():
            sub    = emb[mask]
            norms  = np.linalg.norm(sub, axis=1)
            sub_n  = (sub / (norms[:, None] + 1e-12)).astype(np.float32)
            mean_v = sub.mean(axis=0)
            iso    = float(np.linalg.norm(mean_v) / (norms.mean() + 1e-12))
            mean_d = (mean_v / (np.linalg.norm(mean_v) + 1e-12)).astype(np.float32)
            row[f"{name}_norm_mean"]       = float(norms.mean())
            row[f"{name}_isotropy_ratio"]  = iso
            row[f"{name}_cos_with_mean"]   = float((sub_n @ mean_d).mean())
        rows.append(row)
    return pd.DataFrame(rows)


# ── Figures ────────────────────────────────────────────────────────────────────

def _plot_component_norms(
    tok_data: dict[str, np.ndarray],
    title: str,
    output_path: Path,
    input_layer: str,
    output_layer: str,
) -> None:
    names = list(tok_data.keys())
    means = [float(np.linalg.norm(tok_data[n], axis=1).mean()) for n in names]
    stds  = [float(np.linalg.norm(tok_data[n], axis=1).std())  for n in names]
    colors = [_component_color(n, input_layer, output_layer) for n in names]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.5), 4))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, edgecolor="none",
           capsize=4, error_kw={"linewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Mean L2 norm  (± std)", fontsize=9)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    _savefig(fig, output_path)


def _plot_alignment_with_delta(
    tok_data: dict[str, np.ndarray],
    delta_key: str,
    title: str,
    output_path: Path,
    input_layer: str,
    output_layer: str,
) -> None:
    delta_raw   = tok_data[delta_key]
    comparisons = [n for n in tok_data if n != delta_key]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    per_means, per_q05, per_q95, mean_cos = [], [], [], []
    for n in comparisons:
        cc = _cross_cosines(tok_data[n], delta_raw)
        per_means.append(cc["per_token_mean"])
        per_q05.append(cc["per_token_q05"])
        per_q95.append(cc["per_token_q95"])
        mean_cos.append(cc["cos_mean_vecs"])

    x = np.arange(len(comparisons))
    colors = [_component_color(n, input_layer, output_layer) for n in comparisons]

    axes[0].bar(x, per_means, color=colors, alpha=0.85, edgecolor="none")
    axes[0].errorbar(x, per_means,
                     yerr=[np.array(per_means) - np.array(per_q05),
                           np.array(per_q95) - np.array(per_means)],
                     fmt="none", color="black", capsize=4, linewidth=1.2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(comparisons, rotation=25, ha="right", fontsize=9)
    axes[0].set_ylim(-0.1, 1.05)
    axes[0].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel(f"cos(component_i, {delta_key}_i)", fontsize=9)
    axes[0].set_title("Per-token cosine with delta_total\n(mean ± [q05, q95])", fontsize=9)

    axes[1].bar(x, mean_cos, color=colors, alpha=0.85, edgecolor="none")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(comparisons, rotation=25, ha="right", fontsize=9)
    axes[1].set_ylim(-0.1, 1.05)
    axes[1].set_ylabel(f"cos(mean_component, mean_{delta_key})", fontsize=9)
    axes[1].set_title("Mean-vector cosine with mean delta_total", fontsize=9)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    _savefig(fig, output_path)


def _plot_anisotropy(
    aniso_df: pd.DataFrame,
    title: str,
    output_path: Path,
    input_layer: str,
    output_layer: str,
) -> None:
    components = aniso_df["component"].tolist()
    x      = np.arange(len(components))
    colors = [_component_color(c, input_layer, output_layer) for c in components]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, ylabel in zip(
        axes,
        ["isotropy_ratio", "cos_with_mean_mean", "cos_pairwise_mean"],
        ["Isotropy ratio\n(0=isotropic, 1=collapsed)",
         "Cos with component\nmean direction",
         "Pairwise cosine\n(sampled)"],
    ):
        ax.bar(x, aniso_df[col], color=colors, alpha=0.85, edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(components, rotation=25, ha="right", fontsize=9)
        ax.set_ylim(0, 1.07)
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylabel(ylabel, fontsize=9)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    _savefig(fig, output_path)


def _plot_position_metrics(
    pos_df: pd.DataFrame,
    title: str,
    output_path: Path,
    input_layer: str,
    output_layer: str,
) -> None:
    bins = pos_df["position_bin"].tolist()
    x    = np.arange(len(bins))

    iso_cols = [c for c in pos_df.columns if c.endswith("_isotropy_ratio")]
    if not iso_cols:
        return

    fig, axes = plt.subplots(1, len(iso_cols), figsize=(4 * len(iso_cols), 4))
    if len(iso_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, iso_cols):
        name = col.replace("_isotropy_ratio", "")
        ax.bar(x, pos_df[col],
               color=_component_color(name, input_layer, output_layer),
               alpha=0.85, edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1.07)
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(f"{name}\nisotropy_ratio", fontsize=9)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    _savefig(fig, output_path)


# ── Core: block decomposition ──────────────────────────────────────────────────

def decompose_block(
    block,
    h_in: torch.Tensor,   # [D, L] saved input layer
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Apply one Block's submodules step-by-step.

    Returns dict of [D, L] float32 CPU tensors:
        attn_update, mlp_update, output_recomp
    """
    attn_scale = float(getattr(block, "attn_scale", 1.0))
    x = h_in.T.unsqueeze(0).to(device)   # [1, L, D]

    with torch.inference_mode():
        attn_raw   = block.attn(block.norm1(x))       # [1, L, D]
        attn_upd   = attn_scale * attn_raw
        after_attn = x + attn_upd
        mlp_upd    = block.mlp(block.norm2(after_attn))
        out_r      = after_attn + mlp_upd

    def _to_dl(t: torch.Tensor) -> torch.Tensor:
        return t.squeeze(0).T.float().cpu()   # [D, L]

    return {
        "attn_update":   _to_dl(attn_upd),
        "mlp_update":    _to_dl(mlp_upd),
        "output_recomp": _to_dl(out_r),
        "_attn_scale":   torch.tensor(attn_scale),
    }


def verify_recon(recomp: torch.Tensor, saved: torch.Tensor, label: str) -> dict:
    if recomp.shape != saved.shape:
        print(f"    [{label}] shape mismatch: {recomp.shape} vs {saved.shape}  ✗ FAIL")
        return {"label": label, "shape_ok": False,
                "max_abs": None, "mean_abs": None, "rel_err": None, "pass": False}

    diff     = (recomp - saved).abs()
    max_abs  = float(diff.max())
    mean_abs = float(diff.mean())
    denom    = saved.abs().mean().clamp(min=1e-12)
    rel_err  = float(mean_abs / denom)
    passed   = max_abs < 1e-3

    print(f"    [{label}] max_abs={max_abs:.3e}  mean_abs={mean_abs:.3e}  "
          f"rel={rel_err:.3e}  {'✓ PASS' if passed else '✗ FAIL'}")
    return {"label": label, "shape_ok": True, "max_abs": max_abs,
            "mean_abs": mean_abs, "rel_err": rel_err, "pass": passed}


# ── Phoneme pooling ────────────────────────────────────────────────────────────

def pool_components_by_phoneme(
    item_id:  str,
    meta:     pd.DataFrame,
    tok_comp: dict[str, np.ndarray],  # [L, D] each
) -> list[dict]:
    rows = []
    L    = next(iter(tok_comp.values())).shape[0]
    for _, row in meta[meta["item_id"] == item_id].iterrows():
        if not bool(row["valid_embedding"]) or bool(row["is_silence"]):
            continue
        s, e = int(row["token_start_idx"]), int(row["token_end_idx"])
        if s < 0 or e >= L or e < s:
            continue
        ph_row: dict = {
            "item_id":      item_id,
            "phoneme":      row["phoneme"],
            "phoneme_type": row.get("phoneme_type", "?"),
        }
        for name, arr in tok_comp.items():
            ph_row[name] = arr[s : e + 1].mean(axis=0)
        rows.append(ph_row)
    return rows


# ── CLI + main ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", required=True,
                   help="Extraction run directory.")
    p.add_argument("--block-index", type=int, required=True, dest="block_index",
                   help="0-based Transformer block index (e.g. 0 for first, 47 for last).")
    p.add_argument("--input-layer", required=True, dest="input_layer",
                   help="Name of saved input layer (e.g. embedding, block_47).")
    p.add_argument("--output-layer", required=True, dest="output_layer",
                   help="Name of saved output layer (e.g. block_01, block_48).")
    p.add_argument("--max-items", type=int, default=30, dest="max_items",
                   help="Number of items to process (default: 30).")
    p.add_argument("--n-pos-bins", type=int, default=5, dest="n_pos_bins",
                   help="Absolute token position bins (default: 5).")
    p.add_argument("--random-state", type=int, default=0, dest="random_state")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--output", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = (REPO_ROOT / args.run).resolve()
    emb_dir = run_dir / "phoneme_embeddings"
    rng     = np.random.default_rng(args.random_state)

    with open(run_dir / "manifest.json") as f:
        manifest = json.load(f)
    run_id = manifest["run_id"]

    transition = f"{args.input_layer}_to_{args.output_layer}"
    output_dir = (
        Path(args.output).resolve() if args.output
        else REPO_ROOT / "reproduce" / "figures" / "audio"
             / "auristream_block_components" / run_id / transition
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "block_component_verification.json"
    if not args.overwrite and json_path.exists():
        print(f"[comp_diag] Output exists: {output_dir}\nPass --overwrite to regenerate.")
        return

    # Validate that required layers are in the manifest
    available = set(manifest["run_params"]["layers"])
    for lyr in (args.input_layer, args.output_layer):
        if lyr not in available:
            print(f"[comp_diag] Layer {lyr!r} not in run manifest. "
                  f"Available: {sorted(available)}")
            sys.exit(1)

    item_ids = list(manifest["items"].keys())[: args.max_items]
    print(f"[comp_diag] Transition   : {transition}")
    print(f"[comp_diag] Block index  : {args.block_index}  (transformer.h[{args.block_index}])")
    print(f"[comp_diag] max_items    : {args.max_items}  →  n_items_used = {len(item_ids)}")
    print(f"[comp_diag] Run          : {run_id}")

    # ── Load model ─────────────────────────────────────────────────────────────
    print("\n[comp_diag] Loading AuriStream model …")
    importlib.import_module("swp.audio.models.auristream")
    from swp.audio.models.registry import get_model
    model = get_model("auristream")
    device = model.device

    n_blocks = len(model._lm.transformer.h)
    if args.block_index >= n_blocks or args.block_index < 0:
        print(f"[comp_diag] block_index={args.block_index} out of range (0…{n_blocks-1})")
        sys.exit(1)

    block      = model._lm.transformer.h[args.block_index].eval()
    ln_f       = model._lm.transformer.ln_f.eval()
    attn_scale = float(getattr(block, "attn_scale", 1.0))
    is_last    = (args.block_index == n_blocks - 1)
    lnf_layer  = f"{args.output_layer}_lnf"
    verify_lnf = is_last and (lnf_layer in available)

    print(f"[comp_diag] attn_scale   : {attn_scale}")
    print(f"[comp_diag] is_last_block: {is_last}  verify_lnf={verify_lnf}")
    del model

    # ── Load phoneme metadata (for pooled diagnostics) ─────────────────────────
    meta_loaded: pd.DataFrame | None = None
    if (emb_dir / "metadata_phonemes.csv").exists():
        meta_loaded = pd.read_csv(emb_dir / "metadata_phonemes.csv")
        meta_loaded["phoneme_base"] = (
            meta_loaded["phoneme"].str.replace(r"\d+$", "", regex=True).str.upper()
        )
        meta_loaded["phoneme_type"] = meta_loaded["phoneme_base"].map(_phoneme_type)

    # ── Per-item loop ──────────────────────────────────────────────────────────
    verifications:  list[dict] = []
    tok_attn:  list[np.ndarray] = []
    tok_mlp:   list[np.ndarray] = []
    tok_delta: list[np.ndarray] = []
    tok_in:    list[np.ndarray] = []
    tok_out:   list[np.ndarray] = []
    tok_pos:   list[np.ndarray] = []
    pooled_rows: list[dict] = []
    items_processed = 0

    for item_id in item_ids:
        paths = manifest["items"][item_id]
        print(f"\n  {item_id}")

        h_in  = torch.load(run_dir / paths[args.input_layer],  map_location="cpu",
                           weights_only=True).float()
        h_out = torch.load(run_dir / paths[args.output_layer], map_location="cpu",
                           weights_only=True).float()
        L = h_in.shape[1]
        print(f"    D={h_in.shape[0]}, L={L}")

        comps = decompose_block(block, h_in, device)

        # Reconstruction: output ≈ input + attn_update + mlp_update
        delta_recomp = h_in + comps["attn_update"] + comps["mlp_update"]
        v1 = verify_recon(comps["output_recomp"], h_out,        "output_recomp vs saved")
        v2 = verify_recon(delta_recomp,           h_out,        "in+attn+mlp vs saved")
        verifications += [v1, v2]

        # Optional ln_f verification for last block
        if verify_lnf:
            h_lnf = torch.load(run_dir / paths[lnf_layer], map_location="cpu",
                               weights_only=True).float()
            with torch.inference_mode():
                lnf_r = ln_f(comps["output_recomp"].T.unsqueeze(0).to(device))
            lnf_r_cpu = lnf_r.squeeze(0).T.float().cpu()
            v3 = verify_recon(lnf_r_cpu, h_lnf, f"ln_f(output) vs {lnf_layer}")
            verifications.append(v3)

        if not v1["pass"] or not v2["pass"]:
            print("    ✗ Reconstruction failed — stopping.")
            break

        # Accumulate token-level [L, D]
        attn_arr  = comps["attn_update"].T.numpy().astype(np.float32)
        mlp_arr   = comps["mlp_update"].T.numpy().astype(np.float32)
        delta_arr = (h_out - h_in).T.numpy().astype(np.float32)
        in_arr    = h_in.T.numpy().astype(np.float32)
        out_arr   = h_out.T.numpy().astype(np.float32)

        tok_attn.append(attn_arr)
        tok_mlp.append(mlp_arr)
        tok_delta.append(delta_arr)
        tok_in.append(in_arr)
        tok_out.append(out_arr)
        tok_pos.append(np.arange(L, dtype=np.float32))

        if meta_loaded is not None:
            pooled_rows.extend(
                pool_components_by_phoneme(
                    item_id, meta_loaded,
                    {"attn_update": attn_arr, "mlp_update": mlp_arr,
                     "delta_total": delta_arr,
                     args.input_layer:  in_arr,
                     args.output_layer: out_arr},
                )
            )
        items_processed += 1

    # ── Verification report ────────────────────────────────────────────────────
    all_pass = all(v["pass"] for v in verifications if v["pass"] is not None)
    report: dict = {
        "run_id":          run_id,
        "transition":      transition,
        "block_index":     args.block_index,
        "input_layer":     args.input_layer,
        "output_layer":    args.output_layer,
        "attn_scale":      attn_scale,
        "max_items_arg":   args.max_items,
        "n_items_used":    items_processed,
        "items_used":      item_ids[:items_processed],
        "overall_pass":    all_pass,
        "verifications":   verifications,
    }
    json_path.write_text(json.dumps(report, indent=2))
    print(f"\n[comp_diag] Verification overall PASS: {all_pass}")

    if not all_pass:
        (output_dir / "block_component_verification.md").write_text(
            f"# Block component verification — {transition}\n\n"
            f"**Reconstruction FAILED** — see {json_path.name}\n"
        )
        print("[comp_diag] Reconstruction failed — stopping before analysis.")
        return

    # ── Concatenate token arrays ───────────────────────────────────────────────
    tok_data: dict[str, np.ndarray] = {
        "attn_update":      np.concatenate(tok_attn,  axis=0),
        "mlp_update":       np.concatenate(tok_mlp,   axis=0),
        "delta_total":      np.concatenate(tok_delta, axis=0),
        args.input_layer:   np.concatenate(tok_in,    axis=0),
        args.output_layer:  np.concatenate(tok_out,   axis=0),
    }
    pos_all = np.concatenate(tok_pos, axis=0)
    N = int(pos_all.shape[0])
    print(f"\n[comp_diag] Total tokens: {N}")

    # ── Token-level anisotropy (Task 4) ───────────────────────────────────────
    print("\n[comp_diag] Token-level anisotropy …")
    aniso_rows = []
    for name, arr in tok_data.items():
        m = _anisotropy_metrics(arr, rng)
        aniso_rows.append({"component": name, **m})
        print(f"  {name:24s}: iso={m['isotropy_ratio']:.5f}  "
              f"cos_mean={m['cos_with_mean_mean']:.5f}  "
              f"norm={m['norm_mean']:.3f}  pc1={m['pca_pc1_evr']:.4f}")

    aniso_df = pd.DataFrame(aniso_rows)
    aniso_df.to_csv(output_dir / "component_anisotropy_token_level.csv", index=False)

    # Cross-component alignment with delta_total
    print("\n[comp_diag] Alignment with delta_total:")
    norm_attn  = float(np.linalg.norm(tok_data["attn_update"], axis=1).mean())
    norm_mlp   = float(np.linalg.norm(tok_data["mlp_update"],  axis=1).mean())
    norm_delta = float(np.linalg.norm(tok_data["delta_total"], axis=1).mean())
    for name in ["attn_update", "mlp_update", args.input_layer]:
        cc = _cross_cosines(tok_data[name], tok_data["delta_total"])
        print(f"  cos({name}, delta_total): "
              f"per_token={cc['per_token_mean']:.5f}  "
              f"mean_vecs={cc['cos_mean_vecs']:.5f}")

    print(f"\n  Norm contributions (mean norms):")
    print(f"    attn / delta : {norm_attn  / (norm_delta + 1e-12):.4f}")
    print(f"    mlp  / delta : {norm_mlp   / (norm_delta + 1e-12):.4f}")

    report.update({
        "n_tokens":       N,
        "norm_attn_mean": norm_attn,
        "norm_mlp_mean":  norm_mlp,
        "norm_delta_mean":norm_delta,
        "attn_norm_ratio":norm_attn  / (norm_delta + 1e-12),
        "mlp_norm_ratio": norm_mlp   / (norm_delta + 1e-12),
        "cos_attn_delta_per_token": _cross_cosines(tok_data["attn_update"],
                                                   tok_data["delta_total"])["per_token_mean"],
        "cos_mlp_delta_per_token":  _cross_cosines(tok_data["mlp_update"],
                                                   tok_data["delta_total"])["per_token_mean"],
    })
    json_path.write_text(json.dumps(report, indent=2))

    # ── Pooled-level anisotropy (Task 5) ──────────────────────────────────────
    if pooled_rows:
        print(f"\n[comp_diag] Phoneme-pooled anisotropy ({len(pooled_rows)} phonemes) …")
        pooled_aniso = []
        for name in ["attn_update", "mlp_update", "delta_total",
                     args.input_layer, args.output_layer]:
            arr = np.stack([r[name] for r in pooled_rows if name in r], axis=0)
            m   = _anisotropy_metrics(arr, rng)
            pooled_aniso.append({"component": name, **m})
            print(f"  {name:24s}: iso={m['isotropy_ratio']:.5f}  "
                  f"cos_mean={m['cos_with_mean_mean']:.5f}  norm={m['norm_mean']:.3f}")
        pd.DataFrame(pooled_aniso).to_csv(
            output_dir / "component_anisotropy_pooled_level.csv", index=False
        )

    # ── Position-binned (Task 6) ───────────────────────────────────────────────
    print("\n[comp_diag] Position-binned metrics …")
    pos_df = _position_binned_metrics(
        {k: tok_data[k] for k in ["attn_update", "mlp_update", "delta_total"]},
        pos_all, rng, n_bins=args.n_pos_bins,
    )
    if not pos_df.empty:
        pos_df.to_csv(output_dir / "component_by_absolute_position.csv", index=False)
        print(pos_df[["position_bin", "n",
                       "attn_update_isotropy_ratio",
                       "mlp_update_isotropy_ratio",
                       "delta_total_isotropy_ratio"]].to_string(index=False))

    # ── Figures ────────────────────────────────────────────────────────────────
    suptitle_base = f"{transition}  ({items_processed} items, {N} tokens)"

    _plot_component_norms(
        {k: tok_data[k] for k in ["attn_update", "mlp_update", "delta_total",
                                   args.input_layer, args.output_layer]},
        f"Component norms  —  {suptitle_base}",
        output_dir / "summary_component_norms.png",
        args.input_layer, args.output_layer,
    )

    _plot_alignment_with_delta(
        {k: tok_data[k] for k in ["attn_update", "mlp_update",
                                   args.input_layer, "delta_total"]},
        "delta_total",
        f"Component alignment with delta_total  —  {suptitle_base}",
        output_dir / "summary_component_alignment_with_delta.png",
        args.input_layer, args.output_layer,
    )

    core_aniso = aniso_df[
        aniso_df["component"].isin(["attn_update", "mlp_update", "delta_total"])
    ].reset_index(drop=True)
    _plot_anisotropy(
        core_aniso,
        f"Component anisotropy  —  {suptitle_base}",
        output_dir / "summary_component_anisotropy.png",
        args.input_layer, args.output_layer,
    )

    if not pos_df.empty:
        _plot_position_metrics(
            pos_df,
            f"Component isotropy by position  —  {suptitle_base}",
            output_dir / "summary_component_by_absolute_position.png",
            args.input_layer, args.output_layer,
        )

    # ── Markdown report ────────────────────────────────────────────────────────
    md = [
        f"# Block component report — {transition}",
        f"\n- run_id: `{run_id}`",
        f"- block_index: {args.block_index}  (transformer.h[{args.block_index}])",
        f"- input_layer: `{args.input_layer}`",
        f"- output_layer: `{args.output_layer}`",
        f"- max_items (arg): {args.max_items}",
        f"- n_items_used: {items_processed}",
        f"- n_tokens: {N}",
        f"- attn_scale: {attn_scale}",
        "",
        "## Reconstruction checks",
        f"Overall PASS: **{all_pass}**",
        "",
        "| label | max_abs | mean_abs | rel_err | pass |",
        "|-------|---------|----------|---------|------|",
    ]
    for v in verifications:
        if v["max_abs"] is None:
            md.append(f"| {v['label']} | N/A | N/A | N/A | ✗ |")
        else:
            md.append(
                f"| {v['label']} | {v['max_abs']:.2e} | {v['mean_abs']:.2e} | "
                f"{v['rel_err']:.2e} | {'✓' if v['pass'] else '✗'} |"
            )

    md += [
        "",
        "## Component anisotropy (token level)",
        "",
        "| component | n | isotropy_ratio | cos_with_mean | cos_pairwise | norm_mean | pca_pc1_evr |",
        "|-----------|---|---------------|---------------|--------------|-----------|-------------|",
    ]
    for _, r in aniso_df.iterrows():
        md.append(
            f"| {r['component']} | {r['n']} | {r['isotropy_ratio']:.5f} | "
            f"{r['cos_with_mean_mean']:.5f} | {r['cos_pairwise_mean']:.5f} | "
            f"{r['norm_mean']:.3f} | {r['pca_pc1_evr']:.4f} |"
        )

    md += [
        "",
        "## Relative norm contributions",
        f"- attn_update mean norm : {norm_attn:.3f}",
        f"- mlp_update  mean norm : {norm_mlp:.3f}",
        f"- delta_total mean norm : {norm_delta:.3f}",
        f"- attn / delta : **{norm_attn / (norm_delta + 1e-12):.4f}**",
        f"- mlp  / delta : **{norm_mlp  / (norm_delta + 1e-12):.4f}**",
        f"- cos(attn, delta) per-token mean : {report['cos_attn_delta_per_token']:.5f}",
        f"- cos(mlp,  delta) per-token mean : {report['cos_mlp_delta_per_token']:.5f}",
        "",
        "## Interpretation guide",
        "- **MLP dominates**: mlp_update has high isotropy_ratio + alignment with delta_total;",
        "  near-common update from MLP → output-preparation hypothesis for coch_head.",
        "- **Attention dominates**: attn_update has high isotropy_ratio + alignment.",
        "- **PCA PC1 EVR** is supplementary (PCA centers; high isotropy ≠ high PC1 EVR).",
    ]

    (output_dir / "block_component_verification.md").write_text("\n".join(md) + "\n")
    print(f"\n[comp_diag] Done — {output_dir}")


if __name__ == "__main__":
    main()
