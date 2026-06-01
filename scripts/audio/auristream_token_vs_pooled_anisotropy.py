#!/usr/bin/env python3
"""Compare anisotropy at token level vs phoneme-pooled level.

Determines whether the high anisotropy seen in block_47/block_48/block_48_lnf is:
  (a) already present at the individual token level, or
  (b) introduced or amplified by mean-pooling over phoneme intervals.

Does NOT re-run the model — reads saved token-level activation tensors from
the activations/ directory and phoneme-pooled tensors from phoneme_embeddings/.

Metrics (per layer, per level):
  - isotropy_ratio: ||mean_vector|| / mean(||xᵢ||)
  - cos_with_mean_mean: average cosine similarity with the mean direction
  - cos_pairs_mean: average pairwise cosine (random sample)
  - norm_mean / norm_std
  - pca_pc1_evr (centered PCA — supplementary)

Usage:
    python scripts/audio/auristream_token_vs_pooled_anisotropy.py \\
        --run reproduce/data/audio/auristream__6ee9aeb6 \\
        --layers block_47 block_48 block_48_lnf

Full run (all layers):
    python scripts/audio/auristream_token_vs_pooled_anisotropy.py \\
        --run reproduce/data/audio/auristream__6ee9aeb6

Output:
    reproduce/figures/audio/auristream_token_anisotropy/{run_id}/
        token_vs_pooled_anisotropy.csv
        summary_token_vs_pooled_anisotropy.png
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

_DPI = 150

_DEFAULT_LAYERS = [
    "embedding", "block_01", "block_12", "block_24",
    "block_36", "block_47", "block_48", "block_48_lnf",
]


def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def compute_anisotropy_metrics(
    emb_raw: np.ndarray,
    rng: np.random.Generator,
    n_cos: int = 500,
    n_eu: int = 100,
) -> dict:
    """Compute isotropy_ratio, cos_with_mean, pca_pc1_evr, norms, Euclidean."""
    n, d = emb_raw.shape
    norms   = np.linalg.norm(emb_raw, axis=1)
    emb_n   = (emb_raw / (norms[:, None] + 1e-12)).astype(np.float32)

    mean_vec      = emb_raw.mean(axis=0)
    mean_vec_norm = float(np.linalg.norm(mean_vec))
    isotropy_ratio = mean_vec_norm / (float(norms.mean()) + 1e-12)
    mean_dir = (mean_vec / (mean_vec_norm + 1e-12)).astype(np.float32)
    cos_mean = emb_n @ mean_dir

    n_cos_  = min(n_cos, n)
    idx     = rng.choice(n, n_cos_, replace=False)
    cos_mat = emb_n[idx] @ emb_n[idx].T
    iu      = np.triu_indices(n_cos_, k=1)
    cos_pairs = cos_mat[iu]

    n_eu_  = min(n_eu, n)
    eu_idx = rng.choice(n, n_eu_, replace=False)
    eu_mat = emb_raw[eu_idx].astype(np.float64)
    sq_n   = (eu_mat ** 2).sum(axis=1)
    dot    = eu_mat @ eu_mat.T
    dist_sq = np.clip(sq_n[:, None] + sq_n[None, :] - 2.0 * dot, 0.0, None)
    eu_iu  = np.triu_indices(n_eu_, k=1)
    eu_d   = np.sqrt(dist_sq[eu_iu])

    n_comp = min(10, d, n)
    pca    = PCA(n_components=n_comp)
    pca.fit(emb_raw)
    evr    = pca.explained_variance_ratio_

    return {
        "n": n, "d": d,
        "norm_mean": float(norms.mean()), "norm_std": float(norms.std()),
        "isotropy_ratio": float(isotropy_ratio),
        "cos_with_mean_mean": float(cos_mean.mean()),
        "cos_with_mean_std":  float(cos_mean.std()),
        "cos_pairs_mean": float(cos_pairs.mean()),
        "cos_pairs_q05":  float(np.percentile(cos_pairs, 5)),
        "cos_pairs_q95":  float(np.percentile(cos_pairs, 95)),
        "eu_dist_mean":   float(eu_d.mean()),
        "pca_pc1_evr":    float(evr[0]),
        "pca_top3_evr":   float(sum(evr[:3])),
    }


def load_token_level(
    run_dir: Path,
    manifest: dict,
    layer: str,
    max_items: int | None = None,
) -> np.ndarray | None:
    """Load and concatenate all token-level [D, L] activations for one layer.

    Returns concatenated [N_total_tokens, D] float32 array, or None if not found.
    """
    items = list(manifest["items"].items())
    if max_items is not None:
        items = items[:max_items]

    arrays = []
    for item_id, paths in items:
        if layer not in paths:
            print(f"  [token] {item_id}: layer {layer!r} not in manifest — skipping item")
            continue
        p = run_dir / paths[layer]
        if not p.exists():
            print(f"  [token] {p} not found — skipping item")
            continue
        act = torch.load(p, map_location="cpu", weights_only=True).float()  # [D, L]
        arrays.append(act.T.numpy())  # [L, D]

    if not arrays:
        return None
    return np.concatenate(arrays, axis=0).astype(np.float32)  # [N_total, D]


def load_pooled_level(
    emb_dir: Path,
    layer: str,
    meta: pd.DataFrame,
) -> np.ndarray | None:
    """Load phoneme-pooled embeddings for one layer, apply validity filter.

    Returns [n_valid, D] float32, or None if file not found.
    """
    pt_path = emb_dir / f"embeddings_{layer}.pt"
    if not pt_path.exists():
        return None
    emb = torch.load(pt_path, map_location="cpu", weights_only=True).float()
    keep = (
        meta["valid_embedding"].astype(bool)
        & ~meta["is_silence"].astype(bool)
        & torch.isfinite(emb).all(dim=1).numpy()
    )
    return emb[keep].numpy().astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", required=True,
                   help="Extraction run directory.")
    p.add_argument("--layers", nargs="+", default=None,
                   help="Layers to analyze (default: all in manifest).")
    p.add_argument("--max-items", type=int, default=None, dest="max_items",
                   help="Limit number of items loaded for token-level (default: all).")
    p.add_argument("--random-state", type=int, default=0, dest="random_state")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--output", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = (REPO_ROOT / args.run).resolve()
    emb_dir = run_dir / "phoneme_embeddings"

    with open(run_dir / "manifest.json") as f:
        manifest = json.load(f)
    run_id = manifest["run_id"]

    output_dir = (
        Path(args.output).resolve() if args.output
        else REPO_ROOT / "reproduce" / "figures" / "audio"
             / "auristream_token_anisotropy" / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.overwrite and (output_dir / "token_vs_pooled_anisotropy.csv").exists():
        print(f"[token_aniso] Output exists: {output_dir}\nPass --overwrite to regenerate.")
        return

    layers = args.layers or list(manifest["run_params"]["layers"])
    rng = np.random.default_rng(args.random_state)

    # Load phoneme metadata (needed for pooled filtering)
    meta_path = emb_dir / "metadata_phonemes.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
    else:
        print("[token_aniso] metadata_phonemes.csv not found — pooled level unavailable")
        meta = None

    rows = []
    print(f"[token_aniso] Analyzing {len(layers)} layers …")

    for layer in layers:
        print(f"\n  {layer}")

        # Token level
        token_emb = load_token_level(run_dir, manifest, layer, args.max_items)
        if token_emb is not None:
            print(f"    token-level shape: {token_emb.shape}")
            t_metrics = compute_anisotropy_metrics(token_emb, rng)
            rows.append({"layer": layer, "level": "token", **t_metrics})
            print(f"    isotropy_ratio={t_metrics['isotropy_ratio']:.4f}  "
                  f"cos_with_mean={t_metrics['cos_with_mean_mean']:.4f}  "
                  f"cos_pairs_mean={t_metrics['cos_pairs_mean']:.4f}")
        else:
            print(f"    [warning] token-level data unavailable for {layer}")

        # Pooled level
        if meta is not None:
            pooled_emb = load_pooled_level(emb_dir, layer, meta)
            if pooled_emb is not None:
                print(f"    pooled shape    : {pooled_emb.shape}")
                p_metrics = compute_anisotropy_metrics(pooled_emb, rng)
                rows.append({"layer": layer, "level": "pooled", **p_metrics})
                print(f"    isotropy_ratio={p_metrics['isotropy_ratio']:.4f}  "
                      f"cos_with_mean={p_metrics['cos_with_mean_mean']:.4f}  "
                      f"cos_pairs_mean={p_metrics['cos_pairs_mean']:.4f}")
            else:
                print(f"    [warning] pooled embeddings not found for {layer}")

    if not rows:
        print("[token_aniso] No data collected — check paths and layer names.")
        return

    df = pd.DataFrame(rows)
    csv_path = output_dir / "token_vs_pooled_anisotropy.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[token_aniso] Saved {csv_path}")

    # ── Figure: isotropy_ratio and cos_with_mean_mean, token vs pooled ────────
    present_layers = df["layer"].unique().tolist()
    n_la = len(present_layers)

    fig, axes = plt.subplots(1, 2, figsize=(max(8, 3 * n_la), 5))
    x = np.arange(n_la)
    width = 0.35

    for col, ax, title in zip(
        ["isotropy_ratio", "cos_with_mean_mean"],
        axes,
        ["Isotropy ratio\n(0=isotropic, 1=collapsed)", "Cos with mean direction"],
    ):
        for offset, level, color in [(-width/2, "token", "#4575b4"), (width/2, "pooled", "#d73027")]:
            sub = df[df["level"] == level].set_index("layer")
            vals = [sub.loc[la, col] if la in sub.index else float("nan") for la in present_layers]
            ax.bar(x + offset, vals, width, label=level, color=color, alpha=0.8, edgecolor="none")

        ax.set_xticks(x)
        ax.set_xticklabels(present_layers, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1.07)
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)

    fig.suptitle("Token-level vs phoneme-pooled anisotropy", fontsize=11)
    fig.tight_layout()
    fig_path = output_dir / "summary_token_vs_pooled_anisotropy.png"
    _savefig(fig, fig_path)
    print(f"[token_aniso] Saved {fig_path}")
    print(f"\n[token_aniso] Done — {output_dir}")


if __name__ == "__main__":
    main()
