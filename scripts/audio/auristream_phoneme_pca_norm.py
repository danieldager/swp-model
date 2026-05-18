#!/usr/bin/env python3
"""PCA and norm analysis of phoneme-level AuriStream embeddings.

Reads [n_phonemes, D] embedding tensors and aligned metadata produced by
scripts/audio/build_phoneme_embeddings.py, then runs:

  Analysis A — PCA:
    StandardScaler + PCA(n_components) per layer.
    Scatter plots coloured by phoneme position, lexicality, and phone label.

  Analysis B — Norm as a function of phoneme position:
    L2 norm per phoneme embedding.
    Mean ± SEM curves by position_from_start / position_from_end,
    optionally split by lexicality and length_bin.

This script is currently tailored to AuriStream phoneme embeddings but is
written to be straightforwardly adaptable to phoneme embeddings from other
representation-only speech models (HuBERT, WavLM, etc.) that produce the
same output format from build_phoneme_embeddings.py.

Run from swp-model/ repo root:

    python scripts/audio/auristream_phoneme_pca_norm.py \\
        --embeddings-dir reproduce/data/audio/auristream__9d3f269f/phoneme_embeddings/ \\
        --dataset        data/external/paradigm/processed/subset_male.csv

Smoke test (two layers):

    python scripts/audio/auristream_phoneme_pca_norm.py \\
        --embeddings-dir reproduce/data/audio/auristream__9d3f269f/phoneme_embeddings/ \\
        --dataset        data/external/paradigm/processed/subset_male.csv \\
        --layers         embedding block_48

Full run (all 5 layers):

    python scripts/audio/auristream_phoneme_pca_norm.py \\
        --embeddings-dir reproduce/data/audio/auristream__9d3f269f/phoneme_embeddings/ \\
        --dataset        data/external/paradigm/processed/subset_male.csv \\
        --layers         embedding block_12 block_24 block_36 block_48
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Columns to pull from the dataset CSV when merging
_DATASET_COLS = [
    "item_id", "lexicality", "length_bin", "morphology",
    "frequency_bin", "condition", "speaker",
]

# Visual constants (consistent with existing codec PCA scripts)
_LEX_COLORS: dict[str, str] = {"word": "#0072B2", "nonword": "#ED2727"}
_LEN_COLORS: dict[str, str] = {"short": "#009E73", "long": "#E69F00"}


def _display_lex(lex: str) -> str:
    """Map raw CSV value to display label: 'nonword' → 'pseudoword'."""
    return "pseudoword" if lex == "nonword" else lex


def _layer_sort_key(name: str) -> tuple:
    """Natural sort key: embedding < block_01 < block_12 < …"""
    if name == "embedding":
        return (0, 0)
    m = re.match(r"block_(\d+)$", name)
    if m:
        return (1, int(m.group(1)))
    return (2, name)


# ── Data loading ──────────────────────────────────────────────────────────────


def load_data(
    embeddings_dir: Path,
    dataset_path: Path,
    layers: list[str],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Load embeddings and metadata, filter, and merge with dataset CSV.

    Returns:
        meta_df:    filtered metadata DataFrame, one row per valid phoneme
        embeddings: dict[layer → np.ndarray [n_valid, D]]
    """
    meta_df = pd.read_csv(embeddings_dir / "metadata_phonemes.csv")

    # Load embeddings for all requested layers to build a joint validity mask
    raw: dict[str, np.ndarray] = {}
    for la in layers:
        t = torch.load(
            embeddings_dir / f"embeddings_{la}.pt",
            map_location="cpu",
            weights_only=True,
        )
        raw[la] = t.numpy()  # [n_phonemes, D]

    n = len(meta_df)
    assert all(v.shape[0] == n for v in raw.values()), \
        "Embedding tensor row count does not match metadata_phonemes.csv length."

    # Build validity mask: keep valid, non-silence, fully-finite rows
    mask = meta_df["valid_embedding"].astype(bool) & ~meta_df["is_silence"].astype(bool)
    for la, arr in raw.items():
        mask &= np.isfinite(arr).all(axis=1)

    meta_df = meta_df[mask].reset_index(drop=True)
    embeddings: dict[str, np.ndarray] = {la: raw[la][mask] for la in layers}

    # Merge paradigm factors from the dataset CSV
    ds = pd.read_csv(dataset_path)
    available_cols = [c for c in _DATASET_COLS if c in ds.columns and c != "item_id"]
    ds_subset = ds[["item_id"] + available_cols].drop_duplicates("item_id")
    meta_df = meta_df.merge(ds_subset, on="item_id", how="left")

    return meta_df, embeddings


# ── PCA ───────────────────────────────────────────────────────────────────────


def run_pca(
    X: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, list[float], PCA]:
    """Standardize X and fit PCA. Returns (scores, explained_variance_ratio, pca)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=0)
    scores = pca.fit_transform(X_scaled)
    return scores, pca.explained_variance_ratio_.tolist(), pca


def _scatter_base(ax: plt.Axes, pc1: np.ndarray, pc2: np.ndarray, ev: list[float]) -> None:
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var.)")
    ax.axhline(0, color="lightgray", lw=0.5, zorder=0)
    ax.axvline(0, color="lightgray", lw=0.5, zorder=0)


def plot_pca_by_position(
    scores: np.ndarray,
    meta: pd.DataFrame,
    ev: list[float],
    layer: str,
    out_path: Path,
) -> None:
    """PCA scatter coloured by phoneme_position_from_start."""
    fig, ax = plt.subplots(figsize=(6, 5))
    pos = meta["phoneme_position_from_start"].values
    sc = ax.scatter(scores[:, 0], scores[:, 1], c=pos, cmap="viridis", s=6, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="phoneme position (from start)")
    _scatter_base(ax, scores[:, 0], scores[:, 1], ev)
    ax.set_title(f"PCA by phoneme position — {layer}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_pca_by_lexicality(
    scores: np.ndarray,
    meta: pd.DataFrame,
    ev: list[float],
    layer: str,
    out_path: Path,
) -> None:
    """PCA scatter coloured by lexicality (word / pseudoword)."""
    if "lexicality" not in meta.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    for lex, color in _LEX_COLORS.items():
        sel = meta["lexicality"] == lex
        if not sel.any():
            continue
        ax.scatter(
            scores[sel, 0], scores[sel, 1],
            c=color, label=_display_lex(lex), s=6, alpha=0.6,
        )
    _scatter_base(ax, scores[:, 0], scores[:, 1], ev)
    ax.legend(markerscale=2)
    ax.set_title(f"PCA by lexicality — {layer}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_pca_by_phone(
    scores: np.ndarray,
    meta: pd.DataFrame,
    ev: list[float],
    layer: str,
    out_path: Path,
    top_n: int,
) -> None:
    """PCA scatter coloured by phone label (top N, rest as 'other')."""
    phones = meta["phoneme"].values
    top_phones = (
        pd.Series(phones).value_counts().head(top_n).index.tolist()
    )
    phone_labels = np.where(np.isin(phones, top_phones), phones, "other")

    all_labels = top_phones + (["other"] if "other" in phone_labels else [])
    cmap = matplotlib.colormaps["tab20"].resampled(len(all_labels))
    color_map = {lbl: cmap(i) for i, lbl in enumerate(all_labels)}
    if "other" in color_map:
        color_map["other"] = (0.7, 0.7, 0.7, 1.0)  # neutral gray

    fig, ax = plt.subplots(figsize=(7, 5))
    for lbl in all_labels:
        sel = phone_labels == lbl
        ax.scatter(
            scores[sel, 0], scores[sel, 1],
            c=[color_map[lbl]], label=lbl, s=6, alpha=0.7,
        )
    _scatter_base(ax, scores[:, 0], scores[:, 1], ev)
    ax.legend(markerscale=2, fontsize=7, ncol=2, loc="upper right")
    ax.set_title(f"PCA by phone label (top {top_n}) — {layer}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ── Norm ─────────────────────────────────────────────────────────────────────


def compute_norms(X: np.ndarray) -> np.ndarray:
    """L2 norm of each row of X. Returns [n] float64."""
    return np.linalg.norm(X.astype(np.float64), axis=1)


def _agg_by_position(df: pd.DataFrame, pos_col: str) -> pd.DataFrame:
    """Group by pos_col and compute mean, std, SEM, count of 'norm'."""
    g = df.groupby(pos_col)["norm"]
    agg = pd.DataFrame({
        "position": g.mean().index,
        "mean": g.mean().values,
        "std": g.std().values,
        "sem": (g.std() / g.count().apply(np.sqrt)).values,
        "n": g.count().values,
    })
    return agg.reset_index(drop=True)


def _agg_by_position_factor(
    df: pd.DataFrame,
    pos_col: str,
    factor_col: str,
) -> pd.DataFrame:
    """Group by (pos_col, factor_col) and compute mean, SEM of 'norm'."""
    g = df.groupby([pos_col, factor_col])["norm"]
    agg = g.agg(["mean", "std", "count"]).reset_index()
    agg["sem"] = agg["std"] / agg["count"].apply(np.sqrt)
    agg = agg.rename(columns={"count": "n", pos_col: "position", factor_col: "factor"})
    return agg


def _plot_norm_curve(
    agg: pd.DataFrame,
    pos_col_label: str,
    title: str,
    out_path: Path,
    factor_col: str | None = None,
    factor_colors: dict | None = None,
    display_fn=None,
) -> None:
    """Plot mean norm ± SEM as a function of position, optionally split by factor."""
    fig, ax = plt.subplots(figsize=(6, 4))
    if factor_col is None:
        ax.plot(agg["position"], agg["mean"], marker="o", ms=4, lw=1.5, color="#333333")
        ax.fill_between(
            agg["position"],
            agg["mean"] - agg["sem"],
            agg["mean"] + agg["sem"],
            alpha=0.25, color="#333333",
        )
    else:
        for factor_val, group in agg.groupby("factor"):
            label = display_fn(str(factor_val)) if display_fn else str(factor_val)
            color = (factor_colors or {}).get(str(factor_val), None)
            line = ax.plot(
                group["position"], group["mean"],
                marker="o", ms=4, lw=1.5, label=label,
                color=color,
            )[0]
            ax.fill_between(
                group["position"],
                group["mean"] - group["sem"],
                group["mean"] + group["sem"],
                alpha=0.2,
                color=line.get_color(),
            )
        ax.legend()
    ax.set_xlabel(pos_col_label)
    ax.set_ylabel("L2 norm (mean ± SEM)")
    ax.set_title(title)
    ax.set_xticks(sorted(agg["position"].unique()))
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ── Per-layer pipeline ────────────────────────────────────────────────────────


def process_layer(
    layer: str,
    X: np.ndarray,
    meta: pd.DataFrame,
    output_dir: Path,
    n_components: int,
    top_phones: int,
) -> dict:
    """Run PCA + norm analyses for one layer. Returns dict of created files."""
    layer_dir = output_dir / layer
    layer_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    ev_ratio: list[float]

    # ── PCA ──────────────────────────────────────────────────────────────────
    scores, ev_ratio, _ = run_pca(X, n_components)

    # Scores CSV
    scores_df = meta.copy()
    for k in range(n_components):
        scores_df[f"pc{k+1}"] = scores[:, k]
    scores_csv = layer_dir / f"phoneme_pca_scores_{layer}.csv"
    scores_df.to_csv(scores_csv, index=False)
    created.append(str(scores_csv.relative_to(output_dir)))

    # PCA summary JSON
    summary = {
        "layer": layer,
        "explained_variance_ratio": ev_ratio,
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_components": n_components,
    }
    summary_json = layer_dir / f"phoneme_pca_summary_{layer}.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    created.append(str(summary_json.relative_to(output_dir)))
    print(
        f"  PCA: PC1={ev_ratio[0]*100:.1f}%  PC2={ev_ratio[1]*100:.1f}%  "
        f"n={X.shape[0]}"
    )

    # PCA plots
    p = layer_dir / f"pca_{layer}_by_position.png"
    plot_pca_by_position(scores, meta, ev_ratio, layer, p)
    created.append(str(p.relative_to(output_dir)))

    p = layer_dir / f"pca_{layer}_by_lexicality.png"
    plot_pca_by_lexicality(scores, meta, ev_ratio, layer, p)
    if "lexicality" in meta.columns:
        created.append(str(p.relative_to(output_dir)))

    p = layer_dir / f"pca_{layer}_by_phone.png"
    plot_pca_by_phone(scores, meta, ev_ratio, layer, p, top_n=top_phones)
    created.append(str(p.relative_to(output_dir)))

    # ── Norms ────────────────────────────────────────────────────────────────
    norms = compute_norms(X)
    norms_df = meta.copy()
    norms_df["norm"] = norms

    norms_csv = layer_dir / f"phoneme_norms_{layer}.csv"
    norms_df.to_csv(norms_csv, index=False)
    created.append(str(norms_csv.relative_to(output_dir)))

    # Aggregate by position
    agg_start = _agg_by_position(norms_df, "phoneme_position_from_start")
    agg_end = _agg_by_position(norms_df, "phoneme_position_from_end")

    agg_csv = layer_dir / f"norm_by_position_{layer}.csv"
    agg_all = pd.concat(
        [agg_start.assign(pos_axis="from_start"), agg_end.assign(pos_axis="from_end")],
        ignore_index=True,
    )
    agg_all.to_csv(agg_csv, index=False)
    created.append(str(agg_csv.relative_to(output_dir)))

    # Norm plots — by position from start/end
    p = layer_dir / f"norm_by_position_from_start_{layer}.png"
    _plot_norm_curve(
        agg_start, "phoneme position (from start)", f"Norm by position (start) — {layer}", p
    )
    created.append(str(p.relative_to(output_dir)))

    p = layer_dir / f"norm_by_position_from_end_{layer}.png"
    _plot_norm_curve(
        agg_end, "phoneme position (from end, 0=last)", f"Norm by position (end) — {layer}", p
    )
    created.append(str(p.relative_to(output_dir)))

    # Norm plot — by lexicality (if available)
    if "lexicality" in norms_df.columns and norms_df["lexicality"].notna().any():
        agg_lex = _agg_by_position_factor(
            norms_df.dropna(subset=["lexicality"]),
            "phoneme_position_from_start",
            "lexicality",
        )
        lex_csv = layer_dir / f"norm_by_position_lexicality_{layer}.csv"
        agg_lex.to_csv(lex_csv, index=False)
        created.append(str(lex_csv.relative_to(output_dir)))

        p = layer_dir / f"norm_by_position_lexicality_{layer}.png"
        _plot_norm_curve(
            agg_lex,
            "phoneme position (from start)",
            f"Norm by position × lexicality — {layer}",
            p,
            factor_col="lexicality",
            factor_colors=_LEX_COLORS,
            display_fn=_display_lex,
        )
        created.append(str(p.relative_to(output_dir)))

    # Norm plot — by length_bin (if available)
    if "length_bin" in norms_df.columns and norms_df["length_bin"].notna().any():
        agg_len = _agg_by_position_factor(
            norms_df.dropna(subset=["length_bin"]),
            "phoneme_position_from_start",
            "length_bin",
        )
        len_csv = layer_dir / f"norm_by_position_length_{layer}.csv"
        agg_len.to_csv(len_csv, index=False)
        created.append(str(len_csv.relative_to(output_dir)))

        p = layer_dir / f"norm_by_position_length_{layer}.png"
        _plot_norm_curve(
            agg_len,
            "phoneme position (from start)",
            f"Norm by position × length — {layer}",
            p,
            factor_col="length_bin",
            factor_colors=_LEN_COLORS,
        )
        created.append(str(p.relative_to(output_dir)))

    return {"layer": layer, "files": created}


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PCA and norm analysis of phoneme-level AuriStream embeddings."
    )
    parser.add_argument(
        "--embeddings-dir", required=True, dest="embeddings_dir",
        help=(
            "Path to phoneme embeddings directory "
            "(e.g. reproduce/data/audio/auristream__9d3f269f/phoneme_embeddings/)"
        ),
    )
    parser.add_argument(
        "--dataset", required=True,
        help=(
            "Processed paradigm CSV for merging lexicality, length_bin, etc. "
            "(e.g. data/external/paradigm/processed/subset_male.csv)"
        ),
    )
    parser.add_argument(
        "--layers", nargs="+", default=None,
        help=(
            "Layers to analyse. Default: all embeddings_*.pt files found in "
            "--embeddings-dir."
        ),
    )
    parser.add_argument(
        "--output", default=None,
        help=(
            "Output directory. "
            "Default: reproduce/figures/audio/auristream_phonemes/{source_run_id}/"
        ),
    )
    parser.add_argument(
        "--n-components", type=int, default=2, dest="n_components",
        help="Number of PCA components (default: 2).",
    )
    parser.add_argument(
        "--top-phones", type=int, default=15, dest="top_phones",
        help="Number of most-frequent phone labels to colour individually (default: 15).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite an existing output directory.",
    )
    args = parser.parse_args()

    embeddings_dir = Path(args.embeddings_dir)
    dataset_path = Path(args.dataset)

    if not (embeddings_dir / "metadata_phonemes.csv").exists():
        sys.exit(f"metadata_phonemes.csv not found in: {embeddings_dir}")
    if not dataset_path.exists():
        sys.exit(f"Dataset CSV not found: {dataset_path}")

    # Load phoneme embeddings manifest for provenance
    emb_manifest: dict = {}
    emb_manifest_path = embeddings_dir / "manifest.json"
    if emb_manifest_path.exists():
        with open(emb_manifest_path) as f:
            emb_manifest = json.load(f)
    source_run_id: str = emb_manifest.get("source_run_id", "unknown")

    # Determine layers (natural sort order)
    if args.layers is not None:
        layers = sorted(args.layers, key=_layer_sort_key)
    else:
        found = sorted(
            [p.stem.removeprefix("embeddings_") for p in embeddings_dir.glob("embeddings_*.pt")],
            key=_layer_sort_key,
        )
        if not found:
            sys.exit(f"No embeddings_*.pt files found in {embeddings_dir}")
        layers = found

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = (
            REPO_ROOT
            / "reproduce" / "figures" / "audio" / "auristream_phonemes"
            / source_run_id
        )

    if output_dir.exists() and not args.overwrite:
        sys.exit(
            f"Output directory already exists: {output_dir}\n"
            "Pass --overwrite to overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Embeddings dir : {embeddings_dir}")
    print(f"Source run     : {source_run_id}")
    print(f"Dataset        : {dataset_path}")
    print(f"Layers         : {layers}")
    print(f"Output         : {output_dir}")

    # Load and filter data (done once; embeddings indexed per layer)
    print("\nLoading data …")
    meta_df, embeddings = load_data(embeddings_dir, dataset_path, layers)
    n_valid = len(meta_df)
    n_total = emb_manifest.get("n_phonemes_total", "?")
    print(f"  Valid phonemes: {n_valid} / {n_total}  (silence and invalid excluded)")

    # Per-layer analyses
    layer_results: list[dict] = []
    for layer in layers:
        print(f"\n── {layer} ──")
        result = process_layer(
            layer=layer,
            X=embeddings[layer],
            meta=meta_df,
            output_dir=output_dir,
            n_components=args.n_components,
            top_phones=args.top_phones,
        )
        layer_results.append(result)

    # Write manifest
    manifest_out = {
        "embeddings_dir": str(embeddings_dir.resolve()),
        "dataset": str(dataset_path.resolve()),
        "output": str(output_dir.resolve()),
        "layers": layers,
        "n_components": args.n_components,
        "top_phones": args.top_phones,
        "source_run_id": source_run_id,
        "token_selection_rule": emb_manifest.get("token_selection_rule", "unknown"),
        "n_phonemes_total": n_total,
        "n_valid_analyzed": n_valid,
        "layer_files": {r["layer"]: r["files"] for r in layer_results},
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest_out, f, indent=2)

    print(f"\nDone: {output_dir}")


if __name__ == "__main__":
    main()
