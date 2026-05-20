#!/usr/bin/env python3
"""Diagnostic checks for AuriStream embedding layer and block_48.

Four synthetic checks (always run):

  1. Same token, different positions  — hidden_states[0] must vary with position
  2. Algebraic identity              — hidden_states[0] == wte(token) + wpe(position)
  3. Norm profile                    — ||wpe(p)|| vs ||hidden_states[0][p]|| by position
  4. block_48 before vs after ln_f   — norm spread and cosine similarity

Optional phoneme-level diagnostic (--phoneme-diagnostics):

  For each valid phoneme in the real extraction run, tests whether the phoneme-level
  embedding norm is driven by ||wpe(position)|| rather than by acoustic token content.
  Produces a CSV, a JSON summary, and two plots.

Usage:
    python scripts/audio/auristream_embedding_sanity_checks.py
    python scripts/audio/auristream_embedding_sanity_checks.py --phoneme-diagnostics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

MODEL_ID = "TuKoResearch/AuriStream1B_librilight_ckpt500k"
TOKEN_ID = 42   # arbitrary token, repeated across the whole sequence
SEQ_LEN = 64    # short enough to be fast, long enough to see position effects

_ARPABET_VOWELS: frozenset[str] = frozenset({
    "AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY", "IH", "IY",
    "OW", "OY", "UH", "UW",
})


def load_model() -> torch.nn.Module:
    print(f"Loading {MODEL_ID} …")
    lm = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
    lm.eval()
    print(f"  use_rope={lm.config.use_rope}  dropout={lm.config.dropout}  n_layer={lm.config.n_layer}")
    return lm


# ---------------------------------------------------------------------------
# Check 1 — same token repeated: positions must differ in hidden_states[0]
# ---------------------------------------------------------------------------

def check_1_same_token_different_positions(lm: torch.nn.Module) -> None:
    print("\n=== Check 1: same token, different positions ===")
    seq = torch.full((1, SEQ_LEN), TOKEN_ID, dtype=torch.long)
    with torch.no_grad():
        out = lm(seq, output_hidden_states=True)
    h0 = out.hidden_states[0]  # (1, T, n_embd)
    diffs = (h0[0] - h0[0, 0:1]).norm(dim=-1)  # (T,)  dist from position 0
    print(f"  ||h0[p] - h0[0]|| for p=0..{SEQ_LEN-1}:")
    print(f"    p=0 (self): {diffs[0].item():.6f}")
    print(f"    p=1:        {diffs[1].item():.4f}")
    print(f"    min (p>0):  {diffs[1:].min().item():.4f}   max: {diffs[1:].max().item():.4f}")
    if (diffs[1:] < 1e-5).all():
        print("  [FAIL] All positions identical — wpe has no effect.")
    else:
        print("  [OK] Positions differ, confirming wpe contributes to hidden_states[0].")


# ---------------------------------------------------------------------------
# Check 2 — algebraic identity: hidden_states[0] == wte + wpe (no dropout)
# ---------------------------------------------------------------------------

def check_2_algebraic_identity(lm: torch.nn.Module) -> None:
    print("\n=== Check 2: algebraic identity hidden_states[0] == wte + wpe ===")
    seq = torch.full((1, SEQ_LEN), TOKEN_ID, dtype=torch.long)
    with torch.no_grad():
        out = lm(seq, output_hidden_states=True)
    h0 = out.hidden_states[0][0]  # (T, n_embd)
    wte_vec = lm.transformer.wte.weight[TOKEN_ID].detach()   # (n_embd,)
    max_err = 0.0
    for p in range(SEQ_LEN):
        wpe_vec = lm.transformer.wpe.weight[p].detach()     # (n_embd,)
        err = (h0[p] - (wte_vec + wpe_vec)).abs().max().item()
        max_err = max(max_err, err)
    print(f"  Max abs error over {SEQ_LEN} positions: {max_err:.2e}")
    if max_err < 1e-4:
        print("  [OK] hidden_states[0] == wte + wpe exactly (dropout=0.0 confirmed).")
    else:
        print("  [WARN] Non-trivial discrepancy — check dropout or model loading.")


# ---------------------------------------------------------------------------
# Check 3 — norm profile: ||wpe(p)|| vs ||hidden_states[0][p]|| by position
# ---------------------------------------------------------------------------

def check_3_wpe_norm_profile(lm: torch.nn.Module) -> None:
    print("\n=== Check 3: wpe norm vs hidden_states[0] norm by position ===")
    seq = torch.full((1, SEQ_LEN), TOKEN_ID, dtype=torch.long)
    with torch.no_grad():
        out = lm(seq, output_hidden_states=True)
    h0 = out.hidden_states[0][0]                                     # (T, n_embd)
    wpe_norms = lm.transformer.wpe.weight[:SEQ_LEN].detach().norm(dim=-1)  # (T,)
    wte_norm  = lm.transformer.wte.weight[TOKEN_ID].detach().norm().item()
    h0_norms  = h0.norm(dim=-1)                                      # (T,)
    corr = torch.corrcoef(torch.stack([wpe_norms, h0_norms]))[0, 1].item()
    print(f"  wte norm (single vector): {wte_norm:.3f}")
    print(f"  wpe norm — min: {wpe_norms.min():.3f}  max: {wpe_norms.max():.3f}  mean: {wpe_norms.mean():.3f}")
    print(f"  h0  norm — min: {h0_norms.min():.3f}  max: {h0_norms.max():.3f}  mean: {h0_norms.mean():.3f}")
    print(f"  Pearson(wpe_norm, h0_norm) over {SEQ_LEN} positions: {corr:.4f}")
    if abs(corr) > 0.8:
        print("  [INFO] Strong correlation: positional norm variation in the embedding layer is driven by wpe.")
    else:
        print("  [INFO] Weak correlation: wpe norm alone does not explain h0 norm variation.")


# ---------------------------------------------------------------------------
# Check 4 — block_48 before vs after ln_f: norm and direction
# ---------------------------------------------------------------------------

def check_4_block48_before_after_lnf(lm: torch.nn.Module) -> None:
    print("\n=== Check 4: block_48 (hidden_states[48]) before vs after ln_f ===")
    seq = torch.full((1, SEQ_LEN), TOKEN_ID, dtype=torch.long)
    with torch.no_grad():
        out = lm(seq, output_hidden_states=True)
        h48     = out.hidden_states[48]           # (1, T, n_embd) — before ln_f
        h48_lnf = lm.transformer.ln_f(h48)       # (1, T, n_embd) — after ln_f
    norms_raw = h48[0].norm(dim=-1)               # (T,)
    norms_lnf = h48_lnf[0].norm(dim=-1)          # (T,)
    cos_sim   = F.cosine_similarity(h48[0], h48_lnf[0], dim=-1)  # (T,)
    spread_raw = (norms_raw.max() - norms_raw.min()).item()
    spread_lnf = (norms_lnf.max() - norms_lnf.min()).item()
    print(f"  h48  norm — min: {norms_raw.min():.3f}  max: {norms_raw.max():.3f}  spread: {spread_raw:.3f}")
    print(f"  h48_lnf norm — min: {norms_lnf.min():.3f}  max: {norms_lnf.max():.3f}  spread: {spread_lnf:.3f}")
    print(f"  Cosine sim h48 vs h48_lnf — min: {cos_sim.min():.4f}  max: {cos_sim.max():.4f}  mean: {cos_sim.mean():.4f}")
    if spread_lnf < spread_raw * 0.1:
        print("  [INFO] ln_f strongly collapses norm variation across positions.")
    else:
        print("  [INFO] ln_f does not fully remove norm variation (RMSNorm scale weights may reintroduce it).")
    if cos_sim.mean() > 0.99:
        print("  [INFO] Directions nearly identical — ln_f preserves representational geometry.")
    else:
        print("  [INFO] Directions differ — ln_f has a non-trivial effect on geometry.")


# ---------------------------------------------------------------------------
# Phoneme-level diagnostic (--phoneme-diagnostics)
# ---------------------------------------------------------------------------

def _phoneme_type(base: str) -> str:
    if base in _ARPABET_VOWELS:
        return "vowel"
    return "consonant" if base.isalpha() else "other"


def run_phoneme_diagnostics(
    lm: torch.nn.Module,
    run_dir: Path,
    dataset_path: Path,
    output_dir: Path,
) -> None:
    """Test whether phoneme embedding norms in the 'embedding' layer are driven by wpe.

    For each valid non-silence phoneme, computes:
      embedding_norm    : ||mean_pool(h0)||
      mean_wpe_norm     : mean ||wpe[p]|| over token positions p inside the phoneme
      mean_wte_norm     : mean ||h0[:,p] - wpe[p]||  ≈ mean ||wte[token_id_p]||
                          (exact because dropout=0.0; derived from saved activations)
      mean_h0_tok_norm  : mean ||h0[:,p]|| over token positions p inside the phoneme
      mean_wte_wpe_cos  : mean cosine_similarity(wte[p], wpe[p]) per token, then averaged

    Saves a CSV, a JSON summary, and two diagnostic plots.
    """
    import numpy as np
    import pandas as pd
    from scipy import stats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    embeddings_dir = run_dir / "phoneme_embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Phoneme-level norm diagnostic ===")

    wpe_weight = lm.transformer.wpe.weight.detach().cpu().float()  # [4096, 1280]

    # Load run manifest for activation file paths
    with open(run_dir / "manifest.json") as f:
        manifest = json.load(f)

    # Load metadata + phoneme embeddings
    meta = pd.read_csv(embeddings_dir / "metadata_phonemes.csv")
    emb_raw = torch.load(
        embeddings_dir / "embeddings_embedding.pt",
        map_location="cpu", weights_only=True,
    ).float()  # [n_phonemes_all, 1280]
    assert len(meta) == emb_raw.shape[0]

    # Filter: valid, non-silence, finite
    keep = (
        meta["valid_embedding"].astype(bool)
        & ~meta["is_silence"].astype(bool)
        & torch.isfinite(emb_raw).all(dim=1).numpy()
    )
    meta = meta[keep].reset_index(drop=True)
    emb = emb_raw[keep.values]  # [n_valid, 1280]

    meta["phoneme_base"] = meta["phoneme"].str.replace(r"\d+$", "", regex=True).str.upper()
    meta["phoneme_type"] = meta["phoneme_base"].map(_phoneme_type)

    # Merge paradigm factors
    if dataset_path.exists():
        ds = pd.read_csv(dataset_path)
        merge_cols = [c for c in ["lexicality", "length_bin"] if c in ds.columns]
        if merge_cols:
            meta = meta.merge(
                ds[["item_id"] + merge_cols].drop_duplicates("item_id"),
                on="item_id", how="left",
            )

    n_valid = len(meta)
    print(f"  {n_valid} valid non-silence phonemes across {meta['item_id'].nunique()} items")

    # Per-phoneme norms
    # wte[token_id_p] is derived as h0[:,p] - wpe[p] (exact because dropout=0.0, confirmed in check 2)
    embedding_norm = emb.norm(dim=-1).numpy()
    mean_wpe_norm      = np.full(n_valid, np.nan, dtype=np.float32)
    mean_wte_norm      = np.full(n_valid, np.nan, dtype=np.float32)
    mean_h0_tok_norm   = np.full(n_valid, np.nan, dtype=np.float32)
    mean_wte_wpe_cos   = np.full(n_valid, np.nan, dtype=np.float32)

    print("  Loading token-level activations per item…")
    for item_id, grp in meta.groupby("item_id"):
        act_rel = manifest["items"][item_id]["embedding"]
        act = torch.load(
            run_dir / act_rel, map_location="cpu", weights_only=True
        ).float()  # [1280, L]
        for idx, row in grp.iterrows():
            tok_s, tok_e = int(row["token_start_idx"]), int(row["token_end_idx"])
            if tok_s < 0 or tok_e < tok_s:
                continue
            h0_tok  = act[:, tok_s : tok_e + 1]           # [1280, n_tok]
            wpe_tok = wpe_weight[tok_s : tok_e + 1]        # [n_tok, 1280]
            wte_tok = (h0_tok - wpe_tok.T)                 # [1280, n_tok]  ≈ wte[token_id]

            mean_wpe_norm[idx]    = wpe_tok.norm(dim=-1).mean().item()
            mean_h0_tok_norm[idx] = h0_tok.norm(dim=0).mean().item()
            mean_wte_norm[idx]    = wte_tok.norm(dim=0).mean().item()
            # cosine similarity between wte and wpe vectors, token-by-token, then averaged
            mean_wte_wpe_cos[idx] = F.cosine_similarity(wte_tok.T, wpe_tok, dim=-1).mean().item()

    meta["embedding_norm"]    = embedding_norm
    meta["mean_wpe_norm"]     = mean_wpe_norm
    meta["mean_wte_norm"]     = mean_wte_norm
    meta["mean_h0_tok_norm"]  = mean_h0_tok_norm
    meta["mean_wte_wpe_cos"]  = mean_wte_wpe_cos
    meta = meta.dropna(subset=["mean_wpe_norm"]).reset_index(drop=True)

    # Save CSV
    out_csv = output_dir / "embedding_position_norm_diagnostics.csv"
    meta.to_csv(out_csv, index=False)
    print(f"  Saved {out_csv}  ({len(meta)} rows)")

    # Correlations
    r_wpe,  p_wpe  = stats.pearsonr(meta["embedding_norm"], meta["mean_wpe_norm"])
    rho_wpe,  _    = stats.spearmanr(meta["embedding_norm"], meta["mean_wpe_norm"])
    r_wte,  p_wte  = stats.pearsonr(meta["embedding_norm"], meta["mean_wte_norm"])
    rho_wte,  _    = stats.spearmanr(meta["embedding_norm"], meta["mean_wte_norm"])
    r_h0,   _      = stats.pearsonr(meta["embedding_norm"], meta["mean_h0_tok_norm"])
    r_cos,  _      = stats.pearsonr(meta["embedding_norm"], meta["mean_wte_wpe_cos"])

    summary = {
        "n_phonemes": int(len(meta)),
        "n_items": int(meta["item_id"].nunique()),
        "note_wte": "mean_wte_norm derived as mean ||h0[:,p] - wpe[p]|| from saved activations (no re-extraction).",
        "norms": {
            "embedding_norm":  {"mean": float(meta["embedding_norm"].mean()),  "std": float(meta["embedding_norm"].std())},
            "mean_wpe_norm":   {"mean": float(meta["mean_wpe_norm"].mean()),   "std": float(meta["mean_wpe_norm"].std())},
            "mean_wte_norm":   {"mean": float(meta["mean_wte_norm"].mean()),   "std": float(meta["mean_wte_norm"].std())},
            "mean_h0_tok_norm":{"mean": float(meta["mean_h0_tok_norm"].mean()),"std": float(meta["mean_h0_tok_norm"].std())},
            "mean_wte_wpe_cos":{"mean": float(meta["mean_wte_wpe_cos"].mean()),"std": float(meta["mean_wte_wpe_cos"].std())},
        },
        "correlations": {
            "embedding_norm_vs_mean_wpe_norm": {"pearson": float(r_wpe), "p_pearson": float(p_wpe), "spearman": float(rho_wpe)},
            "embedding_norm_vs_mean_wte_norm": {"pearson": float(r_wte), "p_pearson": float(p_wte), "spearman": float(rho_wte)},
            "embedding_norm_vs_mean_h0_tok_norm": {"pearson": float(r_h0)},
            "embedding_norm_vs_mean_wte_wpe_cos": {"pearson": float(r_cos)},
        },
    }
    out_json = output_dir / "embedding_position_norm_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {out_json}")
    print(f"  embedding_norm vs mean_wpe_norm :  Pearson r = {r_wpe:.4f} (p={p_wpe:.2e}),  Spearman ρ = {rho_wpe:.4f}")
    print(f"  embedding_norm vs mean_wte_norm :  Pearson r = {r_wte:.4f} (p={p_wte:.2e}),  Spearman ρ = {rho_wte:.4f}")
    print(f"  embedding_norm vs mean_h0_tok_norm : Pearson r = {r_h0:.4f}")
    print(f"  embedding_norm vs mean_wte_wpe_cos : Pearson r = {r_cos:.4f}")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: norm by phoneme position — three curves
    by_pos = (
        meta.groupby("phoneme_position_from_start")
        .agg(
            emb_mean=("embedding_norm", "mean"),
            emb_sem=("embedding_norm", lambda x: x.std() / np.sqrt(len(x))),
            wpe_mean=("mean_wpe_norm", "mean"),
            wpe_sem=("mean_wpe_norm", lambda x: x.std() / np.sqrt(len(x))),
            wte_mean=("mean_wte_norm", "mean"),
            wte_sem=("mean_wte_norm", lambda x: x.std() / np.sqrt(len(x))),
            n=("embedding_norm", "count"),
        )
        .query("n >= 3")
        .reset_index()
    )
    ax = axes[0]
    ax.errorbar(by_pos["phoneme_position_from_start"], by_pos["emb_mean"], yerr=by_pos["emb_sem"],
                label="embedding_norm", color="#0072B2", marker="o", ms=5, lw=1.8)
    ax.errorbar(by_pos["phoneme_position_from_start"], by_pos["wpe_mean"], yerr=by_pos["wpe_sem"],
                label="mean_wpe_norm", color="#D55E00", marker="s", ms=5, lw=1.8, linestyle="--")
    ax.errorbar(by_pos["phoneme_position_from_start"], by_pos["wte_mean"], yerr=by_pos["wte_sem"],
                label="mean_wte_norm", color="#009E73", marker="^", ms=5, lw=1.8, linestyle=":")
    ax.set_xlabel("phoneme position from start")
    ax.set_ylabel("L2 norm  (mean ± SEM)")
    ax.set_title("Norm by phoneme position")
    ax.legend(fontsize=9)

    # Right: scatter embedding_norm vs mean_wpe_norm and mean_wte_norm
    ax = axes[1]
    ax.scatter(meta["mean_wpe_norm"], meta["embedding_norm"],
               s=5, alpha=0.3, color="#D55E00", rasterized=True, label=f"vs wpe  r={r_wpe:.3f}")
    ax.scatter(meta["mean_wte_norm"], meta["embedding_norm"],
               s=5, alpha=0.3, color="#009E73", rasterized=True, label=f"vs wte  r={r_wte:.3f}")
    ax.set_xlabel("mean_wpe_norm / mean_wte_norm")
    ax.set_ylabel("embedding_norm")
    ax.set_title("embedding_norm vs wpe and wte norms")
    ax.legend(fontsize=9)

    fig.suptitle(f"AuriStream embedding layer — wpe vs wte norm diagnostic  (n = {len(meta)} phonemes)", fontsize=10)
    fig.tight_layout()
    out_png = output_dir / "embedding_position_norm_diagnostics.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--phoneme-diagnostics", action="store_true",
        help="Also run the phoneme-level wpe norm diagnostic on the real extraction run.",
    )
    parser.add_argument(
        "--run",
        default="reproduce/data/audio/auristream__9d3f269f",
        help="AuriStream extraction run directory (used with --phoneme-diagnostics).",
    )
    parser.add_argument(
        "--dataset",
        default="data/external/paradigm/processed/subset_male.csv",
        help="Paradigm dataset CSV for lexicality/length_bin merge (used with --phoneme-diagnostics).",
    )
    parser.add_argument(
        "--output",
        default="reproduce/figures/audio/auristream_phonemes/auristream__9d3f269f",
        help="Output directory for phoneme diagnostic CSV, JSON, and plots.",
    )
    args = parser.parse_args()

    lm = load_model()
    check_1_same_token_different_positions(lm)
    check_2_algebraic_identity(lm)
    check_3_wpe_norm_profile(lm)
    check_4_block48_before_after_lnf(lm)

    if args.phoneme_diagnostics:
        run_phoneme_diagnostics(
            lm=lm,
            run_dir=REPO_ROOT / args.run,
            dataset_path=REPO_ROOT / args.dataset,
            output_dir=REPO_ROOT / args.output,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()