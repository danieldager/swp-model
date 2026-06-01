#!/usr/bin/env python3
"""Verify block_48_lnf extraction and characterise the block_47→block_48 delta.

Does NOT re-run extraction. Uses saved token-level activations from activations/.

Checks performed (on --max-items items only):
──────────────────────────────────────────────
1. ln_f verification
   • Load saved block_48 token-level [D, L].
   • Apply model.ln_f  →  predicted block_48_lnf.
   • Compare to saved block_48_lnf token-level tensor.
   • Report: max / mean absolute error, relative error, shape match.
   • PASS criterion: max_abs_error < 1e-3 (float32 tolerance).

2. Shape sanity checks
   • block_47, block_48, block_48_lnf all have shape [D, L] with D=1280 and same L.
   • No logit dimensions, no broadcast artefacts.

3. Token-level delta diagnostics  (delta_48 = block_48 − block_47, pooled across items)
   • Concatenate all tokens across the selected items.
   • isotropy_ratio, cos(delta_i, mean_delta), pairwise cos, cos with block_47/block_48.

4. Pooled-level delta diagnostics  (for the same items, pooled phoneme embeddings)
   • Same metrics on phoneme-mean-pooled vectors for the same item subset.
   • Consistency between token-level and pooled-level supports a real model phenomenon.

Usage:
    python scripts/audio/auristream_verify_block48_lnf.py \\
        --run reproduce/data/audio/auristream__6ee9aeb6 \\
        --max-items 3

Output:
    reproduce/figures/audio/auristream_verify/auristream__6ee9aeb6/
        verify_block48_lnf_and_delta.json
        verify_block48_lnf_and_delta.md
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _delta_metrics(
    delta: np.ndarray,       # [N, D]  raw residuals
    ref_a: np.ndarray,       # [N, D]  block_47
    ref_b: np.ndarray,       # [N, D]  block_48
    rng: np.random.Generator,
    n_cos: int = 300,
) -> dict:
    """Compute isotropy, cos-with-mean, pairwise cosine, and cross-layer cosines."""
    n = delta.shape[0]
    norms_d  = np.linalg.norm(delta, axis=1)
    norms_a  = np.linalg.norm(ref_a,  axis=1)
    norms_b  = np.linalg.norm(ref_b,  axis=1)

    delta_n = (delta / (norms_d[:, None] + 1e-12)).astype(np.float32)
    ref_a_n = (ref_a  / (norms_a[:, None]  + 1e-12)).astype(np.float32)
    ref_b_n = (ref_b  / (norms_b[:, None]  + 1e-12)).astype(np.float32)

    mean_d      = delta.mean(axis=0)
    mean_d_norm = float(np.linalg.norm(mean_d))
    iso         = mean_d_norm / (float(norms_d.mean()) + 1e-12)
    mean_dir    = (mean_d / (mean_d_norm + 1e-12)).astype(np.float32)
    cos_dm      = delta_n @ mean_dir        # [N]

    n_s   = min(n_cos, n)
    idx   = rng.choice(n, n_s, replace=False)
    cm    = delta_n[idx] @ delta_n[idx].T
    iu    = np.triu_indices(n_s, k=1)
    cpairs = cm[iu]

    cos_da = (delta_n * ref_a_n).sum(axis=1)   # [N]
    cos_db = (delta_n * ref_b_n).sum(axis=1)   # [N]

    def _s(arr: np.ndarray) -> dict:
        return {
            "mean":  float(arr.mean()),
            "std":   float(arr.std()),
            "q05":   float(np.percentile(arr, 5)),
            "q95":   float(np.percentile(arr, 95)),
        }

    return {
        "n": n,
        "delta_norm": _s(norms_d),
        "isotropy_ratio": float(iso),
        "cos_with_mean_delta": _s(cos_dm),
        "cos_pairwise": _s(cpairs),
        "cos_delta_block47": _s(cos_da),
        "cos_delta_block48": _s(cos_db),
    }


def _load_token(run_dir: Path, manifest: dict, item_id: str, layer: str) -> torch.Tensor:
    """Load [D, L] float32 tensor for one item and layer."""
    path = run_dir / manifest["items"][item_id][layer]
    return torch.load(path, map_location="cpu", weights_only=True).float()


def _pooled_for_items(
    emb_dir: Path,
    meta: pd.DataFrame,
    item_ids: list[str],
    layer: str,
) -> np.ndarray | None:
    """Load pooled embeddings for a specific subset of items."""
    pt_path = emb_dir / f"embeddings_{layer}.pt"
    if not pt_path.exists():
        return None
    emb = torch.load(pt_path, map_location="cpu", weights_only=True).float()
    keep = (
        meta["item_id"].isin(item_ids)
        & meta["valid_embedding"].astype(bool)
        & ~meta["is_silence"].astype(bool)
        & torch.isfinite(emb).all(dim=1).numpy()
    )
    return emb[keep].numpy().astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", required=True,
                   help="Extraction run directory, e.g. reproduce/data/audio/auristream__6ee9aeb6")
    p.add_argument("--max-items", type=int, default=3, dest="max_items",
                   help="Number of items to use (default: 3 — keep tiny).")
    p.add_argument("--random-state", type=int, default=0, dest="random_state")
    p.add_argument("--output", default=None)
    p.add_argument("--overwrite", action="store_true")
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
             / "auristream_verify" / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "verify_block48_lnf_and_delta.json"
    if not args.overwrite and json_path.exists():
        print(f"[verify] Output exists: {json_path}\nPass --overwrite to regenerate.")
        return

    run_params = manifest["run_params"]
    for required in ("block_47", "block_48", "block_48_lnf"):
        if required not in run_params["layers"]:
            print(f"[verify] {required!r} not in run manifest — cannot verify.")
            sys.exit(1)

    item_ids = list(manifest["items"].keys())[: args.max_items]
    print(f"[verify] Items: {item_ids}")
    print(f"[verify] Run  : {run_id}")

    report: dict = {"run_id": run_id, "items": item_ids}

    # ── 1. Load AuriStream ln_f ───────────────────────────────────────────────
    print("\n[verify] Loading AuriStream to extract ln_f …")
    importlib.import_module("swp.audio.models.auristream")   # triggers @register
    from swp.audio.models.registry import get_model
    model  = get_model("auristream")
    ln_f   = model._lm.transformer.ln_f.eval()
    device = model.device
    del model       # free all weights except ln_f (kept by reference in ln_f)
    print(f"[verify] ln_f loaded on {device}")

    # ── 2. Per-item checks ────────────────────────────────────────────────────
    ln_f_checks  = []
    token_b47_list, token_b48_list, token_b48lnf_list = [], [], []

    for item_id in item_ids:
        h47 = _load_token(run_dir, manifest, item_id, "block_47")       # [D, L]
        h48 = _load_token(run_dir, manifest, item_id, "block_48")       # [D, L]
        h48lnf = _load_token(run_dir, manifest, item_id, "block_48_lnf")# [D, L]

        D47, L47 = h47.shape
        D48, L48 = h48.shape
        D48l, L48l = h48lnf.shape

        shapes_ok = (D47 == D48 == D48l == 1280) and (L47 == L48 == L48l)
        print(f"\n  {item_id}")
        print(f"    block_47     : {tuple(h47.shape)}")
        print(f"    block_48     : {tuple(h48.shape)}")
        print(f"    block_48_lnf : {tuple(h48lnf.shape)}")
        print(f"    shapes OK    : {shapes_ok}")

        # Apply ln_f to block_48
        with torch.inference_mode():
            h48_in   = h48.T.unsqueeze(0).to(device)      # (1, L, D)
            h48_pred = ln_f(h48_in).squeeze(0).T.cpu()    # (D, L)

        diff = (h48_pred - h48lnf).abs()
        max_abs  = float(diff.max())
        mean_abs = float(diff.mean())
        denom    = h48lnf.abs().mean().clamp(min=1e-12)
        rel_err  = float(mean_abs / denom)
        passed   = max_abs < 1e-3

        print(f"    ln_f max_abs  : {max_abs:.3e}  {'✓ PASS' if passed else '✗ FAIL'}")
        print(f"    ln_f mean_abs : {mean_abs:.3e}")
        print(f"    ln_f rel_err  : {rel_err:.3e}")

        ln_f_checks.append({
            "item_id": item_id,
            "shapes_ok": shapes_ok,
            "L": int(L48),
            "max_abs_error": max_abs,
            "mean_abs_error": mean_abs,
            "relative_error": rel_err,
            "pass": passed,
        })

        # Accumulate token-level vectors: transpose to [L, D]
        token_b47_list.append(h47.T.numpy())
        token_b48_list.append(h48.T.numpy())
        token_b48lnf_list.append(h48lnf.T.numpy())

    report["ln_f_verification"] = {
        "per_item": ln_f_checks,
        "overall_pass": all(c["pass"] for c in ln_f_checks),
        "max_abs_error_all": float(max(c["max_abs_error"] for c in ln_f_checks)),
        "criterion": "max_abs_error < 1e-3",
    }

    # ── 3. Token-level delta diagnostics ─────────────────────────────────────
    print("\n[verify] Token-level delta diagnostics …")
    tok47 = np.concatenate(token_b47_list, axis=0).astype(np.float32)   # [N_tok, D]
    tok48 = np.concatenate(token_b48_list, axis=0).astype(np.float32)
    tok48lnf = np.concatenate(token_b48lnf_list, axis=0).astype(np.float32)

    delta_tok = tok48 - tok47
    tok_metrics = _delta_metrics(delta_tok, tok47, tok48, rng)
    print(f"    N tokens       : {tok_metrics['n']}")
    print(f"    isotropy_ratio : {tok_metrics['isotropy_ratio']:.5f}")
    print(f"    cos(δ, μ_δ)    : mean={tok_metrics['cos_with_mean_delta']['mean']:.5f}")
    print(f"    pairwise cos   : mean={tok_metrics['cos_pairwise']['mean']:.5f}")
    print(f"    cos(δ, b47)    : mean={tok_metrics['cos_delta_block47']['mean']:.5f}")
    print(f"    cos(δ, b48)    : mean={tok_metrics['cos_delta_block48']['mean']:.5f}")
    report["token_level_delta"] = tok_metrics

    # ── 4. Pooled-level delta diagnostics ─────────────────────────────────────
    meta_path = emb_dir / "metadata_phonemes.csv"
    if meta_path.exists():
        print("\n[verify] Pooled-level delta diagnostics …")
        meta = pd.read_csv(meta_path)
        p47 = _pooled_for_items(emb_dir, meta, item_ids, "block_47")
        p48 = _pooled_for_items(emb_dir, meta, item_ids, "block_48")

        if p47 is not None and p48 is not None and len(p47) > 0:
            delta_pool = p48 - p47
            pool_metrics = _delta_metrics(delta_pool, p47, p48, rng)
            print(f"    N phonemes     : {pool_metrics['n']}")
            print(f"    isotropy_ratio : {pool_metrics['isotropy_ratio']:.5f}")
            print(f"    cos(δ, μ_δ)    : mean={pool_metrics['cos_with_mean_delta']['mean']:.5f}")
            print(f"    pairwise cos   : mean={pool_metrics['cos_pairwise']['mean']:.5f}")
            print(f"    cos(δ, b47)    : mean={pool_metrics['cos_delta_block47']['mean']:.5f}")
            print(f"    cos(δ, b48)    : mean={pool_metrics['cos_delta_block48']['mean']:.5f}")
            report["pooled_level_delta"] = pool_metrics

            # Consistency check
            iso_tok   = tok_metrics["isotropy_ratio"]
            iso_pool  = pool_metrics["isotropy_ratio"]
            consistent = abs(iso_tok - iso_pool) < 0.05
            report["token_pooled_consistency"] = {
                "isotropy_token":  iso_tok,
                "isotropy_pooled": iso_pool,
                "consistent": consistent,
                "note": (
                    "consistent → anisotropy is a model property, not a pooling artefact"
                    if consistent else
                    "inconsistent → pooling may amplify or create anisotropy"
                ),
            }
            print(f"\n    Consistency (|iso_tok - iso_pool| < 0.05): {consistent}")
        else:
            print("  [warning] pooled embeddings not available for selected items")
            report["pooled_level_delta"] = None
    else:
        print("  [warning] metadata_phonemes.csv not found — skipping pooled-level check")
        report["pooled_level_delta"] = None

    # ── 5. Write outputs ───────────────────────────────────────────────────────
    print(f"\n[verify] Overall ln_f PASS: {report['ln_f_verification']['overall_pass']}")

    json_path.write_text(json.dumps(report, indent=2))
    print(f"[verify] Saved {json_path}")

    # Markdown summary
    md_lines = [
        f"# Verification report — {run_id}",
        f"\nItems verified: {', '.join(item_ids)}",
        "",
        "## 1. ln_f verification",
        f"- Criterion: max_abs_error < 1e-3",
        f"- **Overall: {'PASS ✓' if report['ln_f_verification']['overall_pass'] else 'FAIL ✗'}**",
        f"- Max abs error across all items: {report['ln_f_verification']['max_abs_error_all']:.3e}",
        "",
        "| item_id | shapes_ok | L | max_abs | mean_abs | rel_err | pass |",
        "|---------|-----------|---|---------|----------|---------|------|",
    ]
    for c in ln_f_checks:
        md_lines.append(
            f"| {c['item_id']} | {c['shapes_ok']} | {c['L']} | "
            f"{c['max_abs_error']:.2e} | {c['mean_abs_error']:.2e} | "
            f"{c['relative_error']:.2e} | {'✓' if c['pass'] else '✗'} |"
        )

    def _fmt_metrics(m: dict) -> str:
        lines = [
            f"- n: {m['n']}",
            f"- isotropy_ratio: **{m['isotropy_ratio']:.5f}**",
            f"- cos(δ, mean_δ): mean={m['cos_with_mean_delta']['mean']:.5f}  "
            f"std={m['cos_with_mean_delta']['std']:.5f}",
            f"- pairwise cos(δ_i, δ_j): mean={m['cos_pairwise']['mean']:.5f}  "
            f"[q05={m['cos_pairwise']['q05']:.5f}, q95={m['cos_pairwise']['q95']:.5f}]",
            f"- cos(δ, block_47): mean={m['cos_delta_block47']['mean']:.5f}",
            f"- cos(δ, block_48): mean={m['cos_delta_block48']['mean']:.5f}",
        ]
        return "\n".join(lines)

    md_lines += [
        "",
        "## 2. Token-level delta  (block_48 − block_47)",
        _fmt_metrics(tok_metrics),
    ]

    pdelta = report.get("pooled_level_delta")
    if pdelta:
        md_lines += [
            "",
            "## 3. Pooled-level delta  (same items, phoneme-mean-pooled)",
            _fmt_metrics(pdelta),
            "",
            "## 4. Token vs pooled consistency",
        ]
        cons = report["token_pooled_consistency"]
        md_lines += [
            f"- isotropy_ratio  token : {cons['isotropy_token']:.5f}",
            f"- isotropy_ratio pooled : {cons['isotropy_pooled']:.5f}",
            f"- Consistent (|Δ| < 0.05): **{cons['consistent']}**",
            f"- Interpretation: {cons['note']}",
        ]

    md_lines += [
        "",
        "## Interpretation guide",
        "- ln_f PASS + token isotropy ≈ pooled isotropy → "
        "anisotropy is a property of the model, not an extraction bug or pooling artefact.",
        "- High isotropy_ratio (≈ 1) and cos_with_mean (≈ 1) at token level → "
        "the last block adds a near-common direction at every token position.",
        "- PCA PC1 EVR is NOT the primary indicator here (PCA centers internally; "
        "a dominant raw mean direction does not imply a large PC1 EVR).",
    ]

    md_path = output_dir / "verify_block48_lnf_and_delta.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"[verify] Saved {md_path}")
    print(f"\n[verify] Done — {output_dir}")


if __name__ == "__main__":
    main()
