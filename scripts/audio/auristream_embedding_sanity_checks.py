#!/usr/bin/env python3
"""Diagnostic checks for AuriStream embedding layer and block_48.

Four checks on a synthetic repeated-token sequence (no real audio needed):

  1. Same token, different positions  — hidden_states[0] must vary with position
  2. Algebraic identity              — hidden_states[0] == wte(token) + wpe(position)
  3. Norm profile                    — ||wpe(p)|| vs ||hidden_states[0][p]|| by position
  4. block_48 before vs after ln_f   — norm spread and cosine similarity

Usage:
    python scripts/audio/auristream_embedding_sanity_checks.py
"""

from __future__ import annotations

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


def main() -> None:
    lm = load_model()
    check_1_same_token_different_positions(lm)
    check_2_algebraic_identity(lm)
    check_3_wpe_norm_profile(lm)
    check_4_block48_before_after_lnf(lm)
    print("\nDone.")


if __name__ == "__main__":
    main()