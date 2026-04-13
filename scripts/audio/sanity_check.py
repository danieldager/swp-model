#!/usr/bin/env python3
"""Sanity check for the audio model core.

Tests (no real data required — uses synthetic waveforms):
  1. Model loads from checkpoint / HuggingFace
  2. reconstruct() runs on a synthetic 1s waveform and returns finite audio
  3. extract_activations() works for all stable layer names
  4. Activation shapes are consistent:
       encoder_out : 2D  [C, T_enc]
       decoder_in  : 2D  [C, T_enc]
       decoder_out : ≤2D [1, T_audio] or [T_audio]

Run from swp-model/ repo root:
    python scripts/audio/sanity_check.py
    python scripts/audio/sanity_check.py --model encodec --bandwidth 1.5
    python scripts/audio/sanity_check.py --model dac
    python scripts/audio/sanity_check.py --model dac --model-type 16khz
"""

from __future__ import annotations

from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import sys

import torch


def main(model_name: str, model_kwargs: dict) -> None:
    label = f"{model_name}  kwargs={model_kwargs}"
    print(f"── Sanity check: {label} ──────────────────────")

    # ── Import ────────────────────────────────────────────────────────────────
    print("1. Importing swp.audio.models ...")
    if model_name == "encodec":
        import swp.audio.models.encodec  # noqa: F401
    elif model_name == "dac":
        import swp.audio.models.dac  # noqa: F401
    else:
        print(f"   WARNING: no explicit import registered for {model_name!r}; "
              "attempting get_model() anyway")
    from swp.audio.models import get_model
    print("   OK")

    # ── Load model ────────────────────────────────────────────────────────────
    print("2. Loading model ...")
    model = get_model(model_name, **model_kwargs)
    print(f"   OK  name={model.name!r}  sample_rate={model.sample_rate}")
    print(f"   Available layers: {model.available_layers()}")

    # ── Synthetic waveform ────────────────────────────────────────────────────
    sr = model.sample_rate
    duration = 1.0
    waveform = torch.randn(1, int(sr * duration))  # [1, T]
    print(f"\n3. Synthetic waveform shape: {list(waveform.shape)}  sr={sr}")

    # ── reconstruct() ─────────────────────────────────────────────────────────
    print("4. reconstruct() ...")
    result = model.reconstruct(waveform)
    orig = result["original"]
    recon = result["reconstructed"]
    assert orig.shape == waveform.shape, f"original shape mismatch: {orig.shape}"
    assert recon.dim() <= 2, f"unexpected reconstructed ndim: {recon.dim()}"
    assert torch.isfinite(recon).all(), "reconstructed waveform contains non-finite values"
    print(f"   original     : {list(orig.shape)}")
    print(f"   reconstructed: {list(recon.shape)}  finite=True")
    print("   OK")

    # ── extract_activations() ─────────────────────────────────────────────────
    print("5. extract_activations() for all stable layers ...")
    layers = model.available_layers()
    activations = model.extract_activations(waveform, layers)

    all_ok = True
    for layer_name in layers:
        assert layer_name in activations, f"missing layer {layer_name!r} in output"
        t = activations[layer_name]
        finite = torch.isfinite(t).all().item()
        shape = list(t.shape)

        if layer_name in ("encoder_out", "decoder_in"):
            expected_ndim = 2
            ok = t.dim() == 2
        elif layer_name == "decoder_out":
            expected_ndim = "≤2"
            ok = t.dim() <= 2
        else:
            ok = True
            expected_ndim = "any"

        status = "OK" if (ok and finite) else "FAIL"
        if not (ok and finite):
            all_ok = False
        print(
            f"   {layer_name:<15} shape={shape}  ndim={t.dim()}  "
            f"(expected {expected_ndim}D)  finite={finite}  [{status}]"
        )

    if not all_ok:
        print("\nSome checks FAILED.")
        sys.exit(1)

    print("\nAll checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity check for audio model core.")
    parser.add_argument(
        "--model",
        default="encodec",
        help="Model name to test (default: encodec)",
    )
    # EnCodec-specific
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=6.0,
        help="EnCodec bandwidth in kbps (default: 6.0; ignored for other models)",
    )
    # DAC-specific
    parser.add_argument(
        "--model-type",
        dest="model_type",
        default="24khz",
        help="DAC model type: 16khz, 24khz, 44khz (default: 24khz; ignored for other models)",
    )
    args = parser.parse_args()

    if args.model == "encodec":
        model_kwargs = {"bandwidth": args.bandwidth}
    elif args.model == "dac":
        model_kwargs = {"model_type": args.model_type}
    else:
        model_kwargs = {}

    main(args.model, model_kwargs)