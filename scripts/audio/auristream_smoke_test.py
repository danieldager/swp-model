#!/usr/bin/env python3
"""AuriStream + WavCoch smoke test.

Checks:
  1. WavCoch tokenizer loads and produces the expected number of tokens for
     synthetic waveforms (empirical: 1000 for 5 s, 200 for 1 s, 100 for 0.5 s).
  2. AuriStream loads and produces a full-depth temporal hidden-state sequence
     when output_hidden_states=True.
  3. Optionally runs on a real paradigm WAV file.

Run from swp-model/ repo root:
    python scripts/audio/auristream_smoke_test.py
    python scripts/audio/auristream_smoke_test.py --real-audio data/external/paradigm/raw/wav/

Requirements:
    pip install torch torchaudio transformers   (all in requirements.txt)
    huggingface-cli login                        (WavCoch is gated)

Model IDs used:
    WavCoch : TuKoResearch/WavCochV8192          (gated; try WavCochCausalV8192 if 401)
    AuriStream: TuKoResearch/AuriStream1B_librilight_ckpt500k  (public)
              or TuKoResearch/AuriStream100M_1Pred_BigAudioDataset_500k  (gated, smaller)
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse

import torch
import torchaudio

# ---------------------------------------------------------------------------
# Constants — edit to switch model variants
# ---------------------------------------------------------------------------

WAVCOCH_ID = "TuKoResearch/WavCochV8192"
AURISTREAM_ID = "TuKoResearch/AuriStream1B_librilight_ckpt500k"

SAMPLE_RATE = 16_000
HOP_LENGTH = 80   # samples per cochlear token step; empirical token rate = 200 Hz

# Paper-described cochleagram parameters (kept for reference; not used in formula below).
# Paper: FFT window 1001 samples, 988 tokens per 5 s clip.
# Empirical HF model: 1000 tokens per 5 s, 200 per 1 s, 100 per 0.5 s → L = T // HOP_LENGTH.


def token_count(n_samples: int) -> int:
    """Empirical token count: L = T // HOP_LENGTH.

    Verified for T that are exact multiples of HOP_LENGTH (5 s, 1 s, 0.5 s).
    For T not a multiple of 80 samples, floor is used here; behavior is unresolved.
    """
    return n_samples // HOP_LENGTH


def _patch_transformers_tokenization() -> None:
    # WavCoch remote code imports BatchEncoding from transformers.tokenization_utils,
    # but transformers ≥ 5.x moved it to transformers.tokenization_utils_base.
    # Attach it at the old location so the remote import succeeds.
    import transformers.tokenization_utils as _tu
    if not hasattr(_tu, "BatchEncoding"):
        from transformers.tokenization_utils_base import BatchEncoding
        _tu.BatchEncoding = BatchEncoding


def load_models(device: torch.device):
    _patch_transformers_tokenization()
    from transformers import AutoModel

    print(f"\n── Loading WavCoch tokenizer: {WAVCOCH_ID} ──")
    quantizer = AutoModel.from_pretrained(
        WAVCOCH_ID,
        trust_remote_code=True,
    ).to(device).eval()
    print("   OK")

    print(f"\n── Loading AuriStream LM: {AURISTREAM_ID} ──")
    lm = AutoModel.from_pretrained(
        AURISTREAM_ID,
        trust_remote_code=True,
    ).to(device).eval()
    # Print config summary
    cfg = lm.config
    print(f"   OK  n_layer={cfg.n_layer}  n_head={cfg.n_head}  n_embd={cfg.n_embd}  "
          f"vocab={cfg.vocab_size}")

    return quantizer, lm


def run_on_waveform(
    waveform: torch.Tensor,   # (1, T) float32 at 16 kHz
    label: str,
    quantizer,
    lm,
    device: torch.device,
) -> None:
    """Run WavCoch → AuriStream and report shapes."""
    n_samples = waveform.shape[1]
    duration_s = n_samples / SAMPLE_RATE
    expected_tokens = token_count(n_samples)

    print(f"\n── {label} ──")
    print(f"   waveform shape : {list(waveform.shape)}  ({duration_s:.3f} s)")
    print(f"   expected tokens: {expected_tokens}")

    # --- WavCoch tokenization ---
    wav_input = waveform.unsqueeze(0).to(device)   # (1, 1, T)
    with torch.no_grad():
        quant_out = quantizer(wav_input)

    token_ids = quant_out["input_ids"]   # (B, L)
    L = token_ids.shape[1]
    print(f"   token_ids shape : {list(token_ids.shape)}")
    if n_samples % HOP_LENGTH == 0:
        # Exact multiple: empirical formula predicts a precise answer.
        if L == expected_tokens:
            print(f"   token count: {L}  OK  (expected {expected_tokens}, T exact multiple of {HOP_LENGTH})")
        else:
            print(f"   WARNING: got {L} tokens, expected {expected_tokens} (T is exact multiple of {HOP_LENGTH})")
    else:
        # Non-multiple: floor vs ceil behavior unresolved; report both bounds.
        lo, hi = n_samples // HOP_LENGTH, (n_samples + HOP_LENGTH - 1) // HOP_LENGTH
        status = "OK" if L in (lo, hi) else "UNEXPECTED"
        print(f"   token count: {L}  [{status}]  T not a multiple of {HOP_LENGTH} — floor={lo}, ceil={hi}")

    # Log any other keys returned by WavCoch
    extra_keys = [k for k in quant_out if k != "input_ids"]
    if extra_keys:
        for k in extra_keys:
            v = quant_out[k]
            shape_str = list(v.shape) if hasattr(v, "shape") else str(type(v))
            print(f"   WavCoch extra key '{k}': {shape_str}")

    # --- AuriStream hidden states ---
    with torch.no_grad():
        out = lm(token_ids, output_hidden_states=True)

    hidden = out.hidden_states
    n_hidden = len(hidden)
    print(f"\n   AuriStream output_hidden_states: {n_hidden} tensors (expected n_layer+1)")
    for i, h in enumerate(hidden):
        label_str = "embedding" if i == 0 else (f"block_{i}" if i < n_hidden - 1 else f"block_{i} (final)")
        print(f"     [{i:2d}] {label_str:<25}  shape={list(h.shape)}")

    # Check final hidden state shape
    final = hidden[-1]
    B, T_out, D = final.shape
    assert T_out == L, f"token count mismatch: hidden T={T_out} vs token_ids L={L}"
    assert torch.isfinite(final).all(), "non-finite values in final hidden state"
    print(f"\n   final hidden state: ({B}, {T_out}, {D})  finite=True  OK")

    # Token-time mapping — nominal 200 Hz grid (exact WavCoch internal alignment unverified).
    # token i nominally covers [i * 5 ms, (i+1) * 5 ms]; center at (i + 0.5) * 5 ms.
    ms_per_token = HOP_LENGTH / SAMPLE_RATE * 1000
    t_last_start_ms = (L - 1) * ms_per_token
    print(f"\n   token-time mapping (nominal {SAMPLE_RATE // HOP_LENGTH} Hz grid):")
    print(f"     token 0    covers [{0:.1f}, {ms_per_token:.1f}] ms  center {ms_per_token/2:.1f} ms")
    print(f"     token {L-1:<4} covers [{t_last_start_ms:.1f}, {t_last_start_ms + ms_per_token:.1f}] ms")
    print(f"     token rate: {L / duration_s:.1f} tokens/s"
          f"  (note: exact alignment vs WavCoch cochleagram window not yet verified from remote code)")


def main(real_audio_dir: str | None) -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    try:
        quantizer, lm = load_models(device)
    except Exception as e:
        err = str(e).lower()
        print(f"\nFATAL: model loading failed — {e}")
        if any(k in err for k in ("401", "403", "gated", "unauthorized", "token")):
            print("Hint: this looks like an auth/access issue — run: huggingface-cli login")
            print("      and accept model terms at https://huggingface.co/TuKoResearch")
        else:
            print("Hint: this may be a remote-code compatibility issue with the installed")
            print("      transformers version. Try: pip install 'transformers==4.57.6'")
        sys.exit(1)

    # --- Test 1: synthetic 5-second waveform (training length) ---
    wav_5s = torch.randn(1, SAMPLE_RATE * 5)
    run_on_waveform(wav_5s, "Synthetic 5 s", quantizer, lm, device)

    # --- Test 2: synthetic 1-second waveform (typical paradigm stimulus length) ---
    wav_1s = torch.randn(1, SAMPLE_RATE * 1)
    run_on_waveform(wav_1s, "Synthetic 1 s", quantizer, lm, device)

    # --- Test 3: synthetic 0.5-second waveform (short paradigm stimulus) ---
    wav_half = torch.randn(1, SAMPLE_RATE // 2)
    run_on_waveform(wav_half, "Synthetic 0.5 s", quantizer, lm, device)

    # --- Optional: real paradigm WAVs ---
    if real_audio_dir is not None:
        wav_dir = Path(real_audio_dir)
        wav_files = sorted(wav_dir.glob("*.wav"))[:3]
        if not wav_files:
            print(f"\nNo WAV files found in {wav_dir}")
        for wav_path in wav_files:
            wav, sr = torchaudio.load(wav_path)
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != SAMPLE_RATE:
                wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav)
            run_on_waveform(wav, f"Real: {wav_path.name}", quantizer, lm, device)

    print("\n── Smoke test complete ──\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AuriStream + WavCoch smoke test")
    parser.add_argument(
        "--real-audio",
        dest="real_audio_dir",
        default=None,
        help="Optional path to a directory of real paradigm WAV files",
    )
    parser.add_argument(
        "--wavcoch-id",
        default=WAVCOCH_ID,
        help=f"HuggingFace model ID for WavCoch (default: {WAVCOCH_ID})",
    )
    parser.add_argument(
        "--auristream-id",
        default=AURISTREAM_ID,
        help=f"HuggingFace model ID for AuriStream (default: {AURISTREAM_ID})",
    )
    args = parser.parse_args()

    # Allow CLI override of model IDs
    WAVCOCH_ID = args.wavcoch_id
    AURISTREAM_ID = args.auristream_id

    main(args.real_audio_dir)