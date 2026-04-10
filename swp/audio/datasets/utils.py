from __future__ import annotations

from pathlib import Path

import torch
import torchaudio


def load_audio(
    path: str | Path,
    target_sr: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Load an audio file, convert to mono, optionally resample.

    Args:
        path: path to audio file
        target_sr: if provided, resample to this sample rate

    Returns:
        waveform: Tensor [1, T] — mono, float32
        original_sr: original sample rate of the file
    """
    waveform, original_sr = torchaudio.load(path)
    waveform = to_mono(waveform)
    if target_sr is not None and target_sr != original_sr:
        waveform = resample(waveform, original_sr, target_sr)
    return waveform, original_sr


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Average channels to produce a mono waveform.

    Args:
        waveform: Tensor [C, T]

    Returns:
        Tensor [1, T]
    """
    if waveform.shape[0] == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


def resample(
    waveform: torch.Tensor,
    orig_sr: int,
    target_sr: int,
) -> torch.Tensor:
    """Resample waveform from orig_sr to target_sr.

    Args:
        waveform: Tensor [1, T]
        orig_sr: source sample rate
        target_sr: target sample rate

    Returns:
        Tensor [1, T']
    """
    if orig_sr == target_sr:
        return waveform
    return torchaudio.functional.resample(waveform, orig_sr, target_sr)