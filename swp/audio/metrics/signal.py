from __future__ import annotations

import torch
import torchaudio


def si_sdr(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    """Scale-Invariant Signal-to-Distortion Ratio (dB).

    Args:
        reference: Tensor [1, T] or [T], float32
        estimate:  Tensor [1, T'] or [T'], float32

    Returns:
        SI-SDR in dB (float)
    """
    ref = reference.squeeze().float()
    est = estimate.squeeze().float()

    # Align lengths
    min_len = min(ref.shape[-1], est.shape[-1])
    ref = ref[:min_len]
    est = est[:min_len]

    # Zero-mean
    ref = ref - ref.mean()
    est = est - est.mean()

    # Scale factor
    alpha = (est * ref).sum() / (ref * ref).sum().clamp(min=1e-8)
    projection = alpha * ref
    noise = est - projection

    ratio = (projection ** 2).sum() / (noise ** 2).sum().clamp(min=1e-8)
    return 10.0 * torch.log10(ratio.clamp(min=1e-8)).item()


def mel_distance(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    sample_rate: int,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> float:
    """Mean absolute log-mel spectrogram distance.

    Args:
        reference:   Tensor [1, T], float32
        estimate:    Tensor [1, T'], float32
        sample_rate: sample rate of both tensors (Hz)
        n_mels:      number of mel filterbanks (default 80)
        n_fft:       FFT size (default 1024)
        hop_length:  hop size in samples (default 256)

    Returns:
        Mean absolute distance between log-mel spectrograms (float)
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    ref = reference.float()
    est = estimate.float()

    # Align lengths before spectrogramming
    min_len = min(ref.shape[-1], est.shape[-1])
    ref = ref[..., :min_len]
    est = est[..., :min_len]

    ref_mel = torch.log(mel_transform(ref).clamp(min=1e-5))
    est_mel = torch.log(mel_transform(est).clamp(min=1e-5))

    return (ref_mel - est_mel).abs().mean().item()


def compute_all(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    sample_rate: int,
) -> dict[str, float]:
    """Compute all signal metrics between original and reconstructed waveforms.

    Args:
        original:      Tensor [1, T], float32
        reconstructed: Tensor [1, T'], float32
        sample_rate:   sample rate (Hz)

    Returns:
        dict with keys: 'mel_distance', 'si_sdr'
    """
    return {
        "mel_distance": mel_distance(original, reconstructed, sample_rate),
        "si_sdr": si_sdr(original, reconstructed),
    }