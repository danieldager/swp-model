from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from swp.audio.hooks.manager import HookManager
from swp.audio.models.base import ReconstructionResult
from swp.audio.models.device import select_device
from swp.audio.models.registry import register

# Stable functional layer names exposed to callers.
_STABLE_LAYERS = ["encoder_out", "decoder_in", "decoder_out"]

# Mapping: stable name → (hook_type, internal_attr_path, extractor_fn)
# Mirrors the structure of encodec.py exactly.
_LAYER_MAP: dict[str, tuple[str, str, Any]] = {
    # Output of the DAC encoder (before quantizer): [B, D, T_enc]
    "encoder_out": (
        "post",
        "encoder",
        lambda mod, inp, out: out if isinstance(out, torch.Tensor) else out[0],
    ),
    # Input to the DAC decoder (= quantizer output z): [B, D, T_enc]
    # Captured via a pre-hook on the decoder.
    "decoder_in": (
        "pre",
        "decoder",
        lambda mod, inp: inp[0] if isinstance(inp, (tuple, list)) else inp,
    ),
    # Output of the DAC decoder: [B, 1, T_audio]
    "decoder_out": (
        "post",
        "decoder",
        lambda mod, inp, out: out if isinstance(out, torch.Tensor) else out[0],
    ),
}

# Maps model_type string → sample rate (Hz)
_SAMPLE_RATES: dict[str, int] = {
    "16khz": 16_000,
    "24khz": 24_000,
    "44khz": 44_100,
}


@register("dac")
class DACModel:
    """Wrapper around Descript Audio Codec (descript-audio-codec).

    Exposes stable functional layer names for activation extraction:
        encoder_out  — encoder output, shape [D, T_enc]
        decoder_in   — decoder input (quantizer output), shape [D, T_enc]
        decoder_out  — decoder output (waveform), shape [1, T_audio]

    Args:
        model_type: DAC model variant — '16khz', '24khz' (default), or '44khz'.
        device:     torch device string ('cpu', 'cuda', 'mps'), or None to
                    auto-select CUDA > MPS > CPU.
    """

    name = "dac"

    def __init__(
        self,
        model_type: str = "24khz",
        device: str | None = None,
    ) -> None:
        import dac as _dac

        if model_type not in _SAMPLE_RATES:
            raise ValueError(
                f"Unknown model_type {model_type!r}. "
                f"Available: {list(_SAMPLE_RATES)}"
            )

        self.model_type = model_type
        self.sample_rate = _SAMPLE_RATES[model_type]
        self.device = select_device(device)

        model_path = _dac.utils.download(model_type=model_type)
        self._model: nn.Module = _dac.DAC.load(model_path)
        self._model.eval()
        self._model.to(self.device)

    def available_layers(self) -> list[str]:
        return list(_STABLE_LAYERS)

    def reconstruct(self, waveform: torch.Tensor) -> ReconstructionResult:
        """Round-trip: encode → quantize → decode.

        Args:
            waveform: Tensor [1, T], float32

        Returns:
            ReconstructionResult with 'original' [1, T] and 'reconstructed' [1, T']
        """
        wav = self._prepare(waveform)
        with torch.no_grad():
            z, *_ = self._model.encode(wav)
            reconstructed = self._model.decode(z)  # [B, 1, T']
        reconstructed = reconstructed.squeeze(0)    # → [1, T']
        return ReconstructionResult(
            original=waveform.cpu(),
            reconstructed=reconstructed.cpu(),
        )

    def extract_activations(
        self,
        waveform: torch.Tensor,
        layers: list[str],
    ) -> dict[str, torch.Tensor]:
        """Run a forward pass and collect activations at the requested layers.

        Args:
            waveform: Tensor [1, T], float32
            layers:   subset of available_layers()

        Returns:
            dict of layer_name → Tensor, full temporal structure preserved
        """
        unknown = [l for l in layers if l not in _LAYER_MAP]
        if unknown:
            raise ValueError(
                f"Unknown layer(s): {unknown}. Available: {_STABLE_LAYERS}"
            )

        manager = HookManager()
        try:
            for layer_name in layers:
                hook_type, attr_path, extractor_fn = _LAYER_MAP[layer_name]
                module = getattr(self._model, attr_path)
                if hook_type == "post":
                    manager.register_forward_hook(module, layer_name, extractor_fn)
                else:
                    manager.register_forward_pre_hook(module, layer_name, extractor_fn)

            wav = self._prepare(waveform)
            with torch.no_grad():
                z, *_ = self._model.encode(wav)
                self._model.decode(z)

            activations = manager.get_activations()
        finally:
            manager.remove_all()

        # Detach, move to CPU, remove batch dim
        result: dict[str, torch.Tensor] = {}
        for name, tensor in activations.items():
            t = tensor.detach().cpu()
            if t.dim() == 3 and t.shape[0] == 1:
                t = t.squeeze(0)  # [B, D, T] → [D, T]  or  [B, 1, T] → [1, T]
            result[name] = t

        return result

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _prepare(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalise to [1, 1, T], move to device, preprocess for DAC.

        DAC's preprocess() pads the waveform to a multiple of the hop length.
        """
        wav = waveform.to(self.device)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0).unsqueeze(0)   # [T] → [1, 1, T]
        elif wav.dim() == 2:
            wav = wav.unsqueeze(0)                 # [1, T] → [1, 1, T]
        # wav is now [1, 1, T]
        wav = self._model.preprocess(wav, self.sample_rate)
        return wav