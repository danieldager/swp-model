from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from swp.audio.hooks.manager import HookManager
from swp.audio.models.base import ReconstructionResult
from swp.audio.models.registry import register

# Stable functional layer names exposed to callers.
# Internal HF module paths are an implementation detail of this wrapper.
_STABLE_LAYERS = ["encoder_out", "decoder_in", "decoder_out"]

# Mapping: stable name → (hook_type, internal_attr_path, extractor_fn_or_None)
# hook_type: "post" = forward hook, "pre" = forward pre-hook
# internal_attr_path: dotted path relative to the HF EncodecModel instance
# extractor_fn: callable to unwrap the relevant tensor, or None for raw output
_LAYER_MAP: dict[str, tuple[str, str, Any]] = {
    # Output of the encoder (before quantizer): [B, C, T_enc]
    "encoder_out": (
        "post",
        "encoder",
        lambda mod, inp, out: out if isinstance(out, torch.Tensor) else out[0],
    ),
    # Input to the decoder (= quantizer output): [B, C, T_enc]
    # Captured via a pre-hook on the decoder so we see what the decoder receives.
    "decoder_in": (
        "pre",
        "decoder",
        lambda mod, inp: inp[0] if isinstance(inp, (tuple, list)) else inp,
    ),
    # Output of the decoder: [B, 1, T_audio]
    "decoder_out": (
        "post",
        "decoder",
        lambda mod, inp, out: out if isinstance(out, torch.Tensor) else out[0],
    ),
}


def _get_submodule(model: nn.Module, dotted_path: str) -> nn.Module:
    """Resolve a dotted attribute path to a submodule."""
    parts = dotted_path.split(".")
    m = model
    for part in parts:
        m = getattr(m, part)
    return m


@register("encodec")
class EnCodecModel:
    """Wrapper around HuggingFace EnCodec (facebook/encodec_24khz).

    Exposes stable functional layer names for activation extraction:
        encoder_out  — encoder output, shape [B, C, T_enc]
        decoder_in   — decoder input (quantizer output), shape [B, C, T_enc]
        decoder_out  — decoder output (waveform), shape [B, 1, T_audio]

    Args:
        bandwidth: target bandwidth in kbps (default: 6.0).
                   Valid values for encodec_24khz: 1.5, 3.0, 6.0, 12.0, 24.0
        device:    torch device string; defaults to 'cpu'
    """

    name = "encodec"
    sample_rate = 24_000

    def __init__(self, bandwidth: float = 6.0, device: str = "cpu") -> None:
        from transformers import EncodecModel as HFEncodecModel

        self.bandwidth = bandwidth
        self.device = torch.device(device)

        self._model: HFEncodecModel = HFEncodecModel.from_pretrained(
            "facebook/encodec_24khz"
        )
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
            out = self._model(
                input_values=wav,
                bandwidth=self.bandwidth,
            )
        # HF EnCodec output: audio_values shape [B, 1, T']
        reconstructed = out.audio_values.squeeze(0)  # → [1, T']
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
                module = _get_submodule(self._model, attr_path)
                if hook_type == "post":
                    manager.register_forward_hook(module, layer_name, extractor_fn)
                else:
                    manager.register_forward_pre_hook(module, layer_name, extractor_fn)

            wav = self._prepare(waveform)
            with torch.no_grad():
                self._model(input_values=wav, bandwidth=self.bandwidth)

            activations = manager.get_activations()
        finally:
            manager.remove_all()

        # Detach, move to CPU, remove batch dim
        result: dict[str, torch.Tensor] = {}
        for name, tensor in activations.items():
            t = tensor.detach().cpu()
            if t.dim() == 3 and t.shape[0] == 1:
                t = t.squeeze(0)  # [B, C, T] → [C, T]  or  [B, 1, T] → [1, T]
            result[name] = t

        return result

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _prepare(self, waveform: torch.Tensor) -> torch.Tensor:
        """Ensure waveform is [1, 1, T] (batch=1, channels=1) on the right device."""
        wav = waveform.to(self.device)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0).unsqueeze(0)   # [T] → [1, 1, T]
        elif wav.dim() == 2:
            wav = wav.unsqueeze(0)                 # [1, T] → [1, 1, T]
        # wav is now [B, C, T] — already correct
        return wav