from __future__ import annotations

import torch

from swp.audio.models.base import ReconstructionResult
from swp.audio.models.device import select_device
from swp.audio.models.registry import register

_DEFAULT_WAVCOCH_ID = "TuKoResearch/WavCochV8192"
_DEFAULT_AURISTREAM_ID = "TuKoResearch/AuriStream1B_librilight_ckpt500k"

SAMPLE_RATE = 16_000
HOP_LENGTH = 80  # samples per cochlear token; token rate = SAMPLE_RATE // HOP_LENGTH = 200 Hz


def _patch_transformers_tokenization() -> None:
    # WavCoch remote code imports BatchEncoding from transformers.tokenization_utils.
    # In transformers ≥ 5.x it was moved to transformers.tokenization_utils_base.
    # requirements.txt pins transformers < 5 so this is a no-op under normal use,
    # but guards against accidental upgrades.
    import transformers.tokenization_utils as _tu
    if not hasattr(_tu, "BatchEncoding"):
        from transformers.tokenization_utils_base import BatchEncoding
        _tu.BatchEncoding = BatchEncoding


@register("auristream")
class AuriStreamModel:
    """Wrapper around WavCoch tokenizer + AuriStream language model.

    Converts a 16 kHz mono waveform into cochlear token IDs (WavCoch) then
    runs AuriStream with output_hidden_states=True to extract the full
    temporal hidden-state sequence for each requested layer.

    Layer names
    -----------
    "embedding"   initial token embedding     (hidden_states[0])
    "block_01"    transformer block 1 output  (hidden_states[1])
    ...
    "block_N"     transformer block N output  (hidden_states[N])

    Returned tensors are [D, L] float32 on CPU, matching the [C, T] convention
    of EnCodec/DAC wrappers.  D = n_embd, L = floor(n_samples_16k / hop_length).

    Token-time mapping (empirical, 200 Hz):
        token i → [i * 5 ms, (i+1) * 5 ms),  center at (i + 0.5) * 5 ms

    reconstruct() is not supported — AuriStream is a language model.
    """

    name = "auristream"
    sample_rate = SAMPLE_RATE
    hop_length = HOP_LENGTH

    def __init__(
        self,
        wavcoch_id: str = _DEFAULT_WAVCOCH_ID,
        auristream_id: str = _DEFAULT_AURISTREAM_ID,
        device: str | None = None,
    ) -> None:
        _patch_transformers_tokenization()
        from transformers import AutoModel

        self.wavcoch_id = wavcoch_id
        self.auristream_id = auristream_id
        self.device = select_device(device)

        self._quantizer = AutoModel.from_pretrained(
            wavcoch_id, trust_remote_code=True
        ).to(self.device).eval()

        self._lm = AutoModel.from_pretrained(
            auristream_id, trust_remote_code=True
        ).to(self.device).eval()

        n_layer: int = self._lm.config.n_layer
        self._layer_names: list[str] = (
            ["embedding"] + [f"block_{k:02d}" for k in range(1, n_layer + 1)]
        )
        self._layer_index: dict[str, int] = {
            name: i for i, name in enumerate(self._layer_names)
        }

    def available_layers(self) -> list[str]:
        return list(self._layer_names)

    def reconstruct(self, waveform: torch.Tensor) -> ReconstructionResult:
        raise NotImplementedError(
            "AuriStream is a language model and does not support audio "
            "reconstruction. Use extract_activations() to obtain hidden states."
        )

    def extract_activations(
        self,
        waveform: torch.Tensor,
        layers: list[str],
    ) -> dict[str, torch.Tensor]:
        """Extract hidden states for the requested layers.

        Args:
            waveform: [1, T] float32 at model.sample_rate (16 kHz)
            layers:   subset of available_layers()

        Returns:
            dict: layer_name → [D, L] float32 CPU tensor
                  D = n_embd (1280 for AuriStream-1B)
                  L = floor(T / hop_length)
        """
        unknown = [la for la in layers if la not in self._layer_index]
        if unknown:
            raise ValueError(
                f"Unknown layer(s): {unknown}. Available: {self._layer_names}"
            )

        wav_input = self._prepare(waveform)  # (1, 1, T) on device

        with torch.inference_mode():
            token_ids = self._quantizer(wav_input)["input_ids"]   # (1, L)
            out = self._lm(token_ids, output_hidden_states=True)
            # out.hidden_states: tuple of n_layer+1 tensors, each (1, L, D)

        result: dict[str, torch.Tensor] = {}
        for layer_name in layers:
            idx = self._layer_index[layer_name]
            h = out.hidden_states[idx]   # (1, L, D)
            h = h.squeeze(0)             # (L, D)
            h = h.transpose(0, 1)        # (D, L)  — matches [C, T] convention
            result[layer_name] = h.float().cpu()

        return result

    def _prepare(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalise to (1, 1, T) on the model device.

        Sample rate: the caller is responsible for providing audio already at
        model.sample_rate (16 kHz).  ParadigmDataset(target_sr=model.sample_rate)
        handles this when used via extract_representations.py.
        """
        wav = waveform
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)              # (T,) → (1, T)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True) # stereo → mono
        return wav.unsqueeze(0).to(self.device) # (1, T) → (1, 1, T)