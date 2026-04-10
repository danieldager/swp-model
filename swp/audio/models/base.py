from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
from typing_extensions import TypedDict


class ReconstructionResult(TypedDict):
    original: torch.Tensor       # [1, T]  input waveform
    reconstructed: torch.Tensor  # [1, T'] reconstructed waveform


@runtime_checkable
class AudioModel(Protocol):
    """Protocol for pretrained audio models.

    All implementations must expose:
      - name        : str — unique model identifier
      - sample_rate : int — expected input sample rate (Hz)
      - reconstruct()         — encode → quantize → decode round-trip
      - extract_activations() — run forward pass and return layer activations
      - available_layers()    — list stable functional layer names
    """

    name: str
    sample_rate: int

    def reconstruct(self, waveform: torch.Tensor) -> ReconstructionResult:
        """Round-trip the waveform through the model.

        Args:
            waveform: Tensor [1, T], float32, at model's sample_rate

        Returns:
            ReconstructionResult with 'original' and 'reconstructed' tensors
        """
        ...

    def extract_activations(
        self,
        waveform: torch.Tensor,
        layers: list[str],
    ) -> dict[str, torch.Tensor]:
        """Run a forward pass and return activations at the requested layers.

        Args:
            waveform: Tensor [1, T], float32, at model's sample_rate
            layers:   list of stable layer names (see available_layers())

        Returns:
            dict mapping layer name → Tensor; full temporal structure preserved,
            no pooling applied.
        """
        ...

    def available_layers(self) -> list[str]:
        """Return the list of stable functional layer names that can be extracted."""
        ...