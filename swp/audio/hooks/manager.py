from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn


class HookManager:
    """Manages forward and forward-pre hooks on PyTorch modules.

    Supports optional extractor functions to unwrap model-specific output
    types (tuples, dataclasses, etc.) into plain Tensors.

    Full temporal structure is always preserved — no pooling is applied.

    Usage:
        manager = HookManager()
        manager.register_forward_hook(model.encoder, "encoder_out")
        with torch.no_grad():
            model(waveform)
        activations = manager.get_activations()
        manager.clear()
        manager.remove_all()
    """

    def __init__(self) -> None:
        self._activations: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHook] = []

    # ── Registration ─────────────────────────────────────────────────────────

    def register_forward_hook(
        self,
        module: nn.Module,
        layer_name: str,
        extractor_fn: Callable[[Any, Any, Any], torch.Tensor] | None = None,
    ) -> None:
        """Register a forward hook on a module.

        Args:
            module:       PyTorch module to hook
            layer_name:   stable name used as key in get_activations()
            extractor_fn: optional callable (module, input, output) → Tensor
                          used to select/unwrap the relevant tensor from
                          model-specific output types (e.g. tuples).
                          When None, the raw output is stored directly.
        """
        def _hook(mod: nn.Module, inp: Any, out: Any) -> None:
            if extractor_fn is not None:
                tensor = extractor_fn(mod, inp, out)
            else:
                tensor = out
            self._activations[layer_name] = tensor

        handle = module.register_forward_hook(_hook)
        self._handles.append(handle)

    def register_forward_pre_hook(
        self,
        module: nn.Module,
        layer_name: str,
        extractor_fn: Callable[[Any, Any], torch.Tensor] | None = None,
    ) -> None:
        """Register a forward pre-hook on a module.

        Args:
            module:       PyTorch module to hook
            layer_name:   stable name used as key in get_activations()
            extractor_fn: optional callable (module, input) → Tensor
                          used to select/unwrap the relevant tensor from
                          the module's input tuple.
                          When None, input[0] is stored directly.
        """
        def _pre_hook(mod: nn.Module, inp: Any) -> None:
            if extractor_fn is not None:
                tensor = extractor_fn(mod, inp)
            else:
                tensor = inp[0] if isinstance(inp, (tuple, list)) else inp
            self._activations[layer_name] = tensor

        handle = module.register_forward_pre_hook(_pre_hook)
        self._handles.append(handle)

    # ── Access ────────────────────────────────────────────────────────────────

    def get_activations(self) -> dict[str, torch.Tensor]:
        """Return a snapshot of all collected activations."""
        return dict(self._activations)

    def clear(self) -> None:
        """Clear stored activations (keeps hooks registered)."""
        self._activations.clear()

    def remove_all(self) -> None:
        """Remove all registered hooks and clear activations."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._activations.clear()