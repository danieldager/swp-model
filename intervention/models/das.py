"""Distributed Alignment Search (DAS) as an intervention method plugin.

A learned orthogonal rotation R maps the RNN state into a space where each sequence
position owns a contiguous block; an optional soft per-channel mask (the "alignment
matrix") selects which rotated channels carry the causal variable. For a minimal pair
(base, source) differing at one position we rotate both encoder states, swap the edited
position's block from source into base, rotate back, and decode toward ``source``.

This module exposes only the pieces the shared runner needs:
    DASIntervention        — the rotation (+ mask) module
    DASTrainer             — InterventionTrainer with the state-swap batch step
    build_das_intervention — construct a DASIntervention from a MethodConfig

The data, training loop, and I/O all come from the shared pipeline; DAS runs on the same
loaders as the scale method (real-real minimal pairs or synthetic edits, including
``edit_ngram > 1``, where ``span`` adjacent blocks are swapped instead of one).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from intervention.models.repeat_model import decode_with_hidden, get_encoder_hidden
from intervention.experiments.trainer import InterventionTrainer


def gumbel_sigmoid(logits: torch.Tensor, tau: float = 1.0, hard: bool = True, eps: float = 1e-10) -> torch.Tensor:
    """Differentiable Bernoulli sample (concrete relaxation) used to train the mask."""
    u = torch.rand_like(logits)
    logistic = torch.log(u + eps) - torch.log(1 - u + eps)
    y = torch.sigmoid((logits + logistic) / tau)
    if hard:
        y = (y > 0.5).float() + y - y.detach()
    return y


class DASIntervention(nn.Module):
    """Rotation-based interchange intervention on the frozen RNN state.

    ``state_mode`` selects the target: 'concat' = [h, c], else 'h' / 'c' / 'both'.
    """

    def __init__(
        self,
        state_dim: int,
        n_variables: int,
        var_size: int | None = None,
        masking: bool = True,
        reg_loss: float = 1e-3,
        state_mode: str = "concat",
        span: int = 1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_variables = n_variables
        self.var_size = var_size or state_dim // n_variables
        self.intervention_size = self.var_size * self.n_variables
        if self.intervention_size > state_dim:
            raise ValueError(f"var_size*n_variables={self.intervention_size} exceeds state_dim={state_dim}")
        if not 1 <= span <= n_variables:
            raise ValueError(f"span={span} must be in [1, n_variables={n_variables}]")
        self.masking = masking
        self.reg_loss = reg_loss
        self.state_mode = state_mode
        self.span = span  # blocks swapped per intervention (= edit_ngram: 1 phoneme, 2 bigram, ...)

        # Orthogonal rotation R (default matrix_exp parametrisation).
        self.rotation = nn.utils.parametrizations.orthogonal(nn.Linear(state_dim, state_dim, bias=False))
        if masking:
            self.mask = nn.Parameter(torch.zeros(self.intervention_size))
        self.register_buffer("_reg", torch.zeros(()), persistent=False)

    def _rotate_swap(self, base: torch.Tensor, source: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        B = base.shape[0]
        rb = self.rotation(base)      # base in DAS basis
        rs = self.rotation(source)    # source in DAS basis
        I = self.intervention_size

        bb = rb[:, :I].view(B, self.n_variables, self.var_size)
        ss = rs[:, :I].view(B, self.n_variables, self.var_size)
        # Swap the `span` adjacent blocks starting at the edited position (span=1 == one-hot).
        start = position.clamp(max=self.n_variables - self.span).unsqueeze(1)      # (B, 1)
        idx = torch.arange(self.n_variables, device=position.device).unsqueeze(0)  # (1, V)
        sel = ((idx >= start) & (idx < start + self.span)).unsqueeze(-1)           # (B, V, 1)
        swapped = torch.where(sel, ss, bb).reshape(B, I)

        if self.masking:
            swapped = self._align(swapped, rb[:, :I])
        out = torch.cat([swapped, rb[:, I:]], dim=-1)     # untouched dims kept from base
        return F.linear(out, self.rotation.weight.t())    # inverse rotation (R orthogonal)

    def _align(self, intervened: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        m = gumbel_sigmoid(self.mask) if self.training else (self.mask > 0).float()
        self._reg = self._reg + self.reg_loss * torch.sigmoid(self.mask).sum()
        return m * intervened + (1 - m) * original

    def intervene(self, h, c, h_src, c_src, position) -> tuple[torch.Tensor, torch.Tensor]:
        self._reg = torch.zeros((), device=h.device)
        if self.state_mode == "concat":
            out = self._rotate_swap(torch.cat([h, c], -1), torch.cat([h_src, c_src], -1), position)
            return out.chunk(2, dim=-1)
        if self.state_mode == "h":
            return self._rotate_swap(h, h_src, position), c
        if self.state_mode == "c":
            return h, self._rotate_swap(c, c_src, position)
        if self.state_mode == "both":
            return self._rotate_swap(h, h_src, position), self._rotate_swap(c, c_src, position)
        raise ValueError(f"unknown state_mode: {self.state_mode}")

    def reg(self) -> torch.Tensor:
        return self._reg

    @torch.no_grad()
    def get_parameters(self) -> dict[str, np.ndarray]:
        params = {
            "rotation": self.rotation.weight.detach().cpu().numpy(),
            "n_variables": np.int64(self.n_variables),
            "var_size": np.int64(self.var_size),
            "state_mode": self.state_mode,
            "span": np.int64(self.span),
        }
        if self.masking:
            params["mask"] = torch.sigmoid(self.mask).detach().cpu().numpy()
        return params


class DASTrainer(InterventionTrainer):
    """Encodes both words in the minimal pair and swaps the rotated block; the target
    is the ``source`` word (the counterfactual)."""

    def _run_batch(self, batch: dict[str, torch.Tensor]):
        base_ids = batch["input"].to(self.device)     # base word
        source_ids = batch["target"].to(self.device)  # source word == counterfactual target
        position = batch["position"].to(self.device)

        h, c = get_encoder_hidden(self.repeat_model, base_ids, self.device)
        h_src, c_src = get_encoder_hidden(self.repeat_model, source_ids, self.device)
        h_mod, c_mod = self.intervention.intervene(h, c, h_src, c_src, position)
        logits = decode_with_hidden(self.repeat_model, h_mod, c_mod, source_ids, self.device, self.teacher_forcing)

        loss = self._compute_loss(logits, source_ids) + self.intervention.reg()
        return loss, logits.argmax(dim=-1), source_ids, batch["seq_len"]


def build_das_intervention(method_cfg, hidden_size: int, n_variables: int,
                           span: int = 1) -> DASIntervention:
    """Construct a ``DASIntervention`` from a ``MethodConfig``; ``span`` is the number of
    adjacent position blocks swapped per intervention (= the data's ``edit_ngram``)."""
    state_dim = hidden_size * (2 if method_cfg.state_mode == "concat" else 1)
    return DASIntervention(
        state_dim=state_dim,
        n_variables=n_variables,
        var_size=method_cfg.var_size,
        masking=method_cfg.masking,
        reg_loss=method_cfg.reg_loss,
        state_mode=method_cfg.state_mode,
        span=span,
    )
