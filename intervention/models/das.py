"""Distributed Alignment Search (DAS) as an intervention method plugin.

A learned orthogonal rotation R maps the frozen RNN state into a basis where sequence
position is carved into *variables*. For a (base, source) pair we rotate both encoder
states, take the variables the counterfactual needs from source, rotate back, and decode
toward the target.

Two carvings share all the plumbing:

    DASIntervention   fixed contiguous blocks of ``var_size`` channels, one per variable,
                      plus an optional per-channel on/off mask
    AutoSegDAS        no blocks: every channel learns which single variable it belongs to
                      (or "none"), so variable sizes are learned and need not be equal.
                      This is the Csordas parameterisation, and ``var_sizes`` is the
                      measurement it exists to produce.

Which variables move is :func:`select_variables`, shared by both and driven by the data's
``edit_ngram`` and the method's ``var_ngram``.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from intervention.models.repeat_model_utils import decode_with_hidden, get_encoder_hidden
from intervention.experiments.trainer import InterventionTrainer


def gumbel_sigmoid(logits: torch.Tensor, tau: float = 1.0, hard: bool = True, eps: float = 1e-10) -> torch.Tensor:
    """Differentiable Bernoulli sample (concrete relaxation) used to train the mask."""
    u = torch.rand_like(logits)
    logistic = torch.log(u + eps) - torch.log(1 - u + eps)
    y = torch.sigmoid((logits + logistic) / tau)
    if hard:
        y = (y > 0.5).float() + y - y.detach()
    return y


def gumbel_softmax(logits: torch.Tensor, tau: float = 1.0, hard: bool = True, eps: float = 1e-10) -> torch.Tensor:
    """Straight-through categorical sample over the last dim (the channel's variable)."""
    u = torch.empty_like(logits).uniform_(eps, 1 - eps)
    y = torch.softmax((logits - (-u.log()).log()) / tau, dim=-1)
    if hard:
        y = F.one_hot(y.argmax(-1), y.shape[-1]).float() - y.detach() + y
    return y


def select_variables(
    position: torch.Tensor, word_len: torch.Tensor,
    n_variables: int, edit_ngram: int, var_ngram: int,
) -> torch.Tensor:
    """``[B, n_variables]`` mask of the variables the counterfactual has to move.

    Variable ``v`` owns the window ``[v, v+var_ngram-1]``; the edit owns
    ``[position, position+edit_ngram-1]``. A variable moves iff the two overlap, and only
    windows that fit inside the word are addressable. ``var_ngram=1`` gives one variable
    per position (a contiguous span of ``edit_ngram``); ``edit_ngram=1`` with
    ``var_ngram=n`` gives Csordas's overlapping n-gram variables.

    Words shorter than ``var_ngram`` have no addressable variable at all and select
    nothing (rather than a fabricated variable 0) — keep them out of the data with
    ``DataConfig.min_word_len``.
    """
    idx = torch.arange(n_variables, device=position.device)[None]
    lo = (position - var_ngram + 1).clamp(min=0)[:, None]
    hi = torch.minimum(position + edit_ngram - 1, word_len - var_ngram)[:, None]
    return (idx >= lo) & (idx <= hi)


class _RotationIntervention(nn.Module):
    """Orthogonal rotation + ``state_mode`` dispatch; subclasses carve the rotated basis.

    ``state_mode`` selects the target: 'concat' = [h, c], else 'h' / 'c' / 'both'
    ('both' applies the same rotation to h and c independently).
    """

    def __init__(self, state_dim: int, n_variables: int, reg_loss: float,
                 state_mode: str, edit_ngram: int, var_ngram: int):
        super().__init__()
        if state_mode not in ("c", "h", "both", "concat"):
            raise ValueError(f"unknown state_mode: {state_mode}")
        self.state_dim = state_dim
        self.n_variables = n_variables
        self.reg_loss = reg_loss
        self.state_mode = state_mode
        self.edit_ngram = edit_ngram
        self.var_ngram = var_ngram
        self.rotation = nn.utils.parametrizations.orthogonal(nn.Linear(state_dim, state_dim, bias=False))
        self.register_buffer("_reg", torch.zeros(()), persistent=False)

    def _combine(self, rb: torch.Tensor, rs: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _reg_term(self) -> torch.Tensor:
        """Mask penalty. Counted once per batch, not once per rotated state, so
        ``state_mode='both'`` (which swaps twice) is not silently regularised twice."""
        return torch.zeros((), device=self.rotation.weight.device)

    def _swap(self, base: torch.Tensor, source: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
        out = self._combine(self.rotation(base), self.rotation(source), sel)
        return F.linear(out, self.rotation.weight.t())  # inverse rotation (R orthogonal)

    def intervene(self, h, c, h_src, c_src, position, word_len) -> tuple[torch.Tensor, torch.Tensor]:
        self._reg = self._reg_term()
        sel = select_variables(position, word_len, self.n_variables, self.edit_ngram, self.var_ngram)
        if self.state_mode == "concat":
            out = self._swap(torch.cat([h, c], -1), torch.cat([h_src, c_src], -1), sel)
            return out.chunk(2, dim=-1)
        if self.state_mode == "h":
            return self._swap(h, h_src, sel), c
        if self.state_mode == "c":
            return h, self._swap(c, c_src, sel)
        return self._swap(h, h_src, sel), self._swap(c, c_src, sel)  # both

    def reg(self) -> torch.Tensor:
        return self._reg

    @torch.no_grad()
    def _base_params(self) -> dict[str, np.ndarray]:
        return {
            "rotation": self.rotation.weight.detach().cpu().numpy(),
            "n_variables": np.int64(self.n_variables),
            "state_mode": self.state_mode,
            "edit_ngram": np.int64(self.edit_ngram),
            "var_ngram": np.int64(self.var_ngram),
        }


class DASIntervention(_RotationIntervention):
    """Fixed contiguous blocks: variable ``v`` owns channels ``[v*var_size, (v+1)*var_size)``."""

    def __init__(self, state_dim: int, n_variables: int, var_size: int | None = None,
                 masking: bool = True, reg_loss: float = 1e-3, state_mode: str = "concat",
                 edit_ngram: int = 1, var_ngram: int = 1):
        super().__init__(state_dim, n_variables, reg_loss, state_mode, edit_ngram, var_ngram)
        self.var_size = var_size or state_dim // n_variables
        self.intervention_size = self.var_size * n_variables
        if self.intervention_size > state_dim:
            raise ValueError(f"var_size*n_variables={self.intervention_size} exceeds state_dim={state_dim}")
        self.masking = masking
        if masking:
            self.mask = nn.Parameter(torch.zeros(self.intervention_size))

    def _combine(self, rb, rs, sel):
        B, I = rb.shape[0], self.intervention_size
        bb = rb[:, :I].view(B, self.n_variables, self.var_size)
        ss = rs[:, :I].view(B, self.n_variables, self.var_size)
        swapped = torch.where(sel.unsqueeze(-1), ss, bb).reshape(B, I)

        if self.masking:
            m = gumbel_sigmoid(self.mask) if self.training else (self.mask > 0).float()
            swapped = m * swapped + (1 - m) * rb[:, :I]
        return torch.cat([swapped, rb[:, I:]], dim=-1)  # untouched dims kept from base

    def _reg_term(self):
        if not self.masking:
            return super()._reg_term()
        return self.reg_loss * torch.sigmoid(self.mask).sum()  # expected active channels

    @torch.no_grad()
    def get_parameters(self) -> dict[str, np.ndarray]:
        params = {**self._base_params(), "var_size": np.int64(self.var_size)}
        if self.masking:
            params["mask"] = torch.sigmoid(self.mask).detach().cpu().numpy()
        return params


class AutoSegDAS(_RotationIntervention):
    """Learned segmentation: each channel is assigned to one variable, or to "none".

    ``mask`` is ``[state_dim, n_variables + 1]``; the extra column is "this channel is
    never intervened on". Variable sizes are whatever training makes them, which is what
    ``var_sizes`` reports.
    """

    def __init__(self, state_dim: int, n_variables: int, reg_loss: float = 1e-3,
                 state_mode: str = "concat", edit_ngram: int = 1, var_ngram: int = 1):
        super().__init__(state_dim, n_variables, reg_loss, state_mode, edit_ngram, var_ngram)
        self.mask = nn.Parameter(torch.randn(state_dim, n_variables + 1) * 0.01)

    def _assignment(self) -> torch.Tensor:
        if self.training:
            return gumbel_softmax(self.mask)
        return F.one_hot(self.mask.argmax(-1), self.mask.shape[1]).float()

    def _combine(self, rb, rs, sel):
        # One full copy of the rotated state per variable: copy v is the source's if
        # variable v moves, else the base's. Copy n_variables is always the base ("none").
        x = torch.where(sel[..., None], rs.unsqueeze(-2), rb.unsqueeze(-2))   # [B, V, D]
        x = torch.cat([x, rb.unsqueeze(-2)], dim=-2)                          # [B, V+1, D]
        return torch.einsum("bvd,dv->bd", x, self._assignment())

    def _reg_term(self):
        # Csordas's margin: push channels toward the "none" column. Shift-invariant per
        # row, so it only moves the relative preference, never the raw logit scale.
        return self.reg_loss * (self.mask[:, :-1].mean() - self.mask[:, -1].mean())

    @torch.no_grad()
    def get_parameters(self) -> dict[str, np.ndarray]:
        assign = F.one_hot(self.mask.argmax(-1), self.mask.shape[1]).float()
        return {
            **self._base_params(),
            "mask": assign.cpu().numpy(),               # [state_dim, n_variables+1]
            "var_sizes": assign[:, :-1].sum(0).cpu().numpy(),  # channels per variable
            "n_untouched": np.int64(int(assign[:, -1].sum().item())),
        }


class DASTrainer(InterventionTrainer):
    """Encodes base and source, swaps the selected variables, decodes toward the target."""

    def _run_batch(self, batch: dict[str, torch.Tensor]):
        base_ids = batch["input"].to(self.device)
        source_ids = batch["source"].to(self.device)
        target_ids = batch["target"].to(self.device)
        position = batch["position"].to(self.device)
        word_len = (batch["seq_len"] - 1).to(self.device)  # phonemes, excluding EOS

        h, c = get_encoder_hidden(self.repeat_model, base_ids, self.device)
        h_src, c_src = get_encoder_hidden(self.repeat_model, source_ids, self.device)
        h_mod, c_mod = self.intervention.intervene(h, c, h_src, c_src, position, word_len)
        logits = decode_with_hidden(self.repeat_model, h_mod, c_mod, target_ids,
                                    self.device, self.teacher_forcing)

        loss = self._compute_loss(logits, target_ids) + self.intervention.reg()
        return loss, logits.argmax(dim=-1), target_ids, batch["seq_len"]


def n_das_variables(max_position: int, var_ngram: int) -> int:
    """Variable ``v`` owns ``[v, v+var_ngram-1]``, so the windows run out that much early."""
    return max_position - var_ngram + 1


def build_das_intervention(method_cfg, data_cfg, hidden_size: int, max_position: int):
    """Construct the configured DAS variant from a ``MethodConfig`` + ``DataConfig``."""
    state_dim = hidden_size * (2 if method_cfg.state_mode == "concat" else 1)
    n_variables = n_das_variables(max_position, data_cfg.var_ngram)
    shared = dict(state_dim=state_dim, n_variables=n_variables, reg_loss=method_cfg.reg_loss,
                  state_mode=method_cfg.state_mode, edit_ngram=data_cfg.edit_ngram,
                  var_ngram=data_cfg.var_ngram)
    if method_cfg.is_autoseg:
        return AutoSegDAS(**shared)
    return DASIntervention(var_size=method_cfg.var_size, masking=method_cfg.masking, **shared)
