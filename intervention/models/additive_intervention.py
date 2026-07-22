"""Scale-based intervention: state <- state + scale(position) * (emb(new) - emb(old)).

``ScaleIntervention`` supports several parameterisations of ``scale(position)`` (see the
class docstring); ``build_scale_intervention`` constructs one from a ``MethodConfig``,
resolving the token embedding (pretrained / delta-stats / random) along the way.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleIntervention(nn.Module):
    """
    Scale-based intervention on LSTM states.

    state_mode:
      - c: intervene only on c
      - h: intervene only on h
      - both: intervene on both h and c with same transform
      - concat: intervene on concatenated [h, c]

    scale_param:
      update
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        state_mode: str = "c",
        scale_param: str = "onion",
        max_position: int = 15,
        pretrained_embedding: nn.Embedding | None = None,
        train_embedding: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_mode = state_mode
        self.scale_param = scale_param
        self.max_position = max_position
        self.state_dim = hidden_size * 2 if self.state_mode == "concat" else hidden_size

        self.embedding = self._build_embedding(vocab_size, pretrained_embedding, train_embedding)
        self.token_proj = self._build_token_projection(pretrained_embedding)
        self.token_proj.requires_grad_(False) 
        self._build_scale_parameters()

    def _build_embedding(
        self,
        vocab_size: int,
        pretrained_embedding: nn.Embedding | None,
        train_embedding: bool,
    ) -> nn.Embedding:
        if pretrained_embedding is None:
            embedding = nn.Embedding(vocab_size, self.state_dim)
        else:
            embed_dim = pretrained_embedding.embedding_dim
            embedding = nn.Embedding(vocab_size, embed_dim)
            embedding.weight.data.copy_(pretrained_embedding.weight.data)
        embedding.weight.requires_grad = train_embedding
        return embedding

    def _build_token_projection(self, pretrained_embedding: nn.Embedding | None) -> nn.Module:
        if pretrained_embedding is None:
            return nn.Identity()

        embed_dim = pretrained_embedding.embedding_dim
        if embed_dim == self.state_dim:
            return nn.Identity()
        # Parameter-free projection for 'concat' mode
        if self.state_dim % embed_dim == 0:
            repeats = self.state_dim // embed_dim
            
            class RepeatProjection(nn.Module):
                def forward(self, x):
                    # Repeats the last dimension (e.g., turns 128D into 256D by duplicating it)
                    repeats_tuple = [1] * (x.dim() - 1) + [repeats]
                    return x.repeat(*repeats_tuple)
                    
            return RepeatProjection()

        # Fallback for mismatched dimensions that aren't multiples
        return nn.Linear(embed_dim, self.state_dim)

    def _build_scale_parameters(self) -> None:
        if self.scale_param == "per_pos":
            self.scale = nn.Parameter(torch.ones(self.max_position, self.state_dim))
        elif self.scale_param == "one_scale":
            self.scale = nn.Parameter(torch.ones(self.state_dim))
            self.position_weight = nn.Parameter(torch.ones(self.max_position, 1))
        elif self.scale_param == "onion":
            self.gamma = nn.Parameter(torch.ones(self.state_dim) * 0.9)
            self.beta = nn.Parameter(torch.ones(self.state_dim))
            self.g = nn.Parameter(torch.ones(self.state_dim))
            self.b = nn.Parameter(torch.zeros(self.state_dim))
        elif self.scale_param == "linear":
            self.beta = nn.Parameter(torch.ones(self.state_dim))
            self.b = nn.Parameter(torch.zeros(self.state_dim))
        elif self.scale_param.startswith("low_rank"):
            # r = 4  # rank 
            rank = int(self.scale_param.split("-")[-1]) if "-" in self.scale_param else 4
            # r basis directions in state space
            self.scale_basis  = nn.Parameter(torch.randn(self.state_dim, rank) * 0.1)
            # per-position coefficients over those r directions
            self.scale_coeffs = nn.Parameter(torch.randn(self.max_position, rank) * 0.1)
            
        elif self.scale_param == "plane_spiral":
            d = self.state_dim
            # Constant baseline direction
            self.v_const = nn.Parameter(torch.randn(d) * 0.1)
            # Two orthogonal basis vectors of the spiral plane
            self.u1_raw  = nn.Parameter(torch.randn(d) * 0.1)
            self.u2_raw  = nn.Parameter(torch.randn(d) * 0.1)
            # Radial parameters
            self.g       = nn.Parameter(torch.tensor(1.0))
            self.gamma   = nn.Parameter(torch.tensor(1.5))   # sigmoid →  will adapt
            # Angular parameters
            self.alpha     = nn.Parameter(torch.tensor(0.5))
            self.gamma_rot = nn.Parameter(torch.tensor(3.0)) # sigmoid slow decel
            self.phi0      = nn.Parameter(torch.tensor(0.1))

        elif self.scale_param == "expo_decay":
            self.g     = nn.Parameter(torch.ones(self.state_dim))
            self.gamma = nn.Parameter(torch.ones(self.state_dim) * 1.5)  # sigmoid → ~0.82
            self.b     = nn.Parameter(torch.zeros(self.state_dim))
        elif self.scale_param == "expo_unbounded":
            self.g     = nn.Parameter(torch.ones(self.state_dim))
            self.gamma = nn.Parameter(torch.ones(self.state_dim) * 0.5) 
            self.b     = nn.Parameter(torch.zeros(self.state_dim))
        elif self.scale_param == "spiral_expo":
            # Decelerating spiral: functional form matching per_pos behavior
            assert self.state_dim % 2 == 0
            n_pairs = self.state_dim // 2
            # Scale parameters (decaying onion, no linear term)
            # scale(pos) = softplus(g) * sigmoid(gamma)^pos + b
            # softplus/sigmoid ensure γ ∈ (0,1) and g > 0
            self.g      = nn.Parameter(torch.ones(self.state_dim) * 0.5)
            self.gamma  = nn.Parameter(torch.ones(self.state_dim) * 1.5)  # → sigmoid → (0,1)
            self.b      = nn.Parameter(torch.zeros(self.state_dim))
            # Rotation parameters (decelerated RoPE)
            self.alpha        = nn.Parameter(torch.ones(n_pairs) * 0.3)
            self.gamma_rot_raw = nn.Parameter(torch.zeros(n_pairs))  # → sigmoid → (0,1)
            self.phi0         = nn.Parameter(torch.zeros(n_pairs))
        elif self.scale_param == "spiral_rope":
            # RoPE-style: learned per-plane frequencies
            assert self.state_dim % 2 == 0, "state_dim must be even for spiral_rope"
            n_pairs = self.state_dim // 2
            # Learnable frequencies, one per pair of dimensions
            self.rot_freq = nn.Parameter(torch.ones(n_pairs) * 0.1)
            # onion
            # self.gamma = nn.Parameter(torch.ones(self.state_dim) * 1.5)
            # self.g     = nn.Parameter(torch.ones(self.state_dim))
            # self.b     = nn.Parameter(torch.zeros(self.state_dim))
            self.gamma = nn.Parameter(torch.ones(self.state_dim) * 0.9)
            self.beta = nn.Parameter(torch.ones(self.state_dim))
            self.g = nn.Parameter(torch.ones(self.state_dim))
            self.b = nn.Parameter(torch.zeros(self.state_dim))

        elif self.scale_param == "spiral_lie":
            # Lie algebra: skew-symmetric A, R(pos) = expm(pos * A)
            d = self.state_dim
            # Store upper-triangle entries; reconstruct A = W - W^T
            self.skew_weights = nn.Parameter(torch.zeros(d, d))
            # Exponential-decay scale
            self.gamma = nn.Parameter(torch.ones(d) * 1.5)
            self.g     = nn.Parameter(torch.ones(d))
            self.b     = nn.Parameter(torch.zeros(d))
        else:
            raise ValueError("scale_param must be one of 'per_pos', 'one_scale', 'onion', 'linear', 'low_rank' or 'low_rank-<rank>', 'plane_spiral', 'spiral_expo', 'spiral_rope', or 'spiral_lie'")
    def _decelerated_rope_rotate(self, x, position):
        pos = position.float()  # (B,)
        # expo rotate
        # alpha: (n_pairs,), gamma_rot: (n_pairs,) ∈ (0,1)
        gamma_rot = torch.sigmoid(self.gamma_rot_raw)          # constrain to (0,1)
        # theta = self.alpha * (1 - gamma_rot.unsqueeze(0).pow(pos.unsqueeze(1))) \
        #         / (1 - gamma_rot.unsqueeze(0) + 1e-6)
        theta = self.alpha *  gamma_rot.unsqueeze(0).pow(pos.unsqueeze(1)) + self.phi0.unsqueeze(0)  # (B, n_pairs)
                

        cos_a, sin_a = torch.cos(theta), torch.sin(theta)
        x_even, x_odd = x[:, 0::2], x[:, 1::2]
        x_rot = torch.stack([
            x_even * cos_a - x_odd * sin_a,
            x_even * sin_a + x_odd * cos_a,
        ], dim=-1)
        return x_rot.reshape(x.shape)
    def _rope_rotate(self, x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE-style rotation to x using learned frequencies.
        x: (batch, state_dim)
        position: (batch,)
        Returns rotated x of same shape.
        """
        pos = position.float()  # (batch,)
        # θ_i · pos  →  (batch, n_pairs)
        angles = torch.outer(pos, self.rot_freq)          # (batch, n_pairs)
        cos_a  = torch.cos(angles)                        # (batch, n_pairs)
        sin_a  = torch.sin(angles)                        # (batch, n_pairs)

        # Split x into even/odd pairs
        x_even = x[:, 0::2]  # (batch, n_pairs)
        x_odd  = x[:, 1::2]  # (batch, n_pairs)

        # Apply 2D rotation in each pair's plane
        x_even_rot = x_even * cos_a - x_odd * sin_a
        x_odd_rot  = x_even * sin_a + x_odd * cos_a

        # Interleave back
        x_rot = torch.stack([x_even_rot, x_odd_rot], dim=-1)  # (batch, n_pairs, 2)
        return x_rot.reshape(x.shape)                          # (batch, state_dim)


    def _lie_rotate(self, x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        """
        Apply Lie algebra rotation: R(pos) = expm(pos * A), A skew-symmetric.
        x: (batch, state_dim)
        position: (batch,)
        """
        # Build skew-symmetric A from free parameters
        A = self.skew_weights - self.skew_weights.T   # (d, d), skew-symmetric

        # R(pos) = expm(pos * A) — one rotation matrix per position in batch
        # Group by unique positions to avoid redundant expm calls
        results = torch.empty_like(x)
        device = x.device
        for p in position.unique():
            mask = (position == p)
            scaled_A = p.float() * A
            if device.type == "mps":
                scaled_A = scaled_A.to("cpu")
                R_p = torch.matrix_exp(scaled_A).to(device)
            else:
                R_p = torch.matrix_exp(scaled_A)
            results[mask] = x[mask] @ R_p.T          # apply rotation
        return results

    def _expo_decay_scale(
        self,
        position: torch.Tensor,
        gamma: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        pos = position[:, None].float()
        gamma = torch.sigmoid(gamma)
        g = F.softplus(g)
        return g * gamma.pow(pos) + b

    def _scale_for_positions(self, position: torch.Tensor) -> torch.Tensor:
        if self.scale_param == "per_pos":
            return self.scale[position]

        pos = position[:, None].float()
        if self.scale_param == "onion":
            return self.g * self.gamma.pow(pos) + pos * self.beta + self.b
        if self.scale_param == "linear":
            return pos * self.beta + self.b
        if self.scale_param == "one_scale":
            weight = self.position_weight[position]
            return self.scale * weight
        if self.scale_param in ("spiral_rope", "spiral_lie"):
            # return self.g * self.gamma.pow(pos) + pos * self.beta + self.b
            return self._expo_decay_scale(position, self.gamma, self.g, self.b)
        if self.scale_param.startswith("low_rank"):
            # (max_pos, r) @ (r, state_dim) → lookup for this position
            coeff = self.scale_coeffs[position]          # (B, r)
            basis = self.scale_basis                     # (state_dim, r)
            return coeff @ basis.T                       # (B, state_dim)
        if self.scale_param == "plane_spiral":
            pos = position.float()   # (B,)

            # Enforce orthonormal basis via Gram-Schmidt
            u1 = F.normalize(self.u1_raw, dim=0)
            u2 = self.u2_raw - (self.u2_raw @ u1) * u1
            u2 = F.normalize(u2, dim=0)

            # Radial: decay to non-zero baseline
            gamma = torch.sigmoid(self.gamma)
            g     = F.softplus(self.g)
            r     = g * gamma.pow(pos)                   # (B,)

            # Angular: decelerated rotation
            # gamma_rot = torch.sigmoid(self.gamma_rot)
            # theta = self.alpha * (1 - gamma_rot.pow(pos)) \
            #         / (1 - gamma_rot + 1e-6) + self.phi0  # (B,)
            theta = torch.outer(pos, self.gamma_rot) 

            # Spiral in the (u1, u2) plane + constant offset
            spiral = (r * torch.cos(theta)).unsqueeze(1) * u1.unsqueeze(0) \
                + (r * torch.sin(theta)).unsqueeze(1) * u2.unsqueeze(0)
            return self.v_const.unsqueeze(0) + spiral     # (B, state_dim)
        
        if self.scale_param == "expo_decay":
            return self._expo_decay_scale(position, self.gamma, self.g, self.b)
        if self.scale_param == "expo_unbounded":
             # Like expo_decay but without sigmoid/softplus constraints, allowing non-monotone and >1 scaling
             return self.g * self.gamma.pow(pos) + self.b
        
        if self.scale_param == "spiral_expo":
            return self._expo_decay_scale(position, self.gamma, self.g, self.b)          # decaying, no linear term

        raise ValueError("scale_param must be one of 'per_pos', 'one_scale', 'onion', 'linear', 'low_rank', 'plane_spiral', 'spiral_expo', 'spiral_rope', or 'spiral_lie'")


    def _effective_scale_for_positions(self, position: torch.Tensor) -> torch.Tensor:
        """Final per-position scale vector actually applied to the delta, *including* the
        spiral rotation. ``_apply_state`` and ``get_parameters`` both go through this, so
        the saved ``scales`` are exactly what the trained intervention uses."""
        scale = self._scale_for_positions(position)
        # Rotate the scale vector for spiral methods, not the delta,
        # so the intervention follows the plane_spiral-style parameterization.
        if self.scale_param == "spiral_rope":
            scale = self._rope_rotate(scale, position)
        elif self.scale_param == "spiral_lie":
            scale = self._lie_rotate(scale, position)
        elif self.scale_param == "spiral_expo":
            scale = self._decelerated_rope_rotate(scale, position)
        return scale

    def _apply_state(
        self,
        state: torch.Tensor,
        old_token: torch.Tensor,
        new_token: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embedding(old_token).tanh()
        y = self.embedding(new_token).tanh()
        delta = self.token_proj(y - x)
        scale = self._effective_scale_for_positions(position)
        return state + scale * delta


    def intervene(
        self,
        h: torch.Tensor,
        c: torch.Tensor,
        old_token: torch.Tensor,
        new_token: torch.Tensor,
        position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.state_mode == "c":
            return h, self._apply_state(c, old_token, new_token, position)
        if self.state_mode == "h":
            return self._apply_state(h, old_token, new_token, position), c
        if self.state_mode == "both":
            return (
                self._apply_state(h, old_token, new_token, position),
                self._apply_state(c, old_token, new_token, position),
            )
        if self.state_mode == "concat":
            hc = torch.cat([h, c], dim=-1)
            hc_mod = self._apply_state(hc, old_token, new_token, position)
            h_mod, c_mod = hc_mod.chunk(2, dim=-1)
            return h_mod, c_mod
        raise ValueError("state_mode must be one of 'c', 'h', 'both', or 'concat'")

    def forward(
        self,
        state: torch.Tensor,
        old_token: torch.Tensor,
        new_token: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        return self._apply_state(state, old_token, new_token, position)

    def get_scale(self, pos: int) -> torch.Tensor:
        if self.scale_param == "per_pos":
            return self.scale[pos]
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        pos_t = torch.tensor([pos], dtype=torch.long, device=device)
        return self._effective_scale_for_positions(pos_t).squeeze(0)

    def get_parameters(self) -> dict[str, np.ndarray]:
        base = {
            "scale_param": self.scale_param,
            "state_mode": self.state_mode,
            "state_dim": self.state_dim,
            "embedding": self.embedding.weight.detach().cpu().numpy(),
        }
        if self.scale_param == "per_pos":
            base["scales"] = self.scale.detach().cpu().numpy()
        elif self.scale_param == "one_scale":
            base["scale"] = self.scale.detach().cpu().numpy()
            base["position_weight"] = self.position_weight.detach().cpu().numpy()
            positions = torch.arange(self.max_position, device=self.scale.device)
            base["scales"] = self._effective_scale_for_positions(positions).detach().cpu().numpy()

        elif self.scale_param == "onion":
            positions = torch.arange(self.max_position, device=self.gamma.device)
            base.update(
                {
                    "gamma": self.gamma.detach().cpu().numpy(),
                    "beta": self.beta.detach().cpu().numpy(),
                    "g": self.g.detach().cpu().numpy(),
                    "b": self.b.detach().cpu().numpy(),
                    "scales": self._effective_scale_for_positions(positions).detach().cpu().numpy(),
                }
            )
        elif self.scale_param == "spiral_rope":
            positions = torch.arange(self.max_position, device=self.gamma.device)
            base.update(
                {
                    "gamma": self.gamma.detach().cpu().numpy(),
                    "g": self.g.detach().cpu().numpy(),
                    "b": self.b.detach().cpu().numpy(),
                    "rot_freq": self.rot_freq.detach().cpu().numpy(),
                    "scales": self._effective_scale_for_positions(positions).detach().cpu().numpy(),
                }
            )
        elif self.scale_param == "spiral_lie":
            positions = torch.arange(self.max_position, device=self.gamma.device)
            base.update(
                {
                    "gamma": self.gamma.detach().cpu().numpy(),
                    "g": self.g.detach().cpu().numpy(),
                    "b": self.b.detach().cpu().numpy(),
                    "skew_weights": self.skew_weights.detach().cpu().numpy(),
                    "scales": self._effective_scale_for_positions(positions).detach().cpu().numpy(),
                }
            )
        elif self.scale_param.startswith("spiral_expo"):
            positions = torch.arange(self.max_position, device=self.gamma.device)
            base.update(
                {
                    "gamma": self.gamma.detach().cpu().numpy(),
                    "g": self.g.detach().cpu().numpy(),
                    "b": self.b.detach().cpu().numpy(),
                    "alpha": self.alpha.detach().cpu().numpy(),
                    "gamma_rot": torch.sigmoid(self.gamma_rot_raw.detach()).cpu().numpy(),
                    "phi0": self.phi0.detach().cpu().numpy(),
                    "scales": self._effective_scale_for_positions(positions).detach().cpu().numpy(),
                }
            )
        elif self.scale_param == "plane_spiral":
            device = self.v_const.device
            positions = torch.arange(self.max_position, device=device)
            base.update(
                {
                    "v_const": self.v_const.detach().cpu().numpy(),
                    "u1_raw": self.u1_raw.detach().cpu().numpy(),
                    "u2_raw": self.u2_raw.detach().cpu().numpy(),
                    "g": self.g.detach().cpu().numpy(),
                    "gamma": self.gamma.detach().cpu().numpy(),
                    "alpha": self.alpha.detach().cpu().numpy(),
                    "gamma_rot": self.gamma_rot.detach().cpu().numpy(),
                    "phi0": self.phi0.detach().cpu().numpy(),
                    "scales": self._effective_scale_for_positions(positions).detach().cpu().numpy(),
                }
            )
        elif self.scale_param == "linear":
            positions = torch.arange(self.max_position, device=self.beta.device)
            base.update(
                {
                    "beta": self.beta.detach().cpu().numpy(),
                    "b": self.b.detach().cpu().numpy(),
                    "scales": self._effective_scale_for_positions(positions).detach().cpu().numpy(),
                }
            )
        elif self.scale_param.startswith("low_rank"):
            positions = torch.arange(self.max_position, device=self.scale_coeffs.device)
            base.update(
                {
                    "scale_basis": self.scale_basis.detach().cpu().numpy(),
                    "scale_coeffs": self.scale_coeffs.detach().cpu().numpy(),
                    "scales": self._effective_scale_for_positions(positions).detach().cpu().numpy(),
                }
            )
        elif self.scale_param in ("expo_decay", "expo_unbounded"):
            positions = torch.arange(self.max_position, device=self.gamma.device)
            base.update(
                {
                    "gamma": self.gamma.detach().cpu().numpy(),
                    "g": self.g.detach().cpu().numpy(),
                    "b": self.b.detach().cpu().numpy(),
                    "scales": self._effective_scale_for_positions(positions).detach().cpu().numpy(),
                }
            )
        else:
            raise ValueError("scale_param must be one of 'per_pos', 'one_scale', 'onion', 'linear', 'low_rank' or 'low_rank-<rank>', 'plane_spiral', 'spiral_expo', 'spiral_rope', or 'spiral_lie'")
        return base


# Token-embedding statistics used by the delta_* initialisations.
_STATS_DIR = Path(__file__).resolve().parents[1] / "states_ds"
_EMBED_STATS_PATH = _STATS_DIR / "phoneme_state_embeddings.npz"


def _resolve_embedding(embedding_init: str, repeat_model, phoneme_to_id) -> nn.Embedding | None:
    """Pick the token embedding that seeds the (old -> new) delta."""
    if embedding_init == "pretrained":
        return repeat_model.encoder.embedding
    if embedding_init == "none":
        return None
    from intervention.data.delta_embeddings import load_token_embedding_from_stats
    return load_token_embedding_from_stats(
        embedding_init, _EMBED_STATS_PATH, phoneme_to_id, repeat_model.encoder.embedding
    )


def _resolve_ngram_embedding(embedding_init: str, ngram_vocab, phoneme_to_id) -> nn.Embedding | None:
    """delta_* over the n-gram vocab (row i = statistic for the n-gram with id i)."""
    if embedding_init == "none":
        return None
    from intervention.data.delta_embeddings import load_ngram_embedding_from_stats
    id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}
    labels = [None] * len(ngram_vocab)
    for gram, idx in ngram_vocab.items():
        labels[idx] = " ".join(id_to_phoneme[t] for t in gram)
    n = len(next(iter(ngram_vocab)))
    return load_ngram_embedding_from_stats(
        embedding_init, _STATS_DIR / f"ngram{n}_state_embeddings.npz", labels
    )


def build_scale_intervention(method_cfg, repeat_model, hidden_size, max_position,
                             phoneme_to_id, ngram_vocab=None):
    """Construct a ``ScaleIntervention`` from a ``MethodConfig`` (``model`` is the
    parameterisation name, e.g. ``onion`` or ``low_rank-8``).

    With ``ngram_vocab`` (edit_ngram > 1) the embedding table covers the attested n-gram
    inventory: learned from scratch (embedding_init='none') 
    initialised from the n-gram delta statistics ('delta_*') and 'pretrained' is rejected by config validation."""
    if ngram_vocab is not None:
        vocab_size = len(ngram_vocab)
        pretrained = _resolve_ngram_embedding(method_cfg.embedding_init, ngram_vocab, phoneme_to_id)
    else:
        vocab_size = len(phoneme_to_id)
        pretrained = _resolve_embedding(method_cfg.embedding_init, repeat_model, phoneme_to_id)
    return ScaleIntervention(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        state_mode=method_cfg.state_mode,
        scale_param=method_cfg.model,
        max_position=max_position,
        pretrained_embedding=pretrained,
        train_embedding=method_cfg.train_embedding,
    )