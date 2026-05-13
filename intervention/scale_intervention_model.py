
import numpy as np
import torch
import torch.nn as nn

class ScaleIntervention(nn.Module):
    """
    Scale-based intervention on LSTM states.

    state_mode:
      - c: intervene only on c
      - h: intervene only on h
      - both: intervene on both h and c with same transform
      - concat: intervene on concatenated [h, c]

    scale_param:
      - per_pos: per-position scale parameter
      - onion: g * gamma^pos + pos * beta + b
      - linear: pos * beta + b
      - one_scale: scale * w(position)
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        state_mode: str = "c",
        scale_param: str = "onion",
        max_position: int = 15,
        pretrained_embedding: nn.Embedding | None = None,
        freeze_embedding: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_mode = state_mode
        self.scale_param = scale_param
        self.max_position = max_position
        self.state_dim = hidden_size * 2 if self.state_mode == "concat" else hidden_size

        self.embedding = self._build_embedding(vocab_size, pretrained_embedding, freeze_embedding)
        self.token_proj = self._build_token_projection(pretrained_embedding)
        self._build_scale_parameters()

    def _build_embedding(
        self,
        vocab_size: int,
        pretrained_embedding: nn.Embedding | None,
        freeze_embedding: bool,
    ) -> nn.Embedding:
        if pretrained_embedding is None:
            embedding = nn.Embedding(vocab_size, self.state_dim)
        else:
            embed_dim = pretrained_embedding.embedding_dim
            embedding = nn.Embedding(vocab_size, embed_dim)
            embedding.weight.data.copy_(pretrained_embedding.weight.data)
        embedding.weight.requires_grad = not freeze_embedding 
        return embedding

    def _build_token_projection(self, pretrained_embedding: nn.Embedding | None) -> nn.Module:
        if pretrained_embedding is None:
            return nn.Identity()

        embed_dim = pretrained_embedding.embedding_dim
        if embed_dim == self.state_dim:
            return nn.Identity()
        return nn.Linear(embed_dim, self.state_dim)

    def _build_scale_parameters(self) -> None:
        if self.scale_param == "per_pos":
            self.scale = nn.Parameter(torch.ones(self.max_position, self.state_dim))
        elif self.scale_param == "one_scale":
            self.scale = nn.Parameter(torch.ones(self.state_dim))
            self.position_weight = nn.Parameter(torch.ones(self.max_position, 1))
        elif self.scale_param == "onion":
            self.gamma = nn.Parameter(torch.ones(self.state_dim) * 0.9)
            self.beta = nn.Parameter(torch.zeros(self.state_dim))
            self.g = nn.Parameter(torch.ones(self.state_dim))
            self.b = nn.Parameter(torch.zeros(self.state_dim))
        elif self.scale_param == "linear":
            self.beta = nn.Parameter(torch.zeros(self.state_dim))
            self.b = nn.Parameter(torch.zeros(self.state_dim))
        # elif self.scale_param == "spiral":
        #     self.base_scale = nn.Parameter(torch.randn(self.state_dim))
        #     self.decay_rate = nn.Parameter(torch.tensor(-0.1)) # For convergence
        #     self.angle_rate = nn.Parameter(torch.tensor(0.5)) # Controls rotation speed
        else:
            raise ValueError("scale_param must be one of 'per_pos', 'onion', 'linear', or 'one_scale'")

    def _apply_scale(self, position: torch.Tensor) -> torch.Tensor:
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
        # if self.scale_param == "spiral":
        #     return  torch.exp(pos * self.decay_rate) * self._rotate(self.base_scale, pos * self.angle_rate)
        raise ValueError("scale_param must be one of 'per_pos', 'onion', 'linear', or 'one_scale'")

    def _apply_state(
        self,
        state: torch.Tensor,
        old_token: torch.Tensor,
        new_token: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embedding(old_token).tanh()
        y = self.embedding(new_token).tanh()
        delta = y - x
        delta = self.token_proj(delta)
        scale = self._apply_scale(position)
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
        p = torch.tensor(pos, dtype=torch.float, device=device)
        if self.scale_param == "onion":
            return self.g * self.gamma.pow(p) + p * self.beta + self.b
        if self.scale_param == "linear":
            return p * self.beta + self.b
        if self.scale_param == "one_scale":
            return (self.scale * self.position_weight[pos]).squeeze()
        raise ValueError("scale_param must be one of 'per_pos', 'onion', 'linear', or 'one_scale'")

    def scale_parameters(self) -> dict[str, np.ndarray]:
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
            base["scales"] = np.stack(
                [self.get_scale(pos).detach().cpu().numpy() for pos in range(self.max_position)]
            )
        elif self.scale_param == "onion":
            base.update(
                {
                    "gamma": self.gamma.detach().cpu().numpy(),
                    "beta": self.beta.detach().cpu().numpy(),
                    "g": self.g.detach().cpu().numpy(),
                    "b": self.b.detach().cpu().numpy(),
                    "scales": np.stack(
                        [self.get_scale(pos).detach().cpu().numpy() for pos in range(self.max_position)]
                    ),
                }
            )
        elif self.scale_param == "linear":
            base.update(
                {
                    "beta": self.beta.detach().cpu().numpy(),
                    "b": self.b.detach().cpu().numpy(),
                    "scales": np.stack(
                        [self.get_scale(pos).detach().cpu().numpy() for pos in range(self.max_position)]
                    ),
                }
            )
        else:
            raise ValueError("scale_param must be one of 'per_pos', 'one_scale', 'onion', 'linear'")
        return base