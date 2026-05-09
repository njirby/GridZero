"""Flat linear projection encoder — fallback when torch_geometric is unavailable."""
from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from gridzero.encoder.base import ObsEncoder
from gridzero.env.observation import ObsData


class FlatObsEncoder(ObsEncoder):
    """Projects the flat observation vector through an MLP into a fixed-length sequence.

    Output shape: [seq_len, d_model] where seq_len is a fixed hyperparameter.
    Simpler than the graph encoder but loses topology structure.

    Config keys:
        d_model:  output embedding dimension
        seq_len:  number of output tokens (fixed)
        n_layers: number of MLP hidden layers before the reshape
    """

    def __init__(self, cfg: DictConfig, flat_dim: int) -> None:
        super().__init__()
        self._d_model = cfg.d_model
        self._seq_len = cfg.seq_len
        out_dim = cfg.d_model * cfg.seq_len

        layers: list[nn.Module] = [nn.Linear(flat_dim, out_dim), nn.GELU()]
        for _ in range(cfg.n_layers - 1):
            layers += [nn.Linear(out_dim, out_dim), nn.GELU()]
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(cfg.d_model)

        # Per-feature normalization buffers — set via set_normalization_stats()
        self.register_buffer("_obs_mean", torch.zeros(flat_dim))
        self.register_buffer("_obs_std", torch.ones(flat_dim))

    def set_normalization_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Set per-feature normalization statistics.

        Features with std < eps are zeroed out (constant features).
        """
        eps = 1e-6
        safe_std = std.clone()
        safe_std[safe_std < eps] = 1.0
        self._obs_mean.copy_(mean)
        self._obs_std.copy_(safe_std)

    @property
    def output_dim(self) -> int:
        return self._d_model

    def forward(self, obs: ObsData) -> torch.Tensor:
        """Project flat obs to [seq_len, d_model].

        Args:
            obs: ObsData with obs.flat populated.

        Returns:
            Tensor of shape [seq_len, d_model].
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        x = torch.from_numpy(obs.flat).to(device=device, dtype=torch.float32).unsqueeze(0)
        x = (x - self._obs_mean) / self._obs_std
        x = x.to(dtype=dtype)
        x = self.mlp(x)                                       # [1, seq_len * d_model]
        x = x.view(self._seq_len, self._d_model)              # [seq_len, d_model]
        return self.norm(x)
