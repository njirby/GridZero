"""Abstract base class for grid observation encoders."""
from __future__ import annotations

import torch
import torch.nn as nn

from gridzero.env.observation import ObsData


class ObsEncoder(nn.Module):
    """Base class for all observation encoders.

    All encoders take an ObsData and return a variable-length sequence of
    embeddings that are injected into vllm as multimodal prefix tokens.
    """

    @property
    def output_dim(self) -> int:
        """Embedding dimension (d_model). Must match the policy's hidden size."""
        raise NotImplementedError

    def forward(self, obs: ObsData) -> torch.Tensor:
        """Encode a grid observation into a sequence of embeddings.

        Args:
            obs: Parsed observation from ObsParser.

        Returns:
            Float tensor of shape [N, d_model]. N may vary across observations
            (e.g., one embedding per connected substation or per grid element).
        """
        raise NotImplementedError
