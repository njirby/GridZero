"""Graph neural network encoder for grid2op observations (primary encoder)."""
from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from gridzero.encoder.base import ObsEncoder
from gridzero.env.observation import ObsData


class GraphObsEncoder(ObsEncoder):
    """Encodes the power grid as a graph, producing one embedding per substation.

    Architecture:
        node_feat_proj  →  [n_layers × TransformerConv]  →  output_proj
        input:  [n_substations, node_feature_dim]
        output: [n_substations, d_model]

    The output sequence length N = n_substations varies per environment but is
    static within a single training run.

    Config keys:
        d_model:   output embedding dimension
        n_layers:  number of TransformerConv message-passing layers
        n_heads:   attention heads per TransformerConv layer
        dropout:   dropout rate
        node_feat_dim: input node feature dimension (must match ObsParser)
    """

    def __init__(self, cfg: DictConfig, node_feat_dim: int) -> None:
        super().__init__()
        from torch_geometric.nn import TransformerConv

        self._d_model = cfg.d_model
        hidden = cfg.d_model

        self.input_proj = nn.Linear(node_feat_dim, hidden)

        self.conv_layers = nn.ModuleList(
            [
                TransformerConv(
                    in_channels=hidden,
                    out_channels=hidden // cfg.n_heads,
                    heads=cfg.n_heads,
                    dropout=cfg.dropout,
                    concat=True,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(cfg.n_layers)]
        )
        self.dropout = nn.Dropout(cfg.dropout)

    @property
    def output_dim(self) -> int:
        return self._d_model

    def forward(self, obs: ObsData) -> torch.Tensor:
        """Run GNN over the grid graph, return per-substation embeddings.

        Args:
            obs: Must have obs.graph populated (requires torch_geometric).

        Returns:
            Tensor of shape [n_substations, d_model].
        """
        assert obs.graph is not None, "GraphObsEncoder requires obs.graph (torch_geometric)"
        graph = obs.graph
        x = graph.x.float()
        edge_index = graph.edge_index

        x = self.input_proj(x)

        for conv, norm in zip(self.conv_layers, self.norms):
            residual = x
            x = conv(x, edge_index)
            x = self.dropout(x)
            x = norm(x + residual)

        return x  # [n_substations, d_model]
