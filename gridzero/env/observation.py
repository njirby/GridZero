"""Observation parsing: flat vector and graph (PyG) representation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from torch_geometric.data import Data as PyGData
except ImportError:
    PyGData = None


@dataclass
class ObsData:
    """Parsed grid2op observation, ready for the encoder."""

    flat: np.ndarray        # full flat observation vector, shape [flat_dim]
    graph: "PyGData | None" # PyG graph: nodes=grid elements, edges=lines
    n_lines: int
    n_loads: int
    n_gens: int
    n_substations: int
    raw: object | None = None


class ObsParser:
    """Converts a raw grid2op Observation into ObsData.

    Caches grid topology metadata on construction so repeated parse() calls
    don't re-query the environment.
    """

    def __init__(self, g2op_env) -> None:
        """Cache static grid metadata from the environment.

        Args:
            g2op_env: A grid2op Environment (not the GymEnv wrapper).
        """
        obs_space = g2op_env.observation_space
        self._n_lines = g2op_env.n_line
        self._n_loads = g2op_env.n_load
        self._n_gens = g2op_env.n_gen
        self._n_substations = g2op_env.n_sub
        # Static line connectivity: which substations each line endpoint belongs to
        self._line_or_sub = obs_space.line_or_to_subid.copy()
        self._line_ex_sub = obs_space.line_ex_to_subid.copy()
        self._flat_dim: int | None = None  # set on first parse()

    @property
    def flat_dim(self) -> int:
        """Dimension of the flat observation vector (set after first parse)."""
        if self._flat_dim is None:
            raise RuntimeError("flat_dim is available after the first parse() call")
        return self._flat_dim

    @property
    def node_feature_dim(self) -> int:
        """Feature dimension per graph node (substation-level features)."""
        # v, theta, load_p, load_q, gen_p, gen_q, rho_max
        return 7

    def parse(self, obs) -> ObsData:
        """Parse a grid2op Observation into an ObsData.

        Args:
            obs: grid2op Observation object from env.step() or env.reset().

        Returns:
            ObsData with flat vector and (if torch_geometric available) graph.
        """
        flat = obs.to_vect()
        self._flat_dim = flat.shape[0]

        graph = None
        if PyGData is not None:
            graph = self._build_graph(obs)

        return ObsData(
            flat=flat,
            graph=graph,
            n_lines=self._n_lines,
            n_loads=self._n_loads,
            n_gens=self._n_gens,
            n_substations=self._n_substations,
            raw=obs,
        )

    def _build_graph(self, obs) -> "PyGData":
        """Construct a PyG graph from the observation.

        Nodes: one per substation (aggregated features).
        Edges: one per connected powerline (bidirectional).
        Node features: [v, theta, sum_load_p, sum_load_q, sum_gen_p, sum_gen_q, max_rho].
        Edge features: [p_or, q_or, p_ex, q_ex, rho, line_status].
        """
        import torch
        from torch_geometric.data import Data

        n = self._n_substations

        # TODO: aggregate per-substation features from obs
        x = torch.zeros(n, self.node_feature_dim)

        # Build edge index from connected lines only
        connected = obs.line_status  # bool array [n_lines]
        src = self._line_or_sub[connected].tolist()
        dst = self._line_ex_sub[connected].tolist()
        # Bidirectional edges
        edge_index = torch.tensor(
            [src + dst, dst + src], dtype=torch.long
        )

        # TODO: populate edge_attr from obs line flows
        n_connected = int(connected.sum())
        edge_attr = torch.zeros(n_connected * 2, 6)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
