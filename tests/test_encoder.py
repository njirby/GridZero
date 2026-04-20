"""Shape tests for observation encoders."""
import pytest
import numpy as np


def test_flat_encoder_output_shape():
    torch = pytest.importorskip("torch")
    from omegaconf import OmegaConf
    from gridzero.encoder.flat_encoder import FlatObsEncoder
    from gridzero.env.observation import ObsData

    cfg = OmegaConf.create({"d_model": 64, "seq_len": 8, "n_layers": 1})
    enc = FlatObsEncoder(cfg, flat_dim=100)
    obs = ObsData(
        flat=np.random.randn(100).astype(np.float32),
        graph=None,
        n_lines=5, n_loads=3, n_gens=2, n_substations=4,
    )
    out = enc(obs)
    assert out.shape == (8, 64)


def test_flat_encoder_output_dim_property():
    pytest.importorskip("torch")
    from omegaconf import OmegaConf
    from gridzero.encoder.flat_encoder import FlatObsEncoder

    cfg = OmegaConf.create({"d_model": 128, "seq_len": 16, "n_layers": 2})
    enc = FlatObsEncoder(cfg, flat_dim=50)
    assert enc.output_dim == 128


def test_graph_encoder_output_shape():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from omegaconf import OmegaConf
    from torch_geometric.data import Data
    from gridzero.encoder.graph_encoder import GraphObsEncoder
    from gridzero.env.observation import ObsData

    cfg = OmegaConf.create({"d_model": 64, "n_layers": 2, "n_heads": 4, "dropout": 0.0})
    enc = GraphObsEncoder(cfg, node_feat_dim=7)

    n_nodes = 14
    x = torch.zeros(n_nodes, 7)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    graph = Data(x=x, edge_index=edge_index, edge_attr=torch.zeros(3, 6))
    obs = ObsData(
        flat=np.zeros(100, dtype=np.float32),
        graph=graph,
        n_lines=3, n_loads=5, n_gens=3, n_substations=14,
    )
    out = enc(obs)
    assert out.shape == (n_nodes, 64)
