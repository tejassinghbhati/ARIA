"""Unit tests for GNNSceneGraphExtractor."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from aria.rl.extractors.gnn_extractor import GNNSceneGraphExtractor, _DenseGATConv, _GNNEncoder


# ---------------------------------------------------------------------------
# _DenseGATConv
# ---------------------------------------------------------------------------

class TestDenseGATConv:

    def test_output_shape(self) -> None:
        layer = _DenseGATConv(in_channels=9, out_channels=16, heads=4)
        x   = torch.randn(2, 8, 9)      # (B, N, C)
        adj = torch.ones(2, 8, 8)        # fully connected
        out = layer(x, adj)
        assert out.shape == (2, 8, 64)  # 4 heads * 16

    def test_sparse_adjacency(self) -> None:
        """Layer should handle zero-adjacency (isolated nodes) without NaN."""
        layer = _DenseGATConv(in_channels=9, out_channels=8, heads=2)
        x   = torch.randn(1, 4, 9)
        adj = torch.zeros(1, 4, 4)   # no edges
        out = layer(x, adj)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# _GNNEncoder
# ---------------------------------------------------------------------------

class TestGNNEncoder:

    def test_output_shape(self) -> None:
        enc = _GNNEncoder(in_channels=9, hidden_channels=32, num_layers=2,
                          heads=2, dropout=0.0, output_dim=64)
        x   = torch.randn(3, 16, 9)
        adj = torch.zeros(3, 16, 16)
        out = enc(x, adj)
        assert out.shape == (3, 64)

    def test_gradient_flow(self) -> None:
        enc = _GNNEncoder(in_channels=9, hidden_channels=16, num_layers=2,
                          heads=2, dropout=0.0, output_dim=32)
        x   = torch.randn(2, 8, 9, requires_grad=True)
        adj = torch.eye(8).unsqueeze(0).expand(2, -1, -1)
        loss = enc(x, adj).sum()
        loss.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# GNNSceneGraphExtractor (NavEnv-style observation space)
# ---------------------------------------------------------------------------

def _make_nav_obs_space(max_nodes=16, node_feat_dim=9):
    return gym.spaces.Dict({
        "node_features": gym.spaces.Box(
            low=-10, high=10, shape=(max_nodes, node_feat_dim), dtype=np.float32
        ),
        "adj_matrix": gym.spaces.Box(
            low=0, high=1, shape=(max_nodes, max_nodes), dtype=np.float32
        ),
        "agent_state": gym.spaces.Box(
            low=-20, high=20, shape=(6,), dtype=np.float32
        ),
        "goal_idx": gym.spaces.Box(
            low=0, high=max_nodes - 1, shape=(1,), dtype=np.int64
        ),
    })


class TestGNNSceneGraphExtractor:

    def test_output_shape(self) -> None:
        space = _make_nav_obs_space(max_nodes=16)
        ext   = GNNSceneGraphExtractor(space, features_dim=128, gnn_hidden=32,
                                       gnn_layers=2, gnn_heads=2, gnn_output_dim=64)
        obs = {
            "node_features": torch.randn(2, 16, 9),
            "adj_matrix":    torch.zeros(2, 16, 16),
            "agent_state":   torch.randn(2, 6),
            "goal_idx":      torch.zeros(2, 1, dtype=torch.long),
        }
        out = ext(obs)
        assert out.shape == (2, 128)

    def test_manip_obs_space(self) -> None:
        """Extractor should handle ManipEnv observation space without goal_idx."""
        space = gym.spaces.Dict({
            "node_features": gym.spaces.Box(low=-10, high=10, shape=(8, 9), dtype=np.float32),
            "adj_matrix":    gym.spaces.Box(low=0, high=1, shape=(8, 8), dtype=np.float32),
            "ee_pose":       gym.spaces.Box(low=-3, high=3, shape=(7,), dtype=np.float32),
            "ft_sensor":     gym.spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32),
            "joint_pos":     gym.spaces.Box(low=-4, high=4, shape=(7,), dtype=np.float32),
            "joint_vel":     gym.spaces.Box(low=-5, high=5, shape=(7,), dtype=np.float32),
        })
        ext = GNNSceneGraphExtractor(space, features_dim=256, gnn_hidden=32,
                                     gnn_layers=2, gnn_heads=2, gnn_output_dim=64)
        obs = {
            "node_features": torch.randn(1, 8, 9),
            "adj_matrix":    torch.zeros(1, 8, 8),
            "ee_pose":       torch.randn(1, 7),
            "ft_sensor":     torch.randn(1, 6),
            "joint_pos":     torch.randn(1, 7),
            "joint_vel":     torch.randn(1, 7),
        }
        out = ext(obs)
        assert out.shape == (1, 256)

    def test_no_nan_in_output(self) -> None:
        space = _make_nav_obs_space(max_nodes=8)
        ext   = GNNSceneGraphExtractor(space, features_dim=64, gnn_hidden=16,
                                       gnn_layers=2, gnn_heads=2, gnn_output_dim=32)
        obs = {
            "node_features": torch.randn(4, 8, 9),
            "adj_matrix":    torch.zeros(4, 8, 8),
            "agent_state":   torch.randn(4, 6),
            "goal_idx":      torch.randint(0, 7, (4, 1)),
        }
        out = ext(obs)
        assert not torch.isnan(out).any()
