"""
aria.rl.extractors.gnn_extractor
==================================
Stable-Baselines3 custom feature extractor that processes a scene-graph
Dict observation through a Graph Attention Network (GAT) and outputs a
fixed-length embedding vector for the PPO/SAC MLP actor-critic heads.

Architecture
------------
    node_features (B, N, D_node)
    adj_matrix    (B, N, N)
             ↓
    GATConv × num_layers           ← message passing on dense adjacency
             ↓
    Global mean pool               ← (B, hidden_channels * heads)
             ↓
    MLP projection                 ← (B, output_dim)
             ↓
    Concat with flat auxiliary     ← agent_state / ee_pose / ft_sensor …
             ↓
    Final linear                   ← (B, features_dim)

Usage with SB3
--------------
    policy_kwargs = dict(
        features_extractor_class=GNNSceneGraphExtractor,
        features_extractor_kwargs=dict(features_dim=512),
    )
    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs)
"""

from __future__ import annotations

import logging
from typing import Dict, List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lightweight dense GAT implementation
# (avoids the PyG sparse engine for SB3 batch compatibility)
# ---------------------------------------------------------------------------

class _DenseGATConv(nn.Module):
    """
    Multi-head graph attention convolution over dense (B, N, N) adjacency.

    Parameters
    ----------
    in_channels  : int
    out_channels : int  — per head
    heads        : int
    dropout      : float
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels

        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.a = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        nn.init.xavier_uniform_(self.a)
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x   : (B, N, in_channels)
        adj : (B, N, N) — binary or weighted adjacency

        Returns
        -------
        out : (B, N, heads * out_channels)
        """
        B, N, _ = x.shape
        H, D = self.heads, self.out_channels

        # Linear transform: (B, N, H*D)
        Wx = self.W(x).view(B, N, H, D)                               # (B, N, H, D)

        # Attention logits
        Wx_i = Wx.unsqueeze(2).expand(B, N, N, H, D)                  # (B, N, N, H, D)
        Wx_j = Wx.unsqueeze(1).expand(B, N, N, H, D)
        cat  = torch.cat([Wx_i, Wx_j], dim=-1)                        # (B, N, N, H, 2D)
        e    = self.leaky((cat * self.a).sum(dim=-1))                  # (B, N, N, H)

        # Mask by adjacency (add self-loops)
        adj_self = adj + torch.eye(N, device=adj.device).unsqueeze(0)
        mask     = (adj_self == 0).unsqueeze(-1)                       # (B, N, N, 1)
        e        = e.masked_fill(mask, float("-inf"))

        alpha    = torch.softmax(e, dim=2)                             # (B, N, N, H)
        alpha    = self.dropout(alpha)

        # Aggregate
        out = torch.einsum("bnjh,bnjhd->bnhd", alpha, Wx_j)           # (B, N, H, D)
        return out.reshape(B, N, H * D)


# ---------------------------------------------------------------------------
# GNN encoder module
# ---------------------------------------------------------------------------

class _GNNEncoder(nn.Module):
    """Stack of DenseGATConv layers with skip connections."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        heads: int,
        dropout: float,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        c_in = in_channels
        for _ in range(num_layers):
            self.layers.append(_DenseGATConv(c_in, hidden_channels, heads=heads, dropout=dropout))
            c_in = hidden_channels * heads

        self.proj = nn.Sequential(
            nn.Linear(c_in, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Returns global mean-pooled feature: (B, output_dim).
        """
        for layer in self.layers:
            x = torch.relu(layer(x, adj))   # (B, N, H*D)
        x = x.mean(dim=1)                   # (B, H*D)  — global mean pool
        return self.proj(x)                 # (B, output_dim)


# ---------------------------------------------------------------------------
# SB3 Feature Extractor
# ---------------------------------------------------------------------------

class GNNSceneGraphExtractor(BaseFeaturesExtractor):
    """
    Custom SB3 BaseFeaturesExtractor for scene-graph Dict observations.

    Handles both navigation (NavEnv) and manipulation (ManipEnv) obs spaces:
    - Always processes: node_features + adj_matrix via GNN
    - NavEnv extras:   agent_state, goal_idx
    - ManipEnv extras: ee_pose, ft_sensor, joint_pos, joint_vel

    Parameters
    ----------
    observation_space : gym.spaces.Dict
    features_dim      : int  — output dimensionality
    gnn_hidden        : int
    gnn_layers        : int
    gnn_heads         : int
    gnn_output_dim    : int
    gnn_dropout       : float
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 512,
        gnn_hidden: int = 128,
        gnn_layers: int = 3,
        gnn_heads: int = 4,
        gnn_output_dim: int = 256,
        gnn_dropout: float = 0.1,
    ) -> None:
        super().__init__(observation_space, features_dim)

        # Infer node feature dimension from observation space
        node_feat_dim = observation_space["node_features"].shape[1]

        self._gnn = _GNNEncoder(
            in_channels=node_feat_dim,
            hidden_channels=gnn_hidden,
            num_layers=gnn_layers,
            heads=gnn_heads,
            dropout=gnn_dropout,
            output_dim=gnn_output_dim,
        )

        # Determine auxiliary input size from available observation keys
        aux_dim = 0
        self._aux_keys: List[str] = []
        for key in ["agent_state", "ee_pose", "ft_sensor", "joint_pos", "joint_vel"]:
            if key in observation_space.spaces:
                space = observation_space.spaces[key]
                aux_dim += int(np.prod(space.shape))
                self._aux_keys.append(key)
                logger.debug("GNNExtractor: using aux key '%s' (dim=%d)", key, np.prod(space.shape))

        # Handle goal_idx: embed as one-hot or just flatten
        self._has_goal_idx = "goal_idx" in observation_space.spaces
        if self._has_goal_idx:
            max_nodes = observation_space["node_features"].shape[0]
            self._goal_embed = nn.Embedding(max_nodes, 16)
            aux_dim += 16

        concat_dim = gnn_output_dim + aux_dim
        self._head = nn.Sequential(
            nn.Linear(concat_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(inplace=True),
        )

        logger.info(
            "GNNSceneGraphExtractor: node_dim=%d, gnn_out=%d, aux_dim=%d → features=%d",
            node_feat_dim, gnn_output_dim, aux_dim, features_dim,
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        node_feat = observations["node_features"]    # (B, N, D)
        adj_mat   = observations["adj_matrix"]        # (B, N, N)

        gnn_out = self._gnn(node_feat, adj_mat)       # (B, gnn_output_dim)

        parts = [gnn_out]
        for key in self._aux_keys:
            flat = observations[key].flatten(start_dim=1).float()
            parts.append(flat)

        if self._has_goal_idx:
            goal_idx = observations["goal_idx"].long().squeeze(-1)   # (B,)
            goal_emb = self._goal_embed(goal_idx)                     # (B, 16)
            parts.append(goal_emb)

        concat = torch.cat(parts, dim=1)
        return self._head(concat)
