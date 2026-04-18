"""
aria.perception.pointnet_backbone
===================================
PointNet++ Multi-Scale Grouping (MSG) backbone encoder implemented in PyTorch.

Architecture
------------
    Input: (B, N, C_in)   — batch of point clouds, C_in = xyz + features
    ├── SA Module 1  (fine scale)   → (B, N1, D1)
    ├── SA Module 2  (mid scale)    → (B, N2, D2)
    ├── SA Module 3  (coarse scale) → (B, N3, D3)
    └── Global pooling             → (B, global_feat_dim)

Reference: Qi et al., "PointNet++: Deep Hierarchical Feature Learning on
           Point Sets in a Metric Space", NeurIPS 2017.

Usage
-----
    backbone = PointNetBackbone(in_channels=6, global_feat_dim=512)
    cloud = torch.randn(2, 2048, 6)           # (B, N, C)
    per_point, global_feat = backbone(cloud)   # (B, N, D), (B, 512)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level geometric operations
# ---------------------------------------------------------------------------

def _square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Compute squared L2 distance between all pairs of points.

    Parameters
    ----------
    src : (B, N, 3)
    dst : (B, M, 3)

    Returns
    -------
    dist : (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = (
        -2 * torch.bmm(src, dst.permute(0, 2, 1))
        + torch.sum(src ** 2, dim=-1, keepdim=True)
        + torch.sum(dst ** 2, dim=-1, keepdim=True).permute(0, 2, 1)
    )
    return dist.clamp(min=0.0)


def _farthest_point_sample(xyz: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Iterative farthest-point sampling (FPS).

    Parameters
    ----------
    xyz : (B, N, 3)
    n_samples : int

    Returns
    -------
    idx : (B, n_samples)  — indices into the N dimension
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    distances = torch.full((B, N), float("inf"), device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    for i in range(n_samples):
        centroids[:, i] = farthest
        centroid_xyz = xyz[torch.arange(B), farthest, :].unsqueeze(1)  # (B,1,3)
        dist = torch.sum((xyz - centroid_xyz) ** 2, dim=-1)            # (B,N)
        distances = torch.min(distances, dist)
        farthest = distances.max(dim=1)[1]

    return centroids


def _ball_query(
    radius: float,
    n_samples: int,
    xyz: torch.Tensor,
    query_xyz: torch.Tensor,
) -> torch.Tensor:
    """
    Ball-query grouping: for each point in query_xyz, find the nearest
    n_samples neighbours within radius in xyz.

    Returns
    -------
    idx : (B, S, n_samples)
    """
    device = xyz.device
    B, N, _ = xyz.shape
    S = query_xyz.shape[1]
    idx = torch.arange(N, device=device).unsqueeze(0).unsqueeze(0).expand(B, S, N)
    sq_dists = _square_distance(query_xyz, xyz)               # (B, S, N)
    idx[sq_dists > radius ** 2] = N                           # sentinel for out-of-range

    # Sort and take first n_samples; repeat the first valid index if insufficient
    idx = idx.sort(dim=-1)[0][:, :, :n_samples]              # (B, S, K)
    # Replace sentinel with the first valid index
    group_first = idx[:, :, 0:1].expand_as(idx)
    mask = idx == N
    idx[mask] = group_first[mask]
    return idx


# ---------------------------------------------------------------------------
# Shared MLP helper
# ---------------------------------------------------------------------------

class _SharedMLP(nn.Sequential):
    """1-D convolution implemented as shared MLP across all points."""

    def __init__(self, channels: list[int], bn: bool = True) -> None:
        layers: list[nn.Module] = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv1d(channels[i], channels[i + 1], 1, bias=not bn))
            if bn:
                layers.append(nn.BatchNorm1d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


# ---------------------------------------------------------------------------
# Set-Abstraction Module (MSG variant)
# ---------------------------------------------------------------------------

class SetAbstractionMSG(nn.Module):
    """
    Set Abstraction with Multi-Scale Grouping.

    Parameters
    ----------
    n_points   : int          — number of centroids to sample (FPS)
    radii      : list[float]  — ball-query radii for each scale
    n_samples  : list[int]    — neighbour counts for each scale
    mlps       : list[list[int]] — MLP widths for each scale
    in_channels: int          — input feature channels (not counting xyz)
    """

    def __init__(
        self,
        n_points: int,
        radii: list[float],
        n_samples: list[int],
        mlps: list[list[int]],
        in_channels: int,
    ) -> None:
        super().__init__()
        assert len(radii) == len(n_samples) == len(mlps)
        self.n_points = n_points
        self.radii = radii
        self.n_samples = n_samples
        self.convs = nn.ModuleList()
        for i, mlp in enumerate(mlps):
            full_mlp = [in_channels + 3] + mlp
            self.convs.append(_SharedMLP(full_mlp))

    @property
    def out_channels(self) -> int:
        return sum(conv[-2].out_channels for conv in self.convs)  # type: ignore[index]

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        xyz      : (B, N, 3)
        features : (B, N, C)

        Returns
        -------
        new_xyz  : (B, n_points, 3)
        new_feat : (B, n_points, sum_of_mlp_outs)
        """
        B, N, _ = xyz.shape
        centroid_idx = _farthest_point_sample(xyz, self.n_points)          # (B, S)
        new_xyz = xyz[torch.arange(B).unsqueeze(1), centroid_idx, :]       # (B, S, 3)

        scale_feats = []
        for radius, k, conv in zip(self.radii, self.n_samples, self.convs):
            group_idx = _ball_query(radius, k, xyz, new_xyz)               # (B, S, K)
            grouped_xyz = xyz[
                torch.arange(B).unsqueeze(1).unsqueeze(2),
                group_idx, :
            ]                                                               # (B, S, K, 3)
            grouped_xyz -= new_xyz.unsqueeze(2)                            # relative coords
            grouped_feat = features[
                torch.arange(B).unsqueeze(1).unsqueeze(2),
                group_idx, :
            ]                                                               # (B, S, K, C)
            grouped = torch.cat([grouped_xyz, grouped_feat], dim=-1)       # (B, S, K, 3+C)
            grouped = grouped.permute(0, 3, 2, 1)                          # (B, 3+C, K, S)
            B_, C_, K_, S_ = grouped.shape

            # Flatten (K, S) → apply 1-D conv treating K as the sequence dim
            grouped = grouped.reshape(B_, C_, K_ * S_)
            out = conv(grouped)                                             # (B, D, K*S)
            out = out.reshape(B_, -1, K_, S_)
            out = out.max(dim=2)[0].permute(0, 2, 1)                       # (B, S, D)
            scale_feats.append(out)

        new_feat = torch.cat(scale_feats, dim=-1)                          # (B, S, sum_D)
        return new_xyz, new_feat


# ---------------------------------------------------------------------------
# PointNet++ Backbone
# ---------------------------------------------------------------------------

class PointNetBackbone(nn.Module):
    """
    PointNet++ MSG backbone.

    Parameters
    ----------
    in_channels     : int  — number of input feature channels (excluding xyz)
    global_feat_dim : int  — dimension of the final global embedding
    sa_radii        : list[list[float]]  — ball-query radii per SA module
    sa_nsamples     : list[list[int]]    — neighbourhood sizes per SA module
    sa_mlps         : list[list[int]]    — MLP widths per SA module per scale
    """

    def __init__(
        self,
        in_channels: int = 3,    # rgb only; set 0 for xyz-only
        global_feat_dim: int = 512,
        sa_radii: list[list[float]] | None = None,
        sa_nsamples: list[list[int]] | None = None,
        sa_mlps: list[list[int]] | None = None,
    ) -> None:
        super().__init__()

        # Default 3-level MSG architecture
        if sa_radii is None:
            sa_radii = [[0.1, 0.2], [0.2, 0.4], [0.4, 0.8]]
        if sa_nsamples is None:
            sa_nsamples = [[16, 32], [16, 32], [16, 32]]
        if sa_mlps is None:
            sa_mlps = [[32, 32, 64], [64, 64, 128], [128, 128, 256]]

        self.sa1 = SetAbstractionMSG(
            n_points=512,
            radii=sa_radii[0],
            n_samples=sa_nsamples[0],
            mlps=[sa_mlps[0]] * len(sa_radii[0]),
            in_channels=in_channels,
        )
        self.sa2 = SetAbstractionMSG(
            n_points=128,
            radii=sa_radii[1],
            n_samples=sa_nsamples[1],
            mlps=[sa_mlps[1]] * len(sa_radii[1]),
            in_channels=self.sa1.out_channels,
        )
        self.sa3 = SetAbstractionMSG(
            n_points=32,
            radii=sa_radii[2],
            n_samples=sa_nsamples[2],
            mlps=[sa_mlps[2]] * len(sa_radii[2]),
            in_channels=self.sa2.out_channels,
        )

        # Global aggregation head
        self.global_head = nn.Sequential(
            nn.Linear(self.sa3.out_channels, global_feat_dim),
            nn.BatchNorm1d(global_feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(global_feat_dim, global_feat_dim),
        )
        self.global_feat_dim = global_feat_dim
        logger.info(
            "PointNetBackbone: in_ch=%d, global_dim=%d",
            in_channels,
            global_feat_dim,
        )

    def forward(self, cloud: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        cloud : (B, N, 3 + in_channels) — xyz + features

        Returns
        -------
        per_point_feat : (B, N3, D3)   — per-point features at coarsest level
        global_feat    : (B, global_feat_dim)
        """
        xyz = cloud[:, :, :3].contiguous()
        feat = cloud[:, :, 3:].contiguous()

        xyz1, feat1 = self.sa1(xyz, feat)
        xyz2, feat2 = self.sa2(xyz1, feat1)
        xyz3, feat3 = self.sa3(xyz2, feat2)

        # Global feature: mean-pool over coarsest centroids
        g = feat3.mean(dim=1)                              # (B, D3)
        global_feat = self.global_head(g)                  # (B, global_feat_dim)

        return feat3, global_feat

    def load_pretrained(self, path: str | Path) -> None:
        """Load saved weights, ignoring mismatched head dimensions."""
        state = torch.load(str(path), map_location="cpu")
        missing, unexpected = self.load_state_dict(state, strict=False)
        logger.info("Loaded pretrained PointNet++: missing=%d, unexpected=%d",
                    len(missing), len(unexpected))


def pointnet_from_config(cfg: dict) -> PointNetBackbone:
    """Construct a PointNetBackbone from a config dict."""
    backbone = PointNetBackbone(
        in_channels=cfg.get("in_channels", 3),
        global_feat_dim=cfg.get("global_feat_dim", 512),
        sa_radii=cfg.get("sa_radii"),
        sa_nsamples=cfg.get("sa_nsamples"),
        sa_mlps=cfg.get("sa_mlps"),
    )
    if cfg.get("pretrained_weights"):
        backbone.load_pretrained(cfg["pretrained_weights"])
    return backbone
