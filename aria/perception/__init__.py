"""aria.perception — 3D vision pipeline for ARIA.

Exports:
    SensorFusion        — fuse RGB-D, LiDAR, and IMU streams into point clouds
    PointNetBackbone    — PointNet++ multi-scale grouping feature encoder
    OccupancyMap        — 3D voxel grid with sliding-window spatial memory
    SceneGraph          — semantic scene graph with spatial-predicate edges
    NLPGrounder         — LLM-based task planner + CLIP node grounding
"""

from __future__ import annotations


def __getattr__(name: str):
    if name == "SensorFusion":
        from aria.perception.sensor_fusion import SensorFusion
        return SensorFusion
    if name == "PointNetBackbone":
        from aria.perception.pointnet_backbone import PointNetBackbone
        return PointNetBackbone
    if name == "OccupancyMap":
        from aria.perception.occupancy_map import OccupancyMap
        return OccupancyMap
    if name == "SceneGraph":
        from aria.perception.scene_graph import SceneGraph
        return SceneGraph
    if name == "NLPGrounder":
        from aria.perception.nlp_grounding import NLPGrounder
        return NLPGrounder
    raise AttributeError(f"module 'aria.perception' has no attribute {name!r}")


__all__ = [
    "SensorFusion",
    "PointNetBackbone",
    "OccupancyMap",
    "SceneGraph",
    "NLPGrounder",
]
