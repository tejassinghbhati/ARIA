"""aria.perception — 3D vision pipeline for ARIA.

Exports:
    SensorFusion        — fuse RGB-D, LiDAR, and IMU streams into point clouds
    PointNetBackbone    — PointNet++ multi-scale grouping feature encoder
    OccupancyMap        — 3D voxel grid with sliding-window spatial memory
    SceneGraph          — semantic scene graph with spatial-predicate edges
    NLPGrounder         — LLM-based task planner + CLIP node grounding
"""

from aria.perception.sensor_fusion import SensorFusion
from aria.perception.pointnet_backbone import PointNetBackbone
from aria.perception.occupancy_map import OccupancyMap
from aria.perception.scene_graph import SceneGraph
from aria.perception.nlp_grounding import NLPGrounder

__all__ = [
    "SensorFusion",
    "PointNetBackbone",
    "OccupancyMap",
    "SceneGraph",
    "NLPGrounder",
]
