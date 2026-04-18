"""
Integration test: end-to-end perception pipeline.

Verifies that SensorFusion → OccupancyMap → SceneGraph produces a
non-empty, structurally valid graph from a synthetic point cloud.
"""

from __future__ import annotations

import numpy as np
import pytest

from aria.perception.sensor_fusion import SensorFusion, RGBDFrame, CameraIntrinsics
from aria.perception.occupancy_map import OccupancyMap
from aria.perception.scene_graph import SceneGraph, SceneNode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fusion() -> SensorFusion:
    intr = CameraIntrinsics(fx=615, fy=615, cx=320, cy=240, width=640, height=480,
                             depth_scale=0.001, max_depth_m=5.0)
    return SensorFusion(intrinsics=intr, voxel_size_m=0.05)


@pytest.fixture
def rgbd_frame() -> RGBDFrame:
    """Synthetic RGB-D frame: flat table at 1 metre depth."""
    rgb   = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
    depth = np.full((480, 640), 1.0, dtype=np.float32)   # 1 m uniform depth
    return RGBDFrame(rgb=rgb, depth=depth, timestamp=0.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPerceptionPipeline:

    def test_fusion_produces_points(self, fusion, rgbd_frame) -> None:
        frame = fusion.process(rgbd_frame)
        assert frame.num_points > 100
        assert frame.points_world.shape[1] == 3
        assert frame.colors.shape[1] == 3

    def test_occupancy_map_update(self, fusion, rgbd_frame) -> None:
        frame = fusion.process(rgbd_frame)
        omap  = OccupancyMap(resolution_m=0.05, extent_m=(6.0, 6.0, 3.0))
        omap.update(frame.points_world)
        # After one update, some voxels should be above prior
        grid = omap.probability_grid()
        assert float(grid.max()) > 0.5

    def test_scene_graph_accepts_nodes(self) -> None:
        sg = SceneGraph(max_nodes=16, use_clip=False)
        nodes_to_add = [
            SceneNode(
                node_id=-1,
                class_label=label,
                centroid=np.array(pos, dtype=np.float32),
                bbox_3d=np.array([pos[0]-0.1, pos[1]-0.1, 0.0,
                                   pos[0]+0.1, pos[1]+0.1, 0.2], dtype=np.float32),
                embedding=np.zeros(64, dtype=np.float32),
            )
            for label, pos in [
                ("mug",   [0.5, 0.0, 0.85]),
                ("shelf", [0.0, 1.0, 1.20]),
                ("table", [0.5, 0.0, 0.40]),
            ]
        ]
        for node in nodes_to_add:
            sg.add_or_update_node(node)
        assert sg.num_nodes() == 3
        assert sg._graph.number_of_edges() > 0

    def test_full_pipeline_observation_tensors(self, fusion, rgbd_frame) -> None:
        """Full pipeline: fusion → omap → graph → GNN tensors."""
        frame = fusion.process(rgbd_frame)
        omap = OccupancyMap(resolution_m=0.05, extent_m=(6.0, 6.0, 3.0))
        omap.update(frame.points_world)

        sg = SceneGraph(max_nodes=8, use_clip=False)
        # Simulate segmentation: use a few point-cloud centroids as nodes
        pts = frame.points_world
        step = max(1, len(pts) // 3)
        for i in range(0, min(3 * step, len(pts)), step):
            c = pts[i:i + step].mean(0)
            node = SceneNode(
                node_id=-1,
                class_label=f"object_{i // step}",
                centroid=c,
                bbox_3d=np.concatenate([c - 0.1, c + 0.1]),
                embedding=np.zeros(64, dtype=np.float32),
            )
            sg.add_or_update_node(node)

        nf, adj, valid = sg.to_observation_tensors(max_nodes=8, node_feat_dim=70)
        assert nf.shape  == (8, 70)
        assert adj.shape == (8, 8)
        assert len(valid) <= 3
        assert not np.isnan(nf).any()
