"""Unit tests for SceneGraph."""

import numpy as np
import pytest

from aria.perception.scene_graph import SceneGraph, SceneNode, _infer_spatial_predicates


def _make_node(nid: int, label: str, centroid: list, z_min=0.0, z_max=0.2) -> SceneNode:
    cx, cy, cz = centroid
    bbox = np.array([cx - 0.1, cy - 0.1, z_min, cx + 0.1, cy + 0.1, z_max], dtype=np.float32)
    return SceneNode(
        node_id=nid,
        class_label=label,
        centroid=np.array(centroid, dtype=np.float32),
        bbox_3d=bbox,
        embedding=np.zeros(64, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Predicate inference
# ---------------------------------------------------------------------------

class TestSpatialPredicates:
    def test_near(self) -> None:
        a = _make_node(0, "mug", [0.0, 0.0, 0.1])
        b = _make_node(1, "shelf", [0.5, 0.0, 0.1])
        preds = _infer_spatial_predicates(a, b, near_thresh=1.0)
        assert "near" in preds

    def test_not_near(self) -> None:
        a = _make_node(0, "mug",   [0.0, 0.0, 0.1])
        b = _make_node(1, "shelf", [5.0, 0.0, 0.1])
        preds = _infer_spatial_predicates(a, b, near_thresh=1.0)
        assert "near" not in preds

    def test_on_predicate(self) -> None:
        # mug sits on top of table
        mug   = _make_node(0, "mug",   [0.0, 0.0, 0.85], z_min=0.80, z_max=0.90)
        table = _make_node(1, "table", [0.0, 0.0, 0.40], z_min=0.00, z_max=0.80)
        preds = _infer_spatial_predicates(mug, table, near_thresh=2.0, on_vert_thresh=0.05)
        assert "on" in preds

    def test_left_of(self) -> None:
        a = _make_node(0, "cup", [-0.5, 0.0, 0.1])
        b = _make_node(1, "mug", [ 0.5, 0.0, 0.1])
        preds = _infer_spatial_predicates(a, b)
        assert "left_of" in preds

    def test_above(self) -> None:
        a = _make_node(0, "shelf", [0.0, 0.0, 1.5])
        b = _make_node(1, "table", [0.0, 0.0, 0.5])
        preds = _infer_spatial_predicates(a, b, near_thresh=2.0)
        assert "above" in preds


# ---------------------------------------------------------------------------
# SceneGraph operations
# ---------------------------------------------------------------------------

class TestSceneGraph:

    def test_add_node_and_query(self) -> None:
        sg = SceneGraph(max_nodes=16, use_clip=False)
        node = _make_node(0, "red_mug", [0.5, 0.0, 0.85])
        nid = sg.add_or_update_node(node)
        assert sg.num_nodes() == 1
        results = sg.query_node("red mug", use_embedding=False)
        assert any(r[0] == nid for r in results)

    def test_update_existing_node(self) -> None:
        sg = SceneGraph(max_nodes=16, use_clip=False)
        node1 = _make_node(0, "mug", [0.5, 0.0, 0.85])
        nid1 = sg.add_or_update_node(node1)
        # Same class within 0.5m → should update, not add
        node2 = _make_node(0, "mug", [0.52, 0.0, 0.85])
        nid2 = sg.add_or_update_node(node2)
        assert nid1 == nid2
        assert sg.num_nodes() == 1

    def test_edges_created(self) -> None:
        sg = SceneGraph(max_nodes=16, use_clip=False)
        mug   = _make_node(0, "mug",   [0.0, 0.0, 0.90], z_min=0.85, z_max=0.95)
        table = _make_node(1, "table", [0.0, 0.0, 0.40], z_min=0.00, z_max=0.85)
        sg.add_or_update_node(mug)
        sg.add_or_update_node(table)
        assert sg._graph.number_of_edges() > 0

    def test_remove_node(self) -> None:
        sg = SceneGraph(max_nodes=16, use_clip=False)
        nid = sg.add_or_update_node(_make_node(0, "bottle", [1.0, 0.0, 0.5]))
        assert sg.num_nodes() == 1
        sg.remove_node(nid)
        assert sg.num_nodes() == 0

    def test_eviction_at_max_nodes(self) -> None:
        sg = SceneGraph(max_nodes=4, use_clip=False)
        for i in range(5):
            n = _make_node(i, f"obj_{i}", [float(i) * 2, 0.0, 0.5])
            n.confidence = float(i) / 4    # increasing confidence
            sg.add_or_update_node(n)
        assert sg.num_nodes() == 4   # evicted the lowest-confidence node

    def test_observation_tensors_shape(self) -> None:
        sg = SceneGraph(max_nodes=8, use_clip=False)
        for i in range(3):
            sg.add_or_update_node(_make_node(i, f"obj_{i}", [float(i), 0.0, 0.5]))
        nf, adj, valid = sg.to_observation_tensors(max_nodes=8, node_feat_dim=70)
        assert nf.shape  == (8, 70)
        assert adj.shape == (8, 8)
        assert len(valid) == 3

    def test_lexical_query_order(self) -> None:
        sg = SceneGraph(max_nodes=16, use_clip=False)
        mug_id   = sg.add_or_update_node(_make_node(0, "red mug", [0.0, 0.0, 0.5]))
        bottle_id = sg.add_or_update_node(_make_node(1, "bottle", [1.0, 0.0, 0.5]))
        results = sg.query_node("mug", top_k=2, use_embedding=False)
        # "red mug" should score higher than "bottle"
        assert results[0][0] == mug_id
