"""
aria.perception.scene_graph
============================
Semantic 3-D scene graph backed by NetworkX DiGraph.

Each **node** represents a detected object:
    id, class_label, centroid (3,), bbox_3d (6,), embedding (D,), confidence

Each **edge** represents a spatial predicate:
    on | near | inside | left_of | right_of | above | below

Predicate inference
-------------------
1. Geometric rules  — fast, deterministic (centroid overlap, bbox stacking)
2. CLIP verification — zero-shot rescoring of borderline cases (optional)

Querying
--------
    sg.query_node("the red mug")   → [node_id, ...] sorted by match score
    sg.get_node(nid)               → dict with node attributes
    sg.neighbours(nid)             → [(neighbour_id, predicate), ...]
    sg.to_observation_tensors()    → node_matrix (N,D), adj_matrix (N,N)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)

# Optional CLIP import — soft dependency
try:
    import clip
    import torch
    from PIL import Image as PILImage
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False
    logger.warning("CLIP not available — semantic edge verification disabled")


# ---------------------------------------------------------------------------
# Node dataclass
# ---------------------------------------------------------------------------

@dataclass
class SceneNode:
    """A single object node in the scene graph."""
    node_id: int
    class_label: str
    centroid: np.ndarray          # (3,) world coords
    bbox_3d: np.ndarray           # (6,) [x_min, y_min, z_min, x_max, y_max, z_max]
    embedding: np.ndarray         # (D,) visual feature vector
    confidence: float = 1.0
    color_rgb: np.ndarray = field(default_factory=lambda: np.zeros(3))  # mean RGB in [0,1]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        c = self.centroid.round(3).tolist()
        return (
            f"SceneNode(id={self.node_id}, label={self.class_label!r}, "
            f"centroid={c}, conf={self.confidence:.2f})"
        )

    def to_feature_vector(self, max_embed_dim: int = 64) -> np.ndarray:
        """Compact node feature: [centroid(3), color_rgb(3), embed[:max_embed_dim]]."""
        embed = self.embedding
        if len(embed) > max_embed_dim:
            embed = embed[:max_embed_dim]
        elif len(embed) < max_embed_dim:
            embed = np.pad(embed, (0, max_embed_dim - len(embed)))
        return np.concatenate([self.centroid, self.color_rgb, embed]).astype(np.float32)


# ---------------------------------------------------------------------------
# Spatial predicate engine
# ---------------------------------------------------------------------------

_PREDICATES = ("on", "near", "inside", "left_of", "right_of", "above", "below")


def _infer_spatial_predicates(
    a: SceneNode,
    b: SceneNode,
    near_thresh: float = 1.0,
    on_vert_thresh: float = 0.05,
    inside_overlap_thresh: float = 0.5,
) -> list[str]:
    """
    Return a list of predicates describing a's relationship to b.

    - 'on'      : a's bottom face is within on_vert_thresh above b's top face
    - 'near'    : centroid distance < near_thresh
    - 'inside'  : a's xyz centroid is largely within b's bbox
    - 'left_of' : a.centroid.x < b.centroid.x
    - 'right_of': a.centroid.x > b.centroid.x
    - 'above'   : a.centroid.z significantly above b.centroid.z
    - 'below'   : a.centroid.z significantly below b.centroid.z
    """
    predicates = []

    a_min, a_max = a.bbox_3d[:3], a.bbox_3d[3:]
    b_min, b_max = b.bbox_3d[:3], b.bbox_3d[3:]

    dist = float(np.linalg.norm(a.centroid - b.centroid))
    if dist < near_thresh:
        predicates.append("near")

    # 'on': a is resting on top of b
    a_bottom_z = a_min[2]
    b_top_z    = b_max[2]
    if abs(a_bottom_z - b_top_z) < on_vert_thresh and dist < near_thresh:
        predicates.append("on")

    # 'inside': intersection over b-volume
    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter_dims = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = float(np.prod(inter_dims))
    b_vol = float(np.prod(b_max - b_min)) + 1e-8
    if inter_vol / b_vol > inside_overlap_thresh:
        predicates.append("inside")

    # Horizontal relations
    dx = a.centroid[0] - b.centroid[0]
    h_thresh = 0.3
    if dx < -h_thresh:
        predicates.append("left_of")
    elif dx > h_thresh:
        predicates.append("right_of")

    # Vertical relations
    dz = a.centroid[2] - b.centroid[2]
    v_thresh = 0.3
    if dz > v_thresh:
        predicates.append("above")
    elif dz < -v_thresh:
        predicates.append("below")

    return predicates


# ---------------------------------------------------------------------------
# Scene Graph
# ---------------------------------------------------------------------------

class SceneGraph:
    """
    Live semantic scene graph for the ARIA agent.

    Parameters
    ----------
    max_nodes : int
        Maximum number of nodes kept in the graph simultaneously.
    near_threshold_m : float
        Distance threshold for the 'near' predicate.
    on_vertical_threshold_m : float
        Vertical tolerance for the 'on' predicate.
    inside_overlap_threshold : float
        Fractional overlap threshold for the 'inside' predicate.
    clip_model_name : str
        CLIP model variant for semantic verification ("ViT-B/32", etc.).
    use_clip : bool
        Whether to run CLIP-based predicate verification.
    """

    def __init__(
        self,
        max_nodes: int = 64,
        near_threshold_m: float = 1.0,
        on_vertical_threshold_m: float = 0.05,
        inside_overlap_threshold: float = 0.5,
        clip_model_name: str = "ViT-B/32",
        use_clip: bool = True,
    ) -> None:
        self.max_nodes = max_nodes
        self._near_thresh = near_threshold_m
        self._on_vert_thresh = on_vertical_threshold_m
        self._inside_thresh = inside_overlap_threshold
        self._graph: nx.DiGraph = nx.DiGraph()
        self._next_id: int = 0
        self._node_registry: Dict[int, SceneNode] = {}

        # CLIP model
        self._clip_model = self._clip_preprocess = None
        if use_clip and _CLIP_AVAILABLE:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._clip_model, self._clip_preprocess = clip.load(clip_model_name, device=device)
                self._clip_device = device
                logger.info("CLIP loaded: %s on %s", clip_model_name, device)
            except Exception as exc:
                logger.warning("Failed to load CLIP: %s", exc)

        logger.info("SceneGraph: max_nodes=%d", max_nodes)

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def add_or_update_node(self, node: SceneNode) -> int:
        """
        Insert a new node or update an existing one (matched by class_label
        and proximity).  Returns the assigned node_id.
        """
        # Try to match existing node within 0.5m with same class
        for nid, existing in self._node_registry.items():
            if (existing.class_label == node.class_label and
                    np.linalg.norm(existing.centroid - node.centroid) < 0.5):
                # Update in place
                existing.centroid = node.centroid
                existing.bbox_3d  = node.bbox_3d
                existing.embedding = node.embedding
                existing.confidence = node.confidence
                existing.color_rgb = node.color_rgb
                self._graph.nodes[nid].update({"label": node.class_label})
                self._rebuild_edges_for(nid)
                return nid

        # New node
        if len(self._node_registry) >= self.max_nodes:
            self._evict_lowest_confidence()

        nid = self._next_id
        self._next_id += 1
        node.node_id = nid
        self._node_registry[nid] = node
        self._graph.add_node(nid, label=node.class_label)
        self._rebuild_edges_for(nid)
        return nid

    def remove_node(self, node_id: int) -> None:
        """Remove a node and all its edges."""
        if node_id in self._node_registry:
            del self._node_registry[node_id]
            self._graph.remove_node(node_id)

    def get_node(self, node_id: int) -> Optional[SceneNode]:
        return self._node_registry.get(node_id)

    def all_nodes(self) -> List[SceneNode]:
        return list(self._node_registry.values())

    def all_node_ids(self) -> List[int]:
        return list(self._node_registry.keys())

    def num_nodes(self) -> int:
        return len(self._node_registry)

    # ------------------------------------------------------------------
    # Edge / predicate management
    # ------------------------------------------------------------------

    def neighbours(self, node_id: int) -> List[Tuple[int, str]]:
        """Return [(neighbour_id, predicate), ...] for outgoing edges."""
        return [
            (nbr, data["predicate"])
            for _, nbr, data in self._graph.out_edges(node_id, data=True)
        ]

    def get_predicate(self, src: int, dst: int) -> Optional[str]:
        data = self._graph.get_edge_data(src, dst)
        return data["predicate"] if data else None

    def get_edge_weight(self, src: int, dst: int) -> float:
        """
        Return the spatial-distance weight on an edge (lower = closer).
        Returns 1.0 if the edge does not exist.
        """
        data = self._graph.get_edge_data(src, dst)
        return float(data["weight"]) if data else 1.0

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query_node(
        self,
        query: str,
        top_k: int = 5,
        use_embedding: bool = True,
    ) -> List[Tuple[int, float]]:
        """
        Find nodes matching a natural-language description.

        Scoring strategy
        ----------------
        1. CLIP text-image embedding similarity (if available).
        2. Fallback: token overlap between query tokens and class_label.

        Returns
        -------
        List of (node_id, score) sorted by descending score.
        """
        if not self._node_registry:
            return []

        if use_embedding and self._clip_model is not None:
            return self._clip_query(query, top_k)

        return self._lexical_query(query, top_k)

    def _clip_query(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """CLIP-based semantic similarity scoring."""
        import torch
        text_tokens = clip.tokenize([query]).to(self._clip_device)
        with torch.no_grad():
            text_feat = self._clip_model.encode_text(text_tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        scores = []
        for nid, node in self._node_registry.items():
            node_feat = torch.tensor(
                node.embedding, dtype=torch.float32, device=self._clip_device
            ).unsqueeze(0)
            node_feat = node_feat / (node_feat.norm(dim=-1, keepdim=True) + 1e-8)
            sim = float((text_feat * node_feat).sum())
            scores.append((nid, sim))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def _lexical_query(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Token-overlap fallback scoring."""
        q_tokens = set(query.lower().split())
        scores = []
        for nid, node in self._node_registry.items():
            label_tokens = set(node.class_label.lower().split("_"))
            overlap = len(q_tokens & label_tokens) / (len(q_tokens) + 1e-8)
            scores.append((nid, overlap))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    # ------------------------------------------------------------------
    # Tensor representation for RL observation space
    # ------------------------------------------------------------------

    def to_observation_tensors(
        self,
        max_nodes: Optional[int] = None,
        node_feat_dim: int = 70,         # 3+3+64
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Export the graph as dense tensors suitable for GNN input.

        Returns
        -------
        node_features : (max_nodes, node_feat_dim)  — padded with zeros
        adj_matrix    : (max_nodes, max_nodes)       — binary adjacency
        valid_mask    : list[int]                    — indices of real nodes
        """
        N = max_nodes or self.max_nodes
        node_features = np.zeros((N, node_feat_dim), dtype=np.float32)
        adj_matrix    = np.zeros((N, N), dtype=np.float32)

        node_ids = list(self._node_registry.keys())[:N]
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        for nid in node_ids:
            i = id_to_idx[nid]
            node = self._node_registry[nid]
            feat = node.to_feature_vector(max_embed_dim=node_feat_dim - 6)
            node_features[i, :len(feat)] = feat

        for src, dst, _ in self._graph.edges(data=True):
            if src in id_to_idx and dst in id_to_idx:
                adj_matrix[id_to_idx[src], id_to_idx[dst]] = 1.0

        return node_features, adj_matrix, [id_to_idx[nid] for nid in node_ids]

    def summary(self) -> str:
        """Human-readable summary of the current graph state."""
        lines = [f"SceneGraph: {self.num_nodes()} nodes, {self._graph.number_of_edges()} edges"]
        for nid, node in self._node_registry.items():
            lines.append(f"  [{nid}] {node.class_label} @ {node.centroid.round(2)} (conf={node.confidence:.2f})")
            for nbr_id, predicate in self.neighbours(nid):
                nbr_label = self._node_registry[nbr_id].class_label
                lines.append(f"       --{predicate}--> [{nbr_id}] {nbr_label}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Remove all nodes and edges, resetting the graph to an empty state."""
        self._graph.clear()
        self._node_registry.clear()
        self._next_id = 0
        logger.debug("SceneGraph cleared.")

    def subgraph_around(
        self,
        node_id: int,
        depth: int = 1,
    ) -> "SceneGraph":
        """
        Extract the *ego subgraph* centred on ``node_id`` up to ``depth`` hops.

        Returns a new :class:`SceneGraph` containing only the nodes reachable
        within ``depth`` directed steps from ``node_id`` (inclusive), together
        with the edges between them.  The new graph is independent and does not
        share state with the original.

        Parameters
        ----------
        node_id : int   — centre of the ego subgraph
        depth   : int   — maximum number of edge hops to include

        Returns
        -------
        SceneGraph
            A new SceneGraph containing the subgraph.
        """
        if node_id not in self._node_registry:
            raise KeyError(f"node_id {node_id} not found in graph")

        # BFS from node_id up to 'depth' hops
        visited: set[int] = {node_id}
        frontier: set[int] = {node_id}
        for _ in range(depth):
            next_frontier: set[int] = set()
            for nid in frontier:
                for nbr in self._graph.successors(nid):
                    if nbr not in visited:
                        visited.add(nbr)
                        next_frontier.add(nbr)
            frontier = next_frontier

        sub = SceneGraph(
            max_nodes=self.max_nodes,
            near_threshold_m=self._near_thresh,
            on_vertical_threshold_m=self._on_vert_thresh,
            inside_overlap_threshold=self._inside_thresh,
            use_clip=False,   # avoid reloading CLIP for a temporary subgraph
        )
        # Insert nodes in discovery order so IDs are preserved where possible
        for nid in sorted(visited):
            node = self._node_registry[nid]
            sub._node_registry[nid] = node
            sub._graph.add_node(nid, label=node.class_label)
        # Copy edges that fall within the subgraph
        for src, dst, data in self._graph.edges(data=True):
            if src in visited and dst in visited:
                sub._graph.add_edge(src, dst, **data)
        sub._next_id = max(visited) + 1
        logger.debug(
            "subgraph_around(%d, depth=%d): %d nodes, %d edges",
            node_id, depth, len(visited), sub._graph.number_of_edges(),
        )
        return sub

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rebuild_edges_for(self, node_id: int) -> None:
        """Recompute spatial edges between node_id and every other node."""
        # Remove stale edges involving this node
        edges_to_remove = list(self._graph.in_edges(node_id)) + list(self._graph.out_edges(node_id))
        self._graph.remove_edges_from(edges_to_remove)

        node_a = self._node_registry[node_id]
        for other_id, node_b in self._node_registry.items():
            if other_id == node_id:
                continue
            dist = float(np.linalg.norm(node_a.centroid - node_b.centroid))
            # a → b
            for pred in _infer_spatial_predicates(
                node_a, node_b, self._near_thresh, self._on_vert_thresh, self._inside_thresh
            ):
                self._graph.add_edge(node_id, other_id, predicate=pred, weight=dist)
            # b → a
            for pred in _infer_spatial_predicates(
                node_b, node_a, self._near_thresh, self._on_vert_thresh, self._inside_thresh
            ):
                self._graph.add_edge(other_id, node_id, predicate=pred, weight=dist)

    def _evict_lowest_confidence(self) -> None:
        """Remove the node with the lowest confidence score to make room."""
        if not self._node_registry:
            return
        worst_id = min(self._node_registry, key=lambda k: self._node_registry[k].confidence)
        logger.debug("Evicting node %d (%s)", worst_id, self._node_registry[worst_id].class_label)
        self.remove_node(worst_id)


def scene_graph_from_config(cfg: dict) -> SceneGraph:
    sp = cfg.get("spatial_predicates", {})
    return SceneGraph(
        max_nodes=cfg.get("max_nodes", 64),
        near_threshold_m=sp.get("near_distance_threshold_m", 1.0),
        on_vertical_threshold_m=sp.get("on_vertical_threshold_m", 0.05),
        inside_overlap_threshold=sp.get("inside_overlap_threshold", 0.5),
        clip_model_name=cfg.get("clip_model", "ViT-B/32"),
        use_clip=True,
    )
