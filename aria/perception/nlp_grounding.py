"""
aria.perception.nlp_grounding
==============================
Language → action sub-goal pipeline.

Architecture
------------
1.  **LLM Task Planner** (Phi-3-mini via HuggingFace Transformers)
    Decomposes a natural-language command into a structured sequence of
    primitive sub-goals:  navigate_to | pick_up | place_on | inspect

2.  **CLIP / SentenceTransformer Grounding**
    Maps each sub-goal's object description to a specific node ID in the
    live scene graph by embedding similarity scoring.

3.  **Rule-based fallback**
    If the LLM is unavailable (offline / OOM), a lightweight regex parser
    handles common command patterns.

Usage
-----
    grounder = NLPGrounder(cfg)
    plan = grounder.plan("fetch the red mug from the shelf")
    # plan → [
    #   SubGoal(action='navigate_to', object_desc='shelf', node_id=3),
    #   SubGoal(action='pick_up',     object_desc='red mug', node_id=7),
    #   SubGoal(action='navigate_to', object_desc='table',   node_id=2),
    #   SubGoal(action='place_on',    object_desc='table',   node_id=2),
    # ]
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sub-goal dataclass
# ---------------------------------------------------------------------------

@dataclass
class SubGoal:
    """A single decomposed action primitive."""
    action: str           # "navigate_to" | "pick_up" | "place_on" | "inspect"
    object_desc: str      # natural language description of the target object
    node_id: Optional[int] = None    # resolved scene graph node ID (after grounding)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_grounded(self) -> bool:
        return self.node_id is not None


# ---------------------------------------------------------------------------
# Phi-3-mini LLM Planner
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a task planner for a robot arm in an indoor environment.
Given a natural language instruction and a list of objects visible in the
scene, decompose the instruction into a JSON array of sub-goal steps.

Each step must be a JSON object with exactly two fields:
  "action":      one of ["navigate_to", "pick_up", "place_on", "inspect"]
  "object_desc": a short noun phrase identifying the target object

Return ONLY a valid JSON array, no other text.

Example:
Instruction: "Put the blue cup on the table"
Scene objects: ["blue cup", "table", "shelf", "chair"]
Response:
[
  {"action": "navigate_to", "object_desc": "blue cup"},
  {"action": "pick_up",     "object_desc": "blue cup"},
  {"action": "navigate_to", "object_desc": "table"},
  {"action": "place_on",    "object_desc": "table"}
]
"""


class _LLMPlanner:
    """Wrapper around a Phi-3-mini (or any instruct-tuned) HuggingFace model."""

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        device: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._pipeline = None
        self._loaded = False

        try:
            from transformers import pipeline, AutoTokenizer
            import torch

            if device == "auto":
                device_id = 0 if torch.cuda.is_available() else -1
            elif device == "cuda":
                device_id = 0
            else:
                device_id = -1

            logger.info("Loading LLM planner: %s (device=%s)…", model_name, device)
            self._pipeline = pipeline(
                "text-generation",
                model=model_name,
                device=device_id,
                torch_dtype="auto",
                trust_remote_code=True,
            )
            self._loaded = True
            logger.info("LLM planner loaded successfully.")
        except Exception as exc:
            logger.warning("LLM planner could not be loaded (%s) — fallback enabled.", exc)

    def plan(self, instruction: str, scene_objects: List[str]) -> Optional[List[SubGoal]]:
        """
        Call the LLM to produce a structured task plan.

        Returns None if the LLM is unavailable or output cannot be parsed.
        """
        if not self._loaded or self._pipeline is None:
            return None

        objects_str = json.dumps(scene_objects)
        user_msg = (
            f"Instruction: \"{instruction}\"\n"
            f"Scene objects: {objects_str}\n"
            "Response:"
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]

        try:
            outputs = self._pipeline(
                messages,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                return_full_text=False,
            )
            generated = outputs[0]["generated_text"]
            # Extract JSON array from the response
            return self._parse_json_plan(generated)
        except Exception as exc:
            logger.warning("LLM inference failed: %s", exc)
            return None

    @staticmethod
    def _parse_json_plan(text: str) -> Optional[List[SubGoal]]:
        """Extract a JSON array from raw LLM output."""
        # Find the first '[' to ']' block
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if not match:
            return None
        try:
            raw = json.loads(match.group(0))
            goals = []
            for item in raw:
                action = item.get("action", "").strip().lower()
                obj_desc = item.get("object_desc", "").strip()
                if action and obj_desc:
                    goals.append(SubGoal(action=action, object_desc=obj_desc))
            return goals if goals else None
        except (json.JSONDecodeError, KeyError, TypeError):
            return None


# ---------------------------------------------------------------------------
# Rule-based fallback parser
# ---------------------------------------------------------------------------

_FETCH_RE    = re.compile(r"(?:fetch|get|bring|grab|retrieve)\s+(.+?)(?:\s+from\s+(.+))?$", re.I)
_PLACE_RE    = re.compile(r"(?:put|place|set|drop)\s+(.+?)\s+(?:on|onto|in|into)\s+(.+)$", re.I)
_INSPECT_RE  = re.compile(r"(?:look\s*at|inspect|check|examine)\s+(.+)$", re.I)
_NAVIGATE_RE = re.compile(r"(?:go\s+to|move\s+to|navigate\s+to|approach)\s+(.+)$", re.I)


def _rule_based_plan(instruction: str) -> List[SubGoal]:
    """Lightweight regex-based command parser for common patterns."""
    instr = instruction.strip()

    if m := _FETCH_RE.match(instr):
        obj = m.group(1).strip()
        src = m.group(2).strip() if m.group(2) else None
        goals = []
        if src:
            goals.append(SubGoal(action="navigate_to", object_desc=src))
        goals.append(SubGoal(action="navigate_to", object_desc=obj))
        goals.append(SubGoal(action="pick_up",     object_desc=obj))
        return goals

    if m := _PLACE_RE.match(instr):
        obj = m.group(1).strip()
        target = m.group(2).strip()
        return [
            SubGoal(action="navigate_to", object_desc=obj),
            SubGoal(action="pick_up",     object_desc=obj),
            SubGoal(action="navigate_to", object_desc=target),
            SubGoal(action="place_on",    object_desc=target),
        ]

    if m := _INSPECT_RE.match(instr):
        obj = m.group(1).strip()
        return [
            SubGoal(action="navigate_to", object_desc=obj),
            SubGoal(action="inspect",     object_desc=obj),
        ]

    if m := _NAVIGATE_RE.match(instr):
        return [SubGoal(action="navigate_to", object_desc=m.group(1).strip())]

    # Generic fallback: treat entire instruction as a navigate goal
    return [SubGoal(action="navigate_to", object_desc=instr)]


# ---------------------------------------------------------------------------
# SentenceTransformer / CLIP grounding
# ---------------------------------------------------------------------------

class _EmbeddingGrounder:
    """Grounds object descriptions to scene-graph node IDs using embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            logger.info("SentenceTransformer loaded: %s", model_name)
        except Exception as exc:
            logger.warning("SentenceTransformer unavailable (%s) — lexical grounding only", exc)

    def ground(
        self,
        object_desc: str,
        scene_nodes: List[Any],   # List[SceneNode]
        top_k: int = 1,
    ) -> List[tuple[int, float]]:
        """
        Returns [(node_id, score), ...] sorted by descending score.

        Falls back to token-overlap if embeddings unavailable.
        """
        if not scene_nodes:
            return []

        labels = [n.class_label for n in scene_nodes]

        if self._model is not None:
            import numpy as np
            query_emb = self._model.encode([object_desc], normalize_embeddings=True)
            label_embs = self._model.encode(labels, normalize_embeddings=True)
            sims = (label_embs @ query_emb.T).flatten()
            ranked = sorted(zip([n.node_id for n in scene_nodes], sims.tolist()),
                            key=lambda x: -x[1])
            return ranked[:top_k]

        # Lexical fallback
        q_tokens = set(object_desc.lower().split())
        scored = []
        for node in scene_nodes:
            label_tokens = set(node.class_label.lower().replace("_", " ").split())
            overlap = len(q_tokens & label_tokens) / (len(q_tokens) + 1e-8)
            scored.append((node.node_id, overlap))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


# ---------------------------------------------------------------------------
# NLPGrounder (public interface)
# ---------------------------------------------------------------------------

class NLPGrounder:
    """
    Full natural-language → grounded task-plan pipeline.

    Parameters
    ----------
    cfg : dict
        Parsed `nlp_grounding` section from perception.yaml.
    """

    def __init__(self, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        use_llm = cfg.get("use_llm", True)

        self._llm = None
        if use_llm:
            self._llm = _LLMPlanner(
                model_name=cfg.get("llm_model", "microsoft/Phi-3-mini-4k-instruct"),
                device=cfg.get("llm_device", "auto"),
                max_new_tokens=cfg.get("llm_max_new_tokens", 256),
                temperature=cfg.get("llm_temperature", 0.1),
            )

        self._grounder = _EmbeddingGrounder(
            model_name=cfg.get("sentence_transformer", "all-MiniLM-L6-v2")
        )
        self._top_k = cfg.get("clip_grounding_topk", 5)

    def plan(
        self,
        instruction: str,
        scene_graph=None,   # Optional[SceneGraph]
    ) -> List[SubGoal]:
        """
        Produce a fully grounded task plan from a natural language instruction.

        Parameters
        ----------
        instruction  : str            — e.g. "fetch the red mug from the shelf"
        scene_graph  : SceneGraph     — live scene graph to ground objects against

        Returns
        -------
        List[SubGoal] with node_id populated where possible.
        """
        # 1. Get scene objects for LLM context
        scene_nodes = scene_graph.all_nodes() if scene_graph is not None else []
        scene_objects = [n.class_label for n in scene_nodes]

        # 2. LLM planning → fallback to rules
        sub_goals: Optional[List[SubGoal]] = None
        if self._llm is not None:
            sub_goals = self._llm.plan(instruction, scene_objects)
        if sub_goals is None:
            logger.debug("Using rule-based fallback planner.")
            sub_goals = _rule_based_plan(instruction)

        # 3. Ground each sub-goal object to a scene graph node
        if scene_nodes:
            for sg in sub_goals:
                matches = self._grounder.ground(sg.object_desc, scene_nodes, top_k=1)
                if matches:
                    sg.node_id = matches[0][0]
                    sg.confidence = float(matches[0][1])

        self._log_plan(instruction, sub_goals)
        return sub_goals

    @staticmethod
    def _log_plan(instruction: str, plan: List[SubGoal]) -> None:
        logger.info("Plan for: '%s'", instruction)
        for i, sg in enumerate(plan):
            logger.info(
                "  [%d] %s → '%s' (node_id=%s, conf=%.2f)",
                i, sg.action, sg.object_desc, sg.node_id, sg.confidence,
            )

    def ground_description(
        self,
        description: str,
        scene_graph=None,
        top_k: int | None = None,
    ) -> List[tuple[int, float]]:
        """
        Standalone: ground a single object description to node IDs.

        Returns
        -------
        List[(node_id, score)]
        """
        scene_nodes = scene_graph.all_nodes() if scene_graph is not None else []
        k = top_k or self._top_k
        return self._grounder.ground(description, scene_nodes, top_k=k)
