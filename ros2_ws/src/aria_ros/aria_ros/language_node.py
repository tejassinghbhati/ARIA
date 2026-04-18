"""
aria_ros.language_node
=======================
ROS2 node: natural language command → task plan publisher.

Subscribes  /aria/nl_command          std_msgs/String
Publishes   /aria/task_plan           std_msgs/String (JSON SubGoal list)
            /aria/current_goal_node   std_msgs/Int32  (node_id of active goal)

Also subscribes to /aria/scene_graph to keep NLP grounding up-to-date.
"""

from __future__ import annotations

import json
import logging

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String, Int32
    _ROS2_AVAILABLE = True
except ImportError:
    _ROS2_AVAILABLE = False
    class Node:  # type: ignore
        def __init__(self, name): pass

from aria.perception.scene_graph import SceneGraph, SceneNode
from aria.perception.nlp_grounding import NLPGrounder

logger = logging.getLogger(__name__)


class LanguageNode(Node if _ROS2_AVAILABLE else object):
    """Processes NL commands and publishes structured task plans."""

    def __init__(self) -> None:
        if _ROS2_AVAILABLE:
            super().__init__("aria_language_node")
        self._grounder = NLPGrounder()
        self._scene_graph = SceneGraph()

        if _ROS2_AVAILABLE:
            self._cmd_sub  = self.create_subscription(
                String, "/aria/nl_command", self._command_callback, 10
            )
            self._sg_sub   = self.create_subscription(
                String, "/aria/scene_graph", self._scene_graph_callback, 10
            )
            self._plan_pub = self.create_publisher(String, "/aria/task_plan", 10)
            self._goal_pub = self.create_publisher(Int32, "/aria/current_goal_node", 10)
            self.get_logger().info("LanguageNode ready")

    def _scene_graph_callback(self, msg) -> None:
        """Deserialise incoming scene graph JSON and update local graph."""
        try:
            data = json.loads(msg.data)
            self._scene_graph = SceneGraph()
            for n in data.get("nodes", []):
                node = SceneNode(
                    node_id=n["id"],
                    class_label=n["label"],
                    centroid=np.array(n["pos"], dtype=np.float32),
                    bbox_3d=np.zeros(6, dtype=np.float32),
                    embedding=np.zeros(64, dtype=np.float32),
                    color_rgb=np.array(n.get("color", [0.5, 0.5, 0.5]), dtype=np.float32),
                    confidence=float(n.get("conf", 1.0)),
                )
                self._scene_graph.add_or_update_node(node)
        except Exception as exc:
            logger.warning("SceneGraph deserialisation error: %s", exc)

    def _command_callback(self, msg) -> None:
        instruction = msg.data.strip()
        logger.info("Received NL command: '%s'", instruction)

        plan = self._grounder.plan(instruction, self._scene_graph)

        # Publish JSON plan
        plan_data = [
            {"action": sg.action, "object_desc": sg.object_desc,
             "node_id": sg.node_id, "confidence": sg.confidence}
            for sg in plan
        ]
        if _ROS2_AVAILABLE:
            out_msg = String()
            out_msg.data = json.dumps(plan_data)
            self._plan_pub.publish(out_msg)

            # Publish first grounded navigate_to goal node
            for sg in plan:
                if sg.action == "navigate_to" and sg.node_id is not None:
                    goal_msg = Int32()
                    goal_msg.data = sg.node_id
                    self._goal_pub.publish(goal_msg)
                    break


def main(args=None) -> None:
    if not _ROS2_AVAILABLE:
        return
    rclpy.init(args=args)
    node = LanguageNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
