"""
aria_ros.nav_policy_node
=========================
ROS2 Lifecycle Node: runs the trained navigation RL policy.

Subscribes  /aria/scene_graph         std_msgs/String (JSON)
            /aria/current_goal_node   std_msgs/Int32

Publishes   /cmd_vel                  geometry_msgs/Twist
"""

from __future__ import annotations

import json
import logging

import numpy as np

try:
    import rclpy
    from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
    from std_msgs.msg import String, Int32
    from geometry_msgs.msg import Twist
    _ROS2_AVAILABLE = True
except ImportError:
    _ROS2_AVAILABLE = False
    class LifecycleNode:  # type: ignore
        def __init__(self, name): pass
    class TransitionCallbackReturn:  # type: ignore
        SUCCESS = None

from aria.production.tensorrt_engine import build_engine, ONNXEngine
from aria.production.metrics_exporter import ARIAMetricsExporter

logger = logging.getLogger(__name__)


class NavPolicyNode(LifecycleNode if _ROS2_AVAILABLE else object):
    """Runs the navigation policy and publishes velocity commands."""

    def __init__(self) -> None:
        if _ROS2_AVAILABLE:
            super().__init__("aria_nav_policy_node")
        self._engine = None
        self._metrics = ARIAMetricsExporter(port=9092)
        self._current_goal_node: int = 0
        self._node_features = np.zeros((64, 9), dtype=np.float32)
        self._adj_matrix    = np.zeros((64, 64), dtype=np.float32)

    def on_configure(self, state: "State") -> "TransitionCallbackReturn":
        onnx_path = "/opt/aria/exports/onnx/aria_nav_policy.onnx"
        try:
            self._engine = build_engine(onnx_path, device="cpu")
            logger.info("Nav policy engine loaded")
        except FileNotFoundError:
            logger.warning("Nav policy ONNX not found at %s — node will not publish actions", onnx_path)

        if _ROS2_AVAILABLE:
            self._sg_sub   = self.create_subscription(
                String, "/aria/scene_graph", self._sg_callback, 10
            )
            self._goal_sub = self.create_subscription(
                Int32, "/aria/current_goal_node", self._goal_callback, 10
            )
            self._cmd_pub  = self.create_publisher(Twist, "/cmd_vel", 10)
            self._timer    = self.create_timer(0.05, self._policy_step)  # 20 Hz
        self._metrics.start()
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: "State") -> "TransitionCallbackReturn":
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: "State") -> "TransitionCallbackReturn":
        if _ROS2_AVAILABLE:
            twist = Twist()
            self._cmd_pub.publish(twist)   # stop robot
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: "State") -> "TransitionCallbackReturn":
        if _ROS2_AVAILABLE:
            self.destroy_timer(self._timer)
        return TransitionCallbackReturn.SUCCESS

    def _sg_callback(self, msg) -> None:
        try:
            data = json.loads(msg.data)
            self._node_features[:] = 0.0
            self._adj_matrix[:] = 0.0
            for i, n in enumerate(data.get("nodes", [])[:64]):
                pos = n.get("pos", [0, 0, 0])
                self._node_features[i, :3] = pos
            for e in data.get("edges", []):
                s, d = e.get("src", 0), e.get("dst", 0)
                if 0 <= s < 64 and 0 <= d < 64:
                    self._adj_matrix[s, d] = 1.0
        except Exception as exc:
            logger.warning("NavPolicyNode sg_callback error: %s", exc)

    def _goal_callback(self, msg) -> None:
        self._current_goal_node = msg.data

    def _policy_step(self) -> None:
        if self._engine is None or not _ROS2_AVAILABLE:
            return
        inputs = {
            "node_features": self._node_features[np.newaxis],   # (1,64,9)
            "adj_matrix":    self._adj_matrix[np.newaxis],      # (1,64,64)
            "agent_state":   np.zeros((1, 6), dtype=np.float32),
            "goal_idx":      np.array([[self._current_goal_node]], dtype=np.int64),
        }
        with self._metrics.measure_latency("nav_policy"):
            try:
                action = self._engine.infer(inputs)[0][0]   # (3,)
            except Exception as exc:
                logger.warning("Nav policy inference error: %s", exc)
                return

        twist = Twist()
        twist.linear.x  = float(np.clip(action[0], -1.5, 1.5))
        twist.linear.y  = float(np.clip(action[1], -1.0, 1.0))
        twist.angular.z = float(np.clip(action[2], -1.5, 1.5))
        self._cmd_pub.publish(twist)


def main(args=None) -> None:
    if not _ROS2_AVAILABLE:
        return
    rclpy.init(args=args)
    node = NavPolicyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
