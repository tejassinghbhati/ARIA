"""
aria_ros.perception_node
=========================
ROS2 Lifecycle Node: sensor fusion → scene graph publisher.

Subscribes
----------
/camera/color/image_raw     sensor_msgs/Image
/camera/depth/image_raw     sensor_msgs/Image
/lidar/scan                 sensor_msgs/PointCloud2
/imu/data                   sensor_msgs/Imu

Publishes
---------
/aria/scene_graph           std_msgs/String (JSON serialised)
/aria/pointcloud            sensor_msgs/PointCloud2

Lifecycle
---------
configure → activate → deactivate → cleanup
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

import numpy as np

try:
    import rclpy
    from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image, PointCloud2, Imu
    from std_msgs.msg import String
    _ROS2_AVAILABLE = True
except ImportError:
    _ROS2_AVAILABLE = False
    # Provide stub base class for dev/testing without ROS2
    class LifecycleNode:  # type: ignore
        def __init__(self, name): self._name = name
    class TransitionCallbackReturn:  # type: ignore
        SUCCESS = None

try:
    import yaml
except ImportError:
    yaml = None

from aria.perception.sensor_fusion import SensorFusion, RGBDFrame, LiDARScan, IMUMeasurement
from aria.perception.occupancy_map import OccupancyMap
from aria.perception.scene_graph import SceneGraph, SceneNode
from aria.production.metrics_exporter import ARIAMetricsExporter

logger = logging.getLogger(__name__)


def _load_config(path: str = "/opt/aria/configs/perception.yaml") -> dict:
    try:
        with open(path) as f:
            import yaml as _yaml
            return _yaml.safe_load(f)
    except Exception as exc:
        logger.warning("Could not load config (%s) — using defaults", exc)
        return {}


class PerceptionNode(LifecycleNode if _ROS2_AVAILABLE else object):
    """ROS2 Lifecycle Node encapsulating the full ARIA perception pipeline."""

    def __init__(self) -> None:
        if _ROS2_AVAILABLE:
            super().__init__("aria_perception_node")
        self._cfg: dict = {}
        self._fusion: Optional[SensorFusion] = None
        self._omap:   Optional[OccupancyMap]  = None
        self._graph:  Optional[SceneGraph]    = None
        self._metrics = ARIAMetricsExporter(port=9091)
        self._latest_rgb:   Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # Lifecycle callbacks
    # ------------------------------------------------------------------

    def on_configure(self, state: "State") -> "TransitionCallbackReturn":
        self._cfg = _load_config()
        sf_cfg = self._cfg.get("sensor_fusion", {})
        self._fusion = SensorFusion(voxel_size_m=0.03)
        self._omap   = OccupancyMap(**{
            k: self._cfg.get("occupancy_map", {}).get(k, v)
            for k, v in [("resolution_m", 0.05), ("memory_frames", 50)]
        })
        self._graph  = SceneGraph(max_nodes=self._cfg.get("scene_graph", {}).get("max_nodes", 64))

        if _ROS2_AVAILABLE:
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            self._rgb_sub   = self.create_subscription(Image, "/camera/color/image_raw",
                                                        self._rgb_callback, qos)
            self._depth_sub = self.create_subscription(Image, "/camera/depth/image_raw",
                                                        self._depth_callback, qos)
            self._imu_sub   = self.create_subscription(Imu, "/imu/data",
                                                        self._imu_callback, 10)
            self._sg_pub    = self.create_publisher(String, "/aria/scene_graph", 10)
            self._timer     = self.create_timer(0.1, self._pipeline_step)  # 10 Hz
            self.get_logger().info("PerceptionNode configured")

        self._metrics.start()
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: "State") -> "TransitionCallbackReturn":
        if _ROS2_AVAILABLE:
            self.get_logger().info("PerceptionNode activated")
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: "State") -> "TransitionCallbackReturn":
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: "State") -> "TransitionCallbackReturn":
        if _ROS2_AVAILABLE:
            self.destroy_timer(self._timer)
        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------------
    # Sensor callbacks
    # ------------------------------------------------------------------

    def _rgb_callback(self, msg) -> None:
        try:
            self._latest_rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, -1
            )[:, :, :3]
        except Exception as exc:
            self._metrics.record_sensor_drop("rgb")
            logger.warning("RGB decode error: %s", exc)

    def _depth_callback(self, msg) -> None:
        try:
            dtype = np.uint16 if msg.encoding == "16UC1" else np.float32
            raw   = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
            self._latest_depth = raw.astype(np.float32) * 0.001  # → metres
        except Exception as exc:
            self._metrics.record_sensor_drop("depth")
            logger.warning("Depth decode error: %s", exc)

    def _imu_callback(self, msg) -> None:
        meas = IMUMeasurement(
            linear_acc=np.array([msg.linear_acceleration.x,
                                  msg.linear_acceleration.y,
                                  msg.linear_acceleration.z]),
            angular_vel=np.array([msg.angular_velocity.x,
                                   msg.angular_velocity.y,
                                   msg.angular_velocity.z]),
            timestamp=time.time(),
        )
        self._fusion.update_imu(meas)

    # ------------------------------------------------------------------
    # Pipeline step (timer callback)
    # ------------------------------------------------------------------

    def _pipeline_step(self) -> None:
        if self._latest_rgb is None or self._latest_depth is None:
            return

        with self._metrics.measure_latency("perception_pipeline"):
            rgbd  = RGBDFrame(
                rgb=self._latest_rgb.copy(),
                depth=self._latest_depth.copy(),
                timestamp=time.time(),
            )
            fused = self._fusion.process(rgbd)
            self._omap.update(fused.points_world)

            # ---- Object segmentation placeholder ----
            # In production this uses a segmentation model (e.g. Mask3D).
            # Here we demonstrate the graph update API with clustered centroids.
            self._update_scene_graph_from_clusters(fused.points_world, fused.colors)

        # Publish scene graph as JSON
        sg_json = self._serialise_graph()
        if _ROS2_AVAILABLE:
            msg = String()
            msg.data = sg_json
            self._sg_pub.publish(msg)

        self._frame_count += 1
        if self._frame_count % 50 == 0:
            logger.info("Perception: %d frames, %d graph nodes", self._frame_count,
                        self._graph.num_nodes())

    def _update_scene_graph_from_clusters(
        self, points: np.ndarray, colors: np.ndarray
    ) -> None:
        """Cluster point cloud into object segments and update the scene graph."""
        if len(points) < 10:
            return
        # Simple voxel-centroid clustering (production would use Mask3D/DBSCAN)
        step = max(1, len(points) // 8)
        for i in range(0, min(len(points), 8 * step), step):
            centroid = points[i:i + step].mean(axis=0)
            mean_col = colors[i:i + step].mean(axis=0)
            node = SceneNode(
                node_id=-1,
                class_label=f"object_{i // step}",
                centroid=centroid,
                bbox_3d=np.concatenate([centroid - 0.1, centroid + 0.1]),
                embedding=np.zeros(64, dtype=np.float32),
                color_rgb=mean_col,
                confidence=0.8,
            )
            self._graph.add_or_update_node(node)

    def _serialise_graph(self) -> str:
        nodes = []
        for node in self._graph.all_nodes():
            nodes.append({
                "id":    node.node_id,
                "label": node.class_label,
                "pos":   node.centroid.tolist(),
                "color": node.color_rgb.tolist(),
                "conf":  node.confidence,
            })
        edges = []
        for nid in self._graph.all_node_ids():
            for nbr, pred in self._graph.neighbours(nid):
                edges.append({"src": nid, "dst": nbr, "pred": pred})
        return json.dumps({"nodes": nodes, "edges": edges})


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    if not _ROS2_AVAILABLE:
        logger.error("ROS2 (rclpy) not available. Cannot start PerceptionNode.")
        return
    rclpy.init(args=args)
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
