"""
aria.perception.sensor_fusion
==============================
Fuses RGB-D camera frames, LiDAR point clouds, and IMU odometry into a
unified, coloured, metric-scale point cloud suitable for downstream
PointNet++ encoding and occupancy mapping.

Pipeline per frame
------------------
1. Depth image → 3-D point cloud (camera frame) via back-projection.
2. Optionally merge sparse LiDAR scan via ICP-aligned registration.
3. Apply IMU-integrated pose to transform cloud into world frame.
4. Attach RGB colour to each 3-D point.
5. Voxel-downsample to constant density and return.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class RGBDFrame:
    """Single synchronised RGB-D frame."""
    rgb: np.ndarray          # (H, W, 3)  uint8
    depth: np.ndarray        # (H, W)     float32 in metres
    timestamp: float = 0.0   # seconds


@dataclass
class LiDARScan:
    """Single LiDAR scan as an (N, 3) xyz array in sensor frame."""
    points: np.ndarray       # (N, 3) float32
    timestamp: float = 0.0


@dataclass
class IMUMeasurement:
    """Single IMU sample."""
    linear_acc: np.ndarray   # (3,) m/s²
    angular_vel: np.ndarray  # (3,) rad/s
    timestamp: float = 0.0


@dataclass
class FusedFrame:
    """Output of SensorFusion.process()."""
    points_world: np.ndarray    # (N, 3) float32 — xyz in world frame
    colors: np.ndarray          # (N, 3) float32 — RGB in [0, 1]
    pose: np.ndarray            # (4, 4) float64 — world-T-camera
    timestamp: float = 0.0
    num_points: int = 0

    def to_open3d(self) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points_world.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(self.colors.astype(np.float64))
        return pcd

    @property
    def is_empty(self) -> bool:
        """Return True if the fused frame contains no points."""
        return self.num_points == 0

    def __repr__(self) -> str:
        t = self.pose[:3, 3].round(3).tolist()
        return (
            f"FusedFrame(t={self.timestamp:.3f}s, "
            f"pts={self.num_points}, pose_xyz={t})"
        )


# ---------------------------------------------------------------------------
# Camera intrinsics helper
# ---------------------------------------------------------------------------

@dataclass
class CameraIntrinsics:
    fx: float = 615.0
    fy: float = 615.0
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480
    depth_scale: float = 0.001      # 1 depth unit = depth_scale metres
    max_depth_m: float = 6.0

    def to_open3d(self) -> o3d.camera.PinholeCameraIntrinsic:
        return o3d.camera.PinholeCameraIntrinsic(
            width=self.width, height=self.height,
            fx=self.fx, fy=self.fy,
            cx=self.cx, cy=self.cy,
        )


# ---------------------------------------------------------------------------
# IMU pose integrator (simple Euler integration)
# ---------------------------------------------------------------------------

class _IMUIntegrator:
    """Lightweight IMU pose integrator (dead reckoning only)."""

    def __init__(self) -> None:
        self.position = np.zeros(3, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.rotation = np.eye(3, dtype=np.float64)
        self._prev_ts: Optional[float] = None

    def update(self, meas: IMUMeasurement) -> None:
        if self._prev_ts is None:
            self._prev_ts = meas.timestamp
            return
        dt = meas.timestamp - self._prev_ts
        if dt <= 0:
            return
        self._prev_ts = meas.timestamp

        # Integrate angular velocity → rotation (Euler)
        omega = meas.angular_vel.astype(np.float64)
        angle = float(np.linalg.norm(omega) * dt)
        if angle > 1e-8:
            axis = omega / (np.linalg.norm(omega) + 1e-12)
            K = np.array([
                [0,       -axis[2],  axis[1]],
                [axis[2],  0,       -axis[0]],
                [-axis[1], axis[0],  0      ],
            ], dtype=np.float64)
            R_delta = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            self.rotation = self.rotation @ R_delta

        # Integrate linear acceleration → velocity → position
        gravity = np.array([0, 0, -9.81], dtype=np.float64)
        acc_world = self.rotation @ meas.linear_acc.astype(np.float64) + gravity
        self.velocity += acc_world * dt
        self.position += self.velocity * dt

    @property
    def pose_matrix(self) -> np.ndarray:
        """Returns 4×4 world-T-sensor homogeneous transform."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.rotation
        T[:3,  3] = self.position
        return T


# ---------------------------------------------------------------------------
# Main SensorFusion class
# ---------------------------------------------------------------------------

class SensorFusion:
    """
    Fuses RGB-D + LiDAR + IMU data into a coloured metric point cloud.

    Parameters
    ----------
    intrinsics : CameraIntrinsics
        Camera calibration parameters.
    lidar_to_camera : np.ndarray | None
        4×4 extrinsic transform mapping LiDAR frame → camera frame.
        If None, sensors are assumed co-located (identity).
    voxel_size_m : float
        Voxel size used for downsampling the merged cloud.
    use_icp : bool
        Whether to refine LiDAR–RGB-D alignment with ICP.
    icp_max_iter : int
        Maximum ICP iterations per frame.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics | None = None,
        lidar_to_camera: np.ndarray | None = None,
        voxel_size_m: float = 0.03,
        use_icp: bool = False,
        icp_max_iter: int = 30,
        remove_outliers: bool = True,
        outlier_nb_neighbors: int = 20,
        outlier_std_ratio: float = 2.0,
    ) -> None:
        self.intrinsics = intrinsics or CameraIntrinsics()
        self.lidar_to_camera = (
            lidar_to_camera if lidar_to_camera is not None else np.eye(4, dtype=np.float64)
        )
        self.voxel_size_m = voxel_size_m
        self.use_icp = use_icp
        self.icp_max_iter = icp_max_iter
        self.remove_outliers = remove_outliers
        self.outlier_nb_neighbors = outlier_nb_neighbors
        self.outlier_std_ratio = outlier_std_ratio
        self._imu = _IMUIntegrator()
        logger.info(
            "SensorFusion initialised (voxel=%.3fm, icp=%s, outlier_removal=%s)",
            voxel_size_m, use_icp, remove_outliers,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_imu(self, meas: IMUMeasurement) -> None:
        """Feed a new IMU measurement to update the internal pose estimate."""
        self._imu.update(meas)

    def process(
        self,
        rgbd: RGBDFrame,
        lidar: LiDARScan | None = None,
        override_pose: np.ndarray | None = None,
    ) -> FusedFrame:
        """
        Fuse an RGB-D frame (and optional LiDAR scan) into a world-frame cloud.

        Parameters
        ----------
        rgbd : RGBDFrame
            Synchronised colour + depth image pair.
        lidar : LiDARScan | None
            Optional LiDAR scan in sensor frame.
        override_pose : np.ndarray | None
            If provided, use this 4×4 pose instead of IMU integration.

        Returns
        -------
        FusedFrame
            Metric coloured point cloud in world frame.
        """
        pose = override_pose if override_pose is not None else self._imu.pose_matrix

        # 1. Build RGB-D cloud in camera frame
        rgbd_cloud = self._rgbd_to_pointcloud(rgbd)

        # 2. Optionally merge LiDAR
        if lidar is not None:
            lidar_cloud = self._lidar_to_pointcloud(lidar)
            if self.use_icp:
                lidar_cloud = self._icp_align(lidar_cloud, rgbd_cloud)
            rgbd_cloud = rgbd_cloud + lidar_cloud

        # 3. Voxel downsample
        rgbd_cloud = rgbd_cloud.voxel_down_sample(self.voxel_size_m)

        # 4. Statistical outlier removal
        if self.remove_outliers and len(rgbd_cloud.points) > self.outlier_nb_neighbors:
            rgbd_cloud, _ = rgbd_cloud.remove_statistical_outlier(
                nb_neighbors=self.outlier_nb_neighbors,
                std_ratio=self.outlier_std_ratio,
            )

        # 5. Transform to world frame
        rgbd_cloud.transform(pose)

        pts = np.asarray(rgbd_cloud.points, dtype=np.float32)
        cols = np.asarray(rgbd_cloud.colors, dtype=np.float32)

        return FusedFrame(
            points_world=pts,
            colors=cols,
            pose=pose,
            timestamp=rgbd.timestamp,
            num_points=len(pts),
        )

    def reset_pose(self) -> None:
        """Reset IMU integrator to origin (alias: reset_imu)."""
        self._imu = _IMUIntegrator()

    # Alias for discoverable API
    reset_imu = reset_pose

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rgbd_to_pointcloud(self, rgbd: RGBDFrame) -> o3d.geometry.PointCloud:
        """Back-project depth image into a coloured 3-D point cloud."""
        intr = self.intrinsics
        rgb_o3d = o3d.geometry.Image(rgbd.rgb.astype(np.uint8))

        # Clip depth and convert to Open3D image
        depth_clipped = np.clip(rgbd.depth, 0.0, intr.max_depth_m).astype(np.float32)
        depth_o3d = o3d.geometry.Image((depth_clipped / intr.depth_scale).astype(np.uint16))

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            depth_scale=1.0 / intr.depth_scale,
            depth_trunc=intr.max_depth_m,
            convert_rgb_to_intensity=False,
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intr.to_open3d()
        )
        return pcd

    def _lidar_to_pointcloud(self, lidar: LiDARScan) -> o3d.geometry.PointCloud:
        """Convert LiDAR scan to Open3D cloud in camera frame via extrinsics."""
        pts_h = np.hstack([
            lidar.points.astype(np.float64),
            np.ones((len(lidar.points), 1), dtype=np.float64),
        ])
        pts_cam = (self.lidar_to_camera @ pts_h.T).T[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_cam)
        return pcd

    def _icp_align(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
    ) -> o3d.geometry.PointCloud:
        """Refine LiDAR→RGB-D alignment using point-to-point ICP."""
        result = o3d.pipelines.registration.registration_icp(
            source, target,
            max_correspondence_distance=0.2,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.icp_max_iter
            ),
        )
        source.transform(result.transformation)
        return source


# ---------------------------------------------------------------------------
# Factory from config dict
# ---------------------------------------------------------------------------

def sensor_fusion_from_config(cfg: dict) -> SensorFusion:
    """Build a SensorFusion instance from a parsed YAML config dict."""
    cam_cfg = cfg.get("camera", {})
    intr = CameraIntrinsics(
        fx=cam_cfg.get("fx", 615.0),
        fy=cam_cfg.get("fy", 615.0),
        cx=cam_cfg.get("cx", 320.0),
        cy=cam_cfg.get("cy", 240.0),
        width=cam_cfg.get("width", 640),
        height=cam_cfg.get("height", 480),
        depth_scale=cam_cfg.get("depth_scale", 0.001),
        max_depth_m=cam_cfg.get("max_depth_m", 6.0),
    )
    l2c_cfg = cfg.get("lidar_to_camera", {})
    t = np.array(l2c_cfg.get("translation", [0, 0, 0]), dtype=np.float64)
    from scipy.spatial.transform import Rotation
    r_deg = l2c_cfg.get("rotation_euler_deg", [0, 0, 0])
    R = Rotation.from_euler("xyz", r_deg, degrees=True).as_matrix()
    lidar_to_camera = np.eye(4, dtype=np.float64)
    lidar_to_camera[:3, :3] = R
    lidar_to_camera[:3, 3] = t

    return SensorFusion(intrinsics=intr, lidar_to_camera=lidar_to_camera)
