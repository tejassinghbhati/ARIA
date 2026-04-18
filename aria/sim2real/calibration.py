"""
aria.sim2real.calibration
==========================
Sensor calibration utilities for deploying trained models on physical hardware.

Provides:
- CameraCalibration : intrinsic/extrinsic parameters and undistortion
- LiDARCalibration  : extrinsic transform relative to camera frame
- CalibrationBundle : combined multi-sensor calibration loaded from YAML
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Camera calibration
# ---------------------------------------------------------------------------

@dataclass
class CameraCalibration:
    """
    Pinhole camera model with optional distortion coefficients.

    Parameters (OpenCV convention)
    ----------
    K : (3, 3) intrinsic matrix
    D : (1..5,) distortion coefficients [k1, k2, p1, p2, k3]
    R : (3, 3) rotation from camera to reference (extrinsic)
    t : (3,)   translation from camera to reference (extrinsic)
    width, height : image resolution
    """
    K: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    D: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float64))
    R: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    width:  int = 640
    height: int = 480

    @property
    def fx(self) -> float: return float(self.K[0, 0])
    @property
    def fy(self) -> float: return float(self.K[1, 1])
    @property
    def cx(self) -> float: return float(self.K[0, 2])
    @property
    def cy(self) -> float: return float(self.K[1, 2])

    @property
    def extrinsic_matrix(self) -> np.ndarray:
        """4×4 homogeneous world-T-camera transform."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R
        T[:3, 3]  = self.t
        return T

    def project_3d_to_2d(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Project (N, 3) 3-D points in camera frame to (N, 2) pixel coords.
        Applies simple pinhole projection (no distortion correction).
        """
        z = points_3d[:, 2:3] + 1e-8
        u = self.fx * points_3d[:, 0:1] / z + self.cx
        v = self.fy * points_3d[:, 1:2] / z + self.cy
        return np.hstack([u, v]).astype(np.float32)

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """Remove lens distortion from an RGB image (requires OpenCV)."""
        try:
            import cv2
            return cv2.undistort(image, self.K, self.D)
        except ImportError:
            logger.warning("OpenCV not available — returning image without undistortion")
            return image

    @classmethod
    def from_dict(cls, d: dict) -> "CameraCalibration":
        K = np.array(d.get("K", [[615, 0, 320], [0, 615, 240], [0, 0, 1]]), dtype=np.float64)
        D = np.array(d.get("D", [0, 0, 0, 0, 0]), dtype=np.float64)
        R = np.array(d.get("R", np.eye(3).tolist()), dtype=np.float64)
        t = np.array(d.get("t", [0, 0, 0]), dtype=np.float64)
        return cls(K=K, D=D, R=R, t=t,
                   width=d.get("width", 640), height=d.get("height", 480))


# ---------------------------------------------------------------------------
# LiDAR calibration
# ---------------------------------------------------------------------------

@dataclass
class LiDARCalibration:
    """
    Rigid-body extrinsic transform from LiDAR sensor frame to camera frame.

    T_lidar_to_camera : (4, 4) homogeneous transform
    """
    T_lidar_to_camera: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))

    @property
    def rotation(self) -> np.ndarray:
        return self.T_lidar_to_camera[:3, :3]

    @property
    def translation(self) -> np.ndarray:
        return self.T_lidar_to_camera[:3, 3]

    def transform_points(self, pts_lidar: np.ndarray) -> np.ndarray:
        """Map (N, 3) LiDAR points to camera frame."""
        pts_h = np.hstack([pts_lidar, np.ones((len(pts_lidar), 1))])
        return (self.T_lidar_to_camera @ pts_h.T).T[:, :3]

    @classmethod
    def from_dict(cls, d: dict) -> "LiDARCalibration":
        from scipy.spatial.transform import Rotation
        t = np.array(d.get("translation", [0, 0, 0]), dtype=np.float64)
        r_deg = d.get("rotation_euler_deg", [0, 0, 0])
        R = Rotation.from_euler("xyz", r_deg, degrees=True).as_matrix()
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3]  = t
        return cls(T_lidar_to_camera=T)


# ---------------------------------------------------------------------------
# Calibration bundle
# ---------------------------------------------------------------------------

@dataclass
class CalibrationBundle:
    """Combined multi-sensor calibration for a physical robot platform."""
    camera: CameraCalibration = field(default_factory=CameraCalibration)
    lidar:  LiDARCalibration  = field(default_factory=LiDARCalibration)

    @classmethod
    def load(cls, yaml_path: str | Path) -> "CalibrationBundle":
        """Load calibration from a YAML file."""
        with open(str(yaml_path)) as f:
            data = yaml.safe_load(f)
        cam   = CameraCalibration.from_dict(data.get("camera", {}))
        lidar = LiDARCalibration.from_dict(data.get("lidar", {}))
        logger.info("Calibration loaded from: %s", yaml_path)
        return cls(camera=cam, lidar=lidar)

    def save(self, yaml_path: str | Path) -> None:
        """Serialize calibration to YAML."""
        data = {
            "camera": {
                "K":      self.camera.K.tolist(),
                "D":      self.camera.D.tolist(),
                "R":      self.camera.R.tolist(),
                "t":      self.camera.t.tolist(),
                "width":  self.camera.width,
                "height": self.camera.height,
            },
            "lidar": {
                "translation":      self.lidar.translation.tolist(),
                "rotation_euler_deg": [0.0, 0.0, 0.0],  # stored as T matrix
            },
        }
        with open(str(yaml_path), "w") as f:
            yaml.safe_dump(data, f)
        logger.info("Calibration saved to: %s", yaml_path)


def default_realsense_d435() -> CalibrationBundle:
    """Return a CalibrationBundle with typical RealSense D435 parameters."""
    K = np.array([[615.0, 0.0, 320.0],
                  [0.0, 615.0, 240.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return CalibrationBundle(camera=CameraCalibration(K=K))
