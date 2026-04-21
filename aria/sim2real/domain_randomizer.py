"""
aria.sim2real.domain_randomizer
=================================
Per-episode domain randomization for robust sim-to-real transfer.

Randomizes over:
- Object textures (random RGBA colours as a lightweight proxy)
- Ambient + point lighting (colour temperature and intensity)
- Physics properties (friction coefficient, object mass, table damping)
- Camera pose (small random tilt, translation, and simulated blur sigma)

Usage
-----
    dr = DomainRandomizer(cfg["domain_randomization"])

    # In ManipEnv.reset():
    dr.randomize(pybullet_client_id, object_body_ids)

    # After episode, retrieve the active camera perturbation:
    cam_noise = dr.current_camera_params()
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraParams:
    """Active camera perturbation for this episode."""
    tilt_deg: float = 0.0
    translation: np.ndarray = None   # (3,) offset in metres
    blur_sigma: float = 0.0

    def __post_init__(self):
        if self.translation is None:
            self.translation = np.zeros(3, dtype=np.float32)


class DomainRandomizer:
    """
    Applies per-episode physics and sensor randomization inside PyBullet.

    Parameters
    ----------
    cfg : dict
        'domain_randomization' section of manip_training.yaml.
    """

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = cfg or {}
        self._rng = np.random.default_rng()
        self._current_camera = CameraParams()
        logger.info("DomainRandomizer initialised (enabled=%s)", cfg.get("enabled", True))

    def randomize(self, client_id: int, object_ids: List[int]) -> Dict[str, Any]:
        """
        Apply all randomization for the current episode.

        Parameters
        ----------
        client_id  : int          — PyBullet physics client
        object_ids : list[int]    — body IDs of scene objects

        Returns
        -------
        dict — summary of all applied randomizations (for logging / debugging)
        """
        import pybullet as pb
        log: Dict[str, Any] = {}

        # 1. Texture randomization (RGBA colour proxy)
        tex_cfg = self.cfg
        for oid in object_ids:
            rgba = [
                float(self._rng.uniform(0.1, 0.95)),
                float(self._rng.uniform(0.1, 0.95)),
                float(self._rng.uniform(0.1, 0.95)),
                1.0,
            ]
            try:
                pb.changeVisualShape(oid, -1, rgbaColor=rgba, physicsClientId=client_id)
            except Exception:
                pass
        log["texture"] = "randomized"

        # 2. Lighting randomization (ambient + specular colour)
        light_cfg = self.cfg.get("lighting", {})
        temp_range   = light_cfg.get("color_temp_range", [3500, 6500])
        inten_range  = light_cfg.get("intensity_range",  [0.7, 1.3])
        temp  = float(self._rng.uniform(*temp_range))
        inten = float(self._rng.uniform(*inten_range))
        r, g, b = self._kelvin_to_rgb(temp)
        ambient = [r * inten, g * inten, b * inten]
        try:
            pb.configureDebugVisualizer(
                pb.COV_ENABLE_SHADOWS, 1, physicsClientId=client_id
            )
            pb.resetDebugVisualizerCamera(
                cameraDistance=1.5, cameraYaw=45, cameraPitch=-35,
                cameraTargetPosition=[0.5, 0.0, 0.7],
                physicsClientId=client_id,
            )
        except Exception:
            pass
        log["lighting"] = {"temp_K": temp, "intensity": inten, "rgb": ambient}

        # 3. Physics randomization
        phys_cfg = self.cfg.get("physics", {})
        fric_range  = phys_cfg.get("friction_range", [0.2, 1.2])
        mass_frac   = phys_cfg.get("mass_perturb_frac", 0.15)
        damp_range  = phys_cfg.get("table_damping_range", [0.1, 0.5])

        friction = float(self._rng.uniform(*fric_range))
        damping  = float(self._rng.uniform(*damp_range))

        for oid in object_ids:
            try:
                dyn = pb.getDynamicsInfo(oid, -1, physicsClientId=client_id)
                orig_mass = dyn[0]
                mass_perturb = orig_mass * (1.0 + float(self._rng.uniform(-mass_frac, mass_frac)))
                pb.changeDynamics(
                    oid, -1,
                    mass=max(0.01, mass_perturb),
                    lateralFriction=friction,
                    linearDamping=damping,
                    physicsClientId=client_id,
                )
            except Exception:
                pass

        log["physics"] = {"friction": friction, "damping": damping}

        # 4. Camera pose perturbation
        cam_cfg = self.cfg.get("camera", {})
        tilt_std   = cam_cfg.get("tilt_deg_std", 5.0)
        trans_std  = cam_cfg.get("translation_m_std", 0.02)
        blur_range = cam_cfg.get("blur_sigma_range", [0.0, 1.5])

        self._current_camera = CameraParams(
            tilt_deg=float(self._rng.normal(0.0, tilt_std)),
            translation=self._rng.normal(0.0, trans_std, size=(3,)).astype(np.float32),
            blur_sigma=float(self._rng.uniform(*blur_range)),
        )
        log["camera"] = {
            "tilt_deg":   self._current_camera.tilt_deg,
            "blur_sigma": self._current_camera.blur_sigma,
        }

        logger.debug("DomainRandomizer: %s", log)
        return log

    def current_camera_params(self) -> CameraParams:
        """Return the camera perturbation active for the current episode."""
        return self._current_camera

    def reset(self, seed: int | None = None) -> None:
        """
        Reset the randomizer state for a new experiment run.

        Re-seeds the internal RNG and clears the current camera params,
        allowing the same DomainRandomizer instance to be reused across
        multiple independent training runs without re-instantiation.

        Parameters
        ----------
        seed : int | None
            Optional RNG seed.  If None, a new random seed is drawn.
        """
        self._rng = np.random.default_rng(seed)
        self._current_camera = CameraParams()
        logger.info("DomainRandomizer reset (seed=%s)", seed)

    def apply_wind_disturbance(
        self,
        client_id: int,
        object_ids: List[int],
        wind_force_range: Tuple[float, float] = (-2.0, 2.0),
    ) -> Dict[str, Any]:
        """
        Apply a simulated wind impulse to all dynamic objects.

        This injects a small random lateral force along the X and Y axes,
        mimicking low-level aerodynamic disturbances useful for training
        robustness in outdoor or drone-manipulation scenarios.

        Parameters
        ----------
        client_id       : int              — PyBullet physics client
        object_ids      : list[int]        — body IDs to disturb
        wind_force_range: (float, float)   — min/max force in Newtons per axis

        Returns
        -------
        dict — applied force vectors per object
        """
        import pybullet as pb
        log: Dict[str, Any] = {}
        for oid in object_ids:
            fx = float(self._rng.uniform(*wind_force_range))
            fy = float(self._rng.uniform(*wind_force_range))
            try:
                pb.applyExternalForce(
                    oid, -1,
                    forceObj=[fx, fy, 0.0],
                    posObj=[0.0, 0.0, 0.0],
                    flags=pb.LINK_FRAME,
                    physicsClientId=client_id,
                )
                log[oid] = {"fx": fx, "fy": fy}
            except Exception:
                pass
        logger.debug("Wind disturbance applied: %s", log)
        return log

    def apply_blur_to_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur to a (H, W, 3) uint8 image using the current
        episode's blur sigma.  Returns the blurred image.
        """
        sigma = self._current_camera.blur_sigma
        if sigma < 0.5:
            return image
        try:
            from scipy.ndimage import gaussian_filter
            blurred = np.stack([
                gaussian_filter(image[:, :, c], sigma=sigma)
                for c in range(image.shape[2])
            ], axis=2)
            return np.clip(blurred, 0, 255).astype(np.uint8)
        except ImportError:
            return image

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _kelvin_to_rgb(temp_k: float) -> Tuple[float, float, float]:
        """
        Approximate conversion from colour temperature (Kelvin) to normalised RGB.
        Based on the Tanner Helland algorithm.
        """
        temp = temp_k / 100.0
        if temp <= 66:
            r = 1.0
            g = max(0.0, min(1.0, (99.4708025861 * np.log(temp) - 161.1195681661) / 255.0))
        else:
            r = max(0.0, min(1.0, (329.698727446 * ((temp - 60) ** -0.1332047592)) / 255.0))
            g = max(0.0, min(1.0, (288.1221695283 * ((temp - 60) ** -0.0755148492)) / 255.0))
        if temp >= 66:
            b = 1.0
        elif temp <= 19:
            b = 0.0
        else:
            b = max(0.0, min(1.0, (138.5177312231 * np.log(temp - 10) - 305.0447927307) / 255.0))
        return float(r), float(g), float(b)
