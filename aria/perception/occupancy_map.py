"""
aria.perception.occupancy_map
==============================
Maintains a 3-D voxel occupancy grid over the robot's local environment.
The grid uses a sliding-window memory so distant, stale observations are
automatically removed.

Usage
-----
    omap = OccupancyMap(resolution_m=0.05, extent_m=(10, 10, 3))

    # Feed coloured point clouds as the robot moves
    omap.update(points_world)            # np.ndarray (N, 3)

    # Query occupancy
    omap.is_occupied(np.array([1.2, 0.5, 0.8]))  # → True / False

    # Path-planning helper
    corridor = omap.get_free_corridor(start=(0,0,0.5), goal=(2,1,0.5))
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Type alias
Vec3 = Tuple[float, float, float] | np.ndarray


class OccupancyMap:
    """
    Probabilistic 3-D voxel occupancy grid.

    The map maintains a fixed-size array of voxel log-odds.  Observations
    are accumulated via log-odds update and a sliding-window of recent
    frame indices prevents stale readings from persisting indefinitely.

    Parameters
    ----------
    resolution_m : float
        Side-length of each cubic voxel in metres.
    extent_m : tuple(float, float, float)
        Physical extent of the grid (x, y, z) in metres.  The origin is at
        the centre of the grid.
    memory_frames : int
        Sliding-window depth: voxels not observed in the last N frames are
        decayed back towards the prior.
    obstacle_threshold : float
        Probability above which a voxel is considered occupied.
    p_occ : float
        Sensor model probability of occupied given ray hits voxel.
    p_free : float
        Sensor model probability of occupied given ray misses voxel.
    """

    def __init__(
        self,
        resolution_m: float = 0.05,
        extent_m: tuple[float, float, float] = (10.0, 10.0, 3.0),
        memory_frames: int = 50,
        obstacle_threshold: float = 0.6,
        p_occ: float = 0.85,
        p_free: float = 0.25,
    ) -> None:
        self.resolution = resolution_m
        self.extent = np.array(extent_m, dtype=np.float64)
        self.memory_frames = memory_frames
        self.obstacle_threshold = obstacle_threshold

        # Log-odds sensor model values
        self._lo_occ  = np.log(p_occ  / (1 - p_occ))   # +ve update
        self._lo_free = np.log(p_free / (1 - p_free))   # -ve update
        self._lo_min, self._lo_max = -3.0, 3.0

        # Compute grid dimensions
        self._dims = (self.extent / resolution_m).astype(int)  # (Dx, Dy, Dz)
        self._origin = -self.extent / 2.0                       # world coords of [0,0,0] voxel

        # Log-odds grid
        self._grid = np.zeros(self._dims, dtype=np.float32)

        # Sliding-window: queue of sets of voxel indices updated per frame
        self._frame_queue: deque[set] = deque(maxlen=memory_frames)
        self._frame_count = 0

        logger.info(
            "OccupancyMap: %.2fm resolution, grid %s, memory=%d frames",
            resolution_m, tuple(self._dims), memory_frames,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, points_world: np.ndarray) -> None:
        """
        Update the grid from a new point cloud observation.

        Parameters
        ----------
        points_world : (N, 3) float32/64
            Observed 3-D points in world frame.  Points outside the grid
            extent are silently ignored.
        """
        if len(points_world) == 0:
            self._frame_queue.append(set())
            return

        voxel_idx = self._world_to_voxel(points_world)          # (N, 3) int
        in_bounds = self._in_bounds_mask(voxel_idx)
        voxel_idx = voxel_idx[in_bounds]

        # Log-odds update: each hit voxel gets p_occ update
        xi, yi, zi = voxel_idx[:, 0], voxel_idx[:, 1], voxel_idx[:, 2]
        np.add.at(self._grid, (xi, yi, zi), self._lo_occ)

        # Clamp to prevent saturation
        np.clip(self._grid, self._lo_min, self._lo_max, out=self._grid)

        # Record which voxels were touched this frame
        touched = set(zip(xi.tolist(), yi.tolist(), zi.tolist()))
        self._frame_queue.append(touched)
        self._frame_count += 1

        # Decay voxels from expired frames that have NOT been re-observed
        if len(self._frame_queue) == self.memory_frames:
            expired = self._frame_queue[0]  # oldest frame (already popped by maxlen)
            still_active = set().union(*list(self._frame_queue)[1:])
            stale = expired - still_active
            for vi in stale:
                self._grid[vi] = max(self._grid[vi] + self._lo_free, self._lo_min)

    def is_occupied(self, point_world: Vec3, radius_m: float = 0.0) -> bool:
        """
        Check whether a world-frame point is in an occupied voxel.

        Parameters
        ----------
        point_world : array-like (3,)
        radius_m    : if > 0, dilates the query by checking neighbouring voxels.

        Returns
        -------
        bool
        """
        pt = np.asarray(point_world, dtype=np.float64)
        v = self._world_to_voxel(pt[np.newaxis])[0]
        if not self._in_bounds_mask(v[np.newaxis])[0]:
            return False

        if radius_m <= 0:
            return bool(self._prob(self._grid[v[0], v[1], v[2]]) >= self.obstacle_threshold)

        r_vox = max(1, int(np.ceil(radius_m / self.resolution)))
        lo = np.clip(v - r_vox, 0, self._dims - 1)
        hi = np.clip(v + r_vox + 1, 0, self._dims)
        patch = self._grid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        return bool(np.any(self._prob(patch) >= self.obstacle_threshold))

    def get_free_corridor(
        self,
        start: Vec3,
        goal: Vec3,
        n_samples: int = 20,
        safety_radius_m: float = 0.25,
    ) -> bool:
        """
        Check whether a straight-line path between start and goal is
        collision-free under the current occupancy estimate.

        Parameters
        ----------
        start, goal : world-frame 3-D points
        n_samples   : number of interpolation waypoints to test
        safety_radius_m : inflation radius for each waypoint check

        Returns
        -------
        bool — True if all waypoints are free
        """
        s = np.asarray(start, dtype=np.float64)
        g = np.asarray(goal,  dtype=np.float64)
        ts = np.linspace(0, 1, n_samples)
        for t in ts:
            pt = s + t * (g - s)
            if self.is_occupied(pt, radius_m=safety_radius_m):
                return False
        return True

    def probability_grid(self) -> np.ndarray:
        """Return the full 3-D probability grid (Dx, Dy, Dz) in [0,1]."""
        return self._prob(self._grid)

    def reset(self) -> None:
        """Clear the occupancy grid and frame history."""
        self._grid.fill(0.0)
        self._frame_queue.clear()
        self._frame_count = 0

    @property
    def dims(self) -> np.ndarray:
        """Grid dimensions (Dx, Dy, Dz)."""
        return self._dims.copy()

    @property
    def origin_world(self) -> np.ndarray:
        """World coordinates of voxel [0,0,0] corner."""
        return self._origin.copy()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _world_to_voxel(self, points: np.ndarray) -> np.ndarray:
        """Map (N,3) world coords → (N,3) integer voxel indices."""
        shifted = points - self._origin
        return (shifted / self.resolution).astype(int)

    def voxel_to_world(self, idx: np.ndarray) -> np.ndarray:
        """Map (N,3) voxel indices → (N,3) world centroid coords."""
        return idx.astype(np.float64) * self.resolution + self._origin + self.resolution / 2.0

    def _in_bounds_mask(self, idx: np.ndarray) -> np.ndarray:
        """Boolean mask: True where all three voxel indices are within bounds."""
        return np.all((idx >= 0) & (idx < self._dims), axis=-1)

    @staticmethod
    def _prob(log_odds: np.ndarray) -> np.ndarray:
        """Convert log-odds to probability via sigmoid."""
        return 1.0 / (1.0 + np.exp(-log_odds))


def occupancy_map_from_config(cfg: dict) -> OccupancyMap:
    """Build an OccupancyMap from a parsed YAML config dict."""
    return OccupancyMap(
        resolution_m=cfg.get("resolution_m", 0.05),
        extent_m=tuple(cfg.get("extent_m", [10.0, 10.0, 3.0])),
        memory_frames=cfg.get("memory_frames", 50),
        obstacle_threshold=cfg.get("obstacle_threshold", 0.6),
    )
