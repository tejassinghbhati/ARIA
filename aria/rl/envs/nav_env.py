"""
aria.rl.envs.nav_env
=====================
Gymnasium navigation environment backed by PyBullet.

The agent (a differential-drive robot base) must navigate to a goal object
identified by scene graph node ID.  Observations are scene-graph tensors;
actions are continuous base velocity commands.

Episode lifecycle
-----------------
1. Scene is loaded: table, shelves, random objects spawned.
2. Agent is placed at a random free position.
3. Goal node ID is sampled from the scene graph.
4. At each step: scene graph is refreshed → GNN encodes it →
   policy outputs [vx, vy, omega] → physics steps → reward computed.
5. Episode ends on goal reached OR collision OR max_steps.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pybullet as pb
import pybullet_data

from aria.rl.rewards import NavReward, NavRewardConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ROBOT_HALF_H    = 0.3             # metres — collision detection height
_ROBOT_RADIUS    = 0.25            # metres — circular footprint
_GRAVITY         = -9.81
_MAX_WHEEL_VEL   = 5.0             # rad/s

# Observation dimensions (must match GNN extractor)
_MAX_GRAPH_NODES = 64
_NODE_FEAT_DIM   = 9               # x, y, z, r, g, b, cx_norm, cy_norm, conf
_AGENT_STATE_DIM = 6               # pos(3) + vel(3)


# ---------------------------------------------------------------------------
# Helper: load a simple box URDF programmatically (no file needed)
# ---------------------------------------------------------------------------

def _create_box(
    client: int,
    half_extents: list[float],
    position: list[float],
    mass: float = 0.0,
    rgba: list[float] | None = None,
) -> int:
    """Create a static/dynamic box via PyBullet multi-body API."""
    col  = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=half_extents, physicsClientId=client)
    vis  = pb.createVisualShape(
        pb.GEOM_BOX, halfExtents=half_extents,
        rgbaColor=rgba or [0.6, 0.4, 0.2, 1.0],
        physicsClientId=client,
    )
    return pb.createMultiBody(mass, col, vis, position, physicsClientId=client)


def _create_sphere(
    client: int,
    radius: float,
    position: list[float],
    mass: float = 0.1,
    rgba: list[float] | None = None,
) -> int:
    col = pb.createCollisionShape(pb.GEOM_SPHERE, radius=radius, physicsClientId=client)
    vis = pb.createVisualShape(
        pb.GEOM_SPHERE, radius=radius,
        rgbaColor=rgba or [0.8, 0.2, 0.2, 1.0],
        physicsClientId=client,
    )
    return pb.createMultiBody(mass, col, vis, position, physicsClientId=client)


# ---------------------------------------------------------------------------
# NavEnv
# ---------------------------------------------------------------------------

class NavEnv(gym.Env):
    """
    PyBullet navigation environment for ARIA Phase 2.

    Observation space (Dict)
    ------------------------
    node_features : (MAX_NODES, NODE_FEAT_DIM) float32
    adj_matrix    : (MAX_NODES, MAX_NODES)      float32
    agent_state   : (AGENT_STATE_DIM,)          float32
    goal_idx      : (1,)                        int64   — valid node row index

    Action space
    ------------
    Box([-1,-1,-1], [1,1,1]) → scaled to [vx_max, vy_max, omega_max]
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, cfg: dict | None = None, render_mode: str | None = None) -> None:
        super().__init__()
        self.cfg = cfg or {}
        self.render_mode = render_mode

        env_cfg = self.cfg.get("environment", {})
        self._physics_hz     = env_cfg.get("physics_hz", 240)
        self._action_hz      = env_cfg.get("action_hz", 20)
        self._max_steps      = env_cfg.get("max_episode_steps", 500)
        self._spawn_range    = env_cfg.get("spawn_region_m", [-3.0, 3.0])
        self._gui            = render_mode == "human"

        obs_cfg = self.cfg.get("observation", {})
        self._max_nodes      = obs_cfg.get("max_graph_nodes", _MAX_GRAPH_NODES)
        self._node_feat_dim  = obs_cfg.get("node_feat_dim", _NODE_FEAT_DIM)

        # Reward
        self._reward_fn = NavReward()

        # PyBullet client id (-1 = not connected yet)
        self._client: int = -1
        self._robot_id: int = -1
        self._step_count: int = 0
        self._goal_pos: np.ndarray = np.zeros(3)
        self._goal_node_idx: int = 0

        # Simulated scene graph tensors (refreshed per step)
        self._node_features  = np.zeros((self._max_nodes, self._node_feat_dim), dtype=np.float32)
        self._adj_matrix     = np.zeros((self._max_nodes, self._max_nodes), dtype=np.float32)

        # Object bodies for scene tracking
        self._object_ids: list[int] = []
        self._object_goals: list[np.ndarray] = []

        # Define spaces
        self.observation_space = gym.spaces.Dict({
            "node_features": gym.spaces.Box(
                low=-10.0, high=10.0,
                shape=(self._max_nodes, self._node_feat_dim),
                dtype=np.float32,
            ),
            "adj_matrix": gym.spaces.Box(
                low=0.0, high=1.0,
                shape=(self._max_nodes, self._max_nodes),
                dtype=np.float32,
            ),
            "agent_state": gym.spaces.Box(
                low=-20.0, high=20.0,
                shape=(_AGENT_STATE_DIM,),
                dtype=np.float32,
            ),
            "goal_idx": gym.spaces.Box(
                low=0, high=self._max_nodes - 1,
                shape=(1,), dtype=np.int64,
            ),
        })

        # [vx, vy, omega] in normalised [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Max velocities for scaling
        self._vx_max    = 1.5    # m/s
        self._vy_max    = 1.0    # m/s
        self._omega_max = 1.5    # rad/s

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        super().reset(seed=seed)

        self._reward_fn.reset()
        self._step_count = 0

        if self._client >= 0:
            pb.disconnect(self._client)

        self._client = pb.connect(pb.GUI if self._gui else pb.DIRECT)
        pb.setGravity(0, 0, _GRAVITY, physicsClientId=self._client)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)

        # Timestep
        sim_dt = 1.0 / self._physics_hz
        pb.setTimeStep(sim_dt, physicsClientId=self._client)

        # Ground plane
        pb.loadURDF("plane.urdf", physicsClientId=self._client)

        # Spawn scene objects
        self._object_ids = []
        self._object_goals = []
        self._spawn_scene()

        # Spawn robot (simple sphere as holonomic base placeholder)
        spawn_xy = self.np_random.uniform(self._spawn_range[0] + 0.5, self._spawn_range[1] - 0.5, 2)
        self._robot_id = _create_sphere(
            self._client, _ROBOT_RADIUS,
            [float(spawn_xy[0]), float(spawn_xy[1]), _ROBOT_HALF_H],
            mass=5.0, rgba=[0.2, 0.4, 0.9, 1.0],
        )

        # Choose a random goal object
        if self._object_ids:
            goal_obj_idx = self.np_random.integers(0, len(self._object_ids))
            goal_pos_raw, _ = pb.getBasePositionAndOrientation(
                self._object_ids[goal_obj_idx], physicsClientId=self._client
            )
            self._goal_pos = np.array(goal_pos_raw, dtype=np.float32)
            self._goal_node_idx = goal_obj_idx
        else:
            self._goal_pos = np.zeros(3, dtype=np.float32)
            self._goal_node_idx = 0

        self._refresh_scene_graph()
        obs = self._get_obs()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        self._step_count += 1

        # Scale action
        vx    = float(action[0]) * self._vx_max
        vy    = float(action[1]) * self._vy_max
        omega = float(action[2]) * self._omega_max

        # Apply velocity to robot body (kinematic update)
        pos, orn = pb.getBasePositionAndOrientation(self._robot_id, physicsClientId=self._client)
        yaw = pb.getEulerFromQuaternion(orn)[2]

        # Transform local velocity to world frame
        dt = 1.0 / self._action_hz
        dx = (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
        dy = (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
        new_pos = [pos[0] + dx, pos[1] + dy, pos[2]]
        new_yaw = yaw + omega * dt
        new_orn = pb.getQuaternionFromEuler([0, 0, new_yaw])

        pb.resetBasePositionAndOrientation(
            self._robot_id, new_pos, new_orn, physicsClientId=self._client
        )

        # Step physics
        steps_per_action = self._physics_hz // self._action_hz
        for _ in range(steps_per_action):
            pb.stepSimulation(physicsClientId=self._client)

        # Collision check
        collision = self._check_collision()

        # Refresh scene graph observation
        self._refresh_scene_graph()

        # Reward
        agent_pos = np.array(new_pos, dtype=np.float32)
        reward, info = self._reward_fn(
            agent_pos=agent_pos,
            goal_pos=self._goal_pos,
            action=action,
            collision=collision,
        )

        # Termination
        terminated = bool(info["goal_reached"])
        truncated  = self._step_count >= self._max_steps

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            w, h = 640, 480
            view = pb.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0],
                distance=5,
                yaw=45, pitch=-30, roll=0,
                upAxisIndex=2, physicsClientId=self._client,
            )
            proj = pb.computeProjectionMatrixFOV(
                fov=60, aspect=w / h, nearVal=0.1, farVal=100.0, physicsClientId=self._client
            )
            _, _, rgb, _, _ = pb.getCameraImage(w, h, view, proj, physicsClientId=self._client)
            return np.array(rgb, dtype=np.uint8)[:, :, :3]
        return None

    def close(self) -> None:
        if self._client >= 0:
            pb.disconnect(self._client)
            self._client = -1

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _spawn_scene(self) -> None:
        """Spawn a random assortment of objects representing the indoor scene."""
        rng = self.np_random
        # Spawn 6–10 objects (boxes and spheres) at random positions
        n_objects = int(rng.integers(6, 11))
        colours = [
            [0.9, 0.3, 0.2, 1.0],   # red
            [0.2, 0.7, 0.3, 1.0],   # green
            [0.2, 0.3, 0.9, 1.0],   # blue
            [0.9, 0.8, 0.1, 1.0],   # yellow
            [0.6, 0.2, 0.7, 1.0],   # purple
        ]
        for i in range(n_objects):
            x = float(rng.uniform(self._spawn_range[0] + 0.3, self._spawn_range[1] - 0.3))
            y = float(rng.uniform(self._spawn_range[0] + 0.3, self._spawn_range[1] - 0.3))
            rgba = colours[i % len(colours)]
            if i % 2 == 0:
                oid = _create_box(
                    self._client,
                    [0.1, 0.1, 0.15],
                    [x, y, 0.15],
                    mass=0.3, rgba=rgba,
                )
            else:
                oid = _create_sphere(self._client, 0.07, [x, y, 0.07], mass=0.2, rgba=rgba)
            self._object_ids.append(oid)
            self._object_goals.append(np.array([x, y, 0.15], dtype=np.float32))

    def _refresh_scene_graph(self) -> None:
        """Update node_features and adj_matrix from current PyBullet state."""
        self._node_features[:] = 0.0
        self._adj_matrix[:] = 0.0

        for i, oid in enumerate(self._object_ids[:self._max_nodes]):
            pos, _ = pb.getBasePositionAndOrientation(oid, physicsClientId=self._client)
            # Node feature: normalised position + dummy colour + confidence
            feat = np.array([
                pos[0] / 5.0, pos[1] / 5.0, pos[2] / 2.0,  # xyz normalised
                0.5, 0.3, 0.2,                               # dummy rgb
                i / max(len(self._object_ids), 1),           # class pseudo-embedding
                0.0, 1.0,                                    # padding + confidence
            ], dtype=np.float32)
            self._node_features[i, :len(feat)] = feat[:self._node_feat_dim]

        # Build adjacency: connect nodes within 2m of each other
        n = min(len(self._object_ids), self._max_nodes)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                pi = self._node_features[i, :3]
                pj = self._node_features[j, :3]
                if np.linalg.norm(pi - pj) < 0.4:       # threshold in norm space
                    self._adj_matrix[i, j] = 1.0

    def _get_obs(self) -> Dict[str, np.ndarray]:
        pos, vel_lin_ang = pb.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._client
        )
        lin_vel, ang_vel = pb.getBaseVelocity(self._robot_id, physicsClientId=self._client)
        agent_state = np.array(
            list(pos) + list(lin_vel), dtype=np.float32
        )
        return {
            "node_features": self._node_features.copy(),
            "adj_matrix":    self._adj_matrix.copy(),
            "agent_state":   agent_state,
            "goal_idx":      np.array([self._goal_node_idx], dtype=np.int64),
        }

    def _check_collision(self) -> bool:
        """Return True if the robot body is in contact with any scene object."""
        for oid in self._object_ids:
            contacts = pb.getContactPoints(
                self._robot_id, oid, physicsClientId=self._client
            )
            if contacts:
                return True
        return False
