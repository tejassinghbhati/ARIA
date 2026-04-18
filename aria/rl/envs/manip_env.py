"""
aria.rl.envs.manip_env
========================
PyBullet manipulation environment for ARIA Phase 3.

Models a Franka Emika Panda 7-DOF arm on a table performing pick-and-place
of small objects.  Domain randomization (texture, lighting, physics) is
applied at episode reset via the DomainRandomizer.

Observation space (Dict)
------------------------
  node_features     : (MAX_NODES, NODE_FEAT_DIM)  — scene graph
  adj_matrix        : (MAX_NODES, MAX_NODES)
  ee_pose           : (7,)  — position(3) + quaternion(4)
  ft_sensor         : (6,)  — force(3) + torque(3) from simulated F/T
  joint_pos         : (7,)  — Panda joint angles
  joint_vel         : (7,)  — Panda joint velocities

Action space
------------
  Box(-1,1, shape=(7,)) — [Δpose(6) + grasp_force(1)]
  Δpose: [dx, dy, dz, droll, dpitch, dyaw] in normalised units
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pybullet as pb
import pybullet_data

from aria.rl.rewards import ManipReward, ManipRewardConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Franka Panda constants
# ---------------------------------------------------------------------------

PANDA_NUM_JOINTS  = 7
PANDA_EE_LINK     = 11       # end-effector link index in the URDF
PANDA_FINGER_J1   = 9
PANDA_FINGER_J2   = 10
PANDA_REST_JOINTS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

# Workspace AABB (metres, above table surface)
WS_MIN = np.array([0.25, -0.35, 0.63])
WS_MAX = np.array([0.75,  0.35, 1.10])

MAX_DELTA_POS = 0.02   # m per step
MAX_DELTA_ROT = 0.05   # rad per step
MAX_GRASP_N   = 20.0   # Newtons

_MAX_GRAPH_NODES = 32
_NODE_FEAT_DIM   = 9


# ---------------------------------------------------------------------------
# ManipEnv
# ---------------------------------------------------------------------------

class ManipEnv(gym.Env):
    """
    PyBullet Franka Panda pick-and-place environment.

    Parameters
    ----------
    cfg : dict
        Parsed manip_training.yaml.
    domain_randomizer : DomainRandomizer | None
        If provided, called at each episode reset.
    render_mode : str | None
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        cfg: dict | None = None,
        domain_randomizer=None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or {}
        self._randomizer = domain_randomizer
        self.render_mode = render_mode

        env_cfg = self.cfg.get("environment", {})
        self._physics_hz  = env_cfg.get("physics_hz", 240)
        self._action_hz   = env_cfg.get("action_hz", 20)
        self._max_steps   = env_cfg.get("max_episode_steps", 300)
        self._target_zone = np.array(env_cfg.get("target_zone_pos", [0.5, 0.0, 0.65]))

        obs_cfg = self.cfg.get("observation", {})
        self._max_nodes     = obs_cfg.get("max_graph_nodes", _MAX_GRAPH_NODES)
        self._node_feat_dim = obs_cfg.get("node_feat_dim", _NODE_FEAT_DIM)

        self._reward_fn = ManipReward()

        self._client: int = -1
        self._panda_id: int = -1
        self._obj_ids: list[int] = []
        self._target_obj_id: int = -1
        self._target_zone_id: int = -1
        self._step_count: int = 0
        self._grasping: bool = False
        self._grasp_constraint: int = -1

        # Simulated scene graph tensors
        self._node_features = np.zeros((self._max_nodes, self._node_feat_dim), dtype=np.float32)
        self._adj_matrix    = np.zeros((self._max_nodes, self._max_nodes), dtype=np.float32)

        # Spaces
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
            "ee_pose":    gym.spaces.Box(low=-3.0, high=3.0, shape=(7,), dtype=np.float32),
            "ft_sensor":  gym.spaces.Box(low=-100.0, high=100.0, shape=(6,), dtype=np.float32),
            "joint_pos":  gym.spaces.Box(low=-math.pi, high=math.pi, shape=(PANDA_NUM_JOINTS,), dtype=np.float32),
            "joint_vel":  gym.spaces.Box(low=-5.0, high=5.0, shape=(PANDA_NUM_JOINTS,), dtype=np.float32),
        })
        # [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, grasp_force_norm]
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        super().reset(seed=seed)
        self._reward_fn.reset()
        self._step_count = 0
        self._grasping   = False

        if self._client >= 0:
            pb.disconnect(self._client)

        self._client = pb.connect(pb.GUI if self.render_mode == "human" else pb.DIRECT)
        pb.setGravity(0, 0, -9.81, physicsClientId=self._client)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
        pb.setTimeStep(1.0 / self._physics_hz, physicsClientId=self._client)

        # Ground + table
        pb.loadURDF("plane.urdf", physicsClientId=self._client)
        pb.loadURDF(
            "table/table.urdf",
            basePosition=[0.5, 0.0, 0.0],
            baseOrientation=pb.getQuaternionFromEuler([0, 0, math.pi / 2]),
            physicsClientId=self._client,
        )

        # Load Franka Panda
        self._panda_id = pb.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0.0, 0.0, 0.625],
            useFixedBase=True,
            physicsClientId=self._client,
        )
        self._reset_arm()

        # Spawn objects on table
        self._obj_ids = []
        self._spawn_objects()

        # Domain randomization
        if self._randomizer is not None:
            self._randomizer.randomize(self._client, self._obj_ids)

        # Refresh scene graph
        self._refresh_scene_graph()

        obs = self._get_obs()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        self._step_count += 1
        action = np.clip(action, -1.0, 1.0)

        # Decode action
        delta_pos = action[:3] * MAX_DELTA_POS
        delta_rot = action[3:6] * MAX_DELTA_ROT
        grasp_force = float(action[6]) * MAX_GRASP_N   # [-20, 20] N

        # Get current EE state
        ee_state = pb.getLinkState(self._panda_id, PANDA_EE_LINK, physicsClientId=self._client)
        ee_pos   = np.array(ee_state[0], dtype=np.float32)
        ee_orn   = np.array(ee_state[1], dtype=np.float32)

        # Compute target EE pose (clamp to workspace)
        target_pos = np.clip(ee_pos + delta_pos, WS_MIN, WS_MAX)
        target_orn = self._integrate_rotation(ee_orn, delta_rot)

        # IK → joint commands
        joint_poses = pb.calculateInverseKinematics(
            self._panda_id, PANDA_EE_LINK,
            target_pos.tolist(), target_orn.tolist(),
            physicsClientId=self._client,
        )

        for j in range(PANDA_NUM_JOINTS):
            pb.setJointMotorControl2(
                self._panda_id, j,
                pb.POSITION_CONTROL,
                targetPosition=joint_poses[j],
                force=87.0, physicsClientId=self._client,
            )

        # Gripper control
        finger_pos = max(0.0, 0.04 - abs(grasp_force) / MAX_GRASP_N * 0.04)
        for fj in [PANDA_FINGER_J1, PANDA_FINGER_J2]:
            pb.setJointMotorControl2(
                self._panda_id, fj,
                pb.POSITION_CONTROL,
                targetPosition=finger_pos,
                force=20.0, physicsClientId=self._client,
            )

        # Step physics
        steps = self._physics_hz // self._action_hz
        for _ in range(steps):
            pb.stepSimulation(physicsClientId=self._client)

        # Post-step observations
        self._refresh_scene_graph()
        obs = self._get_obs()

        # Detect grasp and placement
        grasping  = self._detect_grasp(grasp_force)
        placed    = self._detect_place()
        dropped   = self._grasping and not grasping and not placed
        collision = self._detect_collision()
        self._grasping = grasping

        # Object position (for reward)
        obj_pos_raw, _ = pb.getBasePositionAndOrientation(
            self._target_obj_id, physicsClientId=self._client
        )
        obj_pos = np.array(obj_pos_raw, dtype=np.float32)

        # Compute reward
        new_ee = np.array(
            pb.getLinkState(self._panda_id, PANDA_EE_LINK, physicsClientId=self._client)[0],
            dtype=np.float32,
        )
        reward, info = self._reward_fn(
            ee_pos=new_ee,
            object_pos=obj_pos,
            goal_pos=self._target_zone,
            grasp_force=abs(grasp_force),
            grasping=grasping,
            placed=placed,
            dropped=dropped,
            collision=collision,
        )

        terminated = placed
        truncated  = self._step_count >= self._max_steps

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            w, h = 640, 480
            view = pb.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0.5, 0.0, 0.8],
                distance=1.2, yaw=45, pitch=-35, roll=0,
                upAxisIndex=2, physicsClientId=self._client,
            )
            proj = pb.computeProjectionMatrixFOV(
                60, w / h, 0.01, 10.0, physicsClientId=self._client
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

    def _reset_arm(self) -> None:
        for j, q in enumerate(PANDA_REST_JOINTS):
            pb.resetJointState(self._panda_id, j, q, physicsClientId=self._client)
        for fj in [PANDA_FINGER_J1, PANDA_FINGER_J2]:
            pb.resetJointState(self._panda_id, fj, 0.04, physicsClientId=self._client)

    def _spawn_objects(self) -> None:
        env_cfg = self.cfg.get("environment", {})
        n_objects = 3
        colours = [[0.8, 0.2, 0.2, 1], [0.2, 0.6, 0.2, 1], [0.2, 0.2, 0.8, 1]]
        rng = self.np_random
        for i in range(n_objects):
            x = float(rng.uniform(0.35, 0.65))
            y = float(rng.uniform(-0.2, 0.2))
            z = 0.67
            col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0.02, 0.02, 0.04],
                                          physicsClientId=self._client)
            vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.02, 0.02, 0.04],
                                       rgbaColor=colours[i], physicsClientId=self._client)
            oid = pb.createMultiBody(0.1, col, vis, [x, y, z], physicsClientId=self._client)
            self._obj_ids.append(oid)
        self._target_obj_id = self._obj_ids[0]

        # Visual goal zone marker (static)
        col_g = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0.06, 0.06, 0.001],
                                        physicsClientId=self._client)
        vis_g = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.06, 0.06, 0.001],
                                     rgbaColor=[0.1, 0.9, 0.1, 0.5], physicsClientId=self._client)
        self._target_zone_id = pb.createMultiBody(
            0, col_g, vis_g, self._target_zone.tolist(), physicsClientId=self._client
        )

    def _refresh_scene_graph(self) -> None:
        self._node_features[:] = 0.0
        self._adj_matrix[:] = 0.0
        for i, oid in enumerate(self._obj_ids[:self._max_nodes]):
            pos, _ = pb.getBasePositionAndOrientation(oid, physicsClientId=self._client)
            feat = np.array([
                pos[0] - 0.5, pos[1], pos[2] - 0.8,  # relative position
                0.5, 0.3, 0.2,                        # placeholder colour
                i / 3.0, 0.0, 1.0,
            ], dtype=np.float32)
            self._node_features[i, :self._node_feat_dim] = feat[:self._node_feat_dim]

    def _get_obs(self) -> Dict[str, np.ndarray]:
        ee_state   = pb.getLinkState(self._panda_id, PANDA_EE_LINK, physicsClientId=self._client)
        ee_pos     = np.array(ee_state[0], dtype=np.float32)
        ee_orn     = np.array(ee_state[1], dtype=np.float32)
        ee_pose    = np.concatenate([ee_pos, ee_orn])

        joint_states = pb.getJointStates(
            self._panda_id, list(range(PANDA_NUM_JOINTS)), physicsClientId=self._client
        )
        joint_pos = np.array([s[0] for s in joint_states], dtype=np.float32)
        joint_vel = np.array([s[1] for s in joint_states], dtype=np.float32)

        # Simulated F/T: reaction forces at EE joint
        ft = np.zeros(6, dtype=np.float32)
        try:
            contact_pts = pb.getContactPoints(
                self._panda_id, self._target_obj_id, physicsClientId=self._client
            )
            if contact_pts:
                forces = np.array([cp[9] for cp in contact_pts], dtype=np.float32)
                ft[2] = float(np.sum(forces))
        except Exception:
            pass

        return {
            "node_features": self._node_features.copy(),
            "adj_matrix":    self._adj_matrix.copy(),
            "ee_pose":       ee_pose,
            "ft_sensor":     ft,
            "joint_pos":     joint_pos,
            "joint_vel":     joint_vel,
        }

    def _detect_grasp(self, grasp_force: float) -> bool:
        """A grasp is detected if the gripper exerts force on the target object."""
        if abs(grasp_force) < 1.0:
            return False
        contacts = pb.getContactPoints(
            self._panda_id, self._target_obj_id, physicsClientId=self._client
        )
        return len(contacts) > 0

    def _detect_place(self) -> bool:
        """Placement: target object within 5cm of goal zone."""
        pos, _ = pb.getBasePositionAndOrientation(self._target_obj_id, physicsClientId=self._client)
        dist = np.linalg.norm(np.array(pos[:2]) - self._target_zone[:2])
        return bool(dist < 0.05)

    def _detect_collision(self) -> bool:
        """Panda arm self-collision or collision with floor/table edges."""
        contacts = pb.getContactPoints(self._panda_id, physicsClientId=self._client)
        for c in contacts:
            if c[8] < -0.001:   # penetration depth > 1mm
                return True
        return False

    @staticmethod
    def _integrate_rotation(orn_quat: np.ndarray, delta_euler: np.ndarray) -> list[float]:
        """Add Euler delta to a quaternion orientation."""
        euler = list(pb.getEulerFromQuaternion(orn_quat.tolist()))
        new_euler = [euler[i] + delta_euler[i] for i in range(3)]
        return list(pb.getQuaternionFromEuler(new_euler))
