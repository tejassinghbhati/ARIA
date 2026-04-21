"""
aria.rl.rewards
================
Reward functions for navigation and manipulation policies.

Both reward classes follow a call-signature contract:
    reward, info = RewardClass()(state, action, next_state, done)

All values are in SI units (metres, radians, Newtons).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Tuple


# ---------------------------------------------------------------------------
# Navigation reward
# ---------------------------------------------------------------------------

@dataclass
class NavRewardConfig:
    goal_reached_bonus: float = 10.0     # sparse terminal reward
    step_penalty: float = -0.1           # living cost per step
    collision_penalty: float = -5.0      # per collision event
    progress_weight: float = 1.0         # scale of ΔD shaping
    goal_radius_m: float = 0.3           # distance to count as "reached"
    angular_efficiency_weight: float = 0.05
    time_efficiency_bonus: float = 5.0   # bonus scaled by fraction of steps remaining


class NavReward:
    """
    Dense + sparse reward for goal-directed navigation.

    Shaping
    -------
    r_progress        = progress_weight * (d_prev - d_now)       ← potential-based
    r_step            = step_penalty                               ← living cost
    r_collision       = collision_penalty    (if collision detected)
    r_goal            = goal_reached_bonus   (if d_now < goal_radius)
    r_angular         = -angular_efficiency_weight * |ω|          ← prefer straight paths
    r_time_efficiency = time_efficiency_bonus * steps_fraction    ← early-arrival bonus
    """

    def __init__(self, cfg: NavRewardConfig | None = None, max_steps: int = 500) -> None:
        self.cfg = cfg or NavRewardConfig()
        self._prev_dist: float | None = None
        self._step_count: int = 0
        self._max_steps: int = max_steps

    def reset(self) -> None:
        self._prev_dist = None
        self._step_count = 0

    def __call__(
        self,
        agent_pos: np.ndarray,        # (3,)
        goal_pos: np.ndarray,         # (3,)
        action: np.ndarray,           # (3,) — [vx, vy, omega]
        collision: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute navigation reward for a single step.

        Returns
        -------
        reward : float
        info   : dict with component breakdown
        """
        self._step_count += 1
        dist = float(np.linalg.norm(agent_pos[:2] - goal_pos[:2]))

        # Progress shaping
        r_progress = 0.0
        if self._prev_dist is not None:
            r_progress = self.cfg.progress_weight * (self._prev_dist - dist)
        self._prev_dist = dist

        # Step penalty
        r_step = self.cfg.step_penalty

        # Collision
        r_collision = self.cfg.collision_penalty if collision else 0.0

        # Goal reached — add an early-arrival efficiency bonus
        goal_reached = dist < self.cfg.goal_radius_m
        r_goal = self.cfg.goal_reached_bonus if goal_reached else 0.0
        r_time_efficiency = 0.0
        if goal_reached and self._max_steps > 0:
            # Fraction of steps *remaining* when the goal is reached [0, 1]
            steps_fraction = max(0.0, 1.0 - self._step_count / self._max_steps)
            r_time_efficiency = self.cfg.time_efficiency_bonus * steps_fraction

        # Angular efficiency — penalise excessive turning
        omega = float(action[2]) if len(action) >= 3 else 0.0
        r_angular = -self.cfg.angular_efficiency_weight * abs(omega)

        total = r_progress + r_step + r_collision + r_goal + r_time_efficiency + r_angular

        info = {
            "r_progress":        r_progress,
            "r_step":            r_step,
            "r_collision":       r_collision,
            "r_goal":            r_goal,
            "r_time_efficiency": r_time_efficiency,
            "r_angular":         r_angular,
            "dist_to_goal":      dist,
            "goal_reached":      goal_reached,
            "step_count":        self._step_count,
        }
        return total, info


# ---------------------------------------------------------------------------
# Manipulation reward
# ---------------------------------------------------------------------------

@dataclass
class ManipRewardConfig:
    grasp_success_bonus: float = 50.0
    place_success_bonus: float = 30.0
    step_penalty: float = -0.5
    drop_penalty: float = -10.0
    collision_penalty: float = -10.0
    ee_distance_weight: float = 2.0           # reward for EE approaching object
    grasp_quality_weight: float = 5.0         # tactile quality bonus
    goal_radius_m: float = 0.05               # place success tolerance
    grasp_force_min: float = 1.0              # N — minimum force to count as grasp
    grasp_force_max: float = 15.0             # N — max before damage penalty
    orientation_alignment_weight: float = 1.0  # reward for top-down EE orientation


class ManipReward:
    """
    Dense + sparse reward for pick-and-place manipulation.

    Reward components
    -----------------
    r_ee_dist      : EE approach shaping (potential-based Δ distance)
    r_grasp        : sparse bonus on successful grasp
    r_tactile      : continuous reward based on grasp-force quality
    r_place        : sparse bonus on placing within goal tolerance
    r_drop         : penalty if held object is dropped
    r_collision    : penalty per collision
    r_step         : living cost
    r_orientation  : reward for maintaining a top-down (−z) EE orientation
    """

    def __init__(self, cfg: ManipRewardConfig | None = None) -> None:
        self.cfg = cfg or ManipRewardConfig()
        self._prev_ee_dist: float | None = None
        self._holding: bool = False

    def reset(self) -> None:
        self._prev_ee_dist = None
        self._holding = False

    def __call__(
        self,
        ee_pos: np.ndarray,              # (3,) end-effector world position
        object_pos: np.ndarray,          # (3,) target object world position
        goal_pos: np.ndarray,            # (3,) goal placement position
        grasp_force: float  = 0.0,       # N — from simulated F/T sensor
        grasping: bool      = False,      # true if gripper is closed on object
        placed: bool        = False,      # true if object is in goal zone
        dropped: bool       = False,
        collision: bool     = False,
        ee_z_axis: np.ndarray | None = None,  # (3,) EE approach axis in world frame
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute manipulation reward for a single step."""
        # EE → object distance shaping
        ee_dist = float(np.linalg.norm(ee_pos - object_pos))
        r_ee_dist = 0.0
        if self._prev_ee_dist is not None:
            r_ee_dist = self.cfg.ee_distance_weight * (self._prev_ee_dist - ee_dist)
        self._prev_ee_dist = ee_dist

        # Grasp quality reward (continuous, based on force being in safe window)
        r_tactile = 0.0
        if grasping and grasp_force > 0:
            in_range = self.cfg.grasp_force_min <= grasp_force <= self.cfg.grasp_force_max
            if in_range:
                # Reward peaks at the midpoint of the safe range
                mid = (self.cfg.grasp_force_min + self.cfg.grasp_force_max) / 2.0
                r_tactile = self.cfg.grasp_quality_weight * (
                    1.0 - abs(grasp_force - mid) / (mid - self.cfg.grasp_force_min + 1e-8)
                )
            else:
                r_tactile = -self.cfg.grasp_quality_weight * 0.5   # excessive force penalty

        # Orientation alignment — reward top-down (gravity-aligned) approach
        # The canonical grasp direction is [0, 0, -1] in world frame.
        r_orientation = 0.0
        if ee_z_axis is not None and self.cfg.orientation_alignment_weight > 0.0:
            axis_norm = ee_z_axis / (np.linalg.norm(ee_z_axis) + 1e-8)
            canonical = np.array([0.0, 0.0, -1.0])
            # dot ∈ [-1, 1]; map to [0, 1] and scale by weight
            alignment = float(np.dot(axis_norm, canonical))
            r_orientation = self.cfg.orientation_alignment_weight * max(0.0, alignment)

        # Sparse signals
        if grasping and not self._holding:
            r_grasp = self.cfg.grasp_success_bonus
            self._holding = True
        else:
            r_grasp = 0.0

        place_dist = float(np.linalg.norm(object_pos - goal_pos)) if grasping else float("inf")
        r_place    = self.cfg.place_success_bonus if placed else 0.0
        r_drop     = self.cfg.drop_penalty        if dropped else 0.0
        r_collision = self.cfg.collision_penalty  if collision else 0.0
        r_step      = self.cfg.step_penalty

        total = (
            r_ee_dist + r_tactile + r_orientation
            + r_grasp + r_place + r_drop + r_collision + r_step
        )

        info = {
            "r_ee_dist":     r_ee_dist,
            "r_tactile":     r_tactile,
            "r_orientation": r_orientation,
            "r_grasp":       r_grasp,
            "r_place":       r_place,
            "r_drop":        r_drop,
            "r_collision":   r_collision,
            "r_step":        r_step,
            "ee_to_obj_m":   ee_dist,
            "place_dist_m":  place_dist,
            "grasping":      grasping,
            "grasp_force_N": grasp_force,
        }
        return total, info
