"""Unit tests for aria.rl.rewards."""

import numpy as np
import pytest

from aria.rl.rewards import (
    NavReward,
    NavRewardConfig,
    ManipReward,
    ManipRewardConfig,
)


# ---------------------------------------------------------------------------
# NavReward
# ---------------------------------------------------------------------------

class TestNavReward:
    def test_progress_shaping(self) -> None:
        """Moving closer to goal yields positive progress reward."""
        cfg = NavRewardConfig(progress_weight=1.0, step_penalty=0.0, angular_efficiency_weight=0.0)
        r = NavReward(cfg)
        action = np.array([1.0, 0.0, 0.0])
        # First call — no prev dist, progress=0
        _, info1 = r(np.array([0.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0]), action)
        assert info1["r_progress"] == pytest.approx(0.0)
        # Second call — moved 1m closer
        _, info2 = r(np.array([1.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0]), action)
        assert info2["r_progress"] > 0.0

    def test_goal_reached_bonus(self) -> None:
        cfg = NavRewardConfig(goal_reached_bonus=10.0, goal_radius_m=0.3)
        r = NavReward(cfg)
        action = np.zeros(3)
        total, info = r(np.array([0.1, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), action)
        assert info["goal_reached"] is True
        assert info["r_goal"] == pytest.approx(10.0)

    def test_collision_penalty(self) -> None:
        cfg = NavRewardConfig(collision_penalty=-5.0, step_penalty=0.0, angular_efficiency_weight=0.0)
        r = NavReward(cfg)
        r(np.zeros(3), np.array([10.0, 0.0, 0.0]), np.zeros(3))  # prime prev_dist
        total, info = r(np.zeros(3), np.array([10.0, 0.0, 0.0]), np.zeros(3), collision=True)
        assert info["r_collision"] == pytest.approx(-5.0)

    def test_angular_efficiency_penalty(self) -> None:
        cfg = NavRewardConfig(angular_efficiency_weight=0.5, step_penalty=0.0)
        r = NavReward(cfg)
        # omega = 1.0
        action = np.array([0.0, 0.0, 1.0])
        _, info = r(np.zeros(3), np.array([10.0, 0.0, 0.0]), action)
        assert info["r_angular"] == pytest.approx(-0.5)

    def test_reset_clears_prev_dist(self) -> None:
        r = NavReward()
        r(np.zeros(3), np.array([5.0, 0.0, 0.0]), np.zeros(3))
        assert r._prev_dist is not None
        r.reset()
        assert r._prev_dist is None

    def test_info_keys_complete(self) -> None:
        r = NavReward()
        _, info = r(np.zeros(3), np.array([3.0, 0.0, 0.0]), np.zeros(3))
        expected = {"r_progress", "r_step", "r_collision", "r_goal", "r_angular", "dist_to_goal", "goal_reached"}
        assert expected == set(info.keys())


# ---------------------------------------------------------------------------
# ManipReward
# ---------------------------------------------------------------------------

class TestManipReward:
    def _pos(self, *args) -> np.ndarray:
        return np.array(args, dtype=np.float32)

    def test_ee_approach_shaping(self) -> None:
        """Moving EE closer to object should yield positive reward."""
        cfg = ManipRewardConfig(ee_distance_weight=2.0, step_penalty=0.0)
        r = ManipReward(cfg)
        obj = self._pos(0.5, 0.0, 0.5)
        goal = self._pos(0.0, 0.0, 0.0)
        r(self._pos(2.0, 0.0, 0.5), obj, goal)        # prime
        _, info = r(self._pos(1.0, 0.0, 0.5), obj, goal)
        assert info["r_ee_dist"] > 0.0

    def test_grasp_bonus_fires_once(self) -> None:
        """Grasp bonus should only fire on the first grasping=True step."""
        cfg = ManipRewardConfig(grasp_success_bonus=50.0, step_penalty=0.0, ee_distance_weight=0.0)
        r = ManipReward(cfg)
        obj = goal = self._pos(0.0, 0.0, 0.0)
        _, info1 = r(obj, obj, goal, grasp_force=5.0, grasping=True)
        assert info1["r_grasp"] == pytest.approx(50.0)
        _, info2 = r(obj, obj, goal, grasp_force=5.0, grasping=True)
        assert info2["r_grasp"] == pytest.approx(0.0)

    def test_place_success_bonus(self) -> None:
        cfg = ManipRewardConfig(place_success_bonus=30.0, step_penalty=0.0, ee_distance_weight=0.0)
        r = ManipReward(cfg)
        p = self._pos(0.0, 0.0, 0.0)
        _, info = r(p, p, p, placed=True)
        assert info["r_place"] == pytest.approx(30.0)

    def test_drop_penalty(self) -> None:
        cfg = ManipRewardConfig(drop_penalty=-10.0, step_penalty=0.0, ee_distance_weight=0.0)
        r = ManipReward(cfg)
        p = self._pos(0.0, 0.0, 0.0)
        _, info = r(p, p, p, dropped=True)
        assert info["r_drop"] == pytest.approx(-10.0)

    def test_tactile_reward_in_safe_range(self) -> None:
        """Force at midpoint of safe range should yield max grasp quality reward."""
        cfg = ManipRewardConfig(
            grasp_quality_weight=5.0, grasp_force_min=1.0, grasp_force_max=15.0,
            step_penalty=0.0, ee_distance_weight=0.0,
        )
        r = ManipReward(cfg)
        p = self._pos(0.0, 0.0, 0.0)
        mid_force = (cfg.grasp_force_min + cfg.grasp_force_max) / 2.0
        _, info = r(p, p, p, grasp_force=mid_force, grasping=True)
        assert info["r_tactile"] == pytest.approx(5.0, abs=1e-4)

    def test_tactile_penalty_excessive_force(self) -> None:
        """Force beyond grasp_force_max should incur a penalty."""
        cfg = ManipRewardConfig(
            grasp_quality_weight=5.0, grasp_force_min=1.0, grasp_force_max=15.0,
            step_penalty=0.0, ee_distance_weight=0.0,
        )
        r = ManipReward(cfg)
        p = self._pos(0.0, 0.0, 0.0)
        _, info = r(p, p, p, grasp_force=99.0, grasping=True)
        assert info["r_tactile"] < 0.0

    def test_reset_clears_state(self) -> None:
        r = ManipReward()
        p = self._pos(0.0, 0.0, 0.0)
        r(p, p, p, grasping=True)
        assert r._holding is True
        r.reset()
        assert r._holding is False
        assert r._prev_ee_dist is None

    def test_info_keys_complete(self) -> None:
        r = ManipReward()
        p = self._pos(0.0, 0.0, 0.0)
        _, info = r(p, p, p)
        expected = {
            "r_ee_dist", "r_tactile", "r_grasp", "r_place",
            "r_drop", "r_collision", "r_step", "ee_to_obj_m",
            "place_dist_m", "grasping", "grasp_force_N",
        }
        assert expected == set(info.keys())
