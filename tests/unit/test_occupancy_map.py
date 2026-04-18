"""Unit tests for OccupancyMap."""

import numpy as np
import pytest

from aria.perception.occupancy_map import OccupancyMap


def test_init() -> None:
    omap = OccupancyMap(resolution_m=0.1, extent_m=(4.0, 4.0, 2.0))
    assert omap.dims[0] == 40
    assert omap.dims[1] == 40
    assert omap.dims[2] == 20


def test_update_and_occupied() -> None:
    omap = OccupancyMap(resolution_m=0.1, extent_m=(4.0, 4.0, 2.0), obstacle_threshold=0.5)
    # Feed the same point 10 times to push it clearly above threshold
    point = np.array([[0.5, 0.0, 0.5]])
    for _ in range(10):
        omap.update(point)
    assert omap.is_occupied(np.array([0.5, 0.0, 0.5]))


def test_empty_after_reset() -> None:
    omap = OccupancyMap(resolution_m=0.1, extent_m=(4.0, 4.0, 2.0), obstacle_threshold=0.5)
    point = np.array([[0.5, 0.0, 0.5]])
    for _ in range(10):
        omap.update(point)
    omap.reset()
    assert not omap.is_occupied(np.array([0.5, 0.0, 0.5]))


def test_free_corridor_clear() -> None:
    omap = OccupancyMap(resolution_m=0.1, extent_m=(6.0, 6.0, 2.0))
    # Empty map — corridor should be free
    assert omap.get_free_corridor((0, 0, 0.5), (1, 0, 0.5))


def test_free_corridor_blocked() -> None:
    omap = OccupancyMap(resolution_m=0.1, extent_m=(6.0, 6.0, 2.0), obstacle_threshold=0.5)
    # Block the midpoint of the corridor
    mid = np.array([[0.5, 0.0, 0.5]])
    for _ in range(20):
        omap.update(mid)
    blocked = not omap.get_free_corridor((0, 0, 0.5), (1, 0, 0.5), safety_radius_m=0.05)
    assert blocked


def test_out_of_bounds_ignored() -> None:
    omap = OccupancyMap(resolution_m=0.1, extent_m=(2.0, 2.0, 2.0))
    far_point = np.array([[100.0, 100.0, 100.0]])
    omap.update(far_point)   # Should not raise
    assert not omap.is_occupied(np.array([100.0, 100.0, 100.0]))


def test_probability_grid_shape() -> None:
    omap = OccupancyMap(resolution_m=0.1, extent_m=(2.0, 2.0, 1.0))
    grid = omap.probability_grid()
    assert grid.shape == (20, 20, 10)
    assert np.all(grid >= 0.0) and np.all(grid <= 1.0)
