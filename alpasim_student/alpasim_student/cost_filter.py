# SPDX-License-Identifier: Apache-2.0
"""
Terrain/safety cost functions based on:
"A Survey on Path Planning for Autonomous Ground Vehicles in Unstructured Environments"
MDPI Machines 12(1), 2024.

Three cost components:
  safety  — APF repulsive potential from obstacles
  comfort — trajectory jerk (third derivative of position)
  energy  — path length proxy
"""
from __future__ import annotations

import numpy as np


def safety_cost(
    trajectory_xy: np.ndarray,
    obstacle_positions: np.ndarray,
    d0: float = 5.0,
    k_rep: float = 1.0,
) -> float:
    """APF repulsive potential summed over all waypoints.

    U_rep(d) = 0.5 * k_rep * (1/d - 1/d0)^2  for d < d0, else 0

    Args:
        trajectory_xy:      (T, 2) waypoints in rig frame (x-forward, y-left), metres.
        obstacle_positions: (N, 2) obstacle centres in same frame.
        d0:                 Influence radius in metres.
        k_rep:              Repulsive gain.
    """
    if len(obstacle_positions) == 0:
        return 0.0
    total = 0.0
    for wp in trajectory_xy:
        dists = np.linalg.norm(obstacle_positions - wp, axis=1)
        in_range = dists < d0
        if in_range.any():
            d = max(float(dists[in_range].min()), 1e-3)
            total += 0.5 * k_rep * (1.0 / d - 1.0 / d0) ** 2
    return total


def comfort_cost(trajectory_xy: np.ndarray, dt: float = 0.1) -> float:
    """Mean jerk magnitude along the trajectory.

    Jerk approximated as second finite difference of velocity.
    Lower is smoother/more comfortable.

    Args:
        trajectory_xy: (T, 2) waypoints, metres.
        dt:            Time between waypoints, seconds.
    """
    if len(trajectory_xy) < 4:
        return 0.0
    vel  = np.diff(trajectory_xy, axis=0) / dt   # (T-1, 2)
    acc  = np.diff(vel,           axis=0) / dt   # (T-2, 2)
    jerk = np.diff(acc,           axis=0) / dt   # (T-3, 2)
    return float(np.mean(np.linalg.norm(jerk, axis=1)))


def energy_cost(trajectory_xy: np.ndarray) -> float:
    """Total path length as an energy proxy."""
    if len(trajectory_xy) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(trajectory_xy, axis=0), axis=1)))


def total_cost(
    trajectory_xy: np.ndarray,
    obstacle_positions: np.ndarray,
    w_safety: float = 2.0,
    w_comfort: float = 1.0,
    w_energy: float = 0.5,
) -> float:
    """Weighted composite cost."""
    return (
        w_safety  * safety_cost(trajectory_xy, obstacle_positions)
        + w_comfort * comfort_cost(trajectory_xy)
        + w_energy  * energy_cost(trajectory_xy)
    )
