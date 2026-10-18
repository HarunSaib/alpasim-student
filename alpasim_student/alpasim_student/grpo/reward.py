# SPDX-License-Identifier: Apache-2.0
"""
Reward functions for GRPO driving policy optimization.

All functions operate on numpy arrays and are batched over the N rollouts
that GRPO generates per scene per iteration.
"""
from __future__ import annotations

import numpy as np


def _ade(pred_xy: np.ndarray, ref_xy: np.ndarray) -> float:
    """Average Displacement Error between two (T, 2) trajectories."""
    # Align to the shorter length
    n = min(len(pred_xy), len(ref_xy))
    return float(np.mean(np.linalg.norm(pred_xy[:n] - ref_xy[:n], axis=1)))


def trajectory_reward(
    pred_xy: np.ndarray,
    ref_xy: np.ndarray,
    offroad: float,
    collision: float,
    *,
    offroad_penalty: float = -5.0,
    collision_penalty: float = -5.0,
    ade_scale: float = 2.0,
) -> float:
    """Compute a scalar reward for one student rollout.

    Args:
        pred_xy:          (T, 2) student-driven ego positions (world frame, relative to start).
        ref_xy:           (T_ref, 2) Alpamayo reference trajectory for the same scene.
        offroad:          Binary flag (0 or 1) from metrics.parquet.
        collision:        Binary flag (0 or 1) from metrics.parquet.
        offroad_penalty:  Fixed penalty added when offroad > 0.
        collision_penalty: Fixed penalty added when collision > 0.
        ade_scale:        ADE → reward scale: reward = exp(-ade * ade_scale).

    Returns:
        Scalar reward in approximately [-10, 1].
    """
    ade = _ade(pred_xy, ref_xy)
    r = float(np.exp(-ade * ade_scale))

    if offroad > 0:
        r += offroad_penalty
    if collision > 0:
        r += collision_penalty

    return r


def group_relative_advantages(rewards: np.ndarray) -> np.ndarray:
    """Normalize a group of rewards to zero-mean, unit-std advantages (GRPO).

    Args:
        rewards: (N,) reward for each rollout in the group.

    Returns:
        (N,) normalized advantages. If std ≈ 0, returns zeros.
    """
    mean = rewards.mean()
    std = rewards.std()
    if std < 1e-6:
        return np.zeros_like(rewards)
    return (rewards - mean) / (std + 1e-8)
