# SPDX-License-Identifier: Apache-2.0
"""
Gradient-free MPC trajectory refinement.

Takes an initial trajectory (from Alpamayo or the student) and iteratively
nudges waypoints to reduce the composite cost while staying close to the
original prediction.  Uses finite-difference gradient estimation — no autograd
required, so it runs safely inside the driver inference hot path (~1 ms for
T=10, N_obs=20 on CPU).
"""
from __future__ import annotations

import numpy as np

from .cost_filter import total_cost


def refine_trajectory(
    initial_traj: np.ndarray,
    obstacle_positions: np.ndarray,
    n_iters: int = 20,
    lr: float = 0.05,
    w_safety: float = 2.0,
    w_comfort: float = 1.0,
    w_energy: float = 0.3,
    w_deviation: float = 3.0,
) -> np.ndarray:
    """Refine a trajectory by minimising a composite cost.

    Objective:
        J = w_safety*C_safety + w_comfort*C_comfort + w_energy*C_energy
          + w_deviation * ||traj - initial_traj||^2

    The deviation term keeps the refined path close to the model's prediction
    so we don't override the learned policy more than necessary.

    Args:
        initial_traj:       (T, 2) raw waypoints from teacher/student, metres.
        obstacle_positions: (N, 2) current-frame obstacle centres, metres.
        n_iters:            Number of gradient-descent steps.
        lr:                 Step size.
        w_*:                Cost weights.

    Returns:
        (T, 2) refined trajectory.
    """
    traj = initial_traj.copy()
    eps = 1e-3

    for _ in range(n_iters):
        dev = w_deviation * float(np.mean((traj - initial_traj) ** 2))
        base = total_cost(traj, obstacle_positions, w_safety, w_comfort, w_energy) + dev

        grad = np.zeros_like(traj)
        for i in range(len(traj)):
            for j in range(2):
                traj[i, j] += eps
                dev_p = w_deviation * float(np.mean((traj - initial_traj) ** 2))
                c_plus = (
                    total_cost(traj, obstacle_positions, w_safety, w_comfort, w_energy)
                    + dev_p
                )
                traj[i, j] -= eps
                grad[i, j] = (c_plus - base) / eps

        traj -= lr * grad

    return traj
