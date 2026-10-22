# SPDX-License-Identifier: Apache-2.0
"""Label construction and masked-loss tests (no simulator needed)."""
import math

import numpy as np
import pytest
import torch

from alpasim_student.alpanet.alpanet_model import NUM_WAYPOINTS, winner_takes_all_loss
from alpasim_student.dagger.collector import _STEP_US, _teacher_targets


def _driver_local_positions(x0, y0, yaw, offsets_in_rig):
    """The driver service's rig-offset -> scene-frame conversion (alpasim_driver.main)."""
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    return np.asarray(offsets_in_rig) @ rotation.T + np.array([x0, y0])


def _poses(t_req, xy):
    """Current pose at t_req followed by one pose per 100 ms, as the driver returns them."""
    return [(t_req + k * _STEP_US, float(p[0]), float(p[1])) for k, p in enumerate(xy)]


def test_straight_ahead_at_90_degrees_becomes_forward():
    t0 = 1_000_000
    world = [(0.0, 0.0)] + [(0.0, 0.25 * (k + 1)) for k in range(NUM_WAYPOINTS)]   # along +y
    traj, valid = _teacher_targets(_poses(t0, world), 0.0, 0.0, math.pi / 2, t0)
    assert valid.all()
    assert traj[-1] == pytest.approx([0.25 * NUM_WAYPOINTS, 0.0], abs=1e-4)


@pytest.mark.parametrize("yaw", [0.0, 0.3, -1.2, math.pi / 2, 2.8, -3.0])
def test_round_trip_through_driver_conversion(yaw):
    t0, ego = 5_000_000, (12.0, -7.5)
    rig = np.stack([np.linspace(0.5, 20, NUM_WAYPOINTS), np.linspace(0.0, 3.0, NUM_WAYPOINTS)], 1)
    world = np.vstack([ego, _driver_local_positions(*ego, yaw, rig)])
    traj, valid = _teacher_targets(_poses(t0, world), ego[0], ego[1], yaw, t0)
    assert valid.all()
    np.testing.assert_allclose(traj, rig, atol=1e-4)


def test_current_pose_is_excluded_and_targets_start_at_plus_100ms():
    t0 = 0
    world = [(float(k), 0.0) for k in range(NUM_WAYPOINTS + 1)]
    traj, _ = _teacher_targets(_poses(t0, world), 0.0, 0.0, 0.0, t0)
    assert traj[0, 0] == pytest.approx(1.0)                   # pose at +0.1 s, not the t0 pose
    assert traj[-1, 0] == pytest.approx(float(NUM_WAYPOINTS))  # pose at +4.0 s


def test_missing_and_misaligned_poses_are_masked():
    t0 = 0
    world = [(float(k), 0.0) for k in range(NUM_WAYPOINTS + 1)]
    poses = _poses(t0, world)
    del poses[10]                                              # drop the +1.0 s pose
    poses[5] = (poses[5][0] + 50_000, poses[5][1], poses[5][2])  # shifted by half a step
    poses[6] = (poses[6][0] + 5_000, poses[6][1], poses[6][2])   # within tolerance
    _, valid = _teacher_targets(poses, 0.0, 0.0, 0.0, t0)
    assert not valid[9] and not valid[4]
    assert valid[5] and valid.sum() == NUM_WAYPOINTS - 2


def test_wta_masked_equals_unmasked_when_all_valid():
    torch.manual_seed(0)
    hyps, gt = torch.randn(3, 4, NUM_WAYPOINTS, 2), torch.randn(3, NUM_WAYPOINTS, 2)
    valid = torch.ones(3, NUM_WAYPOINTS, dtype=torch.bool)
    expected = (hyps - gt.unsqueeze(1)).abs().mean(dim=(-2, -1)).min(dim=1).values.mean()
    assert winner_takes_all_loss(hyps, gt, valid).item() == pytest.approx(expected.item(), rel=1e-5)
    assert winner_takes_all_loss(hyps, gt).item() == pytest.approx(expected.item(), rel=1e-5)


def test_wta_ignores_invalid_waypoints_and_only_trains_the_winner():
    gt = torch.zeros(1, NUM_WAYPOINTS, 2)
    hyps = torch.zeros(1, 4, NUM_WAYPOINTS, 2)
    hyps[0, 1] += 1.0                       # worse hypothesis
    hyps[0, 0, 20:] = 99.0                  # large error only where the label is invalid
    valid = torch.ones(1, NUM_WAYPOINTS, dtype=torch.bool)
    valid[0, 20:] = False
    hyps.requires_grad_(True)
    loss = winner_takes_all_loss(hyps, gt, valid)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)   # hypothesis 0 wins on the valid part
    loss.backward()
    assert hyps.grad[0, 1].abs().sum() == 0              # losing hypotheses get no gradient
    assert hyps.grad[0, 0, 20:].abs().sum() == 0         # invalid waypoints get no gradient
