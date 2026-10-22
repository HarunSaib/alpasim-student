# SPDX-License-Identifier: Apache-2.0
"""
Extract (observation, teacher trajectory) pairs from AlpaSim .asl rollout logs.

Usage:
    uv run python -m alpasim_student.dagger.collector <run_dir> <out_dir>

Output layout:
    <out_dir>/
      <rollout_uuid>/
        samples.parquet      # one row per driver decision step
        step_<ts>_imgs.npz   # {camera_id: HWC uint8 ndarray}

Label convention (label_version 3), matching how the driver turns a model's output into poses:
    scene_id, rollout_id   which scene and rollout the step came from (the train/val split key)
    traj_xy   (40, 2)  teacher poses at +0.1 ... +4.0 s after the request, expressed in the ego
                       frame at the request time (x forward, y left)
    valid     (40,)    False where the teacher pose for that time step is missing
    speed     m/s      from the latest ego dynamic state received before the request
    accel     m/s^2    linear_acceleration.x of that same state (what the driver feeds the model)
    yaw_rad, ego_x, ego_y   ego pose at the request (scene frame), kept for debugging

.asl entry kinds used:
    driver_camera_image   – camera frame (camera_image.logical_id / image_bytes / frame_start_us)
    driver_ego_trajectory – ego poses + dynamic states submitted to the driver
    driver_request        – decision step marker (time_now_us)
    driver_return         – teacher trajectory (current pose followed by future poses)
"""
from __future__ import annotations

import asyncio
import io
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from alpasim_utils.logs import async_read_pb_log

from ..alpanet.alpanet_model import NUM_WAYPOINTS
from ..alpanet.camera_utils import CAMERA_IDS

LABEL_VERSION = 3
_STEP_US = 100_000       # 10 Hz
_TIME_TOL_US = 10_000    # tolerance when matching a pose to its target time
_REF_TOL_M = 0.05        # driver_return pose 0 must coincide with the ego pose at the request


def _quat_to_yaw(q) -> float:
    """Yaw about +Z; same formula as the driver service."""
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def _teacher_targets(
    poses,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    t_req_us: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Teacher poses at +0.1 ... +4.0 s in the ego frame, plus a validity mask.

    poses: iterable of (timestamp_us, x, y) in the scene frame.
    """
    c, s = math.cos(ego_yaw), math.sin(ego_yaw)
    traj = np.zeros((NUM_WAYPOINTS, 2), dtype=np.float32)
    valid = np.zeros(NUM_WAYPOINTS, dtype=bool)
    for ts, x, y in poses:
        k = round((ts - t_req_us) / _STEP_US)
        if not 1 <= k <= NUM_WAYPOINTS:
            continue
        if abs(ts - t_req_us - k * _STEP_US) > _TIME_TOL_US:
            continue
        dx, dy = x - ego_x, y - ego_y
        traj[k - 1] = (c * dx + s * dy, -s * dx + c * dy)
        valid[k - 1] = True
    return traj, valid


async def _extract_rollout(asl_path: Path, out_dir: Path) -> tuple[int, dict[str, int]]:
    """Parse one .asl file and write samples to out_dir. Returns (sample count, skip counts)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_id, rollout_id = asl_path.parent.parent.name, asl_path.parent.name

    records: list[dict] = []
    skips = {"no_images": 0, "missing_camera": 0, "ref_mismatch": 0, "no_targets": 0}

    # image_buffer[frame_start_us][cam_id] = raw JPEG/PNG bytes
    image_buffer: dict[int, dict[str, bytes]] = {}

    ego: dict | None = None          # latest ego pose + dynamic state received
    request: dict | None = None      # snapshot taken at driver_request
    pending_images: dict[str, np.ndarray] = {}

    async for entry in async_read_pb_log(str(asl_path)):
        kind = entry.WhichOneof("log_entry")

        if kind == "driver_camera_image":
            ci = entry.driver_camera_image.camera_image
            image_buffer.setdefault(int(ci.frame_start_us), {})[ci.logical_id] = bytes(ci.image_bytes)

        elif kind == "driver_ego_trajectory":
            et = entry.driver_ego_trajectory
            poses, states = list(et.trajectory.poses), list(et.dynamic_states)
            if poses and states:
                last, dyn = poses[-1], states[-1]
                ego = {
                    "x": float(last.pose.vec.x),
                    "y": float(last.pose.vec.y),
                    "yaw": _quat_to_yaw(last.pose.quat),
                    "speed": float(math.hypot(dyn.linear_velocity.x, dyn.linear_velocity.y)),
                    "accel": float(dyn.linear_acceleration.x),
                }

        elif kind == "driver_request":
            t_req = int(entry.driver_request.time_now_us)
            valid_ts = [t for t in image_buffer if t <= t_req]
            pending_images = {}
            if valid_ts:
                for cam_id, raw in image_buffer[max(valid_ts)].items():
                    try:
                        pending_images[cam_id] = np.array(Image.open(io.BytesIO(raw)))
                    except Exception:
                        pass
            request = {"t": t_req, "ego": dict(ego) if ego else None}

        elif kind == "driver_return":
            poses = list(entry.driver_return.trajectory.poses)
            if not poses or request is None or request["ego"] is None:
                continue
            if not pending_images:
                skips["no_images"] += 1
                continue
            if any(cam not in pending_images for cam in CAMERA_IDS):
                skips["missing_camera"] += 1
                continue

            e, t_req = request["ego"], request["t"]
            if math.hypot(poses[0].pose.vec.x - e["x"], poses[0].pose.vec.y - e["y"]) > _REF_TOL_M:
                skips["ref_mismatch"] += 1
                continue

            traj, valid = _teacher_targets(
                [(int(p.timestamp_us), float(p.pose.vec.x), float(p.pose.vec.y)) for p in poses],
                e["x"], e["y"], e["yaw"], t_req,
            )
            if not valid.any():
                skips["no_targets"] += 1
                continue

            img_file = out_dir / f"step_{t_req:015d}_imgs.npz"
            np.savez_compressed(
                str(img_file), **{cam: pending_images[cam] for cam in CAMERA_IDS}
            )
            records.append({
                "scene_id":      scene_id,
                "rollout_id":    rollout_id,
                "timestamp_us":  t_req,
                "traj_xy":       traj.tolist(),
                "valid":         valid.tolist(),
                "speed":         e["speed"],
                "accel":         e["accel"],
                "yaw_rad":       e["yaw"],
                "ego_x":         e["x"],
                "ego_y":         e["y"],
                "img_file":      str(img_file),
                "label_version": LABEL_VERSION,
            })
            pending_images = {}   # consume; next step needs new images

    if records:
        pd.DataFrame(records).to_parquet(out_dir / "samples.parquet", index=False)

    return len(records), skips


def collect_from_run(run_dir: Path, out_dir: Path) -> int:
    """Process every rollout .asl file found under run_dir."""
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    total = 0
    asl_files = sorted(run_dir.glob("rollouts/*/*/rollout.asl"))
    if not asl_files:
        print(f"[collector] No rollout.asl files found under {run_dir}")
        return 0

    for asl_file in asl_files:
        rollout_id = asl_file.parent.name
        n, skips = asyncio.run(_extract_rollout(asl_file, out_dir / rollout_id))
        skipped = {k: v for k, v in skips.items() if v}
        print(f"  {rollout_id}: {n} samples" + (f"  skipped {skipped}" if skipped else ""))
        total += n

    print(f"[collector] Total: {total} samples → {out_dir}")
    return total


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m alpasim_student.dagger.collector <run_dir> <out_dir>")
        sys.exit(1)
    collect_from_run(Path(sys.argv[1]), Path(sys.argv[2]))
