# SPDX-License-Identifier: Apache-2.0
"""
Extract (observation, teacher_trajectory) pairs from AlpaSim .asl rollout logs.

Usage:
    uv run python -m alpasim_student.dagger.collector <run_dir> <out_dir>

Output layout:
    <out_dir>/
      <rollout_uuid>/
        samples.parquet      # one row per sim step that had a driver response
        step_<ts>_imgs.npz   # {camera_id: HWC uint8 ndarray}

Actual .asl entry kinds used:
    driver_camera_image   – camera frame (camera_image.logical_id / image_bytes / frame_start_us)
    driver_request        – ego step marker  (time_now_us)
    driver_return         – teacher trajectory (trajectory.poses[].pose.vec.x/y)
"""
from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from alpasim_utils.logs import async_read_pb_log


def _lookup_speed(speed_buffer: dict[int, float], ts: int) -> float:
    """Return speed at ts by finding the nearest key in speed_buffer."""
    if not speed_buffer:
        return 0.0
    closest = min(speed_buffer.keys(), key=lambda t: abs(t - ts))
    return speed_buffer[closest]


async def _extract_rollout(asl_path: Path, out_dir: Path) -> int:
    """Parse one .asl file and write samples to out_dir.  Returns sample count."""
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []

    # image_buffer[frame_start_us][cam_id] = raw JPEG/PNG bytes
    image_buffer: dict[int, dict[str, bytes]] = {}

    # Images selected for the current driver step
    pending_images: dict[str, np.ndarray] = {}
    current_ts: int = 0

    # speed_buffer[timestamp_us] = speed_m_s from ego dynamic states
    speed_buffer: dict[int, float] = {}

    async for entry in async_read_pb_log(str(asl_path)):
        kind = entry.WhichOneof("log_entry")

        # ------------------------------------------------------------------ #
        # Accumulate rendered camera images keyed by their frame timestamp    #
        # ------------------------------------------------------------------ #
        if kind == "driver_camera_image":
            ci = entry.driver_camera_image.camera_image
            ts = int(ci.frame_start_us)
            cam_id = ci.logical_id
            if ts not in image_buffer:
                image_buffer[ts] = {}
            image_buffer[ts][cam_id] = bytes(ci.image_bytes)

        # ------------------------------------------------------------------ #
        # Ego dynamic states: extract speed at each pose timestamp            #
        # ------------------------------------------------------------------ #
        elif kind == "driver_ego_trajectory":
            ego_traj = entry.driver_ego_trajectory
            for pose, dyn in zip(ego_traj.trajectory.poses, ego_traj.dynamic_states):
                ts = int(pose.timestamp_us)
                vx = dyn.linear_velocity.x
                vy = dyn.linear_velocity.y
                speed_buffer[ts] = float((vx ** 2 + vy ** 2) ** 0.5)

        # ------------------------------------------------------------------ #
        # driver_request marks the start of a new decision step               #
        # Pick the camera frame whose timestamp is closest (≤) time_now_us    #
        # ------------------------------------------------------------------ #
        elif kind == "driver_request":
            req = entry.driver_request
            current_ts = int(req.time_now_us)

            # Find the most recent camera batch at or before current_ts
            valid_ts = [t for t in image_buffer if t <= current_ts]
            if valid_ts:
                best_ts = max(valid_ts)
                pending_images = {}
                for cam_id, raw in image_buffer[best_ts].items():
                    try:
                        img = np.array(Image.open(io.BytesIO(raw)))
                        pending_images[cam_id] = img
                    except Exception:
                        pass
            else:
                pending_images = {}

        # ------------------------------------------------------------------ #
        # driver_return contains the teacher's planned trajectory             #
        # ------------------------------------------------------------------ #
        elif kind == "driver_return":
            ret = entry.driver_return
            poses = list(ret.trajectory.poses)
            if not poses or not pending_images:
                continue

            # Extract (x, y) world coordinates then convert to ego-relative
            traj_world = [[float(p.pose.vec.x), float(p.pose.vec.y)] for p in poses]
            x0, y0 = traj_world[0]
            traj_xy = [[x - x0, y - y0] for x, y in traj_world]

            img_file = out_dir / f"step_{current_ts:015d}_imgs.npz"
            np.savez_compressed(str(img_file), **pending_images)

            records.append({
                "timestamp_us": current_ts,
                "traj_xy":      traj_xy,
                "speed":        _lookup_speed(speed_buffer, current_ts),
                "img_file":     str(img_file),
            })

            pending_images = {}   # consume; next step needs new images

    if records:
        pd.DataFrame(records).to_parquet(out_dir / "samples.parquet", index=False)

    return len(records)


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
        rollout_out = out_dir / rollout_id
        n = asyncio.run(_extract_rollout(asl_file, rollout_out))
        print(f"  {rollout_id}: {n} samples")
        total += n

    print(f"[collector] Total: {total} samples → {out_dir}")
    return total


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m alpasim_student.dagger.collector <run_dir> <out_dir>")
        sys.exit(1)
    collect_from_run(Path(sys.argv[1]), Path(sys.argv[2]))
