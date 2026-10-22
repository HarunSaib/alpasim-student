#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
BEV geometry validation visualizer.

Projects AlpaNet's BEV grid reference points back into each camera image
and renders them as colour-coded dots in Rerun.

PASS criteria:
  - Projected dots form a regular forward-facing grid on the road surface
  - Dots at Z=0 land on asphalt, dots at Z=2 land on building tops / trees
  - No dots floating in sky or sinking below ground

FAIL indicators:
  - Dots in wrong part of image → extrinsic convention error (cam→rig vs rig→cam)
  - Dots near infinity / all invalid → intrinsic scale wrong (focal_length units)
  - Dots on correct object but shifted → principal-point offset

Usage:
    uv run --extra all --no-sync python scripts/validate_bev_geometry.py <rollout_dir>
    uv run --extra all --no-sync python scripts/validate_bev_geometry.py <rollout_dir> --uuid <nre_uuid>
"""
from __future__ import annotations

import argparse
import asyncio
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import rerun as rr
    import rerun.blueprint as rrb
    _HAS_RERUN = True
except ImportError:
    _HAS_RERUN = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alpasim_utils.logs import async_read_pb_log

# Must import from within the alpasim venv where alpasim_student is installed
from alpasim_student.alpanet.camera_utils import (
    CAMERA_IDS,
    load_cam_params,
    _nre_uuid_from_scene_id,
)
from alpasim_student.alpanet.bev_lifter import (
    BEV_H, BEV_W, BEV_X_MIN, BEV_X_MAX, BEV_Y_MIN, BEV_Y_MAX, BEV_RES, SAMPLE_HEIGHTS,
)

CAMERA_DISPLAY = {
    "camera_cross_left_120fov":  "camera/left",
    "camera_front_wide_120fov":  "camera/front_wide",
    "camera_cross_right_120fov": "camera/right",
    "camera_front_tele_30fov":   "camera/front_tele",
}

# Sample every N-th BEV cell to avoid clutter
SUBSAMPLE = 8   # every 8th cell = 4m spacing

# Height colour map: Z=-0.5=blue, Z=0=green, Z=0.5=yellow, Z=1=orange, Z=2=red
HEIGHT_COLORS = {
    -0.5: [100, 100, 255],
    0.0:  [0,   200,  50],
    0.5:  [200, 200,  0 ],
    1.0:  [255, 140,  0 ],
    2.0:  [255,  50,  50],
}


async def _load_first_frame(asl_path: Path) -> dict[str, np.ndarray]:
    """Read the first complete camera frame set from a rollout ASL."""
    buf: dict[int, dict[str, bytes]] = {}
    best: dict[str, np.ndarray] = {}
    async for entry in async_read_pb_log(str(asl_path)):
        kind = entry.WhichOneof("log_entry")
        if kind == "driver_camera_image":
            ci = entry.driver_camera_image.camera_image
            ts = int(ci.frame_start_us)
            if ts not in buf:
                buf[ts] = {}
            buf[ts][ci.logical_id] = bytes(ci.image_bytes)
            # Once we have all 4 cameras at one timestamp, stop
            if all(c in buf[ts] for c in CAMERA_IDS):
                for cam_id, raw in buf[ts].items():
                    if cam_id in CAMERA_IDS:
                        best[cam_id] = np.array(Image.open(io.BytesIO(raw)))
                break
    return best


def _build_bev_pts() -> np.ndarray:
    """Build the subsampled BEV grid points in rig frame."""
    xs = np.linspace(BEV_X_MIN + BEV_RES / 2, BEV_X_MAX - BEV_RES / 2, BEV_H)
    ys = np.linspace(BEV_Y_MIN + BEV_RES / 2, BEV_Y_MAX - BEV_RES / 2, BEV_W)
    grid_x, grid_y = np.meshgrid(xs[::SUBSAMPLE], ys[::SUBSAMPLE], indexing="ij")
    return np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)   # (N, 2) — x, y only


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout_dir", help="Path to a rollout directory containing rollout.asl")
    parser.add_argument("--uuid",      default=None, help="NRE UUID (auto-detected from rollout if omitted)")
    parser.add_argument("--save",      default=None, help="Save .rrd instead of spawning viewer")
    parser.add_argument("--headless",  action="store_true",
                        help="Print projection stats only, no Rerun viewer (use when rerun is not installed)")
    args = parser.parse_args()

    if not args.headless and not _HAS_RERUN:
        print("WARNING: rerun not installed — falling back to --headless mode. "
              "Install with: pip install rerun-sdk")
        args.headless = True

    rollout_dir = Path(args.rollout_dir)
    asl_path    = rollout_dir / "rollout.asl"
    if not asl_path.exists():
        print(f"ERROR: {asl_path} not found")
        sys.exit(1)

    # ── Auto-detect NRE UUID from rollout metadata ────────────────────────────
    nre_uuid = args.uuid
    if not nre_uuid:
        async def _get_uuid():
            async for entry in async_read_pb_log(str(asl_path)):
                if entry.WhichOneof("log_entry") == "rollout_metadata":
                    meta = entry.rollout_metadata
                    sid  = meta.session_metadata.nre_uuid if hasattr(meta, "session_metadata") else ""
                    return sid or ""
                break
            return ""
        nre_uuid = asyncio.run(_get_uuid())
        if nre_uuid:
            print(f"[validate] nre_uuid = {nre_uuid}")
        else:
            print("WARNING: could not detect nre_uuid — use --uuid to specify")

    # ── Load camera params ────────────────────────────────────────────────────
    if not nre_uuid:
        print("ERROR: nre_uuid required")
        sys.exit(1)

    cam_params = load_cam_params(nre_uuid)
    print(f"[validate] Loaded calibration for {len(cam_params)} cameras")
    for cid, cp in cam_params.items():
        print(f"  {cid}: {cp.width}×{cp.height}  cx={cp.cx:.1f} cy={cp.cy:.1f}  poly[1]={cp.poly[1]:.3f}")
        print(f"    pos_in_rig={cp.t_cam_in_rig}")

    # ── BEV grid points ───────────────────────────────────────────────────────
    bev_xy = _build_bev_pts()   # (N, 2) — rig frame XY

    # ── Load first camera frame (only needed for Rerun visualisation) ─────────
    frames: dict = {}
    if not args.headless:
        print("[validate] Loading first camera frame ...")
        frames = asyncio.run(_load_first_frame(asl_path))
        print(f"[validate] Got frames: {list(frames.keys())}")

    # ── Rerun setup ───────────────────────────────────────────────────────────
    if not args.headless:
        rr.init("bev_geometry_validator", spawn=(args.save is None))
        if args.save:
            rr.save(args.save)
        panels = [
            rrb.Spatial2DView(name=display, origin=f"camera/{cam_id.split('_')[1]}")
            for cam_id, display in CAMERA_DISPLAY.items()
            if cam_id in frames
        ]
        rr.send_blueprint(rrb.Blueprint(rrb.Grid(*panels)))
        rr.set_time("frame", sequence=0)

    # ── Per-camera projection ─────────────────────────────────────────────────
    all_pass = True
    for cam_id, display_name in CAMERA_DISPLAY.items():
        if cam_id not in cam_params:
            continue

        cp = cam_params[cam_id]

        n_total = len(bev_xy) * len(SAMPLE_HEIGHTS)
        n_projected = 0

        for h in SAMPLE_HEIGHTS:
            z_col   = np.full(len(bev_xy), h)
            pts_rig = np.column_stack([bev_xy[:, 0], bev_xy[:, 1], z_col])
            uv, valid = cp.project(pts_rig)
            n_projected += int(valid.sum())

            if not args.headless and cam_id in frames:
                panel_path = f"camera/{cam_id.split('_')[1]}"
                if h == min(SAMPLE_HEIGHTS):
                    rr.log(panel_path, rr.Image(frames[cam_id]))
                if valid.sum() > 0:
                    rr.log(
                        f"{panel_path}/bev_grid/h{h:+.1f}m",
                        rr.Points2D(
                            positions=uv[valid],
                            radii=3.0,
                            colors=[HEIGHT_COLORS[h]] * int(valid.sum()),
                            labels=[f"z={h:+.1f}m"] * int(valid.sum()),
                        ),
                    )

        pct = 100 * n_projected / max(n_total, 1)
        status = "PASS" if pct >= 20.0 else "FAIL"
        if pct < 20.0:
            all_pass = False
        print(f"  [{status}] {cam_id}: {n_projected}/{n_total} ({pct:.1f}% valid)")

    print()
    if all_pass:
        print("OVERALL: PASS — ≥20% of BEV points project into each camera.")
    else:
        print("OVERALL: FAIL — some cameras have <20% projection coverage.")
        print("  Check: extrinsic convention (cam→rig vs rig→cam), focal-length units, principal-point offset.")
    if not args.headless:
        print("Rerun: coloured dots should form a regular grid on road surface in all cameras.")


if __name__ == "__main__":
    main()
