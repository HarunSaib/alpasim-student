# SPDX-License-Identifier: Apache-2.0
"""
Camera calibration utilities for AlpaNet BEV lifting.

Cameras use the ftheta (fisheye) model:
    r_px = poly[0] + poly[1]*θ + poly[2]*θ² + ...
where θ = angle from optical axis, r_px = radial pixel distance from principal point.

All geometry uses the AlpaSim rig frame convention: x=forward, y=left, z=up.

The proto field `rig_to_camera` is misnamed; both its vec and quat represent the
cam→rig transform (verified in alpamayo_base.py):
    - vec = camera position in rig frame (t_cam_in_rig)
    - quat = quaternion rotating camera-frame vectors → rig-frame vectors
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

# Camera IDs used by AlpaNet (matches StudentNet / Alpamayo rig).
CAMERA_IDS = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CACHE_DIR = Path.home() / ".cache" / "alpasim"


def _nre_uuid_from_scene_id(scene_id: str) -> str:
    """Return the NRE UUID for a given scene_id from the sim_scenes.csv manifest."""
    csv = _REPO_ROOT / "data" / "scenes" / "sim_scenes.csv"
    df = pd.read_csv(csv)
    rows = df[df["scene_id"] == scene_id]
    if rows.empty:
        raise KeyError(f"scene_id {scene_id!r} not found in {csv}")
    return str(rows.iloc[0]["uuid"])


@dataclass
class CameraParams:
    logical_id: str
    width: int
    height: int
    # ftheta polynomial coefficients, lowest degree first:
    #   r_px = poly[0] + poly[1]*θ + poly[2]*θ² + ...
    poly: list[float]
    cx: float   # principal point x (pixels, original resolution)
    cy: float   # principal point y (pixels, original resolution)
    # Camera position in rig frame (cam→rig translation)
    t_cam_in_rig: list[float]   # [x, y, z]
    # Quaternion rotating camera-frame vectors → rig-frame vectors [x, y, z, w]
    q_cam_to_rig: list[float]

    def project(self, pts_rig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project (N, 3) rig-frame points to pixel coordinates.

        Returns:
            uv:    (N, 2) pixel coordinates (u=col, v=row)
            valid: (N,)   boolean mask — True where projection is on-image
        """
        t = np.array(self.t_cam_in_rig, dtype=np.float64)
        q = np.array(self.q_cam_to_rig, dtype=np.float64)   # cam→rig

        # Transform rig → camera frame using conjugate quaternion (rig→cam)
        shifted = pts_rig.astype(np.float64) - t               # (N, 3)
        qx, qy, qz, qw = -q[0], -q[1], -q[2], q[3]           # conjugate
        q_vec = np.array([qx, qy, qz])
        tc = 2.0 * np.cross(q_vec, shifted)                    # (N, 3)
        pts_cam = shifted + qw * tc + np.cross(q_vec, tc)      # (N, 3)

        xc, yc, zc = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]

        # Angle from optical axis (z+ in camera frame)
        r_xy = np.hypot(xc, yc)
        theta = np.arctan2(r_xy, zc)

        # Ftheta radial distance in pixels
        r_px = np.polynomial.polynomial.polyval(theta, self.poly)

        # Pixel coords
        phi = np.arctan2(yc, xc)
        u = self.cx + r_px * np.cos(phi)
        v = self.cy + r_px * np.sin(phi)

        valid = (
            (zc > 0.1)
            & (r_px >= 0.0)
            & (u >= 0) & (u < self.width)
            & (v >= 0) & (v < self.height)
        )
        return np.stack([u, v], axis=1), valid


async def _parse_cam_params_from_asl(asl_path: Path) -> dict[str, CameraParams]:
    """Extract CameraParams from the available_cameras_return entry in an .asl file."""
    from alpasim_utils.logs import async_read_pb_log

    params: dict[str, CameraParams] = {}
    async for entry in async_read_pb_log(str(asl_path)):
        kind = entry.WhichOneof("log_entry")
        if kind == "available_cameras_return":
            for cam in entry.available_cameras_return.available_cameras:
                if cam.logical_id not in CAMERA_IDS:
                    continue
                intr = cam.intrinsics
                ft   = intr.ftheta_param
                r2c  = cam.rig_to_camera
                params[cam.logical_id] = CameraParams(
                    logical_id=cam.logical_id,
                    width=intr.resolution_w,
                    height=intr.resolution_h,
                    poly=list(ft.angle_to_pixeldist_poly),
                    cx=float(ft.principal_point_x),
                    cy=float(ft.principal_point_y),
                    t_cam_in_rig=[float(r2c.vec.x), float(r2c.vec.y), float(r2c.vec.z)],
                    q_cam_to_rig=[float(r2c.quat.x), float(r2c.quat.y),
                                  float(r2c.quat.z), float(r2c.quat.w)],
                )
            if params:
                return params
    raise RuntimeError(f"No available_cameras_return found in {asl_path}")


def _find_asl_for_nre_uuid(nre_uuid: str) -> Path:
    """Search common dagger_run directories for any rollout.asl with this nre_uuid."""
    search_roots = sorted(Path.cwd().glob("dagger_run*")) + sorted(
        Path.cwd().glob("alpanet_dagger*")
    )
    for root in search_roots:
        for asl in root.rglob("rollout.asl"):
            # Quick scan: only read the first few entries to check nre_uuid
            async def _check(p=asl):
                from alpasim_utils.logs import async_read_pb_log
                async for entry in async_read_pb_log(str(p)):
                    kind = entry.WhichOneof("log_entry")
                    if kind == "rollout_metadata":
                        uid = entry.rollout_metadata.session_metadata.nre_uuid
                        return uid == nre_uuid
                    # available_cameras_return comes before any driver entries
                    if kind == "available_cameras_return":
                        return False
                return False

            if asyncio.run(_check()):
                return asl
    raise FileNotFoundError(
        f"Could not find a rollout.asl for nre_uuid={nre_uuid!r}. "
        f"Searched {[str(r) for r in search_roots]}. "
        f"Run a DAgger iteration first, or pass an explicit asl_path."
    )


def load_cam_params(
    nre_uuid: str,
    asl_path: Path | None = None,
) -> dict[str, CameraParams]:
    """Load camera calibration for the given NRE UUID.

    Results are cached at ~/.cache/alpasim/cam_{nre_uuid}.json so subsequent
    calls are instant.  Pass asl_path to bypass the search and parse directly.
    """
    cache_path = _CACHE_DIR / f"cam_{nre_uuid}.json"

    if asl_path is None and cache_path.exists():
        raw = json.loads(cache_path.read_text())
        return {k: CameraParams(**v) for k, v in raw.items()}

    if asl_path is None:
        asl_path = _find_asl_for_nre_uuid(nre_uuid)

    params = asyncio.run(_parse_cam_params_from_asl(asl_path))

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({k: asdict(v) for k, v in params.items()}, indent=2))

    return params
