#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Visualize a DAgger rollout with Rerun.

Shows:
  - 3D scene: full ground mesh + car body + 4 camera frustums (FOV cones)
  - Bird's eye (2D) view
  - Camera feeds synced to sim time
  - Metrics timeline

The ground mesh is loaded from the USDZ file matched to this rollout's NRE uuid.

Usage:
    uv run --extra all --no-sync python scripts/visualize_rollout.py <rollout_dir>
    uv run --extra all --no-sync python scripts/visualize_rollout.py <rollout_dir> --save out.rrd
    ... --save out.rrd --cam-subsample 2 --cam-resize 0.5
"""
from __future__ import annotations

import argparse
import asyncio
import io
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pickle

import numpy as np
import pandas as pd
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image

from alpasim_utils.logs import async_read_pb_log


METRICS_TO_PLOT = [
    "offroad",
    "collision_any",
    "progress",
    "dist_to_gt_trajectory",
    "plan_deviation",
]

CAMERA_DISPLAY_NAMES = {
    "camera_front_wide_120fov": "camera/front_wide",
    "camera_front_tele_30fov":  "camera/front_tele",
    "camera_cross_left_120fov": "camera/left",
    "camera_cross_right_120fov":"camera/right",
}

# Car bounding box half-extents in metres (length, width, height) in rig/car frame
CAR_HALF_EXTENTS = [2.3, 1.0, 0.75]

# Where the USDZ scene files live
USDZ_DIR = Path(__file__).parent.parent / "data" / "nre-artifacts" / "all-usdzs"


# ── Camera helpers ─────────────────────────────────────────────────────────────

@dataclass
class CameraInfo:
    logical_id: str
    width: int
    height: int
    focal_length: float
    principal_point: tuple[float, float]
    # Quaternion + translation that take camera-frame vectors into rig frame.
    # (proto field is named rig_to_camera but the quaternion is actually cam→rig)
    cam_to_rig_t: np.ndarray   # (3,) camera position in rig frame
    cam_to_rig_q: np.ndarray   # (4,) xyzw  cam→rig rotation


def _quat_rotate(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q_xyzw
    t = 2.0 * np.cross([qx, qy, qz], v)
    return np.array(v) + qw * t + np.cross([qx, qy, qz], t)


def _parse_cameras(cam_entries) -> dict[str, CameraInfo]:
    cameras: dict[str, CameraInfo] = {}
    for cam in cam_entries:
        intr   = cam.intrinsics
        ftheta = intr.ftheta_param
        focal  = (ftheta.angle_to_pixeldist_poly[1]
                  if len(ftheta.angle_to_pixeldist_poly) > 1 else 1000.0)
        r2c    = cam.rig_to_camera
        # The proto quaternion rotates camera +Z → rig +X (verified empirically),
        # so it IS the cam→rig rotation despite the field name.
        cameras[cam.logical_id] = CameraInfo(
            logical_id=cam.logical_id,
            width=intr.resolution_w,
            height=intr.resolution_h,
            focal_length=float(focal),
            principal_point=(float(ftheta.principal_point_x),
                             float(ftheta.principal_point_y)),
            cam_to_rig_t=np.array([r2c.vec.x, r2c.vec.y, r2c.vec.z]),
            cam_to_rig_q=np.array([r2c.quat.x, r2c.quat.y, r2c.quat.z, r2c.quat.w]),
        )
    return cameras


# ── Mesh loading ───────────────────────────────────────────────────────────────

def _load_scene_mesh(
    nre_uuid: str,
    origin: np.ndarray,
    clip_radius_xy: float = 160.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
    """Load the full scene mesh from the matching USDZ.

    Returns (vertices, faces, colors) centered on `origin`, clipped to
    `clip_radius_xy` metres in XY around the car path, with elevation-based colors.
    Returns (None, None, None) if USDZ not found.
    """
    import matplotlib.cm as cm

    usdz_path = USDZ_DIR / f"{nre_uuid}.usdz"
    if not usdz_path.exists():
        print(f"[viz] USDZ not found: {usdz_path}")
        return None, None, None

    with zipfile.ZipFile(usdz_path) as z:
        ply_data = z.read("mesh.ply")

    # Parse PLY header
    lines, i = [], 0
    while True:
        end = ply_data.index(b'\n', i)
        lines.append(ply_data[i:end].decode().strip())
        i = end + 1
        if lines[-1] == 'end_header':
            break

    n_verts = int(next(l for l in lines if l.startswith('element vertex')).split()[-1])
    n_faces = int(next(l for l in lines if l.startswith('element face')).split()[-1])
    print(f"[viz] Full scene mesh: {n_verts} verts, {n_faces} faces — clipping …")

    verts = np.frombuffer(ply_data, dtype=np.float32, count=n_verts * 3,
                          offset=i).reshape(-1, 3).copy()
    face_offset = i + n_verts * 12
    faces_raw = np.frombuffer(
        ply_data, dtype=np.dtype([('n', 'u1'), ('idx', '<i4', 3)]),
        count=n_faces, offset=face_offset,
    )['idx'].copy()

    # Centre on car start position
    verts -= origin

    # Clip to a box around the path so we don't ship 6M verts to Rerun
    mask = (np.abs(verts[:, 0]) < clip_radius_xy) & (np.abs(verts[:, 1]) < clip_radius_xy)
    kept_idx = np.where(mask)[0]
    remap = -np.ones(n_verts, dtype=np.int32)
    remap[kept_idx] = np.arange(len(kept_idx), dtype=np.int32)

    face_remap = remap[faces_raw]                           # (F, 3) with -1 for dropped verts
    valid_faces = np.all(face_remap >= 0, axis=1)
    faces_clip  = face_remap[valid_faces].astype(np.uint32)
    verts_clip  = verts[kept_idx]

    # Compute per-vertex normals from face normals (vectorised)
    v0 = verts_clip[faces_clip[:, 0]]
    v1 = verts_clip[faces_clip[:, 1]]
    v2 = verts_clip[faces_clip[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    fn_len = np.linalg.norm(fn, axis=1, keepdims=True)
    fn /= np.where(fn_len > 0, fn_len, 1)

    vnorm = np.zeros_like(verts_clip)
    np.add.at(vnorm, faces_clip[:, 0], fn)
    np.add.at(vnorm, faces_clip[:, 1], fn)
    np.add.at(vnorm, faces_clip[:, 2], fn)
    vn_len = np.linalg.norm(vnorm, axis=1, keepdims=True)
    vnorm /= np.where(vn_len > 0, vn_len, 1)

    # Elevation-based colour: road=grey, curbs/low=tan, buildings=brown
    z_vals = verts_clip[:, 2]
    z_norm = np.clip((z_vals - z_vals.min()) / max(z_vals.max() - z_vals.min(), 0.1), 0, 1)
    cmap   = cm.get_cmap("gist_earth")
    colors = (cmap(z_norm)[:, :3] * 255).astype(np.uint8)

    print(f"[viz] Clipped mesh: {len(verts_clip)} verts, {len(faces_clip)} faces")
    return verts_clip, faces_clip, colors, vnorm


# ── ASL parsing ───────────────────────────────────────────────────────────────

async def _parse_asl(asl_path: Path):
    ego_poses: list[tuple]  = []
    cam_frames: dict[str, list[tuple[int, np.ndarray]]] = {}
    plans: list[tuple[int, list[tuple[float, float, float]]]] = []
    camera_info: dict[str, CameraInfo] = {}
    nre_uuid: str = ""
    image_buffer: dict[int, dict[str, bytes]] = {}
    current_ts: int = 0

    # actor_defs[actor_id] = (label, half_extents_xyz)
    actor_defs: dict[str, tuple[str, np.ndarray]] = {}
    # actor_poses_by_ts[ts] = [(actor_id, x, y, z, qx, qy, qz, qw), ...]
    actor_poses_by_ts: dict[int, list[tuple]] = {}
    # reasoning_by_ts[ts] = text string from Alpamayo CoT
    reasoning_by_ts: dict[int, str] = {}

    async for entry in async_read_pb_log(str(asl_path)):
        kind = entry.WhichOneof("log_entry")

        if kind == "rollout_metadata":
            nre_uuid = entry.rollout_metadata.session_metadata.nre_uuid
            for aabb_def in entry.rollout_metadata.actor_definitions.actor_aabb:
                a = aabb_def.aabb
                # AABB half-extents: use the bounding box dimensions
                half = np.array([
                    float(a.size_x) / 2.0,
                    float(a.size_y) / 2.0,
                    float(a.size_z) / 2.0,
                ], dtype=np.float32)
                actor_defs[aabb_def.actor_id] = (aabb_def.actor_label, half)

        elif kind == "actor_poses":
            ap = entry.actor_poses
            ts = int(ap.timestamp_us)
            actor_poses_by_ts[ts] = [
                (
                    p.actor_id,
                    float(p.actor_pose.vec.x), float(p.actor_pose.vec.y), float(p.actor_pose.vec.z),
                    float(p.actor_pose.quat.x), float(p.actor_pose.quat.y),
                    float(p.actor_pose.quat.z), float(p.actor_pose.quat.w),
                )
                for p in ap.actor_poses
            ]

        elif kind == "available_cameras_return" and not camera_info:
            camera_info = _parse_cameras(
                entry.available_cameras_return.available_cameras
            )

        elif kind == "driver_ego_trajectory":
            for pose in entry.driver_ego_trajectory.trajectory.poses:
                v, q = pose.pose.vec, pose.pose.quat
                ego_poses.append((
                    int(pose.timestamp_us),
                    float(v.x), float(v.y), float(v.z),
                    float(q.x), float(q.y), float(q.z), float(q.w),
                ))

        elif kind == "driver_camera_image":
            ci = entry.driver_camera_image.camera_image
            image_buffer.setdefault(int(ci.frame_start_us), {})[ci.logical_id] = \
                bytes(ci.image_bytes)

        elif kind == "driver_request":
            current_ts = int(entry.driver_request.time_now_us)
            valid = [t for t in image_buffer if t <= current_ts]
            if valid:
                for cam, raw in image_buffer[max(valid)].items():
                    try:
                        img = np.array(Image.open(io.BytesIO(raw)))
                        cam_frames.setdefault(cam, []).append((current_ts, img))
                    except Exception:
                        pass

        elif kind == "driver_return":
            dr = entry.driver_return
            poses = list(dr.trajectory.poses)
            if poses and current_ts:
                plans.append((current_ts, [
                    (float(p.pose.vec.x), float(p.pose.vec.y), float(p.pose.vec.z))
                    for p in poses
                ]))
            # Extract Alpamayo chain-of-thought reasoning from pickled debug blob.
            if dr.HasField("debug_info") and dr.debug_info.unstructured_debug_info:
                try:
                    debug = pickle.loads(dr.debug_info.unstructured_debug_info)
                    text = debug.get("reasoning_text")
                    if text and current_ts:
                        reasoning_by_ts[current_ts] = str(text)
                except Exception:
                    pass

    return ego_poses, cam_frames, plans, camera_info, nre_uuid, actor_defs, actor_poses_by_ts, reasoning_by_ts


# ── Blueprint ─────────────────────────────────────────────────────────────────

def _build_blueprint(cam_ids: list[str], has_reasoning: bool) -> rrb.Blueprint:
    cam_views = [
        rrb.Spatial2DView(
            name=CAMERA_DISPLAY_NAMES.get(c, c),
            origin=CAMERA_DISPLAY_NAMES.get(c, f"camera/{c}"),
        )
        for c in cam_ids
    ]
    metric_views = [
        rrb.TimeSeriesView(name=m, origin=f"metrics/{m}")
        for m in METRICS_TO_PLOT
    ]
    right_col = list(metric_views)
    if has_reasoning:
        right_col.append(rrb.TextLogView(name="Teacher Reasoning", origin="reasoning"))
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial3DView(name="3D Scene", origin="world3d"),
                rrb.Spatial2DView(name="Bird's Eye", origin="world2d"),
            ),
            rrb.Vertical(*cam_views),
            rrb.Vertical(*right_col),
            column_shares=[3, 2, 1],
        ),
        collapse_panels=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def visualize(rollout_dir: Path, save_path: Path | None = None,
              cam_subsample: int = 1, cam_resize: float = 1.0,
              show_hypotheses: bool = False) -> None:
    rollout_dir  = Path(rollout_dir).resolve()
    asl_path     = rollout_dir / "rollout.asl"
    metrics_path = rollout_dir / "metrics.parquet"

    if not asl_path.exists():
        print(f"[error] rollout.asl not found in {rollout_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[viz] Parsing {asl_path.name} ...")
    ego_poses, cam_frames, plans, camera_info, nre_uuid, actor_defs, actor_poses_by_ts, reasoning_by_ts = \
        asyncio.run(_parse_asl(asl_path))
    print(f"[viz] {len(ego_poses)} ego poses, {len(plans)} plans, "
          f"{sum(len(v) for v in cam_frames.values())} camera frames  nre={nre_uuid}")
    print(f"[viz] {len(actor_defs)} actor definitions, "
          f"{len(actor_poses_by_ts)} actor pose snapshots, "
          f"{len(reasoning_by_ts)} reasoning entries")

    metrics  = _load_metrics(metrics_path) if metrics_path.exists() else {}

    # AlpaNet hypotheses: optional per-timestep npz {ts_us: (M, T, 2) array}
    hyp_path   = rollout_dir / "hypotheses.npz"
    hyp_by_ts: dict[int, np.ndarray] = {}
    if show_hypotheses and hyp_path.exists():
        npz = np.load(str(hyp_path))
        for key in npz.files:
            hyp_by_ts[int(key)] = npz[key]
        print(f"[viz] Loaded {len(hyp_by_ts)} hypothesis frames from {hyp_path.name}")

    rr.init("alpasim_rollout_viewer", spawn=(save_path is None))
    if save_path:
        rr.save(str(save_path))

    rr.send_blueprint(_build_blueprint(list(cam_frames.keys()), bool(reasoning_by_ts)))

    # AlpaSim world frame: x=forward, y=left, z=up
    rr.log("world3d", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Determine scene origin from first ego pose (used to centre both mesh and car)
    ego_poses.sort(key=lambda t: t[0])
    origin = np.array([ego_poses[0][1], ego_poses[0][2], ego_poses[0][3]])

    # ── Scene mesh ────────────────────────────────────────────────────────────
    mesh_result = _load_scene_mesh(nre_uuid, origin, clip_radius_xy=160.0)
    if mesh_result[0] is not None:
        mesh_verts, mesh_faces, mesh_colors, mesh_normals = mesh_result
        rr.log(
            "world3d/scene",
            rr.Mesh3D(
                vertex_positions=mesh_verts,
                triangle_indices=mesh_faces,
                vertex_normals=mesh_normals,
                vertex_colors=mesh_colors,
            ),
            static=True,
        )

    # ── Static camera frustums (no images — just FOV cones) ───────────────────
    for cam_id, info in camera_info.items():
        cam_path = f"world3d/car/{CAMERA_DISPLAY_NAMES.get(cam_id, cam_id)}"
        scale = cam_resize
        # Camera convention: x=right, y=down, z=forward (OpenCV / RDF)
        rr.log(cam_path, rr.ViewCoordinates.RDF, static=True)
        rr.log(cam_path, rr.Pinhole(
            focal_length=(info.focal_length * scale, info.focal_length * scale),
            principal_point=(info.principal_point[0] * scale,
                             info.principal_point[1] * scale),
            width=int(info.width * scale),
            height=int(info.height * scale),
        ), static=True)
        # Transform3D: place camera in car/rig frame.
        # cam_to_rig_q is verified to be the cam→rig rotation; cam_to_rig_t is
        # the camera position in rig frame.
        rr.log(cam_path, rr.Transform3D(
            translation=info.cam_to_rig_t.tolist(),
            rotation=rr.Quaternion(xyzw=info.cam_to_rig_q.tolist()),
        ), static=True)

    # Static metrics series labels
    for m in METRICS_TO_PLOT:
        rr.log(f"metrics/{m}", rr.SeriesLines(names=m), static=True)

    # ── Metrics timeline ──────────────────────────────────────────────────────
    for metric_name, df in metrics.items():
        for _, row in df.iterrows():
            rr.set_time("sim_time_us", sequence=int(row["timestamps_us"]))
            rr.log(f"metrics/{metric_name}", rr.Scalars(float(row["values"])))

    # ── Ego path and car pose ─────────────────────────────────────────────────
    ego_ts  = np.array([e[0] for e in ego_poses])
    ego_xyz = np.array([[e[1], e[2], e[3]] for e in ego_poses]) - origin
    ego_q   = np.array([[e[4], e[5], e[6], e[7]] for e in ego_poses])

    plan_by_ts = {ts: xyz for ts, xyz in plans}
    all_ts     = sorted(set(ego_ts.tolist()) | set(plan_by_ts.keys()))

    accumulated_3d: list[list[float]] = []
    accumulated_2d: list[list[float]] = []
    prev_ts = None

    for ts in all_ts:
        rr.set_time("sim_time_us", sequence=int(ts))

        idx = int(np.searchsorted(ego_ts, ts, side="right")) - 1
        if idx >= 0 and ego_ts[idx] != prev_ts:
            prev_ts = ego_ts[idx]
            pos = ego_xyz[idx]
            q   = ego_q[idx]

            accumulated_3d.append(pos.tolist())
            accumulated_2d.append([pos[0], pos[1]])

            # Driven path lines
            if len(accumulated_3d) >= 2:
                rr.log("world3d/ego_path",
                       rr.LineStrips3D([accumulated_3d], colors=[[50, 220, 50]], radii=0.12))
                rr.log("world2d/ego_path",
                       rr.LineStrips2D([accumulated_2d], colors=[[50, 220, 50]], radii=0.12))
            else:
                rr.log("world3d/ego_path",
                       rr.Points3D([pos.tolist()], colors=[[50, 220, 50]], radii=0.3))
                rr.log("world2d/ego_path",
                       rr.Points2D([[pos[0], pos[1]]], colors=[[50, 220, 50]], radii=0.3))

            # Car body: Transform3D places the car in world, box is in local frame
            rr.log("world3d/car",
                   rr.Transform3D(translation=pos.tolist(),
                                  rotation=rr.Quaternion(xyzw=q.tolist())))
            rr.log("world3d/car",
                   rr.Boxes3D(centers=[[0.0, 0.0, 0.75]],
                              half_sizes=[CAR_HALF_EXTENTS],
                              colors=[[30, 144, 255]]))

        # ── Ground-truth actor boxes (from simulator actor_poses) ─────────────
        if actor_poses_by_ts:
            nearest_ap_ts = min(actor_poses_by_ts, key=lambda t: abs(t - ts))
            if abs(nearest_ap_ts - ts) < 200_000:  # within 200 ms
                for actor_id, ax, ay, az, qx, qy, qz, qw in actor_poses_by_ts[nearest_ap_ts]:
                    if actor_id == "EGO":
                        continue
                    label, half = actor_defs.get(actor_id, ("unknown", np.array([1.0, 0.5, 0.75])))
                    world_pos = np.array([ax, ay, az]) - origin
                    color = [255, 80, 80] if "vehicle" in label else [255, 200, 0]
                    rr.log(
                        f"world3d/actors/{actor_id}",
                        rr.Transform3D(
                            translation=world_pos.tolist(),
                            rotation=rr.Quaternion(xyzw=[qx, qy, qz, qw]),
                        ),
                    )
                    rr.log(
                        f"world3d/actors/{actor_id}",
                        rr.Boxes3D(
                            centers=[[0.0, 0.0, float(half[2])]],
                            half_sizes=[half.tolist()],
                            colors=[color],
                            labels=[label],
                        ),
                    )
                    rr.log(
                        f"world2d/actors/{actor_id}",
                        rr.Points2D(
                            [[world_pos[0], world_pos[1]]],
                            colors=[color],
                            radii=0.5,
                            labels=[label],
                        ),
                    )

        # ── Alpamayo teacher reasoning text ───────────────────────────────────
        if ts in reasoning_by_ts:
            rr.log("reasoning", rr.TextLog(reasoning_by_ts[ts]))

        # Planned trajectory (the executed hypothesis)
        if ts in plan_by_ts:
            raw    = plan_by_ts[ts]
            plan3d = [[x - origin[0], y - origin[1], z - origin[2]] for x, y, z in raw]
            plan2d = [[x - origin[0], y - origin[1]] for x, y, z in raw]
            rr.log("world3d/planned_traj",
                   rr.LineStrips3D([plan3d], colors=[[255, 165, 0]], radii=0.08))
            rr.log("world2d/planned_traj",
                   rr.LineStrips2D([plan2d], colors=[[255, 165, 0]], radii=0.08))

        # AlpaNet hypothesis trajectories (from hypotheses.npz if present)
        if show_hypotheses and ts in hyp_by_ts:
            hyps = hyp_by_ts[ts]   # (M, T, 2) — M hypotheses, T waypoints, xy
            # Current ego position for offsetting relative waypoints
            idx = int(np.searchsorted(ego_ts, ts, side="right")) - 1
            if idx >= 0:
                car_pos = ego_xyz[idx]
                car_q   = ego_q[idx]
                for m, traj_m in enumerate(hyps):
                    # Waypoints in rig frame → world frame via car pose
                    # Simple approximation: rotate by car heading (yaw) then translate
                    qx, qy, qz, qw = car_q
                    yaw = 2 * np.arctan2(qz, qw)
                    c, s = np.cos(yaw), np.sin(yaw)
                    pts_world = np.column_stack([
                        car_pos[0] + c * traj_m[:, 0] - s * traj_m[:, 1],
                        car_pos[1] + s * traj_m[:, 0] + c * traj_m[:, 1],
                        np.full(len(traj_m), car_pos[2] + 0.05),
                    ])
                    color = [200, 200, 200] if m > 0 else [255, 80, 80]  # grey / red
                    rr.log(f"world3d/hyp_{m}",
                           rr.LineStrips3D([pts_world.tolist()], colors=[color], radii=0.05))
                    rr.log(f"world2d/hyp_{m}",
                           rr.LineStrips2D([pts_world[:, :2].tolist()], colors=[color], radii=0.05))

    # ── Camera frames (2D panels only — no images in 3D scene) ───────────────
    for cam_id, frames in cam_frames.items():
        panel_path = CAMERA_DISPLAY_NAMES.get(cam_id, f"camera/{cam_id}")
        for i, (frame_ts, img) in enumerate(frames):
            if i % cam_subsample != 0:
                continue
            if cam_resize < 1.0:
                h, w = img.shape[:2]
                img = np.array(Image.fromarray(img).resize(
                    (max(1, int(w * cam_resize)), max(1, int(h * cam_resize))),
                    Image.BILINEAR,
                ))
            rr.set_time("sim_time_us", sequence=int(frame_ts))
            rr.log(panel_path, rr.Image(img))

    print(f"[viz] Done — {'saved to ' + str(save_path) if save_path else 'viewer spawned'}")


def _load_metrics(metrics_path: Path) -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(metrics_path)
    result = {}
    for name in METRICS_TO_PLOT:
        sub = df[df["name"] == name][["timestamps_us", "values"]].sort_values("timestamps_us")
        if not sub.empty:
            result[name] = sub.reset_index(drop=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="Visualize a DAgger rollout in Rerun.")
    parser.add_argument("rollout_dir")
    parser.add_argument("--save",           metavar="OUT.rrd")
    parser.add_argument("--cam-subsample",  type=int,   default=1,   metavar="N")
    parser.add_argument("--cam-resize",     type=float, default=1.0, metavar="SCALE")
    parser.add_argument("--hypotheses",     action="store_true",
                        help="Show AlpaNet multi-modal hypothesis trajectories "
                             "(requires hypotheses.npz in rollout dir)")
    args = parser.parse_args()

    visualize(Path(args.rollout_dir),
              Path(args.save) if args.save else None,
              cam_subsample=args.cam_subsample,
              cam_resize=args.cam_resize,
              show_hypotheses=args.hypotheses)


if __name__ == "__main__":
    main()
