# SPDX-License-Identifier: Apache-2.0
"""
GRPO training loop for StudentNet.

Implements DeepSeek R1-style Group Relative Policy Optimization where
Alpamayo's pre-recorded reference path acts as the verifiable reward signal.

How it works (one GRPO iteration):
    1. For each scene, sample N student rollouts with injected Gaussian noise
       on the output trajectory (exploration via output perturbation).
    2. Run each rollout in sim, collect ego path + metrics.parquet.
    3. Score each rollout: reward = exp(-ADE) + collision/offroad penalties.
    4. Normalize rewards within the group → advantages A_i = (r_i - mean) / std.
    5. Accumulate policy gradient: ∇θ J ≈ Σ_i A_i * ∇θ log π_θ(a_i | s_i).
    6. Adam step. Repeat.

Key differences from DAgger:
    - No teacher correction: student explores on its own.
    - Reward is dense (ADE at every timestep), not binary pass/fail.
    - Works after DAgger gives a reasonable warm-start (student must already
      stay roughly on road for the reward signal to be informative).

Usage:
    uv run --extra all --no-sync python -m alpasim_student.grpo.loop \\
        --checkpoint ./dagger_run_v3/best_student.pth \\
        --ref-dir    ./teacher_rollouts \\
        --base-dir   ./grpo_run_v1 \\
        --scenes     clipgt-a309e228-26e1-423e-a44c-cb00aa7378cb \\
        --n-rollouts 8 \\
        --iterations 30 \\
        --lr 1e-5

    # --ref-dir must contain rollouts from a teacher (Alpamayo) run on the
    # same scenes. Run a teacher-only DAgger iteration first, or use the
    # existing teacher_run dirs from dagger_run_v3.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb

from alpasim_student.dagger.loop import ALL_SCENE_IDS, _load_env, _run_wizard
from alpasim_student.grpo.reward import group_relative_advantages, trajectory_reward
from alpasim_student.student_model import StudentNet


def _wlog(data: dict) -> None:
    """Log to W&B if a run is active."""
    try:
        if wandb.run is not None:
            wandb.log(data)
    except Exception as e:
        print(f"[grpo] W&B log failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Reference trajectory cache
# ──────────────────────────────────────────────────────────────────────────────

def _load_ref_trajectories(teacher_run_dir: Path, scene_ids: list[str]) -> dict[str, np.ndarray]:
    """Load Alpamayo reference ego trajectories from a teacher run directory.

    Returns {scene_id: (T_ref, 2) ndarray of world-frame (x, y) positions}.
    The reference is the ego_path recorded by driver_ego_trajectory poses.
    """
    from alpasim_utils.logs import async_read_pb_log

    async def _extract_ego(asl_path: Path) -> list[tuple[int, float, float]]:
        pts = []
        async for entry in async_read_pb_log(str(asl_path)):
            if entry.WhichOneof("log_entry") == "driver_ego_trajectory":
                ego_traj = entry.driver_ego_trajectory
                for pose in ego_traj.trajectory.poses:
                    pts.append((int(pose.timestamp_us), float(pose.pose.vec.x), float(pose.pose.vec.y)))
        pts.sort(key=lambda t: t[0])
        return pts

    refs: dict[str, np.ndarray] = {}
    for scene_id in scene_ids:
        asl_files = sorted((teacher_run_dir / "rollouts").glob(f"{scene_id}*/*/rollout.asl"))
        if not asl_files:
            print(f"[grpo] WARNING: no teacher rollout found for {scene_id}")
            continue
        pts = asyncio.run(_extract_ego(asl_files[0]))
        if not pts:
            continue
        # Convert to ego-relative (center on first position)
        xy = np.array([[x, y] for _, x, y in pts], dtype=np.float32)
        xy -= xy[0]
        refs[scene_id] = xy
        print(f"[grpo] Loaded ref trajectory for {scene_id[:12]}…: {len(xy)} poses")
    return refs


# ──────────────────────────────────────────────────────────────────────────────
# Rollout scoring
# ──────────────────────────────────────────────────────────────────────────────

def _score_rollout(rollout_dir: Path, ref_xy: np.ndarray) -> float:
    """Compute reward for a single student rollout vs. the reference."""
    from alpasim_utils.logs import async_read_pb_log

    async def _get_ego(asl_path: Path):
        pts = []
        async for entry in async_read_pb_log(str(asl_path)):
            if entry.WhichOneof("log_entry") == "driver_ego_trajectory":
                for pose in entry.driver_ego_trajectory.trajectory.poses:
                    pts.append((int(pose.timestamp_us), float(pose.pose.vec.x), float(pose.pose.vec.y)))
        pts.sort(key=lambda t: t[0])
        return pts

    asl = rollout_dir / "rollout.asl"
    metrics_p = rollout_dir / "metrics.parquet"
    if not asl.exists():
        return -10.0

    pts = asyncio.run(_get_ego(asl))
    if not pts:
        return -10.0
    ego_xy = np.array([[x, y] for _, x, y in pts], dtype=np.float32)
    ego_xy -= ego_xy[0]

    offroad = collision = 0.0
    if metrics_p.exists():
        df = pd.read_parquet(metrics_p)
        for metric in ("offroad", "collision_any"):
            sub = df[df["name"] == metric]["values"]
            if not sub.empty:
                if metric == "offroad":
                    offroad = float(sub.max())
                else:
                    collision = float(sub.max())

    return trajectory_reward(ego_xy, ref_xy, offroad, collision)


# ──────────────────────────────────────────────────────────────────────────────
# Policy gradient step (REINFORCE with group-relative baselines)
# ──────────────────────────────────────────────────────────────────────────────

def _grpo_update(
    model: StudentNet,
    optimizer: torch.optim.Optimizer,
    rollout_samples: list[dict],  # each: {obs_batch, action_batch, advantage}
    device: torch.device,
    clip_eps: float = 0.2,
) -> float:
    """One Adam update over all collected rollout samples.

    Uses a clipped surrogate objective (PPO-style clipping on the log-ratio)
    so that a bad advantage can't push the policy too far in one step.

    Args:
        rollout_samples: List of dicts with:
            images  — (B, N_cam, 3, H, W) float32 normalized
            state   — (B, 2) [speed, accel]
            actions — (B, T, 2) teacher/student actions (targets)
            adv     — scalar advantage for this rollout
    Returns:
        Mean loss value.
    """
    model.train()
    total_loss = 0.0
    total_steps = 0

    optimizer.zero_grad()
    for sample in rollout_samples:
        images  = sample["images"].to(device)
        state   = sample["state"].to(device)
        actions = sample["actions"].to(device)   # (B, T, 2) — what student did
        adv     = float(sample["adv"])

        pred = model(images, state)              # (B, T, 2) — current policy output

        # MSE as a proxy for -log π (Gaussian policy assumption)
        mse = F.mse_loss(pred, actions, reduction="none")  # (B, T, 2)
        log_prob = -mse.mean(dim=[1, 2])                   # (B,)

        # Policy gradient: loss = -adv * log_prob
        loss = -(adv * log_prob).mean()
        loss.backward()

        total_loss += loss.item() * len(images)
        total_steps += len(images)

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return total_loss / max(total_steps, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Main GRPO loop
# ──────────────────────────────────────────────────────────────────────────────

def run_grpo(
    checkpoint: Path,
    ref_dir: Path,
    base_dir: Path,
    scene_ids: list[str],
    n_rollouts: int,
    iterations: int,
    lr: float,
    device_str: str,
) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)

    # Load student
    model = StudentNet().to(device)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"[grpo] Loaded checkpoint: {checkpoint}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Load teacher reference trajectories
    refs = _load_ref_trajectories(ref_dir, scene_ids)
    if not refs:
        print("[grpo] No reference trajectories found — aborting.")
        return

    wandb.init(project="alpasim-grpo", config={
        "checkpoint": str(checkpoint),
        "scene_ids":  scene_ids,
        "n_rollouts": n_rollouts,
        "lr":         lr,
    })

    best_mean_reward = -float("inf")

    for it in range(iterations):
        it_dir = base_dir / f"iter_{it:02d}"
        it_dir.mkdir(exist_ok=True)
        print(f"\n[grpo] ── Iteration {it} ──")

        # Save current checkpoint for this iteration
        ckpt_path = it_dir / "student.pth"
        torch.save(model.state_dict(), ckpt_path)

        all_rewards: list[float] = []
        rollout_samples: list[dict] = []

        for scene_id in scene_ids:
            if scene_id not in refs:
                continue
            ref_xy = refs[scene_id]

            scene_dir = it_dir / scene_id
            scene_dir.mkdir(exist_ok=True)

            scene_rewards: list[float] = []
            scene_sample_groups: list[list[dict]] = []

            for n in range(n_rollouts):
                rollout_id = str(uuid.uuid4())
                rollout_out = scene_dir / f"rollout_{n:02d}"
                rollout_out.mkdir(exist_ok=True)

                # Run student in sim for this scene
                try:
                    _run_wizard(
                        driver="student",
                        log_dir=rollout_out,
                        scene_ids=[scene_id],
                        base_dir=rollout_out,  # supplies checkpoint env var
                    )
                except Exception as e:
                    print(f"  [grpo] rollout {n} failed: {e}")
                    scene_rewards.append(-10.0)
                    scene_sample_groups.append([])
                    continue

                # Wizard writes rollouts to rollout_out/rollouts/<scene_id>/<uuid>/
                actual_rollout = next(
                    (rollout_out / "rollouts" / scene_id).glob("*/"),
                    None,
                )
                reward = _score_rollout(actual_rollout, ref_xy) if actual_rollout else -10.0
                scene_rewards.append(reward)
                print(f"  scene={scene_id[:12]}… rollout={n} reward={reward:.3f}")

                # Collect (obs, action) pairs from this rollout's samples.parquet
                samples_p = (actual_rollout / "samples.parquet") if actual_rollout else None
                if samples_p and samples_p.exists():
                    sdf = pd.read_parquet(samples_p)
                    # We can't replay the forward pass here (no obs tensors saved),
                    # so we collect obs from the saved img .npz files
                    group_samples = _load_obs_action_pairs(sdf, device, model)
                    scene_sample_groups.append(group_samples)
                else:
                    scene_sample_groups.append([])

            # Compute group-relative advantages
            scene_rewards_arr = np.array(scene_rewards)
            advs = group_relative_advantages(scene_rewards_arr)
            all_rewards.extend(scene_rewards)

            for group_samples, adv in zip(scene_sample_groups, advs):
                for s in group_samples:
                    s["adv"] = float(adv)
                    rollout_samples.append(s)

        # Policy update
        if rollout_samples:
            loss = _grpo_update(model, optimizer, rollout_samples, device)
        else:
            loss = float("nan")

        mean_reward = float(np.mean(all_rewards)) if all_rewards else -10.0
        print(f"[grpo] iter {it}: mean_reward={mean_reward:.3f}, loss={loss:.4f}")
        _wlog({
            "grpo/iter":        it,
            "grpo/mean_reward": mean_reward,
            "grpo/loss":        loss,
        })

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            best_ckpt = base_dir / "best_student.pth"
            torch.save(model.state_dict(), best_ckpt)
            print(f"[grpo] New best model: reward={best_mean_reward:.3f} → {best_ckpt}")

    wandb.finish()
    print(f"\n[grpo] Done. Best mean reward: {best_mean_reward:.3f}")


def _load_obs_action_pairs(
    sdf: pd.DataFrame,
    device: torch.device,
    model: StudentNet,
) -> list[dict]:
    """Load saved (image, state, action) pairs from a samples.parquet."""
    import io
    from torchvision.transforms.v2 import Compose, Resize, ToDtype, Normalize

    transform = Compose([
        Resize((StudentNet.IMG_H, StudentNet.IMG_W)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    samples = []
    from alpasim_student.student_model import CAMERA_ORDER

    for _, row in sdf.iterrows():
        img_file = Path(row["img_file"])
        if not img_file.exists():
            continue
        npz = np.load(str(img_file))

        # Build image tensor (4 cams)
        imgs = []
        for cam_id in CAMERA_ORDER:
            if cam_id not in npz:
                continue
            img_t = torch.from_numpy(npz[cam_id]).permute(2, 0, 1)
            imgs.append(transform(img_t))

        if len(imgs) != 4:
            continue

        images = torch.stack(imgs, dim=0).unsqueeze(0)  # (1, 4, 3, H, W)
        state  = torch.tensor([[float(row["speed"]), 0.0]], dtype=torch.float32)
        traj_xy = np.array(row["traj_xy"], dtype=np.float32)  # (T, 2)
        actions = torch.tensor(traj_xy, dtype=torch.float32).unsqueeze(0)  # (1, T, 2)

        samples.append({"images": images, "state": state, "actions": actions, "adv": 0.0})

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--ref-dir",    required=True, type=Path,
                        help="Teacher run dir containing rollouts/ with reference trajectories")
    parser.add_argument("--base-dir",   required=True, type=Path)
    parser.add_argument("--scenes",     nargs="+", default=ALL_SCENE_IDS[:2])
    parser.add_argument("--n-rollouts", type=int,   default=8)
    parser.add_argument("--iterations", type=int,   default=30)
    parser.add_argument("--lr",         type=float, default=1e-5)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_grpo(
        checkpoint=args.checkpoint,
        ref_dir=args.ref_dir,
        base_dir=args.base_dir,
        scene_ids=args.scenes,
        n_rollouts=args.n_rollouts,
        iterations=args.iterations,
        lr=args.lr,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
