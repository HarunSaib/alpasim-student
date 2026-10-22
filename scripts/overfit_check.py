# SPDX-License-Identifier: Apache-2.0
"""Overfit sanity check: can the pipeline fit a few dozen labelled frames?

Trains a fresh AlpaNet on N fixed frames and reports how closely it reproduces the teacher
trajectories, next to two trivial baselines (zero motion, constant speed straight ahead).
A healthy pipeline drives the error far below both.

Run from the AlpaSim root:
    uv run --extra all --no-sync python scripts/overfit_check.py \
        --data alpanet_dagger_labels_v3/iter_0/dataset [--n 48] [--steps 400] [--device cuda:0]
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from alpasim_student.alpanet.alpanet_model import AlpaNet, winner_takes_all_loss
from alpasim_student.alpanet.camera_utils import load_cam_params
from alpasim_student.dagger.alpanet_trainer import _compute_traj_metrics
from alpasim_student.dagger.dataset import AlpaNetDataset
from alpasim_student.splits import TRAIN_SCENE_IDS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hypotheses", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="alpanet_dagger_labels_v3/overfit")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    ds = AlpaNetDataset(Path(args.data), TRAIN_SCENE_IDS)
    idx = np.linspace(0, len(ds) - 1, args.n).astype(int)
    items = [ds[int(i)] for i in idx]
    images = torch.stack([it[0] for it in items])
    state = torch.stack([it[1] for it in items])
    traj = torch.stack([it[2] for it in items])
    valid = torch.stack([it[3] for it in items])
    print(f"[overfit] {args.n} frames from {len(ds)} samples; {args.steps} steps, batch {args.batch}")

    asl = sorted(glob.glob("*/iter_*/*/rollouts/*/*/rollout.asl"))[0]
    model = AlpaNet(load_cam_params("", asl_path=Path(asl)), n_hypotheses=args.hypotheses).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.steps, eta_min=args.lr / 20)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        b = torch.randperm(args.n)[: args.batch]
        _, hyps = model(images[b].to(device), state[b].to(device))
        loss = winner_takes_all_loss(hyps, traj[b].to(device), valid[b].to(device))
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        sched.step()
        if step == 1 or step % 25 == 0:
            print(f"  step {step:4d}  loss={loss.item():.4f}  ({time.time() - t0:.0f}s)", flush=True)

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, args.n, args.batch):
            preds.append(model(images[i:i + args.batch].to(device), state[i:i + args.batch].to(device))[0].cpu())
    pred = torch.cat(preds)

    steps_t = torch.arange(1, traj.shape[1] + 1, dtype=torch.float32) * 0.1
    const_vel = torch.zeros_like(traj)
    const_vel[..., 0] = state[:, :1] * steps_t[None]

    results = {
        "model":              _compute_traj_metrics(pred, traj, valid),
        "zero_motion":        _compute_traj_metrics(torch.zeros_like(traj), traj, valid),
        "constant_velocity":  _compute_traj_metrics(const_vel, traj, valid),
    }
    per_frame = (pred - traj).norm(dim=-1).mean(dim=1)
    results["model_worst_frame_ade"] = per_frame.max().item()
    results["peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0

    print("\n[overfit] error on the training frames (metres)")
    print(f"  {'':20s} {'ADE':>7s} {'FDE':>7s} {'|dx|':>7s} {'|dy|':>7s}")
    for name in ("model", "zero_motion", "constant_velocity"):
        m = results[name]
        print(f"  {name:20s} {m['ade']:7.3f} {m['fde']:7.3f} {m['loss_x']:7.3f} {m['loss_y']:7.3f}")
    print(f"  worst single-frame ADE: {results['model_worst_frame_ade']:.3f}   peak GPU memory: {results['peak_gpu_mem_gb']:.1f} GB")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "overfit_results.json").write_text(json.dumps(results, indent=2))
    show = np.linspace(0, args.n - 1, 8).astype(int)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), tight_layout=True)
    for ax, i in zip(axes.ravel(), show):
        ax.plot(traj[i, :, 0], traj[i, :, 1], "g.-", label="teacher")
        ax.plot(pred[i, :, 0], pred[i, :, 1], "r.-", label="model")
        ax.set_title(f"frame {i}  v={state[i, 0]:.1f} m/s  ADE={per_frame[i]:.2f} m")
        ax.set_xlabel("forward x (m)")
        ax.set_ylabel("left y (m)")
        ax.axis("equal")
    axes[0, 0].legend()
    fig.savefig(out / "overfit_trajectories.png", dpi=110)
    print(f"[overfit] wrote {out}/overfit_results.json and overfit_trajectories.png")


if __name__ == "__main__":
    main()
