# SPDX-License-Identifier: Apache-2.0
"""
Imitation learning trainer for the student model.

Usage:
    uv run python -m alpasim_student.dagger.trainer \
        --data ./dataset/iter_0 [./dataset/iter_1 ...] \
        --out  ./checkpoints/student_iter1.pth \
        [--epochs 30] [--batch 32] [--lr 3e-4] [--device cuda:0] \
        [--resume ./checkpoints/student_iter0.pth] \
        [--dagger-iter 1]
"""
from __future__ import annotations

import argparse
import ast
import io
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import transforms as T

from ..student_model import StudentNet

# Camera order must match StudentModel.camera_ids.
CAMERA_ORDER = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]

# Validation fraction of the full dataset
VAL_FRAC = 0.1


class AlpaSimDataset(Dataset):
    """Loads (images, state, trajectory) triples from a collector output directory."""

    def __init__(self, data_dir: Path, source_iter: int = 0):
        data_dir = Path(data_dir)
        self.records: list[pd.Series] = []
        self.source_iter = source_iter
        for pq in sorted(data_dir.glob("*/samples.parquet")):
            df = pd.read_parquet(pq)
            for _, row in df.iterrows():
                self.records.append(row)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((StudentNet.IMG_H, StudentNet.IMG_W)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        row = self.records[idx]
        npz = np.load(row["img_file"])

        imgs = []
        for cam_id in CAMERA_ORDER:
            imgs.append(
                self.transform(npz[cam_id]) if cam_id in npz
                else torch.zeros(3, StudentNet.IMG_H, StudentNet.IMG_W)
            )
        images = torch.stack(imgs, dim=0)   # (N_cam, 3, H, W)

        traj_raw = row["traj_xy"]
        if isinstance(traj_raw, str):
            traj_raw = ast.literal_eval(traj_raw)
        traj = torch.from_numpy(np.vstack(traj_raw).astype(np.float32))  # (T_teacher, 2)

        T_out = StudentNet.NUM_WAYPOINTS
        if traj.shape[0] >= T_out:
            traj = traj[:T_out]
        else:
            pad = traj[-1:].expand(T_out - traj.shape[0], -1)
            traj = torch.cat([traj, pad], dim=0)

        state = torch.tensor([float(row.get("speed", 0.0)), 0.0], dtype=torch.float32)

        return images, state, traj


def _compute_traj_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    """
    Compute standard trajectory prediction metrics.

    pred, gt: (B, T, 2) tensors
    Returns dict with:
      - ade:   Average Displacement Error  — mean L2 over all timesteps
      - fde:   Final Displacement Error    — L2 at last timestep
      - loss_x: MAE on x component (longitudinal)
      - loss_y: MAE on y component (lateral)
      - max_lat_err: max lateral (y) error across batch & time
    """
    diff = pred - gt                                     # (B, T, 2)
    l2   = diff.norm(dim=-1)                             # (B, T)
    ade  = l2.mean().item()
    fde  = l2[:, -1].mean().item()
    loss_x = diff[..., 0].abs().mean().item()
    loss_y = diff[..., 1].abs().mean().item()
    max_lat = diff[..., 1].abs().max().item()
    return {
        "ade":         ade,
        "fde":         fde,
        "loss_x":      loss_x,
        "loss_y":      loss_y,
        "max_lat_err": max_lat,
    }


def _traj_figure(pred: torch.Tensor, gt: torch.Tensor, n: int = 6) -> "plt.Figure":
    """
    Plot predicted vs GT trajectories for up to `n` samples.
    Returns a matplotlib Figure (caller is responsible for closing).
    """
    pred_np = pred[:n].detach().cpu().numpy()   # (n, T, 2)
    gt_np   = gt[:n].detach().cpu().numpy()

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3), tight_layout=True)
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(gt_np[i, :, 0],   gt_np[i, :, 1],   "g.-", label="teacher", linewidth=1.5)
        ax.plot(pred_np[i, :, 0], pred_np[i, :, 1], "r.-", label="student", linewidth=1.5)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"Sample {i}")
        ax.legend(fontsize=7)
        ax.set_aspect("equal")
    return fig


def train(
    data_dirs: list[Path],
    checkpoint_out: Path,
    num_epochs: int = 30,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    device_str: str = "cuda:0",
    resume_from: str | None = None,
    dagger_iter: int = 0,
):
    device = torch.device(device_str)
    checkpoint_out = Path(checkpoint_out).resolve()

    wandb_mode = "online" if os.environ.get("WANDB_API_KEY") else "disabled"

    # ------------------------------------------------------------------ #
    # Dataset — track how many samples came from each DAgger iteration    #
    # ------------------------------------------------------------------ #
    datasets = [AlpaSimDataset(d, source_iter=i) for i, d in enumerate(data_dirs)]
    full_dataset = ConcatDataset(datasets)
    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * VAL_FRAC))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    samples_per_iter = {f"iter_{i}": len(ds) for i, ds in enumerate(datasets)}

    wandb.init(
        project="alpasim-student",
        name=f"dagger_iter_{dagger_iter}",
        mode=wandb_mode,
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
        config={
            "num_epochs":        num_epochs,
            "batch_size":        batch_size,
            "lr":                lr,
            "weight_decay":      weight_decay,
            "dagger_iter":       dagger_iter,
            "num_datasets":      len(data_dirs),
            "total_samples":     n_total,
            "train_samples":     n_train,
            "val_samples":       n_val,
            "val_frac":          VAL_FRAC,
            "num_waypoints":     StudentNet.NUM_WAYPOINTS,
            "num_cameras":       len(CAMERA_ORDER),
            **{f"samples_{k}": v for k, v in samples_per_iter.items()},
        },
    )
    if wandb_mode == "disabled":
        print("[trainer] W&B disabled (set WANDB_API_KEY to enable)")

    # Log dataset composition as a bar chart
    if wandb_mode == "online":
        fig_data, ax_data = plt.subplots(figsize=(6, 3), tight_layout=True)
        ax_data.bar(list(samples_per_iter.keys()), list(samples_per_iter.values()))
        ax_data.set_xlabel("DAgger iteration")
        ax_data.set_ylabel("Samples")
        ax_data.set_title(f"Dataset composition — {n_total} total")
        wandb.log({"dataset/composition": wandb.Image(fig_data)})
        plt.close(fig_data)

    print(f"[trainer] {n_total} samples across {len(data_dirs)} dataset(s): {samples_per_iter}")
    print(f"[trainer] train={n_train}  val={n_val}  epochs={num_epochs}  lr={lr}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=False,
    )

    # ------------------------------------------------------------------ #
    # Model, optimiser, scheduler                                          #
    # ------------------------------------------------------------------ #
    model = StudentNet(num_cameras=len(CAMERA_ORDER)).to(device)
    if resume_from and Path(resume_from).exists():
        model.load_state_dict(torch.load(resume_from, map_location=device, weights_only=True))
        print(f"[trainer] Resumed from {resume_from}")

    # AdamW with weight decay for better regularisation than plain Adam
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # OneCycleLR: fast warmup + cosine decay → much better than plain cosine for small datasets
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr=lr,
        epochs=num_epochs,
        steps_per_epoch=max(len(train_loader), 1),
        pct_start=0.1,       # 10% warmup
        anneal_strategy="cos",
        div_factor=10.0,     # start lr = max_lr / 10
        final_div_factor=100.0,
    )

    loss_fn = nn.L1Loss()

    best_val_ade = float("inf")
    best_ckpt    = checkpoint_out.parent / (checkpoint_out.stem + "_best.pth")

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #
    for epoch in range(num_epochs):
        # --- train ---
        model.train()
        train_loss = 0.0
        train_metrics: dict[str, float] = {
            "ade": 0.0, "fde": 0.0, "loss_x": 0.0, "loss_y": 0.0,
            "max_lat_err": 0.0, "grad_norm": 0.0,
        }
        last_pred, last_gt = None, None

        for images, state, traj_gt in train_loader:
            images  = images.to(device)
            state   = state.to(device)
            traj_gt = traj_gt.to(device)

            traj_pred = model(images, state)
            loss      = loss_fn(traj_pred, traj_gt)

            optim.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            optim.step()
            scheduler.step()

            n_batch = images.size(0)
            train_loss += loss.item() * n_batch

            m = _compute_traj_metrics(traj_pred.detach(), traj_gt.detach())
            for k in m:
                train_metrics[k] += m[k] * n_batch
            train_metrics["grad_norm"] += grad_norm * n_batch

            last_pred, last_gt = traj_pred.detach(), traj_gt.detach()

        n_train_seen = max(len(train_ds), 1)
        train_loss   /= n_train_seen
        for k in train_metrics:
            train_metrics[k] /= n_train_seen

        current_lr = scheduler.get_last_lr()[0]

        # --- validate ---
        model.eval()
        val_loss = 0.0
        val_metrics: dict[str, float] = {
            "ade": 0.0, "fde": 0.0, "loss_x": 0.0, "loss_y": 0.0,
            "max_lat_err": 0.0,
        }
        val_pred_buf, val_gt_buf = None, None

        with torch.no_grad():
            for images, state, traj_gt in val_loader:
                images  = images.to(device)
                state   = state.to(device)
                traj_gt = traj_gt.to(device)

                traj_pred = model(images, state)
                loss      = loss_fn(traj_pred, traj_gt)
                n_batch   = images.size(0)
                val_loss += loss.item() * n_batch

                m = _compute_traj_metrics(traj_pred, traj_gt)
                for k in m:
                    val_metrics[k] += m[k] * n_batch

                if val_pred_buf is None:
                    val_pred_buf = traj_pred
                    val_gt_buf   = traj_gt

        n_val_seen = max(len(val_ds), 1)
        val_loss  /= n_val_seen
        for k in val_metrics:
            val_metrics[k] /= n_val_seen

        # Save best checkpoint by val ADE
        if val_metrics["ade"] < best_val_ade:
            best_val_ade = val_metrics["ade"]
            best_ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_ckpt)

        # --- console log ---
        print(
            f"  Epoch {epoch + 1:3d}/{num_epochs}"
            f"  loss={train_loss:.4f}  val={val_loss:.4f}"
            f"  ADE={train_metrics['ade']:.3f}m  FDE={train_metrics['fde']:.3f}m"
            f"  lr={current_lr:.2e}"
        )

        # --- W&B log ---
        log_dict: dict = {
            "epoch": epoch + 1,
            "lr":    current_lr,
            # training
            "train/loss":        train_loss,
            "train/loss_x":      train_metrics["loss_x"],
            "train/loss_y":      train_metrics["loss_y"],
            "train/ade":         train_metrics["ade"],
            "train/fde":         train_metrics["fde"],
            "train/max_lat_err": train_metrics["max_lat_err"],
            "train/grad_norm":   train_metrics["grad_norm"],
            # validation
            "val/loss":          val_loss,
            "val/loss_x":        val_metrics["loss_x"],
            "val/loss_y":        val_metrics["loss_y"],
            "val/ade":           val_metrics["ade"],
            "val/fde":           val_metrics["fde"],
            "val/max_lat_err":   val_metrics["max_lat_err"],
            "val/best_ade":      best_val_ade,
        }

        # Trajectory visualisation every 5 epochs
        if wandb_mode == "online" and (epoch + 1) % 5 == 0 and val_pred_buf is not None:
            fig = _traj_figure(val_pred_buf, val_gt_buf, n=min(6, val_pred_buf.size(0)))
            log_dict["val/trajectory_plot"] = wandb.Image(fig)
            plt.close(fig)

        wandb.log(log_dict)

    # ------------------------------------------------------------------ #
    # Save final checkpoint                                                #
    # ------------------------------------------------------------------ #
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_out)
    print(f"[trainer] Saved final → {checkpoint_out}")
    print(f"[trainer] Saved best  → {best_ckpt}  (val ADE={best_val_ade:.4f}m)")

    wandb.summary["best_val_ade"]     = best_val_ade
    wandb.summary["final_train_loss"] = train_loss
    wandb.summary["final_val_loss"]   = val_loss
    wandb.summary["total_samples"]    = n_total
    wandb.summary["dagger_iter"]      = dagger_iter

    if wandb_mode == "online":
        wandb.save(str(checkpoint_out))
        wandb.save(str(best_ckpt))
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",         nargs="+", required=True, help="Dataset directories")
    parser.add_argument("--out",          required=True,            help="Checkpoint output path")
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--batch",        type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device",       default="cuda:0")
    parser.add_argument("--resume",       default=None)
    parser.add_argument("--dagger-iter",  type=int,   default=0)
    args = parser.parse_args()

    train(
        data_dirs      = [Path(d) for d in args.data],
        checkpoint_out = Path(args.out),
        num_epochs     = args.epochs,
        batch_size     = args.batch,
        lr             = args.lr,
        weight_decay   = args.weight_decay,
        device_str     = args.device,
        resume_from    = args.resume,
        dagger_iter    = args.dagger_iter,
    )
