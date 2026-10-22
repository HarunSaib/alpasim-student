# SPDX-License-Identifier: Apache-2.0
"""
AlpaNet trainer.

Trains on teacher data from the training scenes and validates on a fixed set of
held-out scenes (see splits.py). Uses a winner-takes-all L1 loss (plain L1 when
there is a single hypothesis) and a weighted sampler that favours newer iterations.

Usage:
    uv run python -m alpasim_student.dagger.alpanet_trainer \
        --data ./dataset/iter_0 [./dataset/iter_1 ...] \
        --val-data ./val_dataset \
        --out  ./checkpoints/alpanet_iter1.pth \
        [--epochs 30] [--batch 8] [--lr 3e-4] [--device cuda:0] [--hypotheses 1]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from ..alpanet.alpanet_model import AlpaNet, N_HYPOTHESES, winner_takes_all_loss
from ..alpanet.camera_utils import load_cam_params
from ..paths import find_alpasim_root
from ..splits import TRAIN_SCENE_IDS, VAL_SCENE_IDS
from .dataset import AlpaNetDataset

_REPO_ROOT = find_alpasim_root()


def _compute_traj_metrics(
    pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor
) -> dict[str, float]:
    """ADE, FDE, per-axis MAE and max lateral error over the valid waypoints of (B, T, 2)."""
    w = valid.to(pred.dtype)
    n = w.sum().clamp(min=1)
    diff = pred - gt
    l2 = diff.norm(dim=-1)
    last = (valid * torch.arange(valid.shape[1], device=valid.device)).argmax(dim=1)
    return {
        "ade":         ((l2 * w).sum() / n).item(),
        "fde":         l2[torch.arange(len(l2), device=l2.device), last].mean().item(),
        "loss_x":      ((diff[..., 0].abs() * w).sum() / n).item(),
        "loss_y":      ((diff[..., 1].abs() * w).sum() / n).item(),
        "max_lat_err": (diff[..., 1].abs() * w).max().item(),
    }


def _traj_figure(pred: torch.Tensor, gt: torch.Tensor, n: int = 6) -> "plt.Figure":
    """Plot predicted vs teacher trajectories for up to `n` samples."""
    pred_np = pred[:n].detach().cpu().numpy()
    gt_np = gt[:n].detach().cpu().numpy()

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
    val_dirs: list[Path],
    checkpoint_out: Path,
    num_epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    device_str: str = "cuda:0",
    resume_from: str | None = None,
    dagger_iter: int = 0,
    n_hypotheses: int = N_HYPOTHESES,
) -> None:
    os.chdir(_REPO_ROOT)

    device = torch.device(device_str)
    checkpoint_out = Path(checkpoint_out).resolve()

    # Disable the wandb service subprocess (which re-invokes the main module
    # from a wrong CWD).  With the service disabled wandb runs in-process and
    # still uploads metrics live when an API key is present.
    os.environ.setdefault("WANDB_DISABLE_SERVICE", "1")
    wandb_mode = "online" if os.environ.get("WANDB_API_KEY") else "disabled"

    # ------------------------------------------------------------------ #
    # Dataset                                                              #
    # ------------------------------------------------------------------ #
    datasets = [AlpaNetDataset(d, TRAIN_SCENE_IDS) for d in data_dirs]
    train_ds = ConcatDataset(datasets)
    val_ds = ConcatDataset([AlpaNetDataset(d, VAL_SCENE_IDS) for d in val_dirs])
    n_train, n_val = len(train_ds), len(val_ds)
    n_total = n_train + n_val
    if n_train == 0 or n_val == 0:
        raise ValueError(f"need training and validation samples, got train={n_train} val={n_val}")

    samples_per_iter = {f"iter_{i}": len(ds) for i, ds in enumerate(datasets)}

    wandb.init(
        project="alpasim-alpanet",
        name=f"dagger_iter_{dagger_iter}",
        mode=wandb_mode,
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
        config={
            "model":         "alpanet",
            "num_epochs":    num_epochs,
            "batch_size":    batch_size,
            "lr":            lr,
            "weight_decay":  weight_decay,
            "dagger_iter":   dagger_iter,
            "n_hypotheses":  n_hypotheses,
            "total_samples": n_total,
            "train_samples": n_train,
            "val_samples":   n_val,
            **{f"samples_{k}": v for k, v in samples_per_iter.items()},
        },
    )

    print(f"[alpanet_trainer] {n_total} samples, train={n_train}, val={n_val}, epochs={num_epochs}")

    # Weighted sampler: recent iterations 2× weight
    sample_weights: list[float] = []
    for iter_idx, ds in enumerate(datasets):
        sample_weights += [1.0 + iter_idx / max(1, len(datasets) - 1)] * len(ds)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=0, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
    )

    # ------------------------------------------------------------------ #
    # Model — auto-discover camera calibration from any existing rollout  #
    # ------------------------------------------------------------------ #
    # Search: base_dir/iter_N/[teacher|student]_run/rollouts/CLIP/UUID/rollout.asl
    asl_candidates = (
        sorted(Path.cwd().glob("*/iter_*/*/rollouts/*/*/rollout.asl"))
        or sorted(Path.cwd().glob("*/iter_*/rollouts/*/*/rollout.asl"))
    )
    if not asl_candidates:
        raise RuntimeError(
            "No rollout.asl found — cannot initialise BevLifter calibration. "
            f"Searched: {Path.cwd()}/**/rollout.asl"
        )
    cam_params = load_cam_params("", asl_path=asl_candidates[0])

    model = AlpaNet(cam_params, n_hypotheses=n_hypotheses).to(device)
    if resume_from and Path(resume_from).exists():
        missing, _ = model.load_state_dict(
            torch.load(resume_from, map_location=device, weights_only=True), strict=False
        )
        if missing:
            print(f"[alpanet_trainer] New keys (random init): {missing}")
        print(f"[alpanet_trainer] Resumed from {resume_from}")

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=lr,
        epochs=num_epochs, steps_per_epoch=max(len(train_loader), 1),
        pct_start=0.1, anneal_strategy="cos",
        div_factor=10.0, final_div_factor=100.0,
    )

    best_val_ade = float("inf")
    best_ckpt    = checkpoint_out.parent / (checkpoint_out.stem + "_best.pth")

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #
    for epoch in range(num_epochs):
        # ── train ──
        model.train()
        train_loss = 0.0
        train_metrics: dict[str, float] = {
            "ade": 0.0, "fde": 0.0, "loss_x": 0.0, "loss_y": 0.0,
            "max_lat_err": 0.0, "grad_norm": 0.0,
        }

        for images, state, traj_gt, valid in train_loader:
            images  = images.to(device)
            state   = state.to(device)
            traj_gt = traj_gt.to(device)
            valid   = valid.to(device)

            traj_pred, hyps = model(images, state)
            loss = winner_takes_all_loss(hyps, traj_gt, valid)

            optim.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            optim.step()
            scheduler.step()

            n_batch = images.size(0)
            train_loss += loss.item() * n_batch

            m = _compute_traj_metrics(traj_pred.detach(), traj_gt.detach(), valid)
            for k in m:
                train_metrics[k] += m[k] * n_batch
            train_metrics["grad_norm"] += grad_norm * n_batch

        n_seen = max(n_train, 1)
        train_loss /= n_seen
        for k in train_metrics:
            train_metrics[k] /= n_seen

        # ── validate ──
        model.eval()
        val_loss = 0.0
        val_metrics: dict[str, float] = {
            "ade": 0.0, "fde": 0.0, "loss_x": 0.0, "loss_y": 0.0, "max_lat_err": 0.0,
        }
        val_pred_buf, val_gt_buf = None, None

        with torch.no_grad():
            for images, state, traj_gt, valid in val_loader:
                images  = images.to(device)
                state   = state.to(device)
                traj_gt = traj_gt.to(device)
                valid   = valid.to(device)

                traj_pred, hyps = model(images, state)
                loss = winner_takes_all_loss(hyps, traj_gt, valid)
                n_batch = images.size(0)
                val_loss += loss.item() * n_batch

                m = _compute_traj_metrics(traj_pred, traj_gt, valid)
                for k in m:
                    val_metrics[k] += m[k] * n_batch

                if val_pred_buf is None:
                    val_pred_buf, val_gt_buf = traj_pred, traj_gt

        n_val_seen = max(n_val, 1)
        val_loss /= n_val_seen
        for k in val_metrics:
            val_metrics[k] /= n_val_seen

        if val_metrics["ade"] < best_val_ade:
            best_val_ade = val_metrics["ade"]
            best_ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_ckpt)

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"  Epoch {epoch + 1:3d}/{num_epochs}"
            f"  loss={train_loss:.4f}  val={val_loss:.4f}"
            f"  ADE={train_metrics['ade']:.3f}m  FDE={train_metrics['fde']:.3f}m"
            f"  lr={current_lr:.2e}",
            flush=True,
        )

        log_dict: dict = {
            "epoch": epoch + 1,
            "lr":    current_lr,
            "train/loss": train_loss,
            "train/ade":  train_metrics["ade"],
            "train/fde":  train_metrics["fde"],
            "val/loss":   val_loss,
            "val/ade":    val_metrics["ade"],
            "val/fde":    val_metrics["fde"],
            "val/best_ade": best_val_ade,
        }
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
    print(f"[alpanet_trainer] Saved final → {checkpoint_out}", flush=True)
    print(f"[alpanet_trainer] Saved best  → {best_ckpt}  (val ADE={best_val_ade:.4f}m)", flush=True)

    wandb.summary["best_val_ade"] = best_val_ade
    wandb.summary["dagger_iter"]  = dagger_iter
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",         nargs="+", required=True)
    parser.add_argument("--val-data",     nargs="+", required=True)
    parser.add_argument("--out",          required=True)
    parser.add_argument("--hypotheses",   type=int,   default=N_HYPOTHESES)
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--batch",        type=int,   default=8)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device",       default="cuda:0")
    parser.add_argument("--resume",       default=None)
    parser.add_argument("--dagger-iter",  type=int,   default=0)
    args = parser.parse_args()

    train(
        data_dirs      = [Path(d) for d in args.data],
        val_dirs       = [Path(d) for d in args.val_data],
        n_hypotheses   = args.hypotheses,
        checkpoint_out = Path(args.out),
        num_epochs     = args.epochs,
        batch_size     = args.batch,
        lr             = args.lr,
        weight_decay   = args.weight_decay,
        device_str     = args.device,
        resume_from    = args.resume,
        dagger_iter    = args.dagger_iter,
    )
