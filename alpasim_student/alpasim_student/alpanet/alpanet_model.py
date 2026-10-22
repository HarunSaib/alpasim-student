# SPDX-License-Identifier: Apache-2.0
"""
AlpaNet: BEV-encoder multi-hypothesis trajectory model.

Architecture
────────────
  Backbone      ResNet34 up to layer2 (stride 8) → 128-ch feature maps
  BEV Lift      BevLifter: projects 4-camera features into 120×120 BEV grid
  BEV Encoder   3 ConvBN layers with progressive downsampling → global pool
  State Encoder [speed, accel] → 64-d
  Hyp Head      Linear → N_HYPOTHESES × NUM_WAYPOINTS × 2

Inputs
    images: (B, N_cam, 3, IMG_H, IMG_W) — resized, normalised camera frames
    state:  (B, 2)                        — [speed_m_s, accel_m_s2]

Outputs
    traj:  (B, NUM_WAYPOINTS, 2)              — best hypothesis (ego x-fwd, y-left)
    hyps:  (B, N_HYPOTHESES, NUM_WAYPOINTS, 2) — all hypotheses

Training loss (winner-takes-all)
    For each sample find the hypothesis with minimum L1 to GT, backprop only that one.
    This encourages mode coverage over multi-modal futures.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

from .bev_lifter import BEV_H, BEV_W, BevLifter
from .camera_utils import CAMERA_IDS, CameraParams

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_H = 270
IMG_W = 480
NUM_WAYPOINTS = 40      # 10 Hz × 4.0 s
N_HYPOTHESES  = 4
STATE_DIM     = 2       # speed (m/s), acceleration (m/s²)
N_CAM         = len(CAMERA_IDS)
_FEAT_STRIDE  = 8       # ResNet34 stride after layer2


def _conv_bn_relu(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class AlpaNet(nn.Module):
    """AlpaNet core nn.Module — wraps backbone, BEV lifter, and all heads."""

    def __init__(self, cam_params: dict[str, CameraParams]):
        super().__init__()

        # ── Image backbone: ResNet34 up to layer2 ─────────────────────────────
        r34 = tvm.resnet34(weights=None)
        self.backbone = nn.Sequential(
            r34.conv1, r34.bn1, r34.relu, r34.maxpool,  # stride 4
            r34.layer1,                                   # stride 4 (no change)
            r34.layer2,                                   # stride 8 → 128-ch output
        )
        _BACKBONE_CH = 128

        # ── BEV lifter (precomputed geometry, no learnable params) ────────────
        self.bev_lifter = BevLifter(
            cam_params, img_w=IMG_W, img_h=IMG_H, feat_stride=_FEAT_STRIDE
        )
        _BEV_IN_CH = N_CAM * _BACKBONE_CH   # 512

        # ── BEV encoder ───────────────────────────────────────────────────────
        self.bev_enc = nn.Sequential(
            _conv_bn_relu(_BEV_IN_CH, 256),                # (B, 256, 120, 120)
            _conv_bn_relu(256, 128, stride=2),             # (B, 128,  60,  60)
            _conv_bn_relu(128, 128, stride=2),             # (B, 128,  30,  30)
            nn.AdaptiveAvgPool2d(1),                       # (B, 128,   1,   1)
            nn.Flatten(),                                  # (B, 128)
        )
        _BEV_OUT_CH = 128

        # ── State encoder ─────────────────────────────────────────────────────
        self.state_enc = nn.Sequential(
            nn.Linear(STATE_DIM, 64),
            nn.ReLU(inplace=True),
        )

        # ── Multi-hypothesis trajectory head ──────────────────────────────────
        _fused_dim = _BEV_OUT_CH + 64
        self.hyp_head = nn.Sequential(
            nn.Linear(_fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, N_HYPOTHESES * NUM_WAYPOINTS * 2),
        )

    def forward(
        self, images: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (B, N_cam, 3, IMG_H, IMG_W) float32 normalised
            state:  (B, STATE_DIM)
        Returns:
            traj: (B, NUM_WAYPOINTS, 2)               — hypothesis 0 as default output
            hyps: (B, N_HYPOTHESES, NUM_WAYPOINTS, 2) — all hypotheses
        """
        B, N, C, H, W = images.shape

        # Per-camera feature extraction
        feats = self.backbone(images.view(B * N, C, H, W))    # (B*N, 128, H/8, W/8)
        feats = feats.view(B, N, *feats.shape[1:])             # (B, N, 128, H/8, W/8)

        # Lift to BEV
        bev = self.bev_lifter(feats)                           # (B, 512, BEV_H, BEV_W)

        # BEV encode
        bev_feat   = self.bev_enc(bev)                        # (B, 128)
        state_feat = self.state_enc(state)                     # (B, 64)
        fused      = torch.cat([bev_feat, state_feat], dim=1) # (B, 192)

        # Multi-hypothesis output
        raw  = self.hyp_head(fused)                           # (B, M*T*2)
        hyps = raw.view(B, N_HYPOTHESES, NUM_WAYPOINTS, 2)
        traj = hyps[:, 0]                                     # (B, T, 2) — default output

        return traj, hyps


# ── Training utilities ────────────────────────────────────────────────────────

def winner_takes_all_loss(hyps: torch.Tensor, traj_gt: torch.Tensor) -> torch.Tensor:
    """Winner-takes-all L1 loss across multiple trajectory hypotheses.

    For each sample, selects the hypothesis with minimum L1 distance to GT and
    computes the loss only on that hypothesis.  Gradients do not flow through
    the hypothesis-selection step.

    Args:
        hyps:    (B, M, T, 2) — predicted hypotheses
        traj_gt: (B, T, 2)    — teacher trajectory
    Returns:
        scalar loss
    """
    B, M, T, _ = hyps.shape

    gt_expanded = traj_gt.unsqueeze(1).expand_as(hyps)       # (B, M, T, 2)
    per_hyp_l1  = (hyps - gt_expanded).abs().mean(dim=(-2, -1))  # (B, M)

    with torch.no_grad():
        winner_idx = per_hyp_l1.argmin(dim=1)                # (B,)

    winner_hyps = hyps[torch.arange(B, device=hyps.device), winner_idx]  # (B, T, 2)
    return (winner_hyps - traj_gt).abs().mean()
