# SPDX-License-Identifier: Apache-2.0
"""
BEV (Bird's Eye View) feature lifting for AlpaNet.

Projects camera feature maps into a shared BEV grid using precomputed
ftheta projection lookup tables (no learnable geometry parameters).

Grid convention (ego/rig frame):
    x-axis: forward  (BEV_X_MIN → BEV_X_MAX metres ahead)
    y-axis: left     (BEV_Y_MIN → BEV_Y_MAX metres lateral)
    cell (i, j) centre: x = BEV_X_MIN + (i + 0.5) * BEV_RES
                        y = BEV_Y_MIN + (j + 0.5) * BEV_RES
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .camera_utils import CAMERA_IDS, CameraParams

# ── Grid constants ─────────────────────────────────────────────────────────────
BEV_X_MIN =   0.0   # metres forward
BEV_X_MAX =  60.0
BEV_Y_MIN = -30.0   # metres lateral (left positive)
BEV_Y_MAX =  30.0
BEV_RES   =   0.5   # metres per cell

BEV_H = int(round((BEV_X_MAX - BEV_X_MIN) / BEV_RES))   # 120 rows (forward)
BEV_W = int(round((BEV_Y_MAX - BEV_Y_MIN) / BEV_RES))   # 120 cols (lateral)

# Heights at which each BEV cell is sampled for multi-height feature aggregation.
SAMPLE_HEIGHTS: tuple[float, ...] = (-0.5, 0.0, 0.5, 1.0, 2.0)


class BevLifter(nn.Module):
    """Lifts per-camera feature maps into a shared BEV feature volume.

    The sampling grid is precomputed from camera calibration and stored as a
    non-learnable buffer.  Forward pass uses F.grid_sample for differentiable,
    GPU-accelerated bilinear interpolation.

    Args:
        cam_params:  dict camera_id → CameraParams for cameras in CAMERA_IDS order.
        img_w, img_h: dimensions of the resized input images fed to the backbone
                      (NOT original resolution — intrinsics are scaled accordingly).
        feat_stride:  spatial stride of the backbone feature map relative to img.
    """

    def __init__(
        self,
        cam_params: dict[str, CameraParams],
        img_w: int = 480,
        img_h: int = 270,
        feat_stride: int = 8,
    ):
        super().__init__()
        self._cam_ids = [c for c in CAMERA_IDS if c in cam_params]
        self._n_heights = len(SAMPLE_HEIGHTS)
        self._feat_h = img_h // feat_stride
        self._feat_w = img_w // feat_stride

        # Precompute: grid[c, h, i, j, :] = (u_norm, v_norm) for camera c,
        # height index h, BEV row i, BEV col j.
        # Shape stored: (N_cam, N_heights, BEV_H, BEV_W, 2)
        grid_np, valid_np = self._build_grid(cam_params, img_w, img_h, feat_stride)
        self.register_buffer("_grid",  torch.from_numpy(grid_np).float())
        self.register_buffer("_valid", torch.from_numpy(valid_np).float())
        # (N_cam, N_heights, BEV_H, BEV_W) — float mask (0 or 1)

    # ------------------------------------------------------------------
    # Grid precomputation
    # ------------------------------------------------------------------

    def _build_grid(
        self,
        cam_params: dict[str, CameraParams],
        img_w: int,
        img_h: int,
        feat_stride: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        N_cam = len(self._cam_ids)
        N_h   = self._n_heights
        grid  = np.zeros((N_cam, N_h, BEV_H, BEV_W, 2), dtype=np.float32)
        valid = np.zeros((N_cam, N_h, BEV_H, BEV_W),    dtype=np.float32)

        # BEV cell centre coordinates in rig frame (x, y)
        xs = BEV_X_MIN + (np.arange(BEV_H) + 0.5) * BEV_RES   # (BEV_H,)
        ys = BEV_Y_MIN + (np.arange(BEV_W) + 0.5) * BEV_RES   # (BEV_W,)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")     # (BEV_H, BEV_W)
        grid_xy = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)  # (BEV_H*BEV_W, 2)

        # Scale factors from original camera resolution to resized image
        cp_any = cam_params[self._cam_ids[0]]
        scale_u = img_w / cp_any.width
        scale_v = img_h / cp_any.height

        for ci, cam_id in enumerate(self._cam_ids):
            cp = cam_params[cam_id]
            for hi, height in enumerate(SAMPLE_HEIGHTS):
                z_col    = np.full(len(grid_xy), height)
                pts_rig  = np.column_stack([grid_xy[:, 0], grid_xy[:, 1], z_col])
                uv, vis  = cp.project(pts_rig)   # (N, 2), (N,)

                # Scale pixel coords to resized-image space
                u_img = uv[:, 0] * scale_u
                v_img = uv[:, 1] * scale_v

                # Normalize to [-1, 1] for F.grid_sample (align_corners=False):
                # grid_sample maps -1→left edge, +1→right edge of the feature map
                u_norm = 2.0 * u_img / img_w - 1.0
                v_norm = 2.0 * v_img / img_h - 1.0

                uv_norm = np.stack([u_norm, v_norm], axis=1)    # (N, 2)
                uv_norm = uv_norm.reshape(BEV_H, BEV_W, 2)
                vis_map = vis.astype(np.float32).reshape(BEV_H, BEV_W)

                grid[ci, hi]  = uv_norm
                valid[ci, hi] = vis_map

        return grid, valid

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats: (B, N_cam, C, feat_h, feat_w) — per-camera feature maps.
        Returns:
            (B, N_cam * C, BEV_H, BEV_W) — BEV feature volume.
        """
        B, N_cam, C, feat_h, feat_w = feats.shape
        N_h = self._n_heights
        bev_list: list[torch.Tensor] = []

        for ci in range(N_cam):
            cam_feat = feats[:, ci]   # (B, C, feat_h, feat_w)

            # grid[ci]: (N_h, BEV_H, BEV_W, 2)
            # Reshape to (N_h * BEV_H, BEV_W, 2) for grid_sample output (BEV_H, BEV_W*N_h)
            g = self._grid[ci]                                    # (N_h, BEV_H, BEV_W, 2)
            g = g.permute(1, 0, 2, 3)                            # (BEV_H, N_h, BEV_W, 2)
            g = g.reshape(1, BEV_H, N_h * BEV_W, 2)             # (1, BEV_H, N_h*BEV_W, 2)
            g = g.expand(B, -1, -1, -1)                          # (B, BEV_H, N_h*BEV_W, 2)

            sampled = F.grid_sample(
                cam_feat, g, mode="bilinear", padding_mode="zeros", align_corners=False
            )  # (B, C, BEV_H, N_h*BEV_W)

            sampled = sampled.view(B, C, BEV_H, N_h, BEV_W)
            sampled = sampled.permute(0, 1, 2, 4, 3)             # (B, C, BEV_H, BEV_W, N_h)

            # Mask invalid projections
            v = self._valid[ci]                                   # (N_h, BEV_H, BEV_W)
            v = v.permute(1, 2, 0).unsqueeze(0).unsqueeze(0)     # (1, 1, BEV_H, BEV_W, N_h)
            sampled = sampled * v

            # Sum over heights → (B, C, BEV_H, BEV_W)
            bev_list.append(sampled.sum(-1))

        return torch.cat(bev_list, dim=1)   # (B, N_cam*C, BEV_H, BEV_W)
