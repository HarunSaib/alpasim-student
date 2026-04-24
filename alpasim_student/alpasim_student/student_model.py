# SPDX-License-Identifier: Apache-2.0
"""
Student model: ResNet18 visual encoder + MLP trajectory head.

Implements BaseTrajectoryModel so it plugs into AlpaSim's driver service
with no changes to any other component.  Registered as the 'student' entry
point under alpasim.models.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms.v2 as T
from alpasim_driver.models.base import (
    BaseTrajectoryModel,
    DriveCommand,
    ModelPrediction,
    PredictionInput,
)
from alpasim_driver.schema import ModelConfig

from .planner import refine_trajectory

logger = logging.getLogger(__name__)

# Camera order must match the Alpamayo rig used during data collection.
CAMERA_ORDER = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]


class StudentNet(nn.Module):
    """ResNet18 backbone + state fusion + trajectory MLP.

    Architecture:
        Visual branch:  ResNet18 (no final FC) → 512-d per camera
        State branch:   [speed, accel] → Linear(2, 64) → ReLU
        Head:           Linear(512*N_cam + 64, 512) → ReLU
                        → Linear(512, 256) → ReLU
                        → Linear(256, T*2)

    ~30 M parameters.  Fits in < 500 MB VRAM.
    """

    IMG_H = 224
    IMG_W = 224
    NUM_WAYPOINTS = 25   # 10 Hz × 2.5 s horizon
    STATE_DIM = 2        # speed (m/s), acceleration (m/s²)

    def __init__(self, num_cameras: int = 4):
        super().__init__()
        backbone = tvm.resnet18(weights=None)
        # Drop the average-pool + FC — we keep the spatial feature map then pool.
        self.visual = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 512, 1, 1)
        visual_dim = 512 * num_cameras

        self.state_enc = nn.Sequential(
            nn.Linear(self.STATE_DIM, 64),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(visual_dim + 64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.NUM_WAYPOINTS * 2),
        )
        self._num_cameras = num_cameras

    def forward(self, images: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, N_cam, 3, H, W) float32 normalised.
            state:  (B, STATE_DIM) float32.
        Returns:
            (B, NUM_WAYPOINTS, 2) trajectory in rig frame (x-forward, y-left).
        """
        B, N, C, H, W = images.shape
        feats = self.visual(images.view(B * N, C, H, W))   # (B*N, 512, 1, 1)
        feats = feats.view(B, N * 512)                      # (B, N*512)
        state_feat = self.state_enc(state)                  # (B, 64)
        out = self.head(torch.cat([feats, state_feat], dim=1))
        return out.view(B, self.NUM_WAYPOINTS, 2)


class StudentModel(BaseTrajectoryModel):
    """AlpaSim driver plugin wrapping StudentNet.

    Optionally applies MPC cost refinement after prediction when
    obstacle positions are available via the prediction_input context
    (currently a no-op placeholder — hook into your obstacle detector here).
    """

    def __init__(
        self,
        checkpoint_path: str | None,
        device: torch.device,
        camera_ids: list[str],
        use_cost_refinement: bool = False,
    ):
        self._device = device
        self._camera_ids = camera_ids
        self._use_cost_refinement = use_cost_refinement

        self._net = StudentNet(num_cameras=len(camera_ids)).to(device)
        if checkpoint_path and Path(checkpoint_path).exists():
            state = torch.load(checkpoint_path, map_location=device, weights_only=True)
            self._net.load_state_dict(state)
            logger.info("Loaded student checkpoint from %s", checkpoint_path)
        else:
            logger.warning(
                "No checkpoint found at %s — running with random weights.",
                checkpoint_path,
            )
        self._net.eval()

        self._transform = T.Compose([
            T.Resize((StudentNet.IMG_H, StudentNet.IMG_W)),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @classmethod
    def from_config(
        cls,
        model_cfg: ModelConfig,
        device: torch.device,
        camera_ids: list[str],
        context_length: int | None,
        output_frequency_hz: int,
    ) -> "StudentModel":
        return cls(
            checkpoint_path=model_cfg.checkpoint_path,
            device=device,
            camera_ids=camera_ids,
        )

    # ------------------------------------------------------------------
    # BaseTrajectoryModel interface
    # ------------------------------------------------------------------

    @property
    def camera_ids(self) -> list[str]:
        return self._camera_ids

    @property
    def context_length(self) -> int:
        return 1  # single-frame model

    @property
    def output_frequency_hz(self) -> int:
        return 10

    def _encode_command(self, command: DriveCommand):
        return None

    def predict(self, inp: PredictionInput) -> ModelPrediction:
        self._validate_cameras(inp.camera_images)

        # Build image tensor — latest frame per camera, sorted by CAMERA_ORDER.
        imgs = []
        for cam_id in self._camera_ids:
            frames = inp.camera_images[cam_id]
            _, img_np = max(frames, key=lambda f: f[0])   # latest timestamp
            img_t = torch.from_numpy(img_np).permute(2, 0, 1)  # HWC → CHW uint8
            imgs.append(self._transform(img_t))

        images = torch.stack(imgs, dim=0).unsqueeze(0).to(self._device)  # (1, N, 3, H, W)
        state  = torch.tensor(
            [[inp.speed, inp.acceleration]], dtype=torch.float32, device=self._device
        )

        with torch.no_grad():
            traj = self._net(images, state)[0].cpu().numpy()   # (T, 2)

        # Optional MPC cost refinement (no obstacles by default — extend here).
        if self._use_cost_refinement:
            obstacle_positions = np.empty((0, 2), dtype=np.float32)
            traj = refine_trajectory(traj, obstacle_positions)

        headings = self._compute_headings_from_trajectory(traj)
        return ModelPrediction(trajectory_xy=traj, headings=headings)
