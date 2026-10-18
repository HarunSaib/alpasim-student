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


# Maximum number of actor slots the detection head predicts (BEV ego frame).
K_MAX_ACTORS = 8


class DetectionHead(nn.Module):
    """Predicts up to K_MAX_ACTORS actor positions in ego BEV frame.

    Output per slot: (x_ego_m, y_ego_m, existence_logit).
    Trained with Hungarian matching against ground-truth actor_poses from the
    .asl log; at inference the logit > 0 threshold selects present actors.
    """

    def __init__(self, in_dim: int, k_max: int = K_MAX_ACTORS):
        super().__init__()
        self.k_max = k_max
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, k_max * 3),  # (x, y, logit) per slot
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, k_max, 3): columns are (x_ego, y_ego, existence_logit)."""
        return self.net(x).view(x.shape[0], self.k_max, 3)


class StudentNet(nn.Module):
    """ResNet18 backbone + state fusion + trajectory MLP + detection head.

    Architecture:
        Visual branch:  ResNet18 (no final FC) → 512-d per camera
        State branch:   [speed, accel] → Linear(2, 64) → ReLU
        Fused:          concat → 2112-d
        Traj head:      Linear(2112, 512) → ReLU → Linear(512, 256) → ReLU
                        → Linear(256, T*2)
        Det head:       Linear(2112, 256) → ReLU → Linear(256, K*3)

    ~30 M parameters.  Fits in < 500 MB VRAM.
    """

    IMG_H = 224
    IMG_W = 224
    NUM_WAYPOINTS = 40   # 10 Hz × 4.0 s horizon
    STATE_DIM = 2        # speed (m/s), acceleration (m/s²)

    def __init__(self, num_cameras: int = 4):
        super().__init__()
        backbone = tvm.resnet18(weights=None)
        # Drop the average-pool + FC — we keep the spatial feature map then pool.
        self.visual = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 512, 1, 1)
        visual_dim = 512 * num_cameras
        fused_dim = visual_dim + 64

        self.state_enc = nn.Sequential(
            nn.Linear(self.STATE_DIM, 64),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.NUM_WAYPOINTS * 2),
        )
        self.det_head = DetectionHead(fused_dim)
        self._num_cameras = num_cameras

    def forward(
        self, images: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (B, N_cam, 3, H, W) float32 normalised.
            state:  (B, STATE_DIM) float32.
        Returns:
            traj: (B, NUM_WAYPOINTS, 2) trajectory in rig frame (x-forward, y-left).
            det:  (B, K_MAX_ACTORS, 3) actor slots (x_ego, y_ego, existence_logit).
        """
        B, N, C, H, W = images.shape
        feats = self.visual(images.view(B * N, C, H, W))   # (B*N, 512, 1, 1)
        feats = feats.view(B, N * 512)                      # (B, N*512)
        state_feat = self.state_enc(state)                  # (B, 64)
        fused = torch.cat([feats, state_feat], dim=1)       # (B, 2112)
        traj = self.head(fused).view(B, self.NUM_WAYPOINTS, 2)
        det = self.det_head(fused)                          # (B, K_MAX, 3)
        return traj, det


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
            missing, unexpected = self._net.load_state_dict(state, strict=False)
            if missing:
                logger.info("Checkpoint missing keys (new head, random init): %s", missing)
            if unexpected:
                logger.warning("Checkpoint has unexpected keys: %s", unexpected)
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
            traj_t, det_t = self._net(images, state)
            traj = traj_t[0].cpu().numpy()   # (T, 2)

        # Feed detected actors into MPC cost refinement when enabled.
        if self._use_cost_refinement:
            det_np = det_t[0].cpu().numpy()           # (K, 3)
            present = det_np[:, 2] > 0                # existence logit threshold
            obstacle_positions = (
                det_np[present, :2].astype(np.float32)
                if present.any()
                else np.empty((0, 2), dtype=np.float32)
            )
            traj = refine_trajectory(traj, obstacle_positions)

        headings = self._compute_headings_from_trajectory(traj)
        return ModelPrediction(trajectory_xy=traj, headings=headings)
