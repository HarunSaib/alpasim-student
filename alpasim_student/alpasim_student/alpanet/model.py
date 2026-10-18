# SPDX-License-Identifier: Apache-2.0
"""
AlpaNetModel — BaseTrajectoryModel driver plugin for AlpaSim.

Registered as the 'alpanet' entry-point under alpasim.models so the wizard
can select it with `driver=alpanet`.

At inference time, all N_HYPOTHESES trajectories are written to
`hypotheses.npz` in the rollout output directory so visualize_rollout.py
can display them with --hypotheses.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as T
from alpasim_driver.models.base import (
    BaseTrajectoryModel,
    DriveCommand,
    ModelPrediction,
    PredictionInput,
)
from alpasim_driver.schema import ModelConfig

from .alpanet_model import IMG_H, IMG_W, AlpaNet
from .camera_utils import CAMERA_IDS, load_cam_params

logger = logging.getLogger(__name__)


class AlpaNetModel(BaseTrajectoryModel):
    """AlpaSim driver plugin wrapping AlpaNet.

    Requires model_cfg to carry:
        checkpoint_path: path to a .pth state-dict saved by the AlpaNet trainer
        nre_uuid:        NRE scene UUID to load camera calibration (or leave
                         empty to auto-discover from dagger_run directories)
    """

    def __init__(
        self,
        checkpoint_path: str | None,
        device: torch.device,
        camera_ids: list[str],
        nre_uuid: str = "",
    ):
        self._device     = device
        self._camera_ids = [c for c in CAMERA_IDS if c in camera_ids]
        self._hyp_log: dict[int, np.ndarray] = {}   # ts_us → (M, T, 2)

        cam_params = load_cam_params(nre_uuid) if nre_uuid else None
        if cam_params is None:
            # Auto-discover: scan dagger_run dirs for any rollout
            from .camera_utils import _find_asl_for_nre_uuid, _parse_cam_params_from_asl
            import asyncio, glob as _glob
            asls = (
                sorted(Path.cwd().glob("*/iter_*/*/rollouts/*/*/rollout.asl"))
                or sorted(Path.cwd().glob("*/iter_*/rollouts/*/*/rollout.asl"))
            )
            if not asls:
                raise RuntimeError(
                    "No rollout.asl found — "
                    "run at least one DAgger iteration first, or set nre_uuid in the driver config."
                )
            cam_params = asyncio.run(_parse_cam_params_from_asl(asls[0]))
            logger.info("AlpaNetModel: loaded camera calibration from %s", asls[0])

        self._net = AlpaNet(cam_params).to(device)

        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
            missing, unexpected = self._net.load_state_dict(ckpt, strict=False)
            if missing:
                logger.info("Checkpoint missing keys (random init): %s", missing)
            if unexpected:
                logger.warning("Unexpected checkpoint keys: %s", unexpected)
            logger.info("Loaded AlpaNet checkpoint from %s", checkpoint_path)
        else:
            logger.warning(
                "No AlpaNet checkpoint at %s — running with random weights.", checkpoint_path
            )
        self._net.eval()

        self._transform = T.Compose([
            T.Resize((IMG_H, IMG_W)),
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
    ) -> "AlpaNetModel":
        nre_uuid = getattr(model_cfg, "nre_uuid", "") or ""
        return cls(
            checkpoint_path=model_cfg.checkpoint_path,
            device=device,
            camera_ids=camera_ids,
            nre_uuid=nre_uuid,
        )

    # ------------------------------------------------------------------
    # BaseTrajectoryModel interface
    # ------------------------------------------------------------------

    @property
    def camera_ids(self) -> list[str]:
        return self._camera_ids

    @property
    def context_length(self) -> int:
        return 1

    @property
    def output_frequency_hz(self) -> int:
        return 10

    def _encode_command(self, command: DriveCommand):
        return None

    def predict(self, inp: PredictionInput) -> ModelPrediction:
        self._validate_cameras(inp.camera_images)

        imgs = []
        for cam_id in self._camera_ids:
            frames = inp.camera_images[cam_id]
            _, img_np = max(frames, key=lambda f: f[0])
            img_t = torch.from_numpy(img_np).permute(2, 0, 1)
            imgs.append(self._transform(img_t))

        images = torch.stack(imgs, dim=0).unsqueeze(0).to(self._device)  # (1, N, 3, H, W)
        state  = torch.tensor(
            [[inp.speed, inp.acceleration]], dtype=torch.float32, device=self._device
        )

        with torch.no_grad():
            traj_t, hyps_t = self._net(images, state)

        traj = traj_t[0].cpu().numpy()                        # (T, 2)
        hyps = hyps_t[0].cpu().numpy()                        # (M, T, 2)

        # Log hypotheses keyed by request timestamp for later npz export
        ts = getattr(inp, "time_now_us", 0) or 0
        self._hyp_log[int(ts)] = hyps

        headings = self._compute_headings_from_trajectory(traj)
        return ModelPrediction(trajectory_xy=traj, headings=headings)

    def flush_hypotheses(self, out_path: Path) -> None:
        """Write accumulated per-step hypotheses to a .npz file.

        Called by the DAgger loop after each rollout so visualize_rollout.py
        can display them with --hypotheses.
        """
        if not self._hyp_log:
            return
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(out_path), **{str(ts): h for ts, h in self._hyp_log.items()})
        self._hyp_log.clear()
