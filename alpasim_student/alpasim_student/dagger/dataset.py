# SPDX-License-Identifier: Apache-2.0
"""Training dataset: (camera frames, state, teacher trajectory) from collector output."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset

from ..alpanet.alpanet_model import IMG_H, IMG_W, NUM_WAYPOINTS
from ..alpanet.camera_utils import CAMERA_IDS

VAL_FRAC = 0.1

_TRANSFORM = T.Compose([
    T.ToImage(),                                   # numpy HWC uint8 → CHW tensor uint8
    T.Resize((IMG_H, IMG_W)),
    T.ToDtype(torch.float32, scale=True),          # uint8 → float32 in [0, 1]
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class AlpaNetDataset(Dataset):
    """Loads (images, state, trajectory) triples from a collector output directory."""

    def __init__(self, data_dir: Path):
        self.records: list[pd.Series] = []
        for pq in sorted(Path(data_dir).glob("*/samples.parquet")):
            for _, row in pd.read_parquet(pq).iterrows():
                self.records.append(row)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        row = self.records[idx]
        npz = np.load(row["img_file"])

        imgs = []
        for cam_id in CAMERA_IDS:
            if cam_id not in npz:
                raise KeyError(f"camera {cam_id} missing from {row['img_file']}")
            imgs.append(_TRANSFORM(npz[cam_id]))
        images = torch.stack(imgs, dim=0)   # (N_cam, 3, H, W)

        traj_raw = row["traj_xy"]
        if isinstance(traj_raw, str):
            traj_raw = ast.literal_eval(traj_raw)
        traj = torch.from_numpy(np.vstack(traj_raw).astype(np.float32))  # (T_teacher, 2)

        if traj.shape[0] >= NUM_WAYPOINTS:
            traj = traj[:NUM_WAYPOINTS]
        else:
            pad = traj[-1:].expand(NUM_WAYPOINTS - traj.shape[0], -1)
            traj = torch.cat([traj, pad], dim=0)

        state = torch.tensor([float(row.get("speed", 0.0)), 0.0], dtype=torch.float32)
        return images, state, traj
