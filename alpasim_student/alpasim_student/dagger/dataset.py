# SPDX-License-Identifier: Apache-2.0
"""Training dataset: (camera frames, state, teacher trajectory) from collector output."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset

from ..alpanet.alpanet_model import IMG_H, IMG_W, NUM_WAYPOINTS
from ..alpanet.camera_utils import CAMERA_IDS
from .collector import LABEL_VERSION

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
            df = pd.read_parquet(pq)
            if "label_version" not in df.columns or (df["label_version"] != LABEL_VERSION).any():
                raise ValueError(
                    f"{pq} was not written with label_version {LABEL_VERSION}; "
                    "regenerate it with alpasim_student.dagger.collector."
                )
            for _, row in df.iterrows():
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

        traj = torch.from_numpy(np.vstack(row["traj_xy"]).astype(np.float32))   # (40, 2)
        valid = torch.from_numpy(np.asarray(row["valid"], dtype=bool))          # (40,)
        if traj.shape != (NUM_WAYPOINTS, 2) or valid.shape != (NUM_WAYPOINTS,):
            raise ValueError(f"bad label shape {tuple(traj.shape)} in {row['img_file']}")

        state = torch.tensor([float(row["speed"]), float(row["accel"])], dtype=torch.float32)
        return images, state, traj, valid
