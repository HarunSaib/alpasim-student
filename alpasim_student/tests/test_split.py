# SPDX-License-Identifier: Apache-2.0
"""Train/validation scene split and its enforcement."""
from pathlib import Path

import pandas as pd
import pytest

from alpasim_student.dagger.collector import LABEL_VERSION
from alpasim_student.dagger.dataset import AlpaNetDataset
from alpasim_student.dagger.loop import CURRICULUM, _curriculum_scenes, run_dagger
from alpasim_student.splits import TRAIN_SCENE_IDS, VAL_SCENE_IDS


def _write_samples(root: Path, scene_id: str, version: int = LABEL_VERSION) -> Path:
    rollout = root / "rollout-0"
    rollout.mkdir(parents=True)
    pd.DataFrame([{"label_version": version, "scene_id": scene_id, "rollout_id": "rollout-0"}]
                 ).to_parquet(rollout / "samples.parquet", index=False)
    return root


def test_scene_lists_are_disjoint_and_unique():
    assert not set(TRAIN_SCENE_IDS) & set(VAL_SCENE_IDS)
    assert len(set(TRAIN_SCENE_IDS)) == len(TRAIN_SCENE_IDS) == 6
    assert len(set(VAL_SCENE_IDS)) == len(VAL_SCENE_IDS) == 2
    assert all(s.startswith("clipgt-") for s in (*TRAIN_SCENE_IDS, *VAL_SCENE_IDS))


def test_dataset_accepts_its_own_split_and_rejects_the_other(tmp_path):
    val_data = _write_samples(tmp_path / "val", VAL_SCENE_IDS[0])
    assert len(AlpaNetDataset(val_data, VAL_SCENE_IDS)) == 1
    with pytest.raises(ValueError, match="outside this split"):
        AlpaNetDataset(val_data, TRAIN_SCENE_IDS)

    train_data = _write_samples(tmp_path / "train", TRAIN_SCENE_IDS[0])
    with pytest.raises(ValueError, match="outside this split"):
        AlpaNetDataset(train_data, VAL_SCENE_IDS)


def test_dataset_rejects_other_label_versions(tmp_path):
    data = _write_samples(tmp_path / "old", TRAIN_SCENE_IDS[0], version=LABEL_VERSION - 1)
    with pytest.raises(ValueError, match="label_version"):
        AlpaNetDataset(data, TRAIN_SCENE_IDS)


def test_curriculum_stays_inside_training_scenes():
    scenes = list(TRAIN_SCENE_IDS)
    for iteration in range(40):
        active = _curriculum_scenes(iteration, scenes)
        assert active and not set(active) & set(VAL_SCENE_IDS)
    assert [len(_curriculum_scenes(t, scenes)) for t, _ in CURRICULUM] == [n for _, n in CURRICULUM]


def test_loop_refuses_validation_scenes_for_collection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="validation scenes"):
        run_dagger(tmp_path / "run", scene_ids=[VAL_SCENE_IDS[0]])
