# Project Structure & Architecture

## Overview

This project trains **AlpaNet**, a camera-based driving policy, to imitate the **Alpamayo 1.5**
teacher using a DAgger-style loop inside the **AlpaSim** simulator (NVlabs/alpasim, pinned as a
submodule).

```
Plugin repo (GitHub: HarunSaib/alpasim-student)
        │  sync.sh  (rsync plugin + topology + scripts)
        ▼
AlpaSim checkout (/home/harun/alpasim)  ── runs ──►  DAgger loop (dagger/loop.py)
                                                      ├─ Teacher (Alpamayo 1.5) in AlpaSim Docker
                                                      └─ Student (AlpaNet)      in AlpaSim Docker
```

---

## Repository layout

```
alpasim_student/                   ← deployable plugin (copied to <alpasim>/plugins/ by sync.sh)
  pyproject.toml
  alpasim_student/
    paths.py                       ← finds the AlpaSim checkout
    splits.py                      ← frozen train/validation scene lists
    alpanet/
      alpanet_model.py             ← AlpaNet network + winner-takes-all loss
      bev_lifter.py                ← projects camera features into a BEV grid
      camera_utils.py              ← camera calibration from rollout logs, CAMERA_IDS
      model.py                     ← AlpaSim driver plugin (inference wrapper)
    dagger/
      loop.py                      ← orchestration (entry point)
      collector.py                 ← .asl rollout logs → training samples
      dataset.py                   ← torch Dataset over collected samples
      alpanet_trainer.py           ← training loop
    configs/driver/
      alpanet.yaml                 ← driver config (checkpoint path, cameras)
      student_camera_configs.yaml  ← 4-camera simulation spec
  tests/                           ← label geometry, masked loss, split enforcement
topology/2gpu_alpamayo.yaml        ← GPU placement: driver GPU 0, renderer GPU 1
scripts/                           ← visualize_rollout.py (Rerun), validate_bev_geometry.py,
                                     overfit_check.py (single-head overfit sanity check)
third_party/alpasim/               ← pinned simulator (NVlabs/alpasim 04a17b23)
sync.sh
```

The earlier StudentNet model, its trainer, the MPC planner/cost filter and the GRPO prototype were
removed; they remain in git history before the commit that introduced this file.

---

## AlpaNet

Input: 4 camera frames (270 x 480) and `[speed, acceleration]`.

1. ResNet34 up to `layer2` per camera (stride 8, 128 channels).
2. `BevLifter` samples each camera's features onto a 120 x 120 BEV grid (0–60 m forward, ±30 m,
   0.5 m cells) at 5 heights using precomputed projection tables from the camera calibration.
3. Three conv blocks and global pooling give a 128-d BEV feature.
4. State goes through `Linear(2, 64)`; the fused 192-d vector feeds an MLP that outputs
   K hypotheses x 40 waypoints x (x, y). K defaults to 1 (the single-trajectory baseline); the
   inference class reads K from the checkpoint.

Loss: masked L1. With K = 1 it is plain L1. With K > 1 it is winner-takes-all (only the hypothesis
closest to the teacher gets gradient), but the first hypothesis is still the one executed and
reported, so K > 1 needs a selector before it is a fair comparison.

---

## DAgger loop (as implemented)

```
once:             teacher drives the 2 held-out validation scenes → val_dataset (never trained on)
iteration 0:      teacher drives the first 2 training scenes → collect → train
iterations 1..N:  student drives the curriculum scenes → pass rate / failure check
                  teacher drives the curriculum teacher scenes again (fresh run) → collect
                  train on all data collected so far
```

- Split: 6 training scenes and 2 validation scenes (`splits.py`, frozen). Collection uses training
  scenes only; the loop refuses validation scenes, and the dataset raises if a split contains a scene
  from the other.
- Curriculum over the training scenes: iterations 0–5 use 2, 6–11 use 4, 12+ use all 6.
- Failures are collision or offroad in completed student rollouts. They feed the pass rate and the
  stagnation check (exit if pass rate improves < 2% over 4 iterations, once it has reached 25%).
- The teacher run after the student run is a fresh teacher drive from the scene start. It does not
  label the states the student visited.
- Training: AdamW (lr 1e-4, weight decay 1e-4), OneCycleLR, batch 8, gradient clip 1.0, weighted
  sampler (newer iterations up to 2x), validation on the fixed `val_dataset`, best checkpoint by
  validation ADE of hypothesis 0.
- Checkpoints: `<base-dir>/checkpoints/alpanet_iter_N.pth` and `_best.pth`.

GPU placement is read from the topology; the loop prints it and refuses to start if a required GPU
is busy (`ALPASIM_SKIP_GPU_CHECK=1` overrides).

---

## Data format

`collector.py` reads each `rollout.asl` and writes `<dataset>/<rollout_id>/samples.parquet` plus
`step_*_imgs.npz` (4 camera frames). Each row is one decision step (`label_version` 3):

- `scene_id`, `rollout_id`: where the step came from; the train/validation split is keyed on `scene_id`.
- `traj_xy` (40, 2): teacher poses at +0.1 … +4.0 s, in the ego frame at the request (x forward,
  y left). In the pinned simulator `driver_return` poses are in the scene frame with the current pose
  first, so the collector drops pose 0, matches the rest by timestamp and rotates by the ego yaw. This
  is the exact inverse of the driver's rig-to-scene conversion.
- `valid` (40,): False where a target pose is missing; the loss and metrics ignore those waypoints.
- `speed`, `accel`: the latest ego dynamic state received before the request, which is what the
  driver feeds the model at inference.
- `yaw_rad`, `ego_x`, `ego_y`: ego pose at the request, for debugging and overlays.

The dataset loader refuses data written without `label_version` 3; regenerate it with
`python -m alpasim_student.dagger.collector <run_dir> <out_dir>`. `tests/` covers the label geometry,
the masked loss and the split enforcement.

---

## Scenes

8 scenes from `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`, downloaded with `HF_TOKEN` from `.env`.

| Scene ID | Batch | Split | Note |
|----------|-------|-------|------|
| `clipgt-01d503d4-...` | 0005 | train | default, straightforward |
| `clipgt-a309e228-...` | 0005 | train | labelled turning scene |
| `clipgt-804afc4a-...` | 0005 | train | mixed |
| `clipgt-04db1bb2-...` | 0001 | train | |
| `clipgt-e36cf4c2-...` | 0007 | train | |
| `clipgt-19f339ba-...` | 0009 | train | |
| `clipgt-1bccdc21-...` | 0002 | validation | different lighting/geometry |
| `clipgt-6e190b33-...` | 0010 | validation | maximum diversity |

Validation scenes come from batches no training scene shares.

---

## Known issues

- No same-state teacher labelling of student-visited states.
- With K > 1, hypothesis 0 is executed and reported although WTA trains whichever hypothesis wins.
- The collected iteration-0 data is almost entirely straight driving, so lateral behaviour is
  untested; the overfit check cannot exercise it.
- No closed-loop benchmark yet; validation is open-loop ADE on the held-out scenes.
- No validation data exists until the teacher has been run on the validation scenes (the loop does
  this on first start).

## Metrics

| Metric | Meaning |
|--------|---------|
| `collision_any` | 1.0 if the ego collided during the rollout |
| `offroad` | 1.0 if the ego left the drivable area |
| `progress` | fraction of the route completed |
| `dist_traveled_m` | metres driven |
| `ADE` | mean L2 distance over the 40 waypoints |
| `FDE` | L2 distance at the last waypoint |
| `val ADE` | ADE on the held-out validation split |

## DAgger reference

Ross, Gordon & Bagnell, "A Reduction of Imitation Learning and Structured Prediction to No-Regret
Online Learning", AISTATS 2011: run the learner, label the states it visits with the expert,
aggregate, retrain.
