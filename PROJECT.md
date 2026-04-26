# Project Structure & Architecture

## Overview

This project trains a **StudentNet** neural network to drive autonomously by imitating
the **Alpamayo 1.5** teacher model using the **DAgger** algorithm, inside the
**AlpaSim** physics simulation environment.

```
Windows Machine  ──git push──►  GitHub (HarunSaib/alpasim-student)
                                         │
                                    git pull + sync.sh
                                         │
                                         ▼
                               Server (saib) /home/harun/alpasim
                                         │
                               DAgger Loop (loop.py)
                                    ┌────┴────┐
                                 Teacher   Student
                               (Alpamayo)  (StudentNet)
                                    └────┬────┘
                                   AlpaSim Docker
                                  (physics + sensors)
```

---

## Repository Layout

```
alpasim-student/                   ← This repo (your code only)
│
├── sync.sh                        ← Push to GitHub + copy into alpasim + uv sync
├── README.md                      ← Quick-start guide
├── PROJECT.md                     ← This file
│
├── alpasim_student/               ← Python plugin package
│   └── alpasim_student/
│       ├── student_model.py       ← StudentNet architecture + AlpaSim driver interface
│       ├── planner.py             ← MPC trajectory refinement (gradient-free)
│       ├── cost_filter.py         ← Cost functions: safety, comfort, energy
│       │
│       ├── configs/driver/
│       │   ├── student.yaml              ← Driver config (checkpoint path, camera list)
│       │   └── student_camera_configs.yaml  ← 4-camera simulation spec (@package _global_)
│       │
│       └── dagger/
│           ├── loop.py            ← DAgger orchestration (main entry point)
│           ├── trainer.py         ← Neural network training loop
│           └── collector.py       ← Extracts training data from simulation logs
│
└── topology/
    └── 2gpu_alpamayo.yaml         ← AlpaSim wizard topology config (2-GPU server layout)
```

The plugin is installed into the main alpasim repo at:
```
/home/harun/alpasim/plugins/alpasim_student/   ← copied by sync.sh
/home/harun/alpasim/src/wizard/configs/topology/2gpu_alpamayo.yaml
```

---

## Component Interactions

### Full DAgger Loop Data Flow

```
loop.py
  │
  ├─ Iteration 0 (Bootstrap)
  │     │
  │     ├─► _run_wizard("alpamayo1_5")   [2 scenes: 01d503d4 + a309e228]
  │     │        AlpaSim spins up Docker containers:
  │     │          runtime   ← physics engine, clock, events
  │     │          sensorsim ← renders camera frames (4 cameras)
  │     │          driver-0  ← Alpamayo 1.5 teacher model
  │     │          trafficsim← other road agents
  │     │        Output: rollouts/*/rollout.asl  (protobuf log)
  │     │
  │     ├─► collector.py: collect_from_run()
  │     │        Reads rollout.asl → extracts (camera images, speed, teacher trajectory)
  │     │        Speed read from driver_ego_trajectory.dynamic_states (not hardcoded)
  │     │        Output: iter_0/dataset/*/samples.parquet + step_*_imgs.npz
  │     │
  │     └─► trainer.py: train()
  │              Loads all datasets, trains StudentNet with weighted sampling
  │              Output: checkpoints/student_iter_1.pth + student_iter_1_best.pth
  │
  └─ Iterations 1–N (DAgger)
        │
        ├─► _curriculum_scenes()
        │        Expands scene list based on iteration:
        │          iters 0–5:  2 scenes  (basic lane keeping)
        │          iters 6–15: 3 scenes  (turns introduced)
        │          iters 16+:  5 scenes  (full diversity)
        │
        ├─► _run_wizard("student")   [curriculum student scenes]
        │        driver-0 runs StudentNet from /mnt/checkpoints/student_iter_N_best.pth
        │        Output: rollouts/*/rollout.asl + metrics.parquet
        │
        ├─► _detect_failures()
        │        Reads metrics.parquet, flags rollouts where:
        │          collision_any > 0
        │          offroad > 0
        │          plan_deviation > 2.5  (near-miss threshold)
        │        If ALL pass → training converged, loop exits
        │        Stagnation check: exits if pass rate improves < 2% over 4 iters
        │                          (only activates after first non-zero pass rate)
        │
        ├─► _run_wizard("alpamayo1_5")   [curriculum teacher scenes, failed only]
        │        Teacher re-drives failed scenes to demonstrate correct behaviour
        │
        ├─► collector.py: collect_from_run()
        │        Collects teacher corrections as new training data
        │
        └─► trainer.py: train()
                 Trains on ALL data collected so far (dataset aggregation)
                 Recent iterations weighted 2× older ones (WeightedRandomSampler)
                 Output: checkpoints/student_iter_N.pth + student_iter_N_best.pth
```

---

## File-by-File Descriptions

### `dagger/loop.py`
The top-level orchestrator. Runs as:
```bash
uv run --extra all --no-sync python -m alpasim_student.dagger.loop \
    --base-dir ./dagger_run_v2 \
    --iterations 20 \
    --epochs 60
```
Key functions:
- `run_dagger()` — main loop, calls all other components
- `_run_wizard()` — generates docker-compose.yaml via alpasim_wizard, patches it to add volume mounts, then runs `docker compose up`
- `_curriculum_scenes()` — returns the scene list for the current iteration based on CURRICULUM schedule
- `_detect_failures()` — reads `metrics.parquet` to find rollouts with collisions, offroad events, or high plan deviation (> 2.5)
- `_pivot_metrics()` — converts AlpaSim's long-format parquet (rows of name/value pairs) into a flat `{metric_name: value}` dict
- `_patch_student_checkpoint()` — updates `student.yaml` with the latest `_best.pth` checkpoint path before each student run
- `_build_eval_log()` — collects per-rollout metrics into a summary dict for W&B logging

Key constants:
- `ALL_SCENE_IDS` — 5 scenes from HuggingFace `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` (Batch0002, Batch0005 ×3, Batch0010)
- `CURRICULUM` — maps iteration thresholds to scene counts: `[(0, 2 scenes), (6, 3 scenes), (16, 5 scenes)]`

### `dagger/trainer.py`
Trains StudentNet on collected datasets. Key details:
- **Optimizer**: AdamW (weight_decay=1e-4)
- **Scheduler**: OneCycleLR — warmup for 10% of training, then cosine decay
- **Loss**: MSE on (x, y) trajectory waypoints across all 25 waypoints
- **Sampler**: WeightedRandomSampler — recent iterations weighted up to 2× older ones
- **Validation**: 10% random split, saves best checkpoint by val ADE
- **Metrics logged to W&B**: ADE, FDE, loss, grad_norm per epoch

### `dagger/collector.py`
Parses AlpaSim's binary `.asl` protobuf rollout logs. For each simulation step:
1. Reads `driver_camera_image` entries → buffers JPEG frames per camera per timestamp
2. Reads `driver_ego_trajectory` entries → extracts real ego speed from `dynamic_states.linear_velocity`
3. Reads `driver_request` entries → marks a new decision step, picks closest camera frames
4. Reads `driver_return` entries → extracts teacher's planned trajectory (up to 65 poses, 6.4s)
5. Writes `samples.parquet` (metadata) + `step_*_imgs.npz` (camera images) per rollout

Note: speed is read from the actual simulation state, not hardcoded — this was a critical bug fix from v1.

### `student_model.py`
Defines the neural network and its AlpaSim driver interface.

**StudentNet architecture:**
```
4 cameras (224×224 RGB)
    │
    ▼
ResNet18 backbone (shared weights per camera)
    │  → 512-d feature vector per camera
    ▼
Concatenate (4 × 512 = 2048-d)
    │
    ├── Speed + Acceleration → Linear(2→64) → ReLU
    │
Fuse → Linear(2112→512) → ReLU → Linear(512→256) → ReLU → Linear(256→50)
    │
    ▼
25 waypoints × (x, y) = 2.5 second trajectory at 10 Hz
```

The teacher (Alpamayo 1.5) provides 65 poses per step (6.4s), so the full
teacher trajectory is available in collected data — the 25-waypoint target
is a truncation of it, not fabricated.

**StudentModel** wraps StudentNet with the `BaseTrajectoryModel` interface that
AlpaSim's driver service expects — handling camera ordering, image preprocessing
(resize → float → ImageNet normalise), and speed/acceleration state.

### `planner.py`
Optional MPC refinement applied after StudentNet inference. Uses finite-difference
gradient descent to nudge the predicted trajectory away from obstacles while
staying close to the original prediction. Currently disabled by default
(`use_cost_refinement=False`).

### `cost_filter.py`
Three cost functions used by the MPC planner:
- **Safety cost** — Artificial Potential Field (APF) repulsion from nearby obstacles
- **Comfort cost** — trajectory jerk (smoothness penalty)
- **Energy cost** — total path length

### `configs/driver/student.yaml`
Hydra config that tells AlpaSim how to run the student driver:
- Which model class to instantiate (`student` entry point)
- Camera IDs to request from sensorsim
- Checkpoint path inside the container (`/mnt/checkpoints/student_iter_N_best.pth`)

### `configs/driver/student_camera_configs.yaml`
Uses `@package _global_` to inject the 4-camera simulation spec at the root
config level — this is what tells AlpaSim's sensorsim to actually render
all 4 cameras. Without this, the student only gets 2 cameras and crashes.

### `topology/2gpu_alpamayo.yaml`
AlpaSim wizard topology config for a 2-GPU server. Extends the base `2gpu`
topology with `trafficsim.n_concurrent_rollouts` which was missing and caused
crashes when running student eval runs.

---

## Output Directory Structure (per run)

```
dagger_run_v2/
├── checkpoints/
│   ├── student_iter_1.pth          ← After iter 0 (bootstrap)
│   ├── student_iter_1_best.pth     ← Best val ADE from iter 0 training
│   ├── student_iter_2.pth
│   ├── student_iter_2_best.pth
│   └── ...
│
├── iter_0/
│   ├── teacher_run/
│   │   ├── docker-compose.yaml
│   │   └── rollouts/
│   │       └── <scene_id>/<rollout_id>/
│   │           ├── rollout.asl     ← Raw protobuf log (camera images + trajectories)
│   │           └── metrics.parquet ← Long-format metrics (collision, progress, etc.)
│   └── dataset/
│       └── <rollout_id>/
│           ├── samples.parquet     ← (timestamp, speed, trajectory, img_file) per step
│           └── step_*_imgs.npz     ← Camera images per step
│
├── iter_1/
│   ├── student_run/                ← Student drives (curriculum scenes)
│   ├── teacher_correction_run/     ← Teacher re-drives failed scenes
│   └── dataset/
│
└── iter_N/ ...
```

---

## Scenes

5 scenes from HuggingFace `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` (910 scenes total):

| Scene ID | Batch | Role |
|----------|-------|------|
| `clipgt-01d503d4-...` | Batch0005 | Default — straightforward lane keeping |
| `clipgt-a309e228-...` | Batch0005 | Turning scene — historically hardest |
| `clipgt-804afc4a-...` | Batch0005 | Mixed geometry |
| `clipgt-1bccdc21-...` | Batch0002 | Different lighting/geometry |
| `clipgt-6e190b33-...` | Batch0010 | Maximum diversity |

Scenes are downloaded automatically via `HF_TOKEN` in `.env`.

---

## Metrics Glossary

| Metric | Meaning |
|--------|---------|
| `collision_any` | 1.0 if the ego vehicle collided with anything during the rollout |
| `offroad` | 1.0 if the ego vehicle left the drivable area at any point |
| `plan_deviation` | Max deviation from the teacher's planned path (metres). Threshold: > 2.5 triggers correction |
| `progress` | Fraction of the route completed (0.0–1.0) |
| `progress_rel` | Progress relative to current GT segment — better real-time signal |
| `dist_traveled_m` | Total metres driven before failure or completion |
| `wrong_lane` | 1.0 if ego drove in the opposing lane |
| `ADE` | Average Displacement Error — mean distance (metres) between predicted and teacher waypoints across all 25 timesteps |
| `FDE` | Final Displacement Error — distance at the 25th waypoint only |
| `val ADE` | ADE measured on the held-out 10% validation split |

---

## Training History

### v1 (iters 1–30, speed bug present)
- Speed hardcoded to 0.0 for all training samples — model never learned speed conditioning
- 10 waypoints (1s horizon)
- Single teacher scene
- Best val ADE: ~0.008m at iter 21 (but model failed turns in simulation)
- All 3 scenes: `01d503d4` passed, `a309e228` offroad 100%, `804afc4a` passed

### v2 (current)
- Speed fix: reads real ego speed from `driver_ego_trajectory.dynamic_states`
- 25 waypoints (2.5s horizon)
- Curriculum: 2 → 3 → 5 scenes over 16 iterations
- Teacher corrections on 2 scenes from iter 0
- Weighted sampling: recent iters 2× older ones
- Stagnation check: exits after pass rate plateaus
- Best val ADE: ~0.036m at iter 6 (significantly better generalisation expected)

---

## Algorithm Definitions

### DAgger (Dataset Aggregation)
**Paper**: Ross, Gordon & Bagnell, "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning", AISTATS 2011.

Standard behavioural cloning fails because the student drifts off the teacher's
trajectory during deployment, entering states it never saw during training — the
**covariate shift** problem. DAgger fixes this iteratively:

```
1. Collect dataset D_0 by running the teacher policy π*
2. Train student π_1 on D_0
3. For iteration i = 1, 2, ..., N:
   a. Run student π_i in the environment
   b. At every visited state s, query teacher π* for the correct action
   c. Add new (state, teacher_action) pairs to dataset: D_i = D_{i-1} ∪ new_data
   d. Train new student π_{i+1} on all data D_i
4. Return best π_i by validation performance
```

In this project:
- **π\*** = Alpamayo 1.5 (the teacher)
- **π_i** = StudentNet checkpoint `student_iter_i_best.pth`
- **States** = 4-camera image + speed + acceleration
- **Actions** = 25-waypoint trajectory (2.5 second horizon at 10 Hz)

### Imitation Learning (Behavioural Cloning)
The base form of learning from demonstrations:

```
L = (1/T) Σ_t ||π_student(s_t) - π_teacher(s_t)||²
```

Where `s_t` is the observation at time t and T=25 waypoints. No reward signal
needed — only expert demonstrations. DAgger fixes the covariate shift limitation.

### ResNet18 (Visual Encoder)
A residual convolutional neural network with 18 layers. Each of the 4 camera images
is passed through ResNet18 to produce a 512-dimensional feature vector, giving
2048-d total visual features before fusion with the state.

### AdamW (Optimiser)
Adam with decoupled weight decay. Applies weight decay directly to weights
independent of the gradient — cleaner regularisation on small datasets.

### OneCycleLR (Learning Rate Schedule)
Warmup to peak LR (3e-4) over ~10% of training, then cosine decay to near-zero.
Prevents early instability, helps escape shallow local minima, fine-tunes at end.

### ADE / FDE (Trajectory Error Metrics)
- **ADE**: mean Euclidean distance across all 25 predicted waypoints vs teacher
- **FDE**: Euclidean distance at the 25th (final) waypoint only

Both in metres. Lower is better. FDE is more sensitive to long-horizon accuracy.

### APF (Artificial Potential Field)
Used in `cost_filter.py` for obstacle avoidance in the optional MPC refinement:
```
U_rep(d) = 0.5 * k_rep * (1/d - 1/d₀)²   if d < d₀
           0                                otherwise
```
Where `d` is distance to nearest obstacle and `d₀` is the influence radius.

### Next Steps (Post-DAgger)
After DAgger convergence, fine-tuning with **SAC (Soft Actor-Critic)** is the
recommended next step. SAC is preferred over PPO due to:
- Off-policy replay buffer: reuses all rollout data, critical given slow sim (~2hrs/scene)
- DAgger dataset pre-seeds the replay buffer with labelled demonstrations
- Entropy maximisation encourages exploration of edge cases (turns, recovery)
- PPO would require ~42–200 days compute from scratch; SAC fine-tuning from DAgger: ~1–2 weeks

Required changes for SAC:
- Add Q-networks + stochastic policy head to StudentNet
- Replace MSE loss with soft Q-loss + policy loss
- Replace collector with (s, a, r, s', done) tuple extraction
- Pre-fill replay buffer from existing DAgger rollout data
- Reward from simulator's per-timestep metrics: `progress_rel`, `dist_to_gt_trajectory`, `offroad`, `collision_any`
