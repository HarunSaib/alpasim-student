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
├── sync.sh                        ← Pull from GitHub + copy into alpasim
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
  │     ├─► _run_wizard("alpamayo1_5")
  │     │        AlpaSim spins up Docker containers:
  │     │          runtime   ← physics engine, clock, events
  │     │          sensorsim ← renders camera frames (4 cameras)
  │     │          driver-0  ← Alpamayo 1.5 teacher model
  │     │          trafficsim← other road agents
  │     │        Output: rollouts/*/rollout.asl  (protobuf log)
  │     │
  │     ├─► collector.py: collect_from_run()
  │     │        Reads rollout.asl → extracts (camera images, teacher trajectory)
  │     │        Output: iter_0/dataset/*/samples.parquet + step_*_imgs.npz
  │     │
  │     └─► trainer.py: train()
  │              Loads all datasets, trains StudentNet
  │              Output: checkpoints/student_iter_1.pth
  │
  └─ Iterations 1–N (DAgger)
        │
        ├─► _run_wizard("student")
        │        Same Docker setup but driver-0 runs StudentNet instead
        │        Reads checkpoint from /mnt/checkpoints/ (Docker volume mount)
        │        Output: rollouts/*/rollout.asl + metrics.parquet
        │
        ├─► _detect_failures()
        │        Reads metrics.parquet (long-format: name/values/time_aggregation)
        │        Flags rollouts where collision_any > 0 or offroad > 0
        │        If ALL pass → training converged, loop exits early
        │
        ├─► _run_wizard("alpamayo1_5")   [only on failed rollouts]
        │        Teacher re-drives the SAME scene to demonstrate correct behaviour
        │
        ├─► collector.py: collect_from_run()
        │        Collects teacher corrections as new training data
        │
        └─► trainer.py: train()
                 Trains on ALL data collected so far (dataset aggregation)
                 Output: checkpoints/student_iter_N.pth
```

---

## File-by-File Descriptions

### `dagger/loop.py`
The top-level orchestrator. Runs as:
```bash
uv run python -m alpasim_student.dagger.loop --base-dir ./dagger_run --iterations 7 --epochs 30
```
Key functions:
- `run_dagger()` — main loop, calls all other components
- `_run_wizard()` — generates docker-compose.yaml via alpasim_wizard, patches it to add volume mounts, then runs `docker compose up`
- `patch_driver_block()` — injects `/mnt/checkpoints` and `/repo/plugins` volume mounts into the driver container so it can find the student checkpoint and plugin code
- `_detect_failures()` — reads `metrics.parquet` to find which rollouts had collisions or offroad events
- `_pivot_metrics()` — converts AlpaSim's long-format parquet (rows of name/value pairs) into a flat `{metric_name: value}` dict
- `_patch_student_checkpoint()` — updates `student.yaml` with the latest checkpoint path before each student run
- `_build_eval_log()` — collects per-rollout metrics into a summary dict for logging

### `dagger/trainer.py`
Trains StudentNet on collected datasets. Key details:
- **Optimizer**: AdamW (weight_decay=1e-4) — better regularisation than plain Adam
- **Scheduler**: OneCycleLR — warmup for 10% of training, then cosine decay
- **Loss**: MSE on (x, y) trajectory waypoints
- **Validation**: 10% random split, saves best checkpoint by val ADE
- **Metrics logged**: ADE, FDE, loss_x, loss_y, grad_norm per epoch

### `dagger/collector.py`
Parses AlpaSim's binary `.asl` protobuf rollout logs. For each simulation step:
1. Reads `driver_camera_image` entries → buffers JPEG frames per camera per timestamp
2. Reads `driver_request` entries → marks a new decision step, picks closest camera frames
3. Reads `driver_return` entries → extracts teacher's planned trajectory (x, y waypoints)
4. Writes `samples.parquet` (metadata) + `step_*_imgs.npz` (camera images) per rollout

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
Fuse → Linear(2112→512) → ReLU → Linear(512→256) → ReLU → Linear(256→20)
    │
    ▼
10 waypoints × (x, y) = 1 second trajectory at 10 Hz
```

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
- Checkpoint path inside the container (`/mnt/checkpoints/student_iter_N.pth`)

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
│           ├── samples.parquet     ← (timestamp, trajectory, img_file) per step
│           └── step_*_imgs.npz     ← Camera images per step
│
├── iter_1/
│   ├── student_run/                ← Student drives
│   ├── teacher_correction_run/     ← Teacher re-drives failed scenes
│   └── dataset/
│
└── iter_N/ ...
```

---

## Metrics Glossary

| Metric | Meaning |
|--------|---------|
| `collision_any` | 1.0 if the ego vehicle collided with anything during the rollout |
| `offroad` | 1.0 if the ego vehicle left the drivable area |
| `progress` | Fraction of the route completed (0.0–1.0) |
| `dist_traveled` | Total metres driven before failure or completion |
| `ADE` | Average Displacement Error — mean distance (metres) between predicted and teacher waypoints across all timesteps |
| `FDE` | Final Displacement Error — distance at the last waypoint only |
| `val ADE` | ADE measured on the held-out 10% validation split |

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

The key insight is step 3b — even though the **student** is driving and making
mistakes, the **teacher** labels every state the student visits. Over iterations,
the training distribution shifts to match the student's actual deployment
distribution, eliminating covariate shift.

In this project:
- **π\*** = Alpamayo 1.5 (the teacher)
- **π_i** = StudentNet checkpoint `student_iter_i.pth`
- **States** = 4-camera image + speed/acceleration
- **Actions** = 10-waypoint trajectory (1 second horizon)

---

### Imitation Learning (Behavioural Cloning)
The base form of learning from demonstrations. The student is trained as a
supervised regression problem:

```
L = (1/T) Σ_t ||π_student(s_t) - π_teacher(s_t)||²
```

Where `s_t` is the observation at time t and the output is a trajectory.
No reward signal is needed — only expert demonstrations. The limitation (fixed
by DAgger) is that errors compound: a small deviation puts the student in a
state the teacher never demonstrated, leading to larger deviations.

---

### ResNet18 (Visual Encoder)
A residual convolutional neural network with 18 layers. "Residual" means each
block learns a residual `F(x) = H(x) - x` rather than the full mapping `H(x)`,
with the input added back: `output = F(x) + x`. This makes gradients flow more
easily through deep networks, solving the vanishing gradient problem.

Used here as a frozen-weight-free feature extractor: each of the 4 camera images
is passed through ResNet18 to produce a 512-dimensional feature vector.

---

### AdamW (Optimiser)
Adam with **decoupled weight decay**. Standard Adam applies weight decay inside
the gradient update (L2 regularisation), which interacts badly with the adaptive
learning rate. AdamW applies weight decay directly to the weights, independent
of the gradient — this gives cleaner regularisation and better generalisation,
especially on small datasets like ours.

```
θ_t = θ_{t-1} - α * (m̂_t / (√v̂_t + ε)) - α * λ * θ_{t-1}
```
Where `m̂_t` and `v̂_t` are bias-corrected first and second moment estimates,
and `λ` is the weight decay coefficient.

---

### OneCycleLR (Learning Rate Schedule)
Starts with a low learning rate, ramps up to a peak over ~10% of training
(warmup), then decays back down via cosine annealing. Proposed by Smith &
Topin (2018) as "Super-Convergence". Benefits:
- The warmup phase prevents early instability on a new dataset
- The high peak LR helps escape shallow local minima
- The cosine decay fine-tunes the solution at the end

In this project: `lr_min → lr_max (3e-4) → lr_final` over 30 epochs.

---

### ADE / FDE (Trajectory Error Metrics)
Standard evaluation metrics for trajectory prediction:

- **ADE** (Average Displacement Error): mean Euclidean distance between the
  predicted waypoints and the ground-truth (teacher) waypoints, averaged over
  all T timesteps in the horizon:
  ```
  ADE = (1/T) Σ_t ||ŷ_t - y_t||₂
  ```

- **FDE** (Final Displacement Error): Euclidean distance at only the last
  waypoint, measuring how well the model captures the long-term intent:
  ```
  FDE = ||ŷ_T - y_T||₂
  ```

Lower is better for both. The student reached val ADE ~0.029m after 6 DAgger
iterations (compared to the raw prediction at iter 0 before any training).

---

### APF (Artificial Potential Field)
A classical path planning technique where the ego vehicle is treated as a
particle in a potential field. Obstacles create **repulsive** potentials and
the goal creates an **attractive** potential. The vehicle follows the gradient
of the combined field.

Used in `cost_filter.py` for the safety cost component of MPC refinement:
```
U_rep(d) = 0.5 * k_rep * (1/d - 1/d₀)²   if d < d₀
           0                                otherwise
```
Where `d` is distance to the nearest obstacle and `d₀` is the influence radius.
