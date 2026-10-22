# alpasim-student

DAgger imitation learning plugin for [NVlabs/alpasim](https://github.com/NVlabs/alpasim).

Trains camera-based student models (StudentNet, AlpaNet) to imitate the Alpamayo 1.5 teacher in AlpaSim.
The simulator is credited to NVIDIA and pinned as a submodule; this repository contains only the plugin.

## Structure

```
alpasim_student/              ← deployable plugin (copied into <alpasim>/plugins/ by sync.sh)
  pyproject.toml
  alpasim_student/
    paths.py                  ← finds the AlpaSim checkout (ALPASIM_ROOT, deployed location, or submodule)
    dagger/                   ← DAgger loop, collector, StudentNet and AlpaNet trainers
    alpanet/                  ← AlpaNet: ResNet34 + BEV lifter + multi-hypothesis head
    grpo/                     ← GRPO prototype (unfinished)
    student_model.py          ← StudentNet driver plugin
    configs/driver/           ← driver configs for the simulator
topology/2gpu_alpamayo.yaml   ← GPU placement: driver on GPU 0, renderer on GPU 1
scripts/                      ← rollout visualizer (Rerun), BEV geometry check
third_party/alpasim/          ← NVlabs/alpasim, pinned (submodule)
sync.sh                       ← push/pull, deploy plugin + topology + scripts, uv sync
```

## Simulator pin

`third_party/alpasim` is pinned to NVlabs/alpasim `04a17b23` (2026-04-09), the version the first
runs used. The runtime `src/` tree is that commit plus `topology/2gpu_alpamayo.yaml`.

```bash
git clone --recurse-submodules git@github.com:HarunSaib/alpasim-student.git
```

## Setup

`sync.sh` deploys into `$ALPASIM_ROOT` (default `/home/harun/alpasim`):

```bash
./sync.sh
```

## GPUs

Placement comes from `topology/2gpu_alpamayo.yaml` (driver on GPU 0, sensorsim on GPU 1) and is not
rewritten by the loop. Before each simulation phase the loop prints the effective placement and
refuses to start if a required GPU has more than 2 GiB in use. Set `ALPASIM_SKIP_GPU_CHECK=1` to
override. If the driver and renderer are ever placed on the same GPU, the renderer cache and worker
count are reduced automatically.

## Running DAgger

```bash
cd /home/harun/alpasim
nohup uv run --extra all --no-sync python -m alpasim_student.dagger.loop \
    --base-dir ./dagger_run_v2 --model alpanet \
    > /tmp/dagger_loop_v2.log 2>&1 &
```

Defaults: 20 iterations, 60 epochs, all 8 scenes with the curriculum expanding 2 → 4 → 6 → 8.

## Key design choices

- **40 waypoints (4 s at 10 Hz)** — taken from the teacher's 65-pose output
- **Curriculum learning** — the scene set expands every few iterations
- **Weighted sampling** — newer iterations weighted up to 2x older ones
- **Stagnation early stopping** — exits if pass rate improvement < 2% over 4 iterations
- **Best checkpoint inference** — uses `_best.pth` (lowest validation ADE) for student runs

## Resuming a run

```bash
uv run --extra all --no-sync python -m alpasim_student.dagger.loop \
    --base-dir ./dagger_run_v2 --model alpanet \
    --start-iteration 10 \
    --initial-checkpoint ./dagger_run_v2/checkpoints/alpanet_iter_10_best.pth
```
