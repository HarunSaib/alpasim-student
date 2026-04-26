# alpasim-student

DAgger imitation learning plugin for [NVlabs/alpasim](https://github.com/NVlabs/alpasim).

Trains a StudentNet model to imitate the Alpamayo 1.5 teacher using the DAgger algorithm.

## Structure

```
alpasim_student/          ← Python plugin (install into alpasim)
  dagger/
    loop.py               ← DAgger orchestration loop (curriculum, early stopping)
    trainer.py            ← StudentNet training (AdamW + OneCycleLR + weighted sampling)
    collector.py          ← Collects training data from AlpaSim rollouts
  student_model.py        ← StudentNet architecture (ResNet18 × 4 + MLP, 25 waypoints)
  configs/driver/
    student.yaml          ← Driver config for student in simulation
    student_camera_configs.yaml  ← 4-camera simulation spec
topology/
  2gpu_alpamayo.yaml      ← AlpaSim wizard topology config
sync.sh                   ← Push to GitHub + copy into alpasim + uv sync
```

## Setup

Place `alpasim_student/` inside `plugins/` in your alpasim repo, and `topology/2gpu_alpamayo.yaml` inside `src/wizard/configs/topology/`. Run `sync.sh` to do this automatically.

```bash
cd /home/harun/alpasim-student
./sync.sh
```

## Running DAgger

```bash
cd /home/harun/alpasim
nohup uv run --extra all --no-sync python -m alpasim_student.dagger.loop \
    --base-dir ./dagger_run_v2 \
    > /tmp/dagger_loop_v2.log 2>&1 &
```

Defaults: 20 iterations, 60 epochs. Monitor with:

```bash
tail -f /tmp/dagger_loop_v2.log
```

## Key Design Decisions

- **25 waypoints (2.5s horizon)** — teacher provides 65 poses per step; longer horizon allows the model to anticipate turns
- **Curriculum learning** — starts with 2 scenes, expands to 5 over 16 iterations
- **Weighted sampling** — recent iterations weighted up to 2× older ones to prevent hard scenes being diluted
- **Stagnation early stopping** — exits if pass rate improvement < 2% over 4 iters (only after first non-zero pass rate)
- **Best checkpoint inference** — uses `_best.pth` (lowest val ADE) for student sim runs, not final epoch

## Resuming a Run

```bash
uv run --extra all --no-sync python -m alpasim_student.dagger.loop \
    --base-dir ./dagger_run_v2 \
    --start-iteration 10 \
    --initial-checkpoint ./dagger_run_v2/checkpoints/student_iter_10_best.pth
```
