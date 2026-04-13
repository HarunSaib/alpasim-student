# alpasim-student

DAgger imitation learning plugin for [NVlabs/alpasim](https://github.com/NVlabs/alpasim).

Trains a StudentNet model to imitate the Alpamayo 1.5 teacher using the DAgger algorithm.

## Structure

```
alpasim_student/          ← Python plugin (install into alpasim)
  dagger/
    loop.py               ← DAgger orchestration loop
    trainer.py            ← StudentNet training (AdamW + OneCycleLR)
    collector.py          ← Collects training data from AlpaSim rollouts
  student_model.py        ← StudentNet architecture (CNN + MLP)
  configs/driver/
    student.yaml          ← Driver config for student in simulation
    student_camera_configs.yaml  ← 4-camera simulation spec
topology/
  2gpu_alpamayo.yaml      ← AlpaSim wizard topology config
```

## Setup

Place `alpasim_student/` inside `plugins/` in your alpasim repo, and `topology/2gpu_alpamayo.yaml` inside `src/wizard/configs/topology/`.

## Running DAgger

```bash
uv run --extra all --no-sync python -m alpasim_student.dagger.loop \
    --base-dir ./dagger_run \
    --iterations 7 \
    --epochs 30
```
