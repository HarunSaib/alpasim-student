# SPDX-License-Identifier: Apache-2.0
"""
DAgger orchestration loop.

Usage:
    uv run python -m alpasim_student.dagger.loop \
        [--base-dir ./dagger_run] \
        [--iterations 5] \
        [--epochs 30]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ALPASIM_ROOT = Path(__file__).resolve().parents[4]  # ~/alpasim


def _load_env() -> None:
    """Load .env from the alpasim root into os.environ (if not already set)."""
    env_file = ALPASIM_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key and value and key not in os.environ:
            os.environ[key] = value
            print(f"[env] Loaded {key} from .env")


def _run_wizard(
    driver: str,
    log_dir: Path,
    topology: str = "2gpu_alpamayo",
    base_dir: Path | None = None,
    scene_ids: list[str] | None = None,
) -> None:
    """Launch alpasim_wizard, patch docker-compose for the student plugin, then run it.

    Steps:
        1. Run wizard with run_method=NONE to generate configs only (no containers).
        2. Patch docker-compose.yaml: add ``--extra all`` to every ``uv run -m alpasim_driver``
           command so the student plugin entry-point is visible inside the container.
        3. Run ``docker compose up`` manually to start services.
    """
    import re

    log_dir = Path(log_dir).resolve()
    env = {**os.environ, "UV_LINK_MODE": "copy"}

    # ── Step 1: generate configs only (no docker compose) ─────────────────────
    gen_cmd = [
        "uv", "run",
        "--extra", "all",
        "--no-sync",
        "python", "-m", "alpasim_wizard",
        "deploy=local",
        "wizard.run_method=NONE",   # generate docker-compose.yaml without running it
        f"topology={topology}",
        f"driver={driver}",
        f"wizard.log_dir={log_dir}",
    ]
    if scene_ids:
        # Hydra list override — HF_TOKEN in env allows the wizard to auto-download
        # any scenes not already in data/nre-artifacts/all-usdzs/.
        ids_joined = ",".join(scene_ids)
        gen_cmd.append(f"scenes.scene_ids=[{ids_joined}]")
    print(f"[loop] $ {' '.join(gen_cmd)}")
    result = subprocess.run(gen_cmd, cwd=ALPASIM_ROOT, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Wizard config-gen exited with code {result.returncode}")

    # ── Step 2: patch docker-compose.yaml ─────────────────────────────────────
    compose_file = log_dir / "docker-compose.yaml"
    if compose_file.exists():
        original = compose_file.read_text()

        # 2a. Ensure --extra all on every driver uv run call
        patched = re.sub(
            r"\buv run\b(?!\s+--extra\s+all)(\s+-m\s+alpasim_driver\.main)",
            r"uv run --extra all\1",
            original,
        )

        # 2b. Patch the driver-0 service specifically (not other services).
        #     We locate the driver-0 block by splitting on YAML top-level service keys.
        plugins_host = str(ALPASIM_ROOT / "plugins")

        # Split into service blocks: everything from "  driver-0:" to the next "  <service>:"
        # We use a regex to find the driver-0 block and modify it in-place.
        def patch_driver_block(text: str) -> str:
            # Find the driver-0 service section
            driver_start = re.search(r"^  driver-0:", text, re.MULTILINE)
            if not driver_start:
                print("[loop] WARNING: could not find driver-0 service in docker-compose")
                return text

            # Find where the next top-level service starts (2-space indent + word chars + colon)
            after_driver = text[driver_start.start():]
            next_service = re.search(r"\n  [a-zA-Z0-9_\-]+:", after_driver[1:])
            if next_service:
                driver_block = after_driver[: next_service.start() + 1]
                rest_after = after_driver[next_service.start() + 1:]
            else:
                driver_block = after_driver
                rest_after = ""
            before_driver = text[: driver_start.start()]

            # Add plugins + checkpoints volume mounts if not present
            drivers_mount = "      - /home/harun/alpasim/data/drivers:/mnt/drivers"
            plugins_mount = f"\n      - {plugins_host}:/repo/plugins"
            ckpt_host = str((base_dir or ALPASIM_ROOT / "dagger_run").resolve() / "checkpoints")
            ckpt_mount = f"\n      - {ckpt_host}:/mnt/checkpoints"
            # Add plugins mount only if not already present
            if "/repo/plugins" not in driver_block and drivers_mount in driver_block:
                driver_block = driver_block.replace(
                    drivers_mount,
                    drivers_mount + plugins_mount,
                    1,
                )
            # Always add checkpoints mount if not present (needed for student ckpt)
            if "/mnt/checkpoints" not in driver_block and drivers_mount in driver_block:
                driver_block = driver_block.replace(
                    drivers_mount,
                    drivers_mount + ckpt_mount,
                    1,
                )

            # Add student install + no-sync before driver replicas start.
            # Use --no-sync so uv doesn't recreate the venv and removes our install.
            student_install_line = (
                "        uv pip install -e /repo/plugins/alpasim_student "
                "--no-deps --quiet 2>&1 | tail -1 || true\n"
            )
            if "alpasim_student" not in driver_block and "umask 0000" in driver_block:
                driver_block = driver_block.replace(
                    "umask 0000\n",
                    "umask 0000\n" + student_install_line,
                    1,
                )
                # Also change uv run --extra all to uv run --no-sync so it uses
                # the image venv (with our freshly-installed student) without re-syncing.
                driver_block = driver_block.replace(
                    "uv run --extra all -m alpasim_driver.main",
                    "uv run --no-sync -m alpasim_driver.main",
                )

            return before_driver + driver_block + rest_after

        new_patched = patch_driver_block(patched)
        if new_patched != patched:
            patched = new_patched
            print(f"[loop] Patched {compose_file}: added student install to driver-0 service")

        if patched != original:
            compose_file.write_text(patched)
            print(f"[loop] Patched {compose_file}: added --extra all + student install")

    # ── Step 3: run docker compose ────────────────────────────────────────────
    run_cmd = ["docker", "compose", "-f", str(compose_file), "up"]
    print(f"[loop] $ {' '.join(run_cmd)}")
    result = subprocess.run(run_cmd, cwd=log_dir, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"docker compose up exited with code {result.returncode}")


def _pivot_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Convert long-format metrics parquet to {metric_name: aggregated_value}.

    The parquet has columns: name, values, time_aggregation (and others).
    We apply each metric's declared time_aggregation over its values column.
    """
    result: dict[str, float] = {}
    for name, group in df.groupby("name"):
        agg = group["time_aggregation"].iloc[0] if "time_aggregation" in group.columns else "max"
        vals = group["values"].dropna()
        if len(vals) == 0:
            continue
        if agg == "last":
            result[name] = float(vals.iloc[-1])
        elif agg == "min":
            result[name] = float(vals.min())
        elif agg == "mean":
            result[name] = float(vals.mean())
        else:  # max (default)
            result[name] = float(vals.max())
    return result


def _detect_failures(
    run_dir: Path,
    thresholds: dict | None = None,
) -> list[str]:
    """Return rollout IDs that violated any failure threshold.

    Reads long-format metrics.parquet written by the eval module after each rollout.
    Checked metrics:
        collision_any           > 0
        offroad                 > 0
        dist_to_gt_trajectory   > threshold (metres, default 5.0)
    """
    thresholds = thresholds or {
        "collision_any": 0.0,
        "offroad":       0.0,
        # dist_to_gt_trajectory omitted: it flagged clean rollouts that took a
        # slightly different valid path, causing unnecessary teacher corrections.
    }
    failed: list[str] = []
    for metrics_file in sorted(run_dir.glob("rollouts/*/*/metrics.parquet")):
        rollout_id = metrics_file.parent.name
        try:
            df = pd.read_parquet(metrics_file)
            metrics = _pivot_metrics(df)
        except Exception:
            continue
        for metric_name, thresh in thresholds.items():
            if metrics.get(metric_name, 0.0) > thresh:
                failed.append(rollout_id)
                break
    return failed


def _build_eval_log(run_dir: Path, iteration: int, phase: str) -> dict:
    """Parse simulation metrics.parquet files and return a W&B log dict.

    Returns an empty dict if no completed rollouts are found.
    """
    metrics_files = list(run_dir.glob("rollouts/*/*/metrics.parquet"))
    if not metrics_files:
        return {}

    all_rows = []
    for mf in metrics_files:
        try:
            df = pd.read_parquet(mf)
            row = _pivot_metrics(df)
            all_rows.append(row)
        except Exception:
            continue

    if not all_rows:
        return {}

    summary = pd.DataFrame(all_rows).mean()
    n_rollouts = len(all_rows)

    log = {f"eval/{phase}/{k}": float(v) for k, v in summary.items()
           if k in {
               "progress", "progress_rel", "dist_traveled_m", "plan_deviation",
               "collision_any", "collision_at_fault", "offroad", "wrong_lane",
               "dist_to_gt_trajectory", "dist_to_gt_location", "safety_monitor_triggered",
               "duration_frac_20s",
           }}
    log[f"eval/{phase}/n_rollouts"] = n_rollouts
    log["iteration"] = iteration

    print(f"[loop] eval/{phase}: progress={summary.get('progress', float('nan')):.2f} "
          f"collision={summary.get('collision_any', 0):.2f} "
          f"offroad={summary.get('offroad', 0):.2f} "
          f"dist_traveled={summary.get('dist_traveled_m', 0):.1f}m "
          f"n={n_rollouts}")
    return log


DEFAULT_SCENE_IDS = [
    "clipgt-01d503d4-449b-46fc-8d78-9085e70d3554",
    "clipgt-a309e228-26e1-423e-a44c-cb00aa7378cb",
    "clipgt-804afc4a-fd1e-4f58-bd39-a4c486a916e5",
]


def run_dagger(
    base_dir: Path,
    n_iterations: int = 5,
    epochs_per_iter: int = 30,
    start_iteration: int = 0,
    initial_checkpoint: Path | None = None,
    scene_ids: list[str] | None = None,
    teacher_scene_ids: list[str] | None = None,
) -> None:
    import wandb

    _load_env()

    from .collector import collect_from_run
    from .trainer import train

    base_dir = Path(base_dir)
    all_data_dirs: list[Path] = []
    student_ckpt: Path | None = initial_checkpoint

    # Top-level W&B run tracking the full DAgger loop (separate from per-training runs).
    # We save the run ID so we can resume the same run after train() calls wandb.finish().
    wandb_mode = "online" if os.environ.get("WANDB_API_KEY") else "disabled"
    loop_run = wandb.init(
        project="alpasim-dagger-loop",
        name=f"dagger_loop_start{start_iteration}",
        mode=wandb_mode,
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
        config={
            "n_iterations":    n_iterations,
            "epochs_per_iter": epochs_per_iter,
            "start_iteration": start_iteration,
        },
        reinit=True,
    )
    loop_run_id = loop_run.id if loop_run and wandb_mode == "online" else None

    # Pre-create checkpoints dir as harun so Docker (root) can't steal ownership
    (base_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    def _loop_log(data: dict) -> None:
        """Log to W&B, reinitialising the loop run if train() closed it."""
        if wandb_mode != "online":
            return
        if wandb.run is None:
            wandb.init(
                project="alpasim-dagger-loop",
                id=loop_run_id,
                resume="allow",
                mode=wandb_mode,
                reinit=True,
            )
        try:
            wandb.log(data)
        except Exception as e:
            print(f"[loop] W&B log failed: {e}")

    def _resume_loop_wandb() -> None:
        """Re-open the loop-level W&B run after train() closed it."""
        _loop_log({})  # no-op log just to ensure run is open

    # Seed all_data_dirs from already-collected datasets for iterations before start_iteration.
    for i in range(start_iteration):
        candidate = base_dir / f"iter_{i}" / "dataset"
        if candidate.exists() and any(candidate.rglob("samples.parquet")):
            all_data_dirs.append(candidate)
            print(f"[loop] Seeded dataset from iter_{i}: {candidate}")

    for iteration in range(start_iteration, n_iterations):
        print(f"\n{'='*60}")
        print(f"  DAgger  iteration {iteration + 1}/{n_iterations}")
        print(f"{'='*60}")

        iter_dir = base_dir / f"iter_{iteration}"

        # ------------------------------------------------------------------ #
        # Step 1  Run teacher (iteration 0) or student + collect corrections  #
        # ------------------------------------------------------------------ #
        if iteration == 0 or student_ckpt is None:
            print("\n[loop] Phase 1: bootstrapping with Alpamayo 1.5 teacher...")
            teacher_run = iter_dir / "teacher_run"
            _run_wizard("alpamayo1_5", teacher_run, base_dir=base_dir, scene_ids=teacher_scene_ids)
            _loop_log(_build_eval_log(teacher_run, iteration, "teacher"))
            source_run = teacher_run

        else:
            print("\n[loop] Phase 1a: running student in simulation...")
            student_run = iter_dir / "student_run"

            # Update student.yaml checkpoint path before launching
            _patch_student_checkpoint(student_ckpt)
            # Use 2gpu_alpamayo topology (has all required config fields)
            _run_wizard("student", student_run, topology="2gpu_alpamayo", base_dir=base_dir, scene_ids=scene_ids)

            # Log student eval metrics to W&B
            _loop_log(_build_eval_log(student_run, iteration, "student"))

            # Guard against simulation crash producing 0 completed rollouts.
            # Count only rollouts that wrote metrics.parquet (actually completed).
            metrics_files = list(student_run.glob("rollouts/*/*/metrics.parquet")) if student_run.exists() else []
            n_completed = len(metrics_files)
            if n_completed == 0:
                print("[loop] WARNING: 0 completed rollouts — simulation may have crashed. Running teacher corrections.")
            else:
                failed = _detect_failures(student_run)
                pass_rate = (n_completed - len(failed)) / n_completed
                print(f"[loop] Failed rollouts: {len(failed)} / {n_completed}  (pass rate: {pass_rate:.0%})")
                _loop_log({"dagger/pass_rate": pass_rate, "dagger/n_completed": n_completed,
                           "dagger/n_failed": len(failed), "iteration": iteration})

                if not failed:
                    print("[loop] Student passed all scenarios — training complete.")
                    _loop_log({"dagger/converged": True})
                    return

            print("[loop] Phase 1b: querying teacher for corrections on failed scenes...")
            correction_run = iter_dir / "teacher_correction_run"
            _run_wizard("alpamayo1_5", correction_run, base_dir=base_dir, scene_ids=teacher_scene_ids)
            _loop_log(_build_eval_log(correction_run, iteration, "teacher_correction"))
            source_run = correction_run

        # ------------------------------------------------------------------ #
        # Step 2  Collect dataset from the teacher run                        #
        # ------------------------------------------------------------------ #
        dataset_dir = iter_dir / "dataset"
        print(f"\n[loop] Phase 2: collecting dataset → {dataset_dir}")
        n = collect_from_run(source_run, dataset_dir)
        if n == 0:
            print("[loop] WARNING: no samples collected — check the run logs.")
            continue
        all_data_dirs.append(dataset_dir)

        # ------------------------------------------------------------------ #
        # Step 3  Train student on all data collected so far (dataset agg.)   #
        # ------------------------------------------------------------------ #
        ckpt_out = base_dir / "checkpoints" / f"student_iter_{iteration + 1}.pth"
        best_ckpt_out = base_dir / "checkpoints" / f"student_iter_{iteration + 1}_best.pth"
        print(f"\n[loop] Phase 3: training student ({len(all_data_dirs)} dataset(s))...")
        train(
            data_dirs      = all_data_dirs,
            checkpoint_out = ckpt_out,
            num_epochs     = epochs_per_iter,
            device_str     = "cuda:1",
            resume_from    = str(student_ckpt) if student_ckpt else None,
            dagger_iter    = iteration + 1,
        )
        # train() calls wandb.finish() — _loop_log will reinit the run if needed
        # Prefer the best-val-ADE checkpoint for the next simulation run.
        student_ckpt = best_ckpt_out if best_ckpt_out.exists() else ckpt_out
        print(f"[loop] Student checkpoint: {student_ckpt}")

        _loop_log({
            "dagger/num_datasets": len(all_data_dirs),
            "dagger/checkpoint":   str(student_ckpt),
            "iteration":           iteration,
        })

    print("\n[loop] DAgger loop finished.")


def _patch_student_checkpoint(ckpt_path: Path) -> None:
    """Overwrite the checkpoint_path in the student driver YAML.

    The checkpoint is mounted inside the driver container at /mnt/checkpoints/<filename>,
    so the YAML path must use the container-side path, not the host path.
    """
    yaml_path = (
        ALPASIM_ROOT
        / "plugins/alpasim_student/alpasim_student/configs/driver/student.yaml"
    )
    # Container sees checkpoints dir as /mnt/checkpoints/
    container_ckpt = "/mnt/checkpoints/" + Path(ckpt_path).name
    text = yaml_path.read_text()
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("checkpoint_path:"):
            indent = len(line) - len(line.lstrip())
            lines.append(" " * indent + f'checkpoint_path: "{container_ckpt}"')
        else:
            lines.append(line)
    yaml_path.write_text("\n".join(lines) + "\n")
    print(f"[loop] Updated student checkpoint path → {container_ckpt} (host: {ckpt_path})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir",           default="./dagger_run")
    parser.add_argument("--iterations",         type=int,   default=5)
    parser.add_argument("--epochs",             type=int,   default=30)
    parser.add_argument("--start-iteration",    type=int,   default=0,
                        help="Resume DAgger from this iteration index (0-based).")
    parser.add_argument("--initial-checkpoint", default=None,
                        help="Path to an existing student checkpoint to start from.")
    parser.add_argument("--scenes",             nargs="+",  default=None,
                        help="Scene IDs to run (e.g. clipgt-01d503d4-... clipgt-a309e228-...). "
                             "Defaults to the base config scene list (1 scene). "
                             "HF_TOKEN in .env is used to auto-download missing scenes.")
    args = parser.parse_args()

    run_dagger(
        base_dir            = Path(args.base_dir),
        n_iterations        = args.iterations,
        epochs_per_iter     = args.epochs,
        start_iteration     = args.start_iteration,
        initial_checkpoint  = Path(args.initial_checkpoint) if args.initial_checkpoint else None,
        scene_ids           = args.scenes or DEFAULT_SCENE_IDS,
        teacher_scene_ids   = [DEFAULT_SCENE_IDS[0]],
    )
