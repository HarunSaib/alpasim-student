# SPDX-License-Identifier: Apache-2.0
"""Failure detection: exit codes, OOM evidence, GPU peak sampling, and the guarded run."""
import json
import subprocess
from pathlib import Path

import pytest

from alpasim_student.dagger import loop
from alpasim_student.guard import (
    GpuMemoryExceeded,
    GpuPeakSampler,
    SimulationFailed,
    failed_runtime,
    log_shows_oom,
    parse_service_states,
)

OOM_LINE = "runtime-0-1  | torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 56.00 MiB. GPU 0 has 112MB free"


def test_parse_service_states_handles_lines_array_and_empty():
    row = lambda s, c: {"Service": s, "State": "exited", "ExitCode": c}
    lines = "\n".join(json.dumps(r) for r in (row("runtime-0", 1), row("driver-0", 0)))
    assert parse_service_states(lines) == {"runtime-0": ("exited", 1), "driver-0": ("exited", 0)}
    assert parse_service_states(json.dumps([row("runtime-0", 0)])) == {"runtime-0": ("exited", 0)}
    assert parse_service_states("") == {}


def test_only_runtime_exit_codes_count_as_failures():
    states = {"runtime-0": ("exited", 1), "sensorsim-0": ("exited", 137), "driver-0": ("exited", 143)}
    assert failed_runtime(states) == [("runtime-0", 1)]
    assert failed_runtime({"runtime-0": ("exited", 0), "driver-0": ("exited", 137)}) == []


def test_oom_detection():
    assert log_shows_oom(OOM_LINE)
    assert log_shows_oom("RuntimeError: CUDA error: out of memory")
    assert not log_shows_oom("rollout finished; collision_any=0")


def test_peak_sampler_records_first_peak_and_min_free(capsys):
    readings = iter([{"0": (1000, 32000)}, {"0": (20000, 32000)}, {"0": (31800, 32000)}, {"0": (5000, 32000)}])
    sampler = GpuPeakSampler({"0"}, interval_s=3600, read=lambda: next(readings))
    sampler._sample(); sampler._sample(); sampler._sample()
    summary = sampler.stop()          # one more sample on stop
    assert summary["first_used_mib"] == {"0": 1000}
    assert summary["peak_used_mib"] == {"0": 31800}
    assert summary["min_free_mib"] == {"0": 200}
    assert summary["samples"] == 4
    assert "only 200 MiB free" in capsys.readouterr().out


class _FakeCompose:
    """Stands in for subprocess.run during a guarded simulation run."""

    def __init__(self, runtime_exit: int, logs: str, up_code: int = 0):
        self.runtime_exit, self.logs, self.up_code, self.calls = runtime_exit, logs, up_code, []

    def __call__(self, cmd, **kwargs):
        verb = cmd[cmd.index("-f") + 2] if "-f" in cmd else cmd[0]
        self.calls.append(verb)
        out = ""
        if verb == "ps":
            rows = [{"Service": "runtime-0", "State": "exited", "ExitCode": self.runtime_exit},
                    {"Service": "sensorsim-0", "State": "exited", "ExitCode": 137}]
            out = "\n".join(json.dumps(r) for r in rows)
        elif verb == "logs":
            kwargs["stdout"].write(self.logs)
        code = self.up_code if verb == "up" else 0
        return subprocess.CompletedProcess(cmd, code, stdout=out, stderr="")


def _prepare(monkeypatch, fake):
    def fake_generate(driver, log_dir, topology, base_dir, scene_ids):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        compose = Path(log_dir) / "docker-compose.yaml"
        compose.write_text("services:\n  driver-0:\n    deploy:\n      resources:\n"
                           "        reservations:\n          devices:\n            - driver: nvidia\n"
                           "              device_ids:\n                - '0'\n")
        return compose

    monkeypatch.setattr(loop, "_generate_compose", fake_generate)
    monkeypatch.setattr(loop, "_check_gpus_free", lambda ids: None)
    monkeypatch.setattr(loop, "GpuPeakSampler",
                        lambda ids: GpuPeakSampler(ids, interval_s=0.01, read=lambda: {"0": (100, 1000)}))
    monkeypatch.setattr(subprocess, "run", fake)


def test_runtime_oom_is_reported_with_logs_and_peaks_kept(tmp_path, monkeypatch):
    fake = _FakeCompose(runtime_exit=1, logs=OOM_LINE)
    _prepare(monkeypatch, fake)
    with pytest.raises(GpuMemoryExceeded) as err:
        loop._run_wizard("alpanet", tmp_path / "run")
    assert "compose.log" in str(err.value)
    assert OOM_LINE in (tmp_path / "run" / "compose.log").read_text()
    assert json.loads((tmp_path / "run" / "gpu_peak.json").read_text())["peak_used_mib"] == {"0": 100}
    assert fake.calls.index("logs") < fake.calls.index("down")    # logs saved before containers removed


def test_runtime_failure_without_oom_is_a_plain_simulation_failure(tmp_path, monkeypatch):
    _prepare(monkeypatch, _FakeCompose(runtime_exit=2, logs="Traceback: something else"))
    with pytest.raises(SimulationFailed) as err:
        loop._run_wizard("alpanet", tmp_path / "run")
    assert not isinstance(err.value, GpuMemoryExceeded)


def test_successful_run_passes_and_still_saves_logs(tmp_path, monkeypatch):
    fake = _FakeCompose(runtime_exit=0, logs="all rollouts complete")
    _prepare(monkeypatch, fake)
    loop._run_wizard("alpanet", tmp_path / "run")
    assert (tmp_path / "run" / "compose.log").read_text() == "all rollouts complete"
    assert fake.calls[-1] == "down"
