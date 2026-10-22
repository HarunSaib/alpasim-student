# SPDX-License-Identifier: Apache-2.0
"""Failure detection for simulation runs: exit codes, out-of-memory evidence and GPU peaks."""
from __future__ import annotations

import json
import re
import subprocess
import threading
from collections.abc import Callable

_OOM = re.compile(r"out of memory|OutOfMemoryError|cudaErrorMemoryAllocation", re.IGNORECASE)
_LOW_FREE_MIB = 512


class SimulationFailed(RuntimeError):
    """A simulation run ended without producing trustworthy results."""


class GpuMemoryExceeded(SimulationFailed):
    """The run failed and its logs show the GPU ran out of memory."""


def log_shows_oom(text: str) -> bool:
    return bool(_OOM.search(text))


def parse_service_states(ps_output: str) -> dict[str, tuple[str, int]]:
    """Parse `docker compose ps -a --format json` (one object per line, or a JSON array)."""
    text = ps_output.strip()
    if not text:
        return {}
    rows = json.loads(text) if text.startswith("[") else [json.loads(l) for l in text.splitlines()]
    return {r["Service"]: (r.get("State", ""), int(r.get("ExitCode", 0))) for r in rows}


def failed_runtime(states: dict[str, tuple[str, int]]) -> list[tuple[str, int]]:
    """Runtime services that exited non-zero. Other services may be stopped by compose on shutdown,
    so their exit codes are not treated as failures."""
    return [(s, code) for s, (_, code) in sorted(states.items()) if s.startswith("runtime") and code != 0]


def compose_states(compose_file, env) -> dict[str, tuple[str, int]]:
    out = subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "ps", "-a", "--format", "json"],
        capture_output=True, text=True, env=env, check=False,
    ).stdout
    return parse_service_states(out)


def read_gpu_memory() -> dict[str, tuple[int, int]]:
    """{gpu index: (used MiB, total MiB)} from nvidia-smi."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    rows = (line.split(",") for line in out.strip().splitlines())
    return {i.strip(): (int(u), int(t)) for i, u, t in rows}


class GpuPeakSampler:
    """Polls GPU memory while a run is in progress.

    Records the first reading (what other processes were already using), the peak, and the lowest
    free memory seen, and prints a warning when a GPU gets within 512 MiB of full. It only observes:
    a PyTorch process may legitimately cache most of a GPU, so low free memory alone is not a failure.
    """

    def __init__(self, gpu_ids, interval_s: float = 5.0,
                 read: Callable[[], dict[str, tuple[int, int]]] = read_gpu_memory):
        self._ids, self._interval, self._read = set(gpu_ids), interval_s, read
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._first: dict[str, int] = {}
        self._peak: dict[str, int] = {}
        self._total: dict[str, int] = {}
        self._min_free: dict[str, int] = {}
        self._warned: set[str] = set()
        self.samples = 0

    def _sample(self) -> None:
        try:
            memory = self._read()
        except Exception:
            return
        self.samples += 1
        for idx in self._ids & memory.keys():
            used, total = memory[idx]
            self._first.setdefault(idx, used)
            self._peak[idx] = max(self._peak.get(idx, 0), used)
            self._total[idx] = total
            self._min_free[idx] = min(self._min_free.get(idx, total), total - used)
            if total - used < _LOW_FREE_MIB and idx not in self._warned:
                self._warned.add(idx)
                print(f"[guard] GPU {idx} has only {total - used} MiB free", flush=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self._interval)

    def start(self) -> "GpuPeakSampler":
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self._interval + 5)
        self._sample()
        return {
            "samples": self.samples,
            "first_used_mib": self._first,
            "peak_used_mib": self._peak,
            "min_free_mib": self._min_free,
            "total_mib": self._total,
        }
