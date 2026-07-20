# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NVIDIA GPU and process snapshots for benchmark provenance."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from subprocess import CompletedProcess


@dataclass(frozen=True)
class GpuDeviceSnapshot:
    """One NVIDIA device resource sample."""

    index: int
    name: str
    memory_total_mib: int
    memory_used_mib: int
    memory_free_mib: int
    utilization_pct: int
    temperature_c: int
    sm_clock_mhz: int


@dataclass(frozen=True)
class GpuProcessSnapshot:
    """One process visible to NVIDIA management tooling."""

    gpu_uuid: str
    pid: int
    name: str
    memory_mib: int


@dataclass(frozen=True)
class GpuSnapshot:
    """Timestamped device and process resource sample."""

    timestamp: str
    devices: tuple[GpuDeviceSnapshot, ...]
    processes: tuple[GpuProcessSnapshot, ...]


def _rows(text: str, width: int) -> list[list[str]]:
    rows = [[value.strip() for value in line.split(",")] for line in text.splitlines() if line.strip()]
    if any(len(row) != width for row in rows):
        raise ValueError(f"expected {width} NVIDIA CSV columns")
    return rows


def parse_gpu_rows(text: str) -> tuple[GpuDeviceSnapshot, ...]:
    """Parse no-header, no-unit NVIDIA device CSV rows."""
    return tuple(
        GpuDeviceSnapshot(
            index=int(row[0]),
            name=row[1],
            memory_total_mib=int(row[2]),
            memory_used_mib=int(row[3]),
            memory_free_mib=int(row[4]),
            utilization_pct=int(row[5]),
            temperature_c=int(row[6]),
            sm_clock_mhz=int(row[7]),
        )
        for row in _rows(text, 8)
    )


def parse_process_rows(text: str) -> tuple[GpuProcessSnapshot, ...]:
    """Parse no-header, no-unit NVIDIA compute-process CSV rows."""
    return tuple(
        GpuProcessSnapshot(gpu_uuid=row[0], pid=int(row[1]), name=row[2], memory_mib=int(row[3]))
        for row in _rows(text, 4)
    )


def capture_gpu_snapshot(
    *,
    runner: Callable[..., CompletedProcess[str]] = subprocess.run,
    timestamp: str | None = None,
) -> GpuSnapshot:
    """Capture one device/process snapshot using :command:`nvidia-smi`."""
    options = {"check": True, "capture_output": True, "text": True}
    try:
        devices = runner(
            [
                "nvidia-smi",
                "--query-gpu",
                "index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,clocks.sm",
                "--format=csv,noheader,nounits",
            ],
            **options,
        )
        processes = runner(
            [
                "nvidia-smi",
                "--query-compute-apps",
                "gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            **options,
        )
    except FileNotFoundError as error:
        raise RuntimeError("nvidia-smi is required for benchmark preflight") from error
    captured_at = timestamp or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return GpuSnapshot(
        timestamp=captured_at,
        devices=parse_gpu_rows(devices.stdout),
        processes=parse_process_rows(processes.stdout),
    )
