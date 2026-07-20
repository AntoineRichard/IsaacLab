# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for GPU pre-run and periodic resource snapshots."""

from pathlib import Path
from subprocess import CompletedProcess

import pytest

from benchmarks.kamino_dvi.gpu import capture_gpu_snapshot, parse_gpu_rows, parse_process_rows


def test_gpu_and_process_csv_parsers_retain_required_fields():
    """NVIDIA CSV output must map to typed device and visible-process records."""
    devices = parse_gpu_rows("0, NVIDIA GeForce RTX 5090, 32607, 1024, 31583, 41, 67, 2100\n")
    processes = parse_process_rows("GPU-abc, 1234, python, 900\nGPU-abc, 5678, Xorg, 100\n")

    assert len(devices) == 1
    assert devices[0].index == 0
    assert devices[0].name == "NVIDIA GeForce RTX 5090"
    assert devices[0].memory_total_mib == 32607
    assert devices[0].memory_free_mib == 31583
    assert devices[0].utilization_pct == 41
    assert devices[0].temperature_c == 67
    assert devices[0].sm_clock_mhz == 2100
    assert all(process.gpu_uuid == "GPU-abc" for process in processes)
    assert [(process.pid, process.name, process.memory_mib) for process in processes] == [
        (1234, "python", 900),
        (5678, "Xorg", 100),
    ]


def test_capture_gpu_snapshot_invokes_nvidia_smi_without_a_shell():
    """GPU capture must use argv commands and retain timestamped device/process data."""
    calls: list[list[str]] = []

    def runner(command, **kwargs):
        calls.append(command)
        if "--query-gpu" in command:
            return CompletedProcess(command, 0, stdout="0, RTX 5090, 32607, 10, 32597, 0, 40, 300\n", stderr="")
        return CompletedProcess(command, 0, stdout="GPU-abc, 42, python, 10\n", stderr="")

    snapshot = capture_gpu_snapshot(runner=runner, timestamp="2026-07-20T12:00:00Z")

    assert snapshot.timestamp == "2026-07-20T12:00:00Z"
    assert snapshot.devices[0].memory_used_mib == 10
    assert snapshot.processes[0].pid == 42
    assert len(calls) == 2
    assert all(call[0] == "nvidia-smi" for call in calls)


def test_capture_gpu_snapshot_reports_missing_nvidia_smi():
    """A missing GPU observer must stop preflight rather than fabricate values."""

    def missing(command, **kwargs):
        raise FileNotFoundError(Path(command[0]))

    with pytest.raises(RuntimeError, match="nvidia-smi is required"):
        capture_gpu_snapshot(runner=missing, timestamp="2026-07-20T12:00:00Z")
