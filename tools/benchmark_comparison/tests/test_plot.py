# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for deterministic plots generated solely from normalized CSV."""

from __future__ import annotations

import struct
from pathlib import Path

from tools.benchmark_comparison.normalize import NormalizedRun, write_raw_runs_csv
from tools.benchmark_comparison.plot import PLOT_BASENAMES, generate_plots


def _run(task: str, mode: str, seed: int, version: str, value: float) -> NormalizedRun:
    return NormalizedRun(
        version=version,
        version_sha=("a" if version == "lab2" else "b") * 40,
        environment_identity=version,
        logical_task=task,
        concrete_task=f"{task}-{version}",
        mode=mode,
        bound=100,
        bound_unit="iterations" if mode == "training-100" else "steps",
        seed=seed,
        num_envs=4096,
        collection_fps=value,
        gpu_memory_mean_mib=value + 100,
        gpu_memory_peak_mib=value + 200,
        gpu_utilization_mean_pct=value / 10,
        gpu_utilization_sample_count=seed - 30,
        elapsed_time_s=5.0,
        artifact_path=f"final/{task}/{mode}/{seed}/{version}/success",
    )


def _png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    return struct.unpack(">II", data[16:24])


def test_plots_have_fixed_names_dimensions_and_byte_identical_regeneration(tmp_path: Path) -> None:
    rows = tuple(
        _run(task, mode, seed, version, 100 + index * 10)
        for index, (task, mode, seed, version) in enumerate(
            (
                ("cartpole", "runtime-100", 42, "lab2"),
                ("cartpole", "runtime-100", 42, "lab3"),
                ("cartpole", "runtime-100", 43, "lab2"),
                ("cartpole", "runtime-100", 43, "lab3"),
                ("ant", "training-100", 42, "lab2"),
            )
        )
    )
    csv_path = write_raw_runs_csv(tmp_path / "raw_runs.csv", rows)

    first = generate_plots(csv_path, tmp_path / "first")
    second = generate_plots(csv_path, tmp_path / "second")

    expected_names = {f"{basename}.{extension}" for basename in PLOT_BASENAMES for extension in ("png", "svg")}
    assert {path.name for path in first} == expected_names
    assert all(path.stat().st_size > 1000 for path in first)
    assert {_png_dimensions(path) for path in first if path.suffix == ".png"} == {(1800, 1000)}
    assert all(b"Missing" in path.read_bytes() for path in first if path.suffix == ".svg")
    assert {path.name: path.read_bytes() for path in first} == {path.name: path.read_bytes() for path in second}
