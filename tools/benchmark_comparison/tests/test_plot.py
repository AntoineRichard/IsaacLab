# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for deterministic plots generated solely from normalized CSV."""

from __future__ import annotations

import statistics
import struct
from dataclasses import replace
from pathlib import Path

import pytest
from matplotlib import image as matplotlib_image

from tools.benchmark_comparison.matrix import expand_final_matrix, load_matrix, task_aliases_by_category
from tools.benchmark_comparison.models import TaskCategory
from tools.benchmark_comparison.normalize import STARTUP_PHASES, NormalizedRun, write_raw_runs_csv
from tools.benchmark_comparison.plot import PLOT_BASENAMES, _startup_phase_means, generate_plots


def test_plot_task_order_does_not_create_an_rgb_training_slot() -> None:
    from tools.benchmark_comparison.normalize import TASK_ORDER
    from tools.benchmark_comparison.plot import _task_order_for_mode

    assert "cartpole_rgb_kit" in _task_order_for_mode("runtime-100")
    assert "cartpole_rgb_kit" not in _task_order_for_mode("training-100")
    assert _task_order_for_mode("training-100") == tuple(task for task in TASK_ORDER if task != "cartpole_rgb_kit")


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
        startup_total_s=4.41,
        startup_app_launch_s=2.5,
        startup_python_imports_s=0.2,
        startup_task_config_s=0.4,
        startup_env_creation_s=1.3,
        startup_first_step_s=0.01,
        artifact_path=f"final/{task}/{mode}/{seed}/{version}/success",
    )


def _png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    return struct.unpack(">II", data[16:24])


def test_plots_have_fixed_names_dimensions_and_byte_identical_regeneration(tmp_path: Path) -> None:
    expansion = expand_final_matrix(load_matrix())
    category_aliases = task_aliases_by_category(expansion)
    rows = tuple(
        _run(task, mode, seed, version, 100 + index * 10)
        for index, (task, mode, seed, version) in enumerate(
            (
                ("cartpole", "runtime-100", 42, "lab2"),
                ("cartpole", "runtime-100", 42, "lab3"),
                ("cartpole", "runtime-100", 43, "lab2"),
                ("cartpole", "runtime-100", 43, "lab3"),
                ("ant", "training-100", 42, "lab2"),
                ("anymal_d_flat", "runtime-100", 42, "lab2"),
                ("allegro_cube", "runtime-100", 42, "lab2"),
                ("cartpole_rgb_kit", "runtime-100", 42, "lab2"),
            )
        )
    )
    csv_path = write_raw_runs_csv(tmp_path / "raw_runs.csv", rows)

    first = generate_plots(csv_path, tmp_path / "first", expansion=expansion)
    second = generate_plots(csv_path, tmp_path / "second", expansion=expansion)

    expected_names = {f"{basename}.{extension}" for basename in PLOT_BASENAMES for extension in ("png", "svg")}
    assert {path.name for path in first} == expected_names
    assert len(first) == 36
    assert all(path.stat().st_size > 1000 for path in first)
    assert {_png_dimensions(path) for path in first if path.suffix == ".png"} == {(1800, 1000)}
    assert all(b"Missing" in path.read_bytes() for path in first if path.suffix == ".svg")
    assert all(b"rotate(-45)" in path.read_bytes() for path in first if path.suffix == ".svg")
    assert {path.name: path.read_bytes() for path in first} == {path.name: path.read_bytes() for path in second}
    assert b"Total startup time [s]" in (tmp_path / "first" / "classic_startup_total_s.svg").read_bytes()
    for basename in PLOT_BASENAMES:
        category = TaskCategory(basename.split("_", maxsplit=1)[0])
        svg = (tmp_path / "first" / f"{basename}.svg").read_bytes()
        aliases = category_aliases[category]
        assert all(f"<!-- {task.replace('_', ' ')} -->".encode() in svg for task in aliases)
        assert all(
            f"<!-- {task.replace('_', ' ')} -->".encode() not in svg
            for other_category, other_aliases in category_aliases.items()
            if other_category is not category
            for task in other_aliases
        )

    classic_svg = (tmp_path / "first" / "classic_collection_fps.svg").read_bytes()
    runtime_100, runtime_1000, training_100 = (
        classic_svg.split(f'<g id="axes_{index}">'.encode(), maxsplit=1)[1] for index in range(1, 4)
    )
    rgb_label = b"<!-- cartpole rgb kit -->"
    assert rgb_label in runtime_100
    assert rgb_label in runtime_1000
    assert rgb_label not in training_100

    breakdown = (tmp_path / "first" / "classic_startup_phase_breakdown.svg").read_bytes()
    assert all(
        label.encode() in breakdown
        for label in ("App launch", "Python imports", "Task config", "Environment creation", "First step")
    )
    assert b"Isaac Lab 2" in breakdown
    assert b"Isaac Lab 3" in breakdown


def test_startup_phase_means_sum_to_total_bar_height() -> None:
    runs = (
        replace(
            _run("cartpole", "runtime-100", 42, "lab2", 100.0),
            startup_total_s=15.0,
            startup_app_launch_s=1.0,
            startup_python_imports_s=2.0,
            startup_task_config_s=3.0,
            startup_env_creation_s=4.0,
            startup_first_step_s=5.0,
        ),
        replace(
            _run("cartpole", "runtime-100", 43, "lab2", 110.0),
            startup_total_s=20.0,
            startup_app_launch_s=2.0,
            startup_python_imports_s=3.0,
            startup_task_config_s=4.0,
            startup_env_creation_s=5.0,
            startup_first_step_s=6.0,
        ),
    )
    phase_means = _startup_phase_means(runs, "runtime-100", "cartpole", "lab2")
    assert tuple(phase_means) == tuple(attribute for _, attribute in STARTUP_PHASES)
    assert sum(phase_means.values()) == pytest.approx(
        statistics.fmean(run.startup_total_s for run in runs if run.version == "lab2")
    )


@pytest.mark.parametrize(
    ("category", "task", "collection_fps"),
    (
        (TaskCategory.LOCOMOTION, "anymal_d_flat", 350_000.0),
        (TaskCategory.MANIPULATION, "allegro_cube", 700_000.0),
    ),
)
def test_collection_fps_y_axis_label_does_not_touch_left_png_boundary(
    tmp_path: Path, category: TaskCategory, task: str, collection_fps: float
) -> None:
    expansion = expand_final_matrix(load_matrix())
    csv_path = write_raw_runs_csv(
        tmp_path / "raw_runs.csv",
        tuple(_run(task, "runtime-100", 42, version, collection_fps) for version in ("lab2", "lab3")),
    )

    generate_plots(csv_path, tmp_path / "plots", expansion=expansion)

    image = matplotlib_image.imread(tmp_path / "plots" / f"{category.value}_collection_fps.png")
    assert bool((image[:, 0, :3] == 1.0).all())


def test_plot_basenames_include_startup_figures() -> None:
    metric_basenames = (
        "collection_fps",
        "gpu_memory_mean_mib",
        "gpu_memory_peak_mib",
        "gpu_utilization_mean_pct",
        "startup_total_s",
        "startup_phase_breakdown",
    )
    assert (
        tuple(f"{category.value}_{metric}" for category in TaskCategory for metric in metric_basenames)
        == PLOT_BASENAMES
    )
