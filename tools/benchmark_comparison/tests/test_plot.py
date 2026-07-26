# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for deterministic plots generated solely from normalized CSV."""

from __future__ import annotations

import csv
import statistics
import struct
from dataclasses import replace
from pathlib import Path

import pytest
from matplotlib import image as matplotlib_image
from matplotlib.axes import Axes

import tools.benchmark_comparison.plot as plot_module
from tools.benchmark_comparison.aggregate import AggregateDelta, aggregate_paired_summary
from tools.benchmark_comparison.matrix import expand_final_matrix, load_matrix, task_aliases_by_category
from tools.benchmark_comparison.models import MatrixExpansion, TaskCategory
from tools.benchmark_comparison.normalize import (
    PAIRED_SUMMARY_FIELDS,
    STARTUP_PHASES,
    NormalizedRun,
    write_raw_runs_csv,
)
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


def _paired_summary_row(task: str, metric: str, percent_delta: float) -> dict[str, str]:
    lab2_mean = 100.0
    lab3_mean = lab2_mean + percent_delta
    return {
        "logical_task": task,
        "mode": "runtime-100",
        "metric": metric,
        "paired_seed_count": "3",
        "lab2_mean": str(lab2_mean),
        "lab2_std": "1.0",
        "lab3_mean": str(lab3_mean),
        "lab3_std": "2.0",
        "absolute_delta": str(percent_delta),
        "percent_delta": str(percent_delta),
        "percent_delta_status": "available",
    }


def _aggregate_deltas(path: Path, expansion: MatrixExpansion) -> tuple[AggregateDelta, ...]:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=PAIRED_SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            (
                _paired_summary_row("cartpole", "collection_fps", 10.0),
                _paired_summary_row("cartpole_direct", "collection_fps", 10.0),
                _paired_summary_row("ant", "collection_fps", 10.0),
                _paired_summary_row("cartpole", "startup_total_s", -20.0),
                _paired_summary_row("cartpole", "gpu_memory_mean_mib", 10.0),
                _paired_summary_row("cartpole_direct", "gpu_memory_mean_mib", 10.0),
                _paired_summary_row("ant", "gpu_memory_mean_mib", 70.0),
            )
        )
    return aggregate_paired_summary(path, expansion)


def _category_from_plot_basename(basename: str) -> TaskCategory:
    """Return the longest matching report category prefix from a plot basename."""
    for category in sorted(TaskCategory, key=lambda category: len(category.value), reverse=True):
        if basename.startswith(f"{category.value}_"):
            return category
    raise ValueError(f"plot basename has no category prefix: {basename}")


def test_plots_have_fixed_names_dimensions_and_byte_identical_regeneration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    aggregate_deltas = _aggregate_deltas(tmp_path / "paired_summary.csv", expansion)
    color_norms: list[tuple[float, float, float]] = []
    original_imshow = Axes.imshow

    def record_color_norm(axis, *args, **kwargs):
        norm = kwargs["norm"]
        color_norms.append((norm.vmin, norm.vcenter, norm.vmax))
        return original_imshow(axis, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", record_color_norm)

    first = generate_plots(csv_path, aggregate_deltas, tmp_path / "first", expansion=expansion)
    second = generate_plots(csv_path, aggregate_deltas, tmp_path / "second", expansion=expansion)

    expected_names = tuple(f"{basename}.{extension}" for basename in PLOT_BASENAMES for extension in ("png", "svg"))
    assert tuple(path.name for path in first) == expected_names
    assert {path.name for path in first} == set(expected_names)
    assert len(PLOT_BASENAMES) == 26
    assert len(first) == 52
    assert all(path.stat().st_size > 1000 for path in first)
    aggregate_names = {
        f"{basename}.{extension}"
        for basename in ("aggregate_delta_median_pct", "aggregate_delta_mean_pct")
        for extension in ("png", "svg")
    }
    assert {path.name for path in first if path.stem.startswith("aggregate_")} == aggregate_names
    assert {_png_dimensions(path) for path in first if path.suffix == ".png" and path.name in aggregate_names} == {
        (1800, 1200)
    }
    assert {_png_dimensions(path) for path in first if path.suffix == ".png" and path.name not in aggregate_names} == {
        (1800, 1000)
    }
    detail_svgs = tuple(path for path in first if path.suffix == ".svg" and path.name not in aggregate_names)
    assert all(b"Missing" in path.read_bytes() for path in detail_svgs)
    assert all(b"rotate(-45)" in path.read_bytes() for path in detail_svgs)
    assert {path.name: path.read_bytes() for path in first} == {path.name: path.read_bytes() for path in second}
    assert color_norms == [(-30.0, 0.0, 30.0)] * 4

    row_labels = tuple(
        f"{category} — {mode}"
        for category in ("Classic", "Locomotion Flat", "Locomotion Rough", "Manipulation")
        for mode in ("runtime-100", "runtime-1000", "training-100")
    )
    metric_labels = (
        "Collection FPS",
        "Total startup time [s]",
        "Mean GPU memory [MiB]",
        "Peak GPU memory [MiB]",
        "Mean GPU utilization [%]",
    )
    for basename in ("aggregate_delta_median_pct", "aggregate_delta_mean_pct"):
        svg = (tmp_path / "first" / f"{basename}.svg").read_text(encoding="utf-8")
        label_positions = tuple(svg.index(f"<!-- {label} -->") for label in row_labels)
        assert label_positions == tuple(sorted(label_positions))
        assert all(f"<!-- {label} -->" in svg for label in metric_labels)
        assert all(annotation in svg for annotation in ("+10.0%", "-20.0%", "(n=3)", "N/A"))
        image = matplotlib_image.imread(tmp_path / "first" / f"{basename}.png")
        assert bool((image[[0, -1], :, :3] == 1.0).all())
        assert bool((image[:, [0, -1], :3] == 1.0).all())

    assert b"Total startup time [s]" in (tmp_path / "first" / "classic_startup_total_s.svg").read_bytes()
    for basename in plot_module.DETAIL_PLOT_BASENAMES:
        category = _category_from_plot_basename(basename)
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


def test_aggregate_color_norm_is_symmetric_and_zero_centered(tmp_path: Path) -> None:
    expansion = expand_final_matrix(load_matrix())
    cells = _aggregate_deltas(tmp_path / "paired_summary.csv", expansion)

    norm = plot_module._aggregate_color_norm(cells)

    assert norm.vcenter == 0.0
    assert norm.vmin == -norm.vmax == -30.0


def test_plots_reject_aggregate_cells_out_of_order(tmp_path: Path) -> None:
    expansion = expand_final_matrix(load_matrix())
    cells = _aggregate_deltas(tmp_path / "paired_summary.csv", expansion)
    reordered = (cells[1], cells[0], *cells[2:])
    raw_runs_path = write_raw_runs_csv(tmp_path / "raw_runs.csv", ())

    with pytest.raises(ValueError, match="exactly 120 unique cells.*order"):
        generate_plots(raw_runs_path, reordered, tmp_path / "plots", expansion=expansion)

    assert not (tmp_path / "plots").exists()


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
        (TaskCategory.LOCOMOTION_FLAT, "anymal_d_flat", 350_000.0),
        (TaskCategory.LOCOMOTION_ROUGH, "anymal_d_rough", 350_000.0),
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
    aggregate_deltas = _aggregate_deltas(tmp_path / "paired_summary.csv", expansion)

    generate_plots(csv_path, aggregate_deltas, tmp_path / "plots", expansion=expansion)

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
    detail_basenames = tuple(f"{category.value}_{metric}" for category in TaskCategory for metric in metric_basenames)
    assert plot_module.AGGREGATE_PLOT_BASENAMES == (
        "aggregate_delta_median_pct",
        "aggregate_delta_mean_pct",
    )
    assert detail_basenames == plot_module.DETAIL_PLOT_BASENAMES
    assert plot_module.AGGREGATE_PLOT_BASENAMES + detail_basenames == PLOT_BASENAMES
