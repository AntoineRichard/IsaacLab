# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate deterministic comparison plots from normalized inputs."""

from __future__ import annotations

import os
import statistics
from collections.abc import Sequence
from pathlib import Path

from .aggregate import AGGREGATE_METRICS, AGGREGATE_STATISTICS, AggregateDelta, AggregateStatistic
from .matrix import expand_legacy_schema_1_matrix, task_aliases_by_category
from .models import MatrixExpansion, RunSet, TaskCategory
from .normalize import (
    STARTUP_PHASES,
    VERSION_ORDER,
    NormalizedRun,
    expansion_orders,
    read_raw_runs_csv,
    task_order_for_mode,
)

PLOT_METRICS = {
    "collection_fps": ("collection_fps", "Collection FPS", "Collection FPS"),
    "gpu_memory_mean_mib": ("gpu_memory_mean_mib", "Mean GPU memory [MiB]", "Mean GPU memory [MiB]"),
    "gpu_memory_peak_mib": ("gpu_memory_peak_mib", "Peak GPU memory [MiB]", "Peak GPU memory [MiB]"),
    "gpu_utilization_mean_pct": (
        "gpu_utilization_mean_pct",
        "Mean GPU utilization [%]",
        "Mean GPU utilization [%]",
    ),
    "startup_total_s": ("startup_total_s", "Total Startup Time", "Total startup time [s]"),
}
_VERSION_COLORS = {"lab2": "#4C78A8", "lab3": "#F58518"}
_VERSION_LABELS = {"lab2": "Isaac Lab 2", "lab3": "Isaac Lab 3"}
_PHASE_COLORS = {
    "startup_app_launch_s": "#4C78A8",
    "startup_python_imports_s": "#72B7B2",
    "startup_task_config_s": "#F2CF5B",
    "startup_env_creation_s": "#F58518",
    "startup_first_step_s": "#E45756",
}
_PHASE_LABELS = {
    "startup_app_launch_s": "App launch",
    "startup_python_imports_s": "Python imports",
    "startup_task_config_s": "Task config",
    "startup_env_creation_s": "Environment creation",
    "startup_first_step_s": "First step",
}
CATEGORY_LABELS = {
    TaskCategory.CLASSIC: "Classic",
    TaskCategory.LOCOMOTION_FLAT: "Locomotion Flat",
    TaskCategory.LOCOMOTION_ROUGH: "Locomotion Rough",
    TaskCategory.MANIPULATION: "Manipulation",
}
_AGGREGATE_METRIC_LABELS = {
    "collection_fps": "Collection FPS",
    "startup_total_s": "Total startup time [s]",
    "gpu_memory_mean_mib": "Mean GPU memory [MiB]",
    "gpu_memory_peak_mib": "Peak GPU memory [MiB]",
    "gpu_utilization_mean_pct": "Mean GPU utilization [%]",
}
_AGGREGATE_TITLES = {
    AggregateStatistic.MEDIAN: "Median Task-Level Percentage Delta",
    AggregateStatistic.MEAN: "Mean Task-Level Percentage Delta",
}
_AGGREGATE_BASENAMES = {
    AggregateStatistic.MEDIAN: "aggregate_delta_median_pct",
    AggregateStatistic.MEAN: "aggregate_delta_mean_pct",
}
AGGREGATE_PLOT_BASENAMES = tuple(_AGGREGATE_BASENAMES.values())
DETAIL_PLOT_BASENAMES = tuple(
    f"{category.value}_{metric}" for category in TaskCategory for metric in (*PLOT_METRICS, "startup_phase_breakdown")
)
PLOT_BASENAMES = AGGREGATE_PLOT_BASENAMES + DETAIL_PLOT_BASENAMES


def generate_plots(
    raw_runs_path: Path,
    aggregate_deltas: Sequence[AggregateDelta],
    output_directory: Path,
    *,
    expansion: MatrixExpansion | None = None,
) -> tuple[Path, ...]:
    """Generate 26 fixed PNG/SVG figures from normalized CSV inputs.

    Matplotlib is imported only when plotting is requested. This keeps the
    comparison controller's non-plotting commands dependency-free.
    """
    plt, matplotlib = _matplotlib()
    runs = read_raw_runs_csv(raw_runs_path)
    plot_expansion = expansion if expansion is not None else expand_legacy_schema_1_matrix(RunSet.FINAL)
    _task_order, mode_order, _task_modes = expansion_orders(plot_expansion)
    category_aliases = task_aliases_by_category(plot_expansion)
    cells = tuple(aggregate_deltas)
    _validate_aggregate_deltas(cells, mode_order)
    output_directory.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    with matplotlib.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "isaaclab-benchmark-comparison",
        }
    ):
        generated.extend(_generate_aggregate_heatmaps(plt, matplotlib, cells, output_directory, mode_order))
        for category, category_tasks in category_aliases.items():
            for metric, (attribute, title, y_label) in PLOT_METRICS.items():
                generated.extend(
                    _generate_scalar_metric(
                        plt,
                        matplotlib,
                        runs,
                        output_directory,
                        category,
                        category_tasks,
                        metric,
                        attribute,
                        title,
                        y_label,
                        mode_order,
                        plot_expansion,
                    )
                )
            generated.extend(
                _generate_startup_phase_breakdown(
                    plt,
                    matplotlib,
                    runs,
                    output_directory,
                    category,
                    category_tasks,
                    mode_order,
                    plot_expansion,
                )
            )
    return tuple(generated)


def _validate_aggregate_deltas(cells: tuple[AggregateDelta, ...], mode_order: tuple[str, ...]) -> None:
    """Validate exact aggregate-cell uniqueness and rendering order."""
    expected = tuple(
        (statistic, category, mode, metric)
        for statistic in AGGREGATE_STATISTICS
        for category in TaskCategory
        for mode in mode_order
        for metric in AGGREGATE_METRICS
    )
    actual = tuple((cell.statistic, cell.category, cell.mode, cell.metric) for cell in cells)
    if actual != expected:
        raise ValueError(
            f"aggregate deltas must contain exactly {len(expected)} unique cells in statistic, category, mode, "
            "and metric order"
        )


def _aggregate_color_norm(cells: Sequence[AggregateDelta]):
    """Return a shared symmetric color normalization for aggregate cells."""
    _plt, matplotlib = _matplotlib()
    available = [abs(cell.value) for cell in cells if cell.value is not None]
    color_limit = max(available, default=1.0)
    if color_limit == 0.0:
        color_limit = 1.0
    return matplotlib.colors.TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)


def _generate_aggregate_heatmaps(
    plt,
    matplotlib,
    cells: tuple[AggregateDelta, ...],
    output_directory: Path,
    mode_order: tuple[str, ...],
) -> tuple[Path, ...]:
    """Generate median and mean aggregate percentage-delta heatmaps."""
    import numpy as np

    norm = _aggregate_color_norm(cells)
    colormap = matplotlib.colormaps["RdBu_r"].copy()
    colormap.set_bad("#F2F2F2")
    generated: list[Path] = []
    row_labels = tuple(f"{CATEGORY_LABELS[category]} — {mode}" for category in TaskCategory for mode in mode_order)
    metric_labels = tuple(_AGGREGATE_METRIC_LABELS[metric] for metric in AGGREGATE_METRICS)
    cells_per_statistic = len(row_labels) * len(AGGREGATE_METRICS)
    for statistic_index, statistic in enumerate(AGGREGATE_STATISTICS):
        statistic_cells = cells[statistic_index * cells_per_statistic : (statistic_index + 1) * cells_per_statistic]
        values = np.ma.masked_all((len(row_labels), len(AGGREGATE_METRICS)), dtype=float)
        for cell_index, cell in enumerate(statistic_cells):
            if cell.value is not None:
                values[cell_index // len(AGGREGATE_METRICS), cell_index % len(AGGREGATE_METRICS)] = cell.value

        figure, axis = plt.subplots(figsize=(12, 8), dpi=150)
        image = axis.imshow(values, cmap=colormap, norm=norm, aspect="auto")
        axis.grid(False)
        axis.set_title(_AGGREGATE_TITLES[statistic], fontsize=14, pad=16)
        axis.set_xticks(
            range(len(metric_labels)),
            metric_labels,
            rotation=20,
            ha="right",
            rotation_mode="anchor",
        )
        axis.set_yticks(range(len(row_labels)), row_labels)
        for cell_index, cell in enumerate(statistic_cells):
            row_index, column_index = divmod(cell_index, len(AGGREGATE_METRICS))
            annotation = "N/A" if cell.value is None else f"{cell.value:+.1f}%\n(n={cell.task_count})"
            color = "white" if cell.value is not None and abs(cell.value) > norm.vmax * 0.55 else "black"
            axis.text(column_index, row_index, annotation, ha="center", va="center", fontsize=8, color=color)
        colorbar = figure.colorbar(image, ax=axis, orientation="horizontal", pad=0.14, fraction=0.06)
        colorbar.set_label("Isaac Lab 3 - Isaac Lab 2 [%]")
        figure.subplots_adjust(left=0.24, right=0.98, bottom=0.2, top=0.91)
        generated.extend(_save_figure(figure, output_directory, _AGGREGATE_BASENAMES[statistic]))
        plt.close(figure)
    return tuple(generated)


def _task_order_for_mode(mode: str, expansion: MatrixExpansion | None = None) -> tuple[str, ...]:
    """Return the deterministic plot order for one benchmark mode."""
    return task_order_for_mode(mode, expansion)


def _category_task_order_for_mode(
    category_tasks: tuple[str, ...], mode: str, expansion: MatrixExpansion
) -> tuple[str, ...]:
    """Return category tasks that support one benchmark mode."""
    mode_tasks = set(_task_order_for_mode(mode, expansion))
    return tuple(task for task in category_tasks if task in mode_tasks)


def _generate_scalar_metric(
    plt,
    matplotlib,
    runs: Sequence[NormalizedRun],
    output_directory: Path,
    category: TaskCategory,
    category_tasks: tuple[str, ...],
    metric: str,
    attribute: str,
    title: str,
    y_label: str,
    mode_order: tuple[str, ...],
    expansion: MatrixExpansion,
) -> tuple[Path, Path]:
    """Generate one category-specific scalar metric figure."""
    figure, axes = plt.subplots(1, len(mode_order), figsize=(12, 20 / 3), dpi=150, sharey=False)
    figure.suptitle(f"{category.value.title()} — {title}", fontsize=14)
    for axis, mode in zip(axes, mode_order, strict=True):
        task_order = _category_task_order_for_mode(category_tasks, mode, expansion)
        _draw_mode(axis, runs, mode, attribute, task_order)
        axis.set_title(mode)
        axis.set_ylabel(y_label)
        axis.set_xticks(
            range(len(task_order)),
            [task.replace("_", " ") for task in task_order],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        axis.tick_params(axis="x", labelsize=7)
    handles = [
        matplotlib.patches.Patch(color=_VERSION_COLORS[version], label=_VERSION_LABELS[version])
        for version in VERSION_ORDER
    ]
    figure.legend(handles=handles, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 0.955), frameon=False)
    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.88, wspace=0.28)
    paths = _save_figure(figure, output_directory, f"{category.value}_{metric}")
    plt.close(figure)
    return paths


def _draw_mode(
    axis,
    runs: tuple[NormalizedRun, ...],
    mode: str,
    attribute: str,
    task_order: tuple[str, ...],
) -> None:
    width = 0.34
    version_offsets = {"lab2": -width / 2, "lab3": width / 2}
    max_value = max(
        (float(getattr(run, attribute)) for run in runs if run.mode == mode),
        default=1.0,
    )
    label_height = max(max_value * 0.025, 0.1)
    for task_index, task in enumerate(task_order):
        for version in VERSION_ORDER:
            values = sorted(
                (
                    (run.seed, float(getattr(run, attribute)))
                    for run in runs
                    if run.mode == mode and run.logical_task == task and run.version == version
                ),
                key=lambda item: item[0],
            )
            x_position = task_index + version_offsets[version]
            if not values:
                axis.text(
                    x_position,
                    label_height,
                    "Missing",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color=_VERSION_COLORS[version],
                )
                continue
            measurements = [value for _, value in values]
            mean = statistics.fmean(measurements)
            standard_deviation = statistics.stdev(measurements) if len(measurements) > 1 else 0.0
            axis.bar(
                x_position,
                mean,
                width=width * 0.86,
                color=_VERSION_COLORS[version],
                alpha=0.55,
                yerr=standard_deviation,
                capsize=3,
                error_kw={"linewidth": 1},
            )
            offsets = _repeat_offsets(len(measurements), width * 0.38)
            axis.scatter(
                [x_position + offset for offset in offsets],
                measurements,
                s=13,
                color=_VERSION_COLORS[version],
                edgecolors="white",
                linewidths=0.35,
                zorder=3,
            )
    axis.set_xlim(-0.55, len(task_order) - 0.45)
    axis.set_ylim(bottom=0)


def _repeat_offsets(count: int, spread: float) -> list[float]:
    if count <= 1:
        return [0.0] * count
    step = spread / (count - 1)
    return [-spread / 2 + index * step for index in range(count)]


def _startup_phase_means(runs: Sequence[NormalizedRun], mode: str, task: str, version: str) -> dict[str, float]:
    selected = tuple(run for run in runs if run.mode == mode and run.logical_task == task and run.version == version)
    return {attribute: statistics.fmean(getattr(run, attribute) for run in selected) for _, attribute in STARTUP_PHASES}


def _generate_startup_phase_breakdown(
    plt,
    matplotlib,
    runs: Sequence[NormalizedRun],
    output_directory: Path,
    category: TaskCategory,
    category_tasks: tuple[str, ...],
    mode_order: tuple[str, ...],
    expansion: MatrixExpansion,
) -> tuple[Path, Path]:
    figure, axes = plt.subplots(1, len(mode_order), figsize=(12, 20 / 3), dpi=150, sharey=False)
    figure.suptitle(f"{category.value.title()} — Startup Phase Breakdown", fontsize=14)
    width = 0.34
    version_offsets = {"lab2": -width / 2, "lab3": width / 2}
    version_hatches = {"lab2": "/", "lab3": "\\"}
    for axis, mode in zip(axes, mode_order, strict=True):
        task_order = _category_task_order_for_mode(category_tasks, mode, expansion)
        max_value = max((run.startup_total_s for run in runs if run.mode == mode), default=1.0)
        label_height = max(max_value * 0.025, 0.1)
        for task_index, task in enumerate(task_order):
            for version in VERSION_ORDER:
                x_position = task_index + version_offsets[version]
                selected = tuple(
                    run for run in runs if run.mode == mode and run.logical_task == task and run.version == version
                )
                if not selected:
                    axis.text(
                        x_position,
                        label_height,
                        "Missing",
                        rotation=90,
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        color=_VERSION_COLORS[version],
                    )
                    continue
                phase_means = _startup_phase_means(runs, mode, task, version)
                bottom = 0.0
                for _, attribute in STARTUP_PHASES:
                    value = phase_means[attribute]
                    axis.bar(
                        x_position,
                        value,
                        bottom=bottom,
                        width=width * 0.86,
                        color=_PHASE_COLORS[attribute],
                        hatch=version_hatches[version],
                        edgecolor="black",
                        linewidth=0.3,
                    )
                    bottom += value
        axis.set_title(mode)
        axis.set_ylabel("Startup time [s]")
        axis.set_xticks(
            range(len(task_order)),
            [task.replace("_", " ") for task in task_order],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        axis.tick_params(axis="x", labelsize=7)
        axis.set_xlim(-0.55, len(task_order) - 0.45)
        axis.set_ylim(bottom=0)
    phase_handles = [
        matplotlib.patches.Patch(color=_PHASE_COLORS[attribute], label=_PHASE_LABELS[attribute])
        for _, attribute in STARTUP_PHASES
    ]
    version_handles = [
        matplotlib.patches.Patch(
            facecolor="white",
            edgecolor="black",
            hatch=version_hatches[version],
            label=_VERSION_LABELS[version],
        )
        for version in VERSION_ORDER
    ]
    figure.legend(handles=phase_handles, loc="upper center", ncols=5, bbox_to_anchor=(0.5, 0.955), frameon=False)
    figure.legend(handles=version_handles, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 0.915), frameon=False)
    figure.subplots_adjust(left=0.06, right=0.99, bottom=0.22, top=0.84, wspace=0.28)
    paths = _save_figure(figure, output_directory, f"{category.value}_startup_phase_breakdown")
    plt.close(figure)
    return paths


def _save_figure(figure, output_directory: Path, basename: str) -> tuple[Path, Path]:
    """Save one figure atomically in PNG and SVG formats."""
    paths: list[Path] = []
    for extension in ("png", "svg"):
        path = output_directory / f"{basename}.{extension}"
        temporary = path.with_suffix(path.suffix + ".tmp")
        metadata = (
            {"Software": "Isaac Lab benchmark comparison"}
            if extension == "png"
            else {"Date": None, "Creator": "Isaac Lab benchmark comparison"}
        )
        figure.savefig(
            temporary,
            format=extension,
            dpi=150,
            metadata=metadata,
            facecolor="white",
            edgecolor="white",
        )
        os.replace(temporary, path)
        paths.append(path)
    return tuple(paths)


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Plot generation requires Matplotlib. Run this command from the pinned Isaac Lab 3 uv "
            "environment (`uv run --project <lab3-worktree> ...`) or install matplotlib in the "
            "reporting environment."
        ) from error
    return plt, matplotlib
