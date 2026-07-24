# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate deterministic comparison plots from ``raw_runs.csv`` alone."""

from __future__ import annotations

import os
import statistics
from pathlib import Path

from .models import MatrixExpansion
from .normalize import VERSION_ORDER, NormalizedRun, expansion_orders, read_raw_runs_csv, task_order_for_mode

PLOT_METRICS = {
    "collection_fps": ("collection_fps", "Collection FPS", "Collection FPS"),
    "gpu_memory_mean_mib": ("gpu_memory_mean_mib", "Mean GPU memory [MiB]", "Mean GPU memory [MiB]"),
    "gpu_memory_peak_mib": ("gpu_memory_peak_mib", "Peak GPU memory [MiB]", "Peak GPU memory [MiB]"),
    "gpu_utilization_mean_pct": (
        "gpu_utilization_mean_pct",
        "Mean GPU utilization [%]",
        "Mean GPU utilization [%]",
    ),
}
PLOT_BASENAMES = tuple(PLOT_METRICS)

_VERSION_COLORS = {"lab2": "#4C78A8", "lab3": "#F58518"}
_VERSION_LABELS = {"lab2": "Isaac Lab 2", "lab3": "Isaac Lab 3"}


def generate_plots(
    raw_runs_path: Path, output_directory: Path, *, expansion: MatrixExpansion | None = None
) -> tuple[Path, ...]:
    """Generate four fixed PNG/SVG figures using normalized successful runs.

    Matplotlib is imported only when plotting is requested. This keeps the
    comparison controller's non-plotting commands dependency-free.
    """
    plt, matplotlib = _matplotlib()
    runs = read_raw_runs_csv(raw_runs_path)
    _task_order, mode_order, _task_modes = expansion_orders(expansion)
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
        for basename, (attribute, title, y_label) in PLOT_METRICS.items():
            figure, axes = plt.subplots(1, len(mode_order), figsize=(12, 20 / 3), dpi=150, sharey=False)
            figure.suptitle(title, fontsize=14)
            for axis, mode in zip(axes, mode_order, strict=True):
                task_order = _task_order_for_mode(mode, expansion)
                _draw_mode(axis, runs, mode, attribute, task_order)
                axis.set_title(mode)
                axis.set_ylabel(y_label)
                axis.set_xticks(range(len(task_order)), [task.replace("_", "\n") for task in task_order])
                axis.tick_params(axis="x", labelsize=8)
            handles = [
                matplotlib.patches.Patch(color=_VERSION_COLORS[version], label=_VERSION_LABELS[version])
                for version in VERSION_ORDER
            ]
            figure.legend(handles=handles, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 0.955), frameon=False)
            figure.subplots_adjust(left=0.06, right=0.99, bottom=0.12, top=0.88, wspace=0.28)
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
                generated.append(path)
            plt.close(figure)
    return tuple(generated)


def _task_order_for_mode(mode: str, expansion: MatrixExpansion | None = None) -> tuple[str, ...]:
    """Return the deterministic plot order for one benchmark mode."""
    return task_order_for_mode(mode, expansion)


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
