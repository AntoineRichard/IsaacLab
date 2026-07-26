# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Aggregate validated task-level benchmark deltas without pooling task scales."""

from __future__ import annotations

import csv
import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .matrix import task_aliases_by_category
from .models import MatrixExpansion, TaskCategory
from .normalize import PAIRED_SUMMARY_FIELDS, SUMMARY_METRICS, expansion_orders


class AggregateStatistic(str, Enum):
    """Statistic applied across equally weighted task percentage deltas."""

    MEDIAN = "median"
    MEAN = "mean"


AGGREGATE_METRICS: tuple[str, ...] = (
    "collection_fps",
    "startup_total_s",
    "gpu_memory_mean_mib",
    "gpu_memory_peak_mib",
    "gpu_utilization_mean_pct",
)
AGGREGATE_STATISTICS: tuple[AggregateStatistic, ...] = (AggregateStatistic.MEDIAN, AggregateStatistic.MEAN)

_FLOAT_FIELDS = (
    "lab2_mean",
    "lab2_std",
    "lab3_mean",
    "lab3_std",
    "absolute_delta",
)
_PERCENT_DELTA_STATUSES = ("available", "undefined_zero_baseline")


@dataclass(frozen=True)
class AggregateDelta:
    """One typed aggregate percentage-delta cell."""

    category: TaskCategory
    mode: str
    metric: str
    statistic: AggregateStatistic
    value: float | None
    task_count: int


def aggregate_paired_summary(
    paired_summary_path: Path,
    expansion: MatrixExpansion,
) -> tuple[AggregateDelta, ...]:
    """Aggregate validated task-level percentage deltas in report order.

    Args:
        paired_summary_path: Normalized paired-summary CSV to aggregate.
        expansion: Matrix expansion defining exact task and mode membership.

    Returns:
        Immutable aggregate cells in statistic, category, mode, and metric order.

    Raises:
        ValueError: If matrix metadata or normalized CSV contents are inconsistent.
    """
    category_tasks = task_aliases_by_category(expansion)
    task_order, mode_order, task_modes = expansion_orders(expansion)
    _validate_category_tasks(category_tasks, task_order)
    task_values = _read_task_values(paired_summary_path, task_order, mode_order, task_modes)

    cells: list[AggregateDelta] = []
    for statistic in AGGREGATE_STATISTICS:
        for category in TaskCategory:
            for mode in mode_order:
                for metric in AGGREGATE_METRICS:
                    values = tuple(
                        task_values[(task, mode, metric)]
                        for task in category_tasks[category]
                        if (task, mode, metric) in task_values
                    )
                    value = _aggregate_value(statistic, values)
                    cells.append(AggregateDelta(category, mode, metric, statistic, value, len(values)))
    return tuple(cells)


def _validate_category_tasks(
    category_tasks: Mapping[TaskCategory, tuple[str, ...]], task_order: tuple[str, ...]
) -> None:
    if tuple(category_tasks) != tuple(TaskCategory):
        raise ValueError("missing category metadata")
    assigned_tasks = tuple(task for category in TaskCategory for task in category_tasks[category])
    if len(assigned_tasks) != len(set(assigned_tasks)):
        raise ValueError("duplicate category metadata")
    if set(assigned_tasks) != set(task_order):
        raise ValueError("missing category metadata for expansion task")


def _read_task_values(
    path: Path,
    task_order: tuple[str, ...],
    mode_order: tuple[str, ...],
    task_modes: Mapping[str, tuple[str, ...]],
) -> dict[tuple[str, str, str], float]:
    task_values: dict[tuple[str, str, str], float] = {}
    seen: set[tuple[str, str, str]] = set()
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != PAIRED_SUMMARY_FIELDS:
            raise ValueError("paired summary has an unexpected header")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("paired summary row does not match header")
            key = _validate_row_identity(row, task_order, mode_order, task_modes)
            if key in seen:
                raise ValueError(f"duplicate paired summary row: {_key_description(key)}")
            seen.add(key)
            _positive_int(row["paired_seed_count"], "paired_seed_count")
            for field in _FLOAT_FIELDS:
                _finite_float(row[field], field)
            value = _task_percent_delta(row)
            if key[2] in AGGREGATE_METRICS and value is not None:
                task_values[key] = value
    return task_values


def _validate_row_identity(
    row: Mapping[str, str],
    task_order: tuple[str, ...],
    mode_order: tuple[str, ...],
    task_modes: Mapping[str, tuple[str, ...]],
) -> tuple[str, str, str]:
    task = row["logical_task"]
    mode = row["mode"]
    metric = row["metric"]
    if task not in task_order:
        raise ValueError(f"unknown task in paired summary: {task}")
    if mode not in mode_order:
        raise ValueError(f"unknown mode in paired summary: {mode}")
    if metric not in SUMMARY_METRICS:
        raise ValueError(f"unknown metric in paired summary: {metric}")
    if row["percent_delta_status"] not in _PERCENT_DELTA_STATUSES:
        raise ValueError(f"unknown percentage status: {row['percent_delta_status']}")
    if mode not in task_modes[task]:
        raise ValueError(f"unsupported mode for task {task}: {mode}")
    return task, mode, metric


def _task_percent_delta(row: Mapping[str, str]) -> float | None:
    lab2_mean = _finite_float(row["lab2_mean"], "lab2_mean")
    lab3_mean = _finite_float(row["lab3_mean"], "lab3_mean")
    description = _key_description((row["logical_task"], row["mode"], row["metric"]))
    if lab2_mean == 0.0:
        if row["percent_delta_status"] != "undefined_zero_baseline" or row["percent_delta"]:
            raise ValueError(f"zero Lab 2 baseline has inconsistent percentage status: {description}")
        return None
    expected = (lab3_mean - lab2_mean) / lab2_mean * 100.0
    serialized = _finite_float(row["percent_delta"], "percent_delta")
    if row["percent_delta_status"] != "available" or not math.isclose(serialized, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"serialized percentage delta disagrees with means: {description}")
    return serialized


def _aggregate_value(statistic: AggregateStatistic, values: tuple[float, ...]) -> float | None:
    if not values:
        return None
    value = statistics.median(values) if statistic is AggregateStatistic.MEDIAN else statistics.fmean(values)
    if not math.isfinite(value):
        raise ValueError("aggregate percentage delta is not finite")
    return value


def _finite_float(value: str, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be a finite number")
    return parsed


def _positive_int(value: str, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a positive integer") from error
    if parsed <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return parsed


def _key_description(key: tuple[str, str, str]) -> str:
    return f"task={key[0]}, mode={key[1]}, metric={key[2]}"
