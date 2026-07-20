# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure capacity-fallback scheduling for the benchmark runner."""

from dataclasses import dataclass

from .matrix import expand_preflights
from .models import BenchmarkMatrix, FailureCategory, Phase, RunIdentity, TaskName


@dataclass(frozen=True)
class CapacityRetry:
    """Re-preflight decision after one explicit capacity failure."""

    task: TaskName
    invalidated_count: int
    next_count: int
    preflights: tuple[RunIdentity, ...]
    full_results_invalidated: bool
    parent_run_id: str


def next_environment_count(
    matrix: BenchmarkMatrix,
    current_count: int,
    category: FailureCategory,
) -> int | None:
    """Return the next approved count only for a capacity failure."""
    try:
        index = matrix.environment_counts.index(current_count)
    except ValueError as error:
        raise ValueError(f"{current_count} is not in environment-count ladder") from error
    if category is not FailureCategory.CAPACITY or index == len(matrix.environment_counts) - 1:
        return None
    return matrix.environment_counts[index + 1]


def capacity_retry(
    matrix: BenchmarkMatrix,
    task: TaskName,
    failed_phase: Phase,
    current_count: int,
    *,
    failed_run_id: str,
) -> CapacityRetry:
    """Schedule all task variants for preflight at the next common count."""
    next_count = next_environment_count(matrix, current_count, FailureCategory.CAPACITY)
    if next_count is None:
        raise RuntimeError(f"capacity ladder exhausted for {task} at {current_count} environments")
    preflights = tuple(run for run in expand_preflights(matrix, next_count) if run.task is task)
    return CapacityRetry(
        task=task,
        invalidated_count=current_count,
        next_count=next_count,
        preflights=preflights,
        full_results_invalidated=failed_phase is Phase.FULL,
        parent_run_id=failed_run_id,
    )
