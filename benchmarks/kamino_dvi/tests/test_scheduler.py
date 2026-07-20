# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for benchmark capacity fallback scheduling."""

import pytest

from benchmarks.kamino_dvi.matrix import DEFAULT_MATRIX_PATH, load_matrix
from benchmarks.kamino_dvi.models import FailureCategory, Phase, TaskName
from benchmarks.kamino_dvi.scheduler import capacity_retry, next_environment_count


@pytest.fixture
def matrix():
    return load_matrix(DEFAULT_MATRIX_PATH)


def test_only_capacity_failure_lowers_environment_count(matrix):
    """Numerical, timeout, crash, incomplete, and artifact failures stay at the selected count."""
    assert next_environment_count(matrix, 4096, FailureCategory.CAPACITY) == 2048
    for category in FailureCategory:
        if category is not FailureCategory.CAPACITY:
            assert next_environment_count(matrix, 4096, category) is None


def test_capacity_ladder_stops_cleanly_at_128(matrix):
    """Capacity fallback must never invent a count below the approved ladder."""
    assert next_environment_count(matrix, 256, FailureCategory.CAPACITY) == 128
    assert next_environment_count(matrix, 128, FailureCategory.CAPACITY) is None
    with pytest.raises(ValueError, match="not in environment-count ladder"):
        next_environment_count(matrix, 777, FailureCategory.CAPACITY)


@pytest.mark.parametrize("phase", [Phase.PREFLIGHT, Phase.FULL])
def test_capacity_retry_repreflights_every_task_variant_at_new_common_count(matrix, phase):
    """A capacity failure invalidates the task count and repeats all variant preflights."""
    decision = capacity_retry(matrix, TaskName.ANT, phase, 4096, failed_run_id="failed")

    assert decision.invalidated_count == 4096
    assert decision.next_count == 2048
    assert decision.full_results_invalidated is (phase is Phase.FULL)
    assert decision.parent_run_id == "failed"
    assert len(decision.preflights) == 5
    assert all(run.task is TaskName.ANT for run in decision.preflights)
    assert all(run.phase is Phase.PREFLIGHT for run in decision.preflights)
    assert all(run.num_envs == 2048 for run in decision.preflights)


def test_capacity_retry_fails_clearly_when_ladder_is_exhausted(matrix):
    """The scheduler must surface an irrecoverable 128-environment capacity failure."""
    with pytest.raises(RuntimeError, match="capacity ladder exhausted"):
        capacity_retry(matrix, TaskName.DR_LEGS, Phase.PREFLIGHT, 128, failed_run_id="failed")
