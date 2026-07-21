# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for benchmark run aggregation."""

from dataclasses import replace

import pytest

from benchmarks.kamino_dvi import analysis
from benchmarks.kamino_dvi.analysis import RunMetrics, summarize_records
from benchmarks.kamino_dvi.matrix import DEFAULT_MATRIX_PATH, load_matrix


def test_summarize_records_excludes_warmup_and_uses_final_learning_window():
    """Per-seed runtime and learning reductions follow the approved protocol."""
    records = [
        RunMetrics(
            task="task",
            variant="dvi",
            seed=seed,
            num_envs=4096,
            iteration_time_s=tuple([100.0] * 10 + [float(seed - 41)] * 20),
            total_fps=tuple([1.0] * 10 + [1000.0] * 20),
            reward=tuple([0.0] * 10 + [10.0] * 20),
            ep_length=tuple([0.0] * 10 + [20.0] * 20),
            success_rate=tuple([0.0] * 10 + [1.0] * 20),
        )
        for seed in range(42, 45)
    ]

    summary = summarize_records(records)[0]

    assert summary.task == "task"
    assert summary.variant == "dvi"
    assert summary.num_envs == 4096
    assert summary.iteration_time_s.mean == 2.0
    assert summary.total_fps.mean == 1000.0
    assert summary.reward.mean == 10.0
    assert summary.ep_length.mean == 20.0
    assert summary.success_rate is not None
    assert summary.success_rate.mean == 1.0


def test_summarize_partial_records_returns_descriptive_successful_seeds_only():
    """Omitted cells retain per-seed windows without entering confidence estimates."""
    omitted = {("task", "kamino_current")}
    records = [
        RunMetrics(
            task="task",
            variant="kamino_current",
            seed=42,
            num_envs=4096,
            iteration_time_s=tuple([100.0] * 10 + [2.0] * 20),
            total_fps=tuple([1.0] * 10 + [2000.0] * 20),
            reward=tuple([0.0] * 10 + [10.0] * 20),
            ep_length=tuple([0.0] * 10 + [20.0] * 20),
            success_rate=tuple([0.0] * 10 + [0.5] * 20),
        ),
        RunMetrics(
            task="task",
            variant="kamino_current",
            seed=44,
            num_envs=4096,
            iteration_time_s=tuple([100.0] * 10 + [4.0] * 20),
            total_fps=tuple([1.0] * 10 + [1000.0] * 20),
            reward=tuple([0.0] * 10 + [30.0] * 20),
            ep_length=tuple([0.0] * 10 + [40.0] * 20),
            success_rate=None,
        ),
        RunMetrics("other", "physx", 42, 4096, (1.0,) * 20, (1.0,) * 20, (1.0,) * 20, (1.0,) * 20, None),
    ]

    partials = analysis.summarize_partial_records(records, omitted, required_seeds=(42, 43, 44))

    assert [row.seed for row in partials] == [42, 44]
    assert partials[0].task == "task"
    assert partials[0].variant == "kamino_current"
    assert partials[0].num_envs == 4096
    assert partials[0].iteration_time_s == 2.0
    assert partials[0].total_fps == 2000.0
    assert partials[0].reward == 10.0
    assert partials[0].ep_length == 20.0
    assert partials[0].success_rate == 0.5
    assert partials[0].completed_runs == 2
    assert partials[0].required_runs == 3
    assert partials[1].success_rate is None


def _record(task: str, variant: str, seed: int, num_envs: int = 4096) -> RunMetrics:
    """Return one valid synthetic full-run record."""
    series = tuple(float(index) for index in range(20))
    return RunMetrics(task, variant, seed, num_envs, series, series, series, series, series)


def test_validate_record_matrix_rejects_incomplete_task_variant_seed_matrix():
    """A report must not silently summarize a subset of the approved matrix."""
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    records = [
        _record(task.name.value, variant.value, seed)
        for task in matrix.tasks
        for variant in task.variants
        for seed in matrix.seeds
    ]

    with pytest.raises(ValueError, match="missing.*Isaac-Cartpole-Direct.*kamino_current.*42"):
        analysis.validate_record_matrix(records[1:], matrix)


def test_validate_record_matrix_accepts_only_incomplete_explicitly_omitted_cell():
    """Validated terminal failures may omit an empty or partially completed cell."""
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    omitted = ("IsaacContrib-DrLegs-Walk", "kamino_pr_dvi")
    records = [
        _record(task.name.value, variant.value, seed)
        for task in matrix.tasks
        for variant in task.variants
        for seed in matrix.seeds
        if (task.name.value, variant.value) != omitted
    ]

    analysis.validate_record_matrix(records, matrix, omitted_cells={omitted})

    partial = [*records, _record(*omitted, matrix.seeds[0]), _record(*omitted, matrix.seeds[2])]
    analysis.validate_record_matrix(partial, matrix, omitted_cells={omitted})

    complete = [*partial, _record(*omitted, matrix.seeds[1])]
    with pytest.raises(ValueError, match="omitted benchmark cell is complete"):
        analysis.validate_record_matrix(complete, matrix, omitted_cells={omitted})


def test_validate_record_matrix_rejects_unexpected_or_duplicate_identity():
    """Stale, mislabeled, and duplicate run identities must not enter a report."""
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    records = [
        _record(task.name.value, variant.value, seed)
        for task in matrix.tasks
        for variant in task.variants
        for seed in matrix.seeds
    ]
    records.append(replace(records[0], seed=99))

    with pytest.raises(ValueError, match="unexpected.*seed=99"):
        analysis.validate_record_matrix(records, matrix)

    with pytest.raises(ValueError, match="duplicate"):
        analysis.validate_record_matrix(records[:-1] + [records[0]], matrix)


def test_validate_record_matrix_rejects_mixed_or_unapproved_environment_counts():
    """Every task uses one approved capacity count across all variants and seeds."""
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    records = [
        _record(task.name.value, variant.value, seed)
        for task in matrix.tasks
        for variant in task.variants
        for seed in matrix.seeds
    ]

    with pytest.raises(ValueError, match="mixes environment counts"):
        analysis.validate_record_matrix([replace(records[0], num_envs=2048), *records[1:]], matrix)

    with pytest.raises(ValueError, match="unapproved environment count 777"):
        analysis.validate_record_matrix([replace(record, num_envs=777) for record in records], matrix)
