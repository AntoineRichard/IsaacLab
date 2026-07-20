# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for benchmark run aggregation."""

from benchmarks.kamino_dvi.analysis import RunMetrics, summarize_records


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
