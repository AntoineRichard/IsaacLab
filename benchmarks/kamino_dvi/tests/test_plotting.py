# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for compact benchmark figures."""

from pathlib import Path

from benchmarks.kamino_dvi.analysis import VariantSummary
from benchmarks.kamino_dvi.plotting import plot_learning, plot_runtime
from benchmarks.kamino_dvi.statistics import Estimate


def test_plot_runtime_writes_nonempty_png(tmp_path: Path):
    """Runtime summaries render as a reusable report figure."""
    estimate = Estimate(1.0, 0.1, 5)
    summary = VariantSummary("task", "dvi", 4096, estimate, estimate, estimate, estimate, estimate)
    output = tmp_path / "runtime.png"

    plot_runtime([summary], output)

    assert output.stat().st_size > 1000


def test_plot_learning_writes_nonempty_three_metric_png(tmp_path: Path):
    """Reward, episode length, and success render with confidence intervals."""
    estimate = Estimate(1.0, 0.1, 3)
    summary = VariantSummary("task", "kamino_pr_dvi", 4096, estimate, estimate, estimate, estimate, estimate)
    output = tmp_path / "learning.png"

    plot_learning([summary], output)

    assert output.stat().st_size > 1000
