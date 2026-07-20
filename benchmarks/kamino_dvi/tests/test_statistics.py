# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for five-seed Kamino DVI benchmark statistics."""

import pytest

from benchmarks.kamino_dvi.statistics import final_window_mean, mean_ci95, paired_ratio, rolling_mean


def test_mean_ci95_uses_approved_five_seed_student_t_interval():
    """Five-seed intervals use t(0.975, 4) rather than a normal interval."""
    estimate = mean_ci95([1.0, 2.0, 3.0, 4.0, 5.0])

    assert estimate.mean == 3.0
    assert estimate.half_width == pytest.approx(1.963243, rel=1e-6)
    assert estimate.n == 5


def test_final_window_and_rolling_means_preserve_iteration_meaning():
    """Learning summaries use the final 20 iterations and exact rolling windows."""
    series = tuple(float(value) for value in range(1, 31))

    assert final_window_mean(series, 20) == 20.5
    assert rolling_mean((1.0, 2.0, 3.0, 4.0), 3) == (2.0, 3.0)


def test_paired_ratio_requires_matching_five_seeds():
    """Speedups pair candidate and baseline values by seed before aggregation."""
    baseline = {42: 10.0, 43: 12.0, 44: 14.0, 45: 16.0, 46: 18.0}
    candidate = {42: 5.0, 43: 6.0, 44: 7.0, 45: 8.0, 46: 9.0}

    estimate = paired_ratio(baseline, candidate)

    assert estimate.mean == 2.0
    assert estimate.half_width == 0.0
    with pytest.raises(ValueError, match="matching five seeds"):
        paired_ratio(baseline, {42: 5.0})
