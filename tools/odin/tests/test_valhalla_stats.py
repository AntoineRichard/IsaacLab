# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for valhalla.stats — mean / std / cv_pct + divergence helper."""

from __future__ import annotations

import math

import pytest

from tools.odin.valhalla.stats import Stats, is_divergent, stats_over


def test_stats_over_n_one_has_zero_std_and_cv():
    s = stats_over([5.0])
    assert s == Stats(mean=5.0, std=0.0, min=5.0, max=5.0, cv_pct=0.0)


def test_stats_over_n_two_uses_population_std():
    # Population std of [4, 6] = sqrt(((-1)^2 + 1^2) / 2) = 1.0
    s = stats_over([4.0, 6.0])
    assert s.mean == 5.0
    assert s.std == pytest.approx(1.0)
    assert s.min == 4.0
    assert s.max == 6.0
    assert s.cv_pct == pytest.approx(20.0)


def test_stats_over_n_three_uses_sample_std():
    # Sample std (ddof=1) of [1, 2, 3] = sqrt(((-1)^2 + 0 + 1^2) / 2) = 1.0
    s = stats_over([1.0, 2.0, 3.0])
    assert s.mean == 2.0
    assert s.std == pytest.approx(1.0)
    assert s.cv_pct == pytest.approx(50.0)


def test_stats_over_mean_zero_yields_zero_cv_not_nan():
    s = stats_over([-1.0, 0.0, 1.0])
    assert s.mean == 0.0
    assert s.cv_pct == 0.0
    assert not math.isnan(s.cv_pct)


def test_stats_over_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        stats_over([])


def test_is_divergent_below_threshold_returns_empty():
    # [1, 1, 1, 1, 1.05] — 1.05 is within 2*std of mean.
    assert is_divergent([1.0, 1.0, 1.0, 1.0, 1.05], z=2.0) == []


def test_is_divergent_outlier_flags_its_index():
    # [1, 1, 1, 1, 1, 10] — 10 is clearly > 2*std from mean.
    # (n>=6 needed so a single outlier's z-score can exceed 2 with ddof=1.)
    assert is_divergent([1.0, 1.0, 1.0, 1.0, 1.0, 10.0], z=2.0) == [5]


def test_is_divergent_higher_z_may_skip_outlier():
    # With z=3.0 the single outlier might not breach, depends on numbers.
    # Use a case that's 2-sigma (flags at z=2) but not 3-sigma (skipped at z=3).
    values = [1.0, 1.0, 1.0, 1.0, 1.0, 3.0]
    out_at_2 = is_divergent(values, z=2.0)
    out_at_3 = is_divergent(values, z=3.0)
    assert len(out_at_2) >= 1  # flags the outlier
    assert out_at_3 == []  # skips at higher threshold


def test_is_divergent_n_less_than_three_returns_empty():
    assert is_divergent([1.0], z=2.0) == []
    assert is_divergent([1.0, 100.0], z=2.0) == []
