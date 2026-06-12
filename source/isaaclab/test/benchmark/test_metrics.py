# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the benchmark metrics helpers (Isaac-Sim-free)."""

import pytest

from isaaclab.test.benchmark.metrics import (
    SUCCESS_RATE_LOG_TAGS,
    SuccessRateTracker,
    check_convergence,
    ema,
    get_success_rate_log,
    mean_std,
    mean_std_peak,
)
from isaaclab.test.benchmark.schema import MeanStd


def test_mean_std_peak_computes_peak():
    ms = mean_std_peak([1.0, 2.0, 3.0])
    assert isinstance(ms, MeanStd)
    assert ms.mean == pytest.approx(2.0)
    assert ms.std == pytest.approx(1.0)
    assert ms.peak == pytest.approx(3.0)


def test_mean_std_omits_peak():
    ms = mean_std([10.0, 20.0])
    assert ms.peak is None
    assert ms.mean == pytest.approx(15.0)


def test_mean_std_empty_is_zero():
    ms = mean_std_peak([])
    assert ms.mean == 0.0 and ms.std == 0.0 and ms.peak == 0.0


def test_ema_matches_manual():
    series = [0.0, 10.0, 10.0]
    a = 0.5
    e = series[0]
    for x in series[1:]:
        e = a * x + (1 - a) * e
    assert ema(series, a) == pytest.approx(e)


def test_ema_empty_is_zero():
    assert ema([], 0.1) == 0.0


def test_check_convergence_passes_on_stable_high_rewards():
    res = check_convergence([100.0] * 10, threshold=50.0)
    assert res["passed"] is True
    assert res["tail_mean"] == pytest.approx(100.0)


def test_check_convergence_fails_when_below_threshold():
    res = check_convergence([1.0] * 10, threshold=50.0)
    assert res["passed"] is False


def test_get_success_rate_log_prefers_first_tag():
    assert SUCCESS_RATE_LOG_TAGS[0] == "Metrics/success_rate"
    data = {"Episode/Metrics/success_rate": [0.5], "Metrics/success_rate": [0.9]}
    assert get_success_rate_log(data) == [0.9]
    assert get_success_rate_log({}) is None


def test_success_rate_tracker_convergence():
    t = SuccessRateTracker(threshold=0.5, window=2, num_steps_per_env=1)
    for v in (0.6, 0.7):
        t.record_step({"log": {"Metrics/success_rate": v}})
        t.end_iteration()
    assert t.converged is True
    assert t.tail_mean == pytest.approx(0.65)
