# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the calibration script's pure-Python helpers."""

from __future__ import annotations

import pytest

from tools.odin.scripts.calibrate_max_iterations import recommend_max_iterations


def test_recommend_fits_inside_budget():
    """Budget = 75% of timeout; subtract startup; divide by per_iter."""
    rec = recommend_max_iterations(
        per_iter_s=1.0, startup_s=60.0, per_job_timeout_s=3600
    )
    # 0.75 * 3600 = 2700 budget, minus 60 startup = 2640 headroom, / 1.0 = 2640.
    assert rec == 2640


def test_recommend_floors_to_int():
    """Truncation, not rounding."""
    rec = recommend_max_iterations(
        per_iter_s=2.5, startup_s=10.0, per_job_timeout_s=100
    )
    # (75 - 10) / 2.5 = 26.0
    assert rec == 26


def test_recommend_negative_budget_raises():
    """Startup alone exceeds 75% of timeout — task can't fit at any iter count."""
    with pytest.raises(ValueError, match="exceeds budget"):
        recommend_max_iterations(per_iter_s=1.0, startup_s=3000.0, per_job_timeout_s=3600)


def test_recommend_zero_per_iter_raises():
    with pytest.raises(ValueError, match="per_iter_s"):
        recommend_max_iterations(per_iter_s=0.0, startup_s=10.0, per_job_timeout_s=100)
