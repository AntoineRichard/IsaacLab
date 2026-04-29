# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the calibration script's pure-Python helpers."""

from __future__ import annotations

import json

import pytest

from tools.odin.scripts.calibrate_max_iterations import (
    _parse_calibration_metrics,
    recommend_max_iterations,
)


def test_recommend_fits_inside_budget():
    """Budget = 75% of timeout; subtract startup; divide by per_iter."""
    rec = recommend_max_iterations(per_iter_s=1.0, startup_s=60.0, per_job_timeout_s=3600)
    # 0.75 * 3600 = 2700 budget, minus 60 startup = 2640 headroom, / 1.0 = 2640.
    assert rec == 2640


def test_recommend_floors_to_int():
    """Truncation, not rounding."""
    rec = recommend_max_iterations(per_iter_s=2.5, startup_s=10.0, per_job_timeout_s=100)
    # (75 - 10) / 2.5 = 26.0
    assert rec == 26


def test_recommend_negative_budget_raises():
    """Startup alone exceeds 75% of timeout — task can't fit at any iter count."""
    with pytest.raises(ValueError, match="exceeds budget"):
        recommend_max_iterations(per_iter_s=1.0, startup_s=3000.0, per_job_timeout_s=3600)


def test_recommend_zero_per_iter_raises():
    with pytest.raises(ValueError, match="per_iter_s"):
        recommend_max_iterations(per_iter_s=0.0, startup_s=10.0, per_job_timeout_s=100)


def test_parse_calibration_metrics_reads_real_schema_paths(tmp_path):
    """Verify the helper reads from runtime.iteration_time_s.mean and run.duration_s."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "training.json").write_text(
        json.dumps(
            {
                "runtime": {"iteration_time_s": {"mean": 1.5, "std": 0.1}},
            }
        )
    )
    (bundle / "startup.json").write_text(
        json.dumps(
            {
                "run": {"duration_s": 80.0},
            }
        )
    )
    per_iter_s, startup_s = _parse_calibration_metrics(bundle)
    assert per_iter_s == 1.5
    assert startup_s == 80.0
