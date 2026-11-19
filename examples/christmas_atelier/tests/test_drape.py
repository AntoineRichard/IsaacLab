# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise the calibration CLI with cloth and static shapes but no rigid bodies."""

import json
import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.kitless, pytest.mark.integration]


def test_drape_without_robot_exports_and_reports_an_incomplete_check(tmp_path):
    """Cloth-only states have no body_q or joint targets; short runs must fail --check."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.demos.christmas_atelier",
            "--mode",
            "drape",
            "--visualizer",
            "none",
            "--device",
            "cpu",
            "--max_steps",
            "2",
            "--substeps",
            "2",
            "--iterations",
            "2",
            "--mesh-rings",
            "8",
            "--mesh-sides",
            "8",
            "--metrics",
            str(tmp_path / "metrics.json"),
            "--output",
            str(tmp_path / "drape.usd"),
            "--check",
        ],
        env={**os.environ, "WARP_CACHE_PATH": os.environ.get("WARP_CACHE_PATH", str(tmp_path / "warp"))},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode != 0
    assert "Physical sequence did not pass" in result.stderr, result.stdout + result.stderr
    report = json.loads((tmp_path / "metrics.json").read_text())
    assert report["frames"] == 2
    assert report["cloth_finite"]
    assert report["cuff_drift_m"] < 1.0e-6
    assert not report["complete"] and not report["passed"]
    assert (tmp_path / "drape.usd").stat().st_size > 0
