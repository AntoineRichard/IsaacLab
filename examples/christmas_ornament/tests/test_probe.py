# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The handoff qualifier: the Agile One hangs the ornament on a standalone branch, by contact only."""

import json
import os
import subprocess
import sys

import pytest
import warp as wp

from examples.christmas_ornament.assets import AGILE_URDF_ENV

pytestmark = [pytest.mark.kitless]


@pytest.mark.skipif(not wp.is_cuda_available(), reason="needs CUDA for the full probe rollout")
@pytest.mark.skipif(not os.environ.get(AGILE_URDF_ENV), reason=f"{AGILE_URDF_ENV} not set (internal asset)")
def test_probe_check_passes(tmp_path):
    metrics = tmp_path / "probe.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.christmas_ornament.run",
            "--mode",
            "probe",
            "--viewer",
            "null",
            "--check",
            "--metrics",
            str(metrics),
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    data = json.loads(metrics.read_text())
    assert data["hung"] is True
    assert data["released_clearance_m"] > 0.005
    assert data["max_hook_error_m"] < 0.025
    assert data["dip_m"] >= 0.003
    assert data["finite"] is True
