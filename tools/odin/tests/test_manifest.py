# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Odin manifest writer."""

import json
import os
from datetime import datetime, timezone

from isaaclab.test.benchmark.standard_schema import ManifestPhase

from tools.odin.common.manifest import write_manifest


def test_write_manifest_minimal(tmp_path):
    start = datetime(2026, 4, 22, 13, 15, 0, tzinfo=timezone.utc)
    end = datetime(2026, 4, 22, 13, 47, 48, tzinfo=timezone.utc)
    path = write_manifest(
        bundle_dir=str(tmp_path),
        run_id="rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-131500_seed42",
        framework="rsl_rl",
        backend="physx",
        task="Isaac-Ant-Direct-v0",
        seed=42,
        num_envs=4096,
        max_iterations=500,
        run_start_dt=start,
        run_end_dt=end,
        startup_phase=ManifestPhase(file="startup.json", status="completed", duration_s=48.7, exit_code=0),
        training_phase=ManifestPhase(file="training.json", status="completed", duration_s=1942.1, exit_code=0),
        repo_root=str(tmp_path),  # not a git repo, so git info is None
    )
    assert os.path.exists(path)
    with open(path) as f:
        data = json.load(f)
    assert data["schema_version"] == "1.0"
    assert data["run_id"].startswith("rsl-rl_physx_")
    assert data["phases"]["training"]["exit_code"] == 0
    assert data["machine"]["git_commit"] is None
