# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test for scripts/benchmarks/training.py with --rl_library sb3."""

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

_TASK = "Isaac-Cartpole-Direct-v0"


def test_training_sb3_writes_training_bundle(tmp_path):
    sh = ROOT / "isaaclab.sh"
    if not sh.exists():
        pytest.skip("isaaclab.sh not found")
    cmd = [
        str(sh),
        "-p",
        "scripts/benchmarks/training.py",
        "--rl_library",
        "sb3",
        "--task",
        _TASK,
        "--num_envs",
        "16",
        "--max_iterations",
        "5",
        "--seed",
        "0",
        "--output_path",
        str(tmp_path),
        "presets=newton_mjwarp",
        "--headless",
    ]
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    out = tmp_path / f"training_{_TASK}.json"
    if res.returncode != 0 or not out.exists():
        blob = (res.stdout + res.stderr).lower()
        env_markers = (
            "isaacsim",
            "isaac sim",
            "no module named",
            "no registered env",
            "registrationerror",
        )
        if any(m in blob for m in env_markers):
            pytest.skip(f"Isaac Sim / task registry unavailable in this env:\n{res.stderr[-1500:]}")
        pytest.fail(f"training.py rc={res.returncode}\nSTDOUT:\n{res.stdout[-2000:]}\nSTDERR:\n{res.stderr[-2000:]}")
    data = json.loads(out.read_text())
    assert data["schema_version"] == "1.0"
    assert data["run"]["framework"] == "sb3"
    assert data["run"]["config"]["physics_backend"] == "newton_mjwarp"
    assert 1 <= data["runtime"]["iterations_completed"] <= 5
    assert data["runtime"]["total_fps"]["mean"] > 0
    assert data["learning"]["reward"]["series_per_iter"] is not None
    assert len(data["learning"]["reward"]["series_per_iter"]) >= 1
    assert data["learning"]["reward"]["final_ema"] is not None
