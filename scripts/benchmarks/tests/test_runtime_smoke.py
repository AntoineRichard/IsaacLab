# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test for scripts/benchmarks/runtime.py (Newton/MJWarp = Isaac-Sim-free)."""

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

# Task id tracks this branch's task naming. Branches that still carry the legacy
# ``-v0`` suffix use ``Isaac-Cartpole-Direct-v0``; update to ``Isaac-Cartpole-Direct``
# once rebased onto a ``develop`` that has dropped the suffix.
_TASK = "Isaac-Cartpole-Direct-v0"


def test_runtime_writes_runtime_bundle(tmp_path):
    sh = ROOT / "isaaclab.sh"
    if not sh.exists():
        pytest.skip("isaaclab.sh not found")
    cmd = [
        str(sh),
        "-p",
        "scripts/benchmarks/runtime.py",
        "--task",
        _TASK,
        "--num_envs",
        "16",
        "--num_frames",
        "20",
        "--seed",
        "0",
        "--output_path",
        str(tmp_path),
        "presets=newton_mjwarp",
        "--headless",
    ]
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    out = tmp_path / f"runtime_{_TASK}.json"
    if res.returncode != 0 or not out.exists():
        blob = (res.stdout + res.stderr).lower()
        # Skip (rather than fail) when the environment can't run the sim or has an
        # inconsistent multi-worktree install: missing Isaac Sim, unimportable
        # packages, or a gym task that isn't registered in this checkout.
        env_markers = (
            "isaacsim",
            "isaac sim",
            "no module named",
            "no registered env",
            "registrationerror",
        )
        if any(m in blob for m in env_markers):
            pytest.skip(f"Isaac Sim / task registry unavailable in this env:\n{res.stderr[-1500:]}")
        pytest.fail(f"runtime.py rc={res.returncode}\nSTDOUT:\n{res.stdout[-2000:]}\nSTDERR:\n{res.stderr[-2000:]}")
    data = json.loads(out.read_text())
    assert data["schema_version"] == "1.0"
    assert data["run"]["framework"] is None
    assert "learning" not in data
    assert data["run"]["config"]["physics_backend"] == "newton_mjwarp"
    assert data["runtime"]["iterations_completed"] == 20
    assert data["runtime"]["total_fps"]["mean"] > 0
