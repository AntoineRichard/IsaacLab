# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test for scripts/benchmarks/startup.py (Newton/MJWarp = Isaac-Sim-free)."""

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

# Task id tracks this branch's task naming. Branches that still carry the legacy
# ``-v0`` suffix use ``Isaac-Cartpole-Direct-v0``; update to ``Isaac-Cartpole-Direct``
# once rebased onto a ``develop`` that has dropped the suffix.
_TASK = "Isaac-Cartpole-Direct-v0"

_EXPECTED_PHASES = {"app_launch", "python_imports", "task_config", "env_creation", "first_step"}


def test_startup_writes_startup_bundle(tmp_path):
    sh = ROOT / "isaaclab.sh"
    if not sh.exists():
        pytest.skip("isaaclab.sh not found")
    cmd = [
        str(sh),
        "-p",
        "scripts/benchmarks/startup.py",
        "--task",
        _TASK,
        "--num_envs",
        "16",
        "--seed",
        "0",
        "--output_path",
        str(tmp_path),
        "presets=newton_mjwarp",
        "--headless",
    ]
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    out = tmp_path / f"startup_{_TASK}.json"
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
        pytest.fail(f"startup.py rc={res.returncode}\nSTDOUT:\n{res.stdout[-2000:]}\nSTDERR:\n{res.stderr[-2000:]}")

    data = json.loads(out.read_text())

    # Top-level schema
    assert data["schema_version"] == "1.0", f"unexpected schema_version: {data['schema_version']}"
    assert data["run"]["framework"] is None, "startup bundle should have framework=null"

    # All five phases must be present
    assert _EXPECTED_PHASES == set(data["phases"].keys()), (
        f"unexpected phases: {set(data['phases'].keys())}"
    )

    # Each phase must have a positive total_time_s
    for phase_name, phase in data["phases"].items():
        assert phase["total_time_s"] > 0, f"phase '{phase_name}' has total_time_s <= 0"

    # At least one phase must have top_functions with a valid 'calls' int
    has_calls = False
    for phase in data["phases"].values():
        for fn in phase.get("top_functions", []):
            if isinstance(fn.get("calls"), int):
                has_calls = True
                break
        if has_calls:
            break
    assert has_calls, "No top_functions entry with an integer 'calls' field found"

    # Config block must be present with top_n
    assert "config" in data, "StartupBundle missing 'config' field"
    assert isinstance(data["config"]["top_n"], int), "config.top_n should be an integer"
