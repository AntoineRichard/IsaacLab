# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Slow-marked integration test: enumerate scripts end-to-end.

Runs both ``enumerate_physx_envs.py`` and ``enumerate_newton_envs.py``
against the live IsaacLab registry, writing into a tmpdir. Asserts the
three YAMLs parse and are internally consistent. Does NOT assert row
content — the registry changes over time.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.odin.common.env_list import load_env_list

REPO_ROOT = Path(__file__).resolve().parents[3]
ISAACLAB_SH = REPO_ROOT / "isaaclab.sh"


@pytest.mark.slow
def test_enumerate_pipeline_end_to_end(tmp_path: Path):
    physx_out = tmp_path / "physx_envs.yaml"
    newton_out = tmp_path / "newton_envs.yaml"
    gap_out = tmp_path / "newton_gap_candidates.yaml"

    # Inherit env plus PYTHONPATH=. so `tools.odin.*` imports resolve.
    import os

    child_env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}

    # --- Step 1: enumerate PhysX envs ---
    result = subprocess.run(
        [
            str(ISAACLAB_SH),
            "-p",
            "tools/odin/scripts/enumerate_physx_envs.py",
            "--output-path",
            str(physx_out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        env=child_env,
    )
    assert result.returncode == 0, (
        f"enumerate_physx_envs.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert physx_out.exists(), "PhysX output YAML was not created"

    physx = load_env_list(physx_out)
    assert physx.groups, "PhysX env list is empty — registry unpopulated?"

    # --- Step 2: enumerate Newton envs ---
    result = subprocess.run(
        [
            str(ISAACLAB_SH),
            "-p",
            "tools/odin/scripts/enumerate_newton_envs.py",
            "--physx-input",
            str(physx_out),
            "--newton-output",
            str(newton_out),
            "--gap-output",
            str(gap_out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        env=child_env,
    )
    assert result.returncode == 0, (
        f"enumerate_newton_envs.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert newton_out.exists()
    assert gap_out.exists()

    newton = load_env_list(newton_out)
    gaps = load_env_list(gap_out)

    # --- Consistency: every Newton/gap row was kept in PhysX ---
    physx_by_id = {e.task_id: e for rows in physx.groups.values() for e in rows}
    for rows in newton.groups.values():
        for e in rows:
            assert e.task_id in physx_by_id, f"Newton row {e.task_id} not in PhysX YAML"
            assert physx_by_id[e.task_id].keep, f"Newton row {e.task_id} was kept in physx but keep=False there"

    # --- Consistency: no row appears in both Newton and gap lists ---
    newton_ids = {e.task_id for rows in newton.groups.values() for e in rows}
    gap_ids = {e.task_id for rows in gaps.groups.values() for e in rows}
    assert not (newton_ids & gap_ids), (
        f"Rows appear in both newton_envs and gap_candidates: {sorted(newton_ids & gap_ids)}"
    )

    # --- Every gap row has suspected_gap set (tbd is fine for fresh runs) ---
    for rows in gaps.groups.values():
        for e in rows:
            assert e.suspected_gap is not None, f"Gap row {e.task_id} missing suspected_gap"
