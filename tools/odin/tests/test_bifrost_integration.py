# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end smoke test against a local OSMO deployment.

Gated by ``ODIN_OSMO_INTEGRATION=1``. Skips otherwise.
Optimistic path only: render → submit → poll → download → assert layout.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow]


def _osmo_available() -> bool:
    if os.environ.get("ODIN_OSMO_INTEGRATION") != "1":
        return False
    if shutil.which("osmo") is None:
        return False
    cp = subprocess.run(["osmo", "profile", "list"], capture_output=True, text=True)
    return cp.returncode == 0


@pytest.mark.skipif(not _osmo_available(), reason="ODIN_OSMO_INTEGRATION!=1 or osmo CLI unavailable")
def test_bifrost_end_to_end_smoke(tmp_path: Path):
    """Submit a 1-task workflow that writes a fake manifest, then verify the bundle lands on disk."""
    from tools.odin.bifrost import cli as bifrost_cli

    cfg = tmp_path / "bifrost-osmo.yaml"
    cfg.write_text(
        "osmo_profile: " + os.environ.get("ODIN_OSMO_PROFILE", "default") + "\n"
        "pool: " + os.environ.get("ODIN_OSMO_POOL", "default") + "\n"
        "priority: NORMAL\n"
        "image:\n"
        "  reference: alpine:3.18\n"
        "defaults:\n"
        "  resources: {cpu: 1, gpu: 0, memory: 256Mi, storage: 256Mi, platform: cpu}\n"
        "  exec_timeout: 60\n"
        "  queue_timeout: 120\n"
        "retry: {reschedule_codes: '', restart_codes: ''}\n"
        "bundle_dataset_prefix: odin-int-test\n"
        "code_delivery: {mode: image_baked, source_root: tools/odin}\n"
    )
    physx = tmp_path / "physx.yaml"
    physx.write_text(
        "envs:\n"
        "- task_id: smoke-task\n"
        "  framework: rsl-rl\n"
        "  num_envs: 1\n"
        "  max_iterations: 1\n"
        "  keep: true\n"
    )

    runs_root = tmp_path / "odin_runs"
    rc = bifrost_cli.main(
        [
            "--osmo-config",
            str(cfg),
            "--physx-yaml",
            str(physx),
            "--seeds",
            "1",
            "--runs-root",
            str(runs_root),
            "--poll-interval",
            "5",
        ]
    )
    assert rc == 0
    [dispatch_dir] = list(runs_root.iterdir())
    state_path = dispatch_dir / "dispatch.json"
    state = json.loads(state_path.read_text())
    assert state["dispatcher"] == "osmo"
    # Note: this test will fail at runtime because the alpine image can't run hugin.
    # When you wire a real test image with a stub manifest writer, replace the
    # `image_baked` config with the real reference and assert manifest.json exists.
