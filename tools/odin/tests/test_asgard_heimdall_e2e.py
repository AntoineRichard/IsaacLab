# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end Heimdall integration test against a loopback fleet.

Skipped when ``ssh localhost`` is unavailable. Verifies that a host
flipping unhealthy mid-dispatch is detected by the watcher.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
from tools.odin.asgard.heimdall import HeimdallWatcher
from tools.odin.asgard.state import SCHEMA_VERSION, DispatchState
from tools.odin.asgard.transport import ShellSSHRunner

pytestmark = pytest.mark.slow


def _ssh_localhost_works() -> bool:
    if shutil.which("ssh") is None:
        return False
    try:
        r = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                "ConnectTimeout=5",
                "localhost",
                "echo ok",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return r.returncode == 0 and "ok" in r.stdout
    except subprocess.SubprocessError:
        return False


def test_heimdall_detects_flipped_host_e2e(tmp_path):
    if not _ssh_localhost_works():
        pytest.skip("ssh localhost not available in this environment")

    sentinel = tmp_path / "host_a_alive"
    sentinel.write_text("alive")

    # The probe runs `docker exec <container_name> nvidia-smi -L`. We don't
    # have docker on the test host; the trick is to abuse `container_name`
    # to inject a sentinel-file check that emits "GPU 0" if the file
    # exists, exits non-zero otherwise. Trailing `#` comments out the
    # tail of the original command (the literal `nvidia-smi -L` token).
    host_a = ValkyrieConfig(
        host="localhost",
        ssh_user=os.environ.get("USER", "root"),
        ssh_key=None,
        isaaclab_path=str(tmp_path),
        container_name=f"sh -c 'test -f {sentinel} && echo GPU 0 || exit 1' #",
    )
    fleet = Fleet(fleet_name="t", hosts=[host_a])

    state = DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id="d",
        started_at="2026-05-08T14:00:00Z",
        ended_at=None,
        seeds=[0],
        commit_sha="x",
        fleet=[],
        jobs=[],
    )

    w = HeimdallWatcher(
        fleet=fleet,
        dispatch_dir=tmp_path,
        ssh=ShellSSHRunner(),
        state_view=lambda: state,
        probe_interval_s=10000,
        stale_threshold_s=180,
        flip_after_k_failures=2,
        probe_timeout_s=10,
    )

    # Tick 1: sentinel present → healthy.
    w._tick_once()
    snap1 = w.latest()
    assert snap1 is not None and snap1.hosts["localhost"].healthy is True

    # Remove sentinel — next probes will fail (exit 1).
    sentinel.unlink()

    w._tick_once()
    w._tick_once()
    snap3 = w.latest()
    assert snap3 is not None
    assert snap3.hosts["localhost"].healthy is False
    # fleet.json was written each tick.
    assert (Path(tmp_path) / "fleet.json").exists()
