# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration test: timeout enforcement leaves remote hosts in a clean state.

Gated on ``ODIN_TEST_FLEET=path/to/fleet.yaml``. Skipped otherwise.
Slow: ~20 min wall-clock. Run with::

    ODIN_TEST_FLEET=fleet.yaml \\
        PYTHONPATH=. python3 -m pytest \\
        tools/odin/tests/test_asgard_timeout_recovery.py \\
        --confcutdir=tools/odin -v -m slow

Verifies, on each host after a 5-minute timeout fires:
    (a) no orphan ``--run_id`` process alive,
    (b) container GPU memory back below 500 MiB,
    (c) container NVML still healthy,
    (d) ``dispatch.json`` job classified ``kind="timeout"``,
    (e) host NOT in ``quarantined_hosts`` (single timeout shouldn't trip
        the circuit-breaker),
    (f) a subsequent dispatch on the same host succeeds for a fast job.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from tools.odin.asgard.fleet import load_fleet
from tools.odin.asgard.runner import DispatchOptions, resolve_dispatch_dir, run_dispatch
from tools.odin.asgard.transport import ShellRsyncRunner, ShellSSHRunner

pytestmark = pytest.mark.slow

_FLEET_ENV = "ODIN_TEST_FLEET"


def _require_fleet() -> Path:
    val = os.environ.get(_FLEET_ENV)
    if not val:
        pytest.skip(f"{_FLEET_ENV} not set; skipping live-fleet integration test")
    p = Path(val)
    if not p.exists():
        pytest.skip(f"{_FLEET_ENV}={val} does not exist")
    return p


def _ssh_run(host, cmd: str, ssh) -> tuple[int, str]:
    r = ssh.run(host, cmd, timeout_s=15.0)
    return r.exit_code, r.stdout.strip()


def test_timeout_leaves_hosts_clean_then_recovers():
    """Integration test on a real fleet — see module docstring."""
    fleet_path = _require_fleet()
    fleet = load_fleet(fleet_path)
    ssh = ShellSSHRunner()
    rsync = ShellRsyncRunner()

    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td)

        # Phase 1: dispatch one Repose-Cube-Allegro seed with a 5-minute timeout.
        # This task takes >5 min to start, so it's guaranteed to time out.
        d1 = resolve_dispatch_dir(runs_root, resume=None)
        physx_yaml = Path("tools/odin/config/physx_envs.yaml")
        state1 = run_dispatch(
            fleet=fleet,
            physx_yaml=physx_yaml,
            newton_yaml=None,
            dispatch_dir=d1,
            options=DispatchOptions(
                seeds=[42],
                per_job_timeout_s=300,
                include_filter=["Isaac-Repose-Cube-Allegro-Direct-v0"],
                skip_aggregate=True,
            ),
            ssh=ssh,
            rsync=rsync,
        )

        # Assert at least one job hit failed/timeout.
        timeouts = [
            j for j in state1.jobs if j.status == "failed" and j.failure is not None and j.failure.kind == "timeout"
        ]
        assert timeouts, f"expected ≥1 timeout, got: {[(j.run_id, j.status) for j in state1.jobs]}"

        # Per host: assert clean state after timeout.
        for host in fleet.hosts:
            for j in timeouts:
                if j.assigned_to != host.host:
                    continue
                # (a) No process matching --run_id alive on the remote.
                rc, _ = _ssh_run(
                    host,
                    f"docker exec {host.container_name} pgrep -f -- '--run_id {j.run_id}'",
                    ssh,
                )
                assert rc != 0, f"orphan process for {j.run_id} alive on {host.host}"

                # (c) NVML healthy in the container.
                rc, out = _ssh_run(
                    host,
                    f"docker exec {host.container_name} nvidia-smi -L",
                    ssh,
                )
                assert rc == 0 and out, f"NVML wedged on {host.host} after timeout"

            # (b) GPU memory back to baseline (< 500 MiB).
            rc, out = _ssh_run(
                host,
                f"docker exec {host.container_name} nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
                ssh,
            )
            assert rc == 0
            mb = int(out.splitlines()[0].strip())
            assert mb < 500, f"GPU mem on {host.host} = {mb} MiB after timeout"

        # (e) Single timeout should NOT have tripped the circuit-breaker.
        for f in state1.fleet:
            assert f.status != "down", f"host {f.host} unexpectedly down after single timeout"

        # Phase 2: subsequent dispatch on same fleet, fast Cartpole job.
        d2 = resolve_dispatch_dir(runs_root, resume=None)
        state2 = run_dispatch(
            fleet=fleet,
            physx_yaml=physx_yaml,
            newton_yaml=None,
            dispatch_dir=d2,
            options=DispatchOptions(
                seeds=[42],
                per_job_timeout_s=900,
                include_filter=["Isaac-Cartpole-Direct-v0"],
                skip_aggregate=True,
            ),
            ssh=ssh,
            rsync=rsync,
        )

        # (f) The Cartpole job must complete on the same fleet.
        completed = [j for j in state2.jobs if j.status == "completed"]
        assert completed, (
            f"subsequent dispatch produced no completions: "
            f"{[(j.run_id, j.status, j.failure.kind if j.failure else None) for j in state2.jobs]}"
        )
