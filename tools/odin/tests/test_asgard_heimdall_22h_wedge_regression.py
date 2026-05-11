# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression — host wedges mid-dispatch must be detected and self-healed.

Models the 2026-04-30 incident where two hosts wedged for 22 hours and the
dispatcher kept assigning no work to them. Expected behaviour: Heimdall
detects the flip after K consecutive failures, attempts recovery, and on
recovery failure quarantines + re-queues all in-flight jobs.

Sanity check: comment out the ``_consume_heimdall_snapshot`` call below
to confirm the test fails (jobs remain ``running``) without the runner
integration.
"""

from __future__ import annotations

from dataclasses import dataclass

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
from tools.odin.asgard.heimdall import HeimdallWatcher
from tools.odin.asgard.jobs import JobEntry
from tools.odin.asgard.recovery import RecoveryResult
from tools.odin.asgard.runner import _consume_heimdall_snapshot
from tools.odin.asgard.state import SCHEMA_VERSION, DispatchState, FleetSnapshot
from tools.odin.asgard.transport import SSHResult


@dataclass
class _FlipsAfterTickSSH:
    """Healthy on first probe, then SSH-timed-out forever after."""

    tick: int = 0

    def run(self, host, cmd, *, timeout_s=None, stdout_tee=None, pty=True):
        self.tick += 1
        if self.tick == 1:
            return SSHResult(exit_code=0, stdout="GPU 0\n", stderr="", duration_s=0.01)
        return SSHResult(exit_code=255, stdout="", stderr="ssh: connect failed", duration_s=15.0, timed_out=True)


def _running_job(run_id: str, host: str) -> JobEntry:
    j = JobEntry(
        run_id=run_id,
        task_id="t",
        framework="rsl_rl",
        backend="physx",
        num_envs=1,
        max_iterations=1,
        seed=0,
        bundle_dir_name=run_id,
    )
    j.transition_to("running", assigned_to=host, now="2026-05-08T14:00:00Z")
    return j


def test_22h_wedge_is_detected_and_quarantined(tmp_path):
    fleet = Fleet(fleet_name="t", hosts=[ValkyrieConfig(host="host-a", ssh_user="u")])
    j1 = _running_job("r1", "host-a")
    j2 = _running_job("r2", "host-a")
    state = DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id="d",
        started_at="2026-05-08T14:00:00Z",
        ended_at=None,
        seeds=[0, 1],
        commit_sha="x",
        fleet=[FleetSnapshot(host="host-a", status="busy", current_run_id="r1")],
        jobs=[j1, j2],
    )

    ssh = _FlipsAfterTickSSH()
    w = HeimdallWatcher(
        fleet=fleet,
        dispatch_dir=tmp_path,
        ssh=ssh,
        state_view=lambda: state,
        probe_interval_s=10000,
        stale_threshold_s=180,
        flip_after_k_failures=2,
        probe_timeout_s=5,
    )

    w._tick_once()
    snap1 = w.latest()
    assert snap1.hosts["host-a"].healthy is True

    w._tick_once()
    snap2 = w.latest()
    assert snap2.hosts["host-a"].healthy is True
    assert snap2.hosts["host-a"].consecutive_failures == 1

    w._tick_once()
    snap3 = w.latest()
    assert snap3.hosts["host-a"].healthy is False

    failed = RecoveryResult(
        host="host-a",
        container_name="isaac-lab-base",
        attempted=True,
        recovered=False,
        duration_s=2.0,
        message="docker_restart_failed: x",
    )
    prev_state: dict = {}
    last_consumed = [None]

    def setter(v):
        last_consumed[0] = v

    def kill_fn(host, run_id, ssh, *, timeout_s):
        pass

    prev_state = dict(snap2.hosts)  # snap2 was the prior tick (healthy)
    _consume_heimdall_snapshot(
        snap3,
        state,
        fleet,
        prev_host_state=prev_state,
        ssh=ssh,
        last_consumed_at=last_consumed[0],
        set_last_consumed=setter,
        recover_fn=lambda h, ssh: failed,
        kill_fn=kill_fn,
    )

    assert state.jobs[0].status == "pending"
    assert state.jobs[1].status == "pending"
    assert "host-a" in state.jobs[0].preferred_not
    assert "host-a" in state.jobs[1].preferred_not
    assert len(state.quarantined_hosts) == 1
    assert state.quarantined_hosts[0].host == "host-a"
