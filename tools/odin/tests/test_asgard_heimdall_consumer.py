# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ``_consume_heimdall_snapshot`` in the runner main loop."""

from __future__ import annotations

from dataclasses import dataclass

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
from tools.odin.asgard.heimdall import HeimdallSnapshot, HostHealth, StaleJob
from tools.odin.asgard.jobs import JobEntry
from tools.odin.asgard.recovery import RecoveryResult
from tools.odin.asgard.runner import _consume_heimdall_snapshot
from tools.odin.asgard.state import SCHEMA_VERSION, DispatchState, FleetSnapshot
from tools.odin.asgard.transport import SSHResult


@dataclass
class _OkSSH:
    def run(self, host, cmd, *, timeout_s=None, stdout_tee=None, pty=True):
        return SSHResult(exit_code=0, stdout="GPU 0\n", stderr="", duration_s=0.01)


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


def _state(jobs, fleet_snap):
    return DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id="d",
        started_at="2026-05-08T14:00:00Z",
        ended_at=None,
        seeds=[0],
        commit_sha="x",
        fleet=fleet_snap,
        jobs=jobs,
    )


def _hh(name: str, healthy: bool) -> HostHealth:
    return HostHealth(
        name=name,
        healthy=healthy,
        last_probe_at="2026-05-08T14:01:00Z",
        consecutive_failures=0 if healthy else 2,
        failure_reason=None if healthy else "ssh_timeout",
        recovery_attempts=0,
        recovery_history=[],
        quarantined=False,
    )


def _snap(hosts, stale_jobs):
    return HeimdallSnapshot(
        generated_at="2026-05-08T14:01:00Z",
        hosts=hosts,
        stale_jobs=stale_jobs,
        recent_events=[],
    )


def test_no_flip_no_action_idempotent_marks_consumed():
    fleet = Fleet(fleet_name="t", hosts=[ValkyrieConfig(host="a", ssh_user="u")])
    state = _state(
        [_running_job("r1", "a")],
        [FleetSnapshot(host="a", status="busy", current_run_id="r1")],
    )
    snap = _snap({"a": _hh("a", True)}, [])
    last_consumed = [None]

    def setter(v):
        last_consumed[0] = v

    _consume_heimdall_snapshot(
        snap,
        state,
        fleet,
        ssh=_OkSSH(),
        last_consumed_at=last_consumed[0],
        set_last_consumed=setter,
        recover_fn=lambda h, ssh: RecoveryResult(
            host=h.host,
            container_name=h.container_name,
            attempted=False,
            recovered=False,
            duration_s=0.0,
            message="not invoked",
        ),
    )
    assert last_consumed[0] == snap.generated_at
    assert state.jobs[0].status == "running"
    assert state.quarantined_hosts == []


def test_flip_with_successful_recovery_clears_failure():
    fleet = Fleet(fleet_name="t", hosts=[ValkyrieConfig(host="a", ssh_user="u")])
    state = _state(
        [_running_job("r1", "a")],
        [FleetSnapshot(host="a", status="busy", current_run_id="r1")],
    )
    state._heimdall_host_state = {"a": _hh("a", True)}
    snap = _snap({"a": _hh("a", False)}, [])
    last_consumed = [None]

    def setter(v):
        last_consumed[0] = v

    recovered = RecoveryResult(
        host="a",
        container_name="isaac-lab-base",
        attempted=True,
        recovered=True,
        duration_s=5.0,
        message="recovered_via_container_restart",
    )

    _consume_heimdall_snapshot(
        snap,
        state,
        fleet,
        ssh=_OkSSH(),
        last_consumed_at=last_consumed[0],
        set_last_consumed=setter,
        recover_fn=lambda h, ssh: recovered,
    )
    assert state.jobs[0].status == "running"
    assert state.quarantined_hosts == []


def test_flip_with_failed_recovery_quarantines_and_requeues_jobs():
    fleet = Fleet(fleet_name="t", hosts=[ValkyrieConfig(host="a", ssh_user="u")])
    state = _state(
        [_running_job("r1", "a"), _running_job("r2", "a")],
        [FleetSnapshot(host="a", status="busy", current_run_id="r1")],
    )
    state._heimdall_host_state = {"a": _hh("a", True)}
    snap = _snap({"a": _hh("a", False)}, [])
    failed = RecoveryResult(
        host="a",
        container_name="isaac-lab-base",
        attempted=True,
        recovered=False,
        duration_s=5.0,
        message="docker_restart_failed: x",
    )
    last_consumed = [None]

    def setter(v):
        last_consumed[0] = v

    _consume_heimdall_snapshot(
        snap,
        state,
        fleet,
        ssh=_OkSSH(),
        last_consumed_at=last_consumed[0],
        set_last_consumed=setter,
        recover_fn=lambda h, ssh: failed,
    )
    assert state.jobs[0].status == "pending"
    assert state.jobs[1].status == "pending"
    assert "a" in state.jobs[0].preferred_not
    assert "a" in state.jobs[1].preferred_not
    assert len(state.quarantined_hosts) == 1
    assert state.quarantined_hosts[0].host == "a"


def test_idempotent_consumption_skips_duplicate_snapshot():
    fleet = Fleet(fleet_name="t", hosts=[ValkyrieConfig(host="a", ssh_user="u")])
    state = _state(
        [_running_job("r1", "a")],
        [FleetSnapshot(host="a", status="busy", current_run_id="r1")],
    )
    state._heimdall_host_state = {"a": _hh("a", True)}
    snap = _snap({"a": _hh("a", False)}, [])
    failed = RecoveryResult(
        host="a",
        container_name="isaac-lab-base",
        attempted=True,
        recovered=False,
        duration_s=5.0,
        message="x",
    )
    calls: list[str] = []

    def recover_fn(h, ssh):
        calls.append(h.host)
        return failed

    last_consumed = [None]

    def setter(v):
        last_consumed[0] = v

    _consume_heimdall_snapshot(
        snap,
        state,
        fleet,
        ssh=_OkSSH(),
        last_consumed_at=last_consumed[0],
        set_last_consumed=setter,
        recover_fn=recover_fn,
    )
    _consume_heimdall_snapshot(
        snap,
        state,
        fleet,
        ssh=_OkSSH(),
        last_consumed_at=last_consumed[0],
        set_last_consumed=setter,
        recover_fn=recover_fn,
    )
    assert calls == ["a"]


def _stale_snap(hosts, stale):
    return HeimdallSnapshot(
        generated_at="2026-05-08T14:01:00Z",
        hosts=hosts,
        stale_jobs=stale,
        recent_events=[],
    )


def _no_recover(h, ssh):
    return RecoveryResult(
        host=h.host,
        container_name=h.container_name,
        attempted=False,
        recovered=False,
        duration_s=0.0,
        message="not invoked",
    )


def test_stale_job_with_healthy_host_marks_failed_timeout():
    fleet = Fleet(fleet_name="t", hosts=[ValkyrieConfig(host="a", ssh_user="u")])
    state = _state(
        [_running_job("r1", "a")],
        [FleetSnapshot(host="a", status="busy", current_run_id="r1")],
    )
    state._heimdall_host_state = {"a": _hh("a", True)}
    snap = _stale_snap(
        {"a": _hh("a", True)},
        [
            StaleJob(
                run_id="r1",
                host="a",
                last_heartbeat_at="2026-05-08T13:55:00Z",
                age_seconds=360.0,
                host_was_healthy=True,
            )
        ],
    )
    last_consumed = [None]
    kill_calls: list[str] = []

    def kill_fn(host, run_id, ssh, *, timeout_s):
        kill_calls.append(run_id)

    def setter(v):
        last_consumed[0] = v

    _consume_heimdall_snapshot(
        snap,
        state,
        fleet,
        ssh=_OkSSH(),
        last_consumed_at=last_consumed[0],
        set_last_consumed=setter,
        recover_fn=_no_recover,
        kill_fn=kill_fn,
    )
    assert kill_calls == ["r1"]
    assert state.jobs[0].status == "failed"
    assert state.jobs[0].failure is not None
    assert state.jobs[0].failure.kind == "timeout"


def test_stale_job_with_unhealthy_host_requeues_as_infrastructure():
    fleet = Fleet(fleet_name="t", hosts=[ValkyrieConfig(host="a", ssh_user="u")])
    state = _state(
        [_running_job("r1", "a")],
        [FleetSnapshot(host="a", status="busy", current_run_id="r1")],
    )
    state._heimdall_host_state = {"a": _hh("a", False)}  # already unhealthy — no fresh flip
    snap = _stale_snap(
        {"a": _hh("a", False)},
        [
            StaleJob(
                run_id="r1",
                host="a",
                last_heartbeat_at="2026-05-08T13:55:00Z",
                age_seconds=360.0,
                host_was_healthy=False,
            )
        ],
    )
    last_consumed = [None]

    def setter(v):
        last_consumed[0] = v

    def kill_fn(host, run_id, ssh, *, timeout_s):
        pass

    _consume_heimdall_snapshot(
        snap,
        state,
        fleet,
        ssh=_OkSSH(),
        last_consumed_at=last_consumed[0],
        set_last_consumed=setter,
        recover_fn=_no_recover,
        kill_fn=kill_fn,
    )
    assert state.jobs[0].status == "pending"
    assert "a" in state.jobs[0].preferred_not


def test_stale_job_skipped_if_already_terminal():
    fleet = Fleet(fleet_name="t", hosts=[ValkyrieConfig(host="a", ssh_user="u")])
    j = _running_job("r1", "a")
    j.transition_to("completed", now="2026-05-08T13:59:00Z")
    state = _state(
        [j],
        [FleetSnapshot(host="a", status="idle", current_run_id=None)],
    )
    state._heimdall_host_state = {"a": _hh("a", True)}
    snap = _stale_snap(
        {"a": _hh("a", True)},
        [
            StaleJob(
                run_id="r1",
                host="a",
                last_heartbeat_at="2026-05-08T13:55:00Z",
                age_seconds=360.0,
                host_was_healthy=True,
            )
        ],
    )
    last_consumed = [None]
    kill_calls: list[str] = []

    def kill_fn(host, run_id, ssh, *, timeout_s):
        kill_calls.append(run_id)

    def setter(v):
        last_consumed[0] = v

    _consume_heimdall_snapshot(
        snap,
        state,
        fleet,
        ssh=_OkSSH(),
        last_consumed_at=last_consumed[0],
        set_last_consumed=setter,
        recover_fn=_no_recover,
        kill_fn=kill_fn,
    )
    assert kill_calls == []
    assert state.jobs[0].status == "completed"
