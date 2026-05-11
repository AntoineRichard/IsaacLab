# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stale-job computation in :meth:`HeimdallWatcher._compute_stale_jobs`."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
from tools.odin.asgard.heimdall import HeimdallWatcher, HostHealth
from tools.odin.asgard.jobs import JobEntry
from tools.odin.asgard.state import SCHEMA_VERSION, DispatchState
from tools.odin.asgard.transport import SSHResult


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _running_job(
    run_id: str,
    host: str,
    *,
    last_heartbeat_at: str | None,
    started_at: str,
) -> JobEntry:
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
    j.transition_to("running", assigned_to=host, now=started_at)
    j.last_heartbeat_at = last_heartbeat_at
    return j


@dataclass
class _OkSSH:
    def run(self, host, cmd, *, timeout_s=None, stdout_tee=None, pty=True):
        return SSHResult(exit_code=0, stdout="GPU 0\n", stderr="", duration_s=0.01)


def _make_watcher(tmp_path, jobs):
    fleet = Fleet(fleet_name="t", hosts=[ValkyrieConfig(host="host-a", ssh_user="u")])
    state = DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id="d",
        started_at="2026-05-08T14:00:00Z",
        ended_at=None,
        seeds=[0],
        commit_sha="x",
        fleet=[],
        jobs=jobs,
    )
    return HeimdallWatcher(
        fleet=fleet,
        dispatch_dir=tmp_path,
        ssh=_OkSSH(),
        state_view=lambda: state,
        probe_interval_s=10000,
        stale_threshold_s=180,
        flip_after_k_failures=2,
        probe_timeout_s=5,
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


def test_no_stale_jobs_when_heartbeat_is_fresh(tmp_path):
    now = datetime.now(timezone.utc)
    fresh = _iso(now - timedelta(seconds=30))
    job = _running_job("run-1", "host-a", last_heartbeat_at=fresh, started_at=fresh)

    w = _make_watcher(tmp_path, jobs=[job])
    assert w._compute_stale_jobs({"host-a": _hh("host-a", True)}) == []


def test_stale_when_heartbeat_older_than_threshold(tmp_path):
    now = datetime.now(timezone.utc)
    old = _iso(now - timedelta(seconds=240))
    job = _running_job("run-1", "host-a", last_heartbeat_at=old, started_at=old)

    w = _make_watcher(tmp_path, jobs=[job])
    stale = w._compute_stale_jobs({"host-a": _hh("host-a", True)})

    assert len(stale) == 1
    assert stale[0].run_id == "run-1"
    assert stale[0].host == "host-a"
    assert stale[0].host_was_healthy is True
    assert stale[0].age_seconds >= 180


def test_stale_with_unhealthy_host_reports_branch(tmp_path):
    now = datetime.now(timezone.utc)
    old = _iso(now - timedelta(seconds=240))
    job = _running_job("run-1", "host-a", last_heartbeat_at=old, started_at=old)

    w = _make_watcher(tmp_path, jobs=[job])
    stale = w._compute_stale_jobs({"host-a": _hh("host-a", False)})

    assert stale[0].host_was_healthy is False


def test_stale_uses_started_at_when_no_heartbeat_yet(tmp_path):
    """Pre-heartbeat resume jobs use ``started_at`` as the staleness baseline."""
    now = datetime.now(timezone.utc)
    old = _iso(now - timedelta(seconds=240))
    job = _running_job("run-1", "host-a", last_heartbeat_at=None, started_at=old)

    w = _make_watcher(tmp_path, jobs=[job])
    stale = w._compute_stale_jobs({"host-a": _hh("host-a", True)})

    assert len(stale) == 1
    assert stale[0].last_heartbeat_at == old


def test_only_running_jobs_are_evaluated(tmp_path):
    """``pending`` / ``completed`` / ``failed`` jobs are never stale."""
    pending = JobEntry(
        run_id="p",
        task_id="t",
        framework="rsl_rl",
        backend="physx",
        num_envs=1,
        max_iterations=1,
        seed=0,
        bundle_dir_name="p",
    )

    w = _make_watcher(tmp_path, jobs=[pending])
    assert w._compute_stale_jobs({"host-a": _hh("host-a", True)}) == []
