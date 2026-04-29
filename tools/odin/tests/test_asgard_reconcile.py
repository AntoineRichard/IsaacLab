# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for tools.odin.asgard.reconcile.reconcile_orphans."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
from tools.odin.asgard.jobs import JobEntry
from tools.odin.asgard.reconcile import ReconcileOutcome, reconcile_orphans
from tools.odin.asgard.transport import RsyncResult, SSHResult


@dataclass
class _FakeSSH:
    scripted: dict
    calls: list = field(default_factory=list)

    def run(self, host, cmd, *, timeout_s=None, stdout_tee=None) -> SSHResult:
        self.calls.append((host.host, cmd))
        for key, result in self.scripted.items():
            if key in cmd:
                return result
        return SSHResult(exit_code=1, stdout="", stderr=f"no fake for {cmd!r}", duration_s=0.0)


@dataclass
class _FakeRsync:
    pulls: list = field(default_factory=list)

    def pull(self, host, remote_path, local_path) -> RsyncResult:
        self.pulls.append((host.host, remote_path, str(local_path)))
        return RsyncResult(exit_code=0, stdout="", stderr="", duration_s=0.01)

    def push(self, host, local_path, remote_path) -> RsyncResult:
        return RsyncResult(exit_code=0, stdout="", stderr="", duration_s=0.0)


def _host() -> ValkyrieConfig:
    return ValkyrieConfig(host="v1", ssh_user="odin", isaaclab_path="/home/odin/IsaacLab")


def _job(run_id: str = "r1", assigned_to: str | None = "v1") -> JobEntry:
    return JobEntry(
        run_id=run_id,
        task_id="Isaac-Ant-Direct-v0",
        framework="rsl_rl",
        backend="physx",
        num_envs=1024,
        max_iterations=100,
        seed=42,
        bundle_dir_name=run_id,
        status="running",
        assigned_to=assigned_to,
    )


def _manifest_completed() -> str:
    return json.dumps({
        "run_id": "r1",
        "phases": {
            "startup": {"status": "completed", "exit_code": 0},
            "training": {"status": "completed", "exit_code": 0},
        },
    })


def _manifest_failed() -> str:
    return json.dumps({
        "run_id": "r1",
        "phases": {
            "startup": {"status": "completed", "exit_code": 0},
            "training": {"status": "failed", "exit_code": 1},
        },
    })


def test_reconcile_completed_manifest_adopts_bundle(tmp_path: Path):
    """Branch (a): manifest exists + both phases completed → rsync, mark completed."""
    fleet = Fleet(fleet_name="t", hosts=[_host()])
    job = _job()
    ssh = _FakeSSH(scripted={
        "cat ": SSHResult(exit_code=0, stdout=_manifest_completed(), stderr="", duration_s=0.0),
    })
    rsync = _FakeRsync()

    outcomes = reconcile_orphans(fleet=fleet, jobs=[job], dispatch_dir=tmp_path, ssh=ssh, rsync=rsync)

    assert outcomes == [ReconcileOutcome(run_id="r1", action="adopted_completed")]
    assert job.status == "completed"
    assert len(rsync.pulls) == 1


def test_reconcile_failed_manifest_adopts_failure(tmp_path: Path):
    """Branch (b): manifest exists but a phase failed → rsync, mark failed."""
    fleet = Fleet(fleet_name="t", hosts=[_host()])
    job = _job()
    ssh = _FakeSSH(scripted={
        "cat ": SSHResult(exit_code=0, stdout=_manifest_failed(), stderr="", duration_s=0.0),
    })
    rsync = _FakeRsync()

    outcomes = reconcile_orphans(fleet=fleet, jobs=[job], dispatch_dir=tmp_path, ssh=ssh, rsync=rsync)

    assert outcomes == [ReconcileOutcome(run_id="r1", action="adopted_failed")]
    assert job.status == "failed"
    assert job.failure is not None
    assert job.failure.kind == "hugin_crash"


def test_reconcile_alive_no_manifest_kills_and_pending(tmp_path: Path):
    """Branch (c): no manifest, process alive → kill, mark pending."""
    fleet = Fleet(fleet_name="t", hosts=[_host()])
    job = _job()
    ssh = _FakeSSH(scripted={
        "cat ": SSHResult(exit_code=1, stdout="", stderr="No such file", duration_s=0.0),  # no manifest
        "pgrep ": SSHResult(exit_code=0, stdout="12345\n", stderr="", duration_s=0.0),  # alive
        "pkill ": SSHResult(exit_code=0, stdout="", stderr="", duration_s=0.0),
    })
    rsync = _FakeRsync()

    outcomes = reconcile_orphans(fleet=fleet, jobs=[job], dispatch_dir=tmp_path, ssh=ssh, rsync=rsync)

    assert outcomes == [ReconcileOutcome(run_id="r1", action="killed_alive_orphan")]
    assert job.status == "pending"
    assert job.assigned_to is None


def test_reconcile_dead_no_manifest_pending(tmp_path: Path):
    """Branch (d): no manifest, no process → mark pending."""
    fleet = Fleet(fleet_name="t", hosts=[_host()])
    job = _job()
    ssh = _FakeSSH(scripted={
        "cat ": SSHResult(exit_code=1, stdout="", stderr="No such file", duration_s=0.0),
        "pgrep ": SSHResult(exit_code=1, stdout="", stderr="", duration_s=0.0),  # nothing matched
    })
    rsync = _FakeRsync()

    outcomes = reconcile_orphans(fleet=fleet, jobs=[job], dispatch_dir=tmp_path, ssh=ssh, rsync=rsync)

    assert outcomes == [ReconcileOutcome(run_id="r1", action="dead_re_pending")]
    assert job.status == "pending"


def test_reconcile_skips_jobs_without_assigned_host(tmp_path: Path):
    """A 'running' job that somehow has assigned_to=None can't be reconciled
    against any host — leave it for the caller's reset_in_flight_to_pending."""
    fleet = Fleet(fleet_name="t", hosts=[_host()])
    job = _job(assigned_to=None)
    ssh = _FakeSSH(scripted={})
    rsync = _FakeRsync()

    outcomes = reconcile_orphans(fleet=fleet, jobs=[job], dispatch_dir=tmp_path, ssh=ssh, rsync=rsync)

    assert outcomes == []
    assert job.status == "running"  # untouched
