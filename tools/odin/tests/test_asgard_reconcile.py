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

    def run(self, host, cmd, *, timeout_s=None, stdout_tee=None, pty=True) -> SSHResult:
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
    return json.dumps(
        {
            "run_id": "r1",
            "phases": {
                "startup": {"status": "completed", "exit_code": 0},
                "training": {"status": "completed", "exit_code": 0},
            },
        }
    )


def _manifest_failed() -> str:
    return json.dumps(
        {
            "run_id": "r1",
            "phases": {
                "startup": {"status": "completed", "exit_code": 0},
                "training": {"status": "failed", "exit_code": 1},
            },
        }
    )


def test_reconcile_completed_manifest_adopts_bundle(tmp_path: Path):
    """Branch (a): manifest exists + both phases completed → rsync, mark completed."""
    fleet = Fleet(fleet_name="t", hosts=[_host()])
    job = _job()
    ssh = _FakeSSH(
        scripted={
            "cat ": SSHResult(exit_code=0, stdout=_manifest_completed(), stderr="", duration_s=0.0),
        }
    )
    rsync = _FakeRsync()

    outcomes = reconcile_orphans(fleet=fleet, jobs=[job], dispatch_dir=tmp_path, ssh=ssh, rsync=rsync)

    assert outcomes == [ReconcileOutcome(run_id="r1", action="adopted_completed")]
    assert job.status == "completed"
    assert len(rsync.pulls) == 1


def test_reconcile_failed_manifest_adopts_failure(tmp_path: Path):
    """Branch (b): manifest exists but a phase failed → rsync, mark failed."""
    fleet = Fleet(fleet_name="t", hosts=[_host()])
    job = _job()
    ssh = _FakeSSH(
        scripted={
            "cat ": SSHResult(exit_code=0, stdout=_manifest_failed(), stderr="", duration_s=0.0),
        }
    )
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
    ssh = _FakeSSH(
        scripted={
            "cat ": SSHResult(exit_code=1, stdout="", stderr="No such file", duration_s=0.0),  # no manifest
            "pgrep ": SSHResult(exit_code=0, stdout="12345\n", stderr="", duration_s=0.0),  # alive
            "pkill ": SSHResult(exit_code=0, stdout="", stderr="", duration_s=0.0),
        }
    )
    rsync = _FakeRsync()

    outcomes = reconcile_orphans(fleet=fleet, jobs=[job], dispatch_dir=tmp_path, ssh=ssh, rsync=rsync)

    assert outcomes == [ReconcileOutcome(run_id="r1", action="killed_alive_orphan")]
    assert job.status == "pending"
    assert job.assigned_to is None
    assert job.started_at is None


def test_reconcile_dead_no_manifest_pending(tmp_path: Path):
    """Branch (d): no manifest, no process → mark pending."""
    fleet = Fleet(fleet_name="t", hosts=[_host()])
    job = _job()
    ssh = _FakeSSH(
        scripted={
            "cat ": SSHResult(exit_code=1, stdout="", stderr="No such file", duration_s=0.0),
            "pgrep ": SSHResult(exit_code=1, stdout="", stderr="", duration_s=0.0),  # nothing matched
        }
    )
    rsync = _FakeRsync()

    outcomes = reconcile_orphans(fleet=fleet, jobs=[job], dispatch_dir=tmp_path, ssh=ssh, rsync=rsync)

    assert outcomes == [ReconcileOutcome(run_id="r1", action="dead_re_pending")]
    assert job.status == "pending"
    assert job.assigned_to is None
    assert job.started_at is None


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


# --- Detached-mode reattach via tracker.json --------------------------------


def _write_tracker(local_bundle: Path, run_id: str, host: str, container: str = "isaac-lab-base") -> None:
    """Helper: write a local-side ``.tracker.json`` so reconcile can find it.

    The dispatcher actually reads the tracker over SSH from the remote
    bundle, but for unit-testing reconcile we just stage a pre-pulled
    bundle dir locally and stub the SSH calls separately.
    """
    from tools.odin.asgard.tracker import Tracker, write_tracker

    local_bundle.mkdir(parents=True, exist_ok=True)
    write_tracker(
        local_bundle,
        Tracker(
            run_id=run_id,
            container_name=container,
            host=host,
            submitted_at="2026-04-30T11:05:34Z",
            pid=12345,
            per_job_timeout_s=43200,
        ),
    )


def test_reconcile_reattaches_inflight_with_tracker_alive(tmp_path: Path):
    """Tracker found + remote process still alive → action=reattached_inflight.

    Reconcile leaves the job in ``running`` for the worker to keep polling;
    the caller seeds the worker's ``_inflight`` map from the outcome list.
    """
    fleet = Fleet(fleet_name="t", hosts=[_host()])
    job = _job()
    _write_tracker(tmp_path / job.bundle_dir_name, job.run_id, host="v1")
    ssh = _FakeSSH(
        scripted={
            # Detached-mode poll script — recognise via 'kill -0' (the manifest
            # cat would also contain ' cat ' so we order this key first).
            "kill -0": SSHResult(exit_code=0, stdout=f"{job.bundle_dir_name} alive\n", stderr="", duration_s=0.0),
            "manifest.json": SSHResult(exit_code=1, stdout="", stderr="No such file", duration_s=0.0),
        }
    )
    rsync = _FakeRsync()

    outcomes = reconcile_orphans(
        fleet=fleet, jobs=[job], dispatch_dir=tmp_path, ssh=ssh, rsync=rsync, detached_mode=True
    )

    assert any(o.action == "reattached_inflight" for o in outcomes)
    assert job.status == "running"


def test_reconcile_finalizes_inflight_with_tracker_done(tmp_path: Path):
    """Tracker found + manifest.json present on remote → adopted_completed."""
    fleet = Fleet(fleet_name="t", hosts=[_host()])
    job = _job()
    _write_tracker(tmp_path / job.bundle_dir_name, job.run_id, host="v1")
    ssh = _FakeSSH(
        scripted={
            "cat ": SSHResult(exit_code=0, stdout=_manifest_completed(), stderr="", duration_s=0.0),
        }
    )
    rsync = _FakeRsync()

    outcomes = reconcile_orphans(
        fleet=fleet, jobs=[job], dispatch_dir=tmp_path, ssh=ssh, rsync=rsync, detached_mode=True
    )

    assert outcomes == [ReconcileOutcome(run_id="r1", action="adopted_completed")]
    assert job.status == "completed"


def test_reconcile_finalizes_inflight_with_tracker_exited_no_manifest(tmp_path: Path):
    """Tracker present, no manifest, process gone → mark failed via remote stderr."""
    fleet = Fleet(fleet_name="t", hosts=[_host()])
    job = _job()
    _write_tracker(tmp_path / job.bundle_dir_name, job.run_id, host="v1")
    ssh = _FakeSSH(
        scripted={
            "kill -0": SSHResult(
                exit_code=0, stdout=f"{job.bundle_dir_name} exited-no-manifest\n", stderr="", duration_s=0.0
            ),
            "manifest.json": SSHResult(exit_code=1, stdout="", stderr="No such file", duration_s=0.0),
        }
    )
    rsync = _FakeRsync()

    outcomes = reconcile_orphans(
        fleet=fleet, jobs=[job], dispatch_dir=tmp_path, ssh=ssh, rsync=rsync, detached_mode=True
    )

    assert any(o.action == "adopted_failed" for o in outcomes)
    assert job.status == "failed"
    assert job.failure is not None


def test_reconcile_applies_pending_skip_for_pending_job(tmp_path: Path):
    """Resume: a 'skip' cancellation arrived while the dispatcher was down.
    The pending job is flipped to failed/skipped and the row marked consumed."""
    from tools.odin.asgard.reconcile import reconcile_orphans
    from tools.odin.valhalla.dashboard.cancel_db import CancelDB

    fleet = Fleet(fleet_name="t", hosts=[_host()])
    job = _job("r-skip-on-resume")
    job.status = "pending"
    job.assigned_to = None
    cancel_db = CancelDB(tmp_path)
    cancel_db.request(tmp_path.name, "r-skip-on-resume", kind="skip")
    ssh = _FakeSSH(scripted={})
    rsync = _FakeRsync()

    outcomes = reconcile_orphans(
        fleet=fleet,
        jobs=[job],
        dispatch_dir=tmp_path,
        ssh=ssh,
        rsync=rsync,
        detached_mode=True,
        cancel_db=cancel_db,
    )

    assert job.status == "failed"
    assert job.failure is not None
    assert job.failure.kind == "skipped"
    assert any(o.run_id == "r-skip-on-resume" and o.action == "adopted_failed" for o in outcomes)
    assert cancel_db.read_pending(tmp_path.name) == {}


def test_reconcile_leaves_skip_for_running_job_unconsumed(tmp_path: Path):
    """Skip on a job still 'running' at resume time → reconcile leaves the
    cancellation row pending. The runner's _consume_cancellations will
    upgrade it to kill on the first main-loop tick.
    """
    from tools.odin.valhalla.dashboard.cancel_db import CancelDB

    fleet = Fleet(fleet_name="t", hosts=[_host()])
    job = _job("r-running-skip")
    job.status = "running"
    job.assigned_to = "v1"
    cancel_db = CancelDB(tmp_path)
    cancel_db.request(tmp_path.name, "r-running-skip", kind="skip")
    # SSH responses for the detached running-job reconcile path: no manifest,
    # alive.
    ssh = _FakeSSH(
        scripted={
            "kill -0": SSHResult(exit_code=0, stdout="r-running-skip alive\n", stderr="", duration_s=0.0),
            "manifest.json": SSHResult(exit_code=1, stdout="", stderr="No such file", duration_s=0.0),
        }
    )
    rsync = _FakeRsync()

    reconcile_orphans(
        fleet=fleet,
        jobs=[job],
        dispatch_dir=tmp_path,
        ssh=ssh,
        rsync=rsync,
        detached_mode=True,
        cancel_db=cancel_db,
    )

    # Job state untouched — re-attached as running.
    assert job.status == "running"
    # Skip row still pending — the runner will pick it up on the first tick.
    assert cancel_db.read_pending(tmp_path.name) == {"r-running-skip": "skip"}
