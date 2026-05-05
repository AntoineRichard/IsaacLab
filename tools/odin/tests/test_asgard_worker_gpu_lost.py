# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for gpu_lost handling in :meth:`ValkyrieWorker._handle_synchronous_failure`.

Bug 3: when ``gpu_lost`` recovery succeeds, the worker re-queued the job
without flipping its status back to ``"pending"``.  The entry sat in
``dispatch.json`` as ``status="running"`` with a stale ``started_at``
while waiting in the queue — operators saw a phantom "running" row that
never resolved.

Concrete victim: Anymal-C-Nav seed43 in run 20260430-110509 sat as
``running started_at=11:34:16Z`` for hours after the worker quickly
recovered and put seed44 in the slot.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

from tools.odin.asgard import worker as worker_mod
from tools.odin.asgard.fleet import ValkyrieConfig
from tools.odin.asgard.jobs import FailureInfo, JobEntry
from tools.odin.asgard.recovery import RecoveryResult
from tools.odin.asgard.transport import RsyncResult, SSHResult
from tools.odin.asgard.worker import StateEvent, ValkyrieWorker, WorkerOptions

# ---------------------------------------------------------------------------
# Shared helpers (mirror test_asgard_worker.py style)
# ---------------------------------------------------------------------------


def _host(host: str = "v1") -> ValkyrieConfig:
    return ValkyrieConfig(host=host, ssh_user="odin", isaaclab_path="~/IsaacLab")


@dataclass
class _FakeSSH:
    log: list[tuple[str, str]] = field(default_factory=list)

    def run(self, host, cmd: str, *, timeout_s=None, stdout_tee=None, pty=True) -> SSHResult:
        self.log.append((host.host, cmd))
        return SSHResult(exit_code=0, stdout="", stderr="", duration_s=0.01)


@dataclass
class _NoopRsync:
    def push(self, host, local_path: Path, remote_path: str) -> RsyncResult:
        return RsyncResult(exit_code=0, stdout="", stderr="", duration_s=0.0)

    def pull(self, host, remote_path: str, local_path: Path) -> RsyncResult:
        return RsyncResult(exit_code=0, stdout="", stderr="", duration_s=0.0)


def _make_worker(tmp_path: Path, ssh=None, rsync=None, host=None) -> ValkyrieWorker:
    return ValkyrieWorker(
        host=host or _host(),
        job_queue=queue.Queue(),
        state_chan=queue.Queue(),
        dispatch_dir=tmp_path,
        options=WorkerOptions(per_job_timeout_s=60, max_infrastructure_retries=2),
        ssh=ssh or _FakeSSH(),
        rsync=rsync or _NoopRsync(),
        shutdown_event=threading.Event(),
    )


def _running_job(run_id: str = "r1") -> JobEntry:
    """Return a JobEntry already in 'running' state (as _handle_synchronous_failure expects)."""
    job = JobEntry(
        run_id=run_id,
        task_id="Isaac-Anymal-C-Nav-v0",
        framework="rsl_rl",
        backend="physx",
        num_envs=1,
        max_iterations=1,
        seed=43,
        bundle_dir_name=run_id,
        status="running",
        assigned_to="v1",
        started_at="2026-05-05T12:00:00Z",
        attempts=1,
    )
    return job


def _fake_recovery(recovered: bool, host: str = "v1") -> RecoveryResult:
    return RecoveryResult(
        host=host,
        container_name="isaac-lab-base",
        attempted=True,
        recovered=recovered,
        duration_s=1.0,
        message="ok" if recovered else "docker_restart_failed: daemon down",
    )


# ---------------------------------------------------------------------------
# Bug 3 regression: recovered branch must flip to pending
# ---------------------------------------------------------------------------


def test_handle_synchronous_failure_gpu_lost_recovery_flips_to_pending(tmp_path):
    """Bug 3 regression: when gpu_lost recovery succeeds in
    _handle_synchronous_failure, the worker re-queues the job. The
    JobEntry must come out with status='pending', no started_at,
    no assigned_to. attempts is preserved (worker bumps at submit,
    not here).

    Concrete scenario: Anymal-C-Nav seed43 in 20260430-110509 sat as
    'running started_at=11:34:16Z' indefinitely after the worker
    quickly recovered + put another job in the slot. Without the
    status flip the JobEntry was orphaned — neither running on any
    host nor visible as pending.
    """
    job = _running_job("r1")
    failure = FailureInfo(kind="gpu_lost", message="probe failed")
    worker = _make_worker(tmp_path)

    with patch.object(worker_mod, "recover_valkyrie_gpu", return_value=_fake_recovery(recovered=True)):
        worker._handle_synchronous_failure(job, failure)

    assert job.status == "pending"
    assert job.started_at is None
    assert job.assigned_to is None
    assert job.failure is None
    assert job.attempts == 1  # preserved across the recovery cycle


def test_handle_synchronous_failure_gpu_lost_recovery_job_requeued(tmp_path):
    """After successful recovery the job must land back on the job queue
    so a worker can pick it up."""
    job = _running_job("r2")
    failure = FailureInfo(kind="gpu_lost", message="probe failed")
    worker = _make_worker(tmp_path)

    with patch.object(worker_mod, "recover_valkyrie_gpu", return_value=_fake_recovery(recovered=True)):
        worker._handle_synchronous_failure(job, failure)

    requeued = worker._job_queue.get_nowait()
    assert requeued is job


def test_handle_synchronous_failure_gpu_lost_recovery_emits_recovered_event(tmp_path):
    """After successful recovery, a 'recovered' StateEvent must be emitted."""
    job = _running_job("r3")
    failure = FailureInfo(kind="gpu_lost", message="probe failed")
    worker = _make_worker(tmp_path)

    with patch.object(worker_mod, "recover_valkyrie_gpu", return_value=_fake_recovery(recovered=True)):
        worker._handle_synchronous_failure(job, failure)

    events: list[StateEvent] = []
    while not worker._state_chan.empty():
        events.append(worker._state_chan.get_nowait())

    transitions = [e.transition for e in events]
    assert "recovered" in transitions


# ---------------------------------------------------------------------------
# host_down branch: transition_to handles preferred_not
# ---------------------------------------------------------------------------


def test_handle_synchronous_failure_gpu_lost_host_down_flips_to_pending(tmp_path):
    """When gpu_lost recovery fails, the job must also be flipped to pending
    (not left as 'running') before being re-queued on a different host."""
    job = _running_job("r4")
    failure = FailureInfo(kind="gpu_lost", message="probe failed")
    worker = _make_worker(tmp_path)

    with patch.object(worker_mod, "recover_valkyrie_gpu", return_value=_fake_recovery(recovered=False)):
        worker._handle_synchronous_failure(job, failure)

    assert job.status == "pending"
    assert job.started_at is None
    assert job.assigned_to is None
    assert job.failure is None


def test_handle_synchronous_failure_gpu_lost_host_down_adds_preferred_not(tmp_path):
    """When recovery fails, the failed host must appear in job.preferred_not
    so a healthy worker on a different host picks it up."""
    job = _running_job("r5")
    failure = FailureInfo(kind="gpu_lost", message="probe failed")
    worker = _make_worker(tmp_path)

    with patch.object(worker_mod, "recover_valkyrie_gpu", return_value=_fake_recovery(recovered=False)):
        worker._handle_synchronous_failure(job, failure)

    assert "v1" in job.preferred_not


def test_handle_synchronous_failure_gpu_lost_host_down_sets_down_event(tmp_path):
    """After a failed recovery the worker's down_event must be set so it
    stops pulling further jobs."""
    job = _running_job("r6")
    failure = FailureInfo(kind="gpu_lost", message="probe failed")
    worker = _make_worker(tmp_path)

    with patch.object(worker_mod, "recover_valkyrie_gpu", return_value=_fake_recovery(recovered=False)):
        worker._handle_synchronous_failure(job, failure)

    assert worker._down_event.is_set()


def test_handle_synchronous_failure_gpu_lost_host_down_emits_host_down_event(tmp_path):
    """After a failed recovery a 'host_down' StateEvent must be emitted."""
    job = _running_job("r7")
    failure = FailureInfo(kind="gpu_lost", message="probe failed")
    worker = _make_worker(tmp_path)

    with patch.object(worker_mod, "recover_valkyrie_gpu", return_value=_fake_recovery(recovered=False)):
        worker._handle_synchronous_failure(job, failure)

    events: list[StateEvent] = []
    while not worker._state_chan.empty():
        events.append(worker._state_chan.get_nowait())

    transitions = [e.transition for e in events]
    assert "host_down" in transitions


# ---------------------------------------------------------------------------
# I1 regression: _execute legacy-PTY gpu_lost retry loop
# ---------------------------------------------------------------------------


@dataclass
class _SequencedSSH:
    """Fake SSH that replays a fixed list of SSHResults, one per call."""

    results: list[SSHResult]
    _call_count: int = field(default=0, init=False)

    def run(self, host, cmd: str, *, timeout_s=None, stdout_tee=None, pty=True) -> SSHResult:
        result = self.results[self._call_count]
        self._call_count += 1
        return result


def _gpu_lost_ssh_result() -> SSHResult:
    """Return an SSHResult whose stderr carries a GPU-loss signature."""
    return SSHResult(
        exit_code=1,
        stdout="",
        stderr="odin: gpu_unavailable",
        duration_s=0.5,
    )


def _success_ssh_result() -> SSHResult:
    return SSHResult(exit_code=0, stdout="", stderr="", duration_s=2.0)


def _pending_job(run_id: str = "r_exec1") -> JobEntry:
    """Return a JobEntry in 'pending' state, ready for _execute to submit."""
    return JobEntry(
        run_id=run_id,
        task_id="Isaac-Anymal-C-Nav-v0",
        framework="rsl_rl",
        backend="physx",
        num_envs=1,
        max_iterations=1,
        seed=1,
        bundle_dir_name=run_id,
        status="pending",
    )


def _write_valid_manifest(bundle_dir: Path) -> None:
    """Write a schema-v1 manifest.json that _validate_bundle accepts."""
    import json

    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.0",
        "phases": {
            "training": {"status": "completed", "exit_code": 0},
        },
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest))


def test_execute_legacy_pty_gpu_lost_recovery_retries_and_completes(tmp_path):
    """I1 regression: the legacy-PTY _execute retry loop's gpu_lost
    recovery path must produce a final completed job after one
    successful recovery + retry. The loop relies on transition_to('running')
    self-looping safely on the second pass — this test catches any
    refactor that breaks that contract.

    Distinct from the _handle_synchronous_failure tests in this file:
    those exercise the detached-mode submit-failure path. This one
    exercises the legacy-PTY in-flight retry loop inside _execute.
    """
    bundle_dir_name = "r_exec1"

    # Pre-create the ssh-tail.log directory so _classify can reference it via
    # relative_to(dispatch_dir), and write a valid manifest.json so that
    # _validate_bundle passes on the success pass (the _NoopRsync.pull does
    # not actually create any files on disk).
    logs_dir = tmp_path / bundle_dir_name / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "ssh-tail.log").touch()
    _write_valid_manifest(tmp_path / bundle_dir_name)

    # First SSH call: GPU-loss signature → _classify returns gpu_lost.
    # Second SSH call: success → _classify returns None → success path.
    ssh = _SequencedSSH(results=[_gpu_lost_ssh_result(), _success_ssh_result()])
    worker = _make_worker(tmp_path, ssh=ssh)
    job = _pending_job(bundle_dir_name)

    with patch.object(worker_mod, "recover_valkyrie_gpu", return_value=_fake_recovery(recovered=True)):
        worker._execute(job)

    assert job.status == "completed"
    assert job.attempts == 2
    assert job.started_at is not None
    assert job.ended_at is not None
