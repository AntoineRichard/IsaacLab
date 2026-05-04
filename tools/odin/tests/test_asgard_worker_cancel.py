# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Worker-side kill / skip handling tests."""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path

from tools.odin.asgard.fleet import ValkyrieConfig
from tools.odin.asgard.jobs import JobEntry
from tools.odin.asgard.tracker import Tracker
from tools.odin.asgard.transport import RsyncResult, SSHResult
from tools.odin.asgard.worker import (
    JobInflight,
    POLL_EXITED_NO_MANIFEST,
    ValkyrieWorker,
    WorkerOptions,
)


def _host(host: str = "v1") -> ValkyrieConfig:
    return ValkyrieConfig(host=host, ssh_user="odin", isaaclab_path="~/IsaacLab")


def _job(run_id: str = "r-cancel") -> JobEntry:
    return JobEntry(
        run_id=run_id,
        task_id="Isaac-Ant-Direct-v0",
        framework="rsl_rl",
        backend="physx",
        num_envs=4096,
        max_iterations=300,
        seed=42,
        bundle_dir_name=run_id,
    )


@dataclass
class _ScriptedSSH:
    scripted: dict = field(default_factory=dict)
    log: list = field(default_factory=list)

    def run(self, host, cmd, *, timeout_s=None, stdout_tee=None, pty=True):
        self.log.append((host.host, cmd, pty))
        for key, result in self.scripted.items():
            if key in cmd:
                return result
        return SSHResult(exit_code=0, stdout="", stderr="", duration_s=0.0)


@dataclass
class _NoopRsync:
    log: list = field(default_factory=list)

    def pull(self, host, remote_path, local_path):
        self.log.append(("pull", remote_path, str(local_path)))
        local_path.mkdir(parents=True, exist_ok=True)
        (local_path / "logs").mkdir(parents=True, exist_ok=True)
        return RsyncResult(exit_code=0, stdout="", stderr="", duration_s=0.0)

    def push(self, host, local_path, remote_path):
        return RsyncResult(exit_code=0, stdout="", stderr="", duration_s=0.0)


def _make_worker(tmp_path: Path, ssh, rsync) -> ValkyrieWorker:
    return ValkyrieWorker(
        host=_host(),
        job_queue=queue.Queue(),
        state_chan=queue.Queue(),
        dispatch_dir=tmp_path,
        options=WorkerOptions(per_job_timeout_s=600, detached_mode=True, poll_interval_s=0),
        ssh=ssh,
        rsync=rsync,
        shutdown_event=threading.Event(),
    )


def test_request_cancel_records_run_id(tmp_path: Path):
    worker = _make_worker(tmp_path, _ScriptedSSH(), _NoopRsync())

    worker.request_cancel("r-cancel")

    assert worker._cancel_request.get("r-cancel") is True


def test_finalize_with_kill_dispatched_classifies_as_killed(tmp_path: Path):
    """When _sweep_cancellations has marked kill_dispatched, _finalize_terminal
    must override _classify_remote and stamp kind=killed."""
    rsync = _NoopRsync()
    worker = _make_worker(tmp_path, _ScriptedSSH(), rsync)
    job = _job("r-killed")
    inflight = JobInflight(
        job=job,
        tracker=Tracker(
            run_id=job.run_id,
            container_name=worker.host.container_name,
            host=worker.host.host,
            submitted_at="2026-05-04T10:00:00Z",
            pid=12345,
            per_job_timeout_s=600,
        ),
        submitted_at_monotonic=0.0,
        kill_dispatched=True,
    )
    worker._inflight[job.run_id] = inflight

    worker._finalize_terminal(inflight, POLL_EXITED_NO_MANIFEST)

    transitions = []
    while not worker._state_chan.empty():
        ev = worker._state_chan.get_nowait()
        transitions.append((ev.transition, ev.failure.kind if ev.failure else None))
    failed = next(t for t in transitions if t[0] == "failed")
    assert failed[1] == "killed"
    # Bundle still pulled (per Q4 answer): partial logs preserved.
    assert any(p[0] == "pull" for p in rsync.log)


def test_finalize_timeout_takes_precedence_over_kill(tmp_path: Path):
    """Both flags set → kind=timeout (job tripped budget before operator clicked)."""
    worker = _make_worker(tmp_path, _ScriptedSSH(), _NoopRsync())
    job = _job("r-timeout-kill")
    inflight = JobInflight(
        job=job,
        tracker=None,
        submitted_at_monotonic=0.0,
        timeout_kill_dispatched=True,
        kill_dispatched=True,
    )
    worker._inflight[job.run_id] = inflight

    worker._finalize_terminal(inflight, POLL_EXITED_NO_MANIFEST)

    transitions = []
    while not worker._state_chan.empty():
        ev = worker._state_chan.get_nowait()
        transitions.append((ev.transition, ev.failure.kind if ev.failure else None))
    failed = next(t for t in transitions if t[0] == "failed")
    assert failed[1] == "timeout"


def test_sweep_cancellations_dispatches_pkill(tmp_path: Path):
    ssh = _ScriptedSSH()
    worker = _make_worker(tmp_path, ssh, _NoopRsync())
    job = _job("r-killing")
    inflight = JobInflight(
        job=job,
        tracker=None,
        submitted_at_monotonic=0.0,
    )
    worker._inflight[job.run_id] = inflight
    worker.request_cancel(job.run_id)

    worker._sweep_cancellations()

    pkill = [cmd for _, cmd, _ in ssh.log if "pkill" in cmd]
    assert len(pkill) == 1
    assert job.run_id in pkill[0]
    assert inflight.kill_dispatched is True
    # _cancel_request consumed (drained after dispatch).
    assert "r-killing" not in worker._cancel_request


def test_sweep_cancellations_drops_unknown_run_id(tmp_path: Path):
    """Cancel for a job that already finished (no inflight entry) → silent drop."""
    ssh = _ScriptedSSH()
    worker = _make_worker(tmp_path, ssh, _NoopRsync())
    worker.request_cancel("ghost-run")

    worker._sweep_cancellations()

    assert "ghost-run" not in worker._cancel_request
    assert not any("pkill" in cmd for _, cmd, _ in ssh.log)


def test_sweep_cancellations_idempotent(tmp_path: Path):
    """Second sweep with the same kill_dispatched=True does not re-pkill."""
    ssh = _ScriptedSSH()
    worker = _make_worker(tmp_path, ssh, _NoopRsync())
    job = _job("r-twice")
    inflight = JobInflight(job=job, tracker=None, submitted_at_monotonic=0.0)
    worker._inflight[job.run_id] = inflight
    worker.request_cancel(job.run_id)
    worker._sweep_cancellations()
    assert sum(1 for _, c, _ in ssh.log if "pkill" in c) == 1

    # Second sweep: nothing new in _cancel_request, kill_dispatched already set.
    worker.request_cancel(job.run_id)  # operator clicked again somehow
    worker._sweep_cancellations()

    # Only the original pkill — kill_dispatched gate prevents the second.
    assert sum(1 for _, c, _ in ssh.log if "pkill" in c) == 1


def test_submit_or_handle_skips_when_status_already_failed(tmp_path: Path):
    """Skip race: runner flipped status before worker pulled the job. Worker
    must NOT submit (no SSH call)."""
    ssh = _ScriptedSSH()
    worker = _make_worker(tmp_path, ssh, _NoopRsync())
    job = _job("r-already-skipped")
    job.status = "failed"  # runner did the flip

    worker._submit_or_handle(job)

    # No SSH submit attempted.
    assert not any("docker exec -i" in cmd for _, cmd, _ in ssh.log)
    # No state event emitted (the runner already handled the transition).
    assert worker._state_chan.empty()
