# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :class:`tools.odin.asgard.worker.ValkyrieWorker` (happy path + classification)."""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path

from tools.odin.asgard.fleet import ValkyrieConfig
from tools.odin.asgard.jobs import JobEntry
from tools.odin.asgard.transport import RsyncResult, SSHResult
from tools.odin.asgard.worker import StateEvent, ValkyrieWorker, WorkerOptions


def _host() -> ValkyrieConfig:
    return ValkyrieConfig(host="v1", ssh_user="odin", isaaclab_path="~/IsaacLab")


def _job(run_id: str = "rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-220000_seed42") -> JobEntry:
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
class _FakeSSH:
    scripted: dict = field(default_factory=dict)
    log: list[tuple[str, str]] = field(default_factory=list)

    def run(self, host, cmd: str, *, timeout_s=None, stdout_tee=None) -> SSHResult:
        self.log.append((host.host, cmd))
        for key, result in self.scripted.items():
            if key in cmd:
                return result
        return SSHResult(exit_code=0, stdout="", stderr="", duration_s=0.01)


@dataclass
class _FakeRsync:
    materialize_bundle: bool = True
    log: list[tuple[str, str, str]] = field(default_factory=list)

    def push(self, host, local_path: Path, remote_path: str) -> RsyncResult:
        self.log.append(("push", str(local_path), remote_path))
        return RsyncResult(exit_code=0, stdout="", stderr="", duration_s=0.0)

    def pull(self, host, remote_path: str, local_path: Path) -> RsyncResult:
        self.log.append(("pull", remote_path, str(local_path)))
        if self.materialize_bundle:
            # Fake Hugin creating manifest + training + startup at local_path.
            local_path.mkdir(parents=True, exist_ok=True)
            (local_path / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "phases": {"startup": {"status": "completed"}, "training": {"status": "completed"}},
                    }
                )
            )
            (local_path / "training.json").write_text(json.dumps({"schema_version": "1.0"}))
            (local_path / "startup.json").write_text(json.dumps({"schema_version": "1.0"}))
        return RsyncResult(exit_code=0, stdout="", stderr="", duration_s=0.0)


def _spin_worker(worker: ValkyrieWorker, jobs: list[JobEntry]) -> list[StateEvent]:
    """Run the worker against the given job list synchronously (no threading) and collect events."""
    for j in jobs:
        worker._job_queue.put(j)
    # Sentinel so the worker exits after draining.
    worker._job_queue.put(None)
    worker.run()
    events: list[StateEvent] = []
    while True:
        try:
            events.append(worker._state_chan.get_nowait())
        except queue.Empty:
            return events


def _make_worker(tmp_path: Path, ssh, rsync, host=None) -> ValkyrieWorker:
    return ValkyrieWorker(
        host=host or _host(),
        job_queue=queue.Queue(),
        state_chan=queue.Queue(),
        dispatch_dir=tmp_path,
        options=WorkerOptions(per_job_timeout_s=60, max_infrastructure_retries=2),
        ssh=ssh,
        rsync=rsync,
        shutdown_event=threading.Event(),
    )


def test_worker_happy_path(tmp_path: Path):
    ssh = _FakeSSH()
    rsync = _FakeRsync()
    w = _make_worker(tmp_path, ssh, rsync)
    events = _spin_worker(w, [_job()])
    kinds = [e.transition for e in events]
    assert "running" in kinds
    assert "completed" in kinds
    # Bundle dir exists locally after pull.
    assert (tmp_path / _job().bundle_dir_name / "manifest.json").exists()


def test_worker_classifies_hugin_crash(tmp_path: Path):
    ssh = _FakeSSH(
        scripted={"docker exec": SSHResult(exit_code=1, stdout="", stderr="CUDA out of memory\n", duration_s=5.0)}
    )
    rsync = _FakeRsync(materialize_bundle=False)
    w = _make_worker(tmp_path, ssh, rsync)
    events = _spin_worker(w, [_job()])
    failed = next(e for e in events if e.transition == "failed")
    assert failed.failure is not None
    assert failed.failure.kind == "hugin_crash"
    assert failed.failure.details["exit_code"] == 1


def test_worker_classifies_timeout(tmp_path: Path):
    ssh = _FakeSSH(
        scripted={"docker exec": SSHResult(exit_code=-15, stdout="", stderr="", duration_s=60.1, timed_out=True)}
    )
    rsync = _FakeRsync(materialize_bundle=False)
    w = _make_worker(tmp_path, ssh, rsync)
    events = _spin_worker(w, [_job()])
    failed = next(e for e in events if e.transition == "failed")
    assert failed.failure.kind == "timeout"


def test_worker_classifies_malformed_bundle(tmp_path: Path):
    class _RsyncNoManifest(_FakeRsync):
        def pull(self, host, remote_path: str, local_path: Path) -> RsyncResult:
            local_path.mkdir(parents=True, exist_ok=True)
            # No manifest.json — should classify as malformed.
            return RsyncResult(exit_code=0, stdout="", stderr="", duration_s=0.0)

    ssh = _FakeSSH()
    rsync = _RsyncNoManifest()
    w = _make_worker(tmp_path, ssh, rsync)
    events = _spin_worker(w, [_job()])
    failed = next(e for e in events if e.transition == "failed")
    assert failed.failure.kind == "hugin_malformed_bundle"


def test_worker_classifies_infrastructure_before_hugin(tmp_path: Path):
    """ssh error on docker exec itself (exit -1 / no such container) is infrastructure, not hugin_crash."""
    ssh = _FakeSSH(
        scripted={
            # docker exec exits 125 when docker itself rejects the command (container not found).
            "docker exec": SSHResult(
                exit_code=125,
                stdout="",
                stderr="Error: No such container: isaac-lab-base\n",
                duration_s=0.5,
            )
        }
    )
    rsync = _FakeRsync(materialize_bundle=False)
    w = _make_worker(tmp_path, ssh, rsync)
    events = _spin_worker(w, [_job()])
    # With max_infrastructure_retries=2 the worker re-queues the job; after
    # exhausting retries (all still 125) it emits failed(infrastructure).
    failed = next(e for e in events if e.transition == "failed")
    assert failed.failure.kind == "infrastructure"


def test_worker_writes_ssh_tail_log(tmp_path: Path):
    class _SSHThatTees(_FakeSSH):
        def run(self, host, cmd, *, timeout_s=None, stdout_tee=None):
            # Emulate the real runner's tee behaviour.
            if stdout_tee is not None:
                stdout_tee.parent.mkdir(parents=True, exist_ok=True)
                stdout_tee.write_text("iter 1\niter 2\ndone\n")
            return SSHResult(exit_code=0, stdout="iter 1\niter 2\ndone\n", stderr="", duration_s=0.01)

    ssh = _SSHThatTees()
    rsync = _FakeRsync()
    w = _make_worker(tmp_path, ssh, rsync)
    _spin_worker(w, [_job()])
    tee = tmp_path / _job().bundle_dir_name / "logs" / "ssh-tail.log"
    assert tee.exists()
    assert "iter 1" in tee.read_text()


def test_worker_respects_shutdown_between_jobs(tmp_path: Path):
    """shutdown_event.set() stops the worker from pulling the next job."""
    ssh = _FakeSSH()
    rsync = _FakeRsync()
    host = _host()
    job_q = queue.Queue()
    state_q = queue.Queue()
    shutdown = threading.Event()
    worker = ValkyrieWorker(
        host=host,
        job_queue=job_q,
        state_chan=state_q,
        dispatch_dir=tmp_path,
        options=WorkerOptions(per_job_timeout_s=60),
        ssh=ssh,
        rsync=rsync,
        shutdown_event=shutdown,
    )
    shutdown.set()
    # Even with jobs queued, the worker should exit without consuming any.
    job_q.put(_job("r-skipped"))
    worker.run()
    events = []
    while not state_q.empty():
        events.append(state_q.get_nowait())
    # No running / completed / failed event for r-skipped.
    assert all(e.run_id != "r-skipped" for e in events)


def test_preferred_not_fallback_no_other_worker(tmp_path: Path):
    """When a job's preferred_not lists our host but NO other worker is around,
    the worker eventually accepts and runs it (we can't leave it stuck)."""
    ssh = _FakeSSH()
    rsync = _FakeRsync()
    w = _make_worker(tmp_path, ssh, rsync)
    j = _job("r-pref")
    j.preferred_not = {w.host.host}
    # The bounded fallback caps refusals at 3 before falling through.
    # Because the worker re-queues to the tail of a FIFO queue, the sentinel
    # must follow enough copies of the job so the worker gets 3 chances to
    # refuse before the sentinel arrives.  Three pre-loaded copies suffice:
    # the first two are refused (put back, cycling to tail), the third
    # triggers the fall-through.
    for _ in range(3):
        w._job_queue.put(j)
    w._job_queue.put(None)
    w.run()
    events = []
    while not w._state_chan.empty():
        events.append(w._state_chan.get_nowait())
    transitions = [e.transition for e in events]
    assert "running" in transitions
    assert "completed" in transitions
