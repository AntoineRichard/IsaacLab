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
from tools.odin.asgard.worker import StateEvent, ValkyrieWorker, WorkerOptions, _build_docker_exec_cmd


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


def test_worker_cleans_up_remote_process_on_timeout(tmp_path: Path):
    """After an SSH timeout, the worker must dispatch a pkill to the container.

    Without this, the training process inside the container outlives its parent
    and hogs the GPU for whatever job the scheduler places next on the same host.
    """
    ssh = _FakeSSH(
        scripted={
            # Only the *initial* docker-exec (the one carrying the training argv)
            # times out. The cleanup pkill is a different command and falls
            # through to the default exit-0 path in _FakeSSH.
            "hugin/run.py": SSHResult(exit_code=-15, stdout="", stderr="", duration_s=60.1, timed_out=True),
        }
    )
    rsync = _FakeRsync(materialize_bundle=False)
    w = _make_worker(tmp_path, ssh, rsync)
    job = _job()
    _spin_worker(w, [job])
    pkill_calls = [cmd for _, cmd in ssh.log if "pkill" in cmd]
    assert len(pkill_calls) == 1
    # Matches on the job's run_id (surgical kill — no risk of hitting unrelated procs).
    assert job.run_id in pkill_calls[0]
    # Targeted at the configured container on this host.
    assert "docker exec" in pkill_calls[0]
    assert w.host.container_name in pkill_calls[0]


def test_worker_does_not_clean_up_on_normal_exit(tmp_path: Path):
    """No pkill dispatched on clean exit — cleanup is only for timeout zombies."""
    ssh = _FakeSSH()  # default: exit 0 for all commands
    rsync = _FakeRsync()
    w = _make_worker(tmp_path, ssh, rsync)
    _spin_worker(w, [_job()])
    assert not any("pkill" in cmd for _, cmd in ssh.log)


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


def test_build_docker_exec_cmd_includes_run_id():
    """worker must pass --run_id so Hugin/Munin write bundles at the dispatcher-expected path."""
    host = _host()
    job = _job()
    cmd = _build_docker_exec_cmd(host, job)
    assert f"--run_id {job.run_id}" in cmd
    # Sanity: other expected args still present.
    assert f"--task {job.task_id}" in cmd
    assert f"--backend {job.backend}" in cmd
    assert "--runs_root odin_runs" in cmd


def test_worker_classifies_preset_unsupported(tmp_path: Path):
    """Stderr containing 'preset_unsupported:' maps to its own kind, not hugin_crash."""
    ssh = _FakeSSH(
        scripted={
            "hugin/run.py": SSHResult(
                exit_code=2,
                stdout="",
                stderr=(
                    "[ERROR] preset_unsupported: task 'Isaac-Foo-v0' has no "
                    "'physx' preset. Inspect raw_cfg.sim.physics.\n"
                ),
                duration_s=3.0,
            )
        }
    )
    rsync = _FakeRsync(materialize_bundle=False)
    w = _make_worker(tmp_path, ssh, rsync)
    events = _spin_worker(w, [_job()])
    failed = next(e for e in events if e.transition == "failed")
    assert failed.failure is not None
    assert failed.failure.kind == "preset_unsupported"
    assert "missing preset" in failed.failure.message.lower()


def test_worker_falls_back_to_hugin_crash_without_marker(tmp_path: Path):
    """Regression: stderr without the marker still classifies as hugin_crash."""
    ssh = _FakeSSH(
        scripted={"hugin/run.py": SSHResult(exit_code=1, stdout="", stderr="generic crash\n", duration_s=2.0)}
    )
    rsync = _FakeRsync(materialize_bundle=False)
    w = _make_worker(tmp_path, ssh, rsync)
    events = _spin_worker(w, [_job()])
    failed = next(e for e in events if e.transition == "failed")
    assert failed.failure.kind == "hugin_crash"


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


def test_classify_gpu_lost_signature_nvml(tmp_path):
    """Stderr containing 'Failed to initialize NVML' → kind='gpu_lost'."""
    import queue
    import threading

    from tools.odin.asgard.fleet import ValkyrieConfig
    from tools.odin.asgard.jobs import JobEntry
    from tools.odin.asgard.transport import SSHResult
    from tools.odin.asgard.worker import ValkyrieWorker, WorkerOptions

    worker = ValkyrieWorker(
        host=ValkyrieConfig(host="v1", ssh_user="horde"),
        job_queue=queue.Queue(),
        state_chan=queue.Queue(),
        dispatch_dir=tmp_path,
        options=WorkerOptions(),
        ssh=None,
        rsync=None,
        shutdown_event=threading.Event(),
    )
    job = JobEntry(
        run_id="rsl-rl_physx_X_seed42",
        task_id="X",
        framework="rsl_rl",
        backend="physx",
        num_envs=4096,
        max_iterations=300,
        seed=42,
        bundle_dir_name="rsl-rl_physx_X_seed42",
    )
    ssh_tail = tmp_path / "rsl-rl_physx_X_seed42" / "logs" / "ssh-tail.log"
    ssh_tail.parent.mkdir(parents=True, exist_ok=True)
    ssh_tail.write_text("")
    r = SSHResult(
        exit_code=1,
        stdout="",
        stderr="Failed to initialize NVML: Unknown Error\n",
        duration_s=10.0,
    )
    failure = worker._classify(r, job, ssh_tail)
    assert failure is not None
    assert failure.kind == "gpu_lost"
    assert "GPU-loss signature" in failure.message


def test_classify_gpu_lost_signature_cuda(tmp_path):
    """Stderr containing 'CUDA error: no CUDA-capable device' → kind='gpu_lost'."""
    import queue
    import threading

    from tools.odin.asgard.fleet import ValkyrieConfig
    from tools.odin.asgard.jobs import JobEntry
    from tools.odin.asgard.transport import SSHResult
    from tools.odin.asgard.worker import ValkyrieWorker, WorkerOptions

    worker = ValkyrieWorker(
        host=ValkyrieConfig(host="v1", ssh_user="horde"),
        job_queue=queue.Queue(),
        state_chan=queue.Queue(),
        dispatch_dir=tmp_path,
        options=WorkerOptions(),
        ssh=None,
        rsync=None,
        shutdown_event=threading.Event(),
    )
    job = JobEntry(
        run_id="rsl-rl_physx_Y_seed42",
        task_id="Y",
        framework="rsl_rl",
        backend="physx",
        num_envs=4096,
        max_iterations=300,
        seed=42,
        bundle_dir_name="rsl-rl_physx_Y_seed42",
    )
    ssh_tail = tmp_path / "rsl-rl_physx_Y_seed42" / "logs" / "ssh-tail.log"
    ssh_tail.parent.mkdir(parents=True, exist_ok=True)
    ssh_tail.write_text("")
    r = SSHResult(
        exit_code=1,
        stdout="",
        stderr="RuntimeError: CUDA error: no CUDA-capable device is detected\n",
        duration_s=10.0,
    )
    failure = worker._classify(r, job, ssh_tail)
    assert failure is not None
    assert failure.kind == "gpu_lost"


def test_classify_gpu_lost_signature_vulkan(tmp_path):
    """Stderr containing 'Vulkan ERROR_INCOMPATIBLE_DRIVER' → kind='gpu_lost'."""
    import queue
    import threading

    from tools.odin.asgard.fleet import ValkyrieConfig
    from tools.odin.asgard.jobs import JobEntry
    from tools.odin.asgard.transport import SSHResult
    from tools.odin.asgard.worker import ValkyrieWorker, WorkerOptions

    worker = ValkyrieWorker(
        host=ValkyrieConfig(host="v1", ssh_user="horde"),
        job_queue=queue.Queue(),
        state_chan=queue.Queue(),
        dispatch_dir=tmp_path,
        options=WorkerOptions(),
        ssh=None,
        rsync=None,
        shutdown_event=threading.Event(),
    )
    job = JobEntry(
        run_id="rsl-rl_physx_Z_seed42",
        task_id="Z",
        framework="rsl_rl",
        backend="physx",
        num_envs=4096,
        max_iterations=300,
        seed=42,
        bundle_dir_name="rsl-rl_physx_Z_seed42",
    )
    ssh_tail = tmp_path / "rsl-rl_physx_Z_seed42" / "logs" / "ssh-tail.log"
    ssh_tail.parent.mkdir(parents=True, exist_ok=True)
    ssh_tail.write_text("")
    r = SSHResult(
        exit_code=1,
        stdout="",
        stderr="[error] Vulkan ERROR_INCOMPATIBLE_DRIVER: cannot create instance\n",
        duration_s=10.0,
    )
    failure = worker._classify(r, job, ssh_tail)
    assert failure is not None
    assert failure.kind == "gpu_lost"


def test_classify_no_false_positive_on_success(tmp_path):
    """Exit 0 + signature in stderr (warning) → _classify returns None."""
    import queue
    import threading

    from tools.odin.asgard.fleet import ValkyrieConfig
    from tools.odin.asgard.jobs import JobEntry
    from tools.odin.asgard.transport import SSHResult
    from tools.odin.asgard.worker import ValkyrieWorker, WorkerOptions

    worker = ValkyrieWorker(
        host=ValkyrieConfig(host="v1", ssh_user="horde"),
        job_queue=queue.Queue(),
        state_chan=queue.Queue(),
        dispatch_dir=tmp_path,
        options=WorkerOptions(),
        ssh=None,
        rsync=None,
        shutdown_event=threading.Event(),
    )
    job = JobEntry(
        run_id="rsl-rl_physx_OK_seed42",
        task_id="OK",
        framework="rsl_rl",
        backend="physx",
        num_envs=4096,
        max_iterations=300,
        seed=42,
        bundle_dir_name="rsl-rl_physx_OK_seed42",
    )
    ssh_tail = tmp_path / "rsl-rl_physx_OK_seed42" / "logs" / "ssh-tail.log"
    ssh_tail.parent.mkdir(parents=True, exist_ok=True)
    ssh_tail.write_text("")
    r = SSHResult(
        exit_code=0,
        stdout="...",
        stderr="warning: Failed to initialize NVML (recoverable)\n",
        duration_s=10.0,
    )
    failure = worker._classify(r, job, ssh_tail)
    assert failure is None


def test_classify_timeout_wins_over_gpu_signature(tmp_path):
    """timed_out=True + stderr has CUDA error → kind='timeout'."""
    import queue
    import threading

    from tools.odin.asgard.fleet import ValkyrieConfig
    from tools.odin.asgard.jobs import JobEntry
    from tools.odin.asgard.transport import SSHResult
    from tools.odin.asgard.worker import ValkyrieWorker, WorkerOptions

    worker = ValkyrieWorker(
        host=ValkyrieConfig(host="v1", ssh_user="horde"),
        job_queue=queue.Queue(),
        state_chan=queue.Queue(),
        dispatch_dir=tmp_path,
        options=WorkerOptions(per_job_timeout_s=14400),
        ssh=None,
        rsync=None,
        shutdown_event=threading.Event(),
    )
    job = JobEntry(
        run_id="rsl-rl_physx_TO_seed42",
        task_id="TO",
        framework="rsl_rl",
        backend="physx",
        num_envs=4096,
        max_iterations=300,
        seed=42,
        bundle_dir_name="rsl-rl_physx_TO_seed42",
    )
    ssh_tail = tmp_path / "rsl-rl_physx_TO_seed42" / "logs" / "ssh-tail.log"
    ssh_tail.parent.mkdir(parents=True, exist_ok=True)
    ssh_tail.write_text("")
    r = SSHResult(
        exit_code=-15,
        stdout="",
        stderr="CUDA error: no CUDA-capable device is detected\n",
        duration_s=14400.0,
    )
    r.timed_out = True
    failure = worker._classify(r, job, ssh_tail)
    assert failure is not None
    assert failure.kind == "timeout"


def test_worker_gpu_lost_recovery_succeeds_retries_same_host(tmp_path, monkeypatch):
    """First attempt: gpu_lost stderr → recover succeeds → second attempt succeeds."""
    import queue
    import threading

    from tools.odin.asgard import worker as worker_mod
    from tools.odin.asgard.fleet import ValkyrieConfig
    from tools.odin.asgard.jobs import JobEntry
    from tools.odin.asgard.recovery import RecoveryResult
    from tools.odin.asgard.transport import SSHResult

    # Scripted SSH: first call (Hugin) fails with NVML; second call (Hugin
    # retry) succeeds with exit 0 + valid bundle.
    ssh_responses = [
        SSHResult(
            exit_code=1,
            stdout="",
            stderr="Failed to initialize NVML: Unknown Error\n",
            duration_s=12.0,
        ),
        SSHResult(exit_code=0, stdout="ok", stderr="", duration_s=600.0),
    ]

    class _SeqSSH:
        def run(self, host, cmd, *, timeout_s=None, stdout_tee=None):
            return ssh_responses.pop(0)

    # Build a minimal valid bundle so _validate_bundle passes after retry.
    job = JobEntry(
        run_id="rsl-rl_physx_R_seed42",
        task_id="R",
        framework="rsl_rl",
        backend="physx",
        num_envs=4096,
        max_iterations=300,
        seed=42,
        bundle_dir_name="rsl-rl_physx_R_seed42",
    )
    bundle = tmp_path / job.bundle_dir_name
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text('{"schema_version": "1.0"}')

    class _FakeRsync:
        def pull(self, host, remote, local):
            return SSHResult(exit_code=0, stdout="", stderr="", duration_s=0.1)

    recover_calls = {"n": 0}

    def _fake_recover(host, *, ssh):
        recover_calls["n"] += 1
        return RecoveryResult(
            host=host.host,
            container_name=host.container_name,
            attempted=True,
            recovered=True,
            duration_s=12.0,
            message="recovered_via_container_restart",
            details={},
        )

    monkeypatch.setattr(worker_mod, "recover_valkyrie_gpu", _fake_recover)

    state_chan: queue.Queue = queue.Queue()
    worker = worker_mod.ValkyrieWorker(
        host=ValkyrieConfig(host="v1", ssh_user="horde"),
        job_queue=queue.Queue(),
        state_chan=state_chan,
        dispatch_dir=tmp_path,
        options=worker_mod.WorkerOptions(),
        ssh=_SeqSSH(),
        rsync=_FakeRsync(),
        shutdown_event=threading.Event(),
    )
    worker._execute(job)

    assert job.status == "completed"
    assert job.attempts == 2
    assert recover_calls["n"] == 1
    transitions = []
    while not state_chan.empty():
        transitions.append(state_chan.get_nowait().transition)
    assert "recovered" in transitions
    assert "completed" in transitions


def test_worker_gpu_lost_recovery_fails_marks_host_down(tmp_path, monkeypatch):
    """First attempt: gpu_lost → recover fails → host_down + re-queue + worker stops pulling.

    Worker no longer terminal-fails the job here — it re-queues it so a
    healthy worker can pick it up via bounded fallback. The runner sweeps
    any still-pending job at dispatch end if no host can run it.
    """
    import queue
    import threading

    from tools.odin.asgard import worker as worker_mod
    from tools.odin.asgard.fleet import ValkyrieConfig
    from tools.odin.asgard.jobs import JobEntry
    from tools.odin.asgard.recovery import RecoveryResult
    from tools.odin.asgard.transport import SSHResult

    class _SingleSSH:
        def run(self, host, cmd, *, timeout_s=None, stdout_tee=None):
            return SSHResult(
                exit_code=1,
                stdout="",
                stderr="CUDA error: no CUDA-capable device is detected\n",
                duration_s=10.0,
            )

    def _fake_recover(host, *, ssh):
        return RecoveryResult(
            host=host.host,
            container_name=host.container_name,
            attempted=True,
            recovered=False,
            duration_s=2.0,
            message="docker_restart_failed: daemon down",
            details={},
        )

    monkeypatch.setattr(worker_mod, "recover_valkyrie_gpu", _fake_recover)

    job_queue: queue.Queue = queue.Queue()
    state_chan: queue.Queue = queue.Queue()
    job = JobEntry(
        run_id="rsl-rl_physx_F_seed42",
        task_id="F",
        framework="rsl_rl",
        backend="physx",
        num_envs=4096,
        max_iterations=300,
        seed=42,
        bundle_dir_name="rsl-rl_physx_F_seed42",
    )
    (tmp_path / job.bundle_dir_name / "logs").mkdir(parents=True)
    (tmp_path / job.bundle_dir_name / "logs" / "ssh-tail.log").write_text("")

    worker = worker_mod.ValkyrieWorker(
        host=ValkyrieConfig(host="v1", ssh_user="horde"),
        job_queue=job_queue,
        state_chan=state_chan,
        dispatch_dir=tmp_path,
        options=worker_mod.WorkerOptions(),
        ssh=_SingleSSH(),
        rsync=None,
        shutdown_event=threading.Event(),
    )
    worker._execute(job)

    # Job is re-queued, NOT terminal-failed. The runner sweeps stuck pending
    # jobs at dispatch end when no healthy host remains.
    assert job.status == "pending"
    assert job.failure is None
    assert "v1" in job.preferred_not

    # Worker has stopped pulling further jobs.
    assert worker._down_event.is_set()

    # Job is back on the queue for another worker.
    requeued = job_queue.get_nowait()
    assert requeued is job

    # State channel got host_down but NOT a terminal "failed" event.
    transitions = []
    while not state_chan.empty():
        transitions.append(state_chan.get_nowait().transition)
    assert "host_down" in transitions
    assert "failed" not in transitions


def test_worker_gpu_lost_three_in_a_row_terminal_failure(tmp_path, monkeypatch):
    """Three consecutive gpu_lost + recovery=True → terminal fail at attempt 3."""
    import queue
    import threading

    from tools.odin.asgard import worker as worker_mod
    from tools.odin.asgard.fleet import ValkyrieConfig
    from tools.odin.asgard.jobs import JobEntry
    from tools.odin.asgard.recovery import RecoveryResult
    from tools.odin.asgard.transport import SSHResult

    nvml_fail = SSHResult(
        exit_code=1,
        stdout="",
        stderr="Failed to initialize NVML: Unknown Error\n",
        duration_s=10.0,
    )
    ssh_responses = [nvml_fail, nvml_fail, nvml_fail]

    class _SeqSSH:
        def run(self, host, cmd, *, timeout_s=None, stdout_tee=None):
            return ssh_responses.pop(0)

    def _fake_recover(host, *, ssh):
        return RecoveryResult(
            host=host.host,
            container_name=host.container_name,
            attempted=True,
            recovered=True,
            duration_s=10.0,
            message="recovered_via_container_restart",
            details={},
        )

    monkeypatch.setattr(worker_mod, "recover_valkyrie_gpu", _fake_recover)

    state_chan: queue.Queue = queue.Queue()
    job = JobEntry(
        run_id="rsl-rl_physx_T_seed42",
        task_id="T",
        framework="rsl_rl",
        backend="physx",
        num_envs=4096,
        max_iterations=300,
        seed=42,
        bundle_dir_name="rsl-rl_physx_T_seed42",
    )
    (tmp_path / job.bundle_dir_name / "logs").mkdir(parents=True)
    (tmp_path / job.bundle_dir_name / "logs" / "ssh-tail.log").write_text("")

    worker = worker_mod.ValkyrieWorker(
        host=ValkyrieConfig(host="v1", ssh_user="horde"),
        job_queue=queue.Queue(),
        state_chan=state_chan,
        dispatch_dir=tmp_path,
        # max_infrastructure_retries=2 → up to 3 attempts.
        options=worker_mod.WorkerOptions(max_infrastructure_retries=2),
        ssh=_SeqSSH(),
        rsync=None,
        shutdown_event=threading.Event(),
    )
    worker._execute(job)

    assert job.status == "failed"
    assert job.failure is not None
    assert job.failure.kind == "gpu_lost"
    assert job.attempts == 3


def test_build_docker_exec_cmd_uses_python_sh_directly():
    """Hugin invocation must bypass ./isaaclab.sh -p whose error_exit trap
    swallows child stderr; call _isaac_sim/python.sh directly instead."""
    host = _host()
    job = JobEntry(
        run_id="rsl-rl_physx_Isaac-Ant-Direct-v0_test_seed42",
        task_id="Isaac-Ant-Direct-v0",
        framework="rsl_rl",
        backend="physx",
        num_envs=1024,
        max_iterations=100,
        seed=42,
        bundle_dir_name="rsl-rl_physx_Isaac-Ant-Direct-v0_test_seed42",
    )
    cmd = _build_docker_exec_cmd(host, job)
    assert "_isaac_sim/python.sh" in cmd
    assert "./isaaclab.sh -p" not in cmd


def test_build_docker_exec_cmd_redirects_streams_into_bundle():
    """Child stdout / stderr must land in bundle-local log files so they
    rsync back regardless of exit code."""
    host = _host()
    job = JobEntry(
        run_id="rsl-rl_physx_Isaac-Ant-Direct-v0_test_seed42",
        task_id="Isaac-Ant-Direct-v0",
        framework="rsl_rl",
        backend="physx",
        num_envs=1024,
        max_iterations=100,
        seed=42,
        bundle_dir_name="rsl-rl_physx_Isaac-Ant-Direct-v0_test_seed42",
    )
    cmd = _build_docker_exec_cmd(host, job)
    assert "odin_runs/rsl-rl_physx_Isaac-Ant-Direct-v0_test_seed42/logs/hugin-stdout.log" in cmd
    assert "odin_runs/rsl-rl_physx_Isaac-Ant-Direct-v0_test_seed42/logs/hugin-stderr.log" in cmd
