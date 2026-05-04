# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the detached-mode poll / finalize paths of :class:`ValkyrieWorker`."""

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
    POLL_ALIVE,
    POLL_DONE,
    POLL_EXITED_NO_MANIFEST,
    POLL_NO_PIDFILE,
    JobInflight,
    ValkyrieWorker,
    WorkerOptions,
    _build_poll_script,
    _parse_poll_output,
)


def _host(host: str = "v1") -> ValkyrieConfig:
    return ValkyrieConfig(host=host, ssh_user="odin", isaaclab_path="~/IsaacLab")


def _job(run_id: str = "r1") -> JobEntry:
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


def _tracker(host: ValkyrieConfig, run_id: str = "r1") -> Tracker:
    return Tracker(
        run_id=run_id,
        container_name=host.container_name,
        host=host.host,
        submitted_at="2026-04-30T11:05:34Z",
        pid=12345,
        per_job_timeout_s=43200,
    )


@dataclass
class _ScriptedSSH:
    """Scripted SSH fake matching by substring (first key-in-cmd wins)."""

    scripted: dict = field(default_factory=dict)
    log: list = field(default_factory=list)

    def run(self, host, cmd, *, timeout_s=None, stdout_tee=None, pty=True):
        self.log.append((host.host, cmd, pty))
        for key, result in self.scripted.items():
            if key in cmd:
                return result
        return SSHResult(exit_code=0, stdout="", stderr="", duration_s=0.0)


@dataclass
class _RsyncMaterialize:
    materialize: bool = True
    log: list = field(default_factory=list)

    def pull(self, host, remote_path, local_path):
        self.log.append(("pull", remote_path, str(local_path)))
        local_path.mkdir(parents=True, exist_ok=True)
        if self.materialize:
            (local_path / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "phases": {
                            "training": {"status": "completed", "exit_code": 0},
                            "startup": {"status": "completed", "exit_code": 0},
                        },
                    }
                )
            )
        return RsyncResult(exit_code=0, stdout="", stderr="", duration_s=0.0)

    def push(self, host, local_path, remote_path):
        return RsyncResult(exit_code=0, stdout="", stderr="", duration_s=0.0)


def _make_worker(tmp_path: Path, ssh, rsync) -> ValkyrieWorker:
    return ValkyrieWorker(
        host=_host(),
        job_queue=queue.Queue(),
        state_chan=queue.Queue(),
        dispatch_dir=tmp_path,
        options=WorkerOptions(per_job_timeout_s=60, detached_mode=True, poll_interval_s=0),
        ssh=ssh,
        rsync=rsync,
        shutdown_event=threading.Event(),
    )


# --- _build_poll_script ------------------------------------------------------


def test_build_poll_script_batches_multiple_bundles():
    """One SSH per host per tick: poll all in-flight bundles in a single script."""
    host = _host()
    script = _build_poll_script(host, ["b1", "b2", "b3"])
    # All three bundle ids appear (in the for-loop iterable).
    assert "b1" in script
    assert "b2" in script
    assert "b3" in script
    # The script tests for manifest.json (done) and .run.pid (alive vs exited).
    assert "manifest.json" in script
    assert ".run.pid" in script
    assert "kill -0" in script


def test_build_poll_script_no_pty_required():
    """The poll command runs through SSH with pty=False (the worker passes
    that argument; we just verify the command itself is short enough that
    an idle TTY is unnecessary)."""
    host = _host()
    script = _build_poll_script(host, ["b1"])
    # No -tt-needed shell features (interactive prompts, term resize). Just a
    # for loop that prints. No reason this couldn't run no-PTY.
    assert "docker exec" in script


# --- _parse_poll_output ------------------------------------------------------


def test_parse_poll_output_recognises_done_alive_exited_no_manifest():
    raw = "b-completed done\nb-running alive\nb-crashed exited-no-manifest\nb-just-submitted no-pidfile\n"
    states = _parse_poll_output(raw)
    assert states == {
        "b-completed": POLL_DONE,
        "b-running": POLL_ALIVE,
        "b-crashed": POLL_EXITED_NO_MANIFEST,
        "b-just-submitted": POLL_NO_PIDFILE,
    }


def test_parse_poll_output_ignores_garbage_lines():
    """SSH banner / login motd / blank lines must not produce phantom states."""
    raw = "Welcome to Ubuntu 22.04\n\nb1 done\nsome extra text\nb2 unexpected-state\n"
    states = _parse_poll_output(raw)
    assert states == {"b1": POLL_DONE}


# --- _finalize_terminal ------------------------------------------------------


def test_finalize_done_emits_completed_after_pull_and_validate(tmp_path: Path):
    ssh = _ScriptedSSH()
    rsync = _RsyncMaterialize(materialize=True)
    worker = _make_worker(tmp_path, ssh, rsync)
    job = _job("r-done")
    inflight = JobInflight(
        job=job,
        tracker=_tracker(worker.host, "r-done"),
        submitted_at_monotonic=0.0,
    )
    worker._inflight[job.run_id] = inflight
    worker._finalize_terminal(inflight, POLL_DONE)
    # Bundle pulled.
    assert any(p[0] == "pull" for p in rsync.log)
    # 'completed' event emitted.
    transitions = []
    while not worker._state_chan.empty():
        transitions.append(worker._state_chan.get_nowait().transition)
    assert "completed" in transitions
    # No longer tracked.
    assert job.run_id not in worker._inflight


def test_finalize_exited_no_manifest_classifies_via_remote_stderr(tmp_path: Path):
    """Crashed mid-run: pull bundle (best effort), classify via remote stderr."""

    class _RsyncBest:
        def pull(self, host, remote_path, local_path):
            local_path.mkdir(parents=True, exist_ok=True)
            (local_path / "logs").mkdir(parents=True, exist_ok=True)
            (local_path / "logs" / "hugin-stderr.log").write_text(
                "Traceback (most recent call last):\n  RuntimeError: CUDA error: no CUDA-capable device is detected\n"
            )
            return RsyncResult(exit_code=0, stdout="", stderr="", duration_s=0.0)

        def push(self, host, local_path, remote_path):
            return RsyncResult(exit_code=0, stdout="", stderr="", duration_s=0.0)

    ssh = _ScriptedSSH()
    rsync = _RsyncBest()
    worker = _make_worker(tmp_path, ssh, rsync)
    job = _job("r-crash")
    job.bundle_dir_name = "r-crash"
    inflight = JobInflight(
        job=job,
        tracker=_tracker(worker.host, "r-crash"),
        submitted_at_monotonic=0.0,
    )
    worker._inflight[job.run_id] = inflight
    worker._finalize_terminal(inflight, POLL_EXITED_NO_MANIFEST)
    failed = [worker._state_chan.get_nowait() for _ in range(worker._state_chan.qsize())]
    fail_evt = next(e for e in failed if e.transition == "failed")
    assert fail_evt.failure is not None
    assert fail_evt.failure.kind == "gpu_lost"


def test_classify_remote_recognises_gpu_lost_signatures(tmp_path: Path):
    """Each known signature → kind='gpu_lost'."""
    cases = [
        "Failed to initialize NVML: Unknown Error",
        "RuntimeError: CUDA error: no CUDA-capable device is detected",
        "[error] Vulkan ERROR_INCOMPATIBLE_DRIVER: cannot create instance",
        "odin: gpu_unavailable: Failed to initialize NVML",
    ]
    ssh = _ScriptedSSH()
    rsync = _RsyncMaterialize()
    worker = _make_worker(tmp_path, ssh, rsync)
    for signature in cases:
        result = worker._classify_remote_text(signature)
        assert result.kind == "gpu_lost", f"signature {signature!r} did not classify as gpu_lost"


def test_classify_remote_falls_back_to_hugin_crash_with_no_signature(tmp_path: Path):
    ssh = _ScriptedSSH()
    rsync = _RsyncMaterialize()
    worker = _make_worker(tmp_path, ssh, rsync)
    result = worker._classify_remote_text("ValueError: unrelated training bug\n")
    assert result.kind == "hugin_crash"


def test_classify_remote_recognises_preset_unsupported(tmp_path: Path):
    ssh = _ScriptedSSH()
    rsync = _RsyncMaterialize()
    worker = _make_worker(tmp_path, ssh, rsync)
    result = worker._classify_remote_text("[ERROR] preset_unsupported: task 'Isaac-Foo-v0' has no 'physx' preset.\n")
    assert result.kind == "preset_unsupported"


# --- _sweep_timeouts ---------------------------------------------------------


def test_sweep_timeouts_kills_remote_and_marks_inflight_for_classify(tmp_path: Path):
    """Inflight job past its budget gets a best-effort pkill and a flag so the
    next poll's exited-no-manifest reclassifies as kind=timeout."""
    ssh = _ScriptedSSH()
    rsync = _RsyncMaterialize()
    worker = _make_worker(tmp_path, ssh, rsync)
    job = _job("r-slow")
    inflight = JobInflight(
        job=job,
        tracker=_tracker(worker.host, "r-slow"),
        submitted_at_monotonic=-99999.0,  # way past budget
    )
    worker._inflight[job.run_id] = inflight
    worker._sweep_timeouts()
    # pkill dispatched.
    pkill_calls = [cmd for _, cmd, _ in ssh.log if "pkill" in cmd]
    assert len(pkill_calls) == 1
    assert job.run_id in pkill_calls[0]
    # Mark recorded for classification on next poll's exited-no-manifest tick.
    assert inflight.timeout_kill_dispatched is True


def test_sweep_timeouts_classifies_timeout_after_kill(tmp_path: Path):
    """After ``_sweep_timeouts`` set the flag, an exited-no-manifest poll
    must classify the failure as kind=timeout (not gpu_lost / hugin_crash)."""
    ssh = _ScriptedSSH()
    rsync = _RsyncMaterialize()
    worker = _make_worker(tmp_path, ssh, rsync)
    job = _job("r-timeout")
    job.bundle_dir_name = "r-timeout"
    inflight = JobInflight(
        job=job,
        tracker=_tracker(worker.host, "r-timeout"),
        submitted_at_monotonic=0.0,
        timeout_kill_dispatched=True,
    )
    worker._inflight[job.run_id] = inflight
    worker._finalize_terminal(inflight, POLL_EXITED_NO_MANIFEST)
    transitions = []
    while not worker._state_chan.empty():
        ev = worker._state_chan.get_nowait()
        transitions.append((ev.transition, ev.failure.kind if ev.failure else None))
    failed = next(t for t in transitions if t[0] == "failed")
    assert failed[1] == "timeout"
