# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the detached-mode submit phase of :class:`ValkyrieWorker`."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path

from tools.odin.asgard.fleet import ValkyrieConfig
from tools.odin.asgard.jobs import JobEntry
from tools.odin.asgard.transport import RsyncResult, SSHResult
from tools.odin.asgard.worker import (
    ValkyrieWorker,
    WorkerOptions,
    _build_submit_script,
)


def _host(host: str = "v1") -> ValkyrieConfig:
    return ValkyrieConfig(host=host, ssh_user="odin", isaaclab_path="~/IsaacLab")


def _job(run_id: str = "rsl-rl_physx_X_seed42") -> JobEntry:
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
    """Sequential-response SSH fake; pops one ``SSHResult`` per call."""

    responses: list[SSHResult] = field(default_factory=list)
    log: list[tuple[str, str, bool]] = field(default_factory=list)

    def run(self, host, cmd, *, timeout_s=None, stdout_tee=None, pty=True):
        self.log.append((host.host, cmd, pty))
        if not self.responses:
            return SSHResult(exit_code=0, stdout="", stderr="", duration_s=0.0)
        return self.responses.pop(0)


@dataclass
class _NoopRsync:
    pulls: list = field(default_factory=list)

    def pull(self, host, remote_path, local_path):
        self.pulls.append((host.host, remote_path, str(local_path)))
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


# --- _build_submit_script ----------------------------------------------------


def test_build_submit_script_includes_setsid_and_pidfile_write():
    """Detached training requires the trainer to outlive the SSH session.

    setsid + nohup + the inner shell's ``echo $$ > .run.pid`` are the three
    ingredients: setsid escapes the SSH-side controlling terminal, nohup
    blocks SIGHUP propagation, and the pidfile is what poll uses with
    ``kill -0`` to detect liveness.
    """
    script = _build_submit_script(
        _host(),
        _job(),
        submitted_at="2026-04-30T11:05:34Z",
        per_job_timeout_s=43200,
    )
    assert "setsid" in script
    assert "nohup" in script
    assert ".run.pid" in script
    # The trainer is launched detached (`&`).
    assert "&" in script


def test_build_submit_script_wipes_bundle_before_launching_trainer():
    """A previous attempt's manifest.json (or any other terminal artifact) must
    not survive into a fresh submit. Without a wipe, the new worker's first
    poll picks up the old manifest, _validate_bundle accepts it as terminal
    (or, post-fix, rejects it as hugin_crash even though the new trainer is
    still alive), and the freshly-submitted run is reported as failed before
    it gets a chance to write its own outcome.

    The wipe must run BEFORE the trainer is backgrounded — otherwise we race
    with the new trainer starting to write its logs."""
    script = _build_submit_script(
        _host(),
        _job("r1"),
        submitted_at="2026-04-30T11:05:34Z",
        per_job_timeout_s=43200,
    )
    # rm -rf the bundle (or at minimum manifest.json + tracker + pidfile).
    assert "rm -rf" in script
    assert "odin_runs/r1" in script
    rm_idx = script.index("rm -rf")
    train_idx = script.index("hugin/run.py")
    assert rm_idx < train_idx, "wipe must precede trainer launch"
    # mkdir -p of the logs dir must come AFTER the wipe — otherwise rm -rf
    # nukes the directory we just created.
    mkdir_idx = script.index("mkdir -p")
    assert rm_idx < mkdir_idx, "mkdir must come after rm so the dir exists"


def test_build_submit_script_runs_nvidia_probe_before_train():
    script = _build_submit_script(
        _host(),
        _job(),
        submitted_at="2026-04-30T11:05:34Z",
        per_job_timeout_s=43200,
    )
    assert "nvidia-smi -L" in script
    probe_idx = script.index("nvidia-smi -L")
    train_idx = script.index("hugin/run.py")
    assert probe_idx < train_idx, "probe must run before training"
    # Probe-failure marker matches the GPU-lost signature set so
    # _classify_remote can match it after a synchronous submit failure.
    assert "odin: gpu_unavailable" in script
    # Probe failure routes to odin-submit-error.log (read by _classify_remote).
    assert "odin-submit-error.log" in script


def test_build_submit_script_writes_tracker_json():
    """Tracker carries everything reconcile.py needs to re-attach an orphan."""
    host = _host()
    job = _job()
    script = _build_submit_script(
        host, job, submitted_at="2026-04-30T11:05:34Z", per_job_timeout_s=43200
    )
    assert ".tracker.json" in script
    # Dispatcher-known fields are stamped in the heredoc body.
    assert f'"run_id": "{job.run_id}"' in script
    assert f'"host": "{host.host}"' in script
    assert f'"container_name": "{host.container_name}"' in script
    assert '"submitted_at": "2026-04-30T11:05:34Z"' in script
    assert '"per_job_timeout_s": 43200' in script
    # PID is substituted by the inside bash via $TRAINING_PID, not by Python.
    assert "$TRAINING_PID" in script


def test_build_submit_script_emits_ok_sentinel():
    script = _build_submit_script(
        _host(),
        _job("r1"),
        submitted_at="2026-04-30T11:05:34Z",
        per_job_timeout_s=43200,
    )
    assert "odin-submit: ok run_id=r1 bundle=r1" in script


def test_build_submit_script_uses_quoted_heredoc_for_outer():
    """Quoting matters here: the OUTER heredoc must be quoted (``<<'TAG'``)
    so the dispatcher's shell on the remote does NOT pre-expand
    ``$TRAINING_PID`` / ``$$`` / ``$!`` before docker exec sees them."""
    script = _build_submit_script(
        _host(),
        _job(),
        submitted_at="2026-04-30T11:05:34Z",
        per_job_timeout_s=43200,
    )
    assert "<<'ASGARD_SUBMIT_EOF'" in script
    assert "ASGARD_SUBMIT_EOF" in script


# --- _submit_job runtime path ------------------------------------------------


def test_submit_parses_ok_sentinel(tmp_path: Path):
    """A submit returning the sentinel on stdout populates inflight."""
    ssh = _ScriptedSSH(
        responses=[SSHResult(exit_code=0, stdout="odin-submit: ok run_id=X bundle=B\n", stderr="", duration_s=0.5)]
    )
    rsync = _NoopRsync()
    worker = _make_worker(tmp_path, ssh, rsync)
    job = _job("r-ok")
    result = worker._submit_job(job)
    assert result.ok
    assert result.failure is None
    # Submit SSH must use pty=False (detached: no SIGHUP path).
    assert ssh.log[0][2] is False, "submit must call SSH with pty=False"


def test_submit_returns_gpu_lost_when_probe_fails(tmp_path: Path):
    """Synchronous probe failure: ssh exits 1 with the marker on stderr.

    ``_classify_remote`` then reads the per-bundle ``odin-submit-error.log``
    via cat and sees the marker — but for the synchronous submit path, the
    SSH stderr captures it directly so we can short-circuit.
    """
    bad_stderr = "odin: gpu_unavailable: Failed to initialize NVML: Unknown Error\n"

    ssh = _ScriptedSSH(
        responses=[
            # Synchronous submit failure (exit 1, marker on stderr).
            SSHResult(exit_code=1, stdout="", stderr=bad_stderr, duration_s=0.4),
        ]
    )
    rsync = _NoopRsync()
    worker = _make_worker(tmp_path, ssh, rsync)
    result = worker._submit_job(_job())
    assert not result.ok
    assert result.failure is not None
    assert result.failure.kind == "gpu_lost"


def test_submit_returns_infrastructure_on_docker_exit_125(tmp_path: Path):
    """``docker exec`` exits 125 when the daemon is unreachable / container is
    gone. That's an infrastructure failure on this host; the run loop should
    retry up to ``max_infrastructure_retries`` per the existing policy."""
    ssh = _ScriptedSSH(
        responses=[
            SSHResult(exit_code=125, stdout="", stderr="Error: No such container", duration_s=0.4),
        ]
    )
    rsync = _NoopRsync()
    worker = _make_worker(tmp_path, ssh, rsync)
    result = worker._submit_job(_job())
    assert not result.ok
    assert result.failure is not None
    assert result.failure.kind == "infrastructure"


def test_submit_retries_on_transient_ssh_error_then_succeeds(tmp_path: Path):
    """A one-second SSH glitch at submit shouldn't kill a job before it
    starts. Retry up to N times with backoff before giving up."""
    ssh = _ScriptedSSH(
        responses=[
            SSHResult(exit_code=255, stdout="", stderr="ssh: connect to host: Connection refused", duration_s=0.4),
            SSHResult(exit_code=0, stdout="odin-submit: ok run_id=r bundle=B\n", stderr="", duration_s=0.5),
        ]
    )
    rsync = _NoopRsync()
    worker = _make_worker(tmp_path, ssh, rsync)
    result = worker._submit_job(_job())
    assert result.ok
    assert len(ssh.log) == 2


def test_submit_returns_infrastructure_when_all_retries_fail(tmp_path: Path):
    """Three consecutive SSH-255 failures → terminal infrastructure failure."""
    fail = SSHResult(exit_code=255, stdout="", stderr="connection refused", duration_s=0.4)
    ssh = _ScriptedSSH(responses=[fail, fail, fail])
    rsync = _NoopRsync()
    worker = _make_worker(tmp_path, ssh, rsync)
    result = worker._submit_job(_job())
    assert not result.ok
    assert result.failure is not None
    assert result.failure.kind == "infrastructure"
