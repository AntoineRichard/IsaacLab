# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :func:`tools.odin.asgard.runner.run_dispatch`.

End-to-end tests with fake SSH + rsync; no threading primitives exercised
at the integration level (that's the slow loopback test). These tests
verify dispatch orchestration, dispatch.json rewrite, and resume.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
from tools.odin.asgard.runner import DispatchOptions, resolve_dispatch_dir, run_dispatch
from tools.odin.asgard.state import read_dispatch_state
from tools.odin.asgard.transport import RsyncResult, ShellRsyncRunner, ShellSSHRunner, SSHResult
from tools.odin.common.env_list import EnvEntry, EnvList, write_env_list
from tools.odin.valhalla.dashboard.retry_db import RetryDB


@dataclass
class _FakeSSH:
    scripted: dict = field(default_factory=dict)

    def run(self, host, cmd: str, *, timeout_s=None, stdout_tee=None, pty=True) -> SSHResult:
        for key, result in self.scripted.items():
            if key in cmd:
                return result
        # docker inspect must return "running" for preflight container_up check.
        if "docker inspect" in cmd:
            return SSHResult(exit_code=0, stdout="running", stderr="", duration_s=0.01)
        return SSHResult(exit_code=0, stdout="ok", stderr="", duration_s=0.01)


@dataclass
class _FakeRsync:
    materialize: bool = True

    def push(self, host, local_path, remote_path) -> RsyncResult:
        return RsyncResult(exit_code=0, stdout="", stderr="", duration_s=0.0)

    def pull(self, host, remote_path: str, local_path: Path) -> RsyncResult:
        if self.materialize:
            local_path.mkdir(parents=True, exist_ok=True)
            (local_path / "manifest.json").write_text(
                json.dumps({"schema_version": "1.0", "phases": {"training": {"status": "completed"}}})
            )
            (local_path / "training.json").write_text(json.dumps({"schema_version": "1.0"}))
            (local_path / "startup.json").write_text(json.dumps({"schema_version": "1.0"}))
        return RsyncResult(exit_code=0, stdout="", stderr="", duration_s=0.0)


def _write_fleet(tmp_path: Path) -> Fleet:
    return Fleet(
        fleet_name="t",
        hosts=[
            ValkyrieConfig(host="v1", ssh_user="odin"),
            ValkyrieConfig(host="v2", ssh_user="odin"),
        ],
    )


def _write_env_list(tmp_path: Path) -> Path:
    from tools.odin.common.env_list import EnvEntry, EnvList, write_env_list

    el = EnvList()
    el.groups["direct/ant"] = [
        EnvEntry(
            task_id="Isaac-Ant-Direct-v0",
            entry_point="isaaclab_tasks.direct.ant:AntEnv",
            env_cfg_entry_point="isaaclab_tasks.direct.ant.ant_env_cfg:AntEnvCfg",
            group="direct/ant",
            has_rsl_rl=True,
            has_skrl=True,
            has_rl_games=False,
            framework="rsl_rl",
            num_envs=4096,
            max_iterations=300,
            keep=True,
            status="current",
        )
    ]
    p = tmp_path / "physx.yaml"
    write_env_list(p, el, generator="test")
    return p


def test_resolve_dispatch_dir_creates_new(tmp_path: Path):
    d = resolve_dispatch_dir(tmp_path / "odin_runs", resume=None)
    assert d.exists()
    assert d.parent == tmp_path / "odin_runs"


def test_resolve_dispatch_dir_resume_latest(tmp_path: Path):
    root = tmp_path / "odin_runs"
    root.mkdir()
    # Create two simulated prior dispatch dirs.
    (root / "20260420-100000").mkdir()
    (root / "20260421-120000").mkdir()
    d = resolve_dispatch_dir(root, resume="LATEST")
    assert d.name == "20260421-120000"


def test_resolve_dispatch_dir_resume_named(tmp_path: Path):
    root = tmp_path / "odin_runs"
    (root / "20260420-100000").mkdir(parents=True)
    d = resolve_dispatch_dir(root, resume="20260420-100000")
    assert d.name == "20260420-100000"


def test_run_dispatch_happy_path(tmp_path: Path):
    fleet = _write_fleet(tmp_path)
    physx = _write_env_list(tmp_path)
    dispatch_dir = tmp_path / "odin_runs" / "20260422-220000"
    dispatch_dir.mkdir(parents=True)
    state = run_dispatch(
        fleet=fleet,
        physx_yaml=physx,
        newton_yaml=None,
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(seeds=[42], detached_mode=False),
        ssh=_FakeSSH(),
        rsync=_FakeRsync(),
    )
    assert state is not None
    assert len(state.jobs) == 1
    assert state.jobs[0].status == "completed"
    assert (dispatch_dir / "dispatch.json").exists()
    # Bundle rsync'd back.
    assert (dispatch_dir / state.jobs[0].bundle_dir_name / "manifest.json").exists()


def test_run_dispatch_sweeps_orphan_trainers_per_host(tmp_path: Path):
    """Pre-dispatch sweep must run on every healthy host before workers
    start — otherwise a wedged GPU from yesterday's legacy-PTY orphan
    will spawn the same cascade we already debugged."""

    @dataclass
    class _SweepRecordingSSH(_FakeSSH):
        sweep_calls: list = field(default_factory=list)

        def run(self, host, cmd: str, *, timeout_s=None, stdout_tee=None, pty=True):
            if "pkill" in cmd and "benchmark_rsl_rl" in cmd:
                self.sweep_calls.append(host.host)
                return SSHResult(exit_code=0, stdout="2\n", stderr="", duration_s=0.01)
            return super().run(host, cmd, timeout_s=timeout_s, stdout_tee=stdout_tee, pty=pty)

    fleet = _write_fleet(tmp_path)
    physx = _write_env_list(tmp_path)
    dispatch_dir = tmp_path / "odin_runs" / "20260430-sweep"
    dispatch_dir.mkdir(parents=True)
    ssh = _SweepRecordingSSH()
    run_dispatch(
        fleet=fleet,
        physx_yaml=physx,
        newton_yaml=None,
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(seeds=[42], detached_mode=False),
        ssh=ssh,
        rsync=_FakeRsync(),
    )
    # Both healthy hosts swept exactly once before workers start.
    assert sorted(ssh.sweep_calls) == ["v1", "v2"]


def test_run_dispatch_preflight_fail_fast(tmp_path: Path):
    fleet = _write_fleet(tmp_path)
    physx = _write_env_list(tmp_path)
    dispatch_dir = tmp_path / "odin_runs" / "20260422-220000"
    dispatch_dir.mkdir(parents=True)
    # docker ps fails → preflight fails → run_dispatch aborts.
    with pytest.raises(RuntimeError, match="preflight"):
        run_dispatch(
            fleet=fleet,
            physx_yaml=physx,
            newton_yaml=None,
            dispatch_dir=dispatch_dir,
            options=DispatchOptions(seeds=[42], detached_mode=False),
            ssh=_FakeSSH(
                scripted={
                    "docker ps": SSHResult(exit_code=1, stdout="", stderr="daemon unreachable", duration_s=0.01),
                }
            ),
            rsync=_FakeRsync(),
        )
    # Preflight.json is written on failure for audit.
    assert (dispatch_dir / "preflight.json").exists()


def test_run_dispatch_resume_preserves_completed(tmp_path: Path):
    fleet = _write_fleet(tmp_path)
    physx = _write_env_list(tmp_path)
    dispatch_dir = tmp_path / "odin_runs" / "20260422-220000"
    dispatch_dir.mkdir(parents=True)

    # First run: completes the single job.
    run_dispatch(
        fleet=fleet,
        physx_yaml=physx,
        newton_yaml=None,
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(seeds=[42], detached_mode=False),
        ssh=_FakeSSH(),
        rsync=_FakeRsync(),
    )
    first = read_dispatch_state(dispatch_dir)
    assert first.jobs[0].status == "completed"

    # Second run (resume) MUST NOT re-run a completed job. Match Hugin/Munin
    # dispatch shape ('hugin/run.py' or 'munin/run.py') rather than any
    # 'docker exec' — preflight's gpu_present probe legitimately uses
    # docker exec nvidia-smi -L without dispatching a job.
    class _AssertNoDispatch(_FakeSSH):
        def run(self, host, cmd, *, timeout_s=None, stdout_tee=None, pty=True):
            if "hugin/run.py" in cmd or "munin/run.py" in cmd:
                raise AssertionError(f"resume should not re-dispatch completed jobs; cmd={cmd!r}")
            return super().run(host, cmd, timeout_s=timeout_s, stdout_tee=stdout_tee)

    run_dispatch(
        fleet=fleet,
        physx_yaml=physx,
        newton_yaml=None,
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(seeds=[42], detached_mode=False),
        ssh=_AssertNoDispatch(),
        rsync=_FakeRsync(),
    )
    second = read_dispatch_state(dispatch_dir)
    assert second.jobs[0].status == "completed"


def test_run_dispatch_writes_aggregate_json(tmp_path: Path):
    """run_dispatch auto-invokes valhalla.aggregator at the tail."""
    fleet = _write_fleet(tmp_path)
    env_yaml = _write_env_list(tmp_path)
    dispatch_dir = tmp_path / "20260423-110000"
    dispatch_dir.mkdir()

    run_dispatch(
        fleet=fleet,
        physx_yaml=env_yaml,
        newton_yaml=None,
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(seeds=[42], detached_mode=False),
        ssh=_FakeSSH(),
        rsync=_FakeRsync(),
    )
    assert (dispatch_dir / "aggregate.json").exists()
    with (dispatch_dir / "aggregate.json").open("r") as fh:
        agg = json.load(fh)
    assert agg["schema_version"] == "1.0"
    # Dispatch enumerated exactly one (task, seed) pair; the aggregator
    # saw it (whether completed or failed is out of scope — this test
    # verifies the wiring, not the bundle-validation correctness).
    assert agg["totals"]["runs"] == 1


def test_run_dispatch_skip_aggregate_leaves_no_file(tmp_path: Path):
    """skip_aggregate=True suppresses the auto-call."""
    fleet = _write_fleet(tmp_path)
    env_yaml = _write_env_list(tmp_path)
    dispatch_dir = tmp_path / "20260423-110000"
    dispatch_dir.mkdir()

    run_dispatch(
        fleet=fleet,
        physx_yaml=env_yaml,
        newton_yaml=None,
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(seeds=[42], skip_aggregate=True, detached_mode=False),
        ssh=_FakeSSH(),
        rsync=_FakeRsync(),
    )
    assert not (dispatch_dir / "aggregate.json").exists()


def test_resume_preserves_skipped_array(tmp_path: Path):
    """A prior dispatch.json's skipped[] survives --resume verbatim.

    Even if the on-disk yaml has changed in the meantime, resume must
    not re-evaluate skipped — the dispatch's identity is fixed at
    first-write.
    """
    from tools.odin.asgard.jobs import SkippedEntry
    from tools.odin.asgard.state import (
        SCHEMA_VERSION,
        DispatchState,
        FleetSnapshot,
        write_dispatch_state,
    )

    fleet = _write_fleet(tmp_path)
    physx = _write_env_list(tmp_path)
    dispatch_dir = tmp_path / "odin_runs" / "20260427-120000"
    dispatch_dir.mkdir(parents=True)

    seed_skipped = [
        SkippedEntry(
            task_id="Isaac-Foo-v0",
            framework="rsl_rl",
            backend="physx",
            seed=42,
            reason="preset_unsupported",
            presets_available=[],
        ),
    ]
    prior = DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id="20260427-120000",
        started_at="2026-04-27T12:00:00Z",
        ended_at=None,
        seeds=[42],
        commit_sha="",
        fleet=[FleetSnapshot(host="v1", status="idle")],
        jobs=[],
        skipped=seed_skipped,
    )
    write_dispatch_state(dispatch_dir, prior)

    state = run_dispatch(
        fleet=fleet,
        physx_yaml=physx,
        newton_yaml=None,
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(seeds=[42], skip_aggregate=True, detached_mode=False),
        ssh=_FakeSSH(),
        rsync=_FakeRsync(),
    )
    assert len(state.skipped) == 1
    assert state.skipped[0].task_id == "Isaac-Foo-v0"
    assert state.skipped[0].reason == "preset_unsupported"


def _ssh_localhost_works() -> bool:
    """Probe: can we `ssh localhost "echo ok"` without a password?"""
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "localhost", "echo ok"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.returncode == 0 and "ok" in r.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


@pytest.fixture
def stub_ssh_runner(monkeypatch, tmp_path: Path):
    """Replace ShellSSHRunner.run's docker-exec command with a local stub that
    materialises a valid bundle and exits 0."""
    from tools.odin.asgard import worker as worker_mod

    real_build = worker_mod._build_docker_exec_cmd

    def _fake_build(host: ValkyrieConfig, job) -> str:
        bundle_dir = f"{host.isaaclab_path}/odin_runs/{job.bundle_dir_name}"
        manifest = {
            "schema_version": "1.0",
            "phases": {"training": {"status": "completed"}, "startup": {"status": "completed"}},
        }
        training = {"schema_version": "1.0"}
        startup = {"schema_version": "1.0"}
        manifest_s = json.dumps(manifest).replace("'", r"\'")
        training_s = json.dumps(training).replace("'", r"\'")
        startup_s = json.dumps(startup).replace("'", r"\'")
        return (
            f"mkdir -p {bundle_dir} && "
            f"printf '%s' '{manifest_s}' > {bundle_dir}/manifest.json && "
            f"printf '%s' '{training_s}' > {bundle_dir}/training.json && "
            f"printf '%s' '{startup_s}' > {bundle_dir}/startup.json"
        )

    monkeypatch.setattr(worker_mod, "_build_docker_exec_cmd", _fake_build)
    yield
    monkeypatch.setattr(worker_mod, "_build_docker_exec_cmd", real_build)


@pytest.fixture
def stub_provisioner(monkeypatch):
    """Preflight (docker ps / docker inspect) would fail on vanilla localhost.
    Short-circuit preflight and provisioner to always pass."""
    from tools.odin.asgard import preflight as pf
    from tools.odin.asgard import provisioner as pv

    def _fake_pf(host, *, ssh, auto_restart=True, newton_cuda_floor=None):
        return pf.PreflightResult(
            host=host.host,
            ok=True,
            checks={"ssh_reach": True, "docker_running": True, "container_up": True, "isaaclab_present": True},
            message="",
        )

    def _fake_pv(host, working_tree, *, fresh, ssh, rsync):
        return pv.ProvisionResult(host=host.host, ok=True, commit_sha="integration-stub")

    monkeypatch.setattr("tools.odin.asgard.runner.preflight_valkyrie", _fake_pf)
    monkeypatch.setattr("tools.odin.asgard.runner.provision_valkyrie", _fake_pv)


@pytest.mark.slow
def test_pre_dispatch_summary_renders_native_mismatch_line(tmp_path: Path, stub_ssh_runner, stub_provisioner, capsys):
    """The [INFO] block grouped by reason shows 'native: <X>' for native_backend_mismatch."""
    if not _ssh_localhost_works():
        pytest.skip("ssh localhost not configured")

    el = EnvList()
    el.groups["direct/quadcopter"] = [
        EnvEntry(
            task_id="Isaac-Quadcopter-Direct-v0",
            entry_point="ep:E",
            env_cfg_entry_point="ec:E",
            group="direct/quadcopter",
            has_rsl_rl=True,
            has_skrl=True,
            framework="rsl_rl",
            num_envs=4096,
            max_iterations=10,
            keep=True,
            presets_available=[],
            native_backend="physx",
        ),
    ]
    physx_yaml = tmp_path / "physx.yaml"
    write_env_list(physx_yaml, el, generator="test")

    dispatch_dir = tmp_path / "20260427-150000"
    dispatch_dir.mkdir()
    fleet = Fleet(
        fleet_name="loopback-test",
        hosts=[
            ValkyrieConfig(
                host="localhost",
                ssh_user=os.environ.get("USER") or "root",
                ssh_key=None,
                isaaclab_path=str(tmp_path / "remote_isaaclab"),
                container_name="loopback-container",
            ),
        ],
    )
    run_dispatch(
        fleet=fleet,
        physx_yaml=None,
        newton_yaml=physx_yaml,  # request newton on a physx-native task
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(seeds=[42], skip_aggregate=True, per_job_timeout_s=60, detached_mode=False),
        ssh=ShellSSHRunner(),
        rsync=ShellRsyncRunner(),
    )
    captured = capsys.readouterr()
    out = captured.out
    assert "native_backend_mismatch" in out
    assert "native: physx" in out


def test_runner_handles_recovered_event(tmp_path):
    """Recovered event updates fleet[host].last_error but not status."""
    from tools.odin.asgard import runner as runner_mod
    from tools.odin.asgard.state import SCHEMA_VERSION, DispatchState, FleetSnapshot
    from tools.odin.asgard.worker import StateEvent

    state = DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id="20260427-200000",
        started_at="2026-04-27T20:00:00Z",
        ended_at=None,
        seeds=[42],
        commit_sha="abc",
        fleet=[FleetSnapshot(host="v1", status="busy", last_error=None)],
        jobs=[],
    )
    ev = StateEvent(run_id="r1", host="v1", transition="recovered")
    runner_mod._apply_state_event(state, ev)
    fs = next(f for f in state.fleet if f.host == "v1")
    assert fs.status == "busy"  # unchanged
    assert fs.last_error == "gpu_lost: recovered"


def test_runner_handles_host_down_event(tmp_path):
    """host_down event marks host status='down' with structured last_error."""
    from tools.odin.asgard import runner as runner_mod
    from tools.odin.asgard.jobs import FailureInfo
    from tools.odin.asgard.state import SCHEMA_VERSION, DispatchState, FleetSnapshot
    from tools.odin.asgard.worker import StateEvent

    state = DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id="20260427-200000",
        started_at="2026-04-27T20:00:00Z",
        ended_at=None,
        seeds=[42],
        commit_sha="abc",
        fleet=[FleetSnapshot(host="v1", status="busy", last_error=None)],
        jobs=[],
    )
    ev = StateEvent(
        run_id="r1",
        host="v1",
        transition="host_down",
        failure=FailureInfo(kind="gpu_lost", message="docker_restart_failed: daemon down"),
    )
    runner_mod._apply_state_event(state, ev)
    fs = next(f for f in state.fleet if f.host == "v1")
    assert fs.status == "down"
    assert fs.last_error == "gpu_lost: docker_restart_failed: daemon down"


def test_runner_host_down_resets_in_flight_job_to_pending(tmp_path):
    """When host_down fires, the in-flight job (matching ev.run_id) was
    re-queued by the worker; the runner must flip its dispatch.json row from
    ``running`` back to ``pending`` so it can be picked up again — and so
    the post-dispatch sweep catches it if no host remains."""
    from tools.odin.asgard import runner as runner_mod
    from tools.odin.asgard.jobs import FailureInfo, JobEntry
    from tools.odin.asgard.state import SCHEMA_VERSION, DispatchState, FleetSnapshot
    from tools.odin.asgard.worker import StateEvent

    in_flight = JobEntry(
        run_id="r-flight",
        task_id="X",
        framework="rsl_rl",
        backend="physx",
        num_envs=4096,
        max_iterations=10,
        seed=42,
        bundle_dir_name="r-flight",
    )
    in_flight.status = "running"
    in_flight.assigned_to = "v1"
    in_flight.started_at = "2026-04-27T20:01:00Z"
    other_done = JobEntry(
        run_id="r-other",
        task_id="X",
        framework="rsl_rl",
        backend="physx",
        num_envs=4096,
        max_iterations=10,
        seed=43,
        bundle_dir_name="r-other",
    )
    other_done.status = "completed"
    state = DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id="20260427-200000",
        started_at="2026-04-27T20:00:00Z",
        ended_at=None,
        seeds=[42, 43],
        commit_sha="abc",
        fleet=[FleetSnapshot(host="v1", status="busy", last_error=None)],
        jobs=[in_flight, other_done],
    )
    ev = StateEvent(
        run_id="r-flight",
        host="v1",
        transition="host_down",
        failure=FailureInfo(kind="circuit_breaker", message="3 consecutive failures on v1; quarantining host"),
    )
    runner_mod._apply_state_event(state, ev)

    j = next(j for j in state.jobs if j.run_id == "r-flight")
    assert j.status == "pending", "in-flight job must be reset to pending so it gets picked up / swept"
    assert j.assigned_to is None
    assert j.started_at is None
    # Other completed job is untouched.
    other = next(j for j in state.jobs if j.run_id == "r-other")
    assert other.status == "completed"


def test_sweep_pending_terminal_fails_when_all_hosts_down(tmp_path):
    """Post-loop sweep marks pending jobs as gpu_lost when no host is healthy."""
    from tools.odin.asgard import runner as runner_mod
    from tools.odin.asgard.jobs import JobEntry
    from tools.odin.asgard.state import SCHEMA_VERSION, DispatchState, FleetSnapshot

    pending_job = JobEntry(
        run_id="r-pending",
        task_id="X",
        framework="rsl_rl",
        backend="physx",
        num_envs=4096,
        max_iterations=300,
        seed=42,
        bundle_dir_name="r-pending",
        preferred_not={"v1", "v2"},
    )
    state = DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id="20260427-200000",
        started_at="2026-04-27T20:00:00Z",
        ended_at=None,
        seeds=[42],
        commit_sha="abc",
        fleet=[
            FleetSnapshot(host="v1", status="down", last_error="gpu_lost: recovery_failed (x)"),
            FleetSnapshot(host="v2", status="down", last_error="gpu_lost: recovery_failed (y)"),
        ],
        jobs=[pending_job],
    )

    runner_mod._sweep_pending_after_dispatch(state)

    j = state.jobs[0]
    assert j.status == "failed"
    assert j.failure is not None
    assert j.failure.kind == "gpu_lost"
    assert "no healthy host" in j.failure.message
    assert j.failure.details["preferred_not"] == ["v1", "v2"]
    assert j.ended_at is not None


def test_sweep_pending_leaves_pending_when_fleet_healthy(tmp_path):
    """Sweep is a no-op if no host is marked down."""
    from tools.odin.asgard import runner as runner_mod
    from tools.odin.asgard.jobs import JobEntry
    from tools.odin.asgard.state import SCHEMA_VERSION, DispatchState, FleetSnapshot

    pending_job = JobEntry(
        run_id="r-pending",
        task_id="X",
        framework="rsl_rl",
        backend="physx",
        num_envs=4096,
        max_iterations=300,
        seed=42,
        bundle_dir_name="r-pending",
    )
    state = DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id="20260427-200000",
        started_at="2026-04-27T20:00:00Z",
        ended_at=None,
        seeds=[42],
        commit_sha="abc",
        fleet=[FleetSnapshot(host="v1", status="idle", last_error=None)],
        jobs=[pending_job],
    )

    runner_mod._sweep_pending_after_dispatch(state)

    assert state.jobs[0].status == "pending"


def test_apply_state_event_circuit_breaker_adds_quarantined_host(tmp_path):
    """host_down(circuit_breaker) appends a QuarantinedHost entry."""
    from tools.odin.asgard import runner as runner_mod
    from tools.odin.asgard.jobs import FailureInfo
    from tools.odin.asgard.state import SCHEMA_VERSION, DispatchState, FleetSnapshot
    from tools.odin.asgard.worker import StateEvent

    state = DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id="20260428-100000",
        started_at="2026-04-28T10:00:00Z",
        ended_at=None,
        seeds=[42],
        commit_sha="abc",
        fleet=[FleetSnapshot(host="v1", status="busy", last_error=None)],
        jobs=[],
    )
    ev = StateEvent(
        run_id="r-cb",
        host="v1",
        transition="host_down",
        failure=FailureInfo(
            kind="circuit_breaker",
            message="3 consecutive failures on v1; quarantining host",
        ),
    )
    runner_mod._apply_state_event(state, ev)

    assert len(state.quarantined_hosts) == 1
    qh = state.quarantined_hosts[0]
    assert qh.host == "v1"
    assert qh.reason == "circuit_breaker"
    assert qh.last_run_id == "r-cb"
    assert qh.at  # non-empty ISO timestamp


def test_apply_state_event_gpu_lost_does_not_add_quarantined_host(tmp_path):
    """host_down(gpu_lost) does NOT add to quarantined_hosts."""
    from tools.odin.asgard import runner as runner_mod
    from tools.odin.asgard.jobs import FailureInfo
    from tools.odin.asgard.state import SCHEMA_VERSION, DispatchState, FleetSnapshot
    from tools.odin.asgard.worker import StateEvent

    state = DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id="20260428-100000",
        started_at="2026-04-28T10:00:00Z",
        ended_at=None,
        seeds=[42],
        commit_sha="abc",
        fleet=[FleetSnapshot(host="v1", status="busy", last_error=None)],
        jobs=[],
    )
    ev = StateEvent(
        run_id="r-gl",
        host="v1",
        transition="host_down",
        failure=FailureInfo(kind="gpu_lost", message="docker_restart_failed: daemon down"),
    )
    runner_mod._apply_state_event(state, ev)

    assert state.quarantined_hosts == []


def test_run_dispatch_marks_newton_jobs_failed_when_no_capable_host(tmp_path: Path, monkeypatch):
    """When no host has newton_available=True after provisioning, pending newton
    jobs are marked failed with kind='newton_floor' and an upgrade-hint message."""
    from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
    from tools.odin.asgard.preflight import PreflightResult
    from tools.odin.asgard.provisioner import ProvisionResult
    from tools.odin.asgard.runner import DispatchOptions, run_dispatch

    # Fake preflight: host healthy, but newton_available=False.
    def _fake_pf(host, *, ssh, auto_restart=True, newton_cuda_floor=None):
        return PreflightResult(
            host=host.host,
            ok=True,
            checks={
                "ssh_reach": True,
                "docker_running": True,
                "container_up": True,
                "isaaclab_present": True,
                "gpu_present": True,
            },
            message="newton unavailable: host CUDA 12.2 < newton floor 12.4",
            cuda_version=(12, 2),
            newton_available=False,
        )

    # Fake provisioner: always succeeds.
    def _fake_pv(host, working_tree, *, fresh, ssh, rsync):
        return ProvisionResult(host=host.host, ok=True, commit_sha="test-stub")

    monkeypatch.setattr("tools.odin.asgard.runner.preflight_valkyrie", _fake_pf)
    monkeypatch.setattr("tools.odin.asgard.runner.provision_valkyrie", _fake_pv)

    # Build minimal newton_yaml using EnvList, creating one Newton-capable task.
    el = EnvList()
    el.groups["test/task"] = [
        EnvEntry(
            task_id="Isaac-Test-Direct-v0",
            entry_point="isaaclab_tasks.test:TestEnv",
            env_cfg_entry_point="isaaclab_tasks.test:TestEnvCfg",
            group="test/task",
            has_rsl_rl=True,
            has_skrl=False,
            has_rl_games=False,
            framework="rsl_rl",
            num_envs=1024,
            max_iterations=100,
            keep=True,
            status="current",
            presets_available=["newton"],
        )
    ]
    newton_yaml = tmp_path / "newton.yaml"
    write_env_list(newton_yaml, el, generator="test")

    fleet = Fleet(
        fleet_name="test",
        hosts=[ValkyrieConfig(host="v1", ssh_user="odin", isaaclab_path="/h/x")],
    )

    dispatch_dir = tmp_path / "20260429-000000"
    dispatch_dir.mkdir()

    state = run_dispatch(
        fleet=fleet,
        physx_yaml=None,
        newton_yaml=newton_yaml,
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(seeds=[42], skip_aggregate=True, skip_preflight=False, detached_mode=False),
        ssh=_FakeSSH(),
        rsync=_FakeRsync(),
    )

    newton_jobs = [j for j in state.jobs if j.backend == "newton"]
    assert newton_jobs, "expected at least one newton job in the state"
    assert all(j.status == "failed" for j in newton_jobs), "all newton jobs should be marked failed"
    assert all(j.failure is not None and j.failure.kind == "newton_floor" for j in newton_jobs), (
        "all newton jobs should have kind='newton_floor'"
    )
    assert all("odin-cuda install --target 12.4" in j.failure.message for j in newton_jobs), (
        "all newton jobs should have upgrade-hint message"
    )


def test_consume_live_retries_requeues_failed_job(tmp_path: Path):
    """A pending DB retry for a failed job resets and requeues that job exactly once."""
    from tools.odin.asgard import runner as runner_mod
    from tools.odin.asgard.jobs import FailureInfo, JobEntry

    dispatch_id = "20260430-110509"
    job = JobEntry(
        run_id="run-a",
        task_id="Isaac-Ant-Direct-v0",
        framework="rsl_rl",
        backend="physx",
        num_envs=4096,
        max_iterations=10,
        seed=42,
        bundle_dir_name="run-a",
    )
    job.status = "failed"
    job.failure = FailureInfo(kind="hugin_crash", message="boom")
    job.assigned_to = "v1"
    job.started_at = "2026-04-30T11:00:00Z"
    job.ended_at = "2026-04-30T11:01:00Z"
    job.attempts = 1
    retry_db = RetryDB(tmp_path)
    retry_db.toggle(dispatch_id, job.run_id)
    job_q: queue.Queue = queue.Queue()
    live_retry_run_ids: set[str] = set()

    added = runner_mod._consume_live_retries(
        retry_db=retry_db,
        dispatch_id=dispatch_id,
        jobs_by_id={job.run_id: job},
        job_q=job_q,
        live_retry_run_ids=live_retry_run_ids,
    )

    assert added == 1
    assert job_q.get_nowait() is job
    assert live_retry_run_ids == {job.run_id}
    assert job.status == "pending"
    assert job.failure is None
    assert job.assigned_to is None
    assert job.started_at is None
    assert job.ended_at is None
    assert job.attempts == 1


def test_consume_live_retries_ignores_non_failed_and_unknown_rows(tmp_path: Path):
    """Only failed jobs in the active dispatch are eligible for live ingestion."""
    from tools.odin.asgard import runner as runner_mod
    from tools.odin.asgard.jobs import FailureInfo, JobEntry

    dispatch_id = "20260430-110509"
    jobs_by_id: dict[str, JobEntry] = {}
    for status in ["pending", "running", "completed"]:
        job = JobEntry(
            run_id=f"run-{status}",
            task_id="Isaac-Ant-Direct-v0",
            framework="rsl_rl",
            backend="physx",
            num_envs=4096,
            max_iterations=10,
            seed=42,
            bundle_dir_name=f"run-{status}",
        )
        job.status = status
        if status == "completed":
            job.failure = None
        elif status == "running":
            job.assigned_to = "v1"
        jobs_by_id[job.run_id] = job

    failed_from_other_dispatch = JobEntry(
        run_id="run-other",
        task_id="Isaac-Ant-Direct-v0",
        framework="rsl_rl",
        backend="physx",
        num_envs=4096,
        max_iterations=10,
        seed=43,
        bundle_dir_name="run-other",
    )
    failed_from_other_dispatch.status = "failed"
    failed_from_other_dispatch.failure = FailureInfo(kind="hugin_crash", message="boom")
    jobs_by_id[failed_from_other_dispatch.run_id] = failed_from_other_dispatch

    retry_db = RetryDB(tmp_path)
    for run_id in ["run-pending", "run-running", "run-completed", "run-unknown"]:
        retry_db.toggle(dispatch_id, run_id)
    retry_db.toggle("20260430-120000", failed_from_other_dispatch.run_id)
    job_q: queue.Queue = queue.Queue()

    added = runner_mod._consume_live_retries(
        retry_db=retry_db,
        dispatch_id=dispatch_id,
        jobs_by_id=jobs_by_id,
        job_q=job_q,
        live_retry_run_ids=set(),
    )

    assert added == 0
    assert job_q.empty()
    assert retry_db.read_pending(dispatch_id) == {"run-pending", "run-running", "run-completed", "run-unknown"}


def test_mark_live_retry_consumed_records_completed_outcome(tmp_path: Path):
    from tools.odin.asgard import runner as runner_mod
    from tools.odin.asgard.worker import StateEvent

    retry_db = RetryDB(tmp_path)
    dispatch_id = "20260430-110509"
    retry_db.toggle(dispatch_id, "run-a")

    runner_mod._mark_live_retry_consumed(
        retry_db=retry_db,
        dispatch_id=dispatch_id,
        ev=StateEvent(run_id="run-a", host="v1", transition="completed"),
        live_retry_run_ids={"run-a"},
    )

    rows = retry_db.list_for_dispatch(dispatch_id)
    assert rows[0].retried_at is not None
    assert rows[0].retry_dispatch_id == dispatch_id
    assert rows[0].retry_outcome == "completed"
    assert rows[0].retry_failure_kind is None


def test_mark_live_retry_consumed_records_failed_outcome(tmp_path: Path):
    from tools.odin.asgard import runner as runner_mod
    from tools.odin.asgard.jobs import FailureInfo
    from tools.odin.asgard.worker import StateEvent

    retry_db = RetryDB(tmp_path)
    dispatch_id = "20260430-110509"
    retry_db.toggle(dispatch_id, "run-a")

    runner_mod._mark_live_retry_consumed(
        retry_db=retry_db,
        dispatch_id=dispatch_id,
        ev=StateEvent(
            run_id="run-a",
            host="v1",
            transition="failed",
            failure=FailureInfo(kind="gpu_lost", message="lost gpu"),
        ),
        live_retry_run_ids={"run-a"},
    )

    rows = retry_db.list_for_dispatch(dispatch_id)
    assert rows[0].retry_outcome == "failed"
    assert rows[0].retry_failure_kind == "gpu_lost"


def test_run_dispatch_consumes_live_retry_for_current_dispatch(tmp_path: Path):
    """A live runner requeues a failed job after the dashboard/CLI adds it to the retry DB."""

    class _LiveRetrySSH(_FakeSSH):
        def __init__(self):
            super().__init__()
            self.seed42_calls = 0
            self.seed43_started = threading.Event()
            self.release_seed43 = threading.Event()

        def run(self, host, cmd: str, *, timeout_s=None, stdout_tee=None, pty=True) -> SSHResult:
            if "hugin/run.py" not in cmd and "munin/run.py" not in cmd:
                return super().run(host, cmd, timeout_s=timeout_s, stdout_tee=stdout_tee, pty=pty)
            if "seed42" in cmd:
                self.seed42_calls += 1
                if self.seed42_calls == 1:
                    return SSHResult(exit_code=1, stdout="", stderr="training failed", duration_s=0.01)
                return SSHResult(exit_code=0, stdout="ok", stderr="", duration_s=0.01)
            if "seed43" in cmd:
                self.seed43_started.set()
                if self.release_seed43.wait(timeout=30):
                    return SSHResult(exit_code=0, stdout="ok", stderr="", duration_s=0.01)
                return SSHResult(exit_code=1, stdout="", stderr="seed43 release timeout", duration_s=30.0)
            return super().run(host, cmd, timeout_s=timeout_s, stdout_tee=stdout_tee, pty=pty)

    fleet = _write_fleet(tmp_path)
    physx = _write_env_list(tmp_path)
    dispatch_id = "20260430-110509"
    dispatch_dir = tmp_path / "odin_runs" / dispatch_id
    dispatch_dir.mkdir(parents=True)
    ssh = _LiveRetrySSH()
    result_holder: dict[str, object] = {}

    def _run_dispatch() -> None:
        try:
            result_holder["state"] = run_dispatch(
                fleet=fleet,
                physx_yaml=physx,
                newton_yaml=None,
                dispatch_dir=dispatch_dir,
                options=DispatchOptions(
                    seeds=[42, 43],
                    skip_aggregate=True,
                    live_retry_poll_s=0.05,
                    detached_mode=False,
                ),
                ssh=ssh,
                rsync=_FakeRsync(),
            )
        except BaseException as exc:
            result_holder["exc"] = exc

    thread = threading.Thread(target=_run_dispatch)
    thread.start()
    assert ssh.seed43_started.wait(timeout=10), "seed43 did not start"
    run_id_42 = f"rsl-rl_physx_Isaac-Ant-Direct-v0_{dispatch_id}_seed42"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        state = read_dispatch_state(dispatch_dir)
        if state is not None:
            job = next(j for j in state.jobs if j.run_id == run_id_42)
            if job.status == "failed":
                break
        time.sleep(0.05)
    else:
        raise AssertionError("seed42 did not reach failed state before live retry")

    RetryDB(dispatch_dir.parent).toggle(dispatch_id, run_id_42)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if ssh.seed42_calls >= 2:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("seed42 was not retried while dispatch was live")

    ssh.release_seed43.set()
    thread.join(timeout=10)

    assert "exc" not in result_holder
    final_state = result_holder["state"]
    final_job = next(j for j in final_state.jobs if j.run_id == run_id_42)
    assert final_job.status == "completed"
    assert final_job.attempts == 2
    rows = RetryDB(dispatch_dir.parent).list_for_dispatch(dispatch_id)
    assert rows[0].retry_dispatch_id == dispatch_id
    assert rows[0].retry_outcome == "completed"
    assert RetryDB(dispatch_dir.parent).read_pending(dispatch_id) == set()
