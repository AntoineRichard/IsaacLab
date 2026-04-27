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
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
from tools.odin.asgard.runner import DispatchOptions, resolve_dispatch_dir, run_dispatch
from tools.odin.asgard.state import read_dispatch_state
from tools.odin.asgard.transport import RsyncResult, SSHResult


@dataclass
class _FakeSSH:
    scripted: dict = field(default_factory=dict)

    def run(self, host, cmd: str, *, timeout_s=None, stdout_tee=None) -> SSHResult:
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
        options=DispatchOptions(seeds=[42]),
        ssh=_FakeSSH(),
        rsync=_FakeRsync(),
    )
    assert state is not None
    assert len(state.jobs) == 1
    assert state.jobs[0].status == "completed"
    assert (dispatch_dir / "dispatch.json").exists()
    # Bundle rsync'd back.
    assert (dispatch_dir / state.jobs[0].bundle_dir_name / "manifest.json").exists()


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
            options=DispatchOptions(seeds=[42]),
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
        options=DispatchOptions(seeds=[42]),
        ssh=_FakeSSH(),
        rsync=_FakeRsync(),
    )
    first = read_dispatch_state(dispatch_dir)
    assert first.jobs[0].status == "completed"

    # Second run (resume) MUST NOT re-run a completed job.
    class _AssertNoDispatch(_FakeSSH):
        def run(self, host, cmd, *, timeout_s=None, stdout_tee=None):
            if "docker exec" in cmd:
                raise AssertionError(f"resume should not re-dispatch completed jobs; cmd={cmd!r}")
            return super().run(host, cmd, timeout_s=timeout_s, stdout_tee=stdout_tee)

    run_dispatch(
        fleet=fleet,
        physx_yaml=physx,
        newton_yaml=None,
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(seeds=[42]),
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
        options=DispatchOptions(seeds=[42]),
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
        options=DispatchOptions(seeds=[42], skip_aggregate=True),
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
        options=DispatchOptions(seeds=[42], skip_aggregate=True),
        ssh=_FakeSSH(),
        rsync=_FakeRsync(),
    )
    assert len(state.skipped) == 1
    assert state.skipped[0].task_id == "Isaac-Foo-v0"
    assert state.skipped[0].reason == "preset_unsupported"
