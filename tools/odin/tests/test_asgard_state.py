# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :mod:`tools.odin.asgard.state`."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.odin.asgard.queue import FailureInfo, JobEntry
from tools.odin.asgard.state import (
    DispatchState,
    FleetSnapshot,
    read_dispatch_state,
    reset_in_flight_to_pending,
    write_dispatch_state,
)


def _job(run_id: str, status: str = "pending", **kw) -> JobEntry:
    return JobEntry(
        run_id=run_id,
        task_id="Isaac-Ant-Direct-v0",
        framework="rsl_rl",
        backend="physx",
        num_envs=4096,
        max_iterations=300,
        seed=42,
        bundle_dir_name=run_id,
        status=status,
        **kw,
    )


def _state(jobs: list[JobEntry]) -> DispatchState:
    return DispatchState(
        schema_version="1.0",
        dispatch_id="20260422-220000",
        started_at="2026-04-22T22:00:00Z",
        ended_at=None,
        seeds=[42],
        commit_sha="abc123",
        fleet=[FleetSnapshot(host="h1", status="idle", current_run_id=None, last_error=None)],
        jobs=jobs,
    )


def test_roundtrip_minimal(tmp_path: Path):
    original = _state([_job("rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-220000_seed42")])
    write_dispatch_state(tmp_path, original)
    reloaded = read_dispatch_state(tmp_path)
    assert reloaded.dispatch_id == "20260422-220000"
    assert len(reloaded.jobs) == 1
    assert reloaded.jobs[0].run_id == original.jobs[0].run_id
    assert reloaded.jobs[0].status == "pending"


def test_roundtrip_preserves_failure_info(tmp_path: Path):
    j = _job("run-x", status="failed")
    j.failure = FailureInfo(
        kind="hugin_crash",
        message="exit code 1",
        details={"exit_code": 1, "log_tail_path": "run-x/logs/ssh-tail.log"},
    )
    j.attempts = 1
    write_dispatch_state(tmp_path, _state([j]))

    reloaded = read_dispatch_state(tmp_path)
    rj = reloaded.jobs[0]
    assert rj.failure is not None
    assert rj.failure.kind == "hugin_crash"
    assert rj.failure.details["exit_code"] == 1
    assert rj.attempts == 1


def test_atomic_write_no_partial(tmp_path: Path, monkeypatch):
    """Atomic write must never leave a partial file visible to readers."""
    # Sanity: the implementation uses a temp file + rename. Simulate a crash
    # between temp-file-close and rename by monkeypatching os.replace to raise
    # on the *second* call; assert the original file still parses cleanly.
    import os

    state1 = _state([_job("run-a")])
    write_dispatch_state(tmp_path, state1)
    first_mtime = (tmp_path / "dispatch.json").stat().st_mtime_ns

    state2 = _state([_job("run-a", status="running"), _job("run-b")])
    real_replace = os.replace
    call_count = {"n": 0}

    def _boom(src, dst, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] >= 1:
            raise RuntimeError("simulated crash between write and rename")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(RuntimeError):
        write_dispatch_state(tmp_path, state2)

    monkeypatch.setattr(os, "replace", real_replace)
    # The pre-existing file must still be readable and reflect state1.
    reloaded = read_dispatch_state(tmp_path)
    assert len(reloaded.jobs) == 1
    assert reloaded.jobs[0].run_id == "run-a"
    assert reloaded.jobs[0].status == "pending"
    assert (tmp_path / "dispatch.json").stat().st_mtime_ns == first_mtime


def test_reset_in_flight_flips_running_and_assigned(tmp_path: Path):
    jobs = [
        _job("r-running", status="running", assigned_to="h1"),
        _job("r-assigned", status="assigned", assigned_to="h1"),
        _job("r-completed", status="completed"),
        _job("r-failed", status="failed"),
        _job("r-pending", status="pending"),
    ]
    s = _state(jobs)
    reset_in_flight_to_pending(s)
    statuses = {j.run_id: j.status for j in s.jobs}
    assert statuses["r-running"] == "pending"
    assert statuses["r-assigned"] == "pending"
    assert statuses["r-completed"] == "completed"
    assert statuses["r-failed"] == "failed"
    assert statuses["r-pending"] == "pending"
    # Assignment cleared on reset.
    running_job = next(j for j in s.jobs if j.run_id == "r-running")
    assert running_job.assigned_to is None


def test_read_missing_returns_none(tmp_path: Path):
    assert read_dispatch_state(tmp_path) is None
