# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :mod:`tools.odin.asgard.state`."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.odin.asgard.jobs import FailureInfo, JobEntry, SkippedEntry
from tools.odin.asgard.state import (
    SCHEMA_VERSION,
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
        schema_version=SCHEMA_VERSION,
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


def _state_with_skipped(jobs: list[JobEntry], skipped: list[SkippedEntry]) -> DispatchState:
    return DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id="20260427-100000",
        started_at="2026-04-27T10:00:00Z",
        ended_at=None,
        seeds=[42],
        commit_sha="abc123",
        fleet=[FleetSnapshot(host="h1", status="idle", current_run_id=None, last_error=None)],
        jobs=jobs,
        skipped=skipped,
    )


def test_roundtrip_skipped_array(tmp_path: Path):
    skipped = [
        SkippedEntry(
            task_id="Isaac-Velocity-Flat-Anymal-C-Direct-v0",
            framework="rsl_rl",
            backend="physx",
            seed=42,
            reason="preset_unsupported",
            presets_available=[],
        ),
        SkippedEntry(
            task_id="Isaac-NewtonOnly-v0",
            framework="rsl_rl",
            backend="physx",
            seed=43,
            reason="preset_unsupported",
            presets_available=["newton"],
        ),
    ]
    write_dispatch_state(tmp_path, _state_with_skipped([_job("run-a")], skipped))
    reloaded = read_dispatch_state(tmp_path)
    assert reloaded.schema_version == SCHEMA_VERSION
    assert len(reloaded.skipped) == 2
    s0 = reloaded.skipped[0]
    assert s0.task_id == "Isaac-Velocity-Flat-Anymal-C-Direct-v0"
    assert s0.reason == "preset_unsupported"
    assert s0.presets_available == []
    assert reloaded.skipped[1].presets_available == ["newton"]


def test_read_v1_0_dispatch_json_loads_with_empty_skipped(tmp_path: Path):
    """Reading a 1.0 file with no skipped key returns DispatchState.skipped == []."""
    path = tmp_path / "dispatch.json"
    path.write_text(
        '{"schema_version": "1.0", "dispatch_id": "old", '
        '"started_at": "2026-01-01T00:00:00Z", "ended_at": null, '
        '"seeds": [42], "commit_sha": "", "fleet": [], "jobs": []}'
    )
    s = read_dispatch_state(tmp_path)
    assert s is not None
    assert s.schema_version == "1.0"
    assert s.skipped == []


def test_read_rejects_major_version_2(tmp_path: Path):
    """Major-version mismatch is still a hard error."""
    path = tmp_path / "dispatch.json"
    path.write_text(
        '{"schema_version": "2.0", "dispatch_id": "future", '
        '"started_at": "x", "ended_at": null, "seeds": [], "commit_sha": "", '
        '"fleet": [], "jobs": []}'
    )
    with pytest.raises(ValueError, match="schema_version"):
        read_dispatch_state(tmp_path)


def test_roundtrip_skipped_with_native_backend(tmp_path: Path):
    """SkippedEntry.native_backend round-trips through dispatch.json."""
    skipped = [
        SkippedEntry(
            task_id="Isaac-Quadcopter-Direct-v0",
            framework="rsl_rl",
            backend="newton",
            seed=42,
            reason="native_backend_mismatch",
            presets_available=[],
            native_backend="physx",
        ),
    ]
    write_dispatch_state(tmp_path, _state_with_skipped([_job("run-a")], skipped))
    reloaded = read_dispatch_state(tmp_path)
    assert reloaded.skipped[0].native_backend == "physx"
    assert reloaded.skipped[0].reason == "native_backend_mismatch"


def test_read_skipped_without_native_backend_defaults_to_none(tmp_path: Path):
    """Skipped entries written by an older writer (no native_backend key) read with native_backend=None."""
    path = tmp_path / "dispatch.json"
    path.write_text(
        '{"schema_version": "1.1", "dispatch_id": "old", '
        '"started_at": "2026-01-01T00:00:00Z", "ended_at": null, '
        '"seeds": [42], "commit_sha": "", "fleet": [], "jobs": [], '
        '"skipped": [{"task_id": "T", "framework": "rsl_rl", "backend": "physx", '
        '"seed": 42, "reason": "preset_unsupported", "presets_available": []}]}'
    )
    s = read_dispatch_state(tmp_path)
    assert s is not None
    assert s.skipped[0].native_backend is None


def test_schema_version_writes_1_3(tmp_path: Path):
    """New dispatches write schema_version='1.3'."""
    write_dispatch_state(tmp_path, _state_with_skipped([_job("run-a")], []))
    import json

    payload = json.loads((tmp_path / "dispatch.json").read_text())
    assert payload["schema_version"] == "1.3"


def test_schema_version_is_1_3():
    """Module-level SCHEMA_VERSION constant is bumped to '1.3'."""
    assert SCHEMA_VERSION == "1.3"


def test_resume_from_1_2_state_works(tmp_path: Path):
    """A 1.2 dispatch.json on disk is readable by the 1.3 reader (major-match)."""
    import json

    payload = {
        "schema_version": "1.2",
        "dispatch_id": "20260424-160119",
        "started_at": "2026-04-24T16:01:19Z",
        "ended_at": None,
        "seeds": [42],
        "commit_sha": "abc123",
        "fleet": [{"host": "v1", "status": "idle", "current_run_id": None, "last_error": None}],
        "jobs": [],
        "skipped": [],
    }
    (tmp_path / "dispatch.json").write_text(json.dumps(payload, indent=2))
    state = read_dispatch_state(tmp_path)
    assert state.dispatch_id == "20260424-160119"
    assert state.schema_version == "1.2"


def test_failure_kind_gpu_lost_round_trips(tmp_path: Path):
    """JobEntry.failure with kind='gpu_lost' survives write→read."""
    j = _job("run-gl", status="failed")
    j.failure = FailureInfo(
        kind="gpu_lost",
        message="GPU-loss signature in stderr",
        details={
            "exit_code": 1,
            "log_tail_path": "run-gl/logs/ssh-tail.log",
        },
    )
    j.attempts = 2
    write_dispatch_state(tmp_path, _state([j]))

    reloaded = read_dispatch_state(tmp_path)
    rj = reloaded.jobs[0]
    assert rj.failure is not None
    assert rj.failure.kind == "gpu_lost"
    assert rj.failure.details["exit_code"] == 1
    assert rj.attempts == 2
