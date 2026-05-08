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
    QuarantinedHost,
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
    j.ended_at = "2026-04-22T22:10:00Z"
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

    state2 = _state(
        [_job("run-a", status="running", started_at="2026-04-22T22:01:00Z", assigned_to="h1"), _job("run-b")]
    )
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


def test_schema_version_writes_1_6(tmp_path: Path):
    """New dispatches write schema_version='1.6'."""
    write_dispatch_state(tmp_path, _state_with_skipped([_job("run-a")], []))
    import json

    payload = json.loads((tmp_path / "dispatch.json").read_text())
    assert payload["schema_version"] == "1.6"


def test_roundtrip_quarantined_hosts(tmp_path: Path):
    """DispatchState with one QuarantinedHost survives write→read."""
    state = DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id="20260428-100000",
        started_at="2026-04-28T10:00:00Z",
        ended_at=None,
        seeds=[42],
        commit_sha="abc",
        fleet=[FleetSnapshot(host="v1", status="down", current_run_id=None, last_error=None)],
        jobs=[],
        quarantined_hosts=[
            QuarantinedHost(
                host="v1",
                reason="circuit_breaker",
                last_run_id="r-cb",
                at="2026-04-28T10:05:00Z",
            )
        ],
    )
    write_dispatch_state(tmp_path, state)
    reloaded = read_dispatch_state(tmp_path)
    assert len(reloaded.quarantined_hosts) == 1
    qh = reloaded.quarantined_hosts[0]
    assert qh.host == "v1"
    assert qh.reason == "circuit_breaker"
    assert qh.last_run_id == "r-cb"
    assert qh.at == "2026-04-28T10:05:00Z"


def test_read_old_state_without_quarantined_hosts_defaults_to_empty(tmp_path: Path):
    """A 1.3 file without a quarantined_hosts key reads as an empty list."""
    import json

    payload = {
        "schema_version": "1.3",
        "dispatch_id": "20260428-110000",
        "started_at": "2026-04-28T11:00:00Z",
        "ended_at": None,
        "seeds": [42],
        "commit_sha": "",
        "fleet": [],
        "jobs": [],
        "skipped": [],
    }
    (tmp_path / "dispatch.json").write_text(json.dumps(payload, indent=2))
    state = read_dispatch_state(tmp_path)
    assert state is not None
    assert state.quarantined_hosts == []


def test_resume_from_1_2_state_works(tmp_path: Path):
    """A 1.2 dispatch.json on disk is readable by the 1.4 reader (major-match)."""
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
    j.ended_at = "2026-04-22T22:10:00Z"
    j.attempts = 2
    write_dispatch_state(tmp_path, _state([j]))

    reloaded = read_dispatch_state(tmp_path)
    rj = reloaded.jobs[0]
    assert rj.failure is not None
    assert rj.failure.kind == "gpu_lost"
    assert rj.failure.details["exit_code"] == 1
    assert rj.attempts == 2


def test_running_substate_round_trips_through_dispatch_json(tmp_path: Path):
    """C1 regression: the running_substate field must survive
    write_dispatch_state → read_dispatch_state. Without this, the
    dashboard's 'pulling bundle' badge breaks across dispatcher
    restarts because the field is stripped on serialize."""
    job = _job("r1", status="running", started_at="2026-04-22T22:00:00Z", assigned_to="v1")
    job.running_substate = "pulling_bundle"
    state = _state([job])
    write_dispatch_state(tmp_path, state)

    reloaded = read_dispatch_state(tmp_path)
    assert reloaded is not None
    assert len(reloaded.jobs) == 1
    assert reloaded.jobs[0].running_substate == "pulling_bundle"


def test_running_substate_training_round_trips(tmp_path: Path):
    """running_substate='training' also round-trips correctly."""
    job = _job("r2", status="running", started_at="2026-04-22T22:00:00Z", assigned_to="v1")
    job.running_substate = "training"
    state = _state([job])
    write_dispatch_state(tmp_path, state)

    reloaded = read_dispatch_state(tmp_path)
    assert reloaded is not None
    assert reloaded.jobs[0].running_substate == "training"


def test_running_substate_none_round_trips(tmp_path: Path):
    """running_substate=None round-trips correctly."""
    job = _job("r3", status="running", started_at="2026-04-22T22:00:00Z", assigned_to="v1")
    # Explicitly set to None to ensure it survives
    job.running_substate = None
    state = _state([job])
    write_dispatch_state(tmp_path, state)

    reloaded = read_dispatch_state(tmp_path)
    assert reloaded is not None
    assert reloaded.jobs[0].running_substate is None


def test_running_substate_absent_in_legacy_dispatch_json_reads_as_none(tmp_path: Path):
    """Backward-compatibility: dispatch.json files written by older code
    don't have the running_substate key. read_dispatch_state must
    tolerate the absence and return None."""
    legacy = """{
  "schema_version": "1.4",
  "dispatch_id": "d",
  "started_at": "2026-04-22T22:00:00Z",
  "ended_at": null,
  "seeds": [42],
  "commit_sha": "abc123",
  "fleet": [{"host": "h1", "status": "idle", "current_run_id": null, "last_error": null}],
  "jobs": [{
    "run_id": "r1",
    "task_id": "Isaac-Ant-Direct-v0",
    "framework": "rsl_rl",
    "backend": "physx",
    "num_envs": 4096,
    "max_iterations": 300,
    "seed": 42,
    "bundle_dir_name": "r1",
    "status": "running",
    "started_at": "2026-04-22T22:00:00Z",
    "assigned_to": "v1",
    "ended_at": null,
    "failure": null,
    "preferred_not": [],
    "attempts": 0,
    "per_job_timeout_s": null
  }],
  "skipped": [],
  "quarantined_hosts": []
}"""
    (tmp_path / "dispatch.json").write_text(legacy)

    loaded = read_dispatch_state(tmp_path)
    assert loaded is not None
    assert loaded.jobs[0].running_substate is None


# ---------------------------------------------------------------------------
# v1.5 schema tests (osmo fields)
# ---------------------------------------------------------------------------


def test_schema_version_is_1_6():
    assert SCHEMA_VERSION == "1.6"


def test_dispatch_state_round_trip_with_osmo_fields(tmp_path: Path):
    # Note: status="completed" requires ended_at per the strict-invariants
    # tripwire added in the state-tracking audit; supplying it here keeps
    # the round-trip valid.
    job = JobEntry(
        run_id="rsl-rl_physx_X_seed42",
        task_id="X",
        framework="rsl-rl",
        backend="physx",
        num_envs=4096,
        max_iterations=500,
        seed=42,
        bundle_dir_name="rsl-rl_physx_X_seed42",
        status="completed",
        ended_at="2026-05-05T15:30:00Z",
        osmo_task_name="rsl-rl-physx-x-seed42",
    )
    state = DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id="20260505-150000",
        started_at="2026-05-05T15:00:00Z",
        ended_at=None,
        seeds=[42],
        commit_sha="deadbeef",
        fleet=[],
        jobs=[job],
        dispatcher="osmo",
        osmo_workflow_id="odin-disp-20260505-150000-1",
        parent_dispatch_id=None,
    )
    write_dispatch_state(tmp_path, state)
    loaded = read_dispatch_state(tmp_path)
    assert loaded is not None
    assert loaded.dispatcher == "osmo"
    assert loaded.osmo_workflow_id == "odin-disp-20260505-150000-1"
    assert loaded.parent_dispatch_id is None
    assert loaded.jobs[0].osmo_task_name == "rsl-rl-physx-x-seed42"


def test_dispatch_state_back_compat_loads_v1_4_without_dispatcher(tmp_path: Path):
    """An old dispatch.json with no `dispatcher` field loads with dispatcher='asgard'."""
    import json

    payload = {
        "schema_version": "1.4",
        "dispatch_id": "20260101-000000",
        "started_at": "2026-01-01T00:00:00Z",
        "ended_at": None,
        "seeds": [42],
        "commit_sha": "",
        "fleet": [],
        "jobs": [],
        "skipped": [],
        "quarantined_hosts": [],
    }
    (tmp_path / "dispatch.json").write_text(json.dumps(payload))
    loaded = read_dispatch_state(tmp_path)
    assert loaded is not None
    assert loaded.dispatcher == "asgard"
    assert loaded.osmo_workflow_id is None
    assert loaded.parent_dispatch_id is None


def test_dispatch_state_round_trip_omits_osmo_task_name(tmp_path: Path):
    """A JobEntry without osmo_task_name (default None) round-trips correctly."""
    job = JobEntry(
        run_id="x",
        task_id="X",
        framework="rsl-rl",
        backend="physx",
        num_envs=1,
        max_iterations=1,
        seed=0,
        bundle_dir_name="x",
    )
    state = DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id="20260101-000000",
        started_at="2026-01-01T00:00:00Z",
        ended_at=None,
        seeds=[0],
        commit_sha="",
        fleet=[],
        jobs=[job],
    )
    write_dispatch_state(tmp_path, state)
    loaded = read_dispatch_state(tmp_path)
    assert loaded is not None
    assert loaded.jobs[0].osmo_task_name is None
    assert loaded.dispatcher == "asgard"


def test_job_entry_last_heartbeat_at_round_trip(tmp_path):
    """JobEntry.last_heartbeat_at survives write_dispatch_state → read_dispatch_state."""
    from tools.odin.asgard.jobs import JobEntry
    from tools.odin.asgard.state import (
        SCHEMA_VERSION,
        DispatchState,
        read_dispatch_state,
        write_dispatch_state,
    )

    job = JobEntry(
        run_id="run-1",
        task_id="cartpole",
        framework="rsl_rl",
        backend="physx",
        num_envs=1024,
        max_iterations=200,
        seed=42,
        bundle_dir_name="run-1",
    )
    job.last_heartbeat_at = "2026-05-08T14:32:18Z"

    state = DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id="20260508-143200",
        started_at="2026-05-08T14:32:00Z",
        ended_at=None,
        seeds=[42],
        commit_sha="deadbeef",
        fleet=[],
        jobs=[job],
    )
    write_dispatch_state(tmp_path, state)
    loaded = read_dispatch_state(tmp_path)
    assert loaded is not None
    assert loaded.jobs[0].last_heartbeat_at == "2026-05-08T14:32:18Z"


def test_job_entry_missing_last_heartbeat_at_is_none():
    """Pre-Heimdall dispatch.json (no last_heartbeat_at field) loads as None."""
    from tools.odin.asgard.state import _job_from_dict

    payload = {
        "run_id": "run-1",
        "task_id": "cartpole",
        "framework": "rsl_rl",
        "backend": "physx",
        "num_envs": 1024,
        "max_iterations": 200,
        "seed": 42,
        "bundle_dir_name": "run-1",
        "status": "pending",
        "assigned_to": None,
        "attempts": 0,
        "preferred_not": [],
        "started_at": None,
        "ended_at": None,
        "running_substate": None,
        "per_job_timeout_s": None,
        "osmo_task_name": None,
        "failure": None,
    }
    job = _job_from_dict(payload)
    assert job.last_heartbeat_at is None


def test_dispatch_state_schema_version_is_minor_bump():
    """Heimdall lands as a minor bump (1.5 → 1.6); same-major resume must work."""
    from tools.odin.asgard.state import SCHEMA_VERSION, _schema_version_compatible

    assert SCHEMA_VERSION.split(".", 1)[0] == "1"
    assert _schema_version_compatible("1.5", SCHEMA_VERSION)
    assert not _schema_version_compatible("2.0", SCHEMA_VERSION)
