# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the JobEntry-invariant tripwire in state.py."""

from __future__ import annotations

import pytest

from tools.odin.asgard.jobs import FailureInfo, JobEntry
from tools.odin.asgard.state import _validate_job_entry_invariants


def _job(**overrides) -> JobEntry:
    defaults = dict(
        run_id="r1",
        task_id="t",
        framework="rsl_rl",
        backend="physx",
        num_envs=1,
        max_iterations=1,
        seed=42,
        bundle_dir_name="r1",
    )
    defaults.update(overrides)
    return JobEntry(**defaults)


# Strict mode tests.


def test_completed_without_ended_at_raises_in_strict_mode():
    job = _job(status="completed", ended_at=None)
    with pytest.raises(AssertionError, match=r"completed.*ended_at"):
        _validate_job_entry_invariants(job, strict=True)


def test_failed_without_failure_raises_in_strict_mode():
    job = _job(status="failed", ended_at="t0", failure=None)
    with pytest.raises(AssertionError, match=r"failed.*failure"):
        _validate_job_entry_invariants(job, strict=True)


def test_failed_without_ended_at_raises_in_strict_mode():
    job = _job(status="failed", ended_at=None, failure=FailureInfo(kind="x", message="y"))
    with pytest.raises(AssertionError, match=r"failed.*ended_at"):
        _validate_job_entry_invariants(job, strict=True)


def test_running_without_started_at_raises_in_strict_mode():
    job = _job(status="running", started_at=None, assigned_to="v1")
    with pytest.raises(AssertionError, match=r"running.*started_at"):
        _validate_job_entry_invariants(job, strict=True)


def test_running_without_assigned_to_raises_in_strict_mode():
    job = _job(status="running", started_at="t0", assigned_to=None)
    with pytest.raises(AssertionError, match=r"running.*assigned_to"):
        _validate_job_entry_invariants(job, strict=True)


def test_pending_with_assigned_to_raises_in_strict_mode():
    job = _job(status="pending", assigned_to="v1")
    with pytest.raises(AssertionError, match=r"pending.*assigned_to"):
        _validate_job_entry_invariants(job, strict=True)


def test_clean_terminal_passes_strict():
    """Healthy completed/failed jobs — no exception."""
    job = _job(status="completed", started_at="t0", assigned_to="v1", ended_at="t1")
    _validate_job_entry_invariants(job, strict=True)  # no raise

    job2 = _job(
        status="failed", started_at="t0", assigned_to="v1", ended_at="t1", failure=FailureInfo(kind="x", message="y")
    )
    _validate_job_entry_invariants(job2, strict=True)


# Lenient mode: auto-repair.


def test_completed_without_ended_at_auto_repairs_in_lenient_mode(caplog):
    job = _job(status="completed", ended_at=None)
    _validate_job_entry_invariants(job, strict=False)
    assert job.ended_at is not None
    assert any("ended_at" in rec.message for rec in caplog.records)


def test_failed_without_failure_auto_repairs_in_lenient_mode(caplog):
    job = _job(status="failed", ended_at="t0", failure=None)
    _validate_job_entry_invariants(job, strict=False)
    assert job.failure is not None
    assert job.failure.kind == "unknown"


def test_pending_with_assigned_to_auto_repairs_in_lenient_mode(caplog):
    job = _job(status="pending", assigned_to="v1", started_at="t0")
    _validate_job_entry_invariants(job, strict=False)
    assert job.assigned_to is None
    assert job.started_at is None


# Integration tests: write_dispatch_state end-to-end.


def _make_state(**overrides):
    """Return a minimal valid DispatchState, accepting job-list overrides."""
    from tools.odin.asgard.state import DispatchState

    defaults = dict(
        schema_version="1.4",
        dispatch_id="d",
        started_at="t0",
        ended_at=None,
        seeds=[0],
        commit_sha="abc123",
        fleet=[],
        jobs=[],
        skipped=[],
        quarantined_hosts=[],
    )
    defaults.update(overrides)
    return DispatchState(**defaults)


def test_write_dispatch_state_strict_raises_on_inconsistent_job(tmp_path, monkeypatch):
    """End-to-end: a corrupt JobEntry in DispatchState.jobs should make
    write_dispatch_state raise AssertionError when strict mode is on."""
    monkeypatch.setenv("ODIN_DISPATCH_STRICT_INVARIANTS", "true")

    from tools.odin.asgard.state import write_dispatch_state

    bad_job = _job(status="completed", ended_at=None)
    state = _make_state(jobs=[bad_job])
    with pytest.raises(AssertionError, match=r"completed.*ended_at"):
        write_dispatch_state(tmp_path, state)


def test_write_dispatch_state_lenient_auto_repairs(tmp_path, monkeypatch, caplog):
    import json

    monkeypatch.setenv("ODIN_DISPATCH_STRICT_INVARIANTS", "false")

    from tools.odin.asgard.state import write_dispatch_state

    bad_job = _job(status="completed", ended_at=None)
    state = _make_state(jobs=[bad_job])
    write_dispatch_state(tmp_path, state)

    # Repaired in place + persisted to disk.
    assert bad_job.ended_at is not None
    on_disk = json.loads((tmp_path / "dispatch.json").read_text())
    assert on_disk["jobs"][0]["ended_at"] is not None
    assert any("auto-repaired" in rec.message for rec in caplog.records)


def test_write_dispatch_state_strict_default_when_env_unset(tmp_path, monkeypatch):
    """Default behavior: strict mode applies when ODIN_DISPATCH_STRICT_INVARIANTS is unset."""
    monkeypatch.delenv("ODIN_DISPATCH_STRICT_INVARIANTS", raising=False)

    from tools.odin.asgard.state import write_dispatch_state

    bad_job = _job(status="completed", ended_at=None)
    state = _make_state(jobs=[bad_job])
    with pytest.raises(AssertionError):
        write_dispatch_state(tmp_path, state)
