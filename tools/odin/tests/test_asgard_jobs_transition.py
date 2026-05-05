# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the JobEntry allowed-transition graph (transition_to tests added in Task 2)."""

from __future__ import annotations

import pytest

from tools.odin.asgard.jobs import FailureInfo, JobEntry


def _job(status: str = "pending", **overrides) -> JobEntry:
    """Build a minimal JobEntry for transition tests. All required fields populated with stubs."""
    defaults = dict(
        run_id="test-run",
        task_id="Isaac-Test-v0",
        framework="rsl_rl",
        backend="physx",
        num_envs=1,
        max_iterations=1,
        seed=42,
        bundle_dir_name="test-run",
        status=status,
    )
    defaults.update(overrides)
    return JobEntry(**defaults)


def test_allowed_transitions_graph_has_eight_legal_edges():
    """The graph encodes seven edges from spec §4.1 plus one back-compat 'assigned'→'pending' edge — eight total."""
    expected = {
        ("assigned", "pending"),
        ("pending", "running"),
        ("pending", "failed"),
        ("running", "completed"),
        ("running", "failed"),
        ("running", "pending"),
        ("failed", "pending"),
        ("completed", "pending"),
    }
    actual = {(src, dst) for src, dsts in JobEntry._ALLOWED_TRANSITIONS.items() for dst in dsts}
    assert actual == expected


def test_pending_to_running_sets_started_at_and_assigned_to():
    job = _job(status="pending")
    assert job.transition_to("running", assigned_to="v1", now="2026-05-05T12:00:00Z") is True
    assert job.status == "running"
    assert job.started_at == "2026-05-05T12:00:00Z"
    assert job.assigned_to == "v1"
    assert job.running_substate == "training"
    assert job.ended_at is None
    assert job.failure is None


def test_running_to_completed_stamps_ended_at_clears_substate():
    job = _job(
        status="running",
        started_at="2026-05-05T12:00:00Z",
        assigned_to="v1",
        running_substate="pulling_bundle",
    )
    assert job.transition_to("completed", now="2026-05-05T12:30:00Z") is True
    assert job.status == "completed"
    assert job.ended_at == "2026-05-05T12:30:00Z"
    assert job.running_substate is None
    assert job.failure is None


def test_running_to_failed_stores_failure_on_job():
    job = _job(status="running", started_at="t0", assigned_to="v1")
    failure = FailureInfo(kind="hugin_crash", message="boom")
    assert job.transition_to("failed", failure=failure, now="t1") is True
    assert job.status == "failed"
    assert job.ended_at == "t1"
    assert job.failure is failure
    assert job.running_substate is None


def test_running_to_pending_clears_runtime_fields_preserves_attempts():
    job = _job(
        status="running",
        started_at="t0",
        assigned_to="v1",
        attempts=2,
        running_substate="training",
    )
    assert job.transition_to("pending") is True
    assert job.status == "pending"
    assert job.started_at is None
    assert job.assigned_to is None
    assert job.ended_at is None
    assert job.failure is None
    assert job.running_substate is None
    assert job.attempts == 2  # NOT reset by default


def test_running_to_pending_with_reset_attempts_zeros_counter():
    job = _job(status="running", started_at="t0", assigned_to="v1", attempts=4)
    job.transition_to("pending", reset_attempts=True)
    assert job.attempts == 0


def test_running_to_pending_with_add_preferred_not_appends_host():
    job = _job(status="running", started_at="t0", assigned_to="v1")
    job.preferred_not = {"v3"}
    job.transition_to("pending", add_preferred_not="v1")
    assert job.preferred_not == {"v1", "v3"}


def test_failed_to_pending_clears_failure():
    job = _job(status="failed", failure=FailureInfo(kind="x", message="y"), ended_at="t0")
    assert job.transition_to("pending") is True
    assert job.status == "pending"
    assert job.failure is None
    assert job.ended_at is None


def test_completed_to_pending_clears_terminal_fields():
    """Live-retry edge: operator may re-run an already-completed seed."""
    job = _job(status="completed", ended_at="t0")
    assert job.transition_to("pending") is True
    assert job.status == "pending"
    assert job.ended_at is None


def test_pending_to_failed_skip_path_requires_failure():
    job = _job(status="pending")
    failure = FailureInfo(kind="newton_floor", message="no host meets cuda floor")
    job.transition_to("failed", failure=failure, now="t0")
    assert job.status == "failed"
    assert job.failure is failure
    assert job.ended_at == "t0"


def test_self_loop_is_noop_returns_false():
    """Calling transition_to(current_state) returns False and mutates nothing."""
    job = _job(status="running", started_at="t0", assigned_to="v1", running_substate="training")
    snapshot = (job.status, job.started_at, job.assigned_to, job.running_substate)
    assert job.transition_to("running", assigned_to="v2") is False
    assert (job.status, job.started_at, job.assigned_to, job.running_substate) == snapshot


def test_now_defaults_to_utc_iso_when_none():
    """Passing now=None on a transition that needs a timestamp uses _utc_now_iso()."""
    import re

    job = _job(status="pending")
    job.transition_to("running", assigned_to="v1")  # now=None
    # _utc_now_iso() format: YYYY-MM-DDTHH:MM:SSZ
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", job.started_at) is not None


@pytest.mark.parametrize(
    "src,dst",
    [
        ("completed", "running"),
        ("completed", "failed"),
        ("failed", "running"),
        ("failed", "completed"),
        ("pending", "completed"),  # must go through running
    ],
)
def test_illegal_edges_raise_value_error(src, dst):
    job = _job(status=src)
    with pytest.raises(ValueError, match=f"illegal transition {src!r} → {dst!r}"):
        job.transition_to(dst, failure=FailureInfo(kind="x", message="y"))


def test_running_to_failed_without_failure_raises():
    job = _job(status="running", started_at="t0", assigned_to="v1")
    with pytest.raises(ValueError, match="requires failure"):
        job.transition_to("failed")


def test_pending_to_running_without_assigned_to_raises():
    job = _job(status="pending")
    with pytest.raises(ValueError, match="requires assigned_to"):
        job.transition_to("running")


def test_running_to_completed_with_failure_raises():
    """The 'completed' contract forbids passing failure. Catches legacy callers
    that thought they could stamp failure on completion."""
    job = _job(status="running", started_at="t0", assigned_to="v1")
    with pytest.raises(ValueError, match="must not pass failure"):
        job.transition_to("completed", failure=FailureInfo(kind="x", message="y"))
