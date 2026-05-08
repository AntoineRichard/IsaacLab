# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :func:`~tools.odin.asgard.runner._apply_state_event`."""

from __future__ import annotations

from tools.odin.asgard.jobs import FailureInfo, JobEntry
from tools.odin.asgard.runner import _apply_state_event
from tools.odin.asgard.state import SCHEMA_VERSION, DispatchState, FleetSnapshot
from tools.odin.asgard.worker import StateEvent


def _running_job(run_id: str, host: str, *, started_at: str = "2026-05-08T14:00:00Z") -> JobEntry:
    j = JobEntry(
        run_id=run_id,
        task_id="t",
        framework="rsl_rl",
        backend="physx",
        num_envs=1,
        max_iterations=1,
        seed=0,
        bundle_dir_name=run_id,
    )
    j.transition_to("running", assigned_to=host, now=started_at)
    return j


def _state(jobs: list[JobEntry], fleet: list[FleetSnapshot]) -> DispatchState:
    return DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id="d",
        started_at="2026-05-08T14:00:00Z",
        ended_at=None,
        seeds=[0],
        commit_sha="x",
        fleet=fleet,
        jobs=jobs,
    )


def test_apply_state_event_heartbeat_bumps_last_heartbeat_at():
    """Event.heartbeat updates JobEntry.last_heartbeat_at without changing status."""
    job = _running_job("run-1", "host-a")
    state = _state(
        jobs=[job],
        fleet=[FleetSnapshot(host="host-a", status="busy", current_run_id="run-1")],
    )

    ev = StateEvent(
        run_id="run-1",
        host="host-a",
        transition="heartbeat",
        at="2026-05-08T14:00:30Z",
    )
    delta = _apply_state_event(state, ev)

    assert delta == 0
    assert job.status == "running"
    assert job.last_heartbeat_at == "2026-05-08T14:00:30Z"


def test_apply_state_event_heartbeat_for_terminal_job_is_noop():
    """Late heartbeat for a terminal job is silently ignored."""
    job = JobEntry(
        run_id="run-1",
        task_id="t",
        framework="rsl_rl",
        backend="physx",
        num_envs=1,
        max_iterations=1,
        seed=0,
        bundle_dir_name="run-1",
        status="failed",
        started_at="2026-05-08T14:00:00Z",
        ended_at="2026-05-08T14:01:00Z",
        failure=FailureInfo(kind="timeout", message="x"),
    )
    state = _state(jobs=[job], fleet=[])

    ev = StateEvent(
        run_id="run-1",
        host="host-a",
        transition="heartbeat",
        at="2026-05-08T14:02:00Z",
    )
    delta = _apply_state_event(state, ev)

    assert delta == 0
    assert job.last_heartbeat_at is None
    assert job.status == "failed"


def test_apply_state_event_heartbeat_for_unknown_run_id_is_noop():
    """Heartbeat for a run_id not in state.jobs is silently ignored."""
    state = _state(jobs=[], fleet=[])

    ev = StateEvent(
        run_id="ghost",
        host="host-a",
        transition="heartbeat",
        at="2026-05-08T14:02:00Z",
    )
    delta = _apply_state_event(state, ev)

    assert delta == 0
    assert state.jobs == []


def test_apply_state_event_heartbeat_without_at_is_noop():
    """Heartbeat without an ``at`` timestamp does not bump the field."""
    job = _running_job("run-1", "host-a")
    state = _state(
        jobs=[job],
        fleet=[FleetSnapshot(host="host-a", status="busy", current_run_id="run-1")],
    )
    ev = StateEvent(run_id="run-1", host="host-a", transition="heartbeat", at=None)

    _apply_state_event(state, ev)

    assert job.last_heartbeat_at is None
