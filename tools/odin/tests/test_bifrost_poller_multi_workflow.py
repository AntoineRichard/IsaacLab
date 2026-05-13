# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-workflow poller behavior (spec §6).

The poller walks every entry in :attr:`DispatchState.osmo_workflow_ids`
each tick so a dispatch split across N OSMO workflows aggregates
correctly. These tests exercise that loop with a recording fake client.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.odin.asgard.jobs import JobEntry
from tools.odin.asgard.state import SCHEMA_VERSION, DispatchState
from tools.odin.bifrost.client import TaskSnapshot, WorkflowSnapshot
from tools.odin.bifrost.poller import poll_until_terminal


def _job(run_id: str, osmo_task_name: str) -> JobEntry:
    return JobEntry(
        run_id=run_id,
        task_id="X",
        framework="rsl-rl",
        backend="physx",
        num_envs=4096,
        max_iterations=500,
        seed=42,
        bundle_dir_name=run_id,
        status="pending",
        osmo_task_name=osmo_task_name,
    )


def _state(jobs: list[JobEntry], *, workflow_ids: list[str]) -> DispatchState:
    return DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id="20260513-100000",
        started_at="2026-05-13T10:00:00Z",
        ended_at=None,
        seeds=[42],
        commit_sha="",
        fleet=[],
        jobs=jobs,
        dispatcher="osmo",
        osmo_workflow_ids=workflow_ids,
    )


class _RecordingPoller:
    """Fake ``status`` source backed by a per-workflow snapshot script.

    ``scripts[wf_id]`` is a list of :class:`WorkflowSnapshot`; each call
    to ``status(wf_id)`` pops the next one. When the script is empty,
    the last snapshot is returned indefinitely.
    """

    def __init__(self, scripts: dict[str, list[WorkflowSnapshot]]):
        self._scripts = {k: list(v) for k, v in scripts.items()}
        self.last_seen: dict[str, WorkflowSnapshot] = {}
        self.calls: list[str] = []

    def status(self, wf_id: str) -> WorkflowSnapshot:
        self.calls.append(wf_id)
        if self._scripts.get(wf_id):
            self.last_seen[wf_id] = self._scripts[wf_id].pop(0)
        return self.last_seen[wf_id]


def test_poll_aggregates_two_workflows_with_three_tasks_each(tmp_path: Path):
    """Tick 1: all RUNNING on both workflows. Tick 2: all COMPLETED.

    After tick 2 the loop exits and every job is ``completed``.
    """
    jobs = [_job(f"r-{i}", f"t-{i}") for i in range(6)]
    state = _state(jobs, workflow_ids=["wf-a", "wf-b"])
    # First three tasks live on wf-a, last three on wf-b.
    wf_a_tasks_running = [TaskSnapshot(f"t-{i}", "RUNNING", None) for i in range(3)]
    wf_b_tasks_running = [TaskSnapshot(f"t-{i}", "RUNNING", None) for i in range(3, 6)]
    wf_a_tasks_done = [TaskSnapshot(f"t-{i}", "COMPLETED", 0) for i in range(3)]
    wf_b_tasks_done = [TaskSnapshot(f"t-{i}", "COMPLETED", 0) for i in range(3, 6)]
    scripts = {
        "wf-a": [
            WorkflowSnapshot("wf-a", "RUNNING", wf_a_tasks_running),
            WorkflowSnapshot("wf-a", "COMPLETED", wf_a_tasks_done),
        ],
        "wf-b": [
            WorkflowSnapshot("wf-b", "RUNNING", wf_b_tasks_running),
            WorkflowSnapshot("wf-b", "COMPLETED", wf_b_tasks_done),
        ],
    }
    client = _RecordingPoller(scripts)
    completed: list[str] = []

    poll_until_terminal(
        client=client,
        state=state,
        dispatch_dir=tmp_path,
        on_task_completed=lambda j: completed.append(j.run_id),
        poll_interval_s=0,
    )
    assert all(j.status == "completed" for j in state.jobs)
    assert sorted(completed) == [f"r-{i}" for i in range(6)]
    # Each tick must query both workflows.
    # Tick 1: both queried (RUNNING transitions). Tick 2: both queried
    # (COMPLETED transitions). After tick 2 the loop exits.
    by_id = {wf: client.calls.count(wf) for wf in ("wf-a", "wf-b")}
    assert by_id == {"wf-a": 2, "wf-b": 2}


def test_poll_continues_while_one_workflow_still_running(tmp_path: Path):
    """One workflow done, the other still RUNNING → loop keeps polling.

    Tick 1: wf-a RUNNING, wf-b COMPLETED.
    Tick 2: wf-a COMPLETED, wf-b COMPLETED (idempotent).
    Loop must exit only after tick 2.
    """
    jobs = [_job("r-0", "t-0"), _job("r-1", "t-1")]
    state = _state(jobs, workflow_ids=["wf-a", "wf-b"])
    scripts = {
        "wf-a": [
            WorkflowSnapshot("wf-a", "RUNNING", [TaskSnapshot("t-0", "RUNNING", None)]),
            WorkflowSnapshot("wf-a", "COMPLETED", [TaskSnapshot("t-0", "COMPLETED", 0)]),
        ],
        "wf-b": [
            WorkflowSnapshot("wf-b", "COMPLETED", [TaskSnapshot("t-1", "COMPLETED", 0)]),
        ],
    }
    client = _RecordingPoller(scripts)
    poll_until_terminal(
        client=client,
        state=state,
        dispatch_dir=tmp_path,
        on_task_completed=lambda j: None,
        poll_interval_s=0,
    )
    assert state.jobs[0].status == "completed"
    assert state.jobs[1].status == "completed"
    # wf-a queried twice (RUNNING → COMPLETED), wf-b twice (COMPLETED on
    # both ticks — the second is a no-op because the job already
    # transitioned, but the loop still issues the query).
    assert client.calls.count("wf-a") == 2
    assert client.calls.count("wf-b") == 2


def test_empty_workflow_ids_with_no_legacy_id_raises(tmp_path: Path):
    """Defensive: poller refuses to spin on an empty ids list with no jobs to track."""
    state = _state([_job("r-0", "t-0")], workflow_ids=[])
    state.osmo_workflow_id = None
    with pytest.raises(ValueError, match="osmo_workflow_ids"):
        poll_until_terminal(
            client=_RecordingPoller({}),
            state=state,
            dispatch_dir=tmp_path,
            on_task_completed=lambda j: None,
            poll_interval_s=0,
        )


def test_legacy_single_workflow_id_still_works(tmp_path: Path):
    """Backward compat: an old state with only ``osmo_workflow_id`` polls fine.

    The migration in ``state._load_osmo_workflow_ids`` promotes the
    single id into the list at load time, but the poller's own fallback
    handles the in-memory case where a caller constructs a state with
    just the single field.
    """
    jobs = [_job("r-0", "t-0")]
    state = _state(jobs, workflow_ids=[])
    state.osmo_workflow_id = "legacy-wf"
    scripts = {"legacy-wf": [WorkflowSnapshot("legacy-wf", "COMPLETED", [TaskSnapshot("t-0", "COMPLETED", 0)])]}
    client = _RecordingPoller(scripts)
    poll_until_terminal(
        client=client,
        state=state,
        dispatch_dir=tmp_path,
        on_task_completed=lambda j: None,
        poll_interval_s=0,
    )
    assert state.jobs[0].status == "completed"
    assert client.calls == ["legacy-wf"]
