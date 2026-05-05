# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tools.odin.asgard.jobs import JobEntry
from tools.odin.asgard.state import SCHEMA_VERSION, DispatchState, read_dispatch_state
from tools.odin.bifrost.client import TaskSnapshot, WorkflowSnapshot
from tools.odin.bifrost.poller import (
    TERMINAL_OSMO_STATES,
    classify_terminal_state,
    is_terminal,
    poll_until_terminal,
)


@pytest.mark.parametrize(
    "osmo_state, expected_kind",
    [
        ("FAILED", "hugin_crash"),
        ("FAILED_EXEC_TIMEOUT", "timeout"),
        ("FAILED_BACKEND_ERROR", "infrastructure"),
        ("FAILED_PREEMPTED", "infrastructure"),
        ("FAILED_EVICTED", "infrastructure"),
        ("FAILED_IMAGE_PULL", "infrastructure"),
        ("FAILED_START_ERROR", "infrastructure"),
        ("FAILED_START_TIMEOUT", "infrastructure"),
        ("FAILED_QUEUE_TIMEOUT", "infrastructure"),
        ("FAILED_SERVER_ERROR", "infrastructure"),
        ("FAILED_CANCELED", "infrastructure"),
    ],
)
def test_failure_kind_for_each_failed_state(osmo_state: str, expected_kind: str):
    assert classify_terminal_state(osmo_state) == expected_kind


def test_completed_returns_none():
    """COMPLETED isn't a failure kind — caller decides hugin_malformed_bundle separately."""
    assert classify_terminal_state("COMPLETED") is None


def test_classify_unknown_state_defaults_to_infrastructure():
    assert classify_terminal_state("FAILED_NOVEL_THING") == "infrastructure"


def test_is_terminal_recognizes_completed_and_failed_family():
    assert is_terminal("COMPLETED")
    assert is_terminal("FAILED")
    assert is_terminal("FAILED_BACKEND_ERROR")
    # Forward-compat: unknown FAILED_* must still terminate the poll loop
    assert is_terminal("FAILED_NOVEL_THING")
    assert is_terminal("FAILED_SOME_NEW_REASON")


def test_is_terminal_excludes_in_flight_states():
    for s in ("PENDING", "WAITING", "PROCESSING", "SCHEDULING", "INITIALIZING", "RUNNING", "RESCHEDULED"):
        assert not is_terminal(s), s


def test_terminal_set_complete():
    """Sanity-check the terminal set contains all FAILED_* and COMPLETED."""
    expected_subset = {
        "COMPLETED",
        "FAILED",
        "FAILED_EXEC_TIMEOUT",
        "FAILED_BACKEND_ERROR",
        "FAILED_PREEMPTED",
        "FAILED_EVICTED",
        "FAILED_IMAGE_PULL",
        "FAILED_START_ERROR",
        "FAILED_START_TIMEOUT",
        "FAILED_QUEUE_TIMEOUT",
        "FAILED_SERVER_ERROR",
        "FAILED_CANCELED",
    }
    assert expected_subset <= TERMINAL_OSMO_STATES


def _job(run_id: str, osmo_task_name: str, status: str = "pending") -> JobEntry:
    return JobEntry(
        run_id=run_id,
        task_id="X",
        framework="rsl-rl",
        backend="physx",
        num_envs=4096,
        max_iterations=500,
        seed=42,
        bundle_dir_name=run_id,
        status=status,
        osmo_task_name=osmo_task_name,
    )


def _state(jobs: list[JobEntry]) -> DispatchState:
    return DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id="20260505-150000",
        started_at="2026-05-05T15:00:00Z",
        ended_at=None,
        seeds=[42],
        commit_sha="",
        fleet=[],
        jobs=jobs,
        dispatcher="osmo",
        osmo_workflow_id="my-wf-1",
    )


def test_poll_marks_completed_and_failed(tmp_path: Path):
    jobs = [_job("run1", "task-1"), _job("run2", "task-2")]
    state = _state(jobs)
    client = MagicMock()
    client.status.side_effect = [
        WorkflowSnapshot(
            "my-wf-1",
            "RUNNING",
            [
                TaskSnapshot("task-1", "RUNNING", None),
                TaskSnapshot("task-2", "RUNNING", None),
            ],
        ),
        WorkflowSnapshot(
            "my-wf-1",
            "COMPLETED",
            [
                TaskSnapshot("task-1", "COMPLETED", 0),
                TaskSnapshot("task-2", "FAILED", 137),
            ],
        ),
    ]
    bundle_calls: list[str] = []

    def on_completed(job: JobEntry) -> None:
        bundle_calls.append(job.run_id)

    poll_until_terminal(
        client=client,
        state=state,
        dispatch_dir=tmp_path,
        on_task_completed=on_completed,
        poll_interval_s=0,
    )
    assert state.jobs[0].status == "completed"
    assert state.jobs[1].status == "failed"
    assert state.jobs[1].failure is not None and state.jobs[1].failure.kind == "hugin_crash"
    assert bundle_calls == ["run1"]
    # dispatch.json was rewritten at least once
    loaded = read_dispatch_state(tmp_path)
    assert loaded is not None
    assert loaded.jobs[0].status == "completed"


def test_poll_handles_unknown_failed_state_as_infrastructure(tmp_path: Path):
    jobs = [_job("run1", "task-1")]
    state = _state(jobs)
    client = MagicMock()
    client.status.return_value = WorkflowSnapshot("my-wf-1", "FAILED", [TaskSnapshot("task-1", "FAILED_NOVEL", 9999)])
    poll_until_terminal(
        client=client,
        state=state,
        dispatch_dir=tmp_path,
        on_task_completed=lambda j: None,
        poll_interval_s=0,
    )
    assert state.jobs[0].failure.kind == "infrastructure"


def test_poll_skips_unknown_task_names(tmp_path: Path):
    """OSMO returning a task name that's not in our state must not crash."""
    jobs = [_job("run1", "task-1")]
    state = _state(jobs)
    client = MagicMock()
    client.status.return_value = WorkflowSnapshot(
        "my-wf-1",
        "COMPLETED",
        [
            TaskSnapshot("task-1", "COMPLETED", 0),
            TaskSnapshot("task-unknown", "COMPLETED", 0),
        ],
    )
    poll_until_terminal(
        client=client,
        state=state,
        dispatch_dir=tmp_path,
        on_task_completed=lambda j: None,
        poll_interval_s=0,
    )
    assert state.jobs[0].status == "completed"
