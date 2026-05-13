# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Poll an OSMO workflow and classify per-task terminal states.

Single source of truth for the OSMO-state → Odin-failure-kind mapping
lives in :data:`OSMO_STATE_TO_FAILURE_KIND` (spec §7).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from tools.odin.asgard.jobs import FailureInfo, JobEntry
from tools.odin.asgard.state import DispatchState, write_dispatch_state
from tools.odin.bifrost.client import OsmoTransientError, WorkflowSnapshot

__all__ = [
    "OSMO_STATE_TO_FAILURE_KIND",
    "TERMINAL_OSMO_STATES",
    "classify_terminal_state",
    "is_terminal",
    "poll_until_terminal",
]


# Maps OSMO terminal task states to Odin failure kinds. Per spec §7.
# COMPLETED is intentionally absent: callers decide
# hugin_malformed_bundle vs success after manifest validation.
OSMO_STATE_TO_FAILURE_KIND: dict[str, str] = {
    "FAILED": "hugin_crash",
    "FAILED_EXEC_TIMEOUT": "timeout",
    "FAILED_BACKEND_ERROR": "infrastructure",
    "FAILED_PREEMPTED": "infrastructure",
    "FAILED_EVICTED": "infrastructure",
    "FAILED_IMAGE_PULL": "infrastructure",
    "FAILED_START_ERROR": "infrastructure",
    "FAILED_START_TIMEOUT": "infrastructure",
    "FAILED_QUEUE_TIMEOUT": "infrastructure",
    "FAILED_SERVER_ERROR": "infrastructure",
    "FAILED_CANCELED": "infrastructure",
}

TERMINAL_OSMO_STATES: frozenset[str] = frozenset({"COMPLETED", *OSMO_STATE_TO_FAILURE_KIND.keys()})


def is_terminal(osmo_state: str) -> bool:
    """Return True iff ``osmo_state`` is one of the known terminal task states.

    Accepts any string starting with ``FAILED`` so OSMO version drift
    (a new ``FAILED_*`` variant we haven't enumerated yet) is still
    treated as terminal — matching the safety-net behavior of
    :func:`classify_terminal_state`.
    """
    if osmo_state in TERMINAL_OSMO_STATES:
        return True
    return osmo_state.startswith("FAILED")


def classify_terminal_state(osmo_state: str) -> str | None:
    """Return the Odin failure kind for an OSMO terminal state, or ``None`` for COMPLETED.

    Unknown ``FAILED_*`` states default to ``"infrastructure"`` to keep behavior
    safe under OSMO version drift.

    Args:
        osmo_state: OSMO task state string.

    Returns:
        Odin failure kind (``hugin_crash`` | ``timeout`` | ``infrastructure``)
        for failed states, ``None`` for ``COMPLETED``, ``None`` for non-terminal
        states.
    """
    if osmo_state == "COMPLETED":
        return None
    if osmo_state in OSMO_STATE_TO_FAILURE_KIND:
        return OSMO_STATE_TO_FAILURE_KIND[osmo_state]
    if osmo_state.startswith("FAILED"):
        return "infrastructure"
    return None


class _StatusClient(Protocol):
    def status(self, workflow_id: str) -> WorkflowSnapshot: ...


def _osmo_status_to_job_status(osmo_state: str) -> str:
    """Map an OSMO task state to the Odin job status string used in dispatch.json."""
    if osmo_state == "COMPLETED":
        return "completed"
    if osmo_state == "RUNNING":
        return "running"
    if is_terminal(osmo_state):
        return "failed"
    # SUBMITTING, WAITING, PROCESSING, SCHEDULING, INITIALIZING, RESCHEDULED.
    return "pending"


def poll_until_terminal(
    *,
    client: _StatusClient,
    state: DispatchState,
    dispatch_dir: Path,
    on_task_completed: Callable[[JobEntry], None],
    poll_interval_s: float,
) -> None:
    """Drive an OSMO workflow to completion, writing dispatch.json atomically.

    Args:
        client: Has ``status(workflow_id) -> WorkflowSnapshot``.
        state: Mutated in place. Must have ``osmo_workflow_id`` set.
        dispatch_dir: Where dispatch.json lives.
        on_task_completed: Called once per task that transitions to COMPLETED.
            Implementations typically enqueue a bundle download.
        poll_interval_s: Seconds between status calls. Set to 0 in tests.

    Returns when every job is in a terminal state.
    """
    # Per spec §6 the poller walks every workflow id each tick so a
    # dispatch split across N OSMO workflows aggregates correctly. The
    # legacy single-id field is honored for back-compat when the new
    # list field is empty.
    workflow_ids: list[str] = list(state.osmo_workflow_ids)
    if not workflow_ids:
        if state.osmo_workflow_id is None:
            raise ValueError("state.osmo_workflow_ids / osmo_workflow_id is required for OSMO polling")
        workflow_ids = [state.osmo_workflow_id]
    by_osmo_name = {j.osmo_task_name: j for j in state.jobs if j.osmo_task_name}
    completed_seen: set[str] = set()
    while not _all_terminal(state):
        for wf_id in workflow_ids:
            try:
                snap = client.status(wf_id)
            except Exception as exc:
                # OSMO occasionally returns 4xx/5xx mid-poll (transient
                # backend hiccup, race against a workflow finishing,
                # CLI parse glitch). None of these should kill the
                # whole poller: skip this workflow's tick, retry next
                # poll. Auth errors will keep failing each tick but
                # that's the operator's signal to fix credentials.
                print(
                    f"[bifrost] osmo workflow query {wf_id} failed; will retry next tick: {exc}",
                    flush=True,
                )
                continue
            for task in snap.tasks:
                job = by_osmo_name.get(task.name)
                if job is None:
                    continue  # Unknown task — log via state's general mechanism in caller.
                new_status = _osmo_status_to_job_status(task.status)
                if new_status == job.status:
                    continue
                # Route through transition_to so the per-target field
                # invariants (started_at on running, ended_at on terminal,
                # FailureInfo on failed) are enforced atomically — same
                # contract every other dispatcher path uses, and required
                # for write_dispatch_state's invariant tripwire to pass.
                if new_status == "running":
                    # OSMO doesn't expose an assigned host; identify the
                    # specific chunk-workflow as the abstract assignee.
                    job.transition_to(
                        "running",
                        assigned_to=f"osmo:{wf_id}",
                    )
                elif new_status == "completed":
                    job.transition_to("completed")
                    if job.run_id not in completed_seen:
                        completed_seen.add(job.run_id)
                        on_task_completed(job)
                elif new_status == "failed":
                    kind = classify_terminal_state(task.status) or "infrastructure"
                    failure = FailureInfo(
                        kind=kind,
                        message=f"OSMO task {task.name} terminal state {task.status} (exit={task.exit_code})",
                        details={"osmo_state": task.status, "exit_code": task.exit_code},
                    )
                    job.transition_to("failed", failure=failure)
                else:  # back to pending (rare — OSMO requeue)
                    job.transition_to("pending")
        write_dispatch_state(dispatch_dir, state)
        if _all_terminal(state):
            break
        if poll_interval_s > 0:
            time.sleep(poll_interval_s)


def _all_terminal(state: DispatchState) -> bool:
    return all(j.status in ("completed", "failed") for j in state.jobs)
