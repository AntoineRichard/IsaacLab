# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runner-side cancellation handling: _consume_cancellations + mark_consumed."""

from __future__ import annotations

from pathlib import Path

from tools.odin.asgard.jobs import FailureInfo, JobEntry
from tools.odin.asgard.runner import _consume_cancellations, _mark_cancellation_consumed
from tools.odin.asgard.worker import StateEvent
from tools.odin.valhalla.dashboard.cancel_db import CancelDB


class _FakeWorker:
    """Minimal stand-in for ValkyrieWorker exposing request_cancel."""

    def __init__(self):
        self.cancel_requests: list[str] = []

    def request_cancel(self, run_id: str) -> None:
        self.cancel_requests.append(run_id)


def _job(run_id: str, status: str = "pending", assigned_to: str | None = None) -> JobEntry:
    return JobEntry(
        run_id=run_id,
        task_id="t",
        framework="rsl_rl",
        backend="physx",
        num_envs=1,
        max_iterations=1,
        seed=42,
        bundle_dir_name=run_id,
        status=status,
        assigned_to=assigned_to,
    )


def test_consume_cancellations_skips_pending_job(tmp_path: Path):
    cancel_db = CancelDB(tmp_path)
    cancel_db.request("d1", "r-skip", kind="skip")
    job = _job("r-skip", status="pending")
    workers_by_host: dict[str, _FakeWorker] = {}

    landed = _consume_cancellations(
        cancel_db=cancel_db,
        dispatch_id="d1",
        jobs_by_id={"r-skip": job},
        workers_by_host=workers_by_host,
    )

    assert landed == 1
    assert job.status == "failed"
    assert job.failure is not None
    assert job.failure.kind == "skipped"
    # Row marked consumed.
    assert cancel_db.read_pending("d1") == {}


def test_consume_cancellations_promotes_skip_to_kill_on_running_job(tmp_path: Path):
    """Skip on a job that already started → promote to kill, signal worker."""
    cancel_db = CancelDB(tmp_path)
    cancel_db.request("d1", "r-running", kind="skip")
    job = _job("r-running", status="running", assigned_to="v1")
    worker = _FakeWorker()
    workers_by_host = {"v1": worker}

    _consume_cancellations(
        cancel_db=cancel_db,
        dispatch_id="d1",
        jobs_by_id={"r-running": job},
        workers_by_host=workers_by_host,
    )

    assert worker.cancel_requests == ["r-running"]
    # Row STILL pending until the worker emits a failed/killed event the
    # runner consumes via _mark_cancellation_consumed.
    assert cancel_db.read_pending("d1") == {"r-running": "kill"}


def test_consume_cancellations_signals_worker_for_kill(tmp_path: Path):
    cancel_db = CancelDB(tmp_path)
    cancel_db.request("d1", "r-kill", kind="kill")
    job = _job("r-kill", status="running", assigned_to="v1")
    worker = _FakeWorker()
    workers_by_host = {"v1": worker}

    _consume_cancellations(
        cancel_db=cancel_db,
        dispatch_id="d1",
        jobs_by_id={"r-kill": job},
        workers_by_host=workers_by_host,
    )

    assert worker.cancel_requests == ["r-kill"]
    # Row stays pending until the worker reports failed/killed back.
    assert cancel_db.read_pending("d1") == {"r-kill": "kill"}


def test_consume_cancellations_marks_noop_on_finished_job(tmp_path: Path):
    """Cancellation arrives after the job already finished → mark_consumed(noop)."""
    cancel_db = CancelDB(tmp_path)
    cancel_db.request("d1", "r-done", kind="kill")
    job = _job("r-done", status="completed")

    _consume_cancellations(
        cancel_db=cancel_db,
        dispatch_id="d1",
        jobs_by_id={"r-done": job},
        workers_by_host={},
    )

    assert cancel_db.read_pending("d1") == {}
    rows = cancel_db.list_for_dispatch("d1")
    assert rows[0].outcome == "noop"


def test_consume_cancellations_returns_added_count_only_for_terminal_skip(tmp_path: Path):
    cancel_db = CancelDB(tmp_path)
    cancel_db.request("d1", "r-skip", kind="skip")
    cancel_db.request("d1", "r-kill", kind="kill")
    job_skip = _job("r-skip", status="pending")
    job_kill = _job("r-kill", status="running", assigned_to="v1")
    worker = _FakeWorker()

    landed = _consume_cancellations(
        cancel_db=cancel_db,
        dispatch_id="d1",
        jobs_by_id={"r-skip": job_skip, "r-kill": job_kill},
        workers_by_host={"v1": worker},
    )

    # Only the skip lands terminal in this call (runner increments 'remaining'
    # only by 1). The kill will land later when the worker emits failed.
    assert landed == 1


def test_mark_cancellation_consumed_on_killed_event(tmp_path: Path):
    """A worker's failed/killed StateEvent triggers mark_consumed(killed)."""
    cancel_db = CancelDB(tmp_path)
    cancel_db.request("d1", "r-kill", kind="kill")

    ev = StateEvent(
        run_id="r-kill",
        host="v1",
        transition="failed",
        failure=FailureInfo(kind="killed", message="operator kill"),
    )

    _mark_cancellation_consumed(cancel_db=cancel_db, dispatch_id="d1", ev=ev)

    rows = cancel_db.list_for_dispatch("d1")
    assert rows[0].outcome == "killed"
    assert rows[0].consumed_at is not None


def test_mark_cancellation_consumed_ignores_unrelated_failed(tmp_path: Path):
    """A failed event for a different kind (gpu_lost, hugin_crash) is unrelated
    to any cancellation row — leave the row alone."""
    cancel_db = CancelDB(tmp_path)
    cancel_db.request("d1", "r-kill", kind="kill")

    ev = StateEvent(
        run_id="r-kill",
        host="v1",
        transition="failed",
        failure=FailureInfo(kind="hugin_crash", message="real crash"),
    )

    _mark_cancellation_consumed(cancel_db=cancel_db, dispatch_id="d1", ev=ev)

    # Row stays pending — operator can still kill the next attempt.
    assert cancel_db.read_pending("d1") == {"r-kill": "kill"}


def test_consume_cancellations_marks_noop_when_worker_missing_for_running_job(tmp_path: Path):
    """Kill on a running job whose host has no live worker (host_down quarantine
    raced ahead of the cancel) → mark noop. Pins the deliberate deviation from
    the spec's 'leave pending' suggestion: the worker is gone, so a sticky
    pending row would never be consumed via the worker's failed/killed event.
    """
    cancel_db = CancelDB(tmp_path)
    cancel_db.request("d1", "r-kill", kind="kill")
    job = _job("r-kill", status="running", assigned_to="v1")  # v1 absent below

    _consume_cancellations(
        cancel_db=cancel_db,
        dispatch_id="d1",
        jobs_by_id={"r-kill": job},
        workers_by_host={},  # worker for "v1" is gone
    )

    rows = cancel_db.list_for_dispatch("d1")
    assert len(rows) == 1
    assert rows[0].outcome == "noop"
