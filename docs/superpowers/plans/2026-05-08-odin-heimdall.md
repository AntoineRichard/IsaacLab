# Heimdall Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Heimdall — an in-process daemon thread inside the Asgard dispatcher — that periodically re-probes the fleet, watches for stale jobs (via worker-thread heartbeats), and triggers `recover_valkyrie_gpu` + quarantine + re-queue through the existing state-event path.

**Architecture:** A `HeimdallWatcher` thread parallel-probes hosts every ~5 min and computes stale jobs from in-memory `DispatchState`, publishing a `HeimdallSnapshot` under a lock. The main `run_dispatch` loop consumes snapshots once per dispatch tick (alongside `_consume_live_retries` / `_consume_cancellations`), applies actions through `_apply_state_event`, and the watcher persists per-host health and recent activity to `<dispatch_dir>/fleet.json` for the Valhalla dashboard. Worker threads emit periodic `heartbeat` state events for each in-flight job; missing heartbeats are the stale-job signal.

**Tech Stack:** Python 3.10+ (`dataclass`, `threading.Thread`, `threading.Event`, `concurrent.futures.ThreadPoolExecutor`). Existing Odin types: `JobEntry`, `DispatchState`, `StateEvent`, `ValkyrieWorker`, `SSHRunner`, `recover_valkyrie_gpu`. No new third-party dependencies. Dash / dash-html-components for the new dashboard card.

**Spec:** `docs/superpowers/specs/2026-05-08-odin-heimdall-design.md`. Worktree branch: `worktree-heimdall-design`. The worktree is currently branched from `main`; the implementor should rebase / re-branch off `antoiner/feat/odin` before Task 1 (the Heimdall code touches Odin modules that only exist on that branch).

---

## Pre-flight

- [ ] **Step 0.1: Confirm branch base.** From the worktree:

```bash
cd /home/antoiner/Documents/IsaacLab/.claude/worktrees/heimdall-design
git log --oneline -1
```

If the top commit is not the spec commit on `antoiner/feat/odin`, rebase the worktree onto that branch:

```bash
git fetch origin antoiner/feat/odin
git rebase origin/antoiner/feat/odin
```

Expected: `tools/odin/asgard/state.py` and `tools/odin/asgard/worker.py` are now present in the worktree.

- [ ] **Step 0.2: Confirm test runner works.**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_state.py -q
```

Expected: passes (any number of tests). If the test directory is missing or the command errors, surface that to the user before continuing.

---

## Task 1: Add `last_heartbeat_at` to JobEntry + schema bump

**Files:**
- Modify: `tools/odin/asgard/jobs.py:86-118` (JobEntry dataclass)
- Modify: `tools/odin/asgard/state.py:38` (SCHEMA_VERSION), `:152-180` (`_job_to_dict`), `:183-210` (`_job_from_dict`)
- Test: `tools/odin/tests/asgard/test_state.py` (add new test)

- [ ] **Step 1.1: Write the failing serialization test**

Append to `tools/odin/tests/asgard/test_state.py`:

```python
def test_job_entry_last_heartbeat_at_round_trip(tmp_path):
    """JobEntry.last_heartbeat_at survives write_dispatch_state → read_dispatch_state."""
    from tools.odin.asgard.jobs import JobEntry
    from tools.odin.asgard.state import (
        DispatchState,
        SCHEMA_VERSION,
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
    # 1.5-era dispatch.json must remain readable.
    assert _schema_version_compatible("1.5", SCHEMA_VERSION)
    # 2.x is not.
    assert not _schema_version_compatible("2.0", SCHEMA_VERSION)
```

- [ ] **Step 1.2: Run the new tests to verify they fail**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_state.py::test_job_entry_last_heartbeat_at_round_trip tools/odin/tests/asgard/test_state.py::test_job_entry_missing_last_heartbeat_at_is_none tools/odin/tests/asgard/test_state.py::test_dispatch_state_schema_version_is_minor_bump -v
```

Expected: 3 failures. The first errors with `AttributeError: 'JobEntry' object has no attribute 'last_heartbeat_at'`; the schema test currently asserts `SCHEMA_VERSION == "1.5"` is the major-1 line which already passes — but the rationale requires the bump to land in Step 1.3.

- [ ] **Step 1.3: Add the field + serializer + schema bump**

In `tools/odin/asgard/jobs.py`, append a new field to `JobEntry` immediately after `osmo_task_name`:

```python
    # Last heartbeat timestamp emitted by the worker thread for this job.
    # Set by _apply_state_event on Event.heartbeat (transition="heartbeat").
    # Heimdall reads this to detect stale jobs whose worker thread is wedged
    # (rsync hang, blocking SSH call, etc.). None on pre-Heimdall dispatches
    # and on jobs that have not yet emitted any heartbeat.
    last_heartbeat_at: str | None = None
```

In `tools/odin/asgard/state.py`, bump the schema constant:

```python
SCHEMA_VERSION = "1.6"
```

Add `last_heartbeat_at` to `_job_to_dict` (insert right before the `failure` block):

```python
        "osmo_task_name": j.osmo_task_name,
        "last_heartbeat_at": j.last_heartbeat_at,
```

Add it to `_job_from_dict` in the `JobEntry(...)` constructor call (after `osmo_task_name=...`):

```python
        osmo_task_name=d.get("osmo_task_name"),
        last_heartbeat_at=d.get("last_heartbeat_at"),
```

- [ ] **Step 1.4: Run the new tests + the existing state suite**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_state.py -v
```

Expected: all tests pass. If a pre-existing test references the old schema version literally (`"1.5"`), update only that literal — do not weaken any other assertion.

- [ ] **Step 1.5: Commit**

```bash
git add tools/odin/asgard/jobs.py tools/odin/asgard/state.py tools/odin/tests/asgard/test_state.py
git commit -m "asgard: add JobEntry.last_heartbeat_at + bump schema to 1.6"
```

---

## Task 2: Add `heartbeat` event handling

**Files:**
- Modify: `tools/odin/asgard/worker.py:108-135` (StateEvent dataclass)
- Modify: `tools/odin/asgard/runner.py:187-300` (`_apply_state_event`)
- Test: `tools/odin/tests/asgard/test_runner_apply_state_event.py` (likely exists; add new tests)

- [ ] **Step 2.1: Write the failing event-handler test**

If `test_runner_apply_state_event.py` does not exist, create it with the standard SPDX header. Otherwise append. Add:

```python
def test_apply_state_event_heartbeat_bumps_last_heartbeat_at():
    """Event.heartbeat updates JobEntry.last_heartbeat_at without changing status."""
    from tools.odin.asgard.jobs import JobEntry
    from tools.odin.asgard.runner import _apply_state_event
    from tools.odin.asgard.state import DispatchState, FleetSnapshot, SCHEMA_VERSION
    from tools.odin.asgard.worker import StateEvent

    job = JobEntry(
        run_id="run-1", task_id="t", framework="rsl_rl", backend="physx",
        num_envs=1, max_iterations=1, seed=0, bundle_dir_name="run-1",
        status="running", assigned_to="host-a", started_at="2026-05-08T14:00:00Z",
    )
    state = DispatchState(
        schema_version=SCHEMA_VERSION, dispatch_id="d", started_at="2026-05-08T14:00:00Z",
        ended_at=None, seeds=[0], commit_sha="x",
        fleet=[FleetSnapshot(host="host-a", status="busy", current_run_id="run-1")],
        jobs=[job],
    )

    ev = StateEvent(run_id="run-1", host="host-a", transition="heartbeat",
                    at="2026-05-08T14:00:30Z")
    delta = _apply_state_event(state, ev)

    assert delta == 0                                        # not a remaining-counter event
    assert job.status == "running"                           # status unchanged
    assert job.last_heartbeat_at == "2026-05-08T14:00:30Z"   # field bumped


def test_apply_state_event_heartbeat_for_terminal_job_is_noop():
    """Late heartbeat for a job that already terminated is silently ignored."""
    from tools.odin.asgard.jobs import FailureInfo, JobEntry
    from tools.odin.asgard.runner import _apply_state_event
    from tools.odin.asgard.state import DispatchState, SCHEMA_VERSION
    from tools.odin.asgard.worker import StateEvent

    job = JobEntry(
        run_id="run-1", task_id="t", framework="rsl_rl", backend="physx",
        num_envs=1, max_iterations=1, seed=0, bundle_dir_name="run-1",
        status="failed", started_at="2026-05-08T14:00:00Z",
        ended_at="2026-05-08T14:01:00Z",
        failure=FailureInfo(kind="timeout", message="x"),
    )
    state = DispatchState(
        schema_version=SCHEMA_VERSION, dispatch_id="d", started_at="2026-05-08T14:00:00Z",
        ended_at=None, seeds=[0], commit_sha="x", fleet=[], jobs=[job],
    )

    ev = StateEvent(run_id="run-1", host="host-a", transition="heartbeat",
                    at="2026-05-08T14:02:00Z")
    delta = _apply_state_event(state, ev)

    assert delta == 0
    assert job.last_heartbeat_at is None    # ignored, not bumped
    assert job.status == "failed"
```

- [ ] **Step 2.2: Run the new tests to verify they fail**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_runner_apply_state_event.py::test_apply_state_event_heartbeat_bumps_last_heartbeat_at tools/odin/tests/asgard/test_runner_apply_state_event.py::test_apply_state_event_heartbeat_for_terminal_job_is_noop -v
```

Expected: 2 failures. The first errors because `StateEvent` has no `at` field; the second because `_apply_state_event` has no `heartbeat` branch.

- [ ] **Step 2.3: Add the `at` field on StateEvent**

In `tools/odin/asgard/worker.py`, extend the `StateEvent` dataclass docstring to mention `heartbeat` and add the field:

```python
@dataclass
class StateEvent:
    """Message posted by a worker to the state channel on every transition.

    ``transition`` values:

    - ``running``: job dispatched to host; ``started_at`` set.
    - ``completed``: job finished, bundle pulled, manifest validated.
    - ``failed``: terminal failure for this job; ``failure`` set.
    - ``recovered``: GPU loss detected, recovery succeeded; retry follows.
    - ``host_down``: GPU loss detected, recovery failed; host transitions
      to ``status="down"``. The worker re-queues the in-flight job (so
      another healthy worker can pick it up) and stops pulling further
      jobs from the queue (its own ``_down_event`` is set).
    - ``shutdown_idle``: worker received its sentinel and exited cleanly.
    - ``heartbeat``: liveness ping from the worker thread for an
      in-flight job. ``at`` carries the emit timestamp; the runner
      bumps the matching :attr:`JobEntry.last_heartbeat_at`. Used by
      Heimdall for stale-job detection.
    """

    run_id: str
    host: str
    transition: str
    failure: FailureInfo | None = None
    started_at: str | None = None
    ended_at: str | None = None
    running_substate: str | None = None
    # Generic timestamp used by transitions that do not fit started_at /
    # ended_at semantics. Today only set by ``transition="heartbeat"``.
    at: str | None = None
```

- [ ] **Step 2.4: Add the `heartbeat` branch in `_apply_state_event`**

In `tools/odin/asgard/runner.py`, add this branch immediately before the final `return 0` of `_apply_state_event` (after the `host_down` branch ends):

```python
    if ev.transition == "heartbeat":
        # Heartbeat from the worker thread for an in-flight job. Bump the
        # JobEntry's last_heartbeat_at so Heimdall's staleness check has
        # an up-to-date reference. Silently ignore if the job has already
        # transitioned to a terminal state — late heartbeats can race
        # past terminal events.
        if j is not None and j.status == "running" and ev.at is not None:
            j.last_heartbeat_at = ev.at
        return 0
```

- [ ] **Step 2.5: Run the new tests + existing runner tests**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_runner_apply_state_event.py -v
```

Expected: all tests pass.

- [ ] **Step 2.6: Commit**

```bash
git add tools/odin/asgard/worker.py tools/odin/asgard/runner.py tools/odin/tests/asgard/test_runner_apply_state_event.py
git commit -m "asgard: handle StateEvent(transition='heartbeat') in _apply_state_event"
```

---

## Task 3: Worker heartbeat thread

**Files:**
- Modify: `tools/odin/asgard/worker.py` (`ValkyrieWorker.run` lifecycle, new helper)
- Test: `tools/odin/tests/asgard/test_worker_heartbeat.py` *(new)*

- [ ] **Step 3.1: Write the failing heartbeat-thread test**

Create `tools/odin/tests/asgard/test_worker_heartbeat.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Heartbeat-thread unit tests — no SSH, no real worker, just the helper."""

from __future__ import annotations

import queue
import threading
import time

import pytest

from tools.odin.asgard.worker import StateEvent, _heartbeat_loop


def _drain(q: queue.Queue) -> list[StateEvent]:
    out: list[StateEvent] = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


def test_heartbeat_loop_emits_one_event_per_inflight_run_id_per_tick():
    state_chan: queue.Queue = queue.Queue()
    inflight = {"run-a": object(), "run-b": object()}
    stop = threading.Event()

    t = threading.Thread(
        target=_heartbeat_loop,
        kwargs={
            "host_name": "host-x",
            "state_chan": state_chan,
            "inflight_view": lambda: list(inflight.keys()),
            "interval_s": 0.05,
            "stop_event": stop,
        },
        daemon=True,
    )
    t.start()
    time.sleep(0.16)   # ~3 ticks at 50 ms
    stop.set()
    t.join(timeout=1.0)
    assert not t.is_alive()

    events = _drain(state_chan)
    runs = {e.run_id for e in events}
    assert runs == {"run-a", "run-b"}
    for e in events:
        assert e.host == "host-x"
        assert e.transition == "heartbeat"
        assert e.at is not None


def test_heartbeat_loop_stops_promptly_on_signal():
    state_chan: queue.Queue = queue.Queue()
    stop = threading.Event()

    t = threading.Thread(
        target=_heartbeat_loop,
        kwargs={
            "host_name": "host-x",
            "state_chan": state_chan,
            "inflight_view": lambda: [],
            "interval_s": 5.0,         # long; relies on stop_event.wait
            "stop_event": stop,
        },
        daemon=True,
    )
    t.start()
    time.sleep(0.05)
    stop.set()
    t.join(timeout=1.0)
    assert not t.is_alive()


def test_heartbeat_loop_double_stop_is_noop():
    stop = threading.Event()
    stop.set()
    state_chan: queue.Queue = queue.Queue()
    # Calling _heartbeat_loop with a pre-set stop should return immediately
    # without raising and without emitting anything.
    _heartbeat_loop(
        host_name="host-x",
        state_chan=state_chan,
        inflight_view=lambda: ["run-a"],
        interval_s=1.0,
        stop_event=stop,
    )
    assert state_chan.qsize() == 0
```

- [ ] **Step 3.2: Run the new tests to verify they fail**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_worker_heartbeat.py -v
```

Expected: 3 failures with `ImportError: cannot import name '_heartbeat_loop'`.

- [ ] **Step 3.3: Add `_heartbeat_loop` helper**

Append to `tools/odin/asgard/worker.py` (somewhere after the `StateEvent` dataclass and before `ValkyrieWorker`; module top-level so unit tests can import it):

```python
def _heartbeat_loop(
    *,
    host_name: str,
    state_chan: "queue.Queue[StateEvent]",
    inflight_view: callable,
    interval_s: float,
    stop_event: threading.Event,
) -> None:
    """Emit one ``StateEvent(transition='heartbeat')`` per in-flight run_id, periodically.

    Args:
        host_name: ``ValkyrieConfig.host``; copied into every emitted event.
        state_chan: The same ``queue.Queue`` the owning ``ValkyrieWorker``
            uses for its other state events.
        inflight_view: Callable returning a list of ``run_id`` strings
            currently in flight on the owning worker. Called once per
            tick; the worker's ``_inflight`` dict iteration is not safe
            across mutation, so callers should pass ``lambda:
            list(self._inflight.keys())`` (a snapshot).
        interval_s: Seconds between ticks. Production default is 30 s
            (set by the worker); unit tests pass smaller values.
        stop_event: Set by the worker on shutdown. The loop returns
            on the next wake.
    """
    if stop_event.is_set():
        return
    while not stop_event.is_set():
        run_ids = inflight_view()
        ts = _utc_now_iso()
        for run_id in run_ids:
            state_chan.put(
                StateEvent(
                    run_id=run_id,
                    host=host_name,
                    transition="heartbeat",
                    at=ts,
                )
            )
        if stop_event.wait(timeout=interval_s):
            return
```

- [ ] **Step 3.4: Run the new tests to verify they pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_worker_heartbeat.py -v
```

Expected: 3 passes.

- [ ] **Step 3.5: Wire the heartbeat thread into ValkyrieWorker.run**

Find the `ValkyrieWorker.run` method (around `tools/odin/asgard/worker.py:498` start of class). Locate the body of `run` — it has a try/finally pattern that handles worker lifecycle. Add a heartbeat thread:

1. Add an instance attribute initialization in `ValkyrieWorker.__init__` (or wherever `self._inflight` is initialized):

```python
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_interval_s = 30.0
```

2. In `ValkyrieWorker.run`, immediately after the worker enters its main loop (and before any job processing), start the heartbeat thread:

```python
        self._heartbeat_thread = threading.Thread(
            target=_heartbeat_loop,
            kwargs={
                "host_name": self.host.host,
                "state_chan": self._state_chan,
                "inflight_view": lambda: list(self._inflight.keys()),
                "interval_s": self._heartbeat_interval_s,
                "stop_event": self._heartbeat_stop,
            },
            name=f"heartbeat-{self.host.host}",
            daemon=True,
        )
        self._heartbeat_thread.start()
```

3. In the same method's `finally` block (or whatever cleanup path the worker uses on exit), stop the heartbeat thread:

```python
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)
```

If the worker has multiple exit paths, route them through one cleanup helper rather than duplicating the stop block. The existing pattern in `worker.py` is "single try/finally in `run()`"; add the stop in that finally clause.

- [ ] **Step 3.6: Run the existing worker test suite**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_worker.py -v
```

Expected: all existing worker tests still pass. The heartbeat thread is daemon and idle when `_inflight` is empty, so it must not perturb any prior behavior.

- [ ] **Step 3.7: Commit**

```bash
git add tools/odin/asgard/worker.py tools/odin/tests/asgard/test_worker_heartbeat.py
git commit -m "asgard: emit periodic heartbeat events for in-flight jobs"
```

---

## Task 4: Heimdall dataclasses + module skeleton

**Files:**
- Create: `tools/odin/asgard/heimdall.py`
- Test: `tools/odin/tests/asgard/test_heimdall_types.py` *(new)*

- [ ] **Step 4.1: Write the failing dataclass test**

Create `tools/odin/tests/asgard/test_heimdall_types.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for Heimdall public dataclasses."""

from __future__ import annotations

from tools.odin.asgard.heimdall import HeimdallSnapshot, HostHealth, StaleJob


def test_host_health_default_history_is_empty():
    h = HostHealth(
        name="host-a",
        healthy=True,
        last_probe_at="2026-05-08T14:32:18Z",
        consecutive_failures=0,
        failure_reason=None,
        recovery_attempts=0,
        recovery_history=[],
        quarantined=False,
    )
    assert h.recovery_history == []
    assert h.healthy is True


def test_stale_job_carries_host_health_branch():
    sj = StaleJob(
        run_id="run-1",
        host="host-a",
        last_heartbeat_at="2026-05-08T14:30:00Z",
        age_seconds=240.0,
        host_was_healthy=True,
    )
    assert sj.host_was_healthy is True


def test_heimdall_snapshot_construction():
    snap = HeimdallSnapshot(
        generated_at="2026-05-08T14:32:18Z",
        hosts={},
        stale_jobs=[],
        recent_events=[],
    )
    assert snap.hosts == {}
    assert snap.stale_jobs == []
    assert snap.recent_events == []
```

- [ ] **Step 4.2: Run to verify it fails**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_heimdall_types.py -v
```

Expected: `ModuleNotFoundError: No module named 'tools.odin.asgard.heimdall'`.

- [ ] **Step 4.3: Create the module with dataclasses**

Create `tools/odin/asgard/heimdall.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Heimdall — periodic fleet watcher for the Asgard dispatcher.

Runs as a daemon thread inside ``run_dispatch``. Periodically re-probes
each Valkyrie's GPU presence (``nvidia-smi -L``) and computes stale jobs
from the dispatcher's in-memory :class:`DispatchState`. Publishes a
:class:`HeimdallSnapshot` consumed once per dispatch tick by
:func:`_consume_heimdall_snapshot`. Persists per-host health and recent
activity to ``<dispatch_dir>/fleet.json`` for the Valhalla dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "HostHealth",
    "StaleJob",
    "HeimdallSnapshot",
    "FLEET_JSON_SCHEMA_VERSION",
]


FLEET_JSON_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class HostHealth:
    """Per-host health snapshot, frozen for safe sharing across threads."""

    name: str
    healthy: bool
    last_probe_at: str
    consecutive_failures: int
    failure_reason: str | None
    recovery_attempts: int
    recovery_history: list[str] = field(default_factory=list)
    quarantined: bool = False


@dataclass(frozen=True)
class StaleJob:
    """One in-flight job whose last heartbeat is older than the threshold."""

    run_id: str
    host: str
    last_heartbeat_at: str
    age_seconds: float
    host_was_healthy: bool


@dataclass(frozen=True)
class HeimdallSnapshot:
    """One watcher tick's published view; consumed at most once by the runner."""

    generated_at: str
    hosts: dict[str, HostHealth]
    stale_jobs: list[StaleJob]
    recent_events: list[dict] = field(default_factory=list)
```

- [ ] **Step 4.4: Run tests to verify pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_heimdall_types.py -v
```

Expected: 3 passes.

- [ ] **Step 4.5: Commit**

```bash
git add tools/odin/asgard/heimdall.py tools/odin/tests/asgard/test_heimdall_types.py
git commit -m "asgard: add heimdall.py module skeleton + dataclasses"
```

---

## Task 5: `fleet.json` round-trip

**Files:**
- Modify: `tools/odin/asgard/heimdall.py` (add I/O functions)
- Test: `tools/odin/tests/asgard/test_heimdall_fleet_json.py` *(new)*

- [ ] **Step 5.1: Write the failing round-trip test**

Create `tools/odin/tests/asgard/test_heimdall_fleet_json.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""fleet.json read/write round-trip + atomic-write semantics."""

from __future__ import annotations

import json

from tools.odin.asgard.heimdall import (
    FLEET_JSON_SCHEMA_VERSION,
    HostHealth,
    read_fleet_json,
    write_fleet_json,
)


def test_fleet_json_round_trip(tmp_path):
    hosts = {
        "host-a": HostHealth(
            name="host-a", healthy=True, last_probe_at="2026-05-08T14:32:18Z",
            consecutive_failures=0, failure_reason=None, recovery_attempts=0,
            recovery_history=[], quarantined=False,
        ),
        "host-b": HostHealth(
            name="host-b", healthy=False, last_probe_at="2026-05-08T14:32:18Z",
            consecutive_failures=2, failure_reason="ssh_timeout",
            recovery_attempts=1,
            recovery_history=["2026-05-08T14:30:00Z"],
            quarantined=True,
        ),
    }
    recent_events = [
        {"ts": "2026-05-08T14:31:00Z", "kind": "host_flipped",
         "host": "host-b", "reason": "ssh_timeout"},
    ]

    write_fleet_json(
        tmp_path,
        generated_at="2026-05-08T14:32:18Z",
        hosts=hosts,
        recent_events=recent_events,
    )

    payload = read_fleet_json(tmp_path)
    assert payload is not None
    assert payload["schema_version"] == FLEET_JSON_SCHEMA_VERSION
    assert payload["generated_at"] == "2026-05-08T14:32:18Z"
    assert payload["hosts"]["host-a"]["healthy"] is True
    assert payload["hosts"]["host-b"]["quarantined"] is True
    assert payload["hosts"]["host-b"]["recovery_history"] == ["2026-05-08T14:30:00Z"]
    assert payload["recent_events"] == recent_events


def test_fleet_json_missing_returns_none(tmp_path):
    assert read_fleet_json(tmp_path) is None


def test_fleet_json_atomic_write_no_partial(tmp_path):
    """A failed write must not leave the .tmp behind, nor truncate fleet.json."""
    write_fleet_json(tmp_path, generated_at="2026-05-08T14:32:18Z",
                     hosts={}, recent_events=[])
    initial = (tmp_path / "fleet.json").read_text()

    # Simulate a writer crash by passing a non-serializable hosts payload.
    class NotSerializable: pass
    try:
        write_fleet_json(
            tmp_path,
            generated_at="2026-05-08T14:33:18Z",
            hosts={"host-a": NotSerializable()},   # type: ignore[dict-item]
            recent_events=[],
        )
    except (TypeError, AttributeError):
        pass

    # fleet.json content unchanged.
    assert (tmp_path / "fleet.json").read_text() == initial
    # No leftover .tmp file.
    leftovers = list(tmp_path.glob(".fleet_*.json.tmp"))
    assert leftovers == []
```

- [ ] **Step 5.2: Run to verify it fails**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_heimdall_fleet_json.py -v
```

Expected: 3 ImportError-style failures on `read_fleet_json` / `write_fleet_json`.

- [ ] **Step 5.3: Add the I/O helpers**

Append to `tools/odin/asgard/heimdall.py`:

```python
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any


_FLEET_FILENAME = "fleet.json"


__all__ = list(__all__) + ["read_fleet_json", "write_fleet_json"]


def _host_health_to_dict(h: HostHealth) -> dict[str, Any]:
    d = asdict(h)
    # asdict copies recovery_history as a list — already JSON-safe.
    return d


def write_fleet_json(
    dispatch_dir: Path,
    *,
    generated_at: str,
    hosts: dict[str, HostHealth],
    recent_events: list[dict],
) -> None:
    """Atomically rewrite ``<dispatch_dir>/fleet.json``.

    Writes to a sibling temporary file and ``os.replace``s into place so
    a concurrent reader (the Valhalla dashboard) never observes a
    truncated file. On serialization failure, the .tmp file is unlinked
    and the existing ``fleet.json`` (if any) is left untouched.
    """
    dispatch_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": FLEET_JSON_SCHEMA_VERSION,
        "generated_at": generated_at,
        "hosts": {name: _host_health_to_dict(h) for name, h in hosts.items()},
        "recent_events": list(recent_events),
    }
    fd, tmp_path_str = tempfile.mkstemp(
        prefix=".fleet_", suffix=".json.tmp", dir=str(dispatch_dir)
    )
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=False)
        os.replace(tmp_path, dispatch_dir / _FLEET_FILENAME)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def read_fleet_json(dispatch_dir: Path) -> dict[str, Any] | None:
    """Read ``<dispatch_dir>/fleet.json`` as a raw dict, or ``None`` if absent.

    Returns the raw payload (not :class:`HostHealth` objects) — the dashboard
    consumes the JSON shape directly.
    """
    path = dispatch_dir / _FLEET_FILENAME
    if not path.exists():
        return None
    with path.open("r") as fh:
        return json.load(fh)
```

- [ ] **Step 5.4: Run tests to verify pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_heimdall_fleet_json.py -v
```

Expected: 3 passes.

- [ ] **Step 5.5: Commit**

```bash
git add tools/odin/asgard/heimdall.py tools/odin/tests/asgard/test_heimdall_fleet_json.py
git commit -m "asgard: add fleet.json read/write with atomic semantics"
```

---

## Task 6: HeimdallWatcher — start/stop/latest/is_alive

**Files:**
- Modify: `tools/odin/asgard/heimdall.py` (add `HeimdallWatcher`)
- Test: `tools/odin/tests/asgard/test_heimdall_watcher_lifecycle.py` *(new)*

- [ ] **Step 6.1: Write the failing lifecycle test**

Create `tools/odin/tests/asgard/test_heimdall_watcher_lifecycle.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HeimdallWatcher start/stop/latest/is_alive — no probing yet."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
from tools.odin.asgard.heimdall import HeimdallWatcher
from tools.odin.asgard.transport import SSHResult


@dataclass
class _NeverProbedSSH:
    """SSH runner that records every call but is never expected to be called."""

    calls: list[str]

    def run(self, host, cmd, *, timeout_s=None, stdout_tee=None, pty=True):
        self.calls.append(cmd)
        return SSHResult(exit_code=0, stdout="GPU 0\n", stderr="", duration_s=0.01)


def _empty_state():
    from tools.odin.asgard.state import DispatchState, SCHEMA_VERSION
    return DispatchState(
        schema_version=SCHEMA_VERSION, dispatch_id="d", started_at="2026-05-08T14:00:00Z",
        ended_at=None, seeds=[0], commit_sha="x", fleet=[], jobs=[],
    )


def test_watcher_starts_and_stops_cleanly(tmp_path):
    fleet = Fleet(fleet_name="t", hosts=[ValkyrieConfig(host="host-a", ssh_user="u")])
    ssh = _NeverProbedSSH(calls=[])
    w = HeimdallWatcher(
        fleet=fleet,
        dispatch_dir=tmp_path,
        ssh=ssh,
        state_view=_empty_state,
        probe_interval_s=3600,        # never fires within the test
        stale_threshold_s=180,
    )
    w.start()
    assert w.is_alive()
    w.stop(timeout_s=2.0)
    assert not w.is_alive()


def test_watcher_latest_returns_none_before_first_tick(tmp_path):
    fleet = Fleet(fleet_name="t", hosts=[ValkyrieConfig(host="host-a", ssh_user="u")])
    ssh = _NeverProbedSSH(calls=[])
    w = HeimdallWatcher(
        fleet=fleet, dispatch_dir=tmp_path, ssh=ssh,
        state_view=_empty_state, probe_interval_s=3600, stale_threshold_s=180,
    )
    w.start()
    try:
        assert w.latest() is None
    finally:
        w.stop(timeout_s=2.0)


def test_watcher_double_stop_is_safe(tmp_path):
    fleet = Fleet(fleet_name="t", hosts=[ValkyrieConfig(host="host-a", ssh_user="u")])
    ssh = _NeverProbedSSH(calls=[])
    w = HeimdallWatcher(
        fleet=fleet, dispatch_dir=tmp_path, ssh=ssh,
        state_view=_empty_state, probe_interval_s=3600, stale_threshold_s=180,
    )
    w.start()
    w.stop(timeout_s=2.0)
    w.stop(timeout_s=2.0)    # must be a no-op


def test_watcher_start_twice_raises(tmp_path):
    fleet = Fleet(fleet_name="t", hosts=[ValkyrieConfig(host="host-a", ssh_user="u")])
    ssh = _NeverProbedSSH(calls=[])
    w = HeimdallWatcher(
        fleet=fleet, dispatch_dir=tmp_path, ssh=ssh,
        state_view=_empty_state, probe_interval_s=3600, stale_threshold_s=180,
    )
    w.start()
    try:
        with pytest.raises(RuntimeError):
            w.start()
    finally:
        w.stop(timeout_s=2.0)
```

- [ ] **Step 6.2: Run to verify it fails**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_heimdall_watcher_lifecycle.py -v
```

Expected: 4 ImportError failures.

- [ ] **Step 6.3: Add `HeimdallWatcher`**

Append to `tools/odin/asgard/heimdall.py`:

```python
import logging
import threading
from typing import Callable

from tools.odin.asgard.fleet import Fleet
from tools.odin.asgard.transport import SSHRunner

_log = logging.getLogger(__name__)


__all__ = list(__all__) + ["HeimdallWatcher"]


class HeimdallWatcher:
    """Periodic fleet probe + stale-job watcher.

    The watcher owns its own thread and is the sole writer of
    ``fleet.json``. Consumers (the dispatcher main loop, the Valhalla
    dashboard) call :meth:`latest` to read the most recent
    :class:`HeimdallSnapshot` and never mutate watcher state.

    Thread safety: ``latest()`` and the publishing path are guarded by
    a single ``threading.Lock``. The probing path may run concurrently
    with main-loop consumption.
    """

    def __init__(
        self,
        fleet: Fleet,
        dispatch_dir,
        ssh: SSHRunner,
        state_view: Callable[[], "DispatchState"],   # noqa: F821 - forward
        *,
        probe_interval_s: int = 300,
        stale_threshold_s: int = 180,
        flip_after_k_failures: int = 2,
        probe_timeout_s: int = 15,
        recent_events_max: int = 20,
    ) -> None:
        self._fleet = fleet
        self._dispatch_dir = Path(dispatch_dir)
        self._ssh = ssh
        self._state_view = state_view
        self._probe_interval_s = probe_interval_s
        self._stale_threshold_s = stale_threshold_s
        self._flip_after_k_failures = flip_after_k_failures
        self._probe_timeout_s = probe_timeout_s
        self._recent_events_max = recent_events_max

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: HeimdallSnapshot | None = None
        self._host_state: dict[str, HostHealth] = {}
        self._recent_events: list[dict] = []

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("HeimdallWatcher already started")
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="heimdall-watcher", daemon=True
        )
        self._thread.start()

    def stop(self, timeout_s: float = 10.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
        self._thread = None

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def latest(self) -> HeimdallSnapshot | None:
        with self._lock:
            return self._latest

    def _run(self) -> None:
        # Body added in Task 7. For now, just sleep until stopped so the
        # lifecycle tests pass.
        self._stop_event.wait()
```

Also add the `Path` import at the top of the module if not already present:

```python
from pathlib import Path
```

- [ ] **Step 6.4: Run tests to verify pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_heimdall_watcher_lifecycle.py -v
```

Expected: 4 passes.

- [ ] **Step 6.5: Commit**

```bash
git add tools/odin/asgard/heimdall.py tools/odin/tests/asgard/test_heimdall_watcher_lifecycle.py
git commit -m "asgard: add HeimdallWatcher lifecycle (no probing yet)"
```

---

## Task 7: Probe cycle with K-failure flip gate

**Files:**
- Modify: `tools/odin/asgard/heimdall.py` (replace `_run` body, add probe helpers)
- Test: `tools/odin/tests/asgard/test_heimdall_probe.py` *(new)*

- [ ] **Step 7.1: Write the failing probe-gate test**

Create `tools/odin/tests/asgard/test_heimdall_probe.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Probe cycle + K-consecutive-failure flip gate."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
from tools.odin.asgard.heimdall import HeimdallWatcher, _probe_host
from tools.odin.asgard.state import DispatchState, SCHEMA_VERSION
from tools.odin.asgard.transport import SSHResult


@dataclass
class _ScriptedSSH:
    """Returns scripted SSHResults from a per-host queue."""

    scripts: dict[str, list[SSHResult]]
    calls: list[tuple[str, str]]

    def run(self, host, cmd, *, timeout_s=None, stdout_tee=None, pty=True):
        self.calls.append((host.host, cmd))
        q = self.scripts.get(host.host, [])
        if not q:
            return SSHResult(exit_code=0, stdout="GPU 0\n", stderr="", duration_s=0.01)
        return q.pop(0)


def _empty_state():
    return DispatchState(
        schema_version=SCHEMA_VERSION, dispatch_id="d", started_at="2026-05-08T14:00:00Z",
        ended_at=None, seeds=[0], commit_sha="x", fleet=[], jobs=[],
    )


def test_probe_host_success_returns_healthy():
    host = ValkyrieConfig(host="host-a", ssh_user="u")
    ssh = _ScriptedSSH(
        scripts={"host-a": [SSHResult(exit_code=0, stdout="GPU 0\n",
                                       stderr="", duration_s=0.01)]},
        calls=[],
    )
    healthy, reason = _probe_host(host, ssh=ssh, timeout_s=5)
    assert healthy is True
    assert reason is None
    assert ssh.calls[0][1].startswith("docker exec")


def test_probe_host_ssh_timeout_returns_unhealthy():
    host = ValkyrieConfig(host="host-a", ssh_user="u")
    ssh = _ScriptedSSH(
        scripts={"host-a": [SSHResult(exit_code=255, stdout="",
                                       stderr="", duration_s=15.0,
                                       timed_out=True)]},
        calls=[],
    )
    healthy, reason = _probe_host(host, ssh=ssh, timeout_s=5)
    assert healthy is False
    assert reason == "ssh_timeout"


def test_probe_host_empty_stdout_means_nvml_missing():
    host = ValkyrieConfig(host="host-a", ssh_user="u")
    ssh = _ScriptedSSH(
        scripts={"host-a": [SSHResult(exit_code=0, stdout="",
                                       stderr="", duration_s=0.5)]},
        calls=[],
    )
    healthy, reason = _probe_host(host, ssh=ssh, timeout_s=5)
    assert healthy is False
    assert reason == "nvml_missing"


def test_watcher_k_failure_gate_does_not_flip_on_first_failure(tmp_path):
    """K=2: one failed probe must NOT flip a host to unhealthy."""
    host_a = ValkyrieConfig(host="host-a", ssh_user="u")
    fleet = Fleet(fleet_name="t", hosts=[host_a])
    ssh = _ScriptedSSH(
        scripts={"host-a": [
            SSHResult(exit_code=255, stdout="", stderr="", duration_s=15.0, timed_out=True),
        ]},
        calls=[],
    )
    w = HeimdallWatcher(
        fleet=fleet, dispatch_dir=tmp_path, ssh=ssh, state_view=_empty_state,
        probe_interval_s=10000, stale_threshold_s=180,
        flip_after_k_failures=2, probe_timeout_s=5,
    )
    # Run one probe cycle inline (don't start the thread).
    w._tick_once()
    snap = w.latest()
    assert snap is not None
    assert snap.hosts["host-a"].consecutive_failures == 1
    assert snap.hosts["host-a"].healthy is True   # not flipped yet


def test_watcher_k_failure_gate_flips_on_second_failure(tmp_path):
    host_a = ValkyrieConfig(host="host-a", ssh_user="u")
    fleet = Fleet(fleet_name="t", hosts=[host_a])
    ssh = _ScriptedSSH(
        scripts={"host-a": [
            SSHResult(exit_code=255, stdout="", stderr="", duration_s=15.0, timed_out=True),
            SSHResult(exit_code=255, stdout="", stderr="", duration_s=15.0, timed_out=True),
        ]},
        calls=[],
    )
    w = HeimdallWatcher(
        fleet=fleet, dispatch_dir=tmp_path, ssh=ssh, state_view=_empty_state,
        probe_interval_s=10000, stale_threshold_s=180,
        flip_after_k_failures=2, probe_timeout_s=5,
    )
    w._tick_once()
    w._tick_once()
    snap = w.latest()
    assert snap is not None
    assert snap.hosts["host-a"].consecutive_failures == 2
    assert snap.hosts["host-a"].healthy is False
    assert snap.hosts["host-a"].failure_reason == "ssh_timeout"


def test_watcher_success_resets_consecutive_failures(tmp_path):
    host_a = ValkyrieConfig(host="host-a", ssh_user="u")
    fleet = Fleet(fleet_name="t", hosts=[host_a])
    ssh = _ScriptedSSH(
        scripts={"host-a": [
            SSHResult(exit_code=255, stdout="", stderr="", duration_s=15.0, timed_out=True),
            SSHResult(exit_code=0, stdout="GPU 0\n", stderr="", duration_s=0.01),
        ]},
        calls=[],
    )
    w = HeimdallWatcher(
        fleet=fleet, dispatch_dir=tmp_path, ssh=ssh, state_view=_empty_state,
        probe_interval_s=10000, stale_threshold_s=180,
        flip_after_k_failures=2, probe_timeout_s=5,
    )
    w._tick_once()
    w._tick_once()
    snap = w.latest()
    assert snap.hosts["host-a"].consecutive_failures == 0
    assert snap.hosts["host-a"].healthy is True


def test_watcher_writes_fleet_json_each_tick(tmp_path):
    host_a = ValkyrieConfig(host="host-a", ssh_user="u")
    fleet = Fleet(fleet_name="t", hosts=[host_a])
    ssh = _ScriptedSSH(
        scripts={"host-a": [SSHResult(exit_code=0, stdout="GPU 0\n",
                                       stderr="", duration_s=0.01)]},
        calls=[],
    )
    w = HeimdallWatcher(
        fleet=fleet, dispatch_dir=tmp_path, ssh=ssh, state_view=_empty_state,
        probe_interval_s=10000, stale_threshold_s=180,
        flip_after_k_failures=2, probe_timeout_s=5,
    )
    w._tick_once()
    assert (tmp_path / "fleet.json").exists()
```

- [ ] **Step 7.2: Run to verify it fails**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_heimdall_probe.py -v
```

Expected: 6 ImportError failures on `_probe_host` and `_tick_once`.

- [ ] **Step 7.3: Implement `_probe_host` and `_tick_once`**

In `tools/odin/asgard/heimdall.py`:

Add the imports at the top of the file (alongside existing imports):

```python
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from tools.odin.asgard.fleet import ValkyrieConfig
```

Add `_utc_now_iso` and `_probe_host` as module-level helpers (after the dataclasses, before `HeimdallWatcher`):

```python
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _probe_host(
    host: ValkyrieConfig, *, ssh: SSHRunner, timeout_s: float
) -> tuple[bool, str | None]:
    """Run ``docker exec <ctr> nvidia-smi -L`` and classify the outcome.

    Returns ``(healthy, failure_reason)``. ``failure_reason`` is one of
    ``"ssh_timeout"``, ``"nvml_missing"``, ``"ssh_error"``, or ``None``
    when ``healthy is True``.
    """
    cmd = f"docker exec {host.container_name} nvidia-smi -L"
    r = ssh.run(host, cmd, timeout_s=timeout_s, pty=False)
    if r.timed_out:
        return False, "ssh_timeout"
    if r.exit_code == 255:
        return False, "ssh_error"
    if r.exit_code != 0:
        return False, "nvml_missing"
    if not r.stdout.strip():
        return False, "nvml_missing"
    return True, None
```

Replace the placeholder `_run` body with a real loop, and add `_tick_once` for testability:

```python
    def _tick_once(self) -> HeimdallSnapshot:
        """Run a single probe + stale-job pass synchronously. Test entry point."""
        host_health = self._probe_all_hosts()
        stale = self._compute_stale_jobs(host_health)
        snap = HeimdallSnapshot(
            generated_at=_utc_now_iso(),
            hosts=host_health,
            stale_jobs=stale,
            recent_events=list(self._recent_events),
        )
        with self._lock:
            self._latest = snap
        try:
            write_fleet_json(
                self._dispatch_dir,
                generated_at=snap.generated_at,
                hosts=host_health,
                recent_events=list(self._recent_events),
            )
        except Exception as exc:
            _log.warning("heimdall: fleet.json write failed: %r", exc)
        return snap

    def _probe_all_hosts(self) -> dict[str, HostHealth]:
        results: dict[str, HostHealth] = {}
        with ThreadPoolExecutor(
            max_workers=max(1, len(self._fleet.hosts)),
            thread_name_prefix="heimdall-probe",
        ) as pool:
            futures = {
                pool.submit(_probe_host, h, ssh=self._ssh,
                            timeout_s=self._probe_timeout_s): h
                for h in self._fleet.hosts
            }
            now = _utc_now_iso()
            for fut, host in futures.items():
                try:
                    healthy, reason = fut.result()
                except Exception as exc:    # SSH runner exception path
                    healthy, reason = False, f"probe_exception:{type(exc).__name__}"
                prev = self._host_state.get(host.host)
                cf = (prev.consecutive_failures if prev else 0)
                if healthy:
                    cf = 0
                else:
                    cf += 1
                effective_healthy = cf < self._flip_after_k_failures
                results[host.host] = HostHealth(
                    name=host.host,
                    healthy=effective_healthy,
                    last_probe_at=now,
                    consecutive_failures=cf,
                    failure_reason=None if effective_healthy else reason,
                    recovery_attempts=(prev.recovery_attempts if prev else 0),
                    recovery_history=list(prev.recovery_history) if prev else [],
                    quarantined=(prev.quarantined if prev else False),
                )
        self._host_state = results
        return results

    def _compute_stale_jobs(
        self, host_health: dict[str, HostHealth]
    ) -> list[StaleJob]:
        # Computed in Task 8. For now, return an empty list so probe-only
        # tests pass.
        return []

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._tick_once()
            except Exception as exc:
                _log.exception("heimdall: tick failed: %r", exc)
            if self._stop_event.wait(timeout=self._probe_interval_s):
                return
```

- [ ] **Step 7.4: Run probe tests to verify pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_heimdall_probe.py -v
```

Expected: 6 passes.

- [ ] **Step 7.5: Re-run lifecycle tests** (no regression)

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_heimdall_watcher_lifecycle.py tools/odin/tests/asgard/test_heimdall_fleet_json.py tools/odin/tests/asgard/test_heimdall_types.py -v
```

Expected: all pass.

- [ ] **Step 7.6: Commit**

```bash
git add tools/odin/asgard/heimdall.py tools/odin/tests/asgard/test_heimdall_probe.py
git commit -m "asgard: implement HeimdallWatcher probe cycle with K-failure gate"
```

---

## Task 8: Stale-job computation

**Files:**
- Modify: `tools/odin/asgard/heimdall.py` (`_compute_stale_jobs`)
- Test: `tools/odin/tests/asgard/test_heimdall_stale_jobs.py` *(new)*

- [ ] **Step 8.1: Write the failing stale-job test**

Create `tools/odin/tests/asgard/test_heimdall_stale_jobs.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stale-job computation in HeimdallWatcher._compute_stale_jobs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
from tools.odin.asgard.heimdall import HeimdallWatcher, HostHealth
from tools.odin.asgard.jobs import JobEntry
from tools.odin.asgard.state import DispatchState, SCHEMA_VERSION
from tools.odin.asgard.transport import SSHResult


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _running_job(run_id: str, host: str, *, last_heartbeat_at: str | None,
                 started_at: str) -> JobEntry:
    j = JobEntry(
        run_id=run_id, task_id="t", framework="rsl_rl", backend="physx",
        num_envs=1, max_iterations=1, seed=0, bundle_dir_name=run_id,
    )
    j.transition_to("running", assigned_to=host, now=started_at)
    j.last_heartbeat_at = last_heartbeat_at
    return j


@dataclass
class _OkSSH:
    def run(self, host, cmd, *, timeout_s=None, stdout_tee=None, pty=True):
        return SSHResult(exit_code=0, stdout="GPU 0\n", stderr="", duration_s=0.01)


def _make_watcher(tmp_path, jobs, host_health):
    fleet = Fleet(fleet_name="t", hosts=[ValkyrieConfig(host="host-a", ssh_user="u")])
    state = DispatchState(
        schema_version=SCHEMA_VERSION, dispatch_id="d",
        started_at="2026-05-08T14:00:00Z", ended_at=None,
        seeds=[0], commit_sha="x", fleet=[], jobs=jobs,
    )
    w = HeimdallWatcher(
        fleet=fleet, dispatch_dir=tmp_path, ssh=_OkSSH(),
        state_view=lambda: state, probe_interval_s=10000,
        stale_threshold_s=180, flip_after_k_failures=2, probe_timeout_s=5,
    )
    return w, state, host_health


def test_no_stale_jobs_when_heartbeat_is_fresh(tmp_path):
    now = datetime.now(timezone.utc)
    fresh = _iso(now - timedelta(seconds=30))
    job = _running_job("run-1", "host-a", last_heartbeat_at=fresh, started_at=fresh)
    health = {"host-a": HostHealth(
        name="host-a", healthy=True, last_probe_at=_iso(now),
        consecutive_failures=0, failure_reason=None, recovery_attempts=0,
        recovery_history=[], quarantined=False,
    )}
    w, _, _ = _make_watcher(tmp_path, jobs=[job], host_health=health)
    stale = w._compute_stale_jobs(health)
    assert stale == []


def test_stale_when_heartbeat_older_than_threshold(tmp_path):
    now = datetime.now(timezone.utc)
    old = _iso(now - timedelta(seconds=240))    # 240 > 180
    job = _running_job("run-1", "host-a", last_heartbeat_at=old, started_at=old)
    health = {"host-a": HostHealth(
        name="host-a", healthy=True, last_probe_at=_iso(now),
        consecutive_failures=0, failure_reason=None, recovery_attempts=0,
        recovery_history=[], quarantined=False,
    )}
    w, _, _ = _make_watcher(tmp_path, jobs=[job], host_health=health)
    stale = w._compute_stale_jobs(health)
    assert len(stale) == 1
    assert stale[0].run_id == "run-1"
    assert stale[0].host == "host-a"
    assert stale[0].host_was_healthy is True
    assert stale[0].age_seconds >= 180


def test_stale_with_unhealthy_host_reports_branch(tmp_path):
    now = datetime.now(timezone.utc)
    old = _iso(now - timedelta(seconds=240))
    job = _running_job("run-1", "host-a", last_heartbeat_at=old, started_at=old)
    health = {"host-a": HostHealth(
        name="host-a", healthy=False, last_probe_at=_iso(now),
        consecutive_failures=2, failure_reason="ssh_timeout",
        recovery_attempts=0, recovery_history=[], quarantined=False,
    )}
    w, _, _ = _make_watcher(tmp_path, jobs=[job], host_health=health)
    stale = w._compute_stale_jobs(health)
    assert stale[0].host_was_healthy is False


def test_stale_uses_started_at_when_no_heartbeat_yet(tmp_path):
    """Pre-heartbeat resume jobs use started_at as the staleness baseline."""
    now = datetime.now(timezone.utc)
    old = _iso(now - timedelta(seconds=240))
    job = _running_job("run-1", "host-a", last_heartbeat_at=None, started_at=old)
    health = {"host-a": HostHealth(
        name="host-a", healthy=True, last_probe_at=_iso(now),
        consecutive_failures=0, failure_reason=None, recovery_attempts=0,
        recovery_history=[], quarantined=False,
    )}
    w, _, _ = _make_watcher(tmp_path, jobs=[job], host_health=health)
    stale = w._compute_stale_jobs(health)
    assert len(stale) == 1
    assert stale[0].last_heartbeat_at == old   # baseline = started_at


def test_only_running_jobs_are_evaluated(tmp_path):
    """pending / completed / failed jobs are never stale."""
    now = datetime.now(timezone.utc)
    old = _iso(now - timedelta(seconds=600))

    pending = JobEntry(run_id="p", task_id="t", framework="rsl_rl", backend="physx",
                      num_envs=1, max_iterations=1, seed=0, bundle_dir_name="p")
    health = {"host-a": HostHealth(
        name="host-a", healthy=True, last_probe_at=_iso(now),
        consecutive_failures=0, failure_reason=None, recovery_attempts=0,
        recovery_history=[], quarantined=False,
    )}
    w, _, _ = _make_watcher(tmp_path, jobs=[pending], host_health=health)
    assert w._compute_stale_jobs(health) == []
```

- [ ] **Step 8.2: Run to verify it fails**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_heimdall_stale_jobs.py -v
```

Expected: tests fail because `_compute_stale_jobs` returns `[]`.

- [ ] **Step 8.3: Implement `_compute_stale_jobs`**

Replace the placeholder body in `tools/odin/asgard/heimdall.py`:

```python
    def _compute_stale_jobs(
        self, host_health: dict[str, HostHealth]
    ) -> list[StaleJob]:
        try:
            state = self._state_view()
        except Exception as exc:
            _log.warning("heimdall: state_view failed: %r", exc)
            return []
        now = datetime.now(timezone.utc)
        stale: list[StaleJob] = []
        for job in state.jobs:
            if job.status != "running":
                continue
            baseline_iso = job.last_heartbeat_at or job.started_at
            if baseline_iso is None:
                continue
            try:
                baseline = datetime.strptime(
                    baseline_iso, "%Y-%m-%dT%H:%M:%SZ"
                ).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            age = (now - baseline).total_seconds()
            if age <= self._stale_threshold_s:
                continue
            host_name = job.assigned_to or ""
            host_was_healthy = host_name in host_health and host_health[host_name].healthy
            stale.append(StaleJob(
                run_id=job.run_id,
                host=host_name,
                last_heartbeat_at=baseline_iso,
                age_seconds=age,
                host_was_healthy=host_was_healthy,
            ))
        return stale
```

- [ ] **Step 8.4: Run tests to verify pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_heimdall_stale_jobs.py -v
```

Expected: 5 passes.

- [ ] **Step 8.5: Run the full Heimdall test suite**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_heimdall_*.py -v
```

Expected: all pass.

- [ ] **Step 8.6: Commit**

```bash
git add tools/odin/asgard/heimdall.py tools/odin/tests/asgard/test_heimdall_stale_jobs.py
git commit -m "asgard: compute stale jobs via heartbeat staleness in HeimdallWatcher"
```

---

## Task 9: `_consume_heimdall_snapshot` — host-flip → recovery → quarantine

**Files:**
- Modify: `tools/odin/asgard/runner.py` (add `_consume_heimdall_snapshot`)
- Test: `tools/odin/tests/asgard/test_heimdall_consumer.py` *(new)*

- [ ] **Step 9.1: Write the failing host-flip test**

Create `tools/odin/tests/asgard/test_heimdall_consumer.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""_consume_heimdall_snapshot — main-loop consumption of HeimdallSnapshot."""

from __future__ import annotations

from dataclasses import dataclass

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
from tools.odin.asgard.heimdall import HeimdallSnapshot, HostHealth, StaleJob
from tools.odin.asgard.jobs import JobEntry
from tools.odin.asgard.recovery import RecoveryResult
from tools.odin.asgard.runner import _consume_heimdall_snapshot
from tools.odin.asgard.state import DispatchState, FleetSnapshot, QuarantinedHost, SCHEMA_VERSION
from tools.odin.asgard.transport import SSHResult


@dataclass
class _OkSSH:
    def run(self, host, cmd, *, timeout_s=None, stdout_tee=None, pty=True):
        return SSHResult(exit_code=0, stdout="GPU 0\n", stderr="", duration_s=0.01)


def _state_with_jobs(jobs, fleet_snap):
    return DispatchState(
        schema_version=SCHEMA_VERSION, dispatch_id="d",
        started_at="2026-05-08T14:00:00Z", ended_at=None,
        seeds=[0], commit_sha="x", fleet=fleet_snap, jobs=jobs,
    )


def _running_job(run_id: str, host: str) -> JobEntry:
    j = JobEntry(run_id=run_id, task_id="t", framework="rsl_rl", backend="physx",
                 num_envs=1, max_iterations=1, seed=0, bundle_dir_name=run_id)
    j.transition_to("running", assigned_to=host, now="2026-05-08T14:00:00Z")
    return j


def _hh(name, healthy):
    return HostHealth(name=name, healthy=healthy,
                      last_probe_at="2026-05-08T14:01:00Z",
                      consecutive_failures=(0 if healthy else 2),
                      failure_reason=(None if healthy else "ssh_timeout"),
                      recovery_attempts=0, recovery_history=[], quarantined=False)


def _snap(hosts, stale_jobs):
    return HeimdallSnapshot(
        generated_at="2026-05-08T14:01:00Z",
        hosts=hosts, stale_jobs=stale_jobs, recent_events=[],
    )


def test_no_flip_no_action_idempotent():
    fleet = Fleet(fleet_name="t", hosts=[ValkyrieConfig(host="a", ssh_user="u")])
    state = _state_with_jobs([_running_job("r1", "a")],
                             [FleetSnapshot(host="a", status="busy", current_run_id="r1")])
    snap = _snap({"a": _hh("a", True)}, [])
    last_consumed: list[str] = [None]
    def setter(v): last_consumed[0] = v

    _consume_heimdall_snapshot(snap, state, fleet, ssh=_OkSSH(),
                               last_consumed_at=last_consumed[0],
                               set_last_consumed=setter,
                               recover_fn=lambda h, ssh: RecoveryResult(
                                   host=h.host, container_name=h.container_name,
                                   attempted=False, recovered=False,
                                   duration_s=0.0, message="not invoked"))
    assert last_consumed[0] == snap.generated_at
    assert state.jobs[0].status == "running"
    assert state.quarantined_hosts == []


def test_flip_with_successful_recovery_clears_failure(tmp_path):
    fleet = Fleet(fleet_name="t", hosts=[ValkyrieConfig(host="a", ssh_user="u")])
    state = _state_with_jobs([_running_job("r1", "a")],
                             [FleetSnapshot(host="a", status="busy", current_run_id="r1")])
    state._heimdall_host_state = {"a": _hh("a", True)}      # prev tick was healthy
    snap = _snap({"a": _hh("a", False)}, [])
    last_consumed = [None]
    def setter(v): last_consumed[0] = v

    recovered = RecoveryResult(host="a", container_name="isaac-lab-base",
                                attempted=True, recovered=True,
                                duration_s=5.0, message="recovered_via_container_restart")

    _consume_heimdall_snapshot(snap, state, fleet, ssh=_OkSSH(),
                               last_consumed_at=last_consumed[0],
                               set_last_consumed=setter,
                               recover_fn=lambda h, ssh: recovered)
    # Job remains running on host a; no quarantine.
    assert state.jobs[0].status == "running"
    assert state.quarantined_hosts == []


def test_flip_with_failed_recovery_quarantines_and_requeues_jobs(tmp_path):
    fleet = Fleet(fleet_name="t", hosts=[ValkyrieConfig(host="a", ssh_user="u")])
    state = _state_with_jobs(
        [_running_job("r1", "a"), _running_job("r2", "a")],
        [FleetSnapshot(host="a", status="busy", current_run_id="r1")],
    )
    state._heimdall_host_state = {"a": _hh("a", True)}
    snap = _snap({"a": _hh("a", False)}, [])

    failed = RecoveryResult(host="a", container_name="isaac-lab-base",
                             attempted=True, recovered=False,
                             duration_s=5.0, message="docker_restart_failed: x")

    last_consumed = [None]
    def setter(v): last_consumed[0] = v

    _consume_heimdall_snapshot(snap, state, fleet, ssh=_OkSSH(),
                               last_consumed_at=last_consumed[0],
                               set_last_consumed=setter,
                               recover_fn=lambda h, ssh: failed)

    # Both jobs flipped back to pending; host added to preferred_not on each.
    assert state.jobs[0].status == "pending"
    assert state.jobs[1].status == "pending"
    assert "a" in state.jobs[0].preferred_not
    assert "a" in state.jobs[1].preferred_not
    # Quarantine record present.
    assert len(state.quarantined_hosts) == 1
    q = state.quarantined_hosts[0]
    assert q.host == "a"
    assert q.reason in {"heimdall_recovery_failed", "ssh_timeout", "gpu_lost"}


def test_idempotent_consumption_skips_duplicate_snapshot():
    fleet = Fleet(fleet_name="t", hosts=[ValkyrieConfig(host="a", ssh_user="u")])
    state = _state_with_jobs([_running_job("r1", "a")],
                             [FleetSnapshot(host="a", status="busy", current_run_id="r1")])
    state._heimdall_host_state = {"a": _hh("a", True)}
    snap = _snap({"a": _hh("a", False)}, [])

    failed = RecoveryResult(host="a", container_name="isaac-lab-base",
                             attempted=True, recovered=False,
                             duration_s=5.0, message="x")
    calls: list[str] = []

    def recover_fn(h, ssh):
        calls.append(h.host)
        return failed

    last_consumed = [None]
    def setter(v): last_consumed[0] = v

    _consume_heimdall_snapshot(snap, state, fleet, ssh=_OkSSH(),
                               last_consumed_at=last_consumed[0],
                               set_last_consumed=setter, recover_fn=recover_fn)
    # Second call with same generated_at must short-circuit.
    _consume_heimdall_snapshot(snap, state, fleet, ssh=_OkSSH(),
                               last_consumed_at=last_consumed[0],
                               set_last_consumed=setter, recover_fn=recover_fn)
    assert calls == ["a"]   # recovery only attempted once
```

- [ ] **Step 9.2: Run to verify it fails**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_heimdall_consumer.py -v
```

Expected: 4 ImportError-style failures on `_consume_heimdall_snapshot`.

- [ ] **Step 9.3: Implement `_consume_heimdall_snapshot` (host-flip half only)**

In `tools/odin/asgard/runner.py`, near the existing `_consume_*` helpers (after `_consume_cancellations` / `_mark_cancellation_consumed`), add:

```python
def _consume_heimdall_snapshot(
    snap: "HeimdallSnapshot",
    state: DispatchState,
    fleet: Fleet,
    *,
    ssh: SSHRunner,
    last_consumed_at: str | None,
    set_last_consumed: callable,
    recover_fn=None,
) -> None:
    """Apply one HeimdallSnapshot to ``state`` exactly once.

    Idempotent on ``snap.generated_at == last_consumed_at``. The runner
    keeps ``last_consumed_at`` between ticks (in a closure or attribute
    on the state). Host flips from healthy → unhealthy trigger one
    ``recover_valkyrie_gpu`` attempt; on failure the host is quarantined
    and any in-flight jobs assigned to it are flipped to pending with
    the host added to ``preferred_not``.

    Stale-job handling lands in Task 10 (currently a no-op).

    Args:
        snap: Latest snapshot from :meth:`HeimdallWatcher.latest`.
        state: Mutable :class:`DispatchState`; the runner is the sole
            writer.
        fleet: Same fleet the watcher is probing; used to look up
            :class:`ValkyrieConfig` for recovery.
        ssh: Same :class:`SSHRunner` the runner uses elsewhere; passed
            into ``recover_fn``.
        last_consumed_at: Previous snapshot's ``generated_at`` (or
            ``None`` on first tick). The runner stores this between
            calls.
        set_last_consumed: Callback the consumer calls with
            ``snap.generated_at`` after a successful pass; the runner
            persists it for the next tick.
        recover_fn: Override for :func:`recover_valkyrie_gpu`, used by
            tests. Production calls pass ``None`` and the import is
            resolved internally.
    """
    if snap is None or snap.generated_at == last_consumed_at:
        return

    if recover_fn is None:
        from tools.odin.asgard.recovery import recover_valkyrie_gpu as recover_fn
    prev_state: dict[str, "HostHealth"] = getattr(state, "_heimdall_host_state", {}) or {}
    host_lookup = {h.host: h for h in fleet.hosts}

    for host_name, h in snap.hosts.items():
        prev = prev_state.get(host_name)
        prev_was_healthy = prev.healthy if prev is not None else True
        if not (prev_was_healthy and not h.healthy):
            continue        # not a healthy → unhealthy flip
        host_cfg = host_lookup.get(host_name)
        if host_cfg is None:
            _log.warning("heimdall: flip on unknown host %r", host_name)
            continue
        rec = recover_fn(host_cfg, ssh=ssh)
        if rec.recovered:
            for f in state.fleet:
                if f.host == host_name:
                    f.last_error = "gpu_lost: heimdall recovery succeeded"
            continue
        # Recovery failed → quarantine + requeue all in-flight jobs.
        from tools.odin.asgard.state import QuarantinedHost
        state.quarantined_hosts.append(QuarantinedHost(
            host=host_name,
            reason="heimdall_recovery_failed",
            last_run_id="",
            at=_utc_now_iso(),
        ))
        for f in state.fleet:
            if f.host == host_name:
                f.status = "down"
                f.last_error = f"heimdall: {rec.message}"
                f.current_run_id = None
        for j in state.jobs:
            if j.assigned_to == host_name and j.status == "running":
                j.transition_to("pending", add_preferred_not=host_name)

    state._heimdall_host_state = dict(snap.hosts)
    set_last_consumed(snap.generated_at)


# Forward-declare type alias so this file doesn't have to import heimdall at
# module load (keeps a tight import graph for unit tests that import only
# state/jobs). Resolved only at type-check / runtime when consumer is called.
# (tools.odin.asgard.heimdall.HeimdallSnapshot)
```

Add to the imports at the top of `runner.py`:

```python
import logging

from tools.odin.asgard.heimdall import HeimdallSnapshot, HeimdallWatcher, HostHealth   # noqa: F401

_log = logging.getLogger(__name__)
```

(`HeimdallWatcher` is imported here for Task 11; including it now keeps the import block stable.)

- [ ] **Step 9.4: Run tests to verify pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_heimdall_consumer.py -v
```

Expected: 4 passes.

- [ ] **Step 9.5: Commit**

```bash
git add tools/odin/asgard/runner.py tools/odin/tests/asgard/test_heimdall_consumer.py
git commit -m "asgard: handle host-flip recovery + quarantine in heimdall consumer"
```

---

## Task 10: Stale-job branch in `_consume_heimdall_snapshot`

**Files:**
- Modify: `tools/odin/asgard/runner.py` (`_consume_heimdall_snapshot` stale-job branch)
- Test: `tools/odin/tests/asgard/test_heimdall_consumer.py` (append)

- [ ] **Step 10.1: Write the failing stale-job tests**

Append to `test_heimdall_consumer.py`:

```python
def test_stale_job_with_healthy_host_marks_failed_timeout(tmp_path):
    fleet = Fleet(fleet_name="t", hosts=[ValkyrieConfig(host="a", ssh_user="u")])
    state = _state_with_jobs([_running_job("r1", "a")],
                             [FleetSnapshot(host="a", status="busy", current_run_id="r1")])
    state._heimdall_host_state = {"a": _hh("a", True)}
    snap = HeimdallSnapshot(
        generated_at="2026-05-08T14:01:00Z",
        hosts={"a": _hh("a", True)},
        stale_jobs=[StaleJob(run_id="r1", host="a",
                             last_heartbeat_at="2026-05-08T13:55:00Z",
                             age_seconds=360.0, host_was_healthy=True)],
        recent_events=[],
    )
    last_consumed = [None]
    kill_calls: list[str] = []
    def kill_fn(host, run_id, ssh, *, timeout_s):
        kill_calls.append(run_id)
    def setter(v): last_consumed[0] = v

    _consume_heimdall_snapshot(snap, state, fleet, ssh=_OkSSH(),
                               last_consumed_at=last_consumed[0],
                               set_last_consumed=setter,
                               recover_fn=lambda h, ssh: RecoveryResult(
                                   host=h.host, container_name=h.container_name,
                                   attempted=False, recovered=False,
                                   duration_s=0.0, message="x"),
                               kill_fn=kill_fn)
    assert kill_calls == ["r1"]
    assert state.jobs[0].status == "failed"
    assert state.jobs[0].failure is not None
    assert state.jobs[0].failure.kind == "timeout"


def test_stale_job_with_unhealthy_host_requeues_as_infrastructure(tmp_path):
    fleet = Fleet(fleet_name="t", hosts=[ValkyrieConfig(host="a", ssh_user="u")])
    state = _state_with_jobs([_running_job("r1", "a")],
                             [FleetSnapshot(host="a", status="busy", current_run_id="r1")])
    # Previous tick already saw host unhealthy — no fresh flip this tick.
    state._heimdall_host_state = {"a": _hh("a", False)}
    snap = HeimdallSnapshot(
        generated_at="2026-05-08T14:01:00Z",
        hosts={"a": _hh("a", False)},
        stale_jobs=[StaleJob(run_id="r1", host="a",
                             last_heartbeat_at="2026-05-08T13:55:00Z",
                             age_seconds=360.0, host_was_healthy=False)],
        recent_events=[],
    )
    last_consumed = [None]
    def setter(v): last_consumed[0] = v
    def kill_fn(host, run_id, ssh, *, timeout_s): pass

    _consume_heimdall_snapshot(snap, state, fleet, ssh=_OkSSH(),
                               last_consumed_at=last_consumed[0],
                               set_last_consumed=setter,
                               recover_fn=lambda h, ssh: RecoveryResult(
                                   host=h.host, container_name=h.container_name,
                                   attempted=False, recovered=False,
                                   duration_s=0.0, message="x"),
                               kill_fn=kill_fn)
    assert state.jobs[0].status == "pending"        # requeued
    assert "a" in state.jobs[0].preferred_not


def test_stale_job_skipped_if_already_terminal():
    fleet = Fleet(fleet_name="t", hosts=[ValkyrieConfig(host="a", ssh_user="u")])
    j = _running_job("r1", "a")
    j.transition_to("completed", now="2026-05-08T13:59:00Z")
    state = _state_with_jobs([j],
                             [FleetSnapshot(host="a", status="idle", current_run_id=None)])
    state._heimdall_host_state = {"a": _hh("a", True)}
    snap = HeimdallSnapshot(
        generated_at="2026-05-08T14:01:00Z",
        hosts={"a": _hh("a", True)},
        stale_jobs=[StaleJob(run_id="r1", host="a",
                             last_heartbeat_at="2026-05-08T13:55:00Z",
                             age_seconds=360.0, host_was_healthy=True)],
        recent_events=[],
    )
    last_consumed = [None]
    def setter(v): last_consumed[0] = v
    kill_calls: list[str] = []
    def kill_fn(host, run_id, ssh, *, timeout_s): kill_calls.append(run_id)

    _consume_heimdall_snapshot(snap, state, fleet, ssh=_OkSSH(),
                               last_consumed_at=last_consumed[0],
                               set_last_consumed=setter,
                               recover_fn=lambda h, ssh: RecoveryResult(
                                   host=h.host, container_name=h.container_name,
                                   attempted=False, recovered=False,
                                   duration_s=0.0, message="x"),
                               kill_fn=kill_fn)
    assert kill_calls == []
    assert state.jobs[0].status == "completed"
```

- [ ] **Step 10.2: Run to verify failure**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_heimdall_consumer.py::test_stale_job_with_healthy_host_marks_failed_timeout tools/odin/tests/asgard/test_heimdall_consumer.py::test_stale_job_with_unhealthy_host_requeues_as_infrastructure tools/odin/tests/asgard/test_heimdall_consumer.py::test_stale_job_skipped_if_already_terminal -v
```

Expected: 3 failures (consumer ignores stale_jobs).

- [ ] **Step 10.3: Add stale-job branch + `kill_fn` parameter**

Update `_consume_heimdall_snapshot` in `tools/odin/asgard/runner.py`:

1. Add a `kill_fn` parameter (default `None`):

```python
def _consume_heimdall_snapshot(
    snap: "HeimdallSnapshot",
    state: DispatchState,
    fleet: Fleet,
    *,
    ssh: SSHRunner,
    last_consumed_at: str | None,
    set_last_consumed: callable,
    recover_fn=None,
    kill_fn=None,
) -> None:
```

2. Add a default `kill_fn` implementation just below the recover_fn import:

```python
    if kill_fn is None:
        def kill_fn(host, run_id, ssh, *, timeout_s):
            from tools.odin.asgard.transport import SSHResult   # noqa: F401
            cmd = (
                f"docker exec {host.container_name} "
                f"sh -c 'pkill -f {run_id} || true'"
            )
            try:
                ssh.run(host, cmd, timeout_s=timeout_s, pty=False)
            except Exception as exc:
                _log.warning("heimdall: kill_fn ssh exception: %r", exc)
```

3. After the host-flip loop (and before the bookkeeping at the end), add:

```python
    jobs_by_id = {j.run_id: j for j in state.jobs}
    for sj in snap.stale_jobs:
        j = jobs_by_id.get(sj.run_id)
        if j is None or j.status != "running":
            continue
        host_cfg = host_lookup.get(sj.host)
        if host_cfg is not None:
            try:
                kill_fn(host_cfg, sj.run_id, ssh, timeout_s=10)
            except Exception as exc:
                _log.warning("heimdall: kill_fn raised: %r", exc)
        if sj.host_was_healthy:
            j.transition_to(
                "failed",
                failure=FailureInfo(
                    kind="timeout",
                    message=(
                        f"heimdall: stale heartbeat (age={sj.age_seconds:.0f}s) "
                        "with healthy host — trainer wedge"
                    ),
                ),
                now=_utc_now_iso(),
            )
        else:
            j.transition_to("pending", add_preferred_not=sj.host)
```

(`FailureInfo` is already imported at the top of `runner.py`.)

- [ ] **Step 10.4: Run tests to verify pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_heimdall_consumer.py -v
```

Expected: all consumer tests pass.

- [ ] **Step 10.5: Commit**

```bash
git add tools/odin/asgard/runner.py tools/odin/tests/asgard/test_heimdall_consumer.py
git commit -m "asgard: classify stale jobs by host health in heimdall consumer"
```

---

## Task 11: Wire watcher into `run_dispatch` + `--no-heimdall` CLI flag

**Files:**
- Modify: `tools/odin/asgard/runner.py` (DispatchOptions, `run_dispatch`)
- Modify: `tools/odin/asgard/cli.py`
- Test: `tools/odin/tests/asgard/test_runner_heimdall_wiring.py` *(new)*

- [ ] **Step 11.1: Write the failing wiring test**

Create `tools/odin/tests/asgard/test_runner_heimdall_wiring.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""run_dispatch starts/stops the HeimdallWatcher when enabled."""

from __future__ import annotations

import threading

from tools.odin.asgard.runner import DispatchOptions


def test_dispatch_options_has_heimdall_flag_default_on():
    opts = DispatchOptions(seeds=[0])
    assert opts.no_heimdall is False


def test_dispatch_options_no_heimdall_flag_can_be_set():
    opts = DispatchOptions(seeds=[0], no_heimdall=True)
    assert opts.no_heimdall is True
```

- [ ] **Step 11.2: Run to verify failure**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_runner_heimdall_wiring.py -v
```

Expected: failures because `no_heimdall` is not a `DispatchOptions` field.

- [ ] **Step 11.3: Add the option + CLI flag**

In `tools/odin/asgard/runner.py`, add to `DispatchOptions`:

```python
    no_heimdall: bool = False
    heimdall_probe_interval_s: int = 300
    heimdall_stale_threshold_s: int = 180
```

In `tools/odin/asgard/cli.py`, find the `argparse` setup and add:

```python
    parser.add_argument(
        "--no-heimdall",
        action="store_true",
        help="Disable the Heimdall fleet watcher (periodic re-probe + stale-job kill).",
    )
    parser.add_argument(
        "--heimdall-probe-interval-s",
        type=int,
        default=300,
        help="Heimdall probe interval in seconds (default 300).",
    )
    parser.add_argument(
        "--heimdall-stale-threshold-s",
        type=int,
        default=180,
        help="Mark a job stale if its heartbeat is older than this (default 180s).",
    )
```

And forward those into the `DispatchOptions(...)` construction:

```python
        no_heimdall=args.no_heimdall,
        heimdall_probe_interval_s=args.heimdall_probe_interval_s,
        heimdall_stale_threshold_s=args.heimdall_stale_threshold_s,
```

- [ ] **Step 11.4: Verify the wiring tests pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_runner_heimdall_wiring.py -v
```

Expected: 2 passes.

- [ ] **Step 11.5: Wire watcher start/stop into `run_dispatch`**

In `tools/odin/asgard/runner.py`'s `run_dispatch`, after `_snapshot_fleet_yaml(fleet, dispatch_dir)` and after the initial preflight + `state` is materialized but before the main dispatch loop begins, add:

```python
    watcher: HeimdallWatcher | None = None
    last_heimdall_consumed_at: str | None = None
    if not options.no_heimdall:
        def _state_view() -> DispatchState:
            return state          # main loop is sole writer; reads are eventually-consistent

        watcher = HeimdallWatcher(
            fleet=fleet,
            dispatch_dir=dispatch_dir,
            ssh=ssh,
            state_view=_state_view,
            probe_interval_s=options.heimdall_probe_interval_s,
            stale_threshold_s=options.heimdall_stale_threshold_s,
        )
        watcher.start()
```

Inside the main loop iteration, alongside the `_consume_live_retries` / `_consume_cancellations` calls, add:

```python
        if watcher is not None:
            if watcher.is_alive():
                snap = watcher.latest()
                if snap is not None:
                    def _set_last(v): nonlocal last_heimdall_consumed_at; last_heimdall_consumed_at = v
                    _consume_heimdall_snapshot(
                        snap, state, fleet,
                        ssh=ssh,
                        last_consumed_at=last_heimdall_consumed_at,
                        set_last_consumed=_set_last,
                    )
            else:
                _log.warning("heimdall: watcher thread is not alive; continuing without it")
```

In the `finally` block of `run_dispatch`, before any other cleanup that depends on `state`:

```python
        if watcher is not None:
            watcher.stop(timeout_s=10.0)
```

- [ ] **Step 11.6: Sanity-run the full asgard test suite**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/ -v
```

Expected: all asgard tests pass (no regressions). Heimdall wiring is opt-in by absence of `--no-heimdall` but the test suite uses `DispatchOptions` directly with `no_heimdall=False`, so existing tests run with the watcher enabled. If any existing dispatcher-level test fails because the watcher's daemon thread leaks, set `no_heimdall=True` in those tests (the simplest fix; the watcher behavior is now covered by the dedicated test files).

- [ ] **Step 11.7: Commit**

```bash
git add tools/odin/asgard/runner.py tools/odin/asgard/cli.py tools/odin/tests/asgard/test_runner_heimdall_wiring.py
git commit -m "asgard: start/stop HeimdallWatcher in run_dispatch + add CLI flags"
```

---

## Task 12: `reconcile_orphans` — try recovery before flip-to-pending

**Files:**
- Modify: `tools/odin/asgard/reconcile.py` (around `reconcile_orphans` at `:170`)
- Test: `tools/odin/tests/asgard/test_reconcile_recovery_first.py` *(new, regression)*

- [ ] **Step 12.1: Write the regression test**

Create `tools/odin/tests/asgard/test_reconcile_recovery_first.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression — reconcile_orphans attempts host recovery before flipping to pending."""

from __future__ import annotations

from dataclasses import dataclass

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
from tools.odin.asgard.recovery import RecoveryResult


@dataclass
class _UnreachableSSH:
    """Always reports SSH unreachable on direct probes (simulates orphan host)."""
    def run(self, host, cmd, *, timeout_s=None, stdout_tee=None, pty=True):
        from tools.odin.asgard.transport import SSHResult
        return SSHResult(exit_code=255, stdout="", stderr="ssh: connect failed",
                         duration_s=0.5)


def test_reconcile_orphan_attempts_recovery_before_flip_to_pending(tmp_path):
    """When a 'running' job's host appears unreachable, reconcile_orphans must
    invoke recovery once. On success it leaves the job in 'running'."""
    from tools.odin.asgard.jobs import JobEntry
    from tools.odin.asgard.reconcile import reconcile_orphans
    from tools.odin.asgard.state import DispatchState, FleetSnapshot, SCHEMA_VERSION

    job = JobEntry(run_id="run-1", task_id="t", framework="rsl_rl", backend="physx",
                   num_envs=1, max_iterations=1, seed=0, bundle_dir_name="run-1")
    job.transition_to("running", assigned_to="host-a", now="2026-05-08T14:00:00Z")
    state = DispatchState(
        schema_version=SCHEMA_VERSION, dispatch_id="d",
        started_at="2026-05-08T14:00:00Z", ended_at=None,
        seeds=[0], commit_sha="x",
        fleet=[FleetSnapshot(host="host-a", status="busy", current_run_id="run-1")],
        jobs=[job],
    )
    fleet = Fleet(fleet_name="t", hosts=[ValkyrieConfig(host="host-a", ssh_user="u")])

    recovery_calls = []

    def recover_fn(host, ssh):
        recovery_calls.append(host.host)
        return RecoveryResult(host=host.host, container_name=host.container_name,
                              attempted=True, recovered=True, duration_s=2.0,
                              message="recovered_via_container_restart")

    reconcile_orphans(
        state=state, fleet=fleet, ssh=_UnreachableSSH(),
        rsync=None, dispatch_dir=tmp_path, cancel_db=None,
        recover_fn=recover_fn,
    )
    # Recovery was attempted, succeeded → job stays running.
    assert recovery_calls == ["host-a"]
    assert state.jobs[0].status == "running"


def test_reconcile_orphan_flips_to_pending_when_recovery_fails(tmp_path):
    from tools.odin.asgard.jobs import JobEntry
    from tools.odin.asgard.reconcile import reconcile_orphans
    from tools.odin.asgard.state import DispatchState, FleetSnapshot, SCHEMA_VERSION

    job = JobEntry(run_id="run-1", task_id="t", framework="rsl_rl", backend="physx",
                   num_envs=1, max_iterations=1, seed=0, bundle_dir_name="run-1")
    job.transition_to("running", assigned_to="host-a", now="2026-05-08T14:00:00Z")
    state = DispatchState(
        schema_version=SCHEMA_VERSION, dispatch_id="d",
        started_at="2026-05-08T14:00:00Z", ended_at=None,
        seeds=[0], commit_sha="x",
        fleet=[FleetSnapshot(host="host-a", status="busy", current_run_id="run-1")],
        jobs=[job],
    )
    fleet = Fleet(fleet_name="t", hosts=[ValkyrieConfig(host="host-a", ssh_user="u")])

    def recover_fn(host, ssh):
        return RecoveryResult(host=host.host, container_name=host.container_name,
                              attempted=True, recovered=False, duration_s=2.0,
                              message="docker_restart_failed")

    reconcile_orphans(
        state=state, fleet=fleet, ssh=_UnreachableSSH(),
        rsync=None, dispatch_dir=tmp_path, cancel_db=None,
        recover_fn=recover_fn,
    )
    # Recovery failed → job flipped back to pending (existing reconcile behavior).
    assert state.jobs[0].status == "pending"
```

- [ ] **Step 12.2: Verify the regression tests fail before the fix**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_reconcile_recovery_first.py -v
```

Expected: both tests fail. The most likely failure is `TypeError: reconcile_orphans() got an unexpected keyword argument 'recover_fn'` (until Step 12.3 adds the parameter). After Step 12.3 lands, the first test must pass because recovery is invoked and succeeds; the second test must pass because recovery is invoked and fails, falling through to the existing flip-to-pending path.

- [ ] **Step 12.3: Read the current `reconcile_orphans` signature**

Open `tools/odin/asgard/reconcile.py` around line 170 and identify the function signature and the path that flips a `running` job to `pending`. Add a `recover_fn` keyword parameter (default `None`, falling back to `recover_valkyrie_gpu`).

In the branch that detects an orphan (host unreachable) for a `running` job, before flipping to `pending`, attempt one recovery:

```python
        if recover_fn is None:
            from tools.odin.asgard.recovery import recover_valkyrie_gpu as recover_fn
        host_cfg = next((h for h in fleet.hosts if h.host == job.assigned_to), None)
        if host_cfg is not None:
            rec = recover_fn(host_cfg, ssh=ssh)
            if rec.recovered:
                # Host recovered — leave job in 'running'; the next
                # heimdall snapshot will pick it up.
                continue
        # Recovery failed (or host config missing) → original flip path.
        job.transition_to("pending")
```

The exact insertion point depends on the existing code structure; the implementor must preserve the existing flow for non-`running` orphans (`assigned`, partial bundles, etc.) and only insert the recovery attempt on the `running` branch.

- [ ] **Step 12.4: Run the regression tests**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_reconcile_recovery_first.py -v
```

Expected: 2 passes.

- [ ] **Step 12.5: Run the existing reconcile suite**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_reconcile*.py -v
```

Expected: all existing reconcile tests still pass.

- [ ] **Step 12.6: Commit**

```bash
git add tools/odin/asgard/reconcile.py tools/odin/tests/asgard/test_reconcile_recovery_first.py
git commit -m "asgard: try recovery before flipping orphan running jobs to pending"
```

---

## Task 13: Heimdall dashboard card — renderer

**Files:**
- Create: `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/heimdall_card.py`
- Test: `tools/odin/tests/valhalla/test_heimdall_card.py` *(new)*

- [ ] **Step 13.1: Write the failing renderer test**

Create `tools/odin/tests/valhalla/test_heimdall_card.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Renderer tests for the Heimdall dashboard card."""

from __future__ import annotations

import json

from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.heimdall_card import (
    render_heimdall_card,
    render_empty_state,
)


def _payload(hosts: dict, recent_events: list) -> dict:
    return {
        "schema_version": "1.0",
        "generated_at": "2026-05-08T14:32:18Z",
        "hosts": hosts,
        "recent_events": recent_events,
    }


def test_render_empty_state_returns_placeholder():
    div = render_empty_state()
    text = json.dumps(div, default=str)
    assert "Heimdall not active" in text or "no fleet.json" in text


def test_render_card_shows_last_lookup_timestamp():
    payload = _payload(
        hosts={"a": {"name": "a", "healthy": True, "last_probe_at": "2026-05-08T14:32:18Z",
                     "consecutive_failures": 0, "failure_reason": None,
                     "recovery_attempts": 0, "recovery_history": [], "quarantined": False}},
        recent_events=[],
    )
    div = render_heimdall_card(payload, now_iso="2026-05-08T14:33:05Z")
    text = json.dumps(div, default=str)
    assert "2026-05-08T14:32:18Z" in text
    assert "47s ago" in text or "47 s ago" in text


def test_render_card_marks_unhealthy_with_reason():
    payload = _payload(
        hosts={"b": {"name": "b", "healthy": False, "last_probe_at": "2026-05-08T14:32:18Z",
                     "consecutive_failures": 2, "failure_reason": "ssh_timeout",
                     "recovery_attempts": 0, "recovery_history": [], "quarantined": False}},
        recent_events=[],
    )
    div = render_heimdall_card(payload, now_iso="2026-05-08T14:33:00Z")
    text = json.dumps(div, default=str)
    assert "b" in text
    assert "ssh_timeout" in text


def test_render_card_marks_quarantined():
    payload = _payload(
        hosts={"c": {"name": "c", "healthy": False, "last_probe_at": "2026-05-08T14:32:18Z",
                     "consecutive_failures": 5, "failure_reason": "ssh_timeout",
                     "recovery_attempts": 1,
                     "recovery_history": ["2026-05-08T14:30:00Z"],
                     "quarantined": True}},
        recent_events=[],
    )
    div = render_heimdall_card(payload, now_iso="2026-05-08T14:33:00Z")
    text = json.dumps(div, default=str)
    assert "quarantined" in text.lower() or "⛔" in text


def test_render_card_shows_recent_events():
    payload = _payload(
        hosts={},
        recent_events=[
            {"ts": "2026-05-08T14:31:00Z", "kind": "host_flipped",
             "host": "b", "reason": "ssh_timeout"},
            {"ts": "2026-05-08T14:31:05Z", "kind": "host_quarantined",
             "host": "b", "reason": "recovery_failed"},
        ],
    )
    div = render_heimdall_card(payload, now_iso="2026-05-08T14:33:00Z")
    text = json.dumps(div, default=str)
    assert "host_flipped" in text
    assert "host_quarantined" in text
```

- [ ] **Step 13.2: Run to verify failure**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/valhalla/test_heimdall_card.py -v
```

Expected: ImportError on `render_heimdall_card` / `render_empty_state`.

- [ ] **Step 13.3: Implement the renderer**

Create `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/heimdall_card.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Heimdall card — sibling panel of fleet_table.py on the dispatch_fleet tab."""

from __future__ import annotations

from datetime import datetime, timezone

from dash import html


__all__ = ["render_heimdall_card", "render_empty_state"]


def _parse_iso(ts: str) -> datetime | None:
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _age_str(generated_at: str, now_iso: str | None) -> str:
    g = _parse_iso(generated_at)
    if g is None:
        return "unknown"
    if now_iso is None:
        now = datetime.now(timezone.utc)
    else:
        now = _parse_iso(now_iso) or datetime.now(timezone.utc)
    delta_s = max(0, int((now - g).total_seconds()))
    if delta_s < 60:
        return f"{delta_s}s ago"
    minutes, secs = divmod(delta_s, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s ago"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m ago"


def _host_row(name: str, h: dict) -> html.Div:
    if h.get("quarantined"):
        icon, status = "⛔", f"quarantined ({h.get('failure_reason') or 'unknown'})"
    elif h.get("healthy"):
        icon, status = "✓", "healthy"
    else:
        icon, status = "✗", f"unhealthy ({h.get('failure_reason') or 'unknown'})"
    return html.Div(
        children=[
            html.Span(icon, className="heimdall-icon"),
            html.Span(f" {name}: ", className="heimdall-host"),
            html.Span(status, className="heimdall-status"),
        ],
        className="heimdall-host-row",
    )


def _event_row(ev: dict) -> html.Div:
    summary = f"{ev.get('ts', '?')} — {ev.get('kind', '?')}"
    host = ev.get("host")
    if host:
        summary += f" {host}"
    reason = ev.get("reason")
    if reason:
        summary += f" ({reason})"
    return html.Div(summary, className="heimdall-event-row")


def render_empty_state() -> html.Div:
    return html.Div(
        children=[
            html.H4("Heimdall"),
            html.Div("Heimdall not active for this dispatch (no fleet.json).",
                     className="heimdall-empty"),
        ],
        className="heimdall-card",
    )


def render_heimdall_card(payload: dict, *, now_iso: str | None = None) -> html.Div:
    """Render the panel for one fleet.json payload.

    Args:
        payload: Raw dict as returned by
            :func:`tools.odin.asgard.heimdall.read_fleet_json`.
        now_iso: Optional UTC timestamp used to compute "X ago" age. Tests
            pin this to make output deterministic. Production callers
            pass ``None`` to use wall-clock time.
    """
    if payload is None:
        return render_empty_state()
    generated_at = payload.get("generated_at", "")
    age = _age_str(generated_at, now_iso)
    hosts = payload.get("hosts", {}) or {}
    events = payload.get("recent_events", []) or []
    stale_count = sum(1 for ev in events if ev.get("kind") == "stale_job_killed")

    return html.Div(
        children=[
            html.H4("Heimdall"),
            html.Div(
                f"Last look-up: {generated_at} ({age})",
                className="heimdall-header",
            ),
            html.Div(
                children=[_host_row(name, h) for name, h in sorted(hosts.items())],
                className="heimdall-hosts",
            ),
            html.Div(
                children=[
                    html.H5("Recent activity"),
                    *[_event_row(ev) for ev in events[-5:]],
                ] if events else [html.H5("Recent activity"),
                                  html.Div("No recent events.",
                                           className="heimdall-empty")],
                className="heimdall-events",
            ),
            html.Div(
                f"Stale jobs killed: {stale_count}" if stale_count else "",
                className="heimdall-stale-badge",
            ),
        ],
        className="heimdall-card",
    )
```

- [ ] **Step 13.4: Run renderer tests**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/valhalla/test_heimdall_card.py -v
```

Expected: 5 passes.

- [ ] **Step 13.5: Commit**

```bash
git add tools/odin/valhalla/dashboard/tabs/dispatch_fleet/heimdall_card.py tools/odin/tests/valhalla/test_heimdall_card.py
git commit -m "valhalla: add Heimdall dashboard card renderer"
```

---

## Task 14: Wire Heimdall card into dispatch_fleet tab

**Files:**
- Modify: `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/layout.py`
- Modify: `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/callbacks.py`
- Modify: `tools/odin/valhalla/dashboard/assets/style.css` (heimdall-* classes — optional polish)

- [ ] **Step 14.1: Read the current layout + callbacks structure**

```bash
sed -n '1,60p' tools/odin/valhalla/dashboard/tabs/dispatch_fleet/layout.py
sed -n '1,80p' tools/odin/valhalla/dashboard/tabs/dispatch_fleet/callbacks.py
```

Note the existing pattern for the fleet table: where `tab-a-fleet-table` is rendered, the polling interval, and the dispatch-id selector wiring.

- [ ] **Step 14.2: Add the placeholder div in `layout.py`**

In `build_layout`, immediately after the existing `html.Div(id="tab-a-fleet-table")` (around `:52`), add:

```python
            html.Div(id="tab-a-heimdall-card", className="heimdall-card-host"),
```

- [ ] **Step 14.3: Register the callback in `callbacks.py`**

Add a new callback that runs on the same interval the fleet-table callback uses. The exact decorator and inputs depend on the existing pattern; the new callback shape is:

```python
from pathlib import Path

from dash import Input, Output, callback

from tools.odin.asgard.heimdall import read_fleet_json
from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.heimdall_card import (
    render_empty_state,
    render_heimdall_card,
)


@callback(
    Output("tab-a-heimdall-card", "children"),
    Input("tab-a-fleet-poll", "n_intervals"),    # reuse the existing poller
    Input("dispatch-id-store", "data"),           # whatever component carries the active dispatch id
)
def _update_heimdall_card(_n, dispatch_id):
    if not dispatch_id:
        return render_empty_state()
    dispatch_dir = Path("odin_runs") / dispatch_id
    payload = read_fleet_json(dispatch_dir)
    if payload is None:
        return render_empty_state()
    return render_heimdall_card(payload)
```

(Component IDs above are placeholders — match the existing fleet-table callback's input IDs verbatim from `callbacks.py`.)

- [ ] **Step 14.4: Smoke-test the dashboard import**

```bash
./isaaclab.sh -p -c "from tools.odin.valhalla.dashboard.tabs.dispatch_fleet import callbacks, layout; print('ok')"
```

Expected: `ok` (no ImportError, no Dash registration failure).

- [ ] **Step 14.5: Commit**

```bash
git add tools/odin/valhalla/dashboard/tabs/dispatch_fleet/layout.py tools/odin/valhalla/dashboard/tabs/dispatch_fleet/callbacks.py
git commit -m "valhalla: render Heimdall card alongside fleet table"
```

---

## Task 15: 22h-wedge regression test

**Files:**
- Test: `tools/odin/tests/asgard/test_heimdall_22h_wedge_regression.py` *(new)*

- [ ] **Step 15.1: Write the regression test**

Create `tools/odin/tests/asgard/test_heimdall_22h_wedge_regression.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression — host wedges mid-dispatch must be detected and self-healed.

Models the 2026-04-30 incident where two hosts wedged for 22 hours and the
dispatcher kept assigning no work to them. Expected behavior: Heimdall
detects the flip after K consecutive failures, attempts recovery, and on
recovery failure quarantines + re-queues all in-flight jobs.
"""

from __future__ import annotations

from dataclasses import dataclass

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
from tools.odin.asgard.heimdall import HeimdallWatcher
from tools.odin.asgard.recovery import RecoveryResult
from tools.odin.asgard.runner import _consume_heimdall_snapshot
from tools.odin.asgard.state import DispatchState, FleetSnapshot, SCHEMA_VERSION
from tools.odin.asgard.transport import SSHResult


@dataclass
class _FlipsAfterTickSSH:
    """Healthy on first probe, then SSH-timed-out forever after."""
    tick: int = 0

    def run(self, host, cmd, *, timeout_s=None, stdout_tee=None, pty=True):
        self.tick += 1
        if self.tick == 1:
            return SSHResult(exit_code=0, stdout="GPU 0\n", stderr="",
                             duration_s=0.01)
        return SSHResult(exit_code=255, stdout="", stderr="ssh: connect failed",
                         duration_s=15.0, timed_out=True)


def test_22h_wedge_is_detected_and_quarantined(tmp_path):
    fleet = Fleet(fleet_name="t",
                  hosts=[ValkyrieConfig(host="host-a", ssh_user="u")])
    from tools.odin.asgard.jobs import JobEntry
    j1 = JobEntry(run_id="r1", task_id="t", framework="rsl_rl", backend="physx",
                  num_envs=1, max_iterations=1, seed=0, bundle_dir_name="r1")
    j1.transition_to("running", assigned_to="host-a", now="2026-05-08T14:00:00Z")
    j2 = JobEntry(run_id="r2", task_id="t", framework="rsl_rl", backend="physx",
                  num_envs=1, max_iterations=1, seed=1, bundle_dir_name="r2")
    j2.transition_to("running", assigned_to="host-a", now="2026-05-08T14:00:00Z")
    state = DispatchState(
        schema_version=SCHEMA_VERSION, dispatch_id="d",
        started_at="2026-05-08T14:00:00Z", ended_at=None,
        seeds=[0, 1], commit_sha="x",
        fleet=[FleetSnapshot(host="host-a", status="busy", current_run_id="r1")],
        jobs=[j1, j2],
    )

    ssh = _FlipsAfterTickSSH()
    w = HeimdallWatcher(
        fleet=fleet, dispatch_dir=tmp_path, ssh=ssh,
        state_view=lambda: state,
        probe_interval_s=10000, stale_threshold_s=180,
        flip_after_k_failures=2, probe_timeout_s=5,
    )

    # Tick 1: healthy.
    w._tick_once()
    snap1 = w.latest()
    assert snap1.hosts["host-a"].healthy is True

    # Tick 2: first failure — not flipped yet (K=2).
    w._tick_once()
    snap2 = w.latest()
    assert snap2.hosts["host-a"].healthy is True
    assert snap2.hosts["host-a"].consecutive_failures == 1

    # Tick 3: second failure — flipped.
    w._tick_once()
    snap3 = w.latest()
    assert snap3.hosts["host-a"].healthy is False

    # Run the consumer with a recovery-fails recover_fn.
    failed = RecoveryResult(host="host-a", container_name="isaac-lab-base",
                             attempted=True, recovered=False,
                             duration_s=2.0, message="docker_restart_failed: x")
    last_consumed = [None]
    def setter(v): last_consumed[0] = v
    def kill_fn(host, run_id, ssh, *, timeout_s): pass

    # Seed prev-tick state so the consumer sees a healthy → unhealthy flip.
    state._heimdall_host_state = dict(snap2.hosts)
    _consume_heimdall_snapshot(snap3, state, fleet, ssh=ssh,
                               last_consumed_at=last_consumed[0],
                               set_last_consumed=setter,
                               recover_fn=lambda h, ssh: failed,
                               kill_fn=kill_fn)

    # Both jobs requeued; host quarantined.
    assert state.jobs[0].status == "pending"
    assert state.jobs[1].status == "pending"
    assert "host-a" in state.jobs[0].preferred_not
    assert "host-a" in state.jobs[1].preferred_not
    assert len(state.quarantined_hosts) == 1
    assert state.quarantined_hosts[0].host == "host-a"
```

- [ ] **Step 15.2: Verify the test fails when Heimdall is bypassed**

To confirm the test really exercises Heimdall, simulate "Heimdall absent" by skipping the consumer call. Temporarily comment out the `_consume_heimdall_snapshot(...)` call. Run:

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_heimdall_22h_wedge_regression.py -v
```

Expected: FAIL — both jobs still in `running`. Restore the consumer call and re-run to confirm PASS.

- [ ] **Step 15.3: Run the full Heimdall + regression suite**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_heimdall_*.py tools/odin/tests/asgard/test_reconcile_recovery_first.py -v
```

Expected: all pass.

- [ ] **Step 15.4: Commit**

```bash
git add tools/odin/tests/asgard/test_heimdall_22h_wedge_regression.py
git commit -m "asgard: add regression test for 22h-wedge mid-dispatch incident"
```

---

## Task 16: End-to-end integration test (slow-marked, ssh localhost)

**Files:**
- Test: `tools/odin/tests/asgard/test_heimdall_e2e.py` *(new, slow-marked)*

- [ ] **Step 16.1: Write the integration test**

Create `tools/odin/tests/asgard/test_heimdall_e2e.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end Heimdall integration test against a loopback fleet.

Skipped when ``ssh localhost`` is unavailable. Verifies that a host
flipping unhealthy mid-dispatch is detected by the watcher, recovery is
attempted, and (on recovery failure) the host is quarantined.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import pytest

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
from tools.odin.asgard.heimdall import HeimdallWatcher
from tools.odin.asgard.transport import ShellSSHRunner


pytestmark = pytest.mark.slow


def _ssh_localhost_works() -> bool:
    if shutil.which("ssh") is None:
        return False
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
             "-o", "ConnectTimeout=5", "localhost", "echo ok"],
            capture_output=True, text=True, timeout=15,
        )
        return r.returncode == 0 and "ok" in r.stdout
    except subprocess.SubprocessError:
        return False


def test_heimdall_detects_flipped_host_e2e(tmp_path):
    if not _ssh_localhost_works():
        pytest.skip("ssh localhost not available in this environment")

    sentinel = tmp_path / "host_a_alive"
    sentinel.write_text("alive")

    # The probe targets a docker container that doesn't exist on the test host;
    # we instead point the probe at a sentinel-file check via a custom
    # container_name → command override. The simplest end-to-end shape is to
    # construct a fake ValkyrieConfig and point _probe_host at a script that
    # checks the sentinel. This keeps the test hermetic.
    host_a = ValkyrieConfig(
        host="localhost", ssh_user=Path.home().name, ssh_key=None,
        isaaclab_path=str(tmp_path),
        container_name=f"sh -c 'test -f {sentinel} && echo GPU 0 || exit 1' #",
    )
    fleet = Fleet(fleet_name="t", hosts=[host_a])

    from tools.odin.asgard.state import DispatchState, SCHEMA_VERSION
    state = DispatchState(
        schema_version=SCHEMA_VERSION, dispatch_id="d",
        started_at="2026-05-08T14:00:00Z", ended_at=None,
        seeds=[0], commit_sha="x", fleet=[], jobs=[],
    )

    w = HeimdallWatcher(
        fleet=fleet, dispatch_dir=tmp_path, ssh=ShellSSHRunner(),
        state_view=lambda: state,
        probe_interval_s=10000, stale_threshold_s=180,
        flip_after_k_failures=2, probe_timeout_s=10,
    )

    # Tick 1: sentinel present → healthy.
    w._tick_once()
    snap1 = w.latest()
    assert snap1 is not None and snap1.hosts["localhost"].healthy is True

    # Remove sentinel — next probe will fail.
    sentinel.unlink()

    w._tick_once()
    w._tick_once()
    snap3 = w.latest()
    assert snap3 is not None
    assert snap3.hosts["localhost"].healthy is False
```

- [ ] **Step 16.2: Run only the e2e test (it should be skipped or pass)**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/asgard/test_heimdall_e2e.py -v
```

Expected: PASS where `ssh localhost` works, SKIP otherwise.

- [ ] **Step 16.3: Commit**

```bash
git add tools/odin/tests/asgard/test_heimdall_e2e.py
git commit -m "asgard: add slow-marked Heimdall end-to-end integration test"
```

---

## Final verification

- [ ] **Step F.1: Full test sweep**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/ -v
```

Expected: every test passes (or is appropriately skipped). No new warnings.

- [ ] **Step F.2: Pre-commit hooks**

```bash
./isaaclab.sh -f
```

Expected: clean. If pre-commit modifies files (formatting), `git add` the changes, re-run, and create one cleanup commit per the project's pre-commit workflow.

- [ ] **Step F.3: Add changelog fragment**

The fragment slug should be the worktree branch with `/` replaced by `-`:

```
source/isaaclab_tools/changelog.d/worktree-heimdall-design.minor.rst
```

(Filename suffix `.minor.rst` reflects the new feature surface; choose `.major.rst` only if the implementor concludes the schema bump or CLI flag warrants it.)

Content:

```rst
Added
^^^^^

* Added Heimdall, a periodic fleet-watcher daemon thread inside the Asgard
  dispatcher. Re-probes each Valkyrie every 5 minutes (cheap
  ``nvidia-smi -L`` probe, K=2 consecutive-failure flip gate), runs
  :func:`~tools.odin.asgard.recovery.recover_valkyrie_gpu` on healthy →
  unhealthy flips, and (on recovery failure) quarantines the host and
  re-queues its in-flight jobs. Workers emit a periodic ``heartbeat``
  state event for each in-flight job; jobs whose heartbeat is older
  than 180 s are flagged stale — classified as ``timeout`` (no retry)
  when the host is healthy or ``infrastructure`` (re-queued) when the
  host has flipped. Persists per-host health and recent activity to
  ``<dispatch_dir>/fleet.json``; rendered as a card next to the fleet
  table on the Valhalla dashboard. Disable via ``--no-heimdall``.

Changed
^^^^^^^

* Bumped ``dispatch.json`` schema to ``1.6`` (added
  ``last_heartbeat_at`` to :class:`JobEntry`). Same-major resume from
  ``1.5`` dispatches works without manual intervention.

* :func:`~tools.odin.asgard.reconcile.reconcile_orphans` now attempts
  one :func:`~tools.odin.asgard.recovery.recover_valkyrie_gpu` call
  before flipping a ``running`` job back to ``pending`` on dispatcher
  restart. Closes the case where a transiently-unreachable host
  caused unnecessary job re-queues at restart time.
```

- [ ] **Step F.4: Commit changelog**

```bash
git add source/isaaclab_tools/changelog.d/worktree-heimdall-design.minor.rst
git commit -m "changelog: add Heimdall + reconcile-recovery-first fragment"
```

- [ ] **Step F.5: Memory follow-ups (post-merge, manual)**

Outside this plan: after merge to `antoiner/feat/odin`, update these memory files:

- `project_odin_periodic_preflight.md` — mark "Heimdall lands the periodic re-preflight" and reference the merged commit.
- `project_odin.md` — under "Component naming reserved", move Heimdall from reserved-but-unused to landed.

These are user-facing memory updates; the plan does not modify them automatically.
