# Odin Dispatcher State-Tracking Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Centralize all `JobEntry` state transitions through a single helper backed by an explicit allowed-transition graph, fixing four observed state-tracking bugs in the Odin dispatcher.

**Architecture:** Add `JobEntry.transition_to(...)` + a class-level `_ALLOWED_TRANSITIONS` graph. Refactor every `j.status = "..."` site in `worker.py`, `reconcile.py`, and `runner.py` to use the helper. Add a defensive `_manifest_indicates_clean_completion` patch (Bug 1), a `running_substate` annotation surfaced through state events (Bug 2), and a serialization-time invariant tripwire on `dispatch.json`.

**Tech Stack:** Python 3.12, pytest (no plugins, run with `--noconftest -p no:cacheprovider`), pre-commit (`./isaaclab.sh -f`). Working directory `/home/antoiner/Documents/IsaacLab` on branch `antoiner/feat/odin`.

**Spec:** `docs/superpowers/specs/2026-05-05-odin-state-tracking-audit-design.md` (commit `f676a3c9494`).

---

## File Structure

**Modified files (existing):**
- `tools/odin/asgard/jobs.py` — add `_ALLOWED_TRANSITIONS`, `transition_to` method, and `running_substate` field on `JobEntry`.
- `tools/odin/asgard/worker.py` — replace 6 direct `status` writes (lines 807, 833, 846, 854, 1137, 1174) with `transition_to(...)` calls; replace gpu_lost recovery re-queue (lines 684-702) with explicit `transition_to("pending", add_preferred_not=...)`; emit a `finalizing` StateEvent before `rsync.pull`.
- `tools/odin/asgard/reconcile.py` — replace 8 status writes (lines 222, 226, 242, 252, 259, 269, 274, 287); tighten `_manifest_indicates_clean_completion` (line 96) to require both `startup` and `training` phases.
- `tools/odin/asgard/runner.py` — replace status writes in `_apply_state_event` (lines 216, 226, 237, 270) and the retry-flip / reset-in-flight / queue-builder sites (lines 226, 237, 306, 334, 340, 364, 438, 727); handle the new `finalizing` transition.
- `tools/odin/asgard/state.py` — add `_validate_job_entry_invariants` checker and call it from `write_dispatch_state` with an `ODIN_DISPATCH_STRICT_INVARIANTS` env-flag opt-out.
- `tools/odin/asgard/worker.py` (StateEvent definition) — add a `running_substate: str | None` field to the `StateEvent` dataclass for the `finalizing` transition.
- `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py` — add a "pulling bundle" badge when `running_substate == "pulling_bundle"`.
- `tools/odin/valhalla/dashboard/assets/style.css` — CSS for the new badge.

**New test files:**
- `tools/odin/tests/test_asgard_jobs_transition.py` — unit tests for `_ALLOWED_TRANSITIONS` and `transition_to` (per legal/illegal edge, per per-target field contract, self-loops).
- `tools/odin/tests/test_asgard_state_invariants.py` — unit tests for `_validate_job_entry_invariants` and the env-flag opt-out.
- `tools/odin/valhalla/dashboard/tests/test_tab_a_pulling_bundle_badge.py` — render test for the new dashboard badge.

**Modified test files (existing):**
- `tools/odin/tests/test_asgard_reconcile.py` — add Bug 1 regression test (manifest with only startup phase placeholder must not adopt as completed); add Bug 4 regression tests (every reconcile path that flips to terminal must set `ended_at`).
- `tools/odin/tests/test_asgard_worker.py` (or sibling) — add Bug 3 regression test (gpu_lost recovery flips status back to `pending`, clears `started_at`/`assigned_to`).
- `tools/odin/tests/test_asgard_integration.py` — add a loopback gpu_lost recovery integration test.

**Out of scope for this plan:** No changes outside `tools/odin/asgard/*`, `tools/odin/valhalla/dashboard/*`, and `tools/odin/tests/*`. The upstream `source/isaaclab*` tree is untouched.

---

## Conventions used throughout this plan

- **Test command:** `python3 -m pytest --noconftest -p no:cacheprovider <path>::<name> -v`. Always include `--noconftest -p no:cacheprovider` — the project's `conftest.py` does heavy Isaac-Sim-aware setup that isn't wanted for these unit tests.
- **Pre-commit:** `./isaaclab.sh -f` before every commit. If pre-commit modifies files (e.g., reformats), re-stage and re-run.
- **Commit messages:** Imperative mood, capitalized, ~50 chars subject, blank line, then body wrapping at 72 chars. No AI-coauthor lines.
- **TDD:** Every code-changing task starts with a failing test, runs it to confirm the failure mode, then implements, then re-runs to confirm pass.
- **Coverage tip for refactor sweeps:** After each batch, also run the full `tools/odin/tests/` directory once to catch incidental regressions.

---

# Batch 1 — Add the helper

Goal: ship `JobEntry.transition_to()` as a standalone, fully-tested unit *without* changing any existing call sites yet. Existing tests must still pass; new helper is dead code until Batch 2.

## Task 1: Add the `_ALLOWED_TRANSITIONS` graph + scaffolding test

**Files:**
- Modify: `tools/odin/asgard/jobs.py:79-104` (add class-level `_ALLOWED_TRANSITIONS` and the `running_substate` field)
- Create: `tools/odin/tests/test_asgard_jobs_transition.py`

- [ ] **Step 1: Write the scaffolding test**

Create `tools/odin/tests/test_asgard_jobs_transition.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the JobEntry.transition_to() helper and the allowed-transition graph."""

from __future__ import annotations

from tools.odin.asgard.jobs import JobEntry


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
    """The graph encodes exactly the spec §4.1 edges: 7 cross-state + completed→pending."""
    expected = {
        ("pending", "running"),
        ("pending", "failed"),
        ("running", "completed"),
        ("running", "failed"),
        ("running", "pending"),
        ("failed", "pending"),
        ("completed", "pending"),
    }
    actual = {
        (src, dst) for src, dsts in JobEntry._ALLOWED_TRANSITIONS.items() for dst in dsts
    }
    assert actual == expected
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_jobs_transition.py -v
```

Expected: FAIL with `AttributeError: type object 'JobEntry' has no attribute '_ALLOWED_TRANSITIONS'`.

- [ ] **Step 3: Add the graph + `running_substate` field on `JobEntry`**

Edit `tools/odin/asgard/jobs.py`. Locate the `@dataclass class JobEntry:` block (currently around line 79). Add the `running_substate` field after `ended_at` and the `_ALLOWED_TRANSITIONS` mapping after the dataclass-field block:

```python
    started_at: str | None = None
    ended_at: str | None = None
    # New: substate annotation while status == "running". Distinguishes
    # the active-training phase from finalization (rsync.pull). Renderers
    # use this to show a "pulling bundle" badge without changing status.
    running_substate: str | None = None  # "training" | "pulling_bundle" | None
    per_job_timeout_s: int | None = None

    # Allowed-transition graph. See spec §4.1. Self-loops are not listed
    # here — `transition_to` short-circuits same-state calls as no-ops
    # before consulting this map.
    _ALLOWED_TRANSITIONS: ClassVar[dict[str, frozenset[str]]] = {
        "pending":   frozenset({"running", "failed"}),
        "running":   frozenset({"completed", "failed", "pending"}),
        "failed":    frozenset({"pending"}),
        "completed": frozenset({"pending"}),
    }
```

Add `from typing import ClassVar` to the imports at the top of the file if not already present.

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_jobs_transition.py -v
```

Expected: PASS (1 test).

- [ ] **Step 5: Run the existing jobs/queue tests to confirm no regression**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_queue.py tools/odin/tests/test_asgard_state.py -v
```

Expected: all PASS. Adding a field with a default value should not break existing tests.

- [ ] **Step 6: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/jobs.py tools/odin/tests/test_asgard_jobs_transition.py
git commit -m "asgard jobs: add allowed-transition graph + running_substate field

Adds _ALLOWED_TRANSITIONS class-level mapping on JobEntry encoding
the seven legal cross-state edges from spec §4.1 (pending→running,
pending→failed, running→{completed,failed,pending}, failed→pending,
completed→pending). Adds the running_substate field that downstream
work will flip to 'pulling_bundle' during rsync finalization.

No call sites change yet — the graph is consulted only by the
forthcoming transition_to() helper."
```

---

## Task 2: Implement `transition_to` and its per-target-state field contract

**Files:**
- Modify: `tools/odin/asgard/jobs.py` (add `transition_to` method on JobEntry)
- Modify: `tools/odin/tests/test_asgard_jobs_transition.py` (extend with edge + contract tests)

- [ ] **Step 1: Write the failing tests for legal edges + per-target field contract**

Append to `tools/odin/tests/test_asgard_jobs_transition.py`:

```python
from tools.odin.asgard.jobs import FailureInfo


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


def test_running_to_failed_requires_failure_kwarg():
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_jobs_transition.py -v
```

Expected: 11 failures, all `AttributeError: 'JobEntry' object has no attribute 'transition_to'`. The graph test from Task 1 still passes.

- [ ] **Step 3: Implement `transition_to` on `JobEntry`**

Edit `tools/odin/asgard/jobs.py`. Add a helper for the timestamp at module level (or import the one already in `tools.odin.asgard.state`; check first to avoid a circular import — `state.py` imports `JobEntry` from `jobs.py`, so add the helper *here* in `jobs.py`):

```python
from datetime import datetime, timezone


def _utc_now_iso() -> str:
    """UTC timestamp in ``YYYY-MM-DDTHH:MM:SSZ`` form. Mirrors the
    runner's ``_utc_now_iso`` so both modules produce identical strings."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
```

Then add the method to `JobEntry`:

```python
    def transition_to(
        self,
        target: str,
        *,
        failure: "FailureInfo | None" = None,
        assigned_to: str | None = None,
        now: str | None = None,
        reset_attempts: bool = False,
        add_preferred_not: str | None = None,
    ) -> bool:
        """Transition this job to ``target`` per spec §4.2.

        Validates the (current, target) edge against
        :data:`_ALLOWED_TRANSITIONS`. Self-loops short-circuit as
        no-ops and return ``False``. Legal cross-state edges apply
        the per-target field contract and return ``True``. Illegal
        edges raise ``ValueError``.

        Args:
            target: One of ``"pending"`` | ``"running"`` | ``"completed"`` | ``"failed"``.
            failure: Required when ``target == "failed"``. Forbidden
                when ``target == "completed"``.
            assigned_to: Required when ``target == "running"``.
            now: ISO-8601 UTC timestamp; defaults to :func:`_utc_now_iso`.
            reset_attempts: Only honored when ``target == "pending"``.
                When True, zeros ``attempts``.
            add_preferred_not: Only honored when ``target == "pending"``.
                When set, adds the host to ``preferred_not``.

        Returns:
            ``True`` when a cross-state edge applied (fields mutated).
            ``False`` for self-loops (no mutation).

        Raises:
            ValueError: For illegal edges or contract violations.
        """
        # Self-loop short-circuit.
        if target == self.status:
            return False

        # Legality check.
        allowed = self._ALLOWED_TRANSITIONS.get(self.status, frozenset())
        if target not in allowed:
            raise ValueError(
                f"illegal transition {self.status!r} → {target!r} for run_id={self.run_id!r}; "
                f"allowed targets from {self.status!r}: {sorted(allowed)}"
            )

        # Contract checks.
        if target == "running" and assigned_to is None:
            raise ValueError(f"transition_to('running') requires assigned_to (run_id={self.run_id!r})")
        if target == "failed" and failure is None:
            raise ValueError(f"transition_to('failed') requires failure (run_id={self.run_id!r})")
        if target == "completed" and failure is not None:
            raise ValueError(
                f"transition_to('completed') must not pass failure (run_id={self.run_id!r})"
            )

        ts = now if now is not None else _utc_now_iso()

        # Apply per-target field contract.
        if target == "pending":
            self.status = "pending"
            self.started_at = None
            self.ended_at = None
            self.assigned_to = None
            self.failure = None
            self.running_substate = None
            if reset_attempts:
                self.attempts = 0
            if add_preferred_not is not None:
                # `preferred_not` is a set; copy-on-write to avoid mutating
                # any caller's reference accidentally shared.
                self.preferred_not = set(self.preferred_not) | {add_preferred_not}
        elif target == "running":
            self.status = "running"
            self.started_at = ts
            self.assigned_to = assigned_to
            self.ended_at = None
            self.failure = None
            self.running_substate = "training"
        elif target == "completed":
            self.status = "completed"
            self.ended_at = ts
            self.failure = None
            self.running_substate = None
        elif target == "failed":
            self.status = "failed"
            self.ended_at = ts
            self.failure = failure
            self.running_substate = None

        return True
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_jobs_transition.py -v
```

Expected: 12 PASS (graph test from Task 1 + 11 new tests).

- [ ] **Step 5: Run the wider asgard test suite to confirm no regression**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_queue.py tools/odin/tests/test_asgard_state.py tools/odin/tests/test_asgard_runner.py tools/odin/tests/test_asgard_worker.py tools/odin/tests/test_asgard_reconcile.py -v 2>&1 | tail -10
```

Expected: All PASS. The helper is dead code at this point — adding it cannot break anything.

- [ ] **Step 6: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/jobs.py tools/odin/tests/test_asgard_jobs_transition.py
git commit -m "asgard jobs: implement JobEntry.transition_to() helper

Centralizes JobEntry state mutations through a single helper that
validates (current → target) edges against _ALLOWED_TRANSITIONS,
enforces per-target field contracts (e.g., 'failed' requires
FailureInfo, 'running' requires assigned_to), and applies all
related field updates atomically. Self-loops short-circuit as
no-ops returning False. Illegal edges raise ValueError naming
the source/target states and the run_id.

This commit only adds the helper. Call sites are still using
direct status writes; Batch 2 sweeps them in."
```

---

## Task 3: Add illegal-edge tests

**Files:**
- Modify: `tools/odin/tests/test_asgard_jobs_transition.py`

- [ ] **Step 1: Write tests for illegal edges and contract-violation paths**

Append to `tools/odin/tests/test_asgard_jobs_transition.py`:

```python
import pytest


@pytest.mark.parametrize("src,dst", [
    ("completed", "running"),
    ("completed", "failed"),
    ("failed", "running"),
    ("failed", "completed"),
    ("pending", "completed"),  # must go through running
])
def test_illegal_edges_raise_value_error(src, dst):
    job = _job(status=src)
    with pytest.raises(ValueError, match=f"illegal transition {src!r} → {dst!r}"):
        job.transition_to(dst, failure=FailureInfo(kind="x", message="y"))


def test_running_to_running_is_self_loop_noop_not_illegal():
    """Self-loops are explicitly allowed (no-op); they don't go through the legality table."""
    job = _job(status="running", started_at="t0", assigned_to="v1")
    assert job.transition_to("running", assigned_to="v2") is False  # no exception


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
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_jobs_transition.py -v
```

Expected: All tests PASS (12 from before + 9 new = ~21).

- [ ] **Step 3: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/tests/test_asgard_jobs_transition.py
git commit -m "asgard jobs: cover illegal-edge + contract-violation paths

Parametrized test for the five most likely illegal edges (e.g.
completed → running, failed → completed) confirming each raises
ValueError with the source/target states named. Three explicit
contract-violation tests confirm running → failed without
FailureInfo, pending → running without assigned_to, and running →
completed with a (forbidden) failure all raise."
```

---

## Task 4: Wire `_emit_failed` to use the helper (proves integration)

**Files:**
- Modify: `tools/odin/asgard/worker.py:1172-1186` (only the body of `_emit_failed`)

- [ ] **Step 1: Read the current `_emit_failed` to confirm shape**

```bash
sed -n '1172,1186p' tools/odin/asgard/worker.py
```

Expected output: a method that sets `job.status = "failed"`, `job.failure = failure`, `job.ended_at = _utc_now_iso()`, then posts a StateEvent.

- [ ] **Step 2: Run the existing worker tests to capture baseline**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_worker.py tools/odin/tests/test_asgard_worker_submit.py tools/odin/tests/test_asgard_worker_poll.py tools/odin/tests/test_asgard_worker_cancel.py 2>&1 | tail -3
```

Expected: a count line like `N passed in M.MMs`. Note this for comparison.

- [ ] **Step 3: Refactor `_emit_failed` to use the helper**

In `tools/odin/asgard/worker.py`, replace the `_emit_failed` method body. The current body is:

```python
    def _emit_failed(self, job: JobEntry, failure: FailureInfo) -> None:
        """Stamp the job as ``failed`` and post the matching :class:`StateEvent`."""
        job.status = "failed"
        job.failure = failure
        job.ended_at = _utc_now_iso()
        self._state_chan.put(
            StateEvent(
                run_id=job.run_id,
                host=self.host.host,
                transition="failed",
                failure=failure,
                ended_at=job.ended_at,
            )
        )
```

Replace with:

```python
    def _emit_failed(self, job: JobEntry, failure: FailureInfo) -> None:
        """Stamp the job as ``failed`` and post the matching :class:`StateEvent`.

        Self-loops (already-failed jobs) short-circuit at the helper level
        and post no event — the worker should not be calling _emit_failed
        on a job that's already terminal, but the no-op safety net is
        cheap insurance.
        """
        if not job.transition_to("failed", failure=failure):
            return
        self._state_chan.put(
            StateEvent(
                run_id=job.run_id,
                host=self.host.host,
                transition="failed",
                failure=failure,
                ended_at=job.ended_at,
            )
        )
```

- [ ] **Step 4: Run worker tests to confirm no regression**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_worker.py tools/odin/tests/test_asgard_worker_submit.py tools/odin/tests/test_asgard_worker_poll.py tools/odin/tests/test_asgard_worker_cancel.py -v 2>&1 | tail -5
```

Expected: same number of tests passing as in Step 2.

- [ ] **Step 5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/worker.py
git commit -m "asgard worker: route _emit_failed through transition_to

First call site to use the new JobEntry.transition_to() helper.
Body shrinks from three direct field writes to one helper call;
the StateEvent post is unchanged. Adds a defensive self-loop
short-circuit (calling _emit_failed on an already-failed job is
a caller bug but the helper's no-op return prevents a duplicate
StateEvent on the runner's queue)."
```

---

# Batch 2 — Sweep all 25 call sites

Goal: replace every direct `JobEntry.status = "..."` write across `worker.py`, `reconcile.py`, and `runner.py` with `transition_to(...)`. Each task is a focused refactor of one file or one logical chunk; tests pass after each.

The line numbers below are from the spec catalog (§5.2), captured at commit `f676a3c9494`. After each commit the line numbers may shift; always re-grep for the pattern in question if a step looks misaligned.

## Task 5: Refactor `reconcile.py` — all 8 status writes

**Files:**
- Modify: `tools/odin/asgard/reconcile.py:222, 226, 242, 252, 259, 269, 274, 287`
- Test: `tools/odin/tests/test_asgard_reconcile.py` (existing tests must still pass; one Bug 4 regression test added at the end)

- [ ] **Step 1: Add Bug 4 regression test**

In `tools/odin/tests/test_asgard_reconcile.py`, find an existing test that exercises the `adopted_completed` path (search for `adopted_completed` or `_manifest_indicates_clean_completion`). Below it, add:

```python
def test_adopted_completed_stamps_ended_at(tmp_path):
    """Bug 4 regression: reconcile flips status to 'completed' but historically
    didn't stamp ended_at. After the transition_to refactor, ended_at must be
    set whenever status flips to a terminal value."""
    # Set up a fake fleet with one host + one running job + a clean remote manifest.
    # Reuse the existing test's scaffolding (fake SSH that returns a
    # both-phases-completed manifest, fake rsync that succeeds). After running
    # reconcile_orphans, the JobEntry should have status='completed' AND
    # ended_at != None.

    # ... [the implementer copies the scaffolding from the nearest existing
    #      adopted_completed test in this file and only changes the assertions]

    # Replace the existing assertions block with:
    assert job.status == "completed"
    assert job.ended_at is not None  # Bug 4: was None before transition_to refactor
```

If the existing test file has a "fake SSH + fake rsync that returns a clean manifest and runs reconcile_orphans" pattern, copy that test, rename it `test_adopted_completed_stamps_ended_at`, and just change the assertions to the two above. If no such test exists, build one from this skeleton:

```python
def test_adopted_completed_stamps_ended_at(tmp_path):
    from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
    from tools.odin.asgard.jobs import JobEntry
    from tools.odin.asgard.reconcile import reconcile_orphans
    from tools.odin.asgard.transport import SSHResult

    class _FakeSSH:
        def run(self, host, cmd, *, timeout_s=None, stdout_tee=None, pty=True):
            if "cat" in cmd and "manifest.json" in cmd:
                return SSHResult(
                    exit_code=0,
                    stdout='{"phases":{"startup":{"status":"completed","exit_code":0},"training":{"status":"completed","exit_code":0}}}',
                    stderr="",
                    duration_s=0.01,
                )
            return SSHResult(exit_code=0, stdout="", stderr="", duration_s=0.01)

    class _FakeRsync:
        def pull(self, host, remote, local):
            return SSHResult(exit_code=0, stdout="", stderr="", duration_s=0.01)

    host = ValkyrieConfig(host="v1", ssh_user="odin")
    fleet = Fleet(fleet_name="t", hosts=[host])
    job = JobEntry(
        run_id="r1",
        task_id="Isaac-Test-v0",
        framework="rsl_rl",
        backend="physx",
        num_envs=1,
        max_iterations=1,
        seed=42,
        bundle_dir_name="r1",
        status="running",
        assigned_to="v1",
        started_at="2026-05-05T12:00:00Z",
    )
    reconcile_orphans(
        fleet=fleet, jobs=[job], dispatch_dir=tmp_path,
        ssh=_FakeSSH(), rsync=_FakeRsync(), detached_mode=False, cancel_db=None,
    )
    assert job.status == "completed"
    assert job.ended_at is not None  # Bug 4 regression
```

- [ ] **Step 2: Run the new test to verify it fails on current code**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_reconcile.py::test_adopted_completed_stamps_ended_at -v
```

Expected: FAIL with `AssertionError: assert None is not None` (the current `j.status = "completed"` write doesn't touch `ended_at`).

- [ ] **Step 3: Replace all 8 reconcile status writes**

Edit `tools/odin/asgard/reconcile.py`. Find each of these lines (use `grep -n "j.status\|job.status" tools/odin/asgard/reconcile.py` to locate; spec line numbers may have shifted):

| Old | New |
|-----|-----|
| `j.status = "completed"` (adopted_completed paths, ~lines 222 and 242) | `j.transition_to("completed")` |
| `j.status = "failed"; j.failure = ...` (adopted_failed paths, ~lines 226 and 252) | `j.transition_to("failed", failure=<the existing FailureInfo expression>)` (then DELETE the `j.failure = ...` line that's now redundant) |
| `j.status = "pending"` (no-pidfile / re-pending paths, ~lines 259, 269, 274) | `j.transition_to("pending")` |
| `job.status = "failed"; job.failure = ...` (terminal failed at line 287) | `job.transition_to("failed", failure=<existing FailureInfo>)` (DELETE the failure assignment) |

For each edit, also remove any redundant trailing `j.ended_at = ...` line that follows the status write — the helper now sets `ended_at` on terminal transitions automatically.

- [ ] **Step 4: Run the regression test to verify it passes**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_reconcile.py::test_adopted_completed_stamps_ended_at -v
```

Expected: PASS.

- [ ] **Step 5: Run the full reconcile test file to confirm no other regression**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_reconcile.py -v 2>&1 | tail -5
```

Expected: all PASS.

- [ ] **Step 6: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/reconcile.py tools/odin/tests/test_asgard_reconcile.py
git commit -m "asgard reconcile: route 8 status writes through transition_to

All eight 'j.status = ...' sites in reconcile_orphans now use
JobEntry.transition_to(...), which atomically applies the per-
target field contract — terminal transitions get ended_at stamped,
'pending' transitions clear started_at/assigned_to/failure, etc.

Fixes Bug 4 ('completed-with-ended_at=None' on the reconcile-
adopted paths). Adds a regression test that walks reconcile_orphans
through an adopted_completed path and asserts ended_at is set."
```

---

## Task 6: Refactor `worker.py` synchronous-failure paths (lines 807, 833, 846)

**Files:**
- Modify: `tools/odin/asgard/worker.py:807, 833, 846` (three terminal-failed writes in `_finalize_terminal` and surrounding methods)

- [ ] **Step 1: Run worker tests to capture baseline pass count**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_worker.py tools/odin/tests/test_asgard_worker_submit.py tools/odin/tests/test_asgard_worker_poll.py 2>&1 | tail -3
```

Note the pass count — must match after refactor.

- [ ] **Step 2: Replace the three writes**

Edit `tools/odin/asgard/worker.py`. For each location, the pattern is:

Before:
```python
            job.status = "failed"
            job.failure = <some_failure_expr>
            job.ended_at = _utc_now_iso()
            self._state_chan.put(
                StateEvent(run_id=job.run_id, host=self.host.host, transition="failed", failure=<failure>)
            )
```

After:
```python
            failure = <some_failure_expr>
            if job.transition_to("failed", failure=failure):
                self._state_chan.put(
                    StateEvent(run_id=job.run_id, host=self.host.host, transition="failed", failure=failure)
                )
```

(Hoist the failure expression into a local variable to avoid evaluating it twice.) The `if job.transition_to(...)` guard makes the StateEvent post conditional on the transition actually firing — defensive; the no-op return only happens if a caller passes a job that's already in target state, which shouldn't happen on these paths but is safe to handle.

Apply this edit at the three line ranges around 807, 833, and 846 (re-grep `grep -n 'job.status = "failed"' tools/odin/asgard/worker.py` to locate after Task 5).

- [ ] **Step 3: Run worker tests**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_worker.py tools/odin/tests/test_asgard_worker_submit.py tools/odin/tests/test_asgard_worker_poll.py -v 2>&1 | tail -5
```

Expected: same pass count as Step 1.

- [ ] **Step 4: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/worker.py
git commit -m "asgard worker: route terminal-failed paths through transition_to

Three sites in _finalize_terminal (rsync-failure, bundle-validation-
failure, classify-remote-failure) now call JobEntry.transition_to(
'failed', failure=...) instead of writing status/failure/ended_at
by hand. The StateEvent post is gated on the transition firing
(no-op returns suppress duplicate events on already-terminal jobs)."
```

---

## Task 7: Refactor `worker.py` terminal-completed paths (lines 854, 1137)

**Files:**
- Modify: `tools/odin/asgard/worker.py:854, 1137`

- [ ] **Step 1: Replace both writes**

Edit `tools/odin/asgard/worker.py`. The pattern is:

Before:
```python
        job.status = "completed"
        job.ended_at = _utc_now_iso()
        self._state_chan.put(
            StateEvent(run_id=job.run_id, host=self.host.host, transition="completed", ended_at=job.ended_at)
        )
        self._fail_tracker.note_success()
```

After:
```python
        if job.transition_to("completed"):
            self._state_chan.put(
                StateEvent(run_id=job.run_id, host=self.host.host, transition="completed", ended_at=job.ended_at)
            )
            self._fail_tracker.note_success()
```

(Note: `note_success` moves under the guard; double-success notes on a no-op transition would skew the fail tracker.) Apply at both line 854 and line 1137 (re-grep `grep -n 'job.status = "completed"' tools/odin/asgard/worker.py` if line numbers shifted).

- [ ] **Step 2: Run worker tests**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_worker.py tools/odin/tests/test_asgard_worker_submit.py tools/odin/tests/test_asgard_worker_poll.py -v 2>&1 | tail -5
```

Expected: all PASS.

- [ ] **Step 3: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/worker.py
git commit -m "asgard worker: route terminal-completed paths through transition_to

Both happy-path completion sites in worker.py now call
transition_to('completed') instead of stamping status + ended_at
by hand. fail_tracker.note_success() and the StateEvent post move
under the transition guard so a no-op return doesn't double-count."
```

---

## Task 8: Refactor `worker.py` gpu_lost recovery — Bug 3 fix

**Files:**
- Modify: `tools/odin/asgard/worker.py:684-702` (the `_handle_synchronous_failure` `gpu_lost` branch)
- Test: `tools/odin/tests/test_asgard_worker_submit.py` (or sibling — pick the file that already has gpu_lost coverage; if none, create `test_asgard_worker_gpu_lost.py`)

This is the central Bug 3 fix described in spec §1.

- [ ] **Step 1: Find or create the gpu_lost test file**

```bash
grep -l "gpu_lost\|recover_valkyrie_gpu" tools/odin/tests/test_asgard_worker*.py 2>/dev/null
```

If a file is found, add the regression test there. Otherwise create `tools/odin/tests/test_asgard_worker_gpu_lost.py` with the standard header.

- [ ] **Step 2: Write the Bug 3 regression test**

Add to the chosen test file:

```python
def test_gpu_lost_recovery_flips_status_to_pending_clears_runtime_fields():
    """Bug 3 regression: when gpu_lost recovery succeeds, the worker re-queues
    the job. The JobEntry must have status='pending', no started_at,
    no assigned_to. attempts is preserved (worker bumps at submit, not here).

    Concrete scenario: Anymal-C-Nav seed43 in 20260430-110509 sat as
    'running started_at=11:34:16Z' indefinitely after the worker quickly
    recovered + put another job in the slot. Without the status flip the
    JobEntry was orphaned — neither running on any host nor visible as
    pending."""
    from unittest.mock import patch

    from tools.odin.asgard.fleet import ValkyrieConfig
    from tools.odin.asgard.jobs import FailureInfo, JobEntry
    from tools.odin.asgard.recovery import RecoveryResult
    from tools.odin.asgard.worker import ValkyrieWorker, WorkerOptions
    # Import any other bits you need based on the existing worker-test scaffolding

    job = JobEntry(
        run_id="r1",
        task_id="Isaac-Test-v0",
        framework="rsl_rl",
        backend="physx",
        num_envs=1,
        max_iterations=1,
        seed=42,
        bundle_dir_name="r1",
        status="running",
        assigned_to="v1",
        started_at="2026-05-05T12:00:00Z",
        attempts=1,
    )
    failure = FailureInfo(kind="gpu_lost", message="probe failed")

    # Build a worker with mocked ssh + recover_valkyrie_gpu that returns recovered=True.
    # Use whatever scaffolding the rest of test_asgard_worker*.py uses (a _FakeSSH,
    # an in-memory state channel, etc.). The relevant call is
    # worker._handle_synchronous_failure(job, failure).

    host = ValkyrieConfig(host="v1", ssh_user="odin")
    # ... build worker as the surrounding test file does ...

    fake_recovery = RecoveryResult(
        host="v1", container_name="isaac-lab-base",
        attempted=True, recovered=True, duration_s=1.0, message="ok",
    )
    with patch("tools.odin.asgard.worker.recover_valkyrie_gpu", return_value=fake_recovery):
        worker._handle_synchronous_failure(job, failure)

    assert job.status == "pending"
    assert job.started_at is None
    assert job.assigned_to is None
    assert job.failure is None
    assert job.attempts == 1  # preserved across the recovery cycle
```

(The implementer fills in the worker construction using whatever fixture pattern the surrounding test file already established; if none, the test_asgard_worker_submit.py file has a clean `_FakeSSH`-style scaffolding to copy.)

- [ ] **Step 3: Run the test to verify it fails on current code**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_worker_gpu_lost.py -v
```

Expected: FAIL with `AssertionError: assert 'running' == 'pending'` — current code never flips status back.

- [ ] **Step 4: Refactor the gpu_lost branch in `_handle_synchronous_failure`**

Edit `tools/odin/asgard/worker.py` around lines 684-703. The current code is:

```python
        elif failure.kind == "gpu_lost":
            rec = recover_valkyrie_gpu(self.host, ssh=self._ssh)
            if rec.recovered:
                self._state_chan.put(StateEvent(run_id=job.run_id, host=self.host.host, transition="recovered"))
                if job.attempts <= self._options.max_infrastructure_retries:
                    self._job_queue.put(job)
                    return
            else:
                self._state_chan.put(
                    StateEvent(
                        run_id=job.run_id,
                        host=self.host.host,
                        transition="host_down",
                        failure=failure,
                    )
                )
                job.preferred_not = set(job.preferred_not) | {self.host.host}
                self._job_queue.put(job)
                self._down_event.set()
                return
```

Replace with:

```python
        elif failure.kind == "gpu_lost":
            rec = recover_valkyrie_gpu(self.host, ssh=self._ssh)
            if rec.recovered:
                self._state_chan.put(StateEvent(run_id=job.run_id, host=self.host.host, transition="recovered"))
                if job.attempts <= self._options.max_infrastructure_retries:
                    # Bug 3 fix: explicitly flip the JobEntry back to pending
                    # before re-queueing. Without this, the entry stays as
                    # 'running' with stale started_at while sitting in the
                    # queue — operator sees a phantom 'running' row forever.
                    job.transition_to("pending")
                    self._job_queue.put(job)
                    return
            else:
                self._state_chan.put(
                    StateEvent(
                        run_id=job.run_id,
                        host=self.host.host,
                        transition="host_down",
                        failure=failure,
                    )
                )
                # Re-queue with this host on preferred_not so a different
                # worker picks it up. transition_to handles both flipping
                # status to pending and adding the host to preferred_not.
                job.transition_to("pending", add_preferred_not=self.host.host)
                self._job_queue.put(job)
                self._down_event.set()
                return
```

Note that the manual `job.preferred_not = set(...) | {self.host.host}` line on the host_down branch is now subsumed by `transition_to(..., add_preferred_not=...)` — delete it.

- [ ] **Step 5: Run the test to verify it passes**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_worker_gpu_lost.py -v
```

Expected: PASS.

- [ ] **Step 6: Run the wider worker test suite**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_worker.py tools/odin/tests/test_asgard_worker_submit.py tools/odin/tests/test_asgard_worker_poll.py tools/odin/tests/test_asgard_worker_cancel.py tools/odin/tests/test_asgard_worker_gpu_lost.py 2>&1 | tail -3
```

Expected: all PASS.

- [ ] **Step 7: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/worker.py tools/odin/tests/test_asgard_worker_gpu_lost.py
git commit -m "asgard worker: flip status to pending on gpu_lost recovery (Bug 3)

When the pre-submit nvidia-smi probe fails, the worker classifies
the failure as gpu_lost and calls recover_valkyrie_gpu. On
successful recovery the JobEntry is re-queued — but the previous
implementation left status='running' with a stale started_at and
assigned_to, so the entry sat in dispatch.json as a phantom
running row indefinitely if another job grabbed the host slot
before the queue cycled back.

Now both branches (recovered + re-queue, and host_down + re-queue
with preferred_not) call transition_to('pending'), which atomically
flips status, clears started_at/assigned_to/failure, and (in the
host_down case) appends the host to preferred_not.

Concrete victim: Anymal-C-Nav seed43 in 20260430-110509 sat as
'running started_at=11:34:16Z' for hours after the worker quickly
recovered + put seed44 in the slot."
```

---

## Task 9: Refactor `runner.py` `_apply_state_event`

**Files:**
- Modify: `tools/odin/asgard/runner.py:214-272` (the `running`, `completed`, `failed`, and `host_down` branches in `_apply_state_event`)

- [ ] **Step 1: Run runner tests for baseline**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_runner.py tools/odin/tests/test_asgard_runner_cancellations.py 2>&1 | tail -3
```

- [ ] **Step 2: Replace four branches in `_apply_state_event`**

Edit `tools/odin/asgard/runner.py`. The current code in `_apply_state_event` (around lines 214-272) writes status directly. Replace each branch:

`running` branch (~lines 214-218):

Before:
```python
    if ev.transition == "running":
        if j is not None:
            j.status = "running"
            j.started_at = ev.started_at
            j.assigned_to = ev.host
```

After:
```python
    if ev.transition == "running":
        if j is not None:
            # Worker has already incremented attempts at submit time
            # (worker.py:_submit_job). transition_to('running') does
            # not touch attempts — it only sets status, started_at,
            # assigned_to, and running_substate='training'.
            j.transition_to("running", assigned_to=ev.host, now=ev.started_at)
```

`completed` branch (~lines 224-227):

Before:
```python
    if ev.transition == "completed":
        if j is not None:
            j.status = "completed"
            j.ended_at = ev.ended_at
```

After:
```python
    if ev.transition == "completed":
        if j is not None:
            j.transition_to("completed", now=ev.ended_at)
```

`failed` branch (~lines 235-239):

Before:
```python
    if ev.transition == "failed":
        if j is not None:
            j.status = "failed"
            j.failure = ev.failure
            j.ended_at = ev.ended_at
```

After:
```python
    if ev.transition == "failed":
        if j is not None and ev.failure is not None:
            j.transition_to("failed", failure=ev.failure, now=ev.ended_at)
```

`host_down` branch (~lines 268-272):

Before:
```python
        # Worker re-queued the in-flight job; reset its dispatch.json row from
        # 'running' back to 'pending' so it (a) is eligible to be picked up by
        # another healthy worker, and (b) is caught by the post-dispatch sweep
        # if no host remains. Without this reset the job stays as 'running'
        # forever in the final state.
        for j in state.jobs:
            if j.run_id == ev.run_id and j.status == "running":
                j.status = "pending"
                j.assigned_to = None
                j.started_at = None
```

After:
```python
        # Worker re-queued the in-flight job; reset its dispatch.json row from
        # 'running' back to 'pending' so it (a) is eligible to be picked up by
        # another healthy worker, and (b) is caught by the post-dispatch sweep
        # if no host remains.
        for j in state.jobs:
            if j.run_id == ev.run_id and j.status == "running":
                j.transition_to("pending")
```

- [ ] **Step 3: Run runner tests to confirm no regression**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_runner.py tools/odin/tests/test_asgard_runner_cancellations.py -v 2>&1 | tail -5
```

Expected: same pass count as Step 1.

- [ ] **Step 4: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/runner.py
git commit -m "asgard runner: route _apply_state_event through transition_to

The four branches in _apply_state_event (running, completed,
failed, host_down) now call JobEntry.transition_to instead of
writing status + started_at/ended_at/assigned_to/failure by hand.

The 'running' transition sets running_substate='training' as a
side effect of the helper's per-target contract — Batch 3 will
flip it to 'pulling_bundle' on the new finalizing transition."
```

---

## Task 10: Refactor `runner.py` retry-flip + reset-in-flight + queue-builder sites

**Files:**
- Modify: `tools/odin/asgard/runner.py:226, 237, 270, 306, 334, 340, 364, 438, 727`

The non-`_apply_state_event` runner sites are scattered: retry-failed/retry-all-failed flips, reset_in_flight_to_pending, the dispatch-finalization loop, and the build_queue / preflight skip paths. All do the same partial-field-write pattern.

- [ ] **Step 1: Locate the remaining sites**

```bash
grep -n '\.status\s*=\s*"\(pending\|running\|completed\|failed\)"' tools/odin/asgard/runner.py
```

This should now show 8-9 sites (the four `_apply_state_event` ones from Task 9 are already converted). Skip any `f.status = ...` lines — those are `FleetEntry`, not `JobEntry`. Pay attention only to `j.status = ...` and `job.status = ...`.

- [ ] **Step 2: Replace each per the pattern**

Walk through each remaining site and apply the appropriate translation:

| Pattern | Replacement |
|---------|-------------|
| `j.status = "pending"` (no-context flip) | `j.transition_to("pending")` |
| `j.status = "pending"; j.failure = None; j.attempts = 0; j.assigned_to = None; j.started_at = None; j.ended_at = None` (the retry-all-failed bulk reset, ~line 340) | `j.transition_to("pending", reset_attempts=True)` |
| `j.status = "pending"; j.failure = None` (the retry-failed targeted flip, ~line 334) | `j.transition_to("pending")` (failure clears automatically; attempts intentionally preserved) |
| `j.status = "failed"; j.failure = <expr>` (skipped/newton_floor/etc., lines 237, 306, 438, 727) | `j.transition_to("failed", failure=<expr>, now=_utc_now_iso())` (the helper's default `now` works too — drop the kwarg if the existing line wasn't computing the timestamp explicitly) |
| `job.status = "pending"` (line 364, in `_consume_live_retries`) | `job.transition_to("pending")` (failure-history clear is part of the contract; remove any redundant `job.failure = None` immediately following) |

Drop any redundant per-field clears that immediately follow each replacement (e.g., `j.failure = None` after `j.transition_to("pending")` is now duplicative).

- [ ] **Step 3: Run the runner test suite**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_runner.py tools/odin/tests/test_asgard_runner_cancellations.py 2>&1 | tail -3
```

Expected: same pass count as Task 9 Step 1.

- [ ] **Step 4: Run the full asgard test suite as a wider sanity check**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/ -k "not slow and not loopback" 2>&1 | tail -3
```

Expected: all PASS.

- [ ] **Step 5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/runner.py
git commit -m "asgard runner: sweep remaining JobEntry status writes

retry-failed / retry-all-failed flips, reset_in_flight_to_pending,
the dispatch-finalization preflight-skip path, and the build_queue
newton_floor + preset_unsupported skip paths all now go through
JobEntry.transition_to. transition_to('pending', reset_attempts=
True) replaces the explicit five-field bulk reset on the
retry-all-failed path; the targeted retry-failed path keeps
attempts as before."
```

---

## Task 11: Audit pass — confirm zero remaining direct status writes

**Files:**
- Read-only audit across `tools/odin/asgard/*.py`.

- [ ] **Step 1: Grep for any remaining JobEntry status writes**

```bash
grep -rn 'j\.status\s*=\|job\.status\s*=' tools/odin/asgard/ 2>/dev/null
```

Expected: zero matches outside test files. The `f.status = ...` lines in runner.py are `FleetEntry` — those stay.

- [ ] **Step 2: If any matches found, replace per Task 10's pattern table and re-run tests**

If matches exist, this is a missed call site. Replace in place using `transition_to(...)`, run the full asgard test suite (`python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/ -k "not slow"`), and amend the previous commit (or make a follow-up commit).

If no matches: proceed.

- [ ] **Step 3: Run the integration test (loopback) to confirm end-to-end behavior**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_integration.py -k "not gpu_lost" 2>&1 | tail -3
```

Expected: all PASS (the gpu_lost loopback test is added in Batch 3; skip it for now).

- [ ] **Step 4: No commit needed if Step 1 was clean.** The audit pass is a checkpoint, not a code change.

---

# Batch 3 — Targeted Bug 1 + Bug 2 fixes

## Task 12: Tighten `_manifest_indicates_clean_completion` (Bug 1)

**Files:**
- Modify: `tools/odin/asgard/reconcile.py:96-100`
- Test: `tools/odin/tests/test_asgard_reconcile.py`

- [ ] **Step 1: Write the Bug 1 regression test**

Add to `tools/odin/tests/test_asgard_reconcile.py`:

```python
def test_manifest_indicates_clean_completion_rejects_startup_only_placeholder():
    """Bug 1 regression: hugin/run.py:138 stamps a placeholder
    phases.startup={status: completed} BEFORE startup actually runs. If
    reconcile reads the manifest at that moment, every present phase looks
    healthy and the run gets adopted as completed even though training
    never even started.

    The fix: require BOTH 'startup' and 'training' phases to be present
    AND completed AND exit=0. A startup-only manifest cannot be trusted."""
    from tools.odin.asgard.reconcile import _manifest_indicates_clean_completion

    placeholder = {"phases": {"startup": {"status": "completed", "exit_code": 0}}}
    assert _manifest_indicates_clean_completion(placeholder) is False


def test_manifest_indicates_clean_completion_accepts_both_phases_completed():
    """Sanity: the legitimate clean-completion manifest still adopts."""
    from tools.odin.asgard.reconcile import _manifest_indicates_clean_completion

    healthy = {
        "phases": {
            "startup": {"status": "completed", "exit_code": 0},
            "training": {"status": "completed", "exit_code": 0},
        }
    }
    assert _manifest_indicates_clean_completion(healthy) is True


def test_manifest_indicates_clean_completion_rejects_training_failed():
    """Defensive: training phase failed must reject."""
    from tools.odin.asgard.reconcile import _manifest_indicates_clean_completion

    crashed = {
        "phases": {
            "startup": {"status": "completed", "exit_code": 0},
            "training": {"status": "failed", "exit_code": 1},
        }
    }
    assert _manifest_indicates_clean_completion(crashed) is False


def test_manifest_indicates_clean_completion_rejects_empty_phases():
    """Defensive: empty phases dict must reject."""
    from tools.odin.asgard.reconcile import _manifest_indicates_clean_completion

    assert _manifest_indicates_clean_completion({"phases": {}}) is False
    assert _manifest_indicates_clean_completion({}) is False
```

- [ ] **Step 2: Run tests to verify the placeholder test fails**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_reconcile.py::test_manifest_indicates_clean_completion_rejects_startup_only_placeholder -v
```

Expected: FAIL — current code returns `True` for the startup-only manifest.

- [ ] **Step 3: Replace `_manifest_indicates_clean_completion`**

Edit `tools/odin/asgard/reconcile.py:96-100`. Replace:

```python
def _manifest_indicates_clean_completion(manifest: dict) -> bool:
    phases = manifest.get("phases", {})
    if not phases:
        return False
    return all(phase.get("status") == "completed" and phase.get("exit_code") == 0 for phase in phases.values())
```

With:

```python
def _manifest_indicates_clean_completion(manifest: dict) -> bool:
    """Return True iff the manifest reports a fully-clean run.

    Hugin's manifest writer stamps a placeholder ``phases.startup =
    {status: "completed", exit_code: 0}`` before startup actually runs
    (`hugin/run.py:138`). A reconcile pass that reads the manifest at
    that instant would see every *present* phase looking healthy and
    erroneously adopt the run as completed (Bug 1 in the
    state-tracking audit). To prevent that, require BOTH ``startup``
    AND ``training`` phases to be present AND completed AND exit=0.
    A manifest with only the startup placeholder is rejected."""
    phases = manifest.get("phases", {})
    required = {"startup", "training"}
    if not required.issubset(phases.keys()):
        return False
    return all(
        phases[k].get("status") == "completed" and phases[k].get("exit_code") == 0
        for k in required
    )
```

- [ ] **Step 4: Run tests to verify all four pass**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_reconcile.py -k "manifest_indicates_clean" -v
```

Expected: 4 PASS.

- [ ] **Step 5: Run the full reconcile + integration tests**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_reconcile.py tools/odin/tests/test_asgard_integration.py -k "not gpu_lost" 2>&1 | tail -3
```

Expected: all PASS.

- [ ] **Step 6: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/reconcile.py tools/odin/tests/test_asgard_reconcile.py
git commit -m "asgard reconcile: require both phases for clean-completion (Bug 1)

_manifest_indicates_clean_completion now requires both 'startup'
and 'training' phases to be present AND completed AND exit=0.
Previously, any non-empty phases dict where every present phase
looked healthy was adopted as completed — which mistakenly
accepted Hugin's pre-startup phases.startup={status:'completed'}
placeholder when reconcile happened to read mid-write.

Concrete victims: Ant44 + Quadcopter44 in 20260505-095154, both
with on-disk manifest showing both phases failed exit=1 but
dispatch.json stamped completed because reconcile read the
placeholder before Hugin overwrote with the failure result."
```

---

## Task 13: Add `running_substate` to the `StateEvent` dataclass

**Files:**
- Modify: `tools/odin/asgard/worker.py` (the `StateEvent` dataclass definition)

- [ ] **Step 1: Locate the `StateEvent` definition**

```bash
grep -n "class StateEvent" tools/odin/asgard/worker.py
```

- [ ] **Step 2: Add the field**

In the `StateEvent` dataclass, add after the existing fields:

```python
    # Bug 2 (UX): annotation on the running state. When set, the runner
    # propagates this onto the matching JobEntry's running_substate. The
    # dashboard renders a "pulling bundle" badge while the value is
    # "pulling_bundle". Untouched on terminal transitions (transition_to
    # clears running_substate as part of its contract).
    running_substate: str | None = None
```

- [ ] **Step 3: Run the worker tests to confirm field addition is non-breaking**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_worker.py tools/odin/tests/test_asgard_worker_submit.py tools/odin/tests/test_asgard_worker_poll.py 2>&1 | tail -3
```

Expected: all PASS.

- [ ] **Step 4: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/worker.py
git commit -m "asgard worker: add running_substate field to StateEvent

Carries the substate annotation (training | pulling_bundle | None)
from the worker thread to the runner's main thread. Default None
keeps it backward-compatible with every existing emit site."
```

---

## Task 14: Worker emits `finalizing` StateEvent before rsync.pull

**Files:**
- Modify: `tools/odin/asgard/worker.py` — `_finalize_terminal` (around the rsync.pull call)
- Test: `tools/odin/tests/test_asgard_worker_poll.py` (or sibling)

- [ ] **Step 1: Locate `_finalize_terminal`'s rsync.pull**

```bash
grep -n "rsync.pull\|self._rsync.pull" tools/odin/asgard/worker.py
```

- [ ] **Step 2: Write a test confirming the finalizing event is emitted before rsync**

Add to `tools/odin/tests/test_asgard_worker_poll.py` (or wherever `_finalize_terminal` is exercised in the existing tests):

```python
def test_finalize_terminal_emits_finalizing_event_before_rsync():
    """Bug 2 fix: when a job is done, the worker emits a 'finalizing'
    StateEvent with running_substate='pulling_bundle' BEFORE invoking
    rsync.pull, so the dashboard can show a 'pulling bundle' badge
    instead of leaving the job's pill stuck at 'running' for the duration
    of a slow transfer."""
    # Build a worker with mocks; drive _finalize_terminal POLL_DONE.
    # Capture all events posted to self._state_chan in order.
    # Assert: the FIRST event after entering _finalize_terminal is
    # transition='finalizing', running_substate='pulling_bundle'.
    # The SECOND event (after the fake rsync resolves) is
    # transition='completed'.

    # ... test scaffolding using existing _FakeSSH / _FakeRsync / channel
    #     pattern from this file ...

    events = list(captured_state_chan)  # whatever the existing helper exposes
    assert events[0].transition == "finalizing"
    assert events[0].running_substate == "pulling_bundle"
    assert events[1].transition == "completed"
```

(The implementer fills in the worker-construction scaffolding using whatever pattern is already established in the file — `_FakeSSH`, `_FakeRsync`, an in-memory `queue.Queue` for the state channel.)

- [ ] **Step 3: Run the test to verify it fails**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_worker_poll.py::test_finalize_terminal_emits_finalizing_event_before_rsync -v
```

Expected: FAIL — no `finalizing` event today.

- [ ] **Step 4: Insert the emit before rsync.pull**

Edit `tools/odin/asgard/worker.py` `_finalize_terminal`. Find both `self._rsync.pull(...)` invocations (one for `POLL_DONE`, one for `POLL_EXITED_NO_MANIFEST`). Immediately before each call, add:

```python
        # Bug 2: announce the finalize phase so the dashboard can render
        # a "pulling bundle" badge instead of leaving the row stuck at
        # 'running' while the rsync runs (~1h on slow ARM tiers for a
        # 360 MB Camera bundle).
        self._state_chan.put(
            StateEvent(
                run_id=job.run_id,
                host=self.host.host,
                transition="finalizing",
                running_substate="pulling_bundle",
            )
        )
```

- [ ] **Step 5: Run the test to confirm it passes**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_worker_poll.py::test_finalize_terminal_emits_finalizing_event_before_rsync -v
```

Expected: PASS.

- [ ] **Step 6: Run the full worker test suite**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_worker.py tools/odin/tests/test_asgard_worker_submit.py tools/odin/tests/test_asgard_worker_poll.py tools/odin/tests/test_asgard_worker_cancel.py tools/odin/tests/test_asgard_worker_gpu_lost.py 2>&1 | tail -3
```

Expected: all PASS.

- [ ] **Step 7: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/worker.py tools/odin/tests/test_asgard_worker_poll.py
git commit -m "asgard worker: emit finalizing StateEvent before rsync.pull

Bug 2 UX fix: when _finalize_terminal kicks off the bundle pull,
emit a finalizing event carrying running_substate='pulling_bundle'.
The runner's _apply_state_event (next task) flips the JobEntry's
running_substate so the dashboard can render a 'pulling bundle'
badge instead of leaving the row apparently 'running' for the
duration of the rsync (~1h on slow ARM tiers for a 360 MB bundle).

The truth model is unchanged — status stays 'running' until the
bundle pull + manifest validation succeed. The substate is purely
an annotation that downstream renderers can use."
```

---

## Task 15: Runner `_apply_state_event` handles `finalizing` transition

**Files:**
- Modify: `tools/odin/asgard/runner.py` — `_apply_state_event`

- [ ] **Step 1: Add the finalizing branch**

Edit `tools/odin/asgard/runner.py`. In `_apply_state_event`, add a branch for the new transition kind. Place it after the `running` branch (before `completed`):

```python
    if ev.transition == "finalizing":
        if j is not None:
            # Annotation on running state — does not change status. The
            # dashboard reads running_substate to show "pulling bundle".
            j.running_substate = ev.running_substate
        return 0
```

- [ ] **Step 2: Write a test for the runner branch**

Add to `tools/odin/tests/test_asgard_runner.py`:

```python
def test_apply_state_event_finalizing_sets_running_substate():
    """The 'finalizing' transition only updates running_substate; status
    stays 'running' (the truth model is unchanged — bundle isn't actually
    pulled yet)."""
    from tools.odin.asgard.jobs import JobEntry
    from tools.odin.asgard.runner import _apply_state_event
    from tools.odin.asgard.state import DispatchState
    from tools.odin.asgard.worker import StateEvent

    job = JobEntry(
        run_id="r1", task_id="t", framework="rsl_rl", backend="physx",
        num_envs=1, max_iterations=1, seed=42, bundle_dir_name="r1",
        status="running", assigned_to="v1", started_at="t0",
        running_substate="training",
    )
    state = DispatchState(
        schema_version="1.4", dispatch_id="d", started_at="t0",
        ended_at=None, jobs=[job], skipped=[], fleet=[], quarantined_hosts=[],
    )
    ev = StateEvent(
        run_id="r1", host="v1", transition="finalizing",
        running_substate="pulling_bundle",
    )

    delta = _apply_state_event(state, ev)
    assert delta == 0  # finalizing doesn't advance the remaining counter
    assert job.status == "running"  # truth model unchanged
    assert job.running_substate == "pulling_bundle"
```

- [ ] **Step 3: Run the test**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_runner.py::test_apply_state_event_finalizing_sets_running_substate -v
```

Expected: PASS.

- [ ] **Step 4: Run the wider runner suite**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_runner.py tools/odin/tests/test_asgard_runner_cancellations.py 2>&1 | tail -3
```

Expected: all PASS.

- [ ] **Step 5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/runner.py tools/odin/tests/test_asgard_runner.py
git commit -m "asgard runner: handle finalizing transition

_apply_state_event now applies running_substate from a finalizing
StateEvent. The finalizing edge does not change job.status — the
job is still authoritatively running until the bundle pull + manifest
validation succeed. running_substate is purely a render annotation
that the dashboard surfaces as a 'pulling bundle' badge."
```

---

## Task 16: Dashboard renders `pulling bundle` badge

**Files:**
- Modify: `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py`
- Modify: `tools/odin/valhalla/dashboard/assets/style.css`
- Create: `tools/odin/valhalla/dashboard/tests/test_tab_a_pulling_bundle_badge.py`

- [ ] **Step 1: Locate the running-row render block**

```bash
grep -n "running\|status_children\|tab-a-running-tail-toggle" tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py | head -20
```

The existing running-row block (the one with the 👁 toggle and the ✕ Kill button from earlier in the session) is the right insertion point.

- [ ] **Step 2: Write the badge render test**

Create `tools/odin/valhalla/dashboard/tests/test_tab_a_pulling_bundle_badge.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tab A render test for the running_substate='pulling_bundle' badge."""

from __future__ import annotations

from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.jobs_table import (
    render_jobs_section,
)


def _payload(jobs):
    return {"dispatch_id": "20260505-095154", "ended_at": None, "jobs": jobs}


def _job(running_substate=None):
    return {
        "run_id": "r1",
        "task_id": "t",
        "framework": "rsl_rl",
        "backend": "physx",
        "seed": 42,
        "status": "running",
        "assigned_to": "v1",
        "running_substate": running_substate,
    }


def _walk(node):
    yield node
    children = getattr(node, "children", None)
    if children is None:
        return
    if isinstance(children, list):
        for child in children:
            yield from _walk(child)
    else:
        yield from _walk(children)


def test_running_row_with_pulling_bundle_substate_renders_badge():
    section = render_jobs_section(_payload([_job(running_substate="pulling_bundle")]))
    badges = [n for n in _walk(section)
              if getattr(n, "className", "") == "tab-a-pulling-bundle-badge"]
    assert len(badges) == 1
    assert "pulling bundle" in (badges[0].children or "").lower()


def test_running_row_without_substate_does_not_render_badge():
    """Default substate (training) → no badge shown."""
    section = render_jobs_section(_payload([_job(running_substate=None)]))
    badges = [n for n in _walk(section)
              if getattr(n, "className", "") == "tab-a-pulling-bundle-badge"]
    assert badges == []


def test_running_row_with_training_substate_does_not_render_badge():
    """Explicit 'training' substate → no badge (only pulling_bundle gets one)."""
    section = render_jobs_section(_payload([_job(running_substate="training")]))
    badges = [n for n in _walk(section)
              if getattr(n, "className", "") == "tab-a-pulling-bundle-badge"]
    assert badges == []
```

- [ ] **Step 3: Run the test to confirm it fails**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/valhalla/dashboard/tests/test_tab_a_pulling_bundle_badge.py -v
```

Expected: FAIL — `tab-a-pulling-bundle-badge` doesn't exist yet.

- [ ] **Step 4: Add the badge in `jobs_table.py`**

Edit `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py`. Find the running-row branch (where the 👁 toggle and ✕ Kill button are appended to `status_children`). Add the following just after the status pill is appended:

```python
    if status == "running":
        substate = job.get("running_substate")
        if substate == "pulling_bundle":
            status_children.append(
                html.Span(
                    "pulling bundle",
                    className="tab-a-pulling-bundle-badge",
                    title="Worker is rsync-pulling the bundle from the host. The trainer has finished; the dispatcher is waiting on the transfer to complete before flipping to 'completed'.",
                )
            )
```

- [ ] **Step 5: Add the CSS**

Edit `tools/odin/valhalla/dashboard/assets/style.css`. Add (near the existing `tab-a-cancel-pending-badge` definition for visual consistency):

```css
.tab-a-pulling-bundle-badge {
  display: inline-block;
  padding: 1px 6px;
  border-radius: 3px;
  margin-left: 6px;
  font-size: 11px;
  font-style: italic;
  background: rgba(118, 185, 0, 0.10);
  border: 1px solid rgba(118, 185, 0, 0.45);
  color: #76b900;
}
```

- [ ] **Step 6: Run the badge test**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/valhalla/dashboard/tests/test_tab_a_pulling_bundle_badge.py -v
```

Expected: 3 PASS.

- [ ] **Step 7: Run the wider dashboard test suite**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/valhalla/dashboard/tests/ 2>&1 | tail -3
```

Expected: all PASS (previous Tab A tests should still pass — we only added a new optional render branch).

- [ ] **Step 8: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py tools/odin/valhalla/dashboard/assets/style.css tools/odin/valhalla/dashboard/tests/test_tab_a_pulling_bundle_badge.py
git commit -m "dashboard: render 'pulling bundle' badge on Tab A running rows

When running_substate='pulling_bundle' the running-row gets a
green-tinted 'pulling bundle' italic badge next to the status pill.
Tooltip explains the trainer has finished and we're waiting on the
rsync; the row stays at status='running' (truth model unchanged).

Closes the visual half of Bug 2 — the back-end rsync transport
hardening lives in project_odin_rsync_no_timeout.md and is filed
separately."
```

---

# Batch 4 — Strict-invariants tripwire

Goal: catch any future drift where a code path slips past `transition_to` and leaves a `JobEntry` inconsistent. Strict mode raises on serialization; lenient mode auto-repairs and logs.

## Task 17: Add `_validate_job_entry_invariants` + tests

**Files:**
- Modify: `tools/odin/asgard/state.py` (add the validator function)
- Create: `tools/odin/tests/test_asgard_state_invariants.py`

- [ ] **Step 1: Write the failing tests**

Create `tools/odin/tests/test_asgard_state_invariants.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the JobEntry-invariant tripwire in state.py."""

from __future__ import annotations

import os

import pytest

from tools.odin.asgard.jobs import FailureInfo, JobEntry
from tools.odin.asgard.state import _validate_job_entry_invariants


def _job(**overrides) -> JobEntry:
    defaults = dict(
        run_id="r1", task_id="t", framework="rsl_rl", backend="physx",
        num_envs=1, max_iterations=1, seed=42, bundle_dir_name="r1",
    )
    defaults.update(overrides)
    return JobEntry(**defaults)


# Strict mode tests (env var unset or set to True).

def test_completed_without_ended_at_raises_in_strict_mode(monkeypatch):
    monkeypatch.delenv("ODIN_DISPATCH_STRICT_INVARIANTS", raising=False)
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
    """Healthy completed job — no exception."""
    job = _job(status="completed", started_at="t0", assigned_to="v1", ended_at="t1")
    _validate_job_entry_invariants(job, strict=True)  # no raise

    job2 = _job(status="failed", started_at="t0", assigned_to="v1", ended_at="t1",
                failure=FailureInfo(kind="x", message="y"))
    _validate_job_entry_invariants(job2, strict=True)


# Lenient mode tests (auto-repair + log).

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
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_state_invariants.py -v
```

Expected: FAIL with `ImportError: cannot import name '_validate_job_entry_invariants'`.

- [ ] **Step 3: Implement `_validate_job_entry_invariants`**

Edit `tools/odin/asgard/state.py`. Add at module level (near the other private helpers):

```python
import logging

from tools.odin.asgard.jobs import FailureInfo, JobEntry

_log = logging.getLogger(__name__)


def _validate_job_entry_invariants(job: JobEntry, strict: bool) -> None:
    """Validate the field invariants implied by ``job.status`` (spec §4.3).

    Strict mode raises ``AssertionError`` on the first violation, naming
    the run_id and the offending field. Lenient mode auto-repairs the
    JobEntry in place (filling missing timestamps with
    :func:`_utc_now_iso`, replacing missing ``failure`` with a stub) and
    logs a WARN for each repair.

    Used by :func:`write_dispatch_state` immediately before serializing
    to ``dispatch.json``."""
    violations: list[str] = []

    if job.status in {"completed", "failed"}:
        if job.ended_at is None:
            violations.append("ended_at")
        if job.status == "failed" and job.failure is None:
            violations.append("failure")
    elif job.status == "running":
        if job.started_at is None:
            violations.append("started_at")
        if job.assigned_to is None:
            violations.append("assigned_to")
    elif job.status == "pending":
        if job.started_at is not None:
            violations.append("started_at")
        if job.ended_at is not None:
            violations.append("ended_at")
        if job.assigned_to is not None:
            violations.append("assigned_to")
        if job.failure is not None:
            violations.append("failure")

    if not violations:
        return

    if strict:
        raise AssertionError(
            f"JobEntry invariant violation for run_id={job.run_id!r} "
            f"in status={job.status!r}: {violations}"
        )

    # Lenient mode: auto-repair.
    for field in violations:
        if job.status in {"completed", "failed"} and field == "ended_at":
            job.ended_at = _utc_now_iso()
        elif job.status == "failed" and field == "failure":
            job.failure = FailureInfo(
                kind="unknown",
                message="state-write invariant violation; see logs",
            )
        elif job.status == "pending":
            setattr(job, field, None)
        # 'running' violations cannot be auto-repaired safely (we don't
        # know what host or timestamp to fill in); fall through to log.
        _log.warning(
            "auto-repaired JobEntry invariant: run_id=%s status=%s field=%s",
            job.run_id, job.status, field,
        )
```

- [ ] **Step 4: Run the tests to confirm pass**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_state_invariants.py -v
```

Expected: 10 PASS.

- [ ] **Step 5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/state.py tools/odin/tests/test_asgard_state_invariants.py
git commit -m "asgard state: add _validate_job_entry_invariants tripwire

Pure validator function — strict mode raises AssertionError on
missing ended_at/failure/started_at/assigned_to per JobEntry.status,
lenient mode auto-repairs and logs WARN. Not yet wired into
write_dispatch_state; the next task does that."
```

---

## Task 18: Wire the validator into `write_dispatch_state` with env-flag opt-out

**Files:**
- Modify: `tools/odin/asgard/state.py` — `write_dispatch_state`
- Modify: `tools/odin/tests/test_asgard_state_invariants.py` — end-to-end tests

- [ ] **Step 1: Locate `write_dispatch_state`**

```bash
grep -n "def write_dispatch_state" tools/odin/asgard/state.py
```

- [ ] **Step 2: Write the integration test**

Append to `tools/odin/tests/test_asgard_state_invariants.py`:

```python
def test_write_dispatch_state_strict_raises_on_inconsistent_job(tmp_path, monkeypatch):
    """End-to-end: a corrupt JobEntry in DispatchState.jobs should make
    write_dispatch_state raise AssertionError when strict mode is on."""
    monkeypatch.setenv("ODIN_DISPATCH_STRICT_INVARIANTS", "true")

    from tools.odin.asgard.state import DispatchState, write_dispatch_state

    bad_job = _job(status="completed", ended_at=None)  # missing ended_at
    state = DispatchState(
        schema_version="1.4", dispatch_id="d", started_at="t0",
        ended_at=None, jobs=[bad_job], skipped=[], fleet=[], quarantined_hosts=[],
    )
    with pytest.raises(AssertionError, match=r"completed.*ended_at"):
        write_dispatch_state(tmp_path, state)


def test_write_dispatch_state_lenient_auto_repairs(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("ODIN_DISPATCH_STRICT_INVARIANTS", "false")

    from tools.odin.asgard.state import DispatchState, write_dispatch_state

    bad_job = _job(status="completed", ended_at=None)
    state = DispatchState(
        schema_version="1.4", dispatch_id="d", started_at="t0",
        ended_at=None, jobs=[bad_job], skipped=[], fleet=[], quarantined_hosts=[],
    )
    write_dispatch_state(tmp_path, state)

    # Repaired in place + persisted to disk.
    assert bad_job.ended_at is not None
    on_disk = (tmp_path / "dispatch.json").read_text()
    assert '"ended_at": null' not in on_disk
    assert any("auto-repaired" in rec.message for rec in caplog.records)


def test_write_dispatch_state_strict_default_when_env_unset(tmp_path, monkeypatch):
    """Default behavior: strict mode applies when ODIN_DISPATCH_STRICT_INVARIANTS is unset."""
    monkeypatch.delenv("ODIN_DISPATCH_STRICT_INVARIANTS", raising=False)

    from tools.odin.asgard.state import DispatchState, write_dispatch_state

    bad_job = _job(status="completed", ended_at=None)
    state = DispatchState(
        schema_version="1.4", dispatch_id="d", started_at="t0",
        ended_at=None, jobs=[bad_job], skipped=[], fleet=[], quarantined_hosts=[],
    )
    with pytest.raises(AssertionError):
        write_dispatch_state(tmp_path, state)
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_state_invariants.py::test_write_dispatch_state_strict_raises_on_inconsistent_job -v
```

Expected: FAIL — `write_dispatch_state` doesn't validate today.

- [ ] **Step 4: Wire the validator into `write_dispatch_state`**

Edit `tools/odin/asgard/state.py`. At the top of `write_dispatch_state`, before the existing serialization logic, add:

```python
def write_dispatch_state(dispatch_dir: Path, state: DispatchState) -> None:
    """Serialize ``state`` to ``<dispatch_dir>/dispatch.json``.

    Validates each JobEntry's invariants before writing (spec §4.3).
    Strict mode (default; controlled by the
    ``ODIN_DISPATCH_STRICT_INVARIANTS`` env var) raises on the first
    inconsistent JobEntry. Lenient mode auto-repairs in place and
    logs a WARN for each repair.

    Strict mode is recommended for tests + dev; production deployments
    can set ``ODIN_DISPATCH_STRICT_INVARIANTS=false`` to keep the
    dispatcher running in the face of upstream state-handling bugs at
    the cost of a few seconds of stale dispatch.json output."""
    strict = os.environ.get("ODIN_DISPATCH_STRICT_INVARIANTS", "true").lower() != "false"
    for job in state.jobs:
        _validate_job_entry_invariants(job, strict=strict)

    # ... existing serialization code unchanged below ...
```

(Add `import os` at module level if not already present.)

- [ ] **Step 5: Run the integration tests**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_state_invariants.py -v
```

Expected: 13 PASS (10 from Task 17 + 3 new).

- [ ] **Step 6: Run the full asgard suite to confirm no production code path violates the new invariants**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/ -k "not slow and not loopback" 2>&1 | tail -5
```

Expected: all PASS. If anything fails with an `AssertionError: JobEntry invariant violation`, that's a bug in either Batch 2 (a missed call site) or in test fixtures that build JobEntries with inconsistent state — investigate and fix before continuing.

- [ ] **Step 7: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/state.py tools/odin/tests/test_asgard_state_invariants.py
git commit -m "asgard state: validate JobEntry invariants on write_dispatch_state

write_dispatch_state now runs _validate_job_entry_invariants on
every JobEntry before serializing dispatch.json. Strict mode
(default) raises on the first inconsistent entry; lenient mode
(ODIN_DISPATCH_STRICT_INVARIANTS=false) auto-repairs and logs.

This is the tripwire that catches any future drift where a code
path slips past JobEntry.transition_to and leaves an entry in an
inconsistent state. The test suite runs in strict mode by default;
production operators who want the safety-pad behavior can opt
into lenient mode via the env var."
```

---

## Task 19: End-to-end loopback regression test for gpu_lost recovery

**Files:**
- Modify: `tools/odin/tests/test_asgard_integration.py`

- [ ] **Step 1: Locate the existing loopback recovery test**

```bash
grep -n "test_loopback_dispatch_recovers_from_gpu_lost\|stub_ssh_runner_first_job_nvml" tools/odin/tests/test_asgard_integration.py
```

The existing `test_loopback_dispatch_recovers_from_gpu_lost` exercises the in-flight gpu_lost path (worker classifies stderr → recovery → second attempt). We extend it (or add a sibling) to cover the synchronous-failure variant (pre-submit nvidia-smi probe failure).

- [ ] **Step 2: Add the new integration test**

Append to `tools/odin/tests/test_asgard_integration.py`:

```python
def test_loopback_dispatch_recovers_from_synchronous_gpu_lost(
    tmp_path: Path,
    monkeypatch,
):
    """Bug 3 end-to-end regression: a pre-submit GPU probe failure
    triggers worker._handle_synchronous_failure(gpu_lost), recovery
    succeeds, and the job re-runs successfully on the second attempt
    with status='completed' in the final dispatch.json. Crucially:
    the JobEntry must NOT spend any time stuck at status='running'
    after the synchronous failure — the first call site that sees
    the JobEntry post-recovery must observe status='pending'."""
    if not _ssh_localhost_works():
        pytest.skip("ssh localhost does not work without a password; skipping integration test")

    from tools.odin.asgard import worker as worker_mod
    from tools.odin.asgard.recovery import RecoveryResult

    # First submit: synthesize a gpu_unavailable failure (the worker's
    # docker-exec command writes 'odin: gpu_unavailable: ...' to the
    # bundle's odin-submit-error.log and exits non-zero). Second submit:
    # materialise a clean bundle. The fixture pattern mirrors
    # stub_ssh_runner_first_job_nvml above.

    seen_per_host: dict[str, int] = {}
    real_build = worker_mod._build_docker_exec_cmd

    def _fake_build(host, job):
        n = seen_per_host.get(host.host, 0)
        seen_per_host[host.host] = n + 1
        if n == 0:
            # Force the synchronous probe path: emit the gpu_unavailable
            # marker on stderr and exit non-zero immediately.
            return (
                "echo 'odin: gpu_unavailable: nvidia-smi failed' 1>&2 && exit 1"
            )
        # Second attempt: write a healthy bundle.
        bundle_dir = f"{host.isaaclab_path}/odin_runs/{job.bundle_dir_name}"
        manifest = '{"schema_version":"1.0","phases":{"startup":{"status":"completed","exit_code":0},"training":{"status":"completed","exit_code":0}}}'
        return (
            f"mkdir -p {bundle_dir} && "
            f"printf '%s' '{manifest}' > {bundle_dir}/manifest.json && "
            f"printf '%s' '{{}}' > {bundle_dir}/training.json && "
            f"printf '%s' '{{}}' > {bundle_dir}/startup.json"
        )

    monkeypatch.setattr(worker_mod, "_build_docker_exec_cmd", _fake_build)

    # Mock recovery to succeed.
    monkeypatch.setattr(
        worker_mod, "recover_valkyrie_gpu",
        lambda host, *, ssh: RecoveryResult(
            host=host.host, container_name=host.container_name,
            attempted=True, recovered=True, duration_s=1.0, message="ok",
        ),
    )

    # ... build a single-row env list, host, dispatch — same scaffolding
    #     as test_loopback_dispatch_recovers_from_gpu_lost ...

    state = run_dispatch(...)  # mirror the existing test's call

    assert len(state.jobs) == 1
    assert state.jobs[0].status == "completed"
    assert state.jobs[0].ended_at is not None  # Bug 4 invariant
    assert state.jobs[0].attempts == 2  # one failed submit, one success
```

(The implementer mirrors the env-list / host / dispatch scaffolding from the existing `test_loopback_dispatch_recovers_from_gpu_lost` test directly above this one in the file.)

- [ ] **Step 3: Run the test (slow-marked; only runs when explicitly invoked)**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/test_asgard_integration.py::test_loopback_dispatch_recovers_from_synchronous_gpu_lost -v
```

Expected: PASS if `_ssh_localhost_works()` succeeds on the runner, otherwise SKIP.

- [ ] **Step 4: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/tests/test_asgard_integration.py
git commit -m "asgard integration: loopback test for synchronous gpu_lost recovery

End-to-end coverage of Bug 3: first submit hits the pre-submit
nvidia-smi probe failure (gpu_unavailable marker → gpu_lost
classification → recover_valkyrie_gpu success), second submit
materialises a clean bundle. Final dispatch.json must show
status='completed' with ended_at set and attempts=2.

Slow-marked at module level (pytestmark = pytest.mark.slow) like
the rest of the file, so it doesn't run in the default unit-test
sweep — only via explicit pytest invocation."
```

---

## Final pass: full test suite + branch sanity

- [ ] **Step 1: Run the full asgard test suite**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/tests/ -k "not slow" 2>&1 | tail -5
```

Expected: all PASS, ~150+ tests.

- [ ] **Step 2: Run the dashboard test suite**

```bash
python3 -m pytest --noconftest -p no:cacheprovider tools/odin/valhalla/dashboard/tests/ 2>&1 | tail -5
```

Expected: all PASS.

- [ ] **Step 3: Pre-commit final check**

```bash
./isaaclab.sh -f
```

Expected: all hooks pass; no files modified.

- [ ] **Step 4: Confirm zero remaining direct-status-write call sites**

```bash
grep -rn 'j\.status\s*=\s*"\|job\.status\s*=\s*"' tools/odin/asgard/ 2>/dev/null
```

Expected: zero matches outside test files.

- [ ] **Step 5: Walk the spec to verify coverage**

Open the spec at `docs/superpowers/specs/2026-05-05-odin-state-tracking-audit-design.md` and check off:

- §4.1 allowed-transition graph → Task 1
- §4.2 transition_to API → Task 2
- §4.3 strict-invariants tripwire → Tasks 17, 18
- §4.4 manifest tightening (Bug 1) → Task 12
- §4.5 running_substate (Bug 2) → Tasks 13–16
- §4.6 attempts handling → covered in Task 2's contract
- §5.2 call-site catalog (25 sites) → Tasks 4–11
- §6 tests (per-edge unit, regression for each bug, loopback) → Tasks 1–3, 5, 8, 12, 16, 19

If any spec section is uncovered, add a remediating task here.

This is a checkpoint, not a code change. No commit.
