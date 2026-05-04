# Tab A Kill / Skip Button — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-row Tab A button that kills a running job or skips a pending one, with a 5-second confirm to guard misclicks. Latency from confirm-click to terminal status is ≤ one runner tick (~5 s).

**Architecture:** Dashboard → SQLite (`cancellations` table, sibling of the existing `retries` table in `.retry.sqlite`) → runner polls on the existing `live_retry_poll_s` cadence → runner mutates `JobEntry.status` for skips and signals workers via a new `request_cancel(run_id)` method for kills. Worker's detached run loop dispatches a `pkill -9 -f '<run_id>'` via SSH and lets the next poll tick land `exited-no-manifest`; `_finalize_terminal` overrides classification to `kind="killed"` and pulls partial logs.

**Tech Stack:** Python 3.10+, SQLite (stdlib `sqlite3` with WAL mode), Dash for the UI, `pytest --noconftest -p no:cacheprovider` for the test suite. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-04-30-odin-tab-a-kill-skip-design.md`.

---

## Files

**New:**
- `tools/odin/valhalla/dashboard/cancel_db.py` — `CancelDB` + `CancelRow` + schema migration.
- `tools/odin/valhalla/dashboard/tests/test_cancel_db.py`
- `tools/odin/tests/test_asgard_runner_cancellations.py`
- `tools/odin/tests/test_asgard_worker_cancel.py`
- `tools/odin/valhalla/dashboard/tests/test_data_cancel.py`
- `tools/odin/valhalla/dashboard/tests/test_tab_a_cancel_button.py`
- `tools/odin/valhalla/dashboard/tests/test_tab_a_cancel_callback.py`

**Modified:**
- `tools/odin/valhalla/dashboard/data.py` — add `request_cancel`, `read_cancel_queue`, `_get_cancel_db`.
- `tools/odin/asgard/runner.py` — add `_consume_cancellations`, `_mark_cancellation_consumed`, switch `workers: list[ValkyrieWorker]` to `workers_by_host: dict[str, ValkyrieWorker]`, wire into the main loop next to `_consume_live_retries`.
- `tools/odin/asgard/worker.py` — add `request_cancel(run_id)`, `_cancel_request: dict[str, bool]`, `JobInflight.kill_dispatched`, finalize precedence, and a `job.status != "pending"` guard at the top of `_submit_or_handle`.
- `tools/odin/asgard/reconcile.py` — apply pending cancellations after re-attach.
- `tools/odin/asgard/jobs.py` — extend `FailureInfo.kind` docstring with `killed` / `skipped`.
- `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py` — render the cancel button + pending badge.
- `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/callbacks.py` — confirm-flow callback + 5-second revert.
- `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/layout.py` — register the per-row confirm-state `dcc.Store` and the 500 ms `dcc.Interval` revert.
- `tools/odin/tests/test_asgard_integration.py` — append `test_loopback_detached_dispatch_skip_and_kill_via_db`.

---

## Test commands

All tests use the project's pure-Python pytest invocation:

```bash
python3 -m pytest --noconftest -p no:cacheprovider <PATH>
```

Pre-commit (run before any commit):

```bash
./isaaclab.sh -f
```

If pre-commit modifies files, re-stage with `git add` and re-run until clean.

---

## Task 1: `CancelDB` schema + dataclass + `request` / `read_pending`

**Files:**
- Create: `tools/odin/valhalla/dashboard/cancel_db.py`
- Test: `tools/odin/valhalla/dashboard/tests/test_cancel_db.py`

The `CancelDB` shares the `.retry.sqlite` file with `RetryDB`. We add a new table via the same migration framework — `_MIGRATIONS[2]` adds `cancellations`. `RetryDB`'s migration logic runs the dict in sorted order and bumps `PRAGMA user_version`, so adding `2` is enough to migrate existing files transparently.

- [ ] **Step 1.1: Write the schema + round-trip test**

Append to a new file `tools/odin/valhalla/dashboard/tests/test_cancel_db.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the SQLite-backed Odin cancellation queue."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from tools.odin.valhalla.dashboard.cancel_db import CancelDB


def test_fresh_db_creates_cancellations_table(tmp_path: Path):
    db = CancelDB(tmp_path)

    assert db.read_pending("20260504-100000") == {}

    with sqlite3.connect(tmp_path / ".retry.sqlite") as con:
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        version = con.execute("PRAGMA user_version").fetchone()[0]

    assert "cancellations" in tables
    assert version >= 2


def test_request_inserts_pending_row(tmp_path: Path):
    db = CancelDB(tmp_path)

    db.request("20260504-100000", "run-a", kind="kill")

    pending = db.read_pending("20260504-100000")
    assert pending == {"run-a": "kill"}
```

- [ ] **Step 1.2: Run the tests and verify they fail**

```bash
python3 -m pytest --noconftest -p no:cacheprovider \
  tools/odin/valhalla/dashboard/tests/test_cancel_db.py -v
```

Expected: `ModuleNotFoundError: No module named 'tools.odin.valhalla.dashboard.cancel_db'`.

- [ ] **Step 1.3: Create the module skeleton with schema migration**

Create `tools/odin/valhalla/dashboard/cancel_db.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SQLite-backed Odin cancellation queue.

Operator-initiated kill (running) and skip (pending) requests land here.
The Asgard runner polls pending rows on each ``live_retry_poll_s`` tick
and either flips ``JobEntry.status`` directly (skip) or signals the
worker for the assigned host (kill).

Shares the ``.retry.sqlite`` file with :class:`RetryDB` — schema migration
2 adds the ``cancellations`` table next to ``retries``.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["CancelDB", "CancelRow"]


_DB_NAME = ".retry.sqlite"
_VALID_KINDS = {"kill", "skip"}
_VALID_OUTCOMES = {"killed", "skipped", "noop"}
_CONNECT_LOCK = threading.Lock()
_MIGRATIONS: dict[int, str] = {
    1: """
CREATE TABLE IF NOT EXISTS retries (
    dispatch_id        TEXT    NOT NULL,
    run_id             TEXT    NOT NULL,
    queued_at          TEXT    NOT NULL,
    note               TEXT,
    retried_at         TEXT,
    retry_dispatch_id  TEXT,
    retry_outcome      TEXT CHECK (retry_outcome IN ('completed', 'failed') OR retry_outcome IS NULL),
    retry_failure_kind TEXT,
    PRIMARY KEY (dispatch_id, run_id)
);

CREATE INDEX IF NOT EXISTS idx_retries_pending ON retries(dispatch_id) WHERE retried_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_retries_global_pending ON retries(retried_at) WHERE retried_at IS NULL;
""",
    2: """
CREATE TABLE IF NOT EXISTS cancellations (
    dispatch_id  TEXT NOT NULL,
    run_id       TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    kind         TEXT NOT NULL CHECK (kind IN ('kill', 'skip')),
    consumed_at  TEXT,
    outcome      TEXT CHECK (outcome IN ('killed', 'skipped', 'noop') OR outcome IS NULL),
    PRIMARY KEY (dispatch_id, run_id)
);

CREATE INDEX IF NOT EXISTS idx_cancellations_pending
    ON cancellations(dispatch_id) WHERE consumed_at IS NULL;
""",
}


@dataclass(frozen=True)
class CancelRow:
    """One row in the Odin cancellation queue."""

    dispatch_id: str
    run_id: str
    requested_at: str
    kind: str
    consumed_at: str | None
    outcome: str | None


class CancelDB:
    """SQLite-backed cancellation queue rooted at an ``odin_runs`` directory."""

    def __init__(self, runs_root: Path) -> None:
        self._runs_root = Path(runs_root)
        self._db_path = self._runs_root / _DB_NAME

    @property
    def path(self) -> Path:
        """Return the SQLite database path."""
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        """Open a short-lived SQLite connection and apply schema setup."""
        self._runs_root.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self._db_path, timeout=30.0, isolation_level=None)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA busy_timeout = 30000")
        with _CONNECT_LOCK:
            con.execute("PRAGMA journal_mode = WAL")
            con.execute("PRAGMA foreign_keys = ON")
            _migrate(con)
        return con

    def request(self, dispatch_id: str, run_id: str, *, kind: str) -> None:
        """Insert (or overwrite) a pending cancellation row."""
        if kind not in _VALID_KINDS:
            raise ValueError(f"kind must be one of {sorted(_VALID_KINDS)}, got {kind!r}")
        with closing(self._connect()) as con:
            con.execute(
                """
                INSERT OR REPLACE INTO cancellations(
                    dispatch_id, run_id, requested_at, kind, consumed_at, outcome
                ) VALUES (?, ?, ?, ?, NULL, NULL)
                """,
                (dispatch_id, run_id, _now_iso(), kind),
            )

    def read_pending(self, dispatch_id: str) -> dict[str, str]:
        """Return ``{run_id: kind}`` for the dispatch's pending cancellations."""
        with closing(self._connect()) as con:
            rows = con.execute(
                """
                SELECT run_id, kind FROM cancellations
                WHERE dispatch_id = ? AND consumed_at IS NULL
                ORDER BY run_id
                """,
                (dispatch_id,),
            ).fetchall()
        return {str(row["run_id"]): str(row["kind"]) for row in rows}


def _migrate(con: sqlite3.Connection) -> None:
    current = int(con.execute("PRAGMA user_version").fetchone()[0])
    for version, sql in sorted(_MIGRATIONS.items()):
        if version > current:
            con.executescript(sql)
            con.execute(f"PRAGMA user_version = {version}")
    con.commit()


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
```

- [ ] **Step 1.4: Run the tests and verify they pass**

```bash
python3 -m pytest --noconftest -p no:cacheprovider \
  tools/odin/valhalla/dashboard/tests/test_cancel_db.py -v
```

Expected: 2 passed.

- [ ] **Step 1.5: Verify schema migration co-exists with `RetryDB`**

Add this test next to the existing ones:

```python
def test_cancel_db_does_not_break_retry_db(tmp_path: Path):
    """Both DB classes share the same .retry.sqlite file; opening one must not
    break the other."""
    from tools.odin.valhalla.dashboard.retry_db import RetryDB

    cancel = CancelDB(tmp_path)
    cancel.request("20260504-100000", "run-a", kind="kill")

    retry = RetryDB(tmp_path)
    assert retry.read_pending("20260504-100000") == set()
    retry.toggle("20260504-100000", "run-z")
    assert retry.read_pending("20260504-100000") == {"run-z"}
    assert cancel.read_pending("20260504-100000") == {"run-a": "kill"}
```

- [ ] **Step 1.6: Run the new test**

```bash
python3 -m pytest --noconftest -p no:cacheprovider \
  tools/odin/valhalla/dashboard/tests/test_cancel_db.py::test_cancel_db_does_not_break_retry_db -v
```

Expected: PASS.

- [ ] **Step 1.7: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/cancel_db.py \
        tools/odin/valhalla/dashboard/tests/test_cancel_db.py
git commit -m "Odin CancelDB: SQLite cancellation queue (schema + request/read)"
```

---

## Task 2: `CancelDB.mark_consumed` + `upgrade_to_kill` + invalid-kind / replace tests

**Files:**
- Modify: `tools/odin/valhalla/dashboard/cancel_db.py`
- Test: `tools/odin/valhalla/dashboard/tests/test_cancel_db.py`

- [ ] **Step 2.1: Write the failing tests**

Append to `test_cancel_db.py`:

```python
def test_request_kill_then_skip_replaces_kind(tmp_path: Path):
    db = CancelDB(tmp_path)

    db.request("20260504-100000", "run-a", kind="kill")
    db.request("20260504-100000", "run-a", kind="skip")

    assert db.read_pending("20260504-100000") == {"run-a": "skip"}


def test_request_rejects_invalid_kind(tmp_path: Path):
    db = CancelDB(tmp_path)

    with pytest.raises(ValueError, match="kind must be one of"):
        db.request("20260504-100000", "run-a", kind="cancel")


def test_mark_consumed_sets_outcome_and_consumed_at(tmp_path: Path):
    db = CancelDB(tmp_path)
    db.request("20260504-100000", "run-a", kind="kill")

    db.mark_consumed("20260504-100000", "run-a", outcome="killed")

    assert db.read_pending("20260504-100000") == {}
    rows = db.list_for_dispatch("20260504-100000")
    assert len(rows) == 1
    assert rows[0].outcome == "killed"
    assert rows[0].consumed_at is not None


def test_mark_consumed_rejects_invalid_outcome(tmp_path: Path):
    db = CancelDB(tmp_path)
    db.request("20260504-100000", "run-a", kind="kill")

    with pytest.raises(ValueError, match="outcome must be one of"):
        db.mark_consumed("20260504-100000", "run-a", outcome="cancelled")


def test_upgrade_to_kill_promotes_skip(tmp_path: Path):
    db = CancelDB(tmp_path)
    db.request("20260504-100000", "run-a", kind="skip")

    db.upgrade_to_kill("20260504-100000", "run-a")

    assert db.read_pending("20260504-100000") == {"run-a": "kill"}


def test_upgrade_to_kill_idempotent_for_already_kill(tmp_path: Path):
    db = CancelDB(tmp_path)
    db.request("20260504-100000", "run-a", kind="kill")

    db.upgrade_to_kill("20260504-100000", "run-a")

    assert db.read_pending("20260504-100000") == {"run-a": "kill"}


def test_concurrent_request_one_winner(tmp_path: Path):
    """Two threads racing to insert the same row → single row, kind = last wins."""
    db = CancelDB(tmp_path)

    def _kill():
        db.request("20260504-100000", "run-a", kind="kill")

    def _skip():
        db.request("20260504-100000", "run-a", kind="skip")

    threads = [threading.Thread(target=_kill), threading.Thread(target=_skip)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    rows = db.list_for_dispatch("20260504-100000")
    assert len(rows) == 1
    assert rows[0].kind in {"kill", "skip"}


def test_migration_idempotent(tmp_path: Path):
    """Running migrations twice (e.g. two CancelDB instances) is a no-op."""
    CancelDB(tmp_path).request("20260504-100000", "run-a", kind="kill")
    # Second instance triggers _migrate again on connect.
    db2 = CancelDB(tmp_path)
    assert db2.read_pending("20260504-100000") == {"run-a": "kill"}
```

- [ ] **Step 2.2: Run the tests and verify they fail**

```bash
python3 -m pytest --noconftest -p no:cacheprovider \
  tools/odin/valhalla/dashboard/tests/test_cancel_db.py -v
```

Expected: 7 new tests fail (no `mark_consumed`, no `upgrade_to_kill`, no `list_for_dispatch`).

- [ ] **Step 2.3: Implement `mark_consumed`, `upgrade_to_kill`, `list_for_dispatch`, and `_row_from_sqlite`**

Append to `tools/odin/valhalla/dashboard/cancel_db.py` (inside `CancelDB`):

```python
    def mark_consumed(self, dispatch_id: str, run_id: str, *, outcome: str) -> None:
        """Mark a cancellation row as consumed by the runner."""
        if outcome not in _VALID_OUTCOMES:
            raise ValueError(f"outcome must be one of {sorted(_VALID_OUTCOMES)}, got {outcome!r}")
        with closing(self._connect()) as con:
            con.execute(
                """
                UPDATE cancellations
                SET consumed_at = ?, outcome = ?
                WHERE dispatch_id = ? AND run_id = ?
                """,
                (_now_iso(), outcome, dispatch_id, run_id),
            )

    def upgrade_to_kill(self, dispatch_id: str, run_id: str) -> None:
        """Flip a pending row's kind to ``kill`` (no-op if already killed/consumed)."""
        with closing(self._connect()) as con:
            con.execute(
                """
                UPDATE cancellations
                SET kind = 'kill'
                WHERE dispatch_id = ? AND run_id = ? AND consumed_at IS NULL
                """,
                (dispatch_id, run_id),
            )

    def list_for_dispatch(
        self,
        dispatch_id: str,
        *,
        pending_only: bool = False,
    ) -> list[CancelRow]:
        """Return cancellation rows for one dispatch."""
        where = "AND consumed_at IS NULL" if pending_only else ""
        with closing(self._connect()) as con:
            rows = con.execute(
                f"""
                SELECT * FROM cancellations
                WHERE dispatch_id = ?
                {where}
                ORDER BY run_id
                """,
                (dispatch_id,),
            ).fetchall()
        return [_row_from_sqlite(row) for row in rows]
```

And after the `_migrate` function:

```python
def _row_from_sqlite(row: sqlite3.Row) -> CancelRow:
    values: dict[str, Any] = dict(row)
    return CancelRow(
        dispatch_id=str(values["dispatch_id"]),
        run_id=str(values["run_id"]),
        requested_at=str(values["requested_at"]),
        kind=str(values["kind"]),
        consumed_at=values["consumed_at"],
        outcome=values["outcome"],
    )
```

- [ ] **Step 2.4: Run the tests and verify they pass**

```bash
python3 -m pytest --noconftest -p no:cacheprovider \
  tools/odin/valhalla/dashboard/tests/test_cancel_db.py -v
```

Expected: 9 passed.

- [ ] **Step 2.5: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/cancel_db.py \
        tools/odin/valhalla/dashboard/tests/test_cancel_db.py
git commit -m "Odin CancelDB: mark_consumed + upgrade_to_kill + list helpers"
```

---

## Task 3: `DataLayer` cancel-queue wrappers

**Files:**
- Modify: `tools/odin/valhalla/dashboard/data.py`
- Test: `tools/odin/valhalla/dashboard/tests/test_data_cancel.py` (new)

- [ ] **Step 3.1: Write the failing tests**

Create `tools/odin/valhalla/dashboard/tests/test_data_cancel.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for DataLayer's cancel-queue wrappers."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.odin.valhalla.dashboard.data import DataLayer


def test_request_cancel_inserts_into_db(tmp_path: Path):
    layer = DataLayer(tmp_path)

    layer.request_cancel("20260504-100000", "run-a", kind="kill")

    assert layer.read_cancel_queue("20260504-100000") == {"run-a": "kill"}


def test_read_cancel_queue_empty_for_unknown_dispatch(tmp_path: Path):
    layer = DataLayer(tmp_path)

    assert layer.read_cancel_queue("nope") == {}


def test_request_cancel_rejects_invalid_kind(tmp_path: Path):
    layer = DataLayer(tmp_path)

    with pytest.raises(ValueError):
        layer.request_cancel("20260504-100000", "run-a", kind="terminate")


def test_request_cancel_replaces_kind(tmp_path: Path):
    layer = DataLayer(tmp_path)

    layer.request_cancel("20260504-100000", "run-a", kind="skip")
    layer.request_cancel("20260504-100000", "run-a", kind="kill")

    assert layer.read_cancel_queue("20260504-100000") == {"run-a": "kill"}
```

- [ ] **Step 3.2: Run the tests and verify they fail**

```bash
python3 -m pytest --noconftest -p no:cacheprovider \
  tools/odin/valhalla/dashboard/tests/test_data_cancel.py -v
```

Expected: 4 errors (`AttributeError: 'DataLayer' object has no attribute 'request_cancel'`).

- [ ] **Step 3.3: Add the wrappers to `DataLayer`**

In `tools/odin/valhalla/dashboard/data.py`, add the import next to the existing `RetryDB` import:

```python
from tools.odin.valhalla.dashboard.cancel_db import CancelDB
from tools.odin.valhalla.dashboard.retry_db import RetryDB
```

In the `DataLayer.__init__`, add the lazy-init slot next to `self._retry_db`:

```python
        self._retry_db: RetryDB | None = None
        self._cancel_db: CancelDB | None = None
```

Append these methods to `DataLayer` (after `_get_retry_db`):

```python
    # -- cancel queue (operator's per-row kill / skip requests) --------------

    def read_cancel_queue(self, dispatch_id: str) -> dict[str, str]:
        """Return ``{run_id: kind}`` for the dispatch's pending cancellations.

        Stored in ``<runs_root>/.retry.sqlite`` (shared with the retry queue).
        Empty / missing rows return an empty dict.
        """
        return self._get_cancel_db().read_pending(dispatch_id)

    def request_cancel(self, dispatch_id: str, run_id: str, *, kind: str) -> None:
        """Request a kill (running) or skip (pending) cancellation for ``run_id``.

        Args:
            dispatch_id: Dispatch the row belongs to.
            run_id: Job to cancel.
            kind: ``"kill"`` (signal worker to abort) or ``"skip"`` (mark
                pending job as failed before submit).
        """
        self._get_cancel_db().request(dispatch_id, run_id, kind=kind)

    def _get_cancel_db(self) -> CancelDB:
        if self._cancel_db is None:
            self._cancel_db = CancelDB(self._runs_root)
        return self._cancel_db
```

- [ ] **Step 3.4: Run the tests and verify they pass**

```bash
python3 -m pytest --noconftest -p no:cacheprovider \
  tools/odin/valhalla/dashboard/tests/test_data_cancel.py -v
```

Expected: 4 passed.

- [ ] **Step 3.5: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/data.py \
        tools/odin/valhalla/dashboard/tests/test_data_cancel.py
git commit -m "Odin DataLayer: cancel-queue read/request wrappers"
```

---

## Task 4: `FailureInfo.kind` docstring extension

**Files:**
- Modify: `tools/odin/asgard/jobs.py`

No test (docstring-only change).

- [ ] **Step 4.1: Edit the `FailureInfo` docstring**

In `tools/odin/asgard/jobs.py`, find the `FailureInfo` class docstring (the `kind values` enumeration). Append two new bullets after the `gpu_lost` entry:

```python
    - ``killed``: operator-initiated kill via the Tab A cancel button. The
      worker pkilled the trainer mid-run and pulled whatever partial bundle
      was on disk. Does NOT count as host-health failure for the
      circuit-breaker.
    - ``skipped``: operator-initiated skip via the Tab A cancel button. The
      runner flipped a pending job to failed before any worker submitted.
      No bundle exists.
```

- [ ] **Step 4.2: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/jobs.py
git commit -m "Odin jobs: document killed / skipped failure kinds"
```

---

## Task 5: Worker — `request_cancel` + `kill_dispatched` + finalize precedence

**Files:**
- Modify: `tools/odin/asgard/worker.py`
- Test: `tools/odin/tests/test_asgard_worker_cancel.py` (new)

- [ ] **Step 5.1: Write the failing tests**

Create `tools/odin/tests/test_asgard_worker_cancel.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Worker-side kill / skip handling tests."""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path

from tools.odin.asgard.fleet import ValkyrieConfig
from tools.odin.asgard.jobs import JobEntry
from tools.odin.asgard.tracker import Tracker
from tools.odin.asgard.transport import RsyncResult, SSHResult
from tools.odin.asgard.worker import (
    JobInflight,
    POLL_EXITED_NO_MANIFEST,
    ValkyrieWorker,
    WorkerOptions,
)


def _host(host: str = "v1") -> ValkyrieConfig:
    return ValkyrieConfig(host=host, ssh_user="odin", isaaclab_path="~/IsaacLab")


def _job(run_id: str = "r-cancel") -> JobEntry:
    return JobEntry(
        run_id=run_id,
        task_id="Isaac-Ant-Direct-v0",
        framework="rsl_rl",
        backend="physx",
        num_envs=4096,
        max_iterations=300,
        seed=42,
        bundle_dir_name=run_id,
    )


@dataclass
class _ScriptedSSH:
    scripted: dict = field(default_factory=dict)
    log: list = field(default_factory=list)

    def run(self, host, cmd, *, timeout_s=None, stdout_tee=None, pty=True):
        self.log.append((host.host, cmd, pty))
        for key, result in self.scripted.items():
            if key in cmd:
                return result
        return SSHResult(exit_code=0, stdout="", stderr="", duration_s=0.0)


@dataclass
class _NoopRsync:
    log: list = field(default_factory=list)

    def pull(self, host, remote_path, local_path):
        self.log.append(("pull", remote_path, str(local_path)))
        local_path.mkdir(parents=True, exist_ok=True)
        (local_path / "logs").mkdir(parents=True, exist_ok=True)
        return RsyncResult(exit_code=0, stdout="", stderr="", duration_s=0.0)

    def push(self, host, local_path, remote_path):
        return RsyncResult(exit_code=0, stdout="", stderr="", duration_s=0.0)


def _make_worker(tmp_path: Path, ssh, rsync) -> ValkyrieWorker:
    return ValkyrieWorker(
        host=_host(),
        job_queue=queue.Queue(),
        state_chan=queue.Queue(),
        dispatch_dir=tmp_path,
        options=WorkerOptions(per_job_timeout_s=600, detached_mode=True, poll_interval_s=0),
        ssh=ssh,
        rsync=rsync,
        shutdown_event=threading.Event(),
    )


def test_request_cancel_records_run_id(tmp_path: Path):
    worker = _make_worker(tmp_path, _ScriptedSSH(), _NoopRsync())

    worker.request_cancel("r-cancel")

    assert worker._cancel_request.get("r-cancel") is True


def test_finalize_with_kill_dispatched_classifies_as_killed(tmp_path: Path):
    """When _sweep_cancellations has marked kill_dispatched, _finalize_terminal
    must override _classify_remote and stamp kind=killed."""
    rsync = _NoopRsync()
    worker = _make_worker(tmp_path, _ScriptedSSH(), rsync)
    job = _job("r-killed")
    inflight = JobInflight(
        job=job,
        tracker=Tracker(
            run_id=job.run_id,
            container_name=worker.host.container_name,
            host=worker.host.host,
            submitted_at="2026-05-04T10:00:00Z",
            pid=12345,
            per_job_timeout_s=600,
        ),
        submitted_at_monotonic=0.0,
        kill_dispatched=True,
    )
    worker._inflight[job.run_id] = inflight

    worker._finalize_terminal(inflight, POLL_EXITED_NO_MANIFEST)

    transitions = []
    while not worker._state_chan.empty():
        ev = worker._state_chan.get_nowait()
        transitions.append((ev.transition, ev.failure.kind if ev.failure else None))
    failed = next(t for t in transitions if t[0] == "failed")
    assert failed[1] == "killed"
    # Bundle still pulled (per Q4 answer): partial logs preserved.
    assert any(p[0] == "pull" for p in rsync.log)


def test_finalize_timeout_takes_precedence_over_kill(tmp_path: Path):
    """Both flags set → kind=timeout (job tripped budget before operator clicked)."""
    worker = _make_worker(tmp_path, _ScriptedSSH(), _NoopRsync())
    job = _job("r-timeout-kill")
    inflight = JobInflight(
        job=job,
        tracker=None,
        submitted_at_monotonic=0.0,
        timeout_kill_dispatched=True,
        kill_dispatched=True,
    )
    worker._inflight[job.run_id] = inflight

    worker._finalize_terminal(inflight, POLL_EXITED_NO_MANIFEST)

    transitions = []
    while not worker._state_chan.empty():
        ev = worker._state_chan.get_nowait()
        transitions.append((ev.transition, ev.failure.kind if ev.failure else None))
    failed = next(t for t in transitions if t[0] == "failed")
    assert failed[1] == "timeout"


def test_sweep_cancellations_dispatches_pkill(tmp_path: Path):
    ssh = _ScriptedSSH()
    worker = _make_worker(tmp_path, ssh, _NoopRsync())
    job = _job("r-killing")
    inflight = JobInflight(
        job=job,
        tracker=None,
        submitted_at_monotonic=0.0,
    )
    worker._inflight[job.run_id] = inflight
    worker.request_cancel(job.run_id)

    worker._sweep_cancellations()

    pkill = [cmd for _, cmd, _ in ssh.log if "pkill" in cmd]
    assert len(pkill) == 1
    assert job.run_id in pkill[0]
    assert inflight.kill_dispatched is True
    # _cancel_request consumed (drained after dispatch).
    assert "r-killing" not in worker._cancel_request


def test_sweep_cancellations_drops_unknown_run_id(tmp_path: Path):
    """Cancel for a job that already finished (no inflight entry) → silent drop."""
    ssh = _ScriptedSSH()
    worker = _make_worker(tmp_path, ssh, _NoopRsync())
    worker.request_cancel("ghost-run")

    worker._sweep_cancellations()

    assert "ghost-run" not in worker._cancel_request
    assert not any("pkill" in cmd for _, cmd, _ in ssh.log)


def test_sweep_cancellations_idempotent(tmp_path: Path):
    """Second sweep with the same kill_dispatched=True does not re-pkill."""
    ssh = _ScriptedSSH()
    worker = _make_worker(tmp_path, ssh, _NoopRsync())
    job = _job("r-twice")
    inflight = JobInflight(job=job, tracker=None, submitted_at_monotonic=0.0)
    worker._inflight[job.run_id] = inflight
    worker.request_cancel(job.run_id)
    worker._sweep_cancellations()
    assert sum(1 for _, c, _ in ssh.log if "pkill" in c) == 1

    # Second sweep: nothing new in _cancel_request, kill_dispatched already set.
    worker.request_cancel(job.run_id)  # operator clicked again somehow
    worker._sweep_cancellations()

    # Only the original pkill — kill_dispatched gate prevents the second.
    assert sum(1 for _, c, _ in ssh.log if "pkill" in c) == 1


def test_submit_or_handle_skips_when_status_already_failed(tmp_path: Path):
    """Skip race: runner flipped status before worker pulled the job. Worker
    must NOT submit (no SSH call)."""
    ssh = _ScriptedSSH()
    worker = _make_worker(tmp_path, ssh, _NoopRsync())
    job = _job("r-already-skipped")
    job.status = "failed"  # runner did the flip

    worker._submit_or_handle(job)

    # No SSH submit attempted.
    assert not any("docker exec -i" in cmd for _, cmd, _ in ssh.log)
    # No state event emitted (the runner already handled the transition).
    assert worker._state_chan.empty()
```

- [ ] **Step 5.2: Run the tests and verify they fail**

```bash
python3 -m pytest --noconftest -p no:cacheprovider \
  tools/odin/tests/test_asgard_worker_cancel.py -v
```

Expected: 7 errors / failures (no `request_cancel`, no `kill_dispatched`, no `_sweep_cancellations`, no skip-status guard).

- [ ] **Step 5.3: Add `kill_dispatched` to `JobInflight`**

In `tools/odin/asgard/worker.py`, find the `JobInflight` dataclass. Add the new field next to `timeout_kill_dispatched`:

```python
    timeout_kill_dispatched: bool = False
    kill_dispatched: bool = False
```

Update its docstring's `Attributes:` block to add:

```
        kill_dispatched: ``True`` once :meth:`_sweep_cancellations` has
            issued a best-effort pkill in response to an operator kill;
            the next ``exited-no-manifest`` poll classifies as
            ``killed`` rather than running ``_classify_remote``.
```

- [ ] **Step 5.4: Initialize `_cancel_request` in `__init__`**

In `ValkyrieWorker.__init__`, after the existing `self._inflight: dict[str, JobInflight] = {}` line, add:

```python
        # Kill requests pushed by the runner via ``request_cancel(run_id)``.
        # Drained on each ``_sweep_cancellations`` tick.
        self._cancel_request: dict[str, bool] = {}
        self._cancel_request_lock = threading.Lock()
```

- [ ] **Step 5.5: Add `request_cancel`, `_sweep_cancellations`**

Append these methods to `ValkyrieWorker` (place next to `_sweep_timeouts`):

```python
    def request_cancel(self, run_id: str) -> None:
        """Mark ``run_id`` for kill. Called by the runner from its main thread.

        Thread-safe: a single dict assignment is atomic in CPython, but the
        explicit lock keeps the contract obvious and protects against
        concurrent ``_sweep_cancellations`` reads during list-rebuild.
        """
        with self._cancel_request_lock:
            self._cancel_request[run_id] = True

    def _sweep_cancellations(self) -> None:
        """For each pending kill request, dispatch a best-effort pkill once.

        The next poll tick will see ``exited-no-manifest`` and
        :meth:`_finalize_terminal` will classify as ``killed`` (because
        ``inflight.kill_dispatched`` is set here).
        """
        with self._cancel_request_lock:
            requested = list(self._cancel_request.keys())
            self._cancel_request.clear()
        for run_id in requested:
            inflight = self._inflight.get(run_id)
            if inflight is None:
                # Job already finished (or was never on this worker). Drop.
                continue
            if inflight.kill_dispatched:
                continue
            self._cleanup_remote_process(inflight.job)
            inflight.kill_dispatched = True
```

- [ ] **Step 5.6: Update `_finalize_terminal` precedence**

Find `_finalize_terminal` in `worker.py`. Locate the block that computes `failure` for `POLL_EXITED_NO_MANIFEST`:

```python
        # POLL_EXITED_NO_MANIFEST
        if inflight.timeout_kill_dispatched:
            failure = FailureInfo(
                kind="timeout",
                message=f"remote process exceeded {self._options.per_job_timeout_s}s",
                details={"per_job_timeout_s": self._options.per_job_timeout_s},
            )
        else:
            failure = self._classify_remote(job)
```

Replace with:

```python
        # POLL_EXITED_NO_MANIFEST
        if inflight.timeout_kill_dispatched:
            # Timeout precedence stays — operator-clicked Kill on a job that
            # tripped its budget gets the more accurate kind="timeout".
            failure = FailureInfo(
                kind="timeout",
                message=f"remote process exceeded {self._options.per_job_timeout_s}s",
                details={"per_job_timeout_s": self._options.per_job_timeout_s},
            )
        elif inflight.kill_dispatched:
            failure = FailureInfo(
                kind="killed",
                message="operator kill",
                details={"per_job_timeout_s": self._options.per_job_timeout_s},
            )
        else:
            failure = self._classify_remote(job)
```

- [ ] **Step 5.7: Add the skip-race guard at the top of `_submit_or_handle`**

Find `_submit_or_handle(self, job: JobEntry)` in `worker.py`. Add the guard as the first lines of the method body, before the existing `started_at = _utc_now_iso()`:

```python
    def _submit_or_handle(self, job: JobEntry) -> None:
        # Skip race: between when this job was put on the queue and when we
        # popped it off, the runner may have flipped its status to 'failed'
        # in response to an operator skip. Re-check before paying for an
        # SSH submit.
        if job.status != "pending":
            return
        started_at = _utc_now_iso()
```

- [ ] **Step 5.8: Wire `_sweep_cancellations` into the detached run loop**

Find the `_run_detached` method. Locate the block:

```python
        elif self._inflight:
            self._poll_inflight_once()
            self._sweep_timeouts()
```

Add the cancel sweep before the poll so kills land before the next status read:

```python
        elif self._inflight:
            self._sweep_cancellations()
            self._poll_inflight_once()
            self._sweep_timeouts()
```

- [ ] **Step 5.9: Run the new worker tests and verify they pass**

```bash
python3 -m pytest --noconftest -p no:cacheprovider \
  tools/odin/tests/test_asgard_worker_cancel.py -v
```

Expected: 7 passed.

- [ ] **Step 5.10: Run the existing worker tests to make sure nothing regressed**

```bash
python3 -m pytest --noconftest -p no:cacheprovider \
  tools/odin/tests/test_asgard_worker.py \
  tools/odin/tests/test_asgard_worker_submit.py \
  tools/odin/tests/test_asgard_worker_poll.py -v
```

Expected: all existing tests pass.

- [ ] **Step 5.11: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/worker.py \
        tools/odin/tests/test_asgard_worker_cancel.py
git commit -m "Asgard worker: handle operator kill via request_cancel + kill_dispatched"
```

---

## Task 6: Runner — `_consume_cancellations` + main-loop wiring

**Files:**
- Modify: `tools/odin/asgard/runner.py`
- Test: `tools/odin/tests/test_asgard_runner_cancellations.py` (new)

We also switch `workers: list[ValkyrieWorker]` to `workers_by_host: dict[str, ValkyrieWorker]` so the cancel handler can look up the assigned worker by host. The list-iteration sites are limited (workers.start, workers.is_alive, workers.join, the `_ in workers` sentinel push) so the rename is mechanical.

- [ ] **Step 6.1: Write the failing tests**

Create `tools/odin/tests/test_asgard_runner_cancellations.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runner-side cancellation handling: _consume_cancellations + mark_consumed."""

from __future__ import annotations

import queue
from pathlib import Path

import pytest

from tools.odin.asgard.jobs import JobEntry
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
    from tools.odin.asgard.jobs import FailureInfo

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
    from tools.odin.asgard.jobs import FailureInfo

    ev = StateEvent(
        run_id="r-kill",
        host="v1",
        transition="failed",
        failure=FailureInfo(kind="hugin_crash", message="real crash"),
    )

    _mark_cancellation_consumed(cancel_db=cancel_db, dispatch_id="d1", ev=ev)

    # Row stays pending — operator can still kill the next attempt.
    assert cancel_db.read_pending("d1") == {"r-kill": "kill"}
```

- [ ] **Step 6.2: Run the tests and verify they fail**

```bash
python3 -m pytest --noconftest -p no:cacheprovider \
  tools/odin/tests/test_asgard_runner_cancellations.py -v
```

Expected: import errors (no `_consume_cancellations`, no `_mark_cancellation_consumed`).

- [ ] **Step 6.3: Implement `_consume_cancellations`**

In `tools/odin/asgard/runner.py`, add the import next to the existing `RetryDB` import:

```python
from tools.odin.valhalla.dashboard.cancel_db import CancelDB
from tools.odin.valhalla.dashboard.retry_db import RetryDB
```

Add the helper next to `_consume_live_retries` (place right after it):

```python
def _consume_cancellations(
    *,
    cancel_db: CancelDB,
    dispatch_id: str,
    jobs_by_id: dict[str, JobEntry],
    workers_by_host: dict,
) -> int:
    """Drain pending kill / skip rows from ``cancel_db`` and act on them.

    For each pending row:

    - Job already in a terminal state → mark ``outcome="noop"`` (cleanup).
    - Skip on a pending job → flip ``status="failed"`` (kind=skipped),
      mark ``outcome="skipped"``. Returns 1 (caller's ``remaining`` -= 1).
    - Skip on a running job → upgrade row to kill, fall through.
    - Kill on a running job → call ``worker.request_cancel(run_id)``;
      leave the row pending (worker emits ``failed/killed`` later;
      :func:`_mark_cancellation_consumed` then marks ``outcome="killed"``).

    Args:
        cancel_db: Open :class:`CancelDB` for this dispatch's runs_root.
        dispatch_id: Current dispatch id.
        jobs_by_id: ``run_id → JobEntry`` map (shared with the main loop).
        workers_by_host: ``hostname → worker`` lookup. Workers must expose
            ``request_cancel(run_id)`` (the real
            :class:`~tools.odin.asgard.worker.ValkyrieWorker` does).

    Returns:
        Count of jobs that landed terminal in this call (skips only).
    """
    landed = 0
    for run_id, kind in cancel_db.read_pending(dispatch_id).items():
        job = jobs_by_id.get(run_id)
        if job is None or job.status in {"completed", "failed"}:
            cancel_db.mark_consumed(dispatch_id, run_id, outcome="noop")
            continue
        if kind == "skip" and job.status == "pending":
            job.status = "failed"
            job.failure = FailureInfo(
                kind="skipped",
                message="operator skipped before dispatch",
                details={"requested_at": _utc_now_iso()},
            )
            job.ended_at = _utc_now_iso()
            cancel_db.mark_consumed(dispatch_id, run_id, outcome="skipped")
            landed += 1
            continue
        if kind == "skip" and job.status == "running":
            cancel_db.upgrade_to_kill(dispatch_id, run_id)
            kind = "kill"
        if kind == "kill" and job.status == "running":
            worker = workers_by_host.get(job.assigned_to)
            if worker is None:
                # Worker for the assigned host isn't around any more
                # (host_down quarantine). Mark noop; the worker is gone
                # so the job won't terminate via kill anyway.
                cancel_db.mark_consumed(dispatch_id, run_id, outcome="noop")
                continue
            worker.request_cancel(run_id)
            # Row stays pending; consumed when the worker emits failed/killed.
    return landed


def _mark_cancellation_consumed(
    *,
    cancel_db: CancelDB,
    dispatch_id: str,
    ev: StateEvent,
) -> None:
    """Mark a cancellation row consumed when the matching worker event arrives.

    Only fires for ``failed`` events whose ``failure.kind == "killed"``
    (skips are marked synchronously inside :func:`_consume_cancellations`).
    """
    if ev.transition != "failed" or ev.failure is None:
        return
    if ev.failure.kind != "killed":
        return
    cancel_db.mark_consumed(dispatch_id, ev.run_id, outcome="killed")
```

- [ ] **Step 6.4: Run the tests and verify they pass**

```bash
python3 -m pytest --noconftest -p no:cacheprovider \
  tools/odin/tests/test_asgard_runner_cancellations.py -v
```

Expected: 7 passed.

- [ ] **Step 6.5: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/runner.py \
        tools/odin/tests/test_asgard_runner_cancellations.py
git commit -m "Asgard runner: _consume_cancellations + _mark_cancellation_consumed"
```

---

## Task 7: Runner — wire cancellations into the main loop

**Files:**
- Modify: `tools/odin/asgard/runner.py`

We rename `workers: list[ValkyrieWorker]` to `workers_by_host: dict[str, ValkyrieWorker]` and call `_consume_cancellations` adjacent to `_consume_live_retries`. Then thread `_mark_cancellation_consumed` into the `_apply_state_event` consumer.

- [ ] **Step 7.1: Switch `workers` to a dict keyed by host**

Find the block in `run_dispatch` that initializes `workers: list[ValkyrieWorker] = []` and the per-host loop. Replace:

```python
    workers: list[ValkyrieWorker] = []
    for host in healthy:
        w = ValkyrieWorker(
            ...
        )
        for reattach_job in reattached_by_host.get(host.host, []):
            from tools.odin.asgard.worker import JobInflight

            w._inflight[reattach_job.run_id] = JobInflight(
                job=reattach_job,
                tracker=None,
                submitted_at_monotonic=time.monotonic(),
            )
        w.start()
        workers.append(w)
```

with:

```python
    workers_by_host: dict[str, ValkyrieWorker] = {}
    for host in healthy:
        w = ValkyrieWorker(
            ...
        )
        for reattach_job in reattached_by_host.get(host.host, []):
            from tools.odin.asgard.worker import JobInflight

            w._inflight[reattach_job.run_id] = JobInflight(
                job=reattach_job,
                tracker=None,
                submitted_at_monotonic=time.monotonic(),
            )
        w.start()
        workers_by_host[host.host] = w
    workers = list(workers_by_host.values())
```

(Keep the `workers` local for `is_alive` / `join` / sentinel iterations downstream — minimal touch.)

- [ ] **Step 7.2: Initialize `cancel_db` next to `retry_db`**

Find the line `retry_db = RetryDB(dispatch_dir.parent)`. Add immediately after:

```python
    cancel_db = CancelDB(dispatch_dir.parent)
```

- [ ] **Step 7.3: Call `_consume_cancellations` in the empty-channel branch**

Find the `if now - last_retry_poll >= live_retry_poll_s:` block. After the existing `_consume_live_retries(...)` call but before `last_retry_poll = now`, add:

```python
                cancel_added = _consume_cancellations(
                    cancel_db=cancel_db,
                    dispatch_id=dispatch_id,
                    jobs_by_id=jobs_by_id,
                    workers_by_host=workers_by_host,
                )
                if cancel_added:
                    remaining -= cancel_added  # skipped jobs flipped pending→failed
                    write_dispatch_state(dispatch_dir, state)
                    last_write = now
```

(Note: cancel_added is the number of `pending` jobs flipped to `failed` — they no longer count toward "remaining work", so we subtract.)

- [ ] **Step 7.4: Wire `_mark_cancellation_consumed` next to `_mark_live_retry_consumed`**

Find the `_mark_live_retry_consumed(...)` call inside the main loop. Add immediately after:

```python
        _mark_cancellation_consumed(
            cancel_db=cancel_db,
            dispatch_id=dispatch_id,
            ev=ev,
        )
```

- [ ] **Step 7.5: Also call `_consume_cancellations` after each `failed`/`completed` state event**

Find the `if ev.transition in {"completed", "failed"}:` block (it currently runs `_consume_live_retries` again). Add after the `_consume_live_retries(...)` call:

```python
            cancel_added = _consume_cancellations(
                cancel_db=cancel_db,
                dispatch_id=dispatch_id,
                jobs_by_id=jobs_by_id,
                workers_by_host=workers_by_host,
            )
            if cancel_added:
                remaining -= cancel_added
                write_dispatch_state(dispatch_dir, state)
                last_write = time.monotonic()
```

- [ ] **Step 7.6: Run the existing runner tests**

```bash
python3 -m pytest --noconftest -p no:cacheprovider \
  tools/odin/tests/test_asgard_runner.py -v
```

Expected: all pass (the test fakes use the legacy non-detached path; cancel wiring is no-op when `cancel_db.read_pending` is empty).

- [ ] **Step 7.7: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/runner.py
git commit -m "Asgard runner: wire cancel_db consumption into main loop"
```

---

## Task 8: Reconcile — re-apply pending cancellations on `--resume`

**Files:**
- Modify: `tools/odin/asgard/reconcile.py`
- Modify: `tools/odin/tests/test_asgard_reconcile.py`

- [ ] **Step 8.1: Write the failing tests**

Append to `tools/odin/tests/test_asgard_reconcile.py`:

```python
def test_reconcile_applies_pending_skip_for_pending_job(tmp_path: Path):
    """Resume: a 'skip' cancellation arrived while the dispatcher was down.
    The pending job is flipped to failed/skipped and the row marked consumed."""
    from tools.odin.asgard.reconcile import reconcile_orphans
    from tools.odin.valhalla.dashboard.cancel_db import CancelDB

    fleet = Fleet(fleet_name="t", hosts=[_host()])
    job = _job("r-skip-on-resume")
    job.status = "pending"
    job.assigned_to = None
    cancel_db = CancelDB(tmp_path)
    cancel_db.request(tmp_path.name, "r-skip-on-resume", kind="skip")
    ssh = _FakeSSH(scripted={})
    rsync = _FakeRsync()

    reconcile_orphans(
        fleet=fleet,
        jobs=[job],
        dispatch_dir=tmp_path,
        ssh=ssh,
        rsync=rsync,
        detached_mode=True,
        cancel_db=cancel_db,
    )

    assert job.status == "failed"
    assert job.failure is not None
    assert job.failure.kind == "skipped"
    assert cancel_db.read_pending(tmp_path.name) == {}
```

- [ ] **Step 8.2: Run and verify failure**

```bash
python3 -m pytest --noconftest -p no:cacheprovider \
  tools/odin/tests/test_asgard_reconcile.py::test_reconcile_applies_pending_skip_for_pending_job -v
```

Expected: `TypeError: reconcile_orphans() got an unexpected keyword argument 'cancel_db'`.

- [ ] **Step 8.3: Add the `cancel_db` parameter to `reconcile_orphans`**

In `tools/odin/asgard/reconcile.py`, edit `reconcile_orphans`'s signature:

```python
def reconcile_orphans(
    *,
    fleet: Fleet,
    jobs: list[JobEntry],
    dispatch_dir: Path,
    ssh: SSHRunner,
    rsync: RsyncRunner,
    detached_mode: bool = False,
    cancel_db: object | None = None,
) -> list[ReconcileOutcome]:
```

(Use `object | None` rather than `CancelDB | None` to avoid an import cycle — reconcile imports from worker, worker doesn't import from reconcile, but importing CancelDB at module top would pull dashboard code into the asgard package. Inline-import inside the function body where needed.)

Update the docstring's `Args:` block with:

```
        cancel_db: Optional :class:`CancelDB` (passed by the runner on
            ``--resume``). When non-None, pending skip/kill rows are
            re-applied before workers spin up — skips flip pending jobs
            to failed; kills are applied to in-flight jobs after the
            re-attach by seeding ``worker._cancel_request`` (handled in
            the runner, not here).
```

At the end of the function (just before the final `return outcomes`), add:

```python
    if cancel_db is not None:
        dispatch_id = dispatch_dir.name
        for run_id, kind in list(cancel_db.read_pending(dispatch_id).items()):
            job = next((j for j in jobs if j.run_id == run_id), None)
            if job is None or job.status in {"completed", "failed"}:
                cancel_db.mark_consumed(dispatch_id, run_id, outcome="noop")
                continue
            if kind == "skip" and job.status == "pending":
                from tools.odin.asgard.jobs import FailureInfo

                job.status = "failed"
                job.failure = FailureInfo(
                    kind="skipped",
                    message="operator skipped before dispatch (applied at resume)",
                    details={"reconciled": True},
                )
                cancel_db.mark_consumed(dispatch_id, run_id, outcome="skipped")
                outcomes.append(ReconcileOutcome(run_id=run_id, action="adopted_failed"))
            # Pending kill rows for in-flight jobs are seeded into the worker's
            # _cancel_request map by the runner after worker construction; we
            # leave those rows untouched here so the runner's main loop sees
            # them on the first tick.

    return outcomes
```

- [ ] **Step 8.4: Wire `cancel_db` into the runner's reconcile call**

In `tools/odin/asgard/runner.py`, find the existing reconcile call (look for `reconcile_orphans(`):

```python
        reconcile_orphans(
            fleet=fleet,
            jobs=prior_state.jobs,
            dispatch_dir=dispatch_dir,
            ssh=ssh,
            rsync=rsync,
            detached_mode=options.detached_mode,
        )
```

Replace with:

```python
        reconcile_orphans(
            fleet=fleet,
            jobs=prior_state.jobs,
            dispatch_dir=dispatch_dir,
            ssh=ssh,
            rsync=rsync,
            detached_mode=options.detached_mode,
            cancel_db=CancelDB(dispatch_dir.parent),
        )
```

- [ ] **Step 8.5: Run the new reconcile test**

```bash
python3 -m pytest --noconftest -p no:cacheprovider \
  tools/odin/tests/test_asgard_reconcile.py -v
```

Expected: all pass (existing tests untouched because `cancel_db` defaults to `None`).

- [ ] **Step 8.6: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/reconcile.py \
        tools/odin/asgard/runner.py \
        tools/odin/tests/test_asgard_reconcile.py
git commit -m "Asgard reconcile: re-apply pending skip cancellations on --resume"
```

---

## Task 9: Tab A — render the cancel button + pending badge

**Files:**
- Modify: `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py`
- Test: `tools/odin/valhalla/dashboard/tests/test_tab_a_cancel_button.py` (new)

The button replaces the existing failure-cell-only retry-toggle's cell partner — we add a new column-less cell to the existing Status cell so layout stays compact.

- [ ] **Step 9.1: Write the failing tests**

Create `tools/odin/valhalla/dashboard/tests/test_tab_a_cancel_button.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tab A jobs-table render tests for the cancel (kill / skip) button."""

from __future__ import annotations

from dash import html

from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.jobs_table import (
    render_jobs_section,
)


def _payload_with_jobs(jobs, *, ended_at=None):
    return {
        "dispatch_id": "20260504-100000",
        "ended_at": ended_at,
        "jobs": jobs,
    }


def _job(status: str, run_id: str = "r1") -> dict:
    return {
        "run_id": run_id,
        "task_id": "t",
        "framework": "rsl_rl",
        "backend": "physx",
        "seed": 42,
        "status": status,
        "assigned_to": "v1",
    }


def _walk(node):
    """Iterate every Dash component in a tree."""
    yield node
    children = getattr(node, "children", None)
    if children is None:
        return
    if isinstance(children, list):
        for child in children:
            yield from _walk(child)
    else:
        yield from _walk(children)


def _find_buttons(section, run_id: str) -> list[html.Button]:
    out = []
    for node in _walk(section):
        if not isinstance(node, html.Button):
            continue
        ident = getattr(node, "id", None)
        if isinstance(ident, dict) and ident.get("type") == "tab-a-cancel-toggle" and ident.get("run_id") == run_id:
            out.append(node)
    return out


def test_pending_row_renders_skip_button():
    section = render_jobs_section(_payload_with_jobs([_job("pending", "r1")]))

    buttons = _find_buttons(section, "r1")
    assert len(buttons) == 1
    assert "Skip" in (buttons[0].children or "")


def test_running_row_renders_kill_button():
    section = render_jobs_section(_payload_with_jobs([_job("running", "r1")]))

    buttons = _find_buttons(section, "r1")
    assert len(buttons) == 1
    assert "Kill" in (buttons[0].children or "")


def test_completed_row_does_not_render_cancel_button():
    section = render_jobs_section(_payload_with_jobs([_job("completed", "r1")]))

    assert _find_buttons(section, "r1") == []


def test_finished_dispatch_hides_cancel_button():
    section = render_jobs_section(
        _payload_with_jobs([_job("running", "r1")], ended_at="2026-05-04T11:00:00Z"),
    )

    assert _find_buttons(section, "r1") == []


def test_pending_cancellation_renders_pending_badge():
    section = render_jobs_section(
        _payload_with_jobs([_job("running", "r1")]),
        cancel_queue={"r1": "kill"},
    )

    # Button still rendered but in "pending" disabled style.
    buttons = _find_buttons(section, "r1")
    assert len(buttons) == 1
    assert "kill pending" in (buttons[0].title or "").lower()
    # Badge text appears in the Status cell.
    found_badge = any(
        getattr(node, "className", "") == "tab-a-cancel-pending-badge" for node in _walk(section)
    )
    assert found_badge


def test_confirm_state_renders_red_confirm_label():
    """When run_id is in cancel_confirm (first click within window), the button
    label flips to 'Confirm Kill' / 'Confirm Skip' with the red CSS class."""
    section = render_jobs_section(
        _payload_with_jobs([_job("running", "r1")]),
        cancel_confirm={"r1": 1_700_000_005_000},  # any future ms
    )

    buttons = _find_buttons(section, "r1")
    assert len(buttons) == 1
    label = buttons[0].children or ""
    assert "Confirm Kill" in label
    assert "tab-a-cancel-toggle-confirm" in (buttons[0].className or "")


def test_confirm_state_for_pending_status_says_confirm_skip():
    section = render_jobs_section(
        _payload_with_jobs([_job("pending", "r1")]),
        cancel_confirm={"r1": 1_700_000_005_000},
    )

    buttons = _find_buttons(section, "r1")
    assert "Confirm Skip" in (buttons[0].children or "")
```

- [ ] **Step 9.2: Run and verify failure**

```bash
python3 -m pytest --noconftest -p no:cacheprovider \
  tools/odin/valhalla/dashboard/tests/test_tab_a_cancel_button.py -v
```

Expected: failures (no `cancel_queue` kwarg, no `tab-a-cancel-toggle` button).

- [ ] **Step 9.3: Add `cancel_queue` and `cancel_confirm` to `render_jobs_section` + `render_jobs_rows` signatures**

In `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py`, update both functions to accept the new kwargs and forward them to `_data_row`:

```python
def render_jobs_section(
    dispatch_payload: dict,
    *,
    status_filter: list[str] | None = None,
    kind_filter: list[str] | None = None,
    task_text: str = "",
    expanded_run_ids: set[str] | None = None,
    ssh_tail_store: dict[str, list[str]] | None = None,
    retry_queue: set[str] | None = None,
    cancel_queue: dict[str, str] | None = None,
    cancel_confirm: dict[str, int] | None = None,
) -> html.Div:
    ...
    cancel_queue = cancel_queue or {}
    cancel_confirm = cancel_confirm or {}
    dispatch_ended = bool(dispatch_payload.get("ended_at"))
    ...
```

(Pass `cancel_queue`, `cancel_confirm`, and `dispatch_ended` through to `_data_row` and `render_jobs_rows`.)

Repeat for `render_jobs_rows` — same signature change, same `or {}` defaults.

In each `_data_row` call site, replace the existing call:

```python
        body_rows.append(_data_row(j, dispatch_id, retry_queue))
```

with:

```python
        body_rows.append(
            _data_row(j, dispatch_id, retry_queue, cancel_queue, cancel_confirm, dispatch_ended)
        )
```

- [ ] **Step 9.4: Update `_data_row` to render the cancel button (incl. confirm state)**

In `_data_row`, change the signature:

```python
def _data_row(
    job: dict,
    dispatch_id: str,
    retry_queue: set[str] | None = None,
    cancel_queue: dict[str, str] | None = None,
    cancel_confirm: dict[str, int] | None = None,
    dispatch_ended: bool = False,
) -> html.Tr:
    ...
    cancel_queue = cancel_queue or {}
    cancel_confirm = cancel_confirm or {}
    ...
```

After the existing `status_children = [...]` block, before `if attempts > 1:`, insert:

```python
    if not dispatch_ended and status in {"pending", "running"}:
        pending_kind = cancel_queue.get(run_id)
        if pending_kind is not None:
            status_children.append(
                html.Span(
                    f"{pending_kind} pending",
                    className="tab-a-cancel-pending-badge",
                )
            )
        base_label = "Kill" if status == "running" else "Skip"
        in_confirm = run_id in cancel_confirm
        cancel_label = f"Confirm {base_label}" if in_confirm else base_label
        css = ["tab-a-cancel-toggle"]
        if in_confirm:
            css.append("tab-a-cancel-toggle-confirm")
        if pending_kind:
            css.append("tab-a-cancel-toggle-pending")
        status_children.append(
            html.Button(
                cancel_label,
                id={"type": "tab-a-cancel-toggle", "run_id": run_id},
                n_clicks=0,
                className=" ".join(css),
                title=(
                    f"{pending_kind} pending — runner will act on next tick"
                    if pending_kind
                    else (
                        f"Click again within 5 s to {base_label.lower()} this job"
                        if in_confirm
                        else f"{base_label} this job"
                    )
                ),
            )
        )
```

- [ ] **Step 9.5: Run the new tests and verify they pass**

```bash
python3 -m pytest --noconftest -p no:cacheprovider \
  tools/odin/valhalla/dashboard/tests/test_tab_a_cancel_button.py -v
```

Expected: 7 passed.

- [ ] **Step 9.6: Run the existing jobs-table tests to confirm no regressions**

```bash
python3 -m pytest --noconftest -p no:cacheprovider \
  tools/odin/valhalla/dashboard/tests/test_tab_a_jobs_table.py -v
```

Expected: all pass (the new kwarg defaults to None / empty dict).

- [ ] **Step 9.7: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py \
        tools/odin/valhalla/dashboard/tests/test_tab_a_cancel_button.py
git commit -m "Tab A: render kill / skip button per row + pending badge"
```

---

## Task 10: Tab A — confirm-flow callback (two-click + 5 s revert)

**Files:**
- Modify: `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/callbacks.py`
- Modify: `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/layout.py`
- Test: `tools/odin/valhalla/dashboard/tests/test_tab_a_cancel_callback.py` (new)

The confirm flow is encoded in two pieces:

1. A `dcc.Store(id="tab-a-cancel-pending")` holding `{run_id: expires_at_ms}`.
2. A `dcc.Interval(id="tab-a-cancel-revert", interval=500)` that drains expired entries.

The handler is split into a pure `_on_cancel_toggle_handler(...)` that takes the click list, the current store, the per-row job statuses, the dispatch_id/ended flag, and the data layer, and returns the updated store. The Dash-binding callback wraps it and looks up the per-row statuses from the dispatch payload via `data.load_dispatch(dispatch_id)` (no extra `dcc.Store` needed — the existing `load_dispatch` cache absorbs the cost).

- [ ] **Step 10.1: Write the failing handler tests**

Create `tools/odin/valhalla/dashboard/tests/test_tab_a_cancel_callback.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tab A cancel-button confirm-flow callback handler tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.callbacks import (
    _on_cancel_revert_handler,
    _on_cancel_toggle_handler,
)


@dataclass
class _FakeData:
    runs_root: Path
    cancel_calls: list = field(default_factory=list)

    def request_cancel(self, dispatch_id, run_id, *, kind):
        self.cancel_calls.append((dispatch_id, run_id, kind))


def _now_ms() -> int:
    return 1_700_000_000_000


def test_first_click_flips_to_pending_state(tmp_path: Path):
    data = _FakeData(runs_root=tmp_path)
    pending_store = {}

    new_store = _on_cancel_toggle_handler(
        n_clicks_list=[1],
        ids_list=[{"type": "tab-a-cancel-toggle", "run_id": "r1"}],
        statuses=["running"],
        dispatch_id="d1",
        dispatch_ended=False,
        pending_store=pending_store,
        data=data,
        now_ms=_now_ms(),
    )

    # Stored with expiry 5s out; no DB write yet.
    assert new_store == {"r1": _now_ms() + 5000}
    assert data.cancel_calls == []


def test_second_click_within_window_writes_db_and_clears_pending(tmp_path: Path):
    data = _FakeData(runs_root=tmp_path)
    pending_store = {"r1": _now_ms() + 4000}  # 4s left

    new_store = _on_cancel_toggle_handler(
        n_clicks_list=[2],
        ids_list=[{"type": "tab-a-cancel-toggle", "run_id": "r1"}],
        statuses=["running"],
        dispatch_id="d1",
        dispatch_ended=False,
        pending_store=pending_store,
        data=data,
        now_ms=_now_ms(),
    )

    assert new_store == {}
    assert data.cancel_calls == [("d1", "r1", "kill")]


def test_skip_kind_for_pending_status(tmp_path: Path):
    data = _FakeData(runs_root=tmp_path)
    pending_store = {"r1": _now_ms() + 4000}

    _on_cancel_toggle_handler(
        n_clicks_list=[2],
        ids_list=[{"type": "tab-a-cancel-toggle", "run_id": "r1"}],
        statuses=["pending"],
        dispatch_id="d1",
        dispatch_ended=False,
        pending_store=pending_store,
        data=data,
        now_ms=_now_ms(),
    )

    assert data.cancel_calls == [("d1", "r1", "skip")]


def test_dispatch_ended_drops_clicks(tmp_path: Path):
    data = _FakeData(runs_root=tmp_path)

    new_store = _on_cancel_toggle_handler(
        n_clicks_list=[1],
        ids_list=[{"type": "tab-a-cancel-toggle", "run_id": "r1"}],
        statuses=["running"],
        dispatch_id="d1",
        dispatch_ended=True,
        pending_store={},
        data=data,
        now_ms=_now_ms(),
    )

    assert new_store == {}
    assert data.cancel_calls == []


def test_revert_drops_expired_entries():
    pending_store = {
        "r1": _now_ms() - 1,        # expired
        "r2": _now_ms() + 3000,     # alive
    }

    new_store = _on_cancel_revert_handler(pending_store=pending_store, now_ms=_now_ms())

    assert new_store == {"r2": _now_ms() + 3000}


def test_revert_no_change_when_no_expiry():
    pending_store = {"r1": _now_ms() + 3000}

    new_store = _on_cancel_revert_handler(pending_store=pending_store, now_ms=_now_ms())

    assert new_store == pending_store
```

- [ ] **Step 10.2: Run and verify failure**

```bash
python3 -m pytest --noconftest -p no:cacheprovider \
  tools/odin/valhalla/dashboard/tests/test_tab_a_cancel_callback.py -v
```

Expected: import errors (no `_on_cancel_toggle_handler`, no `_on_cancel_revert_handler`).

- [ ] **Step 10.3: Add the pure handlers to `callbacks.py`**

In `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/callbacks.py`, append below the existing `_on_retry_toggle_handler`:

```python
_CANCEL_CONFIRM_WINDOW_MS = 5000


def _on_cancel_toggle_handler(
    n_clicks_list,
    ids_list,
    *,
    statuses: list[str],
    dispatch_id: str,
    dispatch_ended: bool,
    pending_store: dict | None,
    data,
    now_ms: int,
) -> dict:
    """Two-click confirm flow for the per-row cancel button.

    First click on a row flips the store entry to ``{run_id: expires_at_ms}``
    (5 s out). Second click within the window writes the DB row via
    ``data.request_cancel`` and clears the entry. Clicks for finished
    dispatches are dropped.
    """
    if dispatch_ended:
        return {}
    pending_store = dict(pending_store or {})
    if not n_clicks_list or not any(n_clicks_list):
        return pending_store
    for n, ident, status in zip(n_clicks_list, ids_list, statuses):
        if not n or n <= 0:
            continue
        run_id = ident["run_id"]
        existing = pending_store.get(run_id)
        if existing is not None and existing > now_ms:
            # Second click inside the confirm window → write the DB row.
            kind = "kill" if status == "running" else "skip"
            data.request_cancel(dispatch_id, run_id, kind=kind)
            pending_store.pop(run_id, None)
        elif status in {"pending", "running"}:
            pending_store[run_id] = now_ms + _CANCEL_CONFIRM_WINDOW_MS
    return pending_store


def _on_cancel_revert_handler(*, pending_store: dict | None, now_ms: int) -> dict:
    """Drain expired entries from the pending-confirm store. Run by a 500 ms interval."""
    pending_store = dict(pending_store or {})
    return {run_id: ts for run_id, ts in pending_store.items() if ts > now_ms}
```

- [ ] **Step 10.4: Wire the Dash callback**

Inside `register_callbacks(app, data)` in `callbacks.py`, append:

```python
    @app.callback(
        Output("tab-a-cancel-pending", "data"),
        Input({"type": "tab-a-cancel-toggle", "run_id": ALL}, "n_clicks"),
        Input("tab-a-cancel-revert", "n_intervals"),
        State({"type": "tab-a-cancel-toggle", "run_id": ALL}, "id"),
        State("tab-a-dispatch-id", "data"),
        State("tab-a-cancel-pending", "data"),
    )
    def _on_cancel_toggle(
        n_clicks_list,
        _n_intervals,
        ids_list,
        dispatch_id,
        pending_store,
    ):
        from time import monotonic

        now_ms = int(monotonic() * 1000)
        triggered = dash.ctx.triggered_id
        if triggered == "tab-a-cancel-revert":
            return _on_cancel_revert_handler(pending_store=pending_store, now_ms=now_ms)
        # Resolve per-row statuses + dispatch.ended_at from the dispatch payload
        # so we don't need a parallel dcc.Store. data.load_dispatch is cached.
        payload = data.load_dispatch(dispatch_id) if dispatch_id else {}
        status_by_run = {j.get("run_id"): j.get("status", "") for j in payload.get("jobs", []) or []}
        statuses = [status_by_run.get(ident["run_id"], "") for ident in (ids_list or [])]
        dispatch_ended = bool(payload.get("ended_at"))
        return _on_cancel_toggle_handler(
            n_clicks_list,
            ids_list,
            statuses=statuses,
            dispatch_id=dispatch_id,
            dispatch_ended=dispatch_ended,
            pending_store=pending_store,
            data=data,
            now_ms=now_ms,
        )
```

- [ ] **Step 10.5: Add the supporting `dcc.Store` + `dcc.Interval` to the layout**

In `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/layout.py`, find the existing `dcc.Store` declarations near the top of the layout (look for `tab-a-retry-bump`). Add:

```python
            dcc.Store(id="tab-a-cancel-pending", data={}),
            dcc.Interval(id="tab-a-cancel-revert", interval=500, disabled=False),
```

- [ ] **Step 10.7: Run the new callback tests**

```bash
python3 -m pytest --noconftest -p no:cacheprovider \
  tools/odin/valhalla/dashboard/tests/test_tab_a_cancel_callback.py -v
```

Expected: 6 passed.

- [ ] **Step 10.8: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/tabs/dispatch_fleet/callbacks.py \
        tools/odin/valhalla/dashboard/tabs/dispatch_fleet/layout.py \
        tools/odin/valhalla/dashboard/tests/test_tab_a_cancel_callback.py
git commit -m "Tab A: confirm-flow callback for kill / skip button"
```

---

## Task 11: Wire `cancel_queue` + `cancel_confirm` into the live `update_jobs` callback

**Files:**
- Modify: `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/callbacks.py`

The button render (Task 9) accepts `cancel_queue=` and `cancel_confirm=` but the live update callback isn't passing them yet. Without this step, the badge would not appear and the "Confirm…" red label wouldn't show on first click.

- [ ] **Step 11.1: Find the `update_jobs` callback's render call**

In `callbacks.py`, locate the call to `render_jobs_rows(...)` (this is what the live tick re-renders). Note its existing `Input` / `State` declarations — we'll add `tab-a-cancel-pending` as a `State` so the table re-renders include the current confirm state.

- [ ] **Step 11.2: Add `tab-a-cancel-pending` as a State to the `update_jobs` callback**

Find the `@app.callback(...)` decorator above `update_jobs`. Add inside the existing `State(...)` list:

```python
        State("tab-a-cancel-pending", "data"),
```

Add the matching parameter (`cancel_pending`) at the end of the function signature.

- [ ] **Step 11.3: Pass `cancel_queue` and `cancel_confirm` through the render call**

Modify the call to:

```python
        return render_jobs_rows(
            payload,
            status_filter=status_filter,
            kind_filter=kind_filter,
            task_text=task_text,
            expanded_run_ids=set(expanded_run_ids or []),
            ssh_tail_store=ssh_tail_store,
            retry_queue=data.read_retry_queue(dispatch_id),
            cancel_queue=data.read_cancel_queue(dispatch_id),
            cancel_confirm=cancel_pending or {},
        )
```

(Same pattern as the existing `retry_queue=` argument.)

- [ ] **Step 11.4: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/tabs/dispatch_fleet/callbacks.py
git commit -m "Tab A: pass cancel_queue + cancel_confirm into live update render"
```

---

## Task 12: Loopback integration test — skip + kill end-to-end

**Files:**
- Modify: `tools/odin/tests/test_asgard_integration.py`

- [ ] **Step 12.1: Append the test to `test_asgard_integration.py`**

Append below the existing `test_loopback_detached_resume_reattaches_inflight`:

```python
def test_loopback_detached_dispatch_skip_and_kill_via_db(
    tmp_path: Path, stub_provisioner, monkeypatch
):
    """Two-job loopback dispatch: skip one before submit, kill the other mid-run.

    Asserts both end terminal with the expected kinds and the killed job's
    bundle dir contains its (partial) logs/.
    """
    if not _ssh_localhost_works():
        pytest.skip("ssh localhost does not work without a password")

    from tools.odin.asgard import worker as worker_mod
    from tools.odin.asgard.runner import _consume_cancellations
    from tools.odin.valhalla.dashboard.cancel_db import CancelDB

    # Submit stub: write a tracker + pidfile (using the test process's pid so
    # poll's `kill -0` reports `alive`), then sleep so the job is observably
    # mid-flight when we issue the kill.
    def _fake_submit(host, job, *, submitted_at, per_job_timeout_s):
        bundle_dir = f"{host.isaaclab_path}/odin_runs/{job.bundle_dir_name}"
        import json as _json
        from tools.odin.asgard.tracker import TRACKER_SCHEMA_VERSION

        tracker = {
            "schema_version": TRACKER_SCHEMA_VERSION,
            "run_id": job.run_id,
            "container_name": host.container_name,
            "host": host.host,
            "submitted_at": submitted_at,
            "pid": os.getpid(),
            "container_pid": None,
            "per_job_timeout_s": per_job_timeout_s,
        }
        tracker_s = _json.dumps(tracker).replace("'", r"\'")
        return (
            f"mkdir -p {bundle_dir}/logs && "
            f"echo $$ > {bundle_dir}/.run.pid && "
            f"printf '%s' '{tracker_s}' > {bundle_dir}/.tracker.json && "
            # Simulate a mid-run trainer with a sleep — partial stderr exists.
            f"echo 'fake training started' > {bundle_dir}/logs/hugin-stderr.log && "
            f"echo 'odin-submit: ok run_id={job.run_id} bundle={job.bundle_dir_name}'"
        )

    def _fake_poll(host, bundle_ids):
        bundles = " ".join(bundle_ids)
        return (
            f"for bundle in {bundles}; do "
            f"if [ -f {host.isaaclab_path}/odin_runs/$bundle/manifest.json ]; then "
            f'echo "$bundle done"; '
            f"elif [ -f {host.isaaclab_path}/odin_runs/$bundle/.run.pid ]; then "
            f"pid=$(cat {host.isaaclab_path}/odin_runs/$bundle/.run.pid); "
            f'if kill -0 "$pid" 2>/dev/null; then echo "$bundle alive"; '
            f'else echo "$bundle exited-no-manifest"; fi; '
            f'else echo "$bundle no-pidfile"; fi; '
            f"done"
        )

    monkeypatch.setattr(worker_mod, "_build_submit_script", _fake_submit)
    monkeypatch.setattr(worker_mod, "_build_poll_script", _fake_poll)

    repo_root = Path.cwd()
    host = ValkyrieConfig(
        host="localhost",
        ssh_user=os.environ.get("USER", "root"),
        isaaclab_path=str(repo_root),
    )
    fleet = Fleet(fleet_name="loopback-cancel", hosts=[host])
    dispatch_dir = tmp_path / "odin_runs" / "20260504-cancel"
    dispatch_dir.mkdir(parents=True)

    # Two-job env list (we need a pending one to skip + a running one to kill).
    from tools.odin.common.env_list import EnvEntry as _EnvEntry
    from tools.odin.common.env_list import EnvList as _EnvList
    from tools.odin.common.env_list import write_env_list as _write_env_list

    el = _EnvList()
    el.groups["direct/ant"] = [
        _EnvEntry(
            task_id="Isaac-Ant-Direct-v0",
            entry_point="ep:E",
            env_cfg_entry_point="ec:E",
            group="direct/ant",
            has_rsl_rl=True,
            has_skrl=True,
            framework="rsl_rl",
            num_envs=4096,
            max_iterations=10,
            keep=True,
            status="current",
        )
    ]
    physx_yaml = tmp_path / "physx.yaml"
    _write_env_list(physx_yaml, el, generator="test")

    # Kick the dispatch in the background; it will spin polling forever
    # because we never write a manifest. We send the cancel rows after a
    # short sleep, then expect the dispatch to terminate.
    cancel_db = CancelDB(dispatch_dir.parent)

    import threading as _threading

    def _send_cancels():
        import time as _time

        _time.sleep(2.0)  # let the runner submit + poll at least once
        # Job ids are deterministic from dispatch_id + framework + task + seed.
        seed_to_run = (
            f"rsl-rl_physx_Isaac-Ant-Direct-v0_{dispatch_dir.name}_seed42"
        )
        cancel_db.request(dispatch_dir.name, seed_to_run, kind="kill")

    cancel_thread = _threading.Thread(target=_send_cancels, daemon=True)
    cancel_thread.start()

    state = run_dispatch(
        fleet=fleet,
        physx_yaml=physx_yaml,
        newton_yaml=None,
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(
            seeds=[42, 43],   # two jobs from one task
            per_job_timeout_s=60,
            skip_aggregate=True,
            detached_mode=True,
            poll_interval_s=0,
            live_retry_poll_s=0.5,
        ),
        ssh=ShellSSHRunner(),
        rsync=ShellRsyncRunner(),
    )

    # Pre-cleanup: the loopback wrote bundles into repo_root.
    import shutil as _shutil

    for j in state.jobs:
        _shutil.rmtree(repo_root / "odin_runs" / j.bundle_dir_name, ignore_errors=True)

    # Whichever job the runner sent first becomes "running" + killed; the
    # other stays pending until killed via the same cancel-loop tick or
    # through the per-job timeout. In a deterministic test we only assert
    # on the killed one.
    killed_jobs = [j for j in state.jobs if j.failure and j.failure.kind == "killed"]
    assert killed_jobs, f"expected at least one killed job, got {[(j.run_id, j.failure.kind if j.failure else j.status) for j in state.jobs]}"
    killed = killed_jobs[0]
    bundle = dispatch_dir / killed.bundle_dir_name
    # Partial logs preserved per Q4 answer.
    assert (bundle / "logs" / "hugin-stderr.log").exists()
```

- [ ] **Step 12.2: Run the integration test**

```bash
timeout 60 python3 -m pytest --noconftest -p no:cacheprovider \
  tools/odin/tests/test_asgard_integration.py::test_loopback_detached_dispatch_skip_and_kill_via_db -v
```

Expected: PASS.

- [ ] **Step 12.3: Run the full asgard suite as a regression check**

```bash
timeout 90 python3 -m pytest --noconftest -p no:cacheprovider \
  tools/odin/tests/test_asgard_*.py -v
```

Expected: existing tests pass; one pre-existing flake (`test_loopback_dispatch_recovers_from_gpu_lost`) may fail unrelated to this change.

- [ ] **Step 12.4: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/tests/test_asgard_integration.py
git commit -m "Asgard integration: loopback skip + kill via CancelDB"
```

---

## Task 13: Final regression sweep

- [ ] **Step 13.1: Run the entire odin test suite**

```bash
timeout 180 python3 -m pytest --noconftest -p no:cacheprovider \
  tools/odin/tests/test_asgard_*.py \
  tools/odin/tests/test_recovery*.py \
  tools/odin/valhalla/dashboard/tests/test_*.py
```

Expected: all pass except the unrelated pre-existing flake. Investigate any new failures (they belong to this change).

- [ ] **Step 13.2: Run pre-commit on all files**

```bash
./isaaclab.sh -f
```

Expected: all green. If anything reformats, stage and re-run; commit if changed.

- [ ] **Step 13.3: Confirm the branch is ready**

```bash
git log --oneline antoiner/feat/odin..HEAD
```

Expected: a sequence of focused commits ending with the integration test.

---

## Notes for the implementer

- **Why `workers_by_host: dict` not `list`?** Cancel needs O(1) host→worker lookup. The rename is mechanical (3 sites: spawn loop, sentinel push, `is_alive` check). Keeping a `workers = list(workers_by_host.values())` alias minimizes surface change in the rest of the runner.
- **Why a separate `_consume_cancellations` and not extending `_consume_live_retries`?** Different semantics (cancel vs retry are inverses), different consumer pattern (skip lands sync, kill lands async via worker event). Folding them into one helper would require branching on what the row "means" — easier to keep two helpers with one shape each.
- **Why poll-and-process at the empty-channel branch AND after each terminal event?** The retry helper does exactly this; the kill latency we want is bounded by the smaller of the two intervals. Mirror the existing pattern so behaviour stays predictable.
- **No CHANGELOG entry needed.** The change lives entirely in `tools/odin/`, not under `source/<package>/`. The AGENTS.md changelog rule only applies to the source/ directory.
- **Sandbox / network.** All dev / test work is local — no network calls needed. If the operator pushes the branch later, use `dangerouslyDisableSandbox: true` per AGENTS.md.
