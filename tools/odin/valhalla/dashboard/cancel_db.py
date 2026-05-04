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


def _migrate(con: sqlite3.Connection) -> None:
    current = int(con.execute("PRAGMA user_version").fetchone()[0])
    for version, sql in sorted(_MIGRATIONS.items()):
        if version > current:
            con.executescript(sql)
            con.execute(f"PRAGMA user_version = {version}")
    con.commit()


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


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
