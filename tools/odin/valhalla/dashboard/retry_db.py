# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SQLite-backed retry queue for Odin dispatches."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["RetryDB", "RetryRow"]


_DB_NAME = ".retry.sqlite"
_VALID_OUTCOMES = {"completed", "failed"}
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
"""
}


@dataclass(frozen=True)
class RetryRow:
    """One row in the Odin retry queue."""

    dispatch_id: str
    run_id: str
    queued_at: str
    note: str | None
    retried_at: str | None
    retry_dispatch_id: str | None
    retry_outcome: str | None
    retry_failure_kind: str | None


class RetryDB:
    """SQLite-backed retry queue rooted at an ``odin_runs`` directory."""

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
        first_connect = not self._db_path.exists()
        con = sqlite3.connect(self._db_path, timeout=30.0, isolation_level=None)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA busy_timeout = 30000")
        with _CONNECT_LOCK:
            con.execute("PRAGMA journal_mode = WAL")
            con.execute("PRAGMA foreign_keys = ON")
            _migrate(con)
            if first_connect:
                _maybe_import_legacy(con, self._runs_root)
        return con

    def read_pending(self, dispatch_id: str) -> set[str]:
        """Return pending retry ``run_id`` values for ``dispatch_id``."""
        with closing(self._connect()) as con:
            rows = con.execute(
                """
                SELECT run_id FROM retries
                WHERE dispatch_id = ? AND retried_at IS NULL
                ORDER BY run_id
                """,
                (dispatch_id,),
            ).fetchall()
        return {str(row["run_id"]) for row in rows}

    def toggle(self, dispatch_id: str, run_id: str, *, note: str | None = None) -> set[str]:
        """Toggle a pending retry row and return the dispatch's new pending set."""
        with closing(self._connect()) as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                """
                SELECT 1 FROM retries
                WHERE dispatch_id = ? AND run_id = ? AND retried_at IS NULL
                """,
                (dispatch_id, run_id),
            ).fetchone()
            if row is not None:
                con.execute("DELETE FROM retries WHERE dispatch_id = ? AND run_id = ?", (dispatch_id, run_id))
            else:
                con.execute(
                    """
                    INSERT OR REPLACE INTO retries(
                        dispatch_id,
                        run_id,
                        queued_at,
                        note,
                        retried_at,
                        retry_dispatch_id,
                        retry_outcome,
                        retry_failure_kind
                    )
                    VALUES (?, ?, ?, ?, NULL, NULL, NULL, NULL)
                    """,
                    (dispatch_id, run_id, _now_iso(), note),
                )
            rows = con.execute(
                """
                SELECT run_id FROM retries
                WHERE dispatch_id = ? AND retried_at IS NULL
                ORDER BY run_id
                """,
                (dispatch_id,),
            ).fetchall()
            con.commit()
        return {str(row["run_id"]) for row in rows}

    def list_all(self, *, pending_only: bool = False) -> list[RetryRow]:
        """Return retry rows across all dispatches."""
        where = "WHERE retried_at IS NULL" if pending_only else ""
        with closing(self._connect()) as con:
            rows = con.execute(
                f"""
                SELECT * FROM retries
                {where}
                ORDER BY dispatch_id, run_id
                """
            ).fetchall()
        return [_row_from_sqlite(row) for row in rows]

    def list_for_dispatch(self, dispatch_id: str, *, pending_only: bool = False) -> list[RetryRow]:
        """Return retry rows for one dispatch."""
        where_pending = "AND retried_at IS NULL" if pending_only else ""
        with closing(self._connect()) as con:
            rows = con.execute(
                f"""
                SELECT * FROM retries
                WHERE dispatch_id = ?
                {where_pending}
                ORDER BY run_id
                """,
                (dispatch_id,),
            ).fetchall()
        return [_row_from_sqlite(row) for row in rows]

    def mark_consumed(
        self,
        dispatch_id: str,
        run_id: str,
        *,
        retry_dispatch_id: str,
        outcome: str,
        failure_kind: str | None = None,
    ) -> None:
        """Record that a queued retry was consumed by a retry attempt."""
        if outcome not in _VALID_OUTCOMES:
            raise ValueError(f"outcome must be one of {sorted(_VALID_OUTCOMES)}, got {outcome!r}")
        with closing(self._connect()) as con:
            con.execute(
                """
                UPDATE retries
                SET retried_at = ?,
                    retry_dispatch_id = ?,
                    retry_outcome = ?,
                    retry_failure_kind = ?
                WHERE dispatch_id = ? AND run_id = ?
                """,
                (_now_iso(), retry_dispatch_id, outcome, failure_kind, dispatch_id, run_id),
            )

    def remove(self, dispatch_id: str, run_id: str) -> None:
        """Hard-delete one retry row."""
        with closing(self._connect()) as con:
            con.execute("DELETE FROM retries WHERE dispatch_id = ? AND run_id = ?", (dispatch_id, run_id))


def _migrate(con: sqlite3.Connection) -> None:
    current = int(con.execute("PRAGMA user_version").fetchone()[0])
    for version, sql in sorted(_MIGRATIONS.items()):
        if version > current:
            con.executescript(sql)
            con.execute(f"PRAGMA user_version = {version}")
    con.commit()


def _maybe_import_legacy(con: sqlite3.Connection, runs_root: Path) -> None:
    count = int(con.execute("SELECT COUNT(*) FROM retries").fetchone()[0])
    if count > 0:
        return
    rows: list[tuple[str, str, str]] = []
    for txt in sorted(runs_root.glob("*/retry_queue.txt")):
        queued_at = _file_mtime_iso(txt)
        dispatch_id = txt.parent.name
        for line in txt.read_text().splitlines():
            run_id = line.strip()
            if run_id:
                rows.append((dispatch_id, run_id, queued_at))
    if not rows:
        return
    con.executemany(
        "INSERT OR IGNORE INTO retries(dispatch_id, run_id, queued_at) VALUES (?, ?, ?)",
        rows,
    )
    con.commit()


def _row_from_sqlite(row: sqlite3.Row) -> RetryRow:
    values: dict[str, Any] = dict(row)
    return RetryRow(
        dispatch_id=str(values["dispatch_id"]),
        run_id=str(values["run_id"]),
        queued_at=str(values["queued_at"]),
        note=values["note"],
        retried_at=values["retried_at"],
        retry_dispatch_id=values["retry_dispatch_id"],
        retry_outcome=values["retry_outcome"],
        retry_failure_kind=values["retry_failure_kind"],
    )


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _file_mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
