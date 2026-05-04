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
