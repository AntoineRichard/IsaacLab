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
