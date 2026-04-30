# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the SQLite-backed Odin retry queue."""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path

import pytest

from tools.odin.valhalla.dashboard.retry_db import RetryDB


def test_fresh_db_creates_schema_and_pragmas(tmp_path: Path):
    db = RetryDB(tmp_path)

    assert db.read_pending("20260430-110509") == set()

    with sqlite3.connect(tmp_path / ".retry.sqlite") as con:
        table_names = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        index_names = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        user_version = con.execute("PRAGMA user_version").fetchone()[0]

    with db._connect() as con:
        journal_mode = con.execute("PRAGMA journal_mode").fetchone()[0]
        foreign_keys = con.execute("PRAGMA foreign_keys").fetchone()[0]

    assert "retries" in table_names
    assert {"idx_retries_pending", "idx_retries_global_pending"} <= index_names
    assert user_version == 1
    assert journal_mode == "wal"
    assert foreign_keys == 1


def test_toggle_adds_then_removes(tmp_path: Path):
    db = RetryDB(tmp_path)

    assert db.toggle("20260430-110509", "run-a") == {"run-a"}
    assert db.read_pending("20260430-110509") == {"run-a"}
    assert db.toggle("20260430-110509", "run-a") == set()
    assert db.read_pending("20260430-110509") == set()
    assert db.list_all() == []


def test_toggle_stores_note(tmp_path: Path):
    db = RetryDB(tmp_path)

    db.toggle("20260430-110509", "run-a", note="network blip")

    rows = db.list_all()
    assert len(rows) == 1
    assert rows[0].note == "network blip"


def test_toggle_repeat_after_consume_re_queues(tmp_path: Path):
    db = RetryDB(tmp_path)
    db.toggle("20260430-110509", "run-a")
    db.mark_consumed(
        "20260430-110509",
        "run-a",
        retry_dispatch_id="20260430-120000",
        outcome="failed",
        failure_kind="hugin_crash",
    )

    assert db.toggle("20260430-110509", "run-a") == {"run-a"}

    row = db.list_all()[0]
    assert row.retried_at is None
    assert row.retry_dispatch_id is None
    assert row.retry_outcome is None
    assert row.retry_failure_kind is None


def test_read_pending_excludes_consumed(tmp_path: Path):
    db = RetryDB(tmp_path)
    db.toggle("20260430-110509", "run-a")
    db.toggle("20260430-110509", "run-b")
    db.mark_consumed("20260430-110509", "run-a", retry_dispatch_id="20260430-120000", outcome="completed")

    assert db.read_pending("20260430-110509") == {"run-b"}


def test_list_all_can_filter_pending(tmp_path: Path):
    db = RetryDB(tmp_path)
    db.toggle("20260430-110509", "run-a")
    db.toggle("20260430-110509", "run-b")
    db.mark_consumed("20260430-110509", "run-a", retry_dispatch_id="20260430-120000", outcome="completed")

    assert [row.run_id for row in db.list_all()] == ["run-a", "run-b"]
    assert [row.run_id for row in db.list_all(pending_only=True)] == ["run-b"]


def test_list_for_dispatch_scopes_rows(tmp_path: Path):
    db = RetryDB(tmp_path)
    db.toggle("20260430-110509", "run-a")
    db.toggle("20260430-120000", "run-b")

    rows = db.list_for_dispatch("20260430-110509")

    assert [(row.dispatch_id, row.run_id) for row in rows] == [("20260430-110509", "run-a")]


def test_mark_consumed_records_outcome_and_kind(tmp_path: Path):
    db = RetryDB(tmp_path)
    db.toggle("20260430-110509", "run-a")

    db.mark_consumed(
        "20260430-110509",
        "run-a",
        retry_dispatch_id="20260430-120000",
        outcome="failed",
        failure_kind="gpu_lost",
    )

    row = db.list_all()[0]
    assert row.retried_at is not None
    assert row.retry_dispatch_id == "20260430-120000"
    assert row.retry_outcome == "failed"
    assert row.retry_failure_kind == "gpu_lost"


def test_mark_consumed_rejects_invalid_outcome(tmp_path: Path):
    db = RetryDB(tmp_path)
    db.toggle("20260430-110509", "run-a")

    with pytest.raises(ValueError, match="outcome"):
        db.mark_consumed("20260430-110509", "run-a", retry_dispatch_id="20260430-120000", outcome="skipped")


def test_remove_hard_deletes(tmp_path: Path):
    db = RetryDB(tmp_path)
    db.toggle("20260430-110509", "run-a")

    db.remove("20260430-110509", "run-a")

    assert db.list_all() == []
    assert db.read_pending("20260430-110509") == set()


def test_concurrent_toggle_serialises(tmp_path: Path):
    db = RetryDB(tmp_path)
    dispatch_id = "20260430-110509"
    run_id = "run-a"
    toggle_count = 100
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            local_db = RetryDB(tmp_path)
            barrier.wait(timeout=5)
            for _ in range(toggle_count):
                local_db.toggle(dispatch_id, run_id)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker), threading.Thread(target=worker)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    assert db.read_pending(dispatch_id) == set()


def test_legacy_txt_imported_on_first_connect(tmp_path: Path):
    dispatch_dir = tmp_path / "20260430-110509"
    dispatch_dir.mkdir(parents=True)
    txt = dispatch_dir / "retry_queue.txt"
    txt.write_text("run-b\n\nrun-a\nrun-a\n")
    os.utime(txt, (1_777_548_000, 1_777_548_000))

    db = RetryDB(tmp_path)

    assert db.read_pending("20260430-110509") == {"run-a", "run-b"}
    assert txt.exists()
    rows = db.list_for_dispatch("20260430-110509")
    assert {row.queued_at for row in rows} == {"2026-04-30T11:20:00Z"}


def test_legacy_txt_import_skipped_when_db_nonempty(tmp_path: Path):
    db = RetryDB(tmp_path)
    db.toggle("20260430-110509", "db-run")
    dispatch_dir = tmp_path / "20260430-120000"
    dispatch_dir.mkdir(parents=True)
    (dispatch_dir / "retry_queue.txt").write_text("legacy-run\n")

    reopened = RetryDB(tmp_path)

    assert reopened.read_pending("20260430-110509") == {"db-run"}
    assert reopened.read_pending("20260430-120000") == set()


def test_rows_are_sorted_for_stable_cli_output(tmp_path: Path):
    db = RetryDB(tmp_path)
    db.toggle("20260430-120000", "z")
    db.toggle("20260430-110509", "b")
    db.toggle("20260430-110509", "a")

    assert [(row.dispatch_id, row.run_id) for row in db.list_all(pending_only=True)] == [
        ("20260430-110509", "a"),
        ("20260430-110509", "b"),
        ("20260430-120000", "z"),
    ]
