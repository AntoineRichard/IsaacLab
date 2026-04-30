# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ``odin-retry`` CLI."""

from __future__ import annotations

from pathlib import Path

from tools.odin.valhalla.dashboard import retry_cli
from tools.odin.valhalla.dashboard.retry_db import RetryDB


def test_list_pending_outputs_tsv(tmp_path: Path, capsys):
    db = RetryDB(tmp_path)
    db.toggle("20260430-110509", "run-a", note="network blip")
    db.toggle("20260430-110509", "run-b")

    rc = retry_cli.main(["--runs-root", str(tmp_path), "list"])

    assert rc == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].split("\t") == [
        "dispatch_id",
        "run_id",
        "queued_at",
        "note",
        "retried_at",
        "retry_dispatch_id",
        "retry_outcome",
        "retry_failure_kind",
    ]
    assert lines[1].split("\t")[:4] == ["20260430-110509", "run-a", db.list_all()[0].queued_at, "network blip"]
    assert lines[2].split("\t")[:2] == ["20260430-110509", "run-b"]


def test_list_dispatch_filters_and_all_includes_history(tmp_path: Path, capsys):
    db = RetryDB(tmp_path)
    db.toggle("20260430-110509", "run-a")
    db.toggle("20260430-120000", "run-b")
    db.mark_consumed("20260430-110509", "run-a", retry_dispatch_id="20260430-130000", outcome="completed")

    rc = retry_cli.main(["--runs-root", str(tmp_path), "list", "--dispatch", "20260430-110509", "--all"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "run-a" in output
    assert "run-b" not in output


def test_queue_and_remove_round_trip(tmp_path: Path, capsys):
    rc = retry_cli.main(
        ["--runs-root", str(tmp_path), "queue", "20260430-110509", "run-a", "--note", "operator requested"]
    )

    assert rc == 0
    assert "queued\t20260430-110509\trun-a" in capsys.readouterr().out
    assert RetryDB(tmp_path).read_pending("20260430-110509") == {"run-a"}

    rc = retry_cli.main(["--runs-root", str(tmp_path), "remove", "20260430-110509", "run-a"])

    assert rc == 0
    assert "removed\t20260430-110509\trun-a" in capsys.readouterr().out
    assert RetryDB(tmp_path).read_pending("20260430-110509") == set()


def test_export_resume_cmd_emits_csv_in_alphabetical_order(tmp_path: Path, capsys):
    db = RetryDB(tmp_path)
    db.toggle("20260430-110509", "run-b")
    db.toggle("20260430-110509", "run-a")

    rc = retry_cli.main(["--runs-root", str(tmp_path), "export-resume-cmd", "20260430-110509"])

    assert rc == 0
    assert capsys.readouterr().out.strip() == "odin-dispatch --resume 20260430-110509 --retry-failed=run-a,run-b"


def test_export_resume_cmd_reports_empty_queue(tmp_path: Path, capsys):
    rc = retry_cli.main(["--runs-root", str(tmp_path), "export-resume-cmd", "20260430-110509"])

    assert rc == 1
    assert "no pending retries for 20260430-110509" in capsys.readouterr().err


def test_status_summarises(tmp_path: Path, capsys):
    db = RetryDB(tmp_path)
    db.toggle("20260430-110509", "run-a")
    db.toggle("20260430-110509", "run-b")
    db.toggle("20260430-120000", "run-c")
    db.mark_consumed("20260430-120000", "run-c", retry_dispatch_id="20260430-130000", outcome="failed")

    rc = retry_cli.main(["--runs-root", str(tmp_path), "status"])

    assert rc == 0
    assert capsys.readouterr().out.strip().splitlines() == [
        "pending\t2",
        "dispatches\t1",
        "20260430-110509\t2",
    ]


def test_runs_root_accepts_snake_case_spelling(tmp_path: Path, capsys):
    RetryDB(tmp_path).toggle("20260430-110509", "run-a")

    rc = retry_cli.main(["--runs_root", str(tmp_path), "status"])

    assert rc == 0
    assert "pending\t1" in capsys.readouterr().out
