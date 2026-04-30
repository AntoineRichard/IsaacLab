# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``odin-retry`` CLI for inspecting and editing the retry queue."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from tools.odin.valhalla.dashboard.retry_db import RetryDB, RetryRow

__all__ = ["main", "parse_args"]


_COLUMNS = [
    "dispatch_id",
    "run_id",
    "queued_at",
    "note",
    "retried_at",
    "retry_dispatch_id",
    "retry_outcome",
    "retry_failure_kind",
]


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse ``odin-retry`` CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="odin-retry",
        description="Inspect and edit the Odin retry queue.",
    )
    parser.add_argument("--runs_root", "--runs-root", dest="runs_root", type=Path, default=Path("odin_runs"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List queued retries.")
    list_parser.add_argument("--dispatch", default=None, help="Restrict output to one dispatch_id.")
    list_parser.add_argument("--all", action="store_true", help="Include consumed retry history.")

    queue_parser = subparsers.add_parser("queue", help="Queue one run_id for retry.")
    queue_parser.add_argument("dispatch_id")
    queue_parser.add_argument("run_id")
    queue_parser.add_argument("--note", default=None)

    remove_parser = subparsers.add_parser("remove", help="Remove one retry row.")
    remove_parser.add_argument("dispatch_id")
    remove_parser.add_argument("run_id")

    subparsers.add_parser("status", help="Print pending retry counts.")

    export_parser = subparsers.add_parser("export-resume-cmd", help="Print an odin-dispatch retry command.")
    export_parser.add_argument("dispatch_id")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the ``odin-retry`` CLI."""
    ns = parse_args(argv if argv is not None else sys.argv[1:])
    db = RetryDB(ns.runs_root)
    if ns.command == "list":
        return _cmd_list(db, dispatch_id=ns.dispatch, include_all=ns.all)
    if ns.command == "queue":
        return _cmd_queue(db, ns.dispatch_id, ns.run_id, note=ns.note)
    if ns.command == "remove":
        return _cmd_remove(db, ns.dispatch_id, ns.run_id)
    if ns.command == "status":
        return _cmd_status(db)
    if ns.command == "export-resume-cmd":
        return _cmd_export_resume_cmd(db, ns.dispatch_id)
    raise AssertionError(f"unhandled command {ns.command!r}")


def _cmd_list(db: RetryDB, *, dispatch_id: str | None, include_all: bool) -> int:
    pending_only = not include_all
    rows = (
        db.list_for_dispatch(dispatch_id, pending_only=pending_only)
        if dispatch_id is not None
        else db.list_all(pending_only=pending_only)
    )
    print("\t".join(_COLUMNS))
    for row in rows:
        print(_format_row(row))
    return 0


def _cmd_queue(db: RetryDB, dispatch_id: str, run_id: str, *, note: str | None) -> int:
    if run_id not in db.read_pending(dispatch_id):
        db.toggle(dispatch_id, run_id, note=note)
    print(f"queued\t{dispatch_id}\t{run_id}")
    return 0


def _cmd_remove(db: RetryDB, dispatch_id: str, run_id: str) -> int:
    db.remove(dispatch_id, run_id)
    print(f"removed\t{dispatch_id}\t{run_id}")
    return 0


def _cmd_status(db: RetryDB) -> int:
    rows = db.list_all(pending_only=True)
    counts = Counter(row.dispatch_id for row in rows)
    print(f"pending\t{len(rows)}")
    print(f"dispatches\t{len(counts)}")
    for dispatch_id, count in sorted(counts.items()):
        print(f"{dispatch_id}\t{count}")
    return 0


def _cmd_export_resume_cmd(db: RetryDB, dispatch_id: str) -> int:
    pending = sorted(db.read_pending(dispatch_id))
    if not pending:
        print(f"odin-retry: no pending retries for {dispatch_id}", file=sys.stderr)
        return 1
    print(f"odin-dispatch --resume {dispatch_id} --retry-failed={','.join(pending)}")
    return 0


def _format_row(row: RetryRow) -> str:
    values = [
        row.dispatch_id,
        row.run_id,
        row.queued_at,
        row.note or "",
        row.retried_at or "",
        row.retry_dispatch_id or "",
        row.retry_outcome or "",
        row.retry_failure_kind or "",
    ]
    return "\t".join(values)


if __name__ == "__main__":
    sys.exit(main())
