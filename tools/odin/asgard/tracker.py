# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-bundle tracker file written by the detached-submit inner script.

A ``.tracker.json`` lives next to ``manifest.json`` inside each bundle on
the remote host. The dispatcher uses it to (a) confirm that a submit
landed on the remote, and (b) re-attach orphaned in-flight jobs after a
``--resume``.

The on-disk schema is intentionally tiny: dispatcher-known fields are
filled by the submit-script HEREDOC, and the inner script substitutes the
detached process's host PID via ``$$``. ``manifest.json`` remains the
source of truth for run outputs; the tracker only carries enough to find
the live process and classify it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "TRACKER_FILENAME",
    "TRACKER_SCHEMA_VERSION",
    "Tracker",
    "read_tracker",
    "validate_tracker_payload",
    "write_tracker",
]


TRACKER_FILENAME = ".tracker.json"
TRACKER_SCHEMA_VERSION = "1.0"

_REQUIRED_FIELDS = ("run_id", "container_name", "host", "submitted_at", "pid", "per_job_timeout_s")


@dataclass(frozen=True)
class Tracker:
    """Snapshot of a detached-submit invocation, persisted on the remote host."""

    run_id: str
    container_name: str
    host: str
    submitted_at: str
    pid: int
    per_job_timeout_s: int
    container_pid: int | None = None
    schema_version: str = TRACKER_SCHEMA_VERSION


def _schema_major_compatible(got: str, expected: str = TRACKER_SCHEMA_VERSION) -> bool:
    if not got:
        return False
    try:
        return got.split(".", 1)[0] == expected.split(".", 1)[0]
    except (AttributeError, IndexError):
        return False


def validate_tracker_payload(payload: dict[str, Any]) -> None:
    """Raise :class:`ValueError` when ``payload`` doesn't look like a tracker.

    Args:
        payload: Decoded JSON object as read from ``.tracker.json``.

    Raises:
        ValueError: When required fields are missing, the schema_version is
            major-incompatible, or ``pid`` / ``per_job_timeout_s`` are not
            integers.
    """
    got_schema = str(payload.get("schema_version", ""))
    if not _schema_major_compatible(got_schema):
        raise ValueError(
            f"unsupported tracker schema_version {got_schema!r} (expected major-compatible with"
            f" {TRACKER_SCHEMA_VERSION!r})"
        )
    for field in _REQUIRED_FIELDS:
        if field not in payload:
            raise ValueError(f"tracker payload missing required field {field!r}")
    if not isinstance(payload["pid"], int):
        raise ValueError(f"tracker pid must be int, got {type(payload['pid']).__name__}")
    if not isinstance(payload["per_job_timeout_s"], int):
        raise ValueError(f"tracker per_job_timeout_s must be int, got {type(payload['per_job_timeout_s']).__name__}")


def write_tracker(bundle_dir: Path, tracker: Tracker) -> None:
    """Write ``tracker`` to ``<bundle_dir>/.tracker.json`` (pretty JSON).

    Used by the integration test harness and reconcile-side fixtures; in
    production the inner submit script writes the file via shell HEREDOC
    (the dispatcher never touches the remote tracker directly).

    Args:
        bundle_dir: Bundle directory to write into. Created if missing.
        tracker: Snapshot to persist.
    """
    bundle_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(tracker)
    (bundle_dir / TRACKER_FILENAME).write_text(json.dumps(payload, indent=2))


def read_tracker(bundle_dir: Path) -> Tracker | None:
    """Read ``<bundle_dir>/.tracker.json`` into a :class:`Tracker`.

    Args:
        bundle_dir: Directory that may contain a tracker file.

    Returns:
        The parsed :class:`Tracker`, or ``None`` when the file is absent.

    Raises:
        ValueError: When the file is present but not valid JSON, or when
            the payload fails :func:`validate_tracker_payload`.
    """
    path = bundle_dir / TRACKER_FILENAME
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"tracker {path} is not valid JSON: {exc}") from exc
    validate_tracker_payload(payload)
    return Tracker(
        run_id=str(payload["run_id"]),
        container_name=str(payload["container_name"]),
        host=str(payload["host"]),
        submitted_at=str(payload["submitted_at"]),
        pid=int(payload["pid"]),
        per_job_timeout_s=int(payload["per_job_timeout_s"]),
        container_pid=(int(payload["container_pid"]) if payload.get("container_pid") is not None else None),
        schema_version=str(payload.get("schema_version", TRACKER_SCHEMA_VERSION)),
    )
