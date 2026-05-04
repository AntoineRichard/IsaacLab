# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DispatchState — on-disk representation of an Asgard dispatch.

``dispatch.json`` lives at ``<dispatch_dir>/dispatch.json`` and is
rewritten atomically (temp-file + rename) after every state transition
and on a periodic heartbeat.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tools.odin.asgard.jobs import FailureInfo, JobEntry, SkippedEntry

__all__ = [
    "DispatchState",
    "FleetSnapshot",
    "QuarantinedHost",
    "SCHEMA_VERSION",
    "read_dispatch_state",
    "write_dispatch_state",
    "reset_in_flight_to_pending",
]


SCHEMA_VERSION = "1.4"
_DISPATCH_FILENAME = "dispatch.json"


@dataclass
class FleetSnapshot:
    """Per-host live state, written into ``dispatch.json``."""

    host: str
    status: str  # "idle" | "busy" | "down"
    current_run_id: str | None = None
    last_error: str | None = None


@dataclass
class QuarantinedHost:
    """Record of a host quarantined by the circuit-breaker.

    Written into the top-level ``quarantined_hosts`` list in
    ``dispatch.json`` when a worker trips the consecutive-failure
    threshold (``kind="circuit_breaker"``).
    """

    host: str
    reason: str  # the FailureInfo.kind that triggered the quarantine
    last_run_id: str  # the run_id whose failure tripped the threshold
    at: str  # UTC ISO-8601 timestamp


@dataclass
class DispatchState:
    """Complete on-disk state for one dispatch."""

    schema_version: str
    dispatch_id: str
    started_at: str  # UTC ISO-8601
    ended_at: str | None
    seeds: list[int]
    commit_sha: str
    fleet: list[FleetSnapshot]
    jobs: list[JobEntry]
    skipped: list[SkippedEntry] = field(default_factory=list)
    quarantined_hosts: list[QuarantinedHost] = field(default_factory=list)


# --- Serialization -----------------------------------------------------------


def _job_to_dict(j: JobEntry) -> dict[str, Any]:
    d: dict[str, Any] = {
        "run_id": j.run_id,
        "task_id": j.task_id,
        "framework": j.framework,
        "backend": j.backend,
        "num_envs": j.num_envs,
        "max_iterations": j.max_iterations,
        "seed": j.seed,
        "bundle_dir_name": j.bundle_dir_name,
        "status": j.status,
        "assigned_to": j.assigned_to,
        "attempts": j.attempts,
        "started_at": j.started_at,
        "ended_at": j.ended_at,
        "preferred_not": sorted(j.preferred_not),
        "per_job_timeout_s": j.per_job_timeout_s,
    }
    if j.failure is None:
        d["failure"] = None
    else:
        d["failure"] = {
            "kind": j.failure.kind,
            "message": j.failure.message,
            "details": j.failure.details,
        }
    return d


def _job_from_dict(d: dict[str, Any]) -> JobEntry:
    failure = None
    if d.get("failure") is not None:
        failure = FailureInfo(
            kind=str(d["failure"]["kind"]),
            message=str(d["failure"].get("message", "")),
            details=dict(d["failure"].get("details") or {}),
        )
    return JobEntry(
        run_id=str(d["run_id"]),
        task_id=str(d["task_id"]),
        framework=str(d["framework"]),
        backend=str(d["backend"]),
        num_envs=int(d["num_envs"]),
        max_iterations=int(d["max_iterations"]),
        seed=int(d["seed"]),
        bundle_dir_name=str(d["bundle_dir_name"]),
        status=str(d.get("status", "pending")),
        assigned_to=d.get("assigned_to"),
        attempts=int(d.get("attempts", 0)),
        failure=failure,
        preferred_not=set(d.get("preferred_not") or []),
        started_at=d.get("started_at"),
        ended_at=d.get("ended_at"),
        per_job_timeout_s=d.get("per_job_timeout_s"),
    )


def _skipped_to_dict(s: SkippedEntry) -> dict[str, Any]:
    return {
        "task_id": s.task_id,
        "framework": s.framework,
        "backend": s.backend,
        "seed": s.seed,
        "reason": s.reason,
        "presets_available": list(s.presets_available),
        "native_backend": s.native_backend,
    }


def _skipped_from_dict(d: dict[str, Any]) -> SkippedEntry:
    return SkippedEntry(
        task_id=str(d["task_id"]),
        framework=str(d["framework"]),
        backend=str(d["backend"]),
        seed=int(d["seed"]),
        reason=str(d.get("reason", "preset_unsupported")),
        presets_available=list(d.get("presets_available") or []),
        native_backend=d.get("native_backend"),
    )


def _state_to_dict(s: DispatchState) -> dict[str, Any]:
    return {
        "schema_version": s.schema_version,
        "dispatch_id": s.dispatch_id,
        "started_at": s.started_at,
        "ended_at": s.ended_at,
        "seeds": list(s.seeds),
        "commit_sha": s.commit_sha,
        "fleet": [
            {
                "host": f.host,
                "status": f.status,
                "current_run_id": f.current_run_id,
                "last_error": f.last_error,
            }
            for f in s.fleet
        ],
        "jobs": [_job_to_dict(j) for j in s.jobs],
        "skipped": [_skipped_to_dict(sk) for sk in s.skipped],
        "quarantined_hosts": [
            {"host": q.host, "reason": q.reason, "last_run_id": q.last_run_id, "at": q.at} for q in s.quarantined_hosts
        ],
    }


def _schema_version_compatible(got: str, expected: str) -> bool:
    """Return True iff ``got`` and ``expected`` share the same major version.

    Additive minor-version bumps (e.g. 1.0 → 1.1) must be tolerated by
    readers per Odin's schema rules in ``docs/odin/architecture.md`` §5;
    a major-version change (1.x → 2.x) is breaking and rejected.
    """
    if not got:
        return False
    try:
        return got.split(".", 1)[0] == expected.split(".", 1)[0]
    except (AttributeError, IndexError):
        return False


def _state_from_dict(d: dict[str, Any]) -> DispatchState:
    got_schema = str(d.get("schema_version", ""))
    if not _schema_version_compatible(got_schema, SCHEMA_VERSION):
        raise ValueError(
            f"Unsupported dispatch.json schema_version {got_schema!r} "
            f"(expected major-compatible with {SCHEMA_VERSION!r})"
        )
    return DispatchState(
        schema_version=got_schema,
        dispatch_id=str(d["dispatch_id"]),
        started_at=str(d["started_at"]),
        ended_at=d.get("ended_at"),
        seeds=[int(s) for s in d.get("seeds") or []],
        commit_sha=str(d.get("commit_sha", "")),
        fleet=[
            FleetSnapshot(
                host=str(f["host"]),
                status=str(f.get("status", "idle")),
                current_run_id=f.get("current_run_id"),
                last_error=f.get("last_error"),
            )
            for f in (d.get("fleet") or [])
        ],
        jobs=[_job_from_dict(j) for j in (d.get("jobs") or [])],
        skipped=[_skipped_from_dict(s) for s in (d.get("skipped") or [])],
        quarantined_hosts=[
            QuarantinedHost(
                host=str(q["host"]),
                reason=str(q["reason"]),
                last_run_id=str(q["last_run_id"]),
                at=str(q["at"]),
            )
            for q in (d.get("quarantined_hosts") or [])
        ],
    )


# --- I/O ---------------------------------------------------------------------


def read_dispatch_state(dispatch_dir: Path) -> DispatchState | None:
    """Read ``<dispatch_dir>/dispatch.json`` into a :class:`DispatchState`.

    Returns ``None`` when the file does not exist (e.g. first-time dispatch).
    Raises :class:`ValueError` when it exists but declares an unsupported
    ``schema_version``.
    """
    path = dispatch_dir / _DISPATCH_FILENAME
    if not path.exists():
        return None
    with path.open("r") as fh:
        payload = json.load(fh)
    return _state_from_dict(payload)


def write_dispatch_state(dispatch_dir: Path, state: DispatchState) -> None:
    """Atomically rewrite ``<dispatch_dir>/dispatch.json``.

    Writes to a sibling temporary file then ``os.replace``s over the final
    path, so a concurrent reader never observes a truncated file.
    """
    dispatch_dir.mkdir(parents=True, exist_ok=True)
    payload = _state_to_dict(state)
    fd, tmp_path_str = tempfile.mkstemp(prefix=".dispatch_", suffix=".json.tmp", dir=str(dispatch_dir))
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=False)
        os.replace(tmp_path, dispatch_dir / _DISPATCH_FILENAME)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


# --- Resume helpers ----------------------------------------------------------


def reset_in_flight_to_pending(state: DispatchState) -> None:
    """Flip ``running`` / ``assigned`` jobs back to ``pending`` for resume.

    Called in-place on the loaded state before a resumed dispatch starts its
    workers. ``completed`` and ``failed`` jobs are left alone — a failed job
    is only re-attempted via an explicit ``--retry-failed`` escape hatch.
    """
    for j in state.jobs:
        if j.status in ("running", "assigned"):
            j.status = "pending"
            j.assigned_to = None
            j.started_at = None
