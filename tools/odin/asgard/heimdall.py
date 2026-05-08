# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Heimdall — periodic fleet watcher for the Asgard dispatcher.

Runs as a daemon thread inside :func:`~tools.odin.asgard.runner.run_dispatch`.
Periodically re-probes each Valkyrie's GPU presence (``nvidia-smi -L``) and
computes stale jobs from the dispatcher's in-memory
:class:`~tools.odin.asgard.state.DispatchState`. Publishes a
:class:`HeimdallSnapshot` consumed once per dispatch tick by the runner's
``_consume_heimdall_snapshot``. Persists per-host health and recent
activity to ``<dispatch_dir>/fleet.json`` for the Valhalla dashboard.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "FLEET_JSON_SCHEMA_VERSION",
    "HeimdallSnapshot",
    "HostHealth",
    "StaleJob",
    "read_fleet_json",
    "write_fleet_json",
]


FLEET_JSON_SCHEMA_VERSION = "1.0"
_FLEET_FILENAME = "fleet.json"


@dataclass(frozen=True)
class HostHealth:
    """Per-host health snapshot.

    Frozen so a published snapshot can be safely shared across threads
    without copy-on-read discipline.
    """

    name: str
    healthy: bool
    last_probe_at: str
    consecutive_failures: int
    failure_reason: str | None
    recovery_attempts: int
    recovery_history: list[str] = field(default_factory=list)
    quarantined: bool = False


@dataclass(frozen=True)
class StaleJob:
    """One in-flight job whose last heartbeat is older than the threshold."""

    run_id: str
    host: str
    last_heartbeat_at: str
    age_seconds: float
    host_was_healthy: bool


@dataclass(frozen=True)
class HeimdallSnapshot:
    """One watcher tick's published view; consumed at most once by the runner."""

    generated_at: str
    hosts: dict[str, HostHealth]
    stale_jobs: list[StaleJob]
    recent_events: list[dict] = field(default_factory=list)


# --- I/O ---------------------------------------------------------------------


def write_fleet_json(
    dispatch_dir: Path,
    *,
    generated_at: str,
    hosts: dict[str, HostHealth],
    recent_events: list[dict],
) -> None:
    """Atomically rewrite ``<dispatch_dir>/fleet.json``.

    Writes to a sibling temporary file and ``os.replace``\\ s into place so
    a concurrent reader (the Valhalla dashboard) never observes a
    truncated file. On serialization failure, the ``.tmp`` file is unlinked
    and the existing ``fleet.json`` (if any) is left untouched.

    Args:
        dispatch_dir: Directory that owns ``fleet.json`` (the dispatch
            directory, alongside ``dispatch.json``).
        generated_at: ISO-8601 UTC timestamp of the watcher tick that
            produced this snapshot.
        hosts: Per-host health snapshot keyed by host name.
        recent_events: Ring buffer of the last few watcher actions
            (flips, recoveries, requeues, stale-job kills); each entry
            is a free-form dict serialized verbatim.
    """
    dispatch_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": FLEET_JSON_SCHEMA_VERSION,
        "generated_at": generated_at,
        "hosts": {name: asdict(h) for name, h in hosts.items()},
        "recent_events": list(recent_events),
    }
    fd, tmp_path_str = tempfile.mkstemp(prefix=".fleet_", suffix=".json.tmp", dir=str(dispatch_dir))
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=False)
        os.replace(tmp_path, dispatch_dir / _FLEET_FILENAME)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def read_fleet_json(dispatch_dir: Path) -> dict[str, Any] | None:
    """Read ``<dispatch_dir>/fleet.json`` as a raw dict, or ``None`` if absent.

    Returns the raw payload (not :class:`HostHealth` objects) — the dashboard
    consumes the JSON shape directly.
    """
    path = dispatch_dir / _FLEET_FILENAME
    if not path.exists():
        return None
    with path.open("r") as fh:
        return json.load(fh)
