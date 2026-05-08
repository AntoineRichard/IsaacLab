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

from dataclasses import dataclass, field

__all__ = [
    "FLEET_JSON_SCHEMA_VERSION",
    "HeimdallSnapshot",
    "HostHealth",
    "StaleJob",
]


FLEET_JSON_SCHEMA_VERSION = "1.0"


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
