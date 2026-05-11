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
import logging
import os
import tempfile
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
from tools.odin.asgard.transport import SSHRunner

if TYPE_CHECKING:
    from tools.odin.asgard.state import DispatchState

__all__ = [
    "FLEET_JSON_SCHEMA_VERSION",
    "HeimdallSnapshot",
    "HeimdallWatcher",
    "HostHealth",
    "StaleJob",
    "read_fleet_json",
    "write_fleet_json",
]


_log = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _probe_host(
    host: ValkyrieConfig,
    *,
    ssh: SSHRunner,
    timeout_s: float,
) -> tuple[bool, str | None]:
    """Run ``docker exec <ctr> nvidia-smi -L`` and classify the outcome.

    Returns ``(healthy, failure_reason)``. ``failure_reason`` is one of
    ``"ssh_timeout"``, ``"nvml_missing"``, ``"ssh_error"``, or ``None``
    when ``healthy is True``.
    """
    cmd = f"docker exec {host.container_name} nvidia-smi -L"
    r = ssh.run(host, cmd, timeout_s=timeout_s, pty=False)
    if r.timed_out:
        return False, "ssh_timeout"
    if r.exit_code == 255:
        return False, "ssh_error"
    if r.exit_code != 0:
        return False, "nvml_missing"
    if not r.stdout.strip():
        return False, "nvml_missing"
    return True, None


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


# --- Watcher -----------------------------------------------------------------


class HeimdallWatcher:
    """Periodic fleet probe + stale-job watcher.

    The watcher owns its own daemon thread and is the sole writer of
    ``fleet.json``. Consumers (the dispatcher main loop, the Valhalla
    dashboard) call :meth:`latest` to read the most recent
    :class:`HeimdallSnapshot` and never mutate watcher state.

    Thread safety: :meth:`latest` and the publishing path inside the
    probing thread are guarded by a single :class:`threading.Lock`. The
    probing path may run concurrently with main-loop consumption.
    """

    def __init__(
        self,
        fleet: Fleet,
        dispatch_dir: Path,
        ssh: SSHRunner,
        state_view: Callable[[], DispatchState],
        *,
        probe_interval_s: int = 300,
        stale_threshold_s: int = 180,
        flip_after_k_failures: int = 2,
        probe_timeout_s: int = 15,
        recent_events_max: int = 20,
    ) -> None:
        self._fleet = fleet
        self._dispatch_dir = Path(dispatch_dir)
        self._ssh = ssh
        self._state_view = state_view
        self._probe_interval_s = probe_interval_s
        self._stale_threshold_s = stale_threshold_s
        self._flip_after_k_failures = flip_after_k_failures
        self._probe_timeout_s = probe_timeout_s
        self._recent_events_max = recent_events_max

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: HeimdallSnapshot | None = None
        self._host_state: dict[str, HostHealth] = {}
        self._recent_events: list[dict] = []

    def start(self) -> None:
        """Spawn the probing thread.

        Raises:
            RuntimeError: If the watcher has already been started.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("HeimdallWatcher already started")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="heimdall-watcher", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 10.0) -> None:
        """Signal stop and join the probing thread (idempotent)."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
        self._thread = None

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def latest(self) -> HeimdallSnapshot | None:
        """Return the most recent published snapshot, or ``None`` before the first tick."""
        with self._lock:
            return self._latest

    def _tick_once(self) -> HeimdallSnapshot:
        """Run a single probe + stale-job pass synchronously.

        Test entry point. The production :meth:`_run` loop calls this once
        per ``probe_interval_s``.
        """
        host_health = self._probe_all_hosts()
        stale = self._compute_stale_jobs(host_health)
        snap = HeimdallSnapshot(
            generated_at=_utc_now_iso(),
            hosts=host_health,
            stale_jobs=stale,
            recent_events=list(self._recent_events),
        )
        with self._lock:
            self._latest = snap
        try:
            write_fleet_json(
                self._dispatch_dir,
                generated_at=snap.generated_at,
                hosts=host_health,
                recent_events=list(self._recent_events),
            )
        except Exception as exc:  # non-fatal — dashboard catches up next tick.
            _log.warning("heimdall: fleet.json write failed: %r", exc)
        return snap

    def _probe_all_hosts(self) -> dict[str, HostHealth]:
        results: dict[str, HostHealth] = {}
        if not self._fleet.hosts:
            self._host_state = results
            return results
        with ThreadPoolExecutor(
            max_workers=max(1, len(self._fleet.hosts)),
            thread_name_prefix="heimdall-probe",
        ) as pool:
            futures = {
                pool.submit(_probe_host, h, ssh=self._ssh, timeout_s=self._probe_timeout_s): h
                for h in self._fleet.hosts
            }
            now = _utc_now_iso()
            for fut, host in futures.items():
                try:
                    healthy, reason = fut.result()
                except Exception as exc:  # SSH-runner exception path.
                    healthy, reason = False, f"probe_exception:{type(exc).__name__}"
                prev = self._host_state.get(host.host)
                cf = prev.consecutive_failures if prev else 0
                cf = 0 if healthy else cf + 1
                effective_healthy = cf < self._flip_after_k_failures
                results[host.host] = HostHealth(
                    name=host.host,
                    healthy=effective_healthy,
                    last_probe_at=now,
                    consecutive_failures=cf,
                    failure_reason=None if effective_healthy else reason,
                    recovery_attempts=(prev.recovery_attempts if prev else 0),
                    recovery_history=list(prev.recovery_history) if prev else [],
                    quarantined=(prev.quarantined if prev else False),
                )
        self._host_state = results
        return results

    def _compute_stale_jobs(self, host_health: dict[str, HostHealth]) -> list[StaleJob]:
        try:
            state = self._state_view()
        except Exception as exc:
            _log.warning("heimdall: state_view failed: %r", exc)
            return []
        now = datetime.now(timezone.utc)
        stale: list[StaleJob] = []
        for job in state.jobs:
            if job.status != "running":
                continue
            baseline_iso = job.last_heartbeat_at or job.started_at
            if baseline_iso is None:
                continue
            try:
                baseline = datetime.strptime(baseline_iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            age = (now - baseline).total_seconds()
            if age <= self._stale_threshold_s:
                continue
            host_name = job.assigned_to or ""
            host_was_healthy = host_name in host_health and host_health[host_name].healthy
            stale.append(
                StaleJob(
                    run_id=job.run_id,
                    host=host_name,
                    last_heartbeat_at=baseline_iso,
                    age_seconds=age,
                    host_was_healthy=host_was_healthy,
                )
            )
        return stale

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._tick_once()
            except Exception as exc:  # don't let one bad tick kill the watcher.
                _log.exception("heimdall: tick failed: %r", exc)
            if self._stop_event.wait(timeout=self._probe_interval_s):
                return
