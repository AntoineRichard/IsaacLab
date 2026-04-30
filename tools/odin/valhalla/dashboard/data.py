# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-Python data layer over ``odin_runs/`` for the Odin dashboard.

Zero Dash imports — exposed APIs are dataclasses and a single :class:`DataLayer`
class. Tab modules and the app shell call into this layer for everything that
touches disk so the UI layer stays orthogonal and the layer itself stays
trivially testable.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["DataLayer", "DispatchSummary", "HardwareInfo"]


_DISPATCH_ID_RE = re.compile(r"^\d{8}-\d{6}$")
_REMOTE_ODIN_RUNS_ROOT = "/workspace/isaaclab/odin_runs"
_RUNNING_TAIL_DEFAULT_LINES = 50
_RUNNING_TAIL_SOURCE_PREFIX = "__odin_tail_source__:"
_RUNNING_TAIL_TIMEOUT_S = 10
_subprocess_run = subprocess.run


@dataclass(frozen=True)
class DispatchSummary:
    """Headline view of one dispatch — what the landing table renders."""

    dispatch_id: str
    started_at: str
    ended_at: str | None
    jobs_total: int
    jobs_completed: int
    jobs_failed: int
    jobs_pending: int
    skipped_total: int
    hostnames: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HardwareInfo:
    """Per-host hardware block, normalized for cross-dispatch comparison."""

    hostname: str
    gpu_devices: list[dict[str, Any]]
    cpu_name: str
    cpu_count: int
    ram_gb: float
    sourced_from: str


class DataLayer:
    """All disk reads for the dashboard go through this class."""

    def __init__(self, runs_root: Path):
        self._runs_root = Path(runs_root).resolve() if runs_root else Path(runs_root)

    # -- list_dispatches ----------------------------------------------------

    def list_dispatches(self) -> list[DispatchSummary]:
        """Return all dispatches under ``runs_root``, newest-first.

        Filters to directories whose name matches ``YYYYMMDD-HHMMSS`` AND that
        contain a ``dispatch.json``. Loose pre-T3.1 bundles (e.g.
        ``rsl-rl_physx_..._seed42``) are excluded.
        """
        if not self._runs_root.exists():
            return []
        results: list[DispatchSummary] = []
        for entry in self._runs_root.iterdir():
            if not entry.is_dir():
                continue
            if not _DISPATCH_ID_RE.match(entry.name):
                continue
            dispatch_json = entry / "dispatch.json"
            if not dispatch_json.exists():
                continue
            try:
                payload = json.loads(dispatch_json.read_text())
            except json.JSONDecodeError:
                continue
            results.append(_summary_from_dispatch(payload))
        results.sort(key=lambda s: s.dispatch_id, reverse=True)
        return results

    # -- raw JSON readers ---------------------------------------------------

    def load_dispatch(self, dispatch_id: str) -> dict[str, Any]:
        """Read ``<runs_root>/<dispatch_id>/dispatch.json``.

        Raises:
            FileNotFoundError: if the file is absent.
        """
        path = self._runs_root / dispatch_id / "dispatch.json"
        if not path.exists():
            raise FileNotFoundError(f"dispatch.json missing for {dispatch_id} at {path}")
        return json.loads(path.read_text())

    def load_aggregate(self, dispatch_id: str) -> dict[str, Any] | None:
        """Read ``aggregate.json`` for the dispatch; ``None`` if absent."""
        path = self._runs_root / dispatch_id / "aggregate.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def load_hardware(self, dispatch_id: str) -> dict[str, Any] | None:
        """Read ``hardware.json`` for the dispatch; ``None`` if absent."""
        path = self._runs_root / dispatch_id / "hardware.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    # -- cross-dispatch lookup ----------------------------------------------

    def lookup_hardware(self, host: str) -> HardwareInfo | None:
        """Walk dispatches newest-first; return the first hardware block
        from any bundle whose ``assigned_to == host``.

        Used as a fall-back when a dispatch's own ``hardware.json`` is
        missing or doesn't list the host (e.g. for pre-feature dispatches).
        """
        for summary in self.list_dispatches():
            try:
                payload = self.load_dispatch(summary.dispatch_id)
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            jobs = payload.get("jobs", []) or []
            for job in jobs:
                if job.get("assigned_to") != host:
                    continue
                run_id = job.get("run_id")
                if not run_id:
                    continue
                training_path = self._runs_root / summary.dispatch_id / run_id / "training.json"
                if not training_path.exists():
                    continue
                try:
                    training = json.loads(training_path.read_text())
                except json.JSONDecodeError:
                    continue
                hw = training.get("hardware")
                if not hw:
                    continue
                return HardwareInfo(
                    hostname=str(hw.get("hostname", "")),
                    gpu_devices=list(hw.get("gpu_devices") or []),
                    cpu_name=str(hw.get("cpu_name", "")),
                    cpu_count=int(hw.get("cpu_count", 0)),
                    ram_gb=float(hw.get("ram_gb", 0.0)),
                    sourced_from=f"{summary.dispatch_id}/{run_id}",
                )
        return None

    # -- trend axis ---------------------------------------------------------

    def trend_dispatches_for(
        self,
        current_dispatch_id: str,
        task: str,
        framework: str,
        backend: str,
        n: int = 10,
    ) -> list[str]:
        """Return the N most recent dispatch_ids that:

        - have a ``hardware.json`` whose fingerprint matches ``current_dispatch_id``
          (excludes pre-feature dispatches and mismatched-hardware dispatches), AND
        - have an ``aggregate.json`` row for ``(task, framework, backend)``.

        Sorted newest-first; trimmed to ``n``.
        """
        current_hw = self.load_hardware(current_dispatch_id)
        if current_hw is None:
            return []
        target_fingerprint = current_hw.get("fingerprint")
        if not target_fingerprint:
            return []
        matches: list[str] = []
        for summary in self.list_dispatches():
            hw = self.load_hardware(summary.dispatch_id)
            if hw is None or hw.get("fingerprint") != target_fingerprint:
                continue
            agg = self.load_aggregate(summary.dispatch_id)
            if agg is None:
                continue
            rows = agg.get("rows", []) or []
            if not any(
                r.get("task") == task and r.get("framework") == framework and r.get("backend") == backend for r in rows
            ):
                continue
            matches.append(summary.dispatch_id)
            if len(matches) >= n:
                break
        return matches

    # -- per-bundle reads ---------------------------------------------------

    def load_training(self, dispatch_id: str, run_id: str) -> dict[str, Any] | None:
        """Read ``<runs_root>/<dispatch_id>/<run_id>/training.json``.

        Returns ``None`` when the file is absent (failed bundle, pulled-in-progress, etc.).
        """
        path = self._runs_root / dispatch_id / run_id / "training.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def load_startup(self, dispatch_id: str, run_id: str) -> dict[str, Any] | None:
        """Read ``<runs_root>/<dispatch_id>/<run_id>/startup.json``.

        Returns ``None`` when the file is absent.
        """
        path = self._runs_root / dispatch_id / run_id / "startup.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    # -- running bundle tail -----------------------------------------------

    def read_running_job_tail(
        self,
        dispatch_id: str,
        run_id: str,
        *,
        host: str,
        ssh_user: str = "horde",
        ssh_key: Path | None = None,
        container_name: str = "isaac-lab-base",
        n: int = _RUNNING_TAIL_DEFAULT_LINES,
    ) -> list[str]:
        """Read the last ``n`` stdout lines from a running remote Hugin bundle.

        The reader tries ``training.stdout.log`` first, then falls back to
        ``startup.stdout.log`` when training has not started yet. SSH or log
        availability failures return an empty list so the dashboard can keep
        rendering running rows.
        """
        return self.read_running_job_tail_payload(
            dispatch_id,
            run_id,
            host=host,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            container_name=container_name,
            n=n,
        )["lines"]

    def lookup_fleet_host_config(self, dispatch_id: str, host: str) -> dict[str, Any] | None:
        """Return the snapshotted fleet config for ``host`` when available."""
        path = self._runs_root / dispatch_id / "fleet.yaml.snapshot"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        for entry in payload.get("hosts", []) or []:
            if entry.get("host") == host:
                return dict(entry)
        return None

    def read_running_job_tail_payload(
        self,
        dispatch_id: str,
        run_id: str,
        *,
        host: str,
        ssh_user: str = "horde",
        ssh_key: Path | None = None,
        container_name: str = "isaac-lab-base",
        n: int = _RUNNING_TAIL_DEFAULT_LINES,
    ) -> dict[str, Any]:
        """Read running-job stdout tail lines and source metadata.

        Returns:
            A payload with ``source`` set to the selected log filename, or
            ``None`` when no log was available, and ``lines`` containing the
            decoded tail without trailing newline characters. Transport
            failures include ``warning`` text for the UI; normal empty logs
            return ``warning=None``.
        """
        n = max(1, int(n))
        ssh_cmd = _build_running_tail_ssh_cmd(run_id=run_id, container_name=container_name, n=n)
        argv = _build_running_tail_ssh_argv(host=host, ssh_user=ssh_user, ssh_key=ssh_key, ssh_cmd=ssh_cmd)
        try:
            result = _subprocess_run(argv, capture_output=True, timeout=_RUNNING_TAIL_TIMEOUT_S, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            warning = f"{type(exc).__name__}: {exc}"
            _warn_running_tail(dispatch_id, run_id, warning)
            return {"source": None, "lines": [], "warning": warning}

        if result.returncode != 0:
            message = _decode_running_tail_bytes(result.stderr).strip() or f"ssh exited with {result.returncode}"
            _warn_running_tail(dispatch_id, run_id, message)
            return {"source": None, "lines": [], "warning": message}

        source, lines = _parse_running_tail_stdout(result.stdout, n)
        return {"source": source, "lines": lines, "warning": None}

    # -- retry queue (operator's per-dispatch "to retry" list) -------------

    def read_retry_queue(self, dispatch_id: str) -> set[str]:
        """Return the set of run_ids the operator has tagged for retry.

        Stored at ``<runs_root>/<dispatch_id>/retry_queue.txt`` (one
        run_id per line). Empty / missing file → empty set.

        dispatch.json is never mutated; this file is the operator's
        TODO list, consumed by the next ``odin-dispatch --resume <id>
        --retry-failed=<csv>`` invocation.
        """
        path = self._runs_root / dispatch_id / "retry_queue.txt"
        if not path.exists():
            return set()
        return {line.strip() for line in path.read_text().splitlines() if line.strip()}

    def toggle_retry_queue(self, dispatch_id: str, run_id: str) -> set[str]:
        """Add ``run_id`` to the retry queue if absent, remove if present.

        Atomic on POSIX (tempfile + ``os.replace``) so an interrupted
        write can't leave a half-truncated file.

        Returns the new contents.
        """
        current = self.read_retry_queue(dispatch_id)
        if run_id in current:
            current.discard(run_id)
        else:
            current.add(run_id)
        dispatch_dir = self._runs_root / dispatch_id
        dispatch_dir.mkdir(parents=True, exist_ok=True)
        target = dispatch_dir / "retry_queue.txt"
        body = "".join(line + "\n" for line in sorted(current))
        fd, tmp_path_str = tempfile.mkstemp(prefix=".retry_queue.", suffix=".tmp", dir=str(dispatch_dir))
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(body)
            os.replace(tmp_path_str, target)
        except Exception:
            # Best-effort cleanup of the temp file on failure.
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_path_str)
            raise
        return current

    # -- cache control ------------------------------------------------------

    def invalidate(self, dispatch_id: str | None = None) -> None:
        """Drop cached state for ``dispatch_id`` (or all if ``None``).

        Callers (notably Tab A's poll on the live → done transition) call
        this before re-reading so the freshly-written aggregate.json /
        hardware.json is picked up. Spec 0 caches nothing yet — Specs 1+
        wrap reads in :func:`functools.lru_cache` and add cache-clear
        plumbing here. Defined now so callers don't need to be edited
        when caching arrives.
        """
        # Intentionally empty in Spec 0. See docstring.
        return


def _summary_from_dispatch(payload: dict[str, Any]) -> DispatchSummary:
    jobs = payload.get("jobs", []) or []
    by_status: dict[str, int] = {}
    for j in jobs:
        s = j.get("status", "unknown")
        by_status[s] = by_status.get(s, 0) + 1
    fleet = payload.get("fleet", []) or []
    hostnames = [h["host"] for h in fleet if "host" in h]
    return DispatchSummary(
        dispatch_id=str(payload.get("dispatch_id", "")),
        started_at=str(payload.get("started_at", "")),
        ended_at=payload.get("ended_at"),
        jobs_total=len(jobs),
        jobs_completed=by_status.get("completed", 0),
        jobs_failed=by_status.get("failed", 0),
        jobs_pending=by_status.get("pending", 0),
        skipped_total=len(payload.get("skipped", []) or []),
        hostnames=hostnames,
    )


def _build_running_tail_ssh_argv(*, host: str, ssh_user: str, ssh_key: Path | None, ssh_cmd: str) -> list[str]:
    argv = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "BatchMode=yes",
    ]
    if ssh_key is not None:
        argv += ["-i", str(ssh_key)]
    argv += [f"{ssh_user}@{host}", ssh_cmd]
    return argv


def _build_running_tail_ssh_cmd(*, run_id: str, container_name: str, n: int) -> str:
    logs_dir = shlex.quote(f"{_REMOTE_ODIN_RUNS_ROOT}/{run_id}/logs")
    inner = (
        f"base={logs_dir}; "
        "for name in training.stdout.log startup.stdout.log; do "
        'f="$base/$name"; '
        'if [ -s "$f" ]; then '
        f'printf "{_RUNNING_TAIL_SOURCE_PREFIX}%s\\n" "$name"; '
        f'tail -n {n} "$f"; '
        "exit 0; "
        "fi; "
        "done"
    )
    return f"docker exec {shlex.quote(container_name)} bash -c {shlex.quote(inner)}"


def _parse_running_tail_stdout(stdout: bytes, n: int) -> tuple[str | None, list[str]]:
    text = _decode_running_tail_bytes(stdout)
    lines = text.splitlines()
    source = None
    if lines and lines[0].startswith(_RUNNING_TAIL_SOURCE_PREFIX):
        source = lines.pop(0)[len(_RUNNING_TAIL_SOURCE_PREFIX) :]
    return source, lines[-n:]


def _decode_running_tail_bytes(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def _warn_running_tail(dispatch_id: str, run_id: str, message: str) -> None:
    print(f"[WARNING] read_running_job_tail {dispatch_id}/{run_id}: {message}", file=sys.stderr)
