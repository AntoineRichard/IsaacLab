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

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["DataLayer", "DispatchSummary", "HardwareInfo"]


_DISPATCH_ID_RE = re.compile(r"^\d{8}-\d{6}$")


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
