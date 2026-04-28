# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the Tab A callback helpers (called directly, not via Dash)."""

from __future__ import annotations

from pathlib import Path

from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.callbacks import (
    _compute_fleet_children,
    _compute_header_children,
    _compute_jobs_children,
    _handle_pill_click,
)


def _job(*, run_id="r", status="completed", kind=None):
    j = {
        "run_id": run_id,
        "task_id": "Isaac-Ant-Direct-v0",
        "framework": "rsl_rl",
        "backend": "physx",
        "seed": 42,
        "status": status,
        "assigned_to": "v1",
        "attempts": 1,
        "started_at": "2026-04-27T14:13:02Z",
        "ended_at": None,
        "preferred_not": [],
        "failure": None,
    }
    if status == "failed" and kind is not None:
        j["failure"] = {"kind": kind, "message": "x", "details": {}}
    return j


def _payload(jobs):
    return {
        "schema_version": "1.3",
        "dispatch_id": "d",
        "started_at": "x",
        "ended_at": None,
        "commit_sha": "abc1234",
        "fleet": [{"host": "v1", "status": "idle", "current_run_id": None, "last_error": None}],
        "jobs": jobs,
        "skipped": [],
    }


class _StubData:
    """Drop-in DataLayer for callback tests."""

    def __init__(self, dispatch_payload, *, hardware=None, lookup_results=None):
        self._dp = dispatch_payload
        self._hw = hardware
        self._lookup = lookup_results or {}
        self.load_dispatch_calls: list[str] = []
        self.load_hardware_calls: list[str] = []
        self.lookup_hardware_calls: list[str] = []
        self._runs_root = Path("/tmp")

    def load_dispatch(self, dispatch_id: str) -> dict:
        self.load_dispatch_calls.append(dispatch_id)
        return self._dp

    def load_hardware(self, dispatch_id: str):
        self.load_hardware_calls.append(dispatch_id)
        return self._hw

    def lookup_hardware(self, host: str):
        self.lookup_hardware_calls.append(host)
        return self._lookup.get(host)


def test_update_header_callback_returns_header_div():
    data = _StubData(_payload([_job()]))
    out = _compute_header_children(data, "d")
    assert getattr(out, "id", None) == "tab-a-header-content"


def test_update_fleet_callback_invokes_data_layer():
    data = _StubData(_payload([_job()]), hardware=None)
    _compute_fleet_children(data, "d")
    assert data.load_dispatch_calls == ["d"]
    assert data.load_hardware_calls == ["d"]
    # Fall-back called once per host (one host in the stub payload).
    assert data.lookup_hardware_calls == ["v1"]


def test_update_jobs_callback_applies_filters():
    jobs = [
        _job(status="completed", run_id="c1"),
        _job(status="failed", kind="hugin_crash", run_id="f1"),
        _job(status="failed", kind="gpu_lost", run_id="f2"),
    ]
    data = _StubData(_payload(jobs))
    out = _compute_jobs_children(
        data,
        dispatch_id="d",
        status_filter=["failed"],
        kind_filter=["hugin_crash"],
        task_text="ant",
        failure_filter=None,
        expanded_run_ids=[],
        ssh_tail_store={},
    )

    def _walk(component):
        yield component
        children = getattr(component, "children", None)
        if children is None:
            return
        if isinstance(children, list):
            for child in children:
                if child is None or isinstance(child, str):
                    continue
                yield from _walk(child)
        elif not isinstance(children, str):
            yield from _walk(children)

    rows = [c for c in _walk(out) if type(c).__name__ == "Tr"]
    # 1 header row + 1 data row.
    assert len(rows) == 2


def test_update_jobs_callback_uses_failure_filter_store():
    """When the failure-filter store carries a kind, it's applied like a kind_filter entry."""
    jobs = [
        _job(status="failed", kind="gpu_lost", run_id="g"),
        _job(status="failed", kind="hugin_crash", run_id="h"),
    ]
    data = _StubData(_payload(jobs))
    out = _compute_jobs_children(
        data,
        dispatch_id="d",
        status_filter=None,
        kind_filter=None,
        task_text="",
        failure_filter="gpu_lost",
        expanded_run_ids=[],
        ssh_tail_store={},
    )

    def _walk(component):
        yield component
        children = getattr(component, "children", None)
        if children is None:
            return
        if isinstance(children, list):
            for child in children:
                if child is None or isinstance(child, str):
                    continue
                yield from _walk(child)
        elif not isinstance(children, str):
            yield from _walk(children)

    rows = [c for c in _walk(out) if type(c).__name__ == "Tr"]
    assert len(rows) == 2  # header + 1 gpu_lost row


def test_failure_pill_click_writes_store_and_dropdown():
    store_value, dropdown_value = _handle_pill_click("gpu_lost")
    assert store_value == "gpu_lost"
    assert dropdown_value == ["gpu_lost"]
