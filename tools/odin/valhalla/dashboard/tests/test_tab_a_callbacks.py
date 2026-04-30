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

    def read_retry_queue(self, dispatch_id: str) -> set[str]:
        return set()


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


def test_toggle_expand_row_adds_then_removes():
    from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.callbacks import _toggle_run_id

    out = _toggle_run_id([], "X")
    assert out == ["X"]

    out = _toggle_run_id(["X"], "X")
    assert out == []


def test_toggle_expand_row_ignores_phantom_click():
    """n_clicks=0 list (Dash phantom fire) returns dash.no_update."""
    from tools.odin.valhalla.dashboard.tabs.dispatch_fleet import callbacks as cb_mod

    out = cb_mod._on_expand_toggle_handler([], [], current=[])
    import dash

    assert out is dash.no_update


def test_load_ssh_tail_callback_writes_store(tmp_path):
    from tools.odin.valhalla.dashboard.tabs.dispatch_fleet import callbacks as cb_mod

    log_dir = tmp_path / "d" / "Y" / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "ssh-tail.log").write_text("a\nb\n")

    class _Data:
        _runs_root = tmp_path

    out = cb_mod._compute_ssh_tail_store(_Data(), "d", "Y", current_store={})
    assert out == {"Y": ["a", "b"]}


def test_load_ssh_tail_callback_ignores_phantom_click():
    """n_clicks=0 list (no clicks yet) returns dash.no_update."""
    import dash

    from tools.odin.valhalla.dashboard.tabs.dispatch_fleet import callbacks as cb_mod

    out = cb_mod._on_ssh_tail_handler([], [], data=None, current_store={})
    assert out is dash.no_update


def test_retry_toggle_round_trip(tmp_path):
    """Click on a retry-toggle button → DataLayer.toggle_retry_queue is
    invoked with the right run_id; the bump counter increments. Click
    again → toggles off, file goes empty, bump increments again."""
    from tools.odin.valhalla.dashboard.data import DataLayer
    from tools.odin.valhalla.dashboard.tabs.dispatch_fleet import callbacks as cb_mod

    (tmp_path / "20260427-141302").mkdir()
    (tmp_path / "20260427-141302" / "dispatch.json").write_text("{}")
    data = DataLayer(tmp_path)

    n_clicks = [1]
    ids = [{"type": "tab-a-retry-toggle", "run_id": "rsl-rl_physx_X_seed42"}]
    out = cb_mod._on_retry_toggle_handler(n_clicks, ids, dispatch_id="20260427-141302", bump=0, data=data)
    assert out == 1
    assert data.read_retry_queue("20260427-141302") == {"rsl-rl_physx_X_seed42"}

    out = cb_mod._on_retry_toggle_handler(n_clicks, ids, dispatch_id="20260427-141302", bump=1, data=data)
    assert out == 2
    assert data.read_retry_queue("20260427-141302") == set()


def test_retry_toggle_ignores_phantom_click():
    import dash

    from tools.odin.valhalla.dashboard.tabs.dispatch_fleet import callbacks as cb_mod

    class _DataNoOp:
        def toggle_retry_queue(self, *a, **k):
            raise AssertionError("must not be called on phantom")

    out = cb_mod._on_retry_toggle_handler([], [], dispatch_id="d", bump=0, data=_DataNoOp())
    assert out is dash.no_update
    out = cb_mod._on_retry_toggle_handler(
        [0], [{"type": "x", "run_id": "y"}], dispatch_id="d", bump=0, data=_DataNoOp()
    )
    assert out is dash.no_update
