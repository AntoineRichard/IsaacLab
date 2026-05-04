# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tab A cancel-button confirm-flow callback handler tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.callbacks import (
    _on_cancel_revert_handler,
    _on_cancel_toggle_handler,
)


@dataclass
class _FakeData:
    runs_root: Path
    cancel_calls: list = field(default_factory=list)

    def request_cancel(self, dispatch_id, run_id, *, kind):
        self.cancel_calls.append((dispatch_id, run_id, kind))


def _now_ms() -> int:
    return 1_700_000_000_000


def test_first_click_flips_to_pending_state(tmp_path: Path):
    data = _FakeData(runs_root=tmp_path)
    pending_store = {}

    new_store = _on_cancel_toggle_handler(
        n_clicks_list=[1],
        ids_list=[{"type": "tab-a-cancel-toggle", "run_id": "r1"}],
        statuses=["running"],
        dispatch_id="d1",
        dispatch_ended=False,
        pending_store=pending_store,
        data=data,
        now_ms=_now_ms(),
    )

    # Stored with expiry 5s out; no DB write yet.
    assert new_store == {"r1": _now_ms() + 5000}
    assert data.cancel_calls == []


def test_second_click_within_window_writes_db_and_clears_pending(tmp_path: Path):
    data = _FakeData(runs_root=tmp_path)
    pending_store = {"r1": _now_ms() + 4000}  # 4s left

    new_store = _on_cancel_toggle_handler(
        n_clicks_list=[2],
        ids_list=[{"type": "tab-a-cancel-toggle", "run_id": "r1"}],
        statuses=["running"],
        dispatch_id="d1",
        dispatch_ended=False,
        pending_store=pending_store,
        data=data,
        now_ms=_now_ms(),
    )

    assert new_store == {}
    assert data.cancel_calls == [("d1", "r1", "kill")]


def test_skip_kind_for_pending_status(tmp_path: Path):
    data = _FakeData(runs_root=tmp_path)
    pending_store = {"r1": _now_ms() + 4000}

    _on_cancel_toggle_handler(
        n_clicks_list=[2],
        ids_list=[{"type": "tab-a-cancel-toggle", "run_id": "r1"}],
        statuses=["pending"],
        dispatch_id="d1",
        dispatch_ended=False,
        pending_store=pending_store,
        data=data,
        now_ms=_now_ms(),
    )

    assert data.cancel_calls == [("d1", "r1", "skip")]


def test_dispatch_ended_drops_clicks(tmp_path: Path):
    data = _FakeData(runs_root=tmp_path)

    new_store = _on_cancel_toggle_handler(
        n_clicks_list=[1],
        ids_list=[{"type": "tab-a-cancel-toggle", "run_id": "r1"}],
        statuses=["running"],
        dispatch_id="d1",
        dispatch_ended=True,
        pending_store={},
        data=data,
        now_ms=_now_ms(),
    )

    assert new_store == {}
    assert data.cancel_calls == []


def test_revert_drops_expired_entries():
    pending_store = {
        "r1": _now_ms() - 1,  # expired
        "r2": _now_ms() + 3000,  # alive
    }

    new_store = _on_cancel_revert_handler(pending_store=pending_store, now_ms=_now_ms())

    assert new_store == {"r2": _now_ms() + 3000}


def test_revert_no_change_when_no_expiry():
    pending_store = {"r1": _now_ms() + 3000}

    new_store = _on_cancel_revert_handler(pending_store=pending_store, now_ms=_now_ms())

    assert new_store == pending_store
