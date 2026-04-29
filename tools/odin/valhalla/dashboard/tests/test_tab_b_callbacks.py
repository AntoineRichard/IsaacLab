# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the Tab B callback helpers."""

from __future__ import annotations

from tools.odin.valhalla.dashboard.tabs.task_drilldown.callbacks import (
    _compute_picker_children,
    _serialize_to_url,
    _sync_url_to_selection,
)


class _StubData:
    """Drop-in DataLayer for callback tests."""

    def __init__(self, aggregate=None):
        self._agg = aggregate

    def load_aggregate(self, dispatch_id: str):
        return self._agg


def _aggregate(rows: list[dict]) -> dict:
    return {"schema_version": "1.0", "rows": rows}


def _row(task: str, framework: str = "rsl_rl", backend: str = "physx") -> dict:
    return {
        "task": task,
        "framework": framework,
        "backend": backend,
        "aggregate": {},
        "seeds": {},
        "divergent_seeds": [],
    }


def test_init_picker_returns_picker_div():
    data = _StubData(aggregate=_aggregate([_row("X")]))
    out = _compute_picker_children(data, "d-1", search="?task=X&framework=rsl_rl&backend=physx")
    assert getattr(out, "id", None) == "tab-b-picker"


def test_sync_url_to_selection_parses_full():
    out = _sync_url_to_selection("?task=A&framework=rsl_rl&backend=physx")
    assert out == "A|rsl_rl|physx"


def test_sync_url_to_selection_handles_empty():
    assert _sync_url_to_selection("") is None


def test_picker_to_url_serializes_value():
    out = _serialize_to_url("A|rsl_rl|physx")
    assert out == "?task=A&framework=rsl_rl&backend=physx"
