# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for tab_b.picker."""

from __future__ import annotations

from tools.odin.valhalla.dashboard.tabs.task_drilldown.picker import (
    list_row_options,
    render_picker,
)
from tools.odin.valhalla.dashboard.tabs.task_drilldown.url_state import TaskSelection


def _row(task: str, framework: str = "rsl_rl", backend: str = "physx") -> dict:
    return {
        "task": task,
        "framework": framework,
        "backend": backend,
        "aggregate": {},
        "seeds": {},
        "divergent_seeds": [],
    }


def _aggregate(rows: list[dict]) -> dict:
    return {"schema_version": "1.0", "rows": rows}


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


def test_list_row_options_one_per_row():
    agg = _aggregate([_row("Isaac-Ant-Direct-v0"), _row("Isaac-Cartpole-Direct-v0"),
                      _row("Isaac-Humanoid-Direct-v0")])
    assert len(list_row_options(agg)) == 3


def test_list_row_options_label_format():
    agg = _aggregate([_row("Isaac-Ant-Direct-v0")])
    opts = list_row_options(agg)
    assert opts[0]["label"] == "Isaac-Ant-Direct-v0 · rsl_rl × physx"


def test_list_row_options_value_format():
    agg = _aggregate([_row("Isaac-Ant-Direct-v0")])
    opts = list_row_options(agg)
    assert opts[0]["value"] == "Isaac-Ant-Direct-v0|rsl_rl|physx"


def test_list_row_options_sorted_by_task_name():
    agg = _aggregate([_row("Isaac-Cartpole-Direct-v0"), _row("Isaac-Ant-Direct-v0"),
                      _row("Isaac-Humanoid-Direct-v0")])
    labels = [o["label"] for o in list_row_options(agg)]
    assert labels[0].startswith("Isaac-Ant-Direct-v0")
    assert labels[1].startswith("Isaac-Cartpole-Direct-v0")
    assert labels[2].startswith("Isaac-Humanoid-Direct-v0")


def test_render_picker_contains_dropdown():
    agg = _aggregate([_row("Isaac-Ant-Direct-v0")])
    component = render_picker(agg, selected=None)
    dropdowns = [c for c in _walk(component) if type(c).__name__ == "Dropdown"]
    assert len(dropdowns) == 1
    dd = dropdowns[0]
    assert getattr(dd, "id", None) == "tab-b-row-select"
    assert getattr(dd, "searchable", False) is True


def test_render_picker_preselects_value_when_in_options():
    agg = _aggregate([_row("Isaac-Ant-Direct-v0")])
    sel = TaskSelection("Isaac-Ant-Direct-v0", "rsl_rl", "physx")
    component = render_picker(agg, selected=sel)
    dd = next(c for c in _walk(component) if type(c).__name__ == "Dropdown")
    assert dd.value == "Isaac-Ant-Direct-v0|rsl_rl|physx"


def test_render_picker_no_preselection_when_selection_missing():
    agg = _aggregate([_row("Isaac-Ant-Direct-v0")])
    sel = TaskSelection("does-not-exist", "rsl_rl", "physx")
    component = render_picker(agg, selected=sel)
    dd = next(c for c in _walk(component) if type(c).__name__ == "Dropdown")
    assert dd.value is None


def test_render_picker_handles_empty_aggregate():
    agg = _aggregate([])
    component = render_picker(agg, selected=None)
    dd = next(c for c in _walk(component) if type(c).__name__ == "Dropdown")
    assert dd.options == []
    assert dd.value is None
