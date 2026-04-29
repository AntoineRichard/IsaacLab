# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for tab_b.url_state."""

from __future__ import annotations

from tools.odin.valhalla.dashboard.tabs.task_drilldown.url_state import (
    TaskSelection,
    parse_query_string,
    serialize,
)


def test_parse_empty_string_returns_empty_selection():
    assert parse_query_string("") == TaskSelection(None, None, None)


def test_parse_full_query_string():
    out = parse_query_string("?task=A&framework=rsl_rl&backend=physx")
    assert out == TaskSelection("A", "rsl_rl", "physx")


def test_parse_partial_query_string():
    assert parse_query_string("?task=A") == TaskSelection("A", None, None)


def test_parse_url_encoded_task():
    out = parse_query_string("?task=Isaac-Repose-Cube-Allegro-Direct-v0")
    assert out.task == "Isaac-Repose-Cube-Allegro-Direct-v0"


def test_parse_duplicate_keys_takes_last():
    out = parse_query_string("?task=a&task=b")
    assert out.task == "b"


def test_serialize_full_selection():
    sel = TaskSelection("A", "rsl_rl", "physx")
    assert serialize(sel) == "?task=A&framework=rsl_rl&backend=physx"


def test_serialize_omits_none_fields():
    sel = TaskSelection("A", None, None)
    assert serialize(sel) == "?task=A"


def test_serialize_empty_selection_returns_empty_string():
    assert serialize(TaskSelection(None, None, None)) == ""


def test_round_trip_preserves_special_characters():
    sel = TaskSelection("Isaac-Repose-Cube-Allegro-Direct-v0", "rsl_rl", "physx")
    out = parse_query_string(serialize(sel))
    assert out == sel
