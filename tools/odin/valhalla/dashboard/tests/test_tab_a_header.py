# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the Tab A header."""

from __future__ import annotations

from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.header import render_header


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


def _text_blob(component) -> str:
    """Concatenate every string child encountered in the tree."""
    parts: list[str] = []
    for c in _walk(component):
        ch = getattr(c, "children", None)
        if isinstance(ch, str):
            parts.append(ch)
    return " ".join(parts)


def _has_class(component, target_class: str) -> bool:
    for c in _walk(component):
        cls = getattr(c, "className", "") or ""
        if target_class in cls.split():
            return True
    return False


def _payload(jobs, *, ended_at=None, commit_sha="abc123def456", fleet=None):
    return {
        "schema_version": "1.3",
        "dispatch_id": "20260427-141302",
        "started_at": "2026-04-27T14:13:02Z",
        "ended_at": ended_at,
        "seeds": [42],
        "commit_sha": commit_sha,
        "fleet": fleet
        if fleet is not None
        else [
            {"host": "v1", "status": "idle", "current_run_id": None, "last_error": None},
            {"host": "v2", "status": "idle", "current_run_id": None, "last_error": None},
        ],
        "jobs": jobs,
        "skipped": [],
    }


def _job(status: str, *, kind: str | None = None, run_id: str = "r"):
    j = {
        "run_id": run_id,
        "task_id": "Isaac-Ant-Direct-v0",
        "framework": "rsl_rl",
        "backend": "physx",
        "num_envs": 4096,
        "max_iterations": 300,
        "seed": 42,
        "bundle_dir_name": run_id,
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


def test_header_live_pill_when_ended_at_null():
    component = render_header(_payload([_job("running")], ended_at=None))
    assert _has_class(component, "tab-a-live-pill")
    blob = _text_blob(component)
    assert "Live" in blob


def test_header_done_pill_when_ended_at_set():
    component = render_header(_payload([_job("completed")], ended_at="2026-04-27T15:00:00Z"))
    assert _has_class(component, "tab-a-done-pill")
    blob = _text_blob(component)
    assert "Done" in blob


def test_header_totals_match_jobs_array():
    jobs = [
        _job("completed", run_id="c1"),
        _job("completed", run_id="c2"),
        _job("completed", run_id="c3"),
        _job("failed", kind="hugin_crash", run_id="f1"),
        _job("failed", kind="gpu_lost", run_id="f2"),
        _job("pending", run_id="p1"),
    ]
    component = render_header(_payload(jobs))
    blob = _text_blob(component)
    assert "6 total" in blob
    assert "3 completed" in blob
    assert "2 failed" in blob
    assert "1 pending" in blob


def test_header_failure_pills_grouped_by_kind():
    jobs = [
        _job("failed", kind="hugin_crash", run_id="a"),
        _job("failed", kind="hugin_crash", run_id="b"),
        _job("failed", kind="gpu_lost", run_id="c"),
        _job("failed", kind="preset_unsupported", run_id="d"),
    ]
    component = render_header(_payload(jobs))
    blob = _text_blob(component)
    assert "hugin_crash: 2" in blob
    assert "gpu_lost: 1" in blob
    assert "preset_unsupported: 1" in blob


def test_header_failure_pill_ids_use_pattern_matching():
    jobs = [_job("failed", kind="hugin_crash", run_id="a")]
    component = render_header(_payload(jobs))
    pill_ids = [
        getattr(c, "id", None)
        for c in _walk(component)
        if isinstance(getattr(c, "id", None), dict)
        and getattr(c, "id", {}).get("type") == "tab-a-failure-pill"
    ]
    assert len(pill_ids) == 1
    assert pill_ids[0] == {"type": "tab-a-failure-pill", "kind": "hugin_crash"}


def test_header_no_failure_pills_when_no_failures():
    component = render_header(_payload([_job("completed")]))
    blob = _text_blob(component)
    assert "Failures:" not in blob


def test_header_short_commit_sha():
    component = render_header(_payload([_job("completed")], commit_sha="abc123def4567890"))
    blob = _text_blob(component)
    assert "abc123d" in blob  # first 7 chars
    assert "abc123def4567890" not in blob  # full sha not displayed


def test_header_handles_missing_commit_sha():
    component = render_header(_payload([_job("completed")], commit_sha=""))
    blob = _text_blob(component)
    assert "commit" not in blob.lower()
