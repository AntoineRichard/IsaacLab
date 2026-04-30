# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Tab A running-job tail rendering."""

from __future__ import annotations

from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.jobs_table import _expand_running_row, render_jobs_section


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


def _job(*, run_id="r", status="running", kind=None):
    job = {
        "run_id": run_id,
        "task_id": "Isaac-Ant-Direct-v0",
        "framework": "rsl_rl",
        "backend": "physx",
        "seed": 42,
        "status": status,
        "assigned_to": "v1",
        "attempts": 1,
        "started_at": "2026-04-30T12:00:00Z",
        "ended_at": None,
        "preferred_not": [],
        "failure": None,
    }
    if status == "failed":
        job["failure"] = {"kind": kind or "hugin_crash", "message": "failed", "details": {}}
    return job


def _payload(jobs):
    return {"schema_version": "1.3", "dispatch_id": "20260430-110509", "jobs": jobs}


def _ids_of_type(component, type_name: str) -> list[dict]:
    return [
        getattr(c, "id", None)
        for c in _walk(component)
        if isinstance(getattr(c, "id", None), dict) and getattr(c, "id", {}).get("type") == type_name
    ]


def _text_blob(component) -> str:
    parts = []
    for c in _walk(component):
        children = getattr(c, "children", None)
        if isinstance(children, str):
            parts.append(children)
    return " ".join(parts)


def test_running_row_has_tail_button():
    component = render_jobs_section(_payload([_job(run_id="rid-running", status="running")]))

    assert _ids_of_type(component, "tab-a-running-tail-toggle") == [
        {"type": "tab-a-running-tail-toggle", "run_id": "rid-running"}
    ]


def test_completed_row_does_not_have_tail_button():
    component = render_jobs_section(_payload([_job(run_id="rid-done", status="completed")]))

    assert _ids_of_type(component, "tab-a-running-tail-toggle") == []


def test_failed_row_has_retry_and_expand_but_not_tail_button():
    component = render_jobs_section(_payload([_job(run_id="rid-failed", status="failed", kind="timeout")]))

    assert _ids_of_type(component, "tab-a-expand-toggle") == [{"type": "tab-a-expand-toggle", "run_id": "rid-failed"}]
    assert _ids_of_type(component, "tab-a-retry-toggle") == [{"type": "tab-a-retry-toggle", "run_id": "rid-failed"}]
    assert _ids_of_type(component, "tab-a-running-tail-toggle") == []


def test_expand_running_row_renders_lines_in_pre_block():
    component = _expand_running_row(
        _job(run_id="rid-running"),
        {"source": "training.stdout.log", "lines": ["iter 1", "iter 2"], "fetched_at": "2026-04-30T12:01:00Z"},
    )

    pre_blocks = [c for c in _walk(component) if type(c).__name__ == "Pre"]
    assert len(pre_blocks) == 1
    assert pre_blocks[0].children == "iter 1\niter 2"
    assert _ids_of_type(component, "tab-a-running-tail-refresh") == [
        {"type": "tab-a-running-tail-refresh", "run_id": "rid-running"}
    ]


def test_expand_running_row_shows_filename_marker():
    for source in ("training.stdout.log", "startup.stdout.log"):
        component = _expand_running_row(
            _job(run_id=f"rid-{source}"),
            {"source": source, "lines": ["x"], "fetched_at": "2026-04-30T12:01:00Z"},
        )
        assert source in _text_blob(component)


def test_expand_running_row_shows_transport_warning():
    component = _expand_running_row(
        _job(run_id="rid-running"),
        {"source": None, "lines": [], "warning": "connection timed out", "fetched_at": "2026-04-30T12:01:00Z"},
    )

    text = _text_blob(component)
    assert "Running stdout tail unavailable" in text
    assert "connection timed out" in text


def test_render_jobs_section_inserts_running_expand_row_when_shown():
    component = render_jobs_section(
        _payload([_job(run_id="rid-open", status="running")]),
        running_tail_shown={"rid-open"},
        running_tail_store={
            "rid-open": {
                "source": "training.stdout.log",
                "lines": ["reward=1.0"],
                "fetched_at": "2026-04-30T12:01:00Z",
            }
        },
    )

    rows = [c for c in _walk(component) if type(c).__name__ == "Tr"]
    assert len(rows) == 3
    assert "reward=1.0" in _text_blob(component)
