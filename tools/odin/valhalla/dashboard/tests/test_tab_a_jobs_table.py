# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the Tab A jobs table (rendering only; expand row in T7)."""

from __future__ import annotations

from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.jobs_table import render_jobs_section


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


def _has_id(component, target_id) -> bool:
    return any(getattr(c, "id", None) == target_id for c in _walk(component))


def _has_class(component, cls) -> bool:
    for c in _walk(component):
        c_cls = getattr(c, "className", "") or ""
        if cls in c_cls.split():
            return True
    return False


def _job(*, run_id="r", task="Isaac-Ant-Direct-v0", status="completed", kind=None,
         attempts=1, started_at="2026-04-27T14:13:02Z", ended_at=None, host="v1"):
    j = {
        "run_id": run_id,
        "task_id": task,
        "framework": "rsl_rl",
        "backend": "physx",
        "seed": 42,
        "status": status,
        "assigned_to": host,
        "attempts": attempts,
        "started_at": started_at,
        "ended_at": ended_at,
        "preferred_not": [],
        "failure": None,
    }
    if status == "failed" and kind is not None:
        j["failure"] = {"kind": kind, "message": "long stderr text here", "details": {}}
    return j


def _payload(jobs):
    return {"schema_version": "1.3", "dispatch_id": "d", "jobs": jobs}


def test_jobs_renders_filter_row_with_three_controls():
    component = render_jobs_section(_payload([_job()]))
    # Status dropdown
    assert _has_id(component, "tab-a-status-filter")
    # Failure-kind dropdown
    assert _has_id(component, "tab-a-kind-filter")
    # Task-text input
    assert _has_id(component, "tab-a-task-text")


def test_jobs_renders_one_row_per_job():
    jobs = [_job(run_id=f"r{i}") for i in range(5)]
    component = render_jobs_section(_payload(jobs))
    # 1 header row + 5 data rows.
    rows = [c for c in _walk(component) if type(c).__name__ == "Tr"]
    assert len(rows) == 6


def test_jobs_status_pill_per_status():
    statuses = [
        ("pending", "tab-a-job-status-pending"),
        ("running", "tab-a-job-status-running"),
        ("completed", "tab-a-job-status-completed"),
        ("failed", "tab-a-job-status-failed"),
    ]
    for status, cls in statuses:
        kind = "hugin_crash" if status == "failed" else None
        component = render_jobs_section(_payload([_job(status=status, kind=kind)]))
        assert _has_class(component, cls), f"missing pill class for status={status!r}"


def test_jobs_failure_kind_column_filled_for_failed_only():
    jobs = [
        _job(status="completed", run_id="c"),
        _job(status="failed", kind="hugin_crash", run_id="f"),
        _job(status="running", run_id="r"),
    ]
    component = render_jobs_section(_payload(jobs))
    # The failed row's kind pill is rendered.
    assert _has_class(component, "tab-a-kind-pill-hugin_crash")


def test_jobs_relative_started_at():
    import re

    component = render_jobs_section(_payload([_job(started_at="2026-04-27T14:13:02Z")]))
    blob_parts = []
    for c in _walk(component):
        ch = getattr(c, "children", None)
        if isinstance(ch, str):
            blob_parts.append(ch)
    blob = " ".join(blob_parts)
    # Loose pattern: at least one "<num><unit> ago" text in the row.
    assert re.search(r"\d+\s*[smhd]\s*ago", blob, re.IGNORECASE) or "ago" in blob


def test_jobs_attempts_badge_only_when_gt_1():
    component_one = render_jobs_section(_payload([_job(attempts=1)]))
    component_two = render_jobs_section(_payload([_job(attempts=2)]))
    assert not _has_class(component_one, "tab-a-attempts-badge")
    assert _has_class(component_two, "tab-a-attempts-badge")
