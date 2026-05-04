# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the Tab A jobs table: rendering, filters, expand row, ssh-tail, empty states."""

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


def _job(
    *,
    run_id="r",
    task="Isaac-Ant-Direct-v0",
    status="completed",
    kind=None,
    attempts=1,
    started_at="2026-04-27T14:13:02Z",
    ended_at=None,
    host="v1",
):
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


def test_jobs_expanded_row_for_failed_in_expanded_set():
    job = _job(status="failed", kind="hugin_crash", run_id="rid-1")
    component = render_jobs_section(
        _payload([job]),
        expanded_run_ids={"rid-1"},
    )
    # The expand row exists.
    assert _has_class(component, "tab-a-expand-row")


def test_jobs_expanded_row_not_rendered_when_collapsed():
    job = _job(status="failed", kind="hugin_crash", run_id="rid-2")
    component = render_jobs_section(_payload([job]), expanded_run_ids=set())
    assert not _has_class(component, "tab-a-expand-row")


def test_jobs_expanded_row_ssh_tail_button_present():
    job = _job(status="failed", kind="hugin_crash", run_id="rid-3")
    component = render_jobs_section(_payload([job]), expanded_run_ids={"rid-3"})
    button_ids = [
        getattr(c, "id", None)
        for c in _walk(component)
        if isinstance(getattr(c, "id", None), dict) and getattr(c, "id", {}).get("type") == "tab-a-ssh-tail-button"
    ]
    assert button_ids == [{"type": "tab-a-ssh-tail-button", "run_id": "rid-3"}]


def test_jobs_expand_toggle_button_on_failed_rows():
    job = _job(status="failed", kind="hugin_crash", run_id="rid-4")
    component = render_jobs_section(_payload([job]))
    button_ids = [
        getattr(c, "id", None)
        for c in _walk(component)
        if isinstance(getattr(c, "id", None), dict) and getattr(c, "id", {}).get("type") == "tab-a-expand-toggle"
    ]
    assert button_ids == [{"type": "tab-a-expand-toggle", "run_id": "rid-4"}]


def test_jobs_expanded_row_ssh_tail_lines_rendered():
    job = _job(status="failed", kind="hugin_crash", run_id="rid-5")
    component = render_jobs_section(
        _payload([job]),
        expanded_run_ids={"rid-5"},
        ssh_tail_store={"rid-5": ["line one", "line two"]},
    )
    pre_blocks = [c for c in _walk(component) if type(c).__name__ == "Pre"]
    # There may be 1 (failure.message <pre>) or 2 (failure.message + ssh-tail) <pre> blocks.
    assert len(pre_blocks) >= 1
    # At least one of them contains the ssh-tail content.
    found = False
    for pre in pre_blocks:
        text = getattr(pre, "children", "")
        if isinstance(text, list):
            text = "".join(t for t in text if isinstance(t, str))
        if "line one" in text and "line two" in text:
            found = True
            break
    assert found


def test_jobs_expanded_row_ssh_tail_lines_empty_renders_not_found():
    job = _job(status="failed", kind="hugin_crash", run_id="rid-6")
    component = render_jobs_section(
        _payload([job]),
        expanded_run_ids={"rid-6"},
        ssh_tail_store={"rid-6": []},
    )
    blob = " ".join(
        getattr(c, "children", "") for c in _walk(component) if isinstance(getattr(c, "children", None), str)
    )
    assert "ssh-tail.log not found" in blob


def test_jobs_expanded_row_no_message_renders_friendly_text():
    job = {
        "run_id": "rid-7",
        "task_id": "x",
        "framework": "rsl_rl",
        "backend": "physx",
        "seed": 42,
        "status": "failed",
        "assigned_to": "v1",
        "attempts": 1,
        "started_at": None,
        "ended_at": None,
        "preferred_not": [],
        "failure": {"kind": "hugin_crash", "message": None, "details": {}},
    }
    component = render_jobs_section(_payload([job]), expanded_run_ids={"rid-7"})
    blob = " ".join(
        getattr(c, "children", "") for c in _walk(component) if isinstance(getattr(c, "children", None), str)
    )
    assert "(no failure message recorded)" in blob


def test_jobs_empty_state_when_filters_match_nothing():
    jobs = [_job(status="completed", run_id="c")]
    component = render_jobs_section(_payload(jobs), status_filter=["failed"])
    assert _has_id(component, "tab-a-jobs-empty")
    blob = " ".join(
        getattr(c, "children", "") for c in _walk(component) if isinstance(getattr(c, "children", None), str)
    )
    assert "No jobs match" in blob
    # Has a Clear button.
    assert _has_id(component, "tab-a-clear-filters")


def test_jobs_empty_state_when_dispatch_has_no_jobs():
    component = render_jobs_section(_payload([]))
    assert _has_id(component, "tab-a-jobs-empty-zero")
    blob = " ".join(
        getattr(c, "children", "") for c in _walk(component) if isinstance(getattr(c, "children", None), str)
    )
    assert "No jobs queued" in blob
    # No clear button when there are zero jobs at all.
    assert not _has_id(component, "tab-a-clear-filters")


def test_jobs_task_cell_links_to_tab_b():
    """The Task cell renders as a dcc.Link to Tab B with task/framework/backend params."""
    job = _job(
        run_id="r1",
        task="Isaac-Ant-Direct-v0",
        status="completed",
    )
    payload = _payload([job])
    payload["dispatch_id"] = "20260427-141302"
    component = render_jobs_section(payload)

    links = [c for c in _walk(component) if type(c).__name__ == "Link"]
    hrefs = [getattr(link, "href", None) for link in links]
    expected = "/20260427-141302/task-drilldown?task=Isaac-Ant-Direct-v0&framework=rsl_rl&backend=physx"
    assert expected in hrefs


def _retry_buttons(component):
    out = []
    for c in _walk(component):
        cls = (getattr(c, "className", "") or "").split()
        if "tab-a-retry-toggle" in cls:
            out.append(c)
    return out


def test_retry_button_only_on_failed_rows():
    """Completed / running rows must NOT carry a retry toggle — the only
    rows that make sense to re-run are failed (kind set) rows."""
    payload = _payload(
        [
            _job(run_id="ok", status="completed"),
            _job(run_id="fail", status="failed", kind="hugin_crash"),
            _job(run_id="run", status="running"),
        ]
    )
    component = render_jobs_section(payload)
    btns = _retry_buttons(component)
    assert len(btns) == 1
    btn_id = getattr(btns[0], "id", {})
    assert isinstance(btn_id, dict)
    assert btn_id.get("type") == "tab-a-retry-toggle"
    assert btn_id.get("run_id") == "fail"


def test_retry_button_renders_queued_state_when_run_id_in_queue():
    """When the run_id is already in the retry queue, the button gets the
    ``tab-a-retry-toggle-queued`` class and shows ``Queued ✓`` so the
    operator can see at a glance which rows are tagged."""
    payload = _payload([_job(run_id="r1", status="failed", kind="timeout")])
    component = render_jobs_section(payload, retry_queue={"r1"})
    btn = _retry_buttons(component)[0]
    classes = (getattr(btn, "className", "") or "").split()
    assert "tab-a-retry-toggle-queued" in classes
    assert btn.children == "Queued ✓"


def test_retry_button_default_state_when_not_in_queue():
    """Untagged failed rows show ``Retry`` next to the status pill,
    visually parallel to the Kill/Skip cancel buttons on running and
    pending rows."""
    payload = _payload([_job(run_id="r1", status="failed", kind="timeout")])
    component = render_jobs_section(payload, retry_queue=set())
    btn = _retry_buttons(component)[0]
    classes = (getattr(btn, "className", "") or "").split()
    assert "tab-a-retry-toggle-queued" not in classes
    assert btn.children == "Retry"


def test_retry_banner_renders_when_queue_nonempty():
    """Banner appears only when at least one row is queued; carries the
    operator's --resume command with the run_ids comma-joined."""
    payload = _payload(
        [
            _job(run_id="r1", status="failed", kind="timeout"),
            _job(run_id="r2", status="failed", kind="timeout"),
        ]
    )
    payload["dispatch_id"] = "20260427-141302"
    component = render_jobs_section(payload, retry_queue={"r1", "r2"})
    assert _has_id(component, "tab-a-retry-banner")
    text = " ".join(
        str(getattr(c, "children", "")) for c in _walk(component) if isinstance(getattr(c, "children", None), str)
    )
    assert "2 job(s) tagged for retry" in text
    assert "--resume 20260427-141302" in text
    # Run ids are sorted in the csv to be deterministic.
    assert "--retry-failed=r1,r2" in text


def test_retry_banner_absent_when_queue_empty():
    payload = _payload([_job(run_id="r1", status="failed", kind="timeout")])
    component = render_jobs_section(payload, retry_queue=set())
    assert not _has_id(component, "tab-a-retry-banner")
