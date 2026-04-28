# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the multi-dispatch landing page."""

from __future__ import annotations

import json
from pathlib import Path

from tools.odin.valhalla.dashboard.app import _landing
from tools.odin.valhalla.dashboard.data import DataLayer


def _write_dispatch(
    runs_root: Path,
    dispatch_id: str,
    *,
    jobs_total: int = 0,
    completed: int = 0,
    failed: int = 0,
    ended_at: str | None = None,
) -> Path:
    d = runs_root / dispatch_id
    d.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i in range(completed):
        jobs.append({"run_id": f"c-{i}", "status": "completed", "assigned_to": "v1"})
    for i in range(failed):
        jobs.append({"run_id": f"f-{i}", "status": "failed", "assigned_to": "v1"})
    while len(jobs) < jobs_total:
        jobs.append({"run_id": f"p-{len(jobs)}", "status": "pending", "assigned_to": None})
    payload = {
        "schema_version": "1.3",
        "dispatch_id": dispatch_id,
        "started_at": "2026-04-27T14:13:02Z",
        "ended_at": ended_at,
        "seeds": [42],
        "commit_sha": "abc",
        "fleet": [{"host": "v1", "status": "idle", "current_run_id": None, "last_error": None}],
        "jobs": jobs,
        "skipped": [],
    }
    (d / "dispatch.json").write_text(json.dumps(payload))
    return d


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


def test_landing_with_no_dispatches_renders_empty_state(tmp_path):
    """No dispatches → an empty-state message instead of an empty table."""
    component = _landing(DataLayer(tmp_path))
    text_components = [
        c for c in _walk(component) if hasattr(c, "children") and isinstance(getattr(c, "children", None), str)
    ]
    text_blob = " ".join(c.children for c in text_components)
    assert "No dispatches" in text_blob


def test_landing_renders_one_row_per_dispatch(tmp_path):
    """Three dispatches → three table rows (in addition to the header)."""
    _write_dispatch(tmp_path, "20260427-141302", jobs_total=3, completed=2, failed=1, ended_at="2026-04-27T14:30:00Z")
    _write_dispatch(tmp_path, "20260425-080000", jobs_total=4, completed=4, ended_at="2026-04-25T08:30:00Z")
    _write_dispatch(tmp_path, "20260424-160119", jobs_total=15, completed=15, ended_at="2026-04-24T16:30:00Z")

    component = _landing(DataLayer(tmp_path))
    table_rows = [c for c in _walk(component) if type(c).__name__ == "Tr"]
    # 1 header row + 3 data rows
    assert len(table_rows) == 4


def test_landing_row_contains_dispatch_id_and_link(tmp_path):
    """Each row has the dispatch_id text and a link to /<id>/."""
    _write_dispatch(tmp_path, "20260427-141302", jobs_total=1)
    component = _landing(DataLayer(tmp_path))
    links = [c for c in _walk(component) if type(c).__name__ == "A"]
    hrefs = [getattr(link, "href", None) for link in links]
    assert "/20260427-141302/" in hrefs


def test_landing_sorts_newest_first(tmp_path):
    """Three dispatches → rendered newest-first in the rendered DOM."""
    _write_dispatch(tmp_path, "20260424-160119", jobs_total=1)
    _write_dispatch(tmp_path, "20260427-141302", jobs_total=1)
    _write_dispatch(tmp_path, "20260425-080000", jobs_total=1)

    component = _landing(DataLayer(tmp_path))
    # Collect anchor hrefs in DOM order — they should match newest-first.
    links = [c for c in _walk(component) if type(c).__name__ == "A"]
    hrefs = [getattr(link, "href", "") for link in links]
    dispatch_hrefs = [h for h in hrefs if h.startswith("/2026")]
    assert dispatch_hrefs == [
        "/20260427-141302/",
        "/20260425-080000/",
        "/20260424-160119/",
    ]
