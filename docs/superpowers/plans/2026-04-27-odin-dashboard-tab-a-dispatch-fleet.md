# Odin Dashboard — Tab A (Spec 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Tab A — Dispatch & Fleet — the per-dispatch health view that renders header (live/done pill, totals, click-to-filter failure pills), fleet table (one row per host with 8 inline columns), and a filtered jobs table with inline failure expansion + ssh-tail loader. Auto-polls `dispatch.json` every 5s while live.

**Architecture:** New sub-package `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/` with seven files split by responsibility (layout / header / fleet_table / jobs_table / filters / ssh_tail / callbacks). Tab module exposes `render(dispatch_id, tab_id)` and `register(app, data)` for Spec 0's registry. Callbacks fire on a 5s `dcc.Interval` and on user clicks (filters, expand toggles, ssh-tail buttons, failure pills).

**Tech Stack:** Python 3.10+, Plotly Dash 4.x (`dcc.Interval`, `dcc.Store`, pattern-matching ids), pure-Python pytest. No new pip deps; reuses Spec 0's `DataLayer` + dash/plotly/pandas.

**Branch:** `antoiner/feat/odin` (continues from spec commit `c2c836a0514`).

**Spec:** `docs/superpowers/specs/2026-04-27-odin-dashboard-tab-a-dispatch-fleet-design.md`.

**Visual reference (read-only, do not edit):**
- `.superpowers/brainstorm/2312694-1777385893/content/fleet-row-vs-card.html` — fleet table layout (option A)
- `.superpowers/brainstorm/2312694-1777385893/content/jobs-table.html` — jobs table layout (option B+C)

These mockups are the visual source of truth for status pill colors, kind pill colors, table styling, and dark theme. Match their structure when wiring Dash components.

---

## Conventions used in every task

- **Test runner:** `PYTHONPATH=. python3 -m pytest <test> -v --tb=short --noconftest -p no:cacheprovider`. Pure-Python tests (no Isaac Sim, no Dash server). 0.03s per test.
- **Run tests one at a time** — never batch. Per the established discipline on this branch.
- **Pre-commit:** `./isaaclab.sh -f` BEFORE `git commit`. Restage and rerun until clean.
- **Commit message:** Imperative subject ≤ 50 chars; body wrapped at 72 chars; **NO** AI co-authorship lines (per `AGENTS.md`).
- **No new pip deps.** dash/plotly/pandas already installed in T1 of Spec 0.
- **Dash 4.x note:** `app.run` (not `run_server`); `app.callback` decorator and `dcc.Store` work normally; `dash.dcc.Location` for redirects; pattern-matching ids use `{"type": "...", "<key>": ALL}` format.

## File map — what gets created or modified

| File | Owner task | Responsibility |
|---|---|---|
| `tools/odin/valhalla/dashboard/app.py` | T1 | Extend `_register_callbacks` to also call `tab_module.register(app, data)` at app startup if the tab module exposes it. |
| `tools/odin/valhalla/dashboard/tests/test_app.py` | T1 | New test for the registry's register-hook walking. |
| `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/__init__.py` | T2 | Package marker; re-exports `render` + `register`. |
| `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/layout.py` | T2 | `build_layout(dispatch_id)` — static layout with stores + slots. |
| `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/header.py` | T3 | `render_header(payload)` — title, live/done pill, totals, failure pills. |
| `tools/odin/valhalla/dashboard/tests/test_tab_a_header.py` | T3 | Header render tests (8 cases). |
| `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/fleet_table.py` | T4 | `render_fleet_table(payload, hw, fallback)` — 8-column rows, hardware lookup. |
| `tools/odin/valhalla/dashboard/tests/test_tab_a_fleet_table.py` | T4 | Fleet table tests (11 cases). |
| `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/filters.py` | T5 | `filter_jobs(jobs, ...)` — pure data function. |
| `tools/odin/valhalla/dashboard/tests/test_tab_a_filters.py` | T5 | Filter tests (8 cases). |
| `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py` (initial) | T6 | `render_jobs_section` — filter row + table rows; no expand yet. |
| `tools/odin/valhalla/dashboard/tests/test_tab_a_jobs_table.py` (partial) | T6 | First 6 cases (filter row + row rendering). |
| `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py` (expand) | T7 | Add expand-row + ssh-tail render to `render_jobs_section`. |
| `tools/odin/valhalla/dashboard/tests/test_tab_a_jobs_table.py` (rest) | T7 | Remaining 10 cases (expand, ssh-tail render, empty states). |
| `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/ssh_tail.py` | T8 | `load_ssh_tail` — file reader with truncation. |
| `tools/odin/valhalla/dashboard/tests/test_tab_a_ssh_tail.py` | T8 | ssh-tail tests (6 cases). |
| `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/callbacks.py` (initial) | T9 | `register_callbacks` + 4 callbacks (header/fleet/jobs/failure-pill). |
| `tools/odin/valhalla/dashboard/tests/test_tab_a_callbacks.py` (partial) | T9 | First 5 callback tests. |
| `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/callbacks.py` (rest) | T10 | Add 2 pattern-matching callbacks (expand-toggle, ssh-tail-load). |
| `tools/odin/valhalla/dashboard/tests/test_tab_a_callbacks.py` (rest) | T10 | Remaining 4 callback tests. |
| `docs/odin/architecture.md` | T11 | Change-log entry; "Last updated" bump. |

---

## Task 1: Spec 0 registry — call `tab_module.register(app, data)` at startup

**Files:**
- Modify: `tools/odin/valhalla/dashboard/app.py` (`_register_callbacks`)
- Modify: `tools/odin/valhalla/dashboard/tests/test_app.py`

Spec 0's `_register_callbacks(app, data)` only registers the URL routing callback. Tab A's `dcc.Interval` callbacks need to be registered at app-startup time too. We extend the function to walk the three known tab module names, import each, and call `register(app, data)` if the module exposes it. Backward-compatible: Spec 0's `_placeholder` doesn't have `register`, so it's never called.

- [ ] **Step 1.1: Append failing test** to `tools/odin/valhalla/dashboard/tests/test_app.py`

```python
def test_create_app_calls_tab_register_when_present(monkeypatch, tmp_path):
    """If a tab module exposes register(), create_app calls it once during startup."""
    import sys
    import types

    fake_module = types.ModuleType("tools.odin.valhalla.dashboard.tabs.dispatch_fleet")
    register_calls: list[tuple] = []

    def _register(app, data):
        register_calls.append((app, data))

    fake_module.register = _register
    monkeypatch.setitem(sys.modules, "tools.odin.valhalla.dashboard.tabs.dispatch_fleet", fake_module)

    app = create_app(tmp_path)
    assert len(register_calls) == 1
    registered_app, registered_data = register_calls[0]
    assert registered_app is app


def test_create_app_skips_tab_register_when_absent(tmp_path):
    """If a tab module has no register() (e.g., not yet implemented), skip silently."""
    # No monkeypatch; in real test environment, none of the three tab modules
    # exist yet (Spec 1 hasn't landed in this test's perspective). create_app
    # should not raise.
    app = create_app(tmp_path)
    assert app is not None
```

- [ ] **Step 1.2: Run new tests, verify they FAIL**

```
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_app.py::test_create_app_calls_tab_register_when_present -v --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_app.py::test_create_app_skips_tab_register_when_absent -v --tb=short --noconftest -p no:cacheprovider
```

Expected: `test_create_app_calls_tab_register_when_present` FAILs because `register` is never called; `test_create_app_skips_tab_register_when_absent` may already PASS (the absence path is implicit today).

- [ ] **Step 1.3: Modify `_register_callbacks` in `tools/odin/valhalla/dashboard/app.py`**

Replace the existing `_register_callbacks` function:

```python
def _register_callbacks(app: dash.Dash, data: DataLayer) -> None:
    @app.callback(Output("page-content", "children"), Input("url", "pathname"))
    def _on_url(pathname: str):
        return route_pathname(pathname or "/", data)

    _register_tab_callbacks(app, data)


def _register_tab_callbacks(app: dash.Dash, data: DataLayer) -> None:
    """Walk the three known tab module names; call register(app, data) if present.

    Spec 0's placeholder has no register(); Specs 1/2/3 add modules that wire
    their dcc.Interval / pattern-matching callbacks at app startup. Importing
    a missing module is silently OK — that just means the tab spec hasn't
    landed yet.
    """
    import importlib

    for module_name in (
        "tools.odin.valhalla.dashboard.tabs.dispatch_fleet",
        "tools.odin.valhalla.dashboard.tabs.task_drilldown",
        "tools.odin.valhalla.dashboard.tabs.startup",
    ):
        try:
            tab_module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        register_fn = getattr(tab_module, "register", None)
        if register_fn is not None:
            register_fn(app, data)
```

- [ ] **Step 1.4: Run new tests, verify they PASS**

Run each individually with the same command shape. Expected: 2/2 pass.

- [ ] **Step 1.5: Run all existing app + landing tests** to confirm no regression

```
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_app.py --tb=short --noconftest -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_app_landing.py --tb=short --noconftest -p no:cacheprovider
```

Expected: all pre-existing tests still pass.

- [ ] **Step 1.6: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/app.py tools/odin/valhalla/dashboard/tests/test_app.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Dashboard registry: call tab.register(app, data) at startup

Spec 0's _register_callbacks only wired the URL routing callback.
Tab A (Spec 1+) needs its dcc.Interval / pattern-matching callbacks
registered at app startup. Walk the three known tab module names;
call register(app, data) if the module exposes it. Backward-
compatible: Spec 0's placeholder has no register(), so it's never
called. Missing tab modules (specs not yet landed) are silently
skipped.
EOF
)"
```

---

## Task 2: Tab A package skeleton + `layout.py`

**Files:**
- Create: `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/__init__.py`
- Create: `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/layout.py`
- Modify: `tools/odin/valhalla/dashboard/tests/test_app.py`

This task delivers the package marker + `build_layout` skeleton. Tests assert that the layout has the expected slot ids; subsequent tasks fill those slots.

- [ ] **Step 2.1: Append failing tests** to `tools/odin/valhalla/dashboard/tests/test_app.py`

```python
def test_tab_a_render_returns_layout_with_expected_slots(tmp_path):
    """Tab A's render() returns a Div with the expected dynamic slots and stores."""
    _write_dispatch(tmp_path, "20260427-141302")
    from tools.odin.valhalla.dashboard.data import DataLayer
    from tools.odin.valhalla.dashboard.tabs import dispatch_fleet

    data = DataLayer(tmp_path)
    component = dispatch_fleet.render("20260427-141302", "dispatch-fleet")

    # Top-level container.
    assert _has_id(component, "tab-a-root")

    # Stores (so callbacks can write to them).
    assert _has_id(component, "tab-a-dispatch-id")
    assert _has_id(component, "tab-a-failure-filter")
    assert _has_id(component, "tab-a-expanded-run-ids")
    assert _has_id(component, "tab-a-ssh-tail-store")

    # Tick interval.
    assert _has_id(component, "tab-a-tick")

    # Empty content slots — populated by callbacks.
    assert _has_id(component, "tab-a-header")
    assert _has_id(component, "tab-a-fleet-table")
    assert _has_id(component, "tab-a-jobs-section")


def test_tab_a_layout_replaces_placeholder(tmp_path):
    """When real tab module is present, route_pathname returns the real layout, not the placeholder."""
    _write_dispatch(tmp_path, "20260427-141302")
    from tools.odin.valhalla.dashboard.data import DataLayer

    data = DataLayer(tmp_path)
    component = route_pathname("/20260427-141302/dispatch-fleet", data)
    assert _has_id(component, "tab-a-root")
    assert not _has_id(component, "tab-placeholder")
```

- [ ] **Step 2.2: Run new tests, verify they FAIL**

Run each individually. Expected: each → FAIL with `ModuleNotFoundError: No module named 'tools.odin.valhalla.dashboard.tabs.dispatch_fleet'`.

- [ ] **Step 2.3: Create `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/__init__.py`**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tab A — Dispatch & Fleet — for the Odin dashboard."""

from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.layout import build_layout

__all__ = ["render", "register"]


def render(dispatch_id: str, tab_id: str):
    """Spec 0 registry hook — return the static layout for this tab."""
    return build_layout(dispatch_id)


def register(app, data):
    """Spec 0 registry hook — wire Tab A's callbacks at app startup.

    Lazy-imported to avoid pulling Dash callbacks into the module graph at
    test collection time when only `render` is needed.
    """
    from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.callbacks import register_callbacks

    register_callbacks(app, data)
```

- [ ] **Step 2.4: Create `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/layout.py`**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static layout for /<id>/dispatch-fleet.

All dynamic content lives in slots (id="tab-a-..."); callbacks fill them on
mount and on every dcc.Interval tick.
"""

from __future__ import annotations

from dash import dcc, html

__all__ = ["build_layout"]


_TICK_MS = 5_000


def build_layout(dispatch_id: str) -> html.Div:
    """Return the Tab A static layout for ``dispatch_id``.

    Stores carry per-page state (filters, expansion set, ssh-tail cache).
    Slots are empty Divs that callbacks populate on every tick.
    """
    return html.Div(
        id="tab-a-root",
        children=[
            dcc.Interval(id="tab-a-tick", interval=_TICK_MS, n_intervals=0),
            dcc.Store(id="tab-a-dispatch-id", storage_type="memory", data=dispatch_id),
            dcc.Store(id="tab-a-failure-filter", storage_type="memory", data=None),
            dcc.Store(id="tab-a-expanded-run-ids", storage_type="memory", data=[]),
            dcc.Store(id="tab-a-ssh-tail-store", storage_type="memory", data={}),
            html.Div(id="tab-a-header"),
            html.Div(id="tab-a-fleet-table"),
            html.Div(id="tab-a-jobs-section"),
        ],
    )
```

- [ ] **Step 2.5: Run each test individually, verify all PASS**

- [ ] **Step 2.6: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add -f tools/odin/valhalla/dashboard/tabs/dispatch_fleet/__init__.py
git add tools/odin/valhalla/dashboard/tabs/dispatch_fleet/layout.py tools/odin/valhalla/dashboard/tests/test_app.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Tab A skeleton: package + layout with stores and slots

build_layout(dispatch_id) returns the static Tab A layout: a tick
interval (5s), four memory-backed stores (dispatch-id, failure-filter,
expanded-run-ids, ssh-tail-store), and three empty content slots
(header, fleet-table, jobs-section). Subsequent tasks fill the slots
via callbacks. The __init__ exports render() and register() for the
Spec 0 registry; register() lazy-imports callbacks so test collection
of just render() doesn't pull in Dash's callback machinery.
EOF
)"
```

---

## Task 3: `header.py` — totals, live/done pill, failure pills

**Files:**
- Create: `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/header.py`
- Create: `tools/odin/valhalla/dashboard/tests/test_tab_a_header.py`

- [ ] **Step 3.1: Write failing tests** — `tools/odin/valhalla/dashboard/tests/test_tab_a_header.py`

```python
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
```

- [ ] **Step 3.2: Run new tests, verify they FAIL**

Each individually. Expected: each → FAIL with `ModuleNotFoundError`.

- [ ] **Step 3.3: Implement `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/header.py`**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the Tab A header: title, live/done pill, totals, failure pills."""

from __future__ import annotations

from collections import Counter

from dash import html

__all__ = ["render_header"]


def render_header(dispatch_payload: dict) -> html.Div:
    """Build the header strip for a parsed dispatch.json payload.

    Pure render — no Dash callbacks, no DataLayer access. Returns a Div with
    id ``tab-a-header-content`` containing the title row, live/done pill,
    totals row, and (when there are failures) the click-to-filter pill row.
    """
    dispatch_id = str(dispatch_payload.get("dispatch_id", ""))
    commit_sha = str(dispatch_payload.get("commit_sha", "") or "")
    ended_at = dispatch_payload.get("ended_at")
    fleet = dispatch_payload.get("fleet", []) or []
    jobs = dispatch_payload.get("jobs", []) or []
    skipped = dispatch_payload.get("skipped", []) or []

    title_parts = [f"Dispatch {dispatch_id}"]
    if commit_sha:
        title_parts.append(f"commit {commit_sha[:7]}")
    title_parts.append(f"{len(fleet)} hosts")
    title_text = " · ".join(title_parts)

    pill = (
        html.Span("● Live", className="tab-a-live-pill")
        if ended_at is None
        else html.Span("✓ Done", className="tab-a-done-pill")
    )

    by_status = Counter(j.get("status", "unknown") for j in jobs)
    totals_text = (
        f"{len(jobs)} total · "
        f"{by_status.get('completed', 0)} completed · "
        f"{by_status.get('failed', 0)} failed · "
        f"{by_status.get('pending', 0)} pending · "
        f"{len(skipped)} skipped"
    )

    children: list = [
        html.Div(title_text, className="tab-a-header-title"),
        html.Div(pill, className="tab-a-header-pill-row"),
        html.Div(totals_text, className="tab-a-header-totals"),
    ]

    by_kind = Counter(
        (j.get("failure") or {}).get("kind", "unknown")
        for j in jobs
        if j.get("status") == "failed" and j.get("failure")
    )
    if by_kind:
        pill_children: list = [html.Span(f"Failures: {sum(by_kind.values())}  ")]
        for kind, count in sorted(by_kind.items()):
            pill_children.append(
                html.Button(
                    f"{kind}: {count}",
                    id={"type": "tab-a-failure-pill", "kind": kind},
                    n_clicks=0,
                    className=f"tab-a-failure-pill tab-a-failure-pill-{kind}",
                )
            )
        children.append(html.Div(pill_children, className="tab-a-header-failure-pills"))

    return html.Div(id="tab-a-header-content", children=children)
```

- [ ] **Step 3.4: Run each test individually, verify all PASS**

8 tests total.

- [ ] **Step 3.5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/tabs/dispatch_fleet/header.py tools/odin/valhalla/dashboard/tests/test_tab_a_header.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Tab A header: live pill + totals + failure pills

render_header(dispatch_payload) → Div(id='tab-a-header-content') with
title row (dispatch_id · commit_sha[:7] · N hosts), live/done pill,
totals (total · completed · failed · pending · skipped), and a row of
clickable per-kind failure pills using pattern-matching ids
({type: tab-a-failure-pill, kind: <kind>}). Empty failure-pill row
when there are zero failures.
EOF
)"
```

---

## Task 4: `fleet_table.py` — 8-column rows + hardware lookup

**Files:**
- Create: `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/fleet_table.py`
- Create: `tools/odin/valhalla/dashboard/tests/test_tab_a_fleet_table.py`

- [ ] **Step 4.1: Write failing tests** — `tools/odin/valhalla/dashboard/tests/test_tab_a_fleet_table.py`

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the Tab A fleet table."""

from __future__ import annotations

from tools.odin.valhalla.dashboard.data import HardwareInfo
from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.fleet_table import render_fleet_table


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


def _payload(fleet, *, jobs=None):
    return {
        "schema_version": "1.3",
        "dispatch_id": "d",
        "fleet": fleet,
        "jobs": jobs or [],
    }


def _hw_payload(hosts):
    return {
        "schema_version": "1.0",
        "dispatch_id": "d",
        "fingerprint": "gpu:NVIDIA-L40",
        "hosts": hosts,
    }


def _hw_block(*, hostname="h", gpu="NVIDIA L40", mem=44.32, cpu="Xeon", count=16, ram=62.79):
    return {
        "hostname": hostname,
        "gpu_devices": [{"name": gpu, "mem_gb": mem, "compute_cap": "8.9"}],
        "cpu_name": cpu,
        "cpu_count": count,
        "ram_gb": ram,
        "sourced_from": "r1",
    }


def test_fleet_renders_one_row_per_host():
    fleet = [
        {"host": "v1", "status": "idle", "current_run_id": None, "last_error": None},
        {"host": "v2", "status": "busy", "current_run_id": "r1", "last_error": None},
    ]
    component = render_fleet_table(_payload(fleet), None, lambda host: None)
    rows = [c for c in _walk(component) if type(c).__name__ == "Tr"]
    # 1 header row + 2 data rows
    assert len(rows) == 3


def test_fleet_status_pill_idle_busy_down():
    for status, expected_class in [
        ("idle", "tab-a-fleet-status-idle"),
        ("busy", "tab-a-fleet-status-busy"),
        ("down", "tab-a-fleet-status-down"),
    ]:
        fleet = [{"host": "v1", "status": status, "current_run_id": None, "last_error": None}]
        component = render_fleet_table(_payload(fleet), None, lambda h: None)
        assert _has_class(component, expected_class), f"missing class for status={status!r}"


def test_fleet_current_run_link_for_busy_host():
    fleet = [
        {"host": "v1", "status": "busy", "current_run_id": "rsl-rl_physx_X_seed42", "last_error": None}
    ]
    component = render_fleet_table(_payload(fleet), None, lambda h: None)
    anchors = [c for c in _walk(component) if type(c).__name__ == "A"]
    assert len(anchors) == 1
    assert "seed42" in (getattr(anchors[0], "children", "") or "")


def test_fleet_current_run_dash_when_idle():
    fleet = [{"host": "v1", "status": "idle", "current_run_id": None, "last_error": None}]
    component = render_fleet_table(_payload(fleet), None, lambda h: None)
    blob = _text_blob(component)
    # Has the em-dash placeholder somewhere on the row.
    assert "—" in blob


def test_fleet_hardware_from_hardware_json():
    fleet = [{"host": "v1", "status": "idle", "current_run_id": None, "last_error": None}]
    hw = _hw_payload({"v1": _hw_block(hostname="Odin-Runner-5")})
    component = render_fleet_table(_payload(fleet), hw, lambda h: None)
    blob = _text_blob(component)
    assert "Odin-Runner-5" in blob
    assert "NVIDIA L40" in blob
    assert "44.32" in blob or "44.3" in blob


def test_fleet_hardware_falls_back_to_lookup():
    fleet = [{"host": "v1", "status": "idle", "current_run_id": None, "last_error": None}]

    def fallback(host):
        if host == "v1":
            return HardwareInfo(
                hostname="Odin-Fallback",
                gpu_devices=[{"name": "NVIDIA L40", "mem_gb": 44.32, "compute_cap": "8.9"}],
                cpu_name="Xeon",
                cpu_count=16,
                ram_gb=62.79,
                sourced_from="prev/r1",
            )
        return None

    component = render_fleet_table(_payload(fleet), None, fallback)
    blob = _text_blob(component)
    assert "Odin-Fallback" in blob


def test_fleet_hardware_dash_when_unknown():
    fleet = [{"host": "v1", "status": "idle", "current_run_id": None, "last_error": None}]
    component = render_fleet_table(_payload(fleet), None, lambda h: None)
    blob = _text_blob(component)
    # Several em-dashes; at minimum the GPU/CPU/RAM cells are dashes.
    assert blob.count("—") >= 3


def test_fleet_last_event_recovered_pill():
    fleet = [
        {"host": "v1", "status": "idle", "current_run_id": None, "last_error": "gpu_lost: recovered"}
    ]
    component = render_fleet_table(_payload(fleet), None, lambda h: None)
    assert _has_class(component, "tab-a-event-recovered")


def test_fleet_last_event_recovery_failed_pill():
    fleet = [
        {
            "host": "v1",
            "status": "down",
            "current_run_id": None,
            "last_error": "gpu_lost: recovery_failed (docker_restart_failed: daemon down)",
        }
    ]
    component = render_fleet_table(_payload(fleet), None, lambda h: None)
    assert _has_class(component, "tab-a-event-recovery-failed")


def test_fleet_last_event_dash_when_no_error():
    fleet = [{"host": "v1", "status": "idle", "current_run_id": None, "last_error": None}]
    component = render_fleet_table(_payload(fleet), None, lambda h: None)
    rows = [c for c in _walk(component) if type(c).__name__ == "Tr"]
    # The last cell of the data row should be an em-dash.
    last_data_row = rows[-1]
    last_cell = list(_walk(last_data_row))[-1]
    last_cell_text = getattr(last_cell, "children", None)
    assert last_cell_text == "—"


def test_fleet_fallback_lookup_called_at_most_once_per_host():
    fleet = [
        {"host": "v1", "status": "idle", "current_run_id": None, "last_error": None},
        {"host": "v2", "status": "idle", "current_run_id": None, "last_error": None},
    ]
    call_log: list[str] = []

    def fallback(host):
        call_log.append(host)
        return None

    render_fleet_table(_payload(fleet), None, fallback)
    assert call_log == ["v1", "v2"]
```

- [ ] **Step 4.2: Run new tests, verify they FAIL** (each individually)

- [ ] **Step 4.3: Implement `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/fleet_table.py`**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the Tab A fleet table — one row per host, 8 inline columns."""

from __future__ import annotations

from collections.abc import Callable

from dash import html

from tools.odin.valhalla.dashboard.data import HardwareInfo

__all__ = ["render_fleet_table"]


_STATUS_CLASS = {
    "idle": "tab-a-fleet-status-idle",
    "busy": "tab-a-fleet-status-busy",
    "down": "tab-a-fleet-status-down",
}


def render_fleet_table(
    dispatch_payload: dict,
    hardware_payload: dict | None,
    fallback_lookup: Callable[[str], HardwareInfo | None],
) -> html.Div:
    """Build the fleet table.

    Args:
        dispatch_payload: Parsed dispatch.json.
        hardware_payload: Parsed hardware.json (or None for pre-feature dispatches).
        fallback_lookup: ``DataLayer.lookup_hardware`` — called at most once per host.

    Returns:
        A Div containing a table with one header row and one data row per host.
    """
    fleet = dispatch_payload.get("fleet", []) or []
    hw_hosts = (hardware_payload or {}).get("hosts", {}) or {}
    fallback_cache: dict[str, HardwareInfo | None] = {}

    header = html.Tr(
        children=[
            html.Th("Host"),
            html.Th("Hostname"),
            html.Th("Status"),
            html.Th("Current run"),
            html.Th("GPU"),
            html.Th("CPU"),
            html.Th("RAM"),
            html.Th("Last event"),
        ]
    )

    rows: list = []
    for host_entry in fleet:
        host = str(host_entry.get("host", ""))
        status = str(host_entry.get("status", "unknown"))
        current_run_id = host_entry.get("current_run_id")
        last_error = host_entry.get("last_error")

        hw = _resolve_hardware(host, hw_hosts, fallback_lookup, fallback_cache)
        hostname_cell = hw["hostname"] if hw else "—"
        gpu_cell = _gpu_cell(hw)
        cpu_cell = _cpu_cell(hw)
        ram_cell = _ram_cell(hw)

        rows.append(
            html.Tr(
                children=[
                    html.Td(host, className="tab-a-fleet-host"),
                    html.Td(hostname_cell),
                    html.Td(_status_pill(status)),
                    html.Td(_current_run_cell(current_run_id)),
                    html.Td(gpu_cell),
                    html.Td(cpu_cell),
                    html.Td(ram_cell),
                    html.Td(_last_event_cell(last_error)),
                ]
            )
        )

    return html.Div(
        id="tab-a-fleet-table-content",
        children=[
            html.Table(
                children=[
                    html.Thead(children=[header]),
                    html.Tbody(children=rows),
                ],
                className="tab-a-fleet-table",
            )
        ],
    )


def _resolve_hardware(
    host: str,
    hw_hosts: dict,
    fallback_lookup: Callable[[str], HardwareInfo | None],
    cache: dict[str, HardwareInfo | None],
) -> dict | None:
    """Return the hardware block for ``host`` as a plain dict, or None."""
    direct = hw_hosts.get(host)
    if direct:
        return direct
    if host not in cache:
        cache[host] = fallback_lookup(host)
    info = cache[host]
    if info is None:
        return None
    return {
        "hostname": info.hostname,
        "gpu_devices": info.gpu_devices,
        "cpu_name": info.cpu_name,
        "cpu_count": info.cpu_count,
        "ram_gb": info.ram_gb,
    }


def _status_pill(status: str) -> html.Span:
    cls = _STATUS_CLASS.get(status, "tab-a-fleet-status-unknown")
    label = status.capitalize()
    return html.Span(label, className=f"tab-a-pill {cls}")


def _current_run_cell(current_run_id):
    if not current_run_id:
        return "—"
    short = current_run_id.split("_seed")[-1] if "_seed" in current_run_id else current_run_id
    short_text = f"…{current_run_id[-30:]}" if len(current_run_id) > 30 else current_run_id
    return html.A(short_text, href="#", title=current_run_id)


def _gpu_cell(hw: dict | None):
    if not hw or not hw.get("gpu_devices"):
        return "—"
    g = hw["gpu_devices"][0]
    return f"{g.get('name', '?')} · {g.get('mem_gb', 0):.2f} GB"


def _cpu_cell(hw: dict | None):
    if not hw:
        return "—"
    return f"{hw.get('cpu_name', '?')} ×{hw.get('cpu_count', 0)}"


def _ram_cell(hw: dict | None):
    if not hw:
        return "—"
    return f"{hw.get('ram_gb', 0):.2f} GB"


def _last_event_cell(last_error):
    if not last_error:
        return "—"
    if last_error == "gpu_lost: recovered":
        return html.Span("gpu_lost: recovered", className="tab-a-pill tab-a-event-recovered")
    if last_error.startswith("gpu_lost: recovery_failed"):
        return html.Span(
            "gpu_lost: recovery_failed",
            title=last_error,
            className="tab-a-pill tab-a-event-recovery-failed",
        )
    return last_error
```

- [ ] **Step 4.4: Run each test individually, verify all PASS**

- [ ] **Step 4.5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/tabs/dispatch_fleet/fleet_table.py tools/odin/valhalla/dashboard/tests/test_tab_a_fleet_table.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Tab A fleet table: 8 inline columns + hardware lookup

render_fleet_table builds a one-row-per-host table with: Host (IP),
Hostname, Status (pill: idle/busy/down), Current run (link or '—'),
GPU (name + mem), CPU (name + count), RAM, Last event (recovery pill
on gpu_lost: recovered / recovery_failed strings, '—' otherwise).
Hardware lookup order: hardware.json → fallback (data.lookup_hardware
called at most once per host) → '—'.
EOF
)"
```

---

## Task 5: `filters.py` — pure data filter

**Files:**
- Create: `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/filters.py`
- Create: `tools/odin/valhalla/dashboard/tests/test_tab_a_filters.py`

- [ ] **Step 5.1: Write failing tests** — `tools/odin/valhalla/dashboard/tests/test_tab_a_filters.py`

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for tab_a.filters.filter_jobs."""

from __future__ import annotations

from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.filters import filter_jobs


def _job(*, task="Isaac-Ant-Direct-v0", status="completed", kind=None, run_id="r"):
    j = {
        "run_id": run_id,
        "task_id": task,
        "framework": "rsl_rl",
        "backend": "physx",
        "seed": 42,
        "status": status,
        "assigned_to": "v1",
        "attempts": 1,
        "failure": None,
    }
    if status == "failed" and kind is not None:
        j["failure"] = {"kind": kind, "message": "x", "details": {}}
    return j


def test_no_filters_returns_all():
    jobs = [_job(run_id=f"r{i}") for i in range(3)]
    assert filter_jobs(jobs) == jobs


def test_status_filter_single():
    jobs = [
        _job(status="completed", run_id="c"),
        _job(status="failed", kind="hugin_crash", run_id="f"),
    ]
    out = filter_jobs(jobs, status_filter=["failed"])
    assert [j["run_id"] for j in out] == ["f"]


def test_status_filter_multi():
    jobs = [
        _job(status="completed", run_id="c"),
        _job(status="failed", kind="hugin_crash", run_id="f"),
        _job(status="pending", run_id="p"),
    ]
    out = filter_jobs(jobs, status_filter=["completed", "failed"])
    assert sorted(j["run_id"] for j in out) == ["c", "f"]


def test_kind_filter_passes_failed_only():
    jobs = [
        _job(status="completed", run_id="c"),
        _job(status="failed", kind="hugin_crash", run_id="f"),
        _job(status="failed", kind="gpu_lost", run_id="g"),
    ]
    out = filter_jobs(jobs, kind_filter=["hugin_crash"])
    assert [j["run_id"] for j in out] == ["f"]


def test_task_text_substring():
    jobs = [
        _job(task="Isaac-Ant-Direct-v0", run_id="ant"),
        _job(task="Isaac-Velocity-Flat-Anymal-C-Direct-v0", run_id="anymal"),
        _job(task="Isaac-Cartpole-Direct-v0", run_id="cart"),
    ]
    out = filter_jobs(jobs, task_text="ant")
    assert sorted(j["run_id"] for j in out) == ["ant", "anymal"]


def test_task_text_empty_string():
    jobs = [_job(task="Isaac-Ant-Direct-v0", run_id="a"), _job(task="Cartpole", run_id="b")]
    out = filter_jobs(jobs, task_text="")
    assert sorted(j["run_id"] for j in out) == ["a", "b"]


def test_task_text_case_insensitive():
    jobs = [
        _job(task="Isaac-Ant-Direct-v0", run_id="ant"),
        _job(task="Isaac-Velocity-Flat-Anymal-C-Direct-v0", run_id="anymal"),
    ]
    out = filter_jobs(jobs, task_text="ANT")
    assert sorted(j["run_id"] for j in out) == ["ant", "anymal"]


def test_combined_filters_intersect():
    jobs = [
        _job(task="Isaac-Ant-Direct-v0", status="completed", run_id="ac"),
        _job(task="Isaac-Ant-Direct-v0", status="failed", kind="hugin_crash", run_id="af"),
        _job(task="Isaac-Velocity-Flat-Anymal-C-Direct-v0", status="failed", kind="hugin_crash", run_id="amf"),
        _job(task="Isaac-Cartpole-Direct-v0", status="failed", kind="hugin_crash", run_id="cf"),
    ]
    out = filter_jobs(
        jobs,
        status_filter=["failed"],
        kind_filter=["hugin_crash"],
        task_text="ant",
    )
    assert sorted(j["run_id"] for j in out) == ["af", "amf"]
```

- [ ] **Step 5.2: Run new tests, verify they FAIL** (each individually)

- [ ] **Step 5.3: Implement `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/filters.py`**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure data filter for the Tab A jobs table."""

from __future__ import annotations

__all__ = ["filter_jobs"]


def filter_jobs(
    jobs: list[dict],
    *,
    status_filter: list[str] | None = None,
    kind_filter: list[str] | None = None,
    task_text: str = "",
) -> list[dict]:
    """Apply the three filters in sequence.

    - ``status_filter``: empty / None = pass through. Otherwise keep jobs whose
      ``status`` is in the list.
    - ``kind_filter``: empty / None = pass through. Otherwise keep jobs whose
      ``failure.kind`` is in the list (which implicitly excludes non-failed jobs).
    - ``task_text``: empty string = pass through. Otherwise keep jobs whose
      ``task_id`` contains the text (case-insensitive substring).

    All three are AND-combined.
    """
    needle = (task_text or "").lower()
    out: list[dict] = []
    for job in jobs:
        if status_filter and job.get("status") not in status_filter:
            continue
        if kind_filter:
            failure = job.get("failure") or {}
            if failure.get("kind") not in kind_filter:
                continue
        if needle:
            task_id = str(job.get("task_id", "")).lower()
            if needle not in task_id:
                continue
        out.append(job)
    return out
```

- [ ] **Step 5.4: Run each test individually, verify all PASS**

- [ ] **Step 5.5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/tabs/dispatch_fleet/filters.py tools/odin/valhalla/dashboard/tests/test_tab_a_filters.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Tab A filters: filter_jobs pure data function

Three composable filters: status (multi-value list), failure-kind
(multi-value list; failed jobs only), task-text (case-insensitive
substring on task_id). Empty / None filter is pass-through. All
three AND-combine. Used by both the initial render and the on-tick
update_jobs callback.
EOF
)"
```

---

## Task 6: `jobs_table.py` — filter row + base rows (no expand yet)

**Files:**
- Create: `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py`
- Create: `tools/odin/valhalla/dashboard/tests/test_tab_a_jobs_table.py`

This task delivers the filter row, table headers, and per-job data rows. Inline expansion + ssh-tail are added in Task 7.

- [ ] **Step 6.1: Write the first 6 failing tests** — `tools/odin/valhalla/dashboard/tests/test_tab_a_jobs_table.py`

```python
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
    for c in _walk(component):
        if getattr(c, "id", None) == target_id:
            return True
    return False


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
```

- [ ] **Step 6.2: Run new tests, verify they FAIL**

- [ ] **Step 6.3: Implement `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py` (no expand yet)**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the Tab A jobs section: filter row + table.

Inline expansion + ssh-tail rendering land in Task 7.
"""

from __future__ import annotations

from datetime import datetime, timezone

from dash import dcc, html

from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.filters import filter_jobs

__all__ = ["render_jobs_section"]


_STATUS_OPTIONS = [
    {"label": "Pending", "value": "pending"},
    {"label": "Running", "value": "running"},
    {"label": "Completed", "value": "completed"},
    {"label": "Failed", "value": "failed"},
]

_KIND_OPTIONS = [
    {"label": "hugin_crash", "value": "hugin_crash"},
    {"label": "gpu_lost", "value": "gpu_lost"},
    {"label": "preset_unsupported", "value": "preset_unsupported"},
    {"label": "timeout", "value": "timeout"},
    {"label": "infrastructure", "value": "infrastructure"},
    {"label": "hugin_malformed_bundle", "value": "hugin_malformed_bundle"},
]


def render_jobs_section(
    dispatch_payload: dict,
    *,
    status_filter: list[str] | None = None,
    kind_filter: list[str] | None = None,
    task_text: str = "",
) -> html.Div:
    """Build the jobs section: filter row + filtered table.

    Spec 1 Task 6 — no expand row support yet (added in Task 7).
    """
    jobs = dispatch_payload.get("jobs", []) or []
    visible = filter_jobs(jobs, status_filter=status_filter, kind_filter=kind_filter, task_text=task_text)

    filter_row = html.Div(
        className="tab-a-jobs-filter-row",
        children=[
            html.Span("Status", className="tab-a-filter-label"),
            dcc.Dropdown(
                id="tab-a-status-filter",
                options=_STATUS_OPTIONS,
                value=status_filter or [],
                multi=True,
                placeholder="All",
                className="tab-a-filter-dropdown",
            ),
            html.Span("Failure", className="tab-a-filter-label"),
            dcc.Dropdown(
                id="tab-a-kind-filter",
                options=_KIND_OPTIONS,
                value=kind_filter or [],
                multi=True,
                placeholder="All",
                className="tab-a-filter-dropdown",
            ),
            html.Span("Task", className="tab-a-filter-label"),
            dcc.Input(
                id="tab-a-task-text",
                type="text",
                value=task_text,
                placeholder="filter task…",
                debounce=True,
                className="tab-a-filter-input",
            ),
        ],
    )

    header = html.Tr(
        children=[
            html.Th("Task"),
            html.Th("Framework × Backend"),
            html.Th("Seed"),
            html.Th("Status"),
            html.Th("Failure"),
            html.Th("Host"),
            html.Th("Started / Ended"),
        ]
    )
    rows = [_data_row(j) for j in visible]

    table = html.Table(
        className="tab-a-jobs-table",
        children=[
            html.Thead(children=[header]),
            html.Tbody(id="tab-a-jobs-rows", children=rows),
        ],
    )

    return html.Div(
        id="tab-a-jobs-section-content",
        children=[filter_row, table],
    )


def _data_row(job: dict) -> html.Tr:
    status = str(job.get("status", "unknown"))
    failure = job.get("failure") or {}
    kind = failure.get("kind")
    attempts = int(job.get("attempts", 1) or 1)
    host = job.get("assigned_to") or "—"
    started = _relative_time(job.get("started_at"))
    ended = _relative_time(job.get("ended_at"))

    status_children = [
        html.Span(status.capitalize(), className=f"tab-a-pill tab-a-job-status-{status}"),
    ]
    if attempts > 1:
        status_children.append(html.Span(f"×{attempts}", className="tab-a-attempts-badge"))

    failure_cell = (
        html.Span(kind, className=f"tab-a-kind-pill tab-a-kind-pill-{kind}") if kind else "—"
    )

    started_ended_text = (
        f"{started} · {ended}" if ended else (f"{started} · —" if started != "—" else "— · —")
    )

    return html.Tr(
        children=[
            html.Td(job.get("task_id", "")),
            html.Td(f"{job.get('framework', '')} × {job.get('backend', '')}", className="tab-a-mono"),
            html.Td(str(job.get("seed", ""))),
            html.Td(status_children),
            html.Td(failure_cell),
            html.Td(host, className="tab-a-mono"),
            html.Td(started_ended_text, className="tab-a-muted"),
        ]
    )


def _relative_time(ts: str | None) -> str:
    """Return a human-readable relative time, e.g. ``'3m ago'``. ``None`` → ``'—'``."""
    if not ts:
        return "—"
    try:
        # Strip a trailing Z for fromisoformat compatibility on Python 3.10.
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return ts
    now = datetime.now(timezone.utc)
    delta = now - dt
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"
```

- [ ] **Step 6.4: Run each test individually, verify all PASS**

- [ ] **Step 6.5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py tools/odin/valhalla/dashboard/tests/test_tab_a_jobs_table.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Tab A jobs table: filter row + base rows

render_jobs_section builds the filter row (status + kind dropdowns,
task-text input) and a 7-column table: Task, Framework × Backend,
Seed, Status (pill), Failure (kind pill or '—'), Host, Started/Ended
(relative time). Failed rows show a kind pill in the Failure column.
attempts > 1 adds an '×N' badge after the status pill. Inline
expansion + ssh-tail render are added in T7.
EOF
)"
```

---

## Task 7: `jobs_table.py` — inline expand row + ssh-tail render

**Files:**
- Modify: `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py`
- Modify: `tools/odin/valhalla/dashboard/tests/test_tab_a_jobs_table.py`

- [ ] **Step 7.1: Append failing tests** to `tools/odin/valhalla/dashboard/tests/test_tab_a_jobs_table.py`

```python
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
        if isinstance(getattr(c, "id", None), dict)
        and getattr(c, "id", {}).get("type") == "tab-a-ssh-tail-button"
    ]
    assert button_ids == [{"type": "tab-a-ssh-tail-button", "run_id": "rid-3"}]


def test_jobs_expand_toggle_button_on_failed_rows():
    job = _job(status="failed", kind="hugin_crash", run_id="rid-4")
    component = render_jobs_section(_payload([job]))
    button_ids = [
        getattr(c, "id", None)
        for c in _walk(component)
        if isinstance(getattr(c, "id", None), dict)
        and getattr(c, "id", {}).get("type") == "tab-a-expand-toggle"
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
    assert len(pre_blocks) == 1
    pre_text = getattr(pre_blocks[0], "children", "")
    if isinstance(pre_text, list):
        pre_text = "".join(t for t in pre_text if isinstance(t, str))
    assert "line one" in pre_text
    assert "line two" in pre_text


def test_jobs_expanded_row_ssh_tail_lines_empty_renders_not_found():
    job = _job(status="failed", kind="hugin_crash", run_id="rid-6")
    component = render_jobs_section(
        _payload([job]),
        expanded_run_ids={"rid-6"},
        ssh_tail_store={"rid-6": []},
    )
    blob = " ".join(
        getattr(c, "children", "")
        for c in _walk(component)
        if isinstance(getattr(c, "children", None), str)
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
        getattr(c, "children", "")
        for c in _walk(component)
        if isinstance(getattr(c, "children", None), str)
    )
    assert "(no failure message recorded)" in blob


def test_jobs_empty_state_when_filters_match_nothing():
    jobs = [_job(status="completed", run_id="c")]
    component = render_jobs_section(_payload(jobs), status_filter=["failed"])
    assert _has_id(component, "tab-a-jobs-empty")
    blob = " ".join(
        getattr(c, "children", "")
        for c in _walk(component)
        if isinstance(getattr(c, "children", None), str)
    )
    assert "No jobs match" in blob
    # Has a Clear button.
    assert _has_id(component, "tab-a-clear-filters")


def test_jobs_empty_state_when_dispatch_has_no_jobs():
    component = render_jobs_section(_payload([]))
    assert _has_id(component, "tab-a-jobs-empty-zero")
    blob = " ".join(
        getattr(c, "children", "")
        for c in _walk(component)
        if isinstance(getattr(c, "children", None), str)
    )
    assert "No jobs queued" in blob
    # No clear button when there are zero jobs at all.
    assert not _has_id(component, "tab-a-clear-filters")
```

- [ ] **Step 7.2: Run new tests, verify they FAIL**

- [ ] **Step 7.3: Modify `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py`** to add expand support

Replace the existing `render_jobs_section` signature + body with:

```python
def render_jobs_section(
    dispatch_payload: dict,
    *,
    status_filter: list[str] | None = None,
    kind_filter: list[str] | None = None,
    task_text: str = "",
    expanded_run_ids: set[str] | None = None,
    ssh_tail_store: dict[str, list[str]] | None = None,
) -> html.Div:
    """Build the jobs section: filter row + table + inline expand rows.

    expanded_run_ids: which failed-row expansions are currently open.
    ssh_tail_store: keyed by run_id; values are the lines from ssh-tail.log
        (loaded on demand via the tab's load_ssh_tail callback).
    """
    jobs = dispatch_payload.get("jobs", []) or []
    expanded_run_ids = expanded_run_ids or set()
    ssh_tail_store = ssh_tail_store or {}

    if not jobs:
        return html.Div(
            id="tab-a-jobs-section-content",
            children=[
                _filter_row(status_filter, kind_filter, task_text),
                html.Div(
                    id="tab-a-jobs-empty-zero",
                    className="tab-a-empty-state",
                    children=[html.P("No jobs queued for this dispatch yet.")],
                ),
            ],
        )

    visible = filter_jobs(jobs, status_filter=status_filter, kind_filter=kind_filter, task_text=task_text)

    if not visible:
        return html.Div(
            id="tab-a-jobs-section-content",
            children=[
                _filter_row(status_filter, kind_filter, task_text),
                html.Div(
                    id="tab-a-jobs-empty",
                    className="tab-a-empty-state",
                    children=[
                        html.P("No jobs match the current filters."),
                        html.Button(
                            "Clear",
                            id="tab-a-clear-filters",
                            n_clicks=0,
                            className="tab-a-clear-button",
                        ),
                    ],
                ),
            ],
        )

    header = html.Tr(
        children=[
            html.Th("Task"),
            html.Th("Framework × Backend"),
            html.Th("Seed"),
            html.Th("Status"),
            html.Th("Failure"),
            html.Th("Host"),
            html.Th("Started / Ended"),
        ]
    )

    body_rows: list = []
    for j in visible:
        body_rows.append(_data_row(j))
        if j.get("status") == "failed" and j.get("run_id") in expanded_run_ids:
            body_rows.append(_expand_row(j, ssh_tail_store.get(j.get("run_id"))))

    table = html.Table(
        className="tab-a-jobs-table",
        children=[
            html.Thead(children=[header]),
            html.Tbody(id="tab-a-jobs-rows", children=body_rows),
        ],
    )

    return html.Div(
        id="tab-a-jobs-section-content",
        children=[_filter_row(status_filter, kind_filter, task_text), table],
    )


def _filter_row(status_filter, kind_filter, task_text):
    return html.Div(
        className="tab-a-jobs-filter-row",
        children=[
            html.Span("Status", className="tab-a-filter-label"),
            dcc.Dropdown(
                id="tab-a-status-filter",
                options=_STATUS_OPTIONS,
                value=status_filter or [],
                multi=True,
                placeholder="All",
                className="tab-a-filter-dropdown",
            ),
            html.Span("Failure", className="tab-a-filter-label"),
            dcc.Dropdown(
                id="tab-a-kind-filter",
                options=_KIND_OPTIONS,
                value=kind_filter or [],
                multi=True,
                placeholder="All",
                className="tab-a-filter-dropdown",
            ),
            html.Span("Task", className="tab-a-filter-label"),
            dcc.Input(
                id="tab-a-task-text",
                type="text",
                value=task_text,
                placeholder="filter task…",
                debounce=True,
                className="tab-a-filter-input",
            ),
        ],
    )
```

Update `_data_row` to add the expand-toggle button on failed rows. Replace the `failure_cell` definition with:

```python
    failure_cell_children: list = []
    if kind:
        failure_cell_children.append(
            html.Span(kind, className=f"tab-a-kind-pill tab-a-kind-pill-{kind}")
        )
        failure_cell_children.append(
            html.Button(
                "▸",
                id={"type": "tab-a-expand-toggle", "run_id": job.get("run_id", "")},
                n_clicks=0,
                className="tab-a-expand-toggle",
                title="Show / hide failure details",
            )
        )
        failure_cell = failure_cell_children
    else:
        failure_cell = "—"
```

Add the `_expand_row` helper at the end of the file:

```python
def _expand_row(job: dict, ssh_tail_lines: list[str] | None) -> html.Tr:
    """Inline expansion row for a failed job: kind, attempts, message, ssh-tail."""
    failure = job.get("failure") or {}
    kind = failure.get("kind", "unknown")
    message = failure.get("message")
    attempts = int(job.get("attempts", 1) or 1)
    run_id = job.get("run_id", "")

    body: list = [
        html.Span("Kind ", className="tab-a-expand-label"),
        html.Span(kind, className=f"tab-a-kind-pill tab-a-kind-pill-{kind}"),
        html.Span(f"  Attempts {attempts}", className="tab-a-expand-label"),
        html.Br(),
        html.Br(),
        html.Span("Message", className="tab-a-expand-label"),
        html.Br(),
        html.Pre(
            message if message else "(no failure message recorded)",
            className="tab-a-failure-message",
        ),
        html.Button(
            "▸ Show ssh-tail.log (last 50 lines)",
            id={"type": "tab-a-ssh-tail-button", "run_id": run_id},
            n_clicks=0,
            className="tab-a-ssh-tail-button",
        ),
    ]

    if ssh_tail_lines is not None:
        if ssh_tail_lines:
            body.append(html.Pre("\n".join(ssh_tail_lines), className="tab-a-ssh-tail-pre"))
        else:
            body.append(
                html.P(
                    f"ssh-tail.log not found at "
                    f"{run_id}/logs/ssh-tail.log (or unreadable)",
                    className="tab-a-ssh-tail-empty",
                )
            )

    return html.Tr(
        className="tab-a-expand-row",
        children=[html.Td(colSpan=7, children=body)],
    )
```

- [ ] **Step 7.4: Run each test individually, verify all PASS**

- [ ] **Step 7.5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py tools/odin/valhalla/dashboard/tests/test_tab_a_jobs_table.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Tab A jobs table: inline expand row + ssh-tail render

render_jobs_section now accepts expanded_run_ids (set) and
ssh_tail_store (dict). Failed rows in the expanded set get an inline
<tr class='tab-a-expand-row'> beneath them showing kind pill,
attempts, full failure.message in a <pre> block, and a 'Show
ssh-tail.log' button. When ssh_tail_store has lines for that run_id,
they're rendered as a <pre>; empty list shows the not-found message.
Empty states: 'No jobs match' (with Clear button) when filters match
nothing; 'No jobs queued' when the dispatch has zero jobs at all.
EOF
)"
```

---

## Task 8: `ssh_tail.py` — file loader with truncation

**Files:**
- Create: `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/ssh_tail.py`
- Create: `tools/odin/valhalla/dashboard/tests/test_tab_a_ssh_tail.py`

- [ ] **Step 8.1: Write failing tests** — `tools/odin/valhalla/dashboard/tests/test_tab_a_ssh_tail.py`

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for tab_a.ssh_tail.load_ssh_tail."""

from __future__ import annotations

from pathlib import Path

from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.ssh_tail import (
    SSH_TAIL_DEFAULT_LINES,
    SSH_TAIL_MAX_BYTES,
    load_ssh_tail,
)


def _write_log(runs_root: Path, dispatch_id: str, run_id: str, content: str) -> Path:
    log_dir = runs_root / dispatch_id / run_id / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "ssh-tail.log"
    log_path.write_text(content)
    return log_path


def test_load_ssh_tail_full_file_under_threshold(tmp_path):
    _write_log(tmp_path, "d", "r", "\n".join(f"line {i}" for i in range(10)) + "\n")
    lines = load_ssh_tail(tmp_path, "d", "r", lines=SSH_TAIL_DEFAULT_LINES)
    assert lines == [f"line {i}" for i in range(10)]


def test_load_ssh_tail_returns_last_n_lines(tmp_path):
    _write_log(tmp_path, "d", "r", "\n".join(f"line {i}" for i in range(100)) + "\n")
    lines = load_ssh_tail(tmp_path, "d", "r", lines=10)
    assert lines == [f"line {i}" for i in range(90, 100)]


def test_load_ssh_tail_returns_empty_when_file_missing(tmp_path):
    assert load_ssh_tail(tmp_path, "d", "r-missing") == []


def test_load_ssh_tail_truncates_huge_file(tmp_path):
    # Write 200 KB so we exceed the 64 KB cap.
    chunk = "abcdefghij" * 100  # 1000 bytes per line
    payload = "\n".join([f"{i:05d} {chunk}" for i in range(200)]) + "\n"
    _write_log(tmp_path, "d", "r", payload)
    lines = load_ssh_tail(tmp_path, "d", "r", lines=20)
    assert len(lines) == 20
    assert lines[0].startswith("…")
    assert "truncated" in lines[0].lower()


def test_load_ssh_tail_handles_partial_first_line_when_seeking(tmp_path):
    # Construct a file where the truncation point is mid-line; the partial first line should be dropped.
    chunk = "x" * 70_000  # bigger than SSH_TAIL_MAX_BYTES
    payload = chunk + "\nfinal-line\n"
    _write_log(tmp_path, "d", "r", payload)
    lines = load_ssh_tail(tmp_path, "d", "r", lines=5)
    # The truncation marker is line[0]; the partial line is dropped; "final-line" is the only real line.
    assert lines[0].startswith("…")
    assert any("final-line" in s for s in lines)


def test_load_ssh_tail_returns_empty_on_permission_error(tmp_path, monkeypatch):
    log_path = _write_log(tmp_path, "d", "r", "hello\n")

    real_open = open

    def _raise(path, *args, **kwargs):
        if str(path) == str(log_path):
            raise PermissionError("denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", _raise)
    assert load_ssh_tail(tmp_path, "d", "r") == []


def test_load_ssh_tail_max_bytes_constant_is_64kb():
    assert SSH_TAIL_MAX_BYTES == 64 * 1024
```

- [ ] **Step 8.2: Run new tests, verify they FAIL**

- [ ] **Step 8.3: Implement `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/ssh_tail.py`**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read the last N lines of a bundle's ssh-tail.log.

Truncates at SSH_TAIL_MAX_BYTES (64 KB) — never reads more into memory.
Failed jobs' logs are typically a few KB so the threshold rarely fires; it
exists to bound memory in pathological cases.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["SSH_TAIL_DEFAULT_LINES", "SSH_TAIL_MAX_BYTES", "load_ssh_tail"]


SSH_TAIL_DEFAULT_LINES = 50
SSH_TAIL_MAX_BYTES = 64 * 1024
_TRUNCATION_MARKER = "… (truncated to last 64 KB) …"


def load_ssh_tail(
    runs_root: Path,
    dispatch_id: str,
    run_id: str,
    lines: int = SSH_TAIL_DEFAULT_LINES,
) -> list[str]:
    """Return the last ``lines`` lines of the bundle's ssh-tail.log.

    Returns an empty list if the file is missing, unreadable, or the read
    raises any ``OSError`` (e.g., PermissionError). When the file exceeds
    ``SSH_TAIL_MAX_BYTES``, only the last 64 KB are read; the first
    (potentially partial) line is dropped and a truncation marker is
    prepended to the returned list.
    """
    path = Path(runs_root) / dispatch_id / run_id / "logs" / "ssh-tail.log"
    if not path.exists():
        return []
    try:
        size = path.stat().st_size
        if size <= SSH_TAIL_MAX_BYTES:
            with open(path) as fh:
                all_lines = fh.read().splitlines()
            return all_lines[-lines:]
        # Large file: seek to the last SSH_TAIL_MAX_BYTES.
        with open(path, "rb") as fh:
            fh.seek(-SSH_TAIL_MAX_BYTES, os.SEEK_END)
            tail_bytes = fh.read()
        text = tail_bytes.decode("utf-8", errors="ignore")
        # Drop the (probably partial) first line.
        all_lines = text.split("\n")[1:]
        # Drop a trailing empty entry if present (file ends with \n).
        if all_lines and all_lines[-1] == "":
            all_lines = all_lines[:-1]
        return [_TRUNCATION_MARKER, *all_lines[-(lines - 1):]] if lines > 1 else [_TRUNCATION_MARKER]
    except OSError:
        return []
```

- [ ] **Step 8.4: Run each test individually, verify all PASS**

- [ ] **Step 8.5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/tabs/dispatch_fleet/ssh_tail.py tools/odin/valhalla/dashboard/tests/test_tab_a_ssh_tail.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Tab A ssh_tail: load last N lines with 64 KB safety cap

load_ssh_tail(runs_root, dispatch_id, run_id, lines=50) reads
<runs_root>/<dispatch_id>/<run_id>/logs/ssh-tail.log. Files <= 64 KB
read whole-file; larger files seek to last 64 KB, drop the partial
first line, prepend a truncation marker. Missing file or any OSError
(permission denied, etc.) → []. Default 50 lines; 64 KB cap is a
hard memory bound.
EOF
)"
```

---

## Task 9: `callbacks.py` — header / fleet / jobs / failure-pill callbacks

**Files:**
- Create: `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/callbacks.py`
- Create: `tools/odin/valhalla/dashboard/tests/test_tab_a_callbacks.py`

This task wires four of the six callbacks. Tasks 4 and 5 (expand-toggle, ssh-tail-load, both pattern-matching) land in T10.

- [ ] **Step 9.1: Write the first 5 failing tests** — `tools/odin/valhalla/dashboard/tests/test_tab_a_callbacks.py`

```python
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
    # Out is a Div(id="tab-a-jobs-section-content"); inside it, the table contains exactly one data row.
    from tools.odin.valhalla.dashboard.tests.test_tab_a_jobs_table import _walk

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
    from tools.odin.valhalla.dashboard.tests.test_tab_a_jobs_table import _walk

    rows = [c for c in _walk(out) if type(c).__name__ == "Tr"]
    assert len(rows) == 2  # header + 1 gpu_lost row


def test_failure_pill_click_writes_store_and_dropdown():
    store_value, dropdown_value = _handle_pill_click("gpu_lost")
    assert store_value == "gpu_lost"
    assert dropdown_value == ["gpu_lost"]
```

- [ ] **Step 9.2: Run new tests, verify they FAIL**

- [ ] **Step 9.3: Implement `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/callbacks.py`** (4 callbacks)

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wire Tab A's dcc.Interval and pattern-matching callbacks."""

from __future__ import annotations

import sys
from typing import Any

import dash
from dash import ALL, Input, Output, State

from tools.odin.valhalla.dashboard.data import DataLayer
from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.fleet_table import render_fleet_table
from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.header import render_header
from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.jobs_table import render_jobs_section

__all__ = ["register_callbacks"]


def register_callbacks(app: dash.Dash, data: DataLayer) -> None:
    """Register the Tab A callbacks against the layout's slot ids."""

    @app.callback(
        Output("tab-a-header", "children"),
        Input("tab-a-tick", "n_intervals"),
        Input("tab-a-dispatch-id", "data"),
    )
    def _update_header(_n, dispatch_id):
        if not dispatch_id:
            return dash.no_update
        try:
            return _compute_header_children(data, dispatch_id)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] tab-a header callback: {type(exc).__name__}: {exc}", file=sys.stderr)
            return _error_banner("Failed to load dispatch.json", exc)

    @app.callback(
        Output("tab-a-fleet-table", "children"),
        Input("tab-a-tick", "n_intervals"),
        Input("tab-a-dispatch-id", "data"),
    )
    def _update_fleet(_n, dispatch_id):
        if not dispatch_id:
            return dash.no_update
        try:
            return _compute_fleet_children(data, dispatch_id)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] tab-a fleet callback: {type(exc).__name__}: {exc}", file=sys.stderr)
            return _error_banner("Failed to render fleet table", exc)

    @app.callback(
        Output("tab-a-jobs-section", "children"),
        Input("tab-a-tick", "n_intervals"),
        Input("tab-a-dispatch-id", "data"),
        Input("tab-a-status-filter", "value"),
        Input("tab-a-kind-filter", "value"),
        Input("tab-a-task-text", "value"),
        Input("tab-a-failure-filter", "data"),
        Input("tab-a-expanded-run-ids", "data"),
        Input("tab-a-ssh-tail-store", "data"),
    )
    def _update_jobs(_n, dispatch_id, status_filter, kind_filter, task_text,
                     failure_filter, expanded_run_ids, ssh_tail_store):
        if not dispatch_id:
            return dash.no_update
        try:
            return _compute_jobs_children(
                data,
                dispatch_id=dispatch_id,
                status_filter=status_filter,
                kind_filter=kind_filter,
                task_text=task_text or "",
                failure_filter=failure_filter,
                expanded_run_ids=expanded_run_ids or [],
                ssh_tail_store=ssh_tail_store or {},
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] tab-a jobs callback: {type(exc).__name__}: {exc}", file=sys.stderr)
            return _error_banner("Failed to render jobs section", exc)

    @app.callback(
        Output("tab-a-failure-filter", "data"),
        Output("tab-a-kind-filter", "value"),
        Input({"type": "tab-a-failure-pill", "kind": ALL}, "n_clicks"),
        State({"type": "tab-a-failure-pill", "kind": ALL}, "id"),
    )
    def _on_failure_pill(n_clicks_list, ids_list):
        if not n_clicks_list or not any(n_clicks_list):
            return dash.no_update, dash.no_update
        # Pick the most-recently clicked pill: any with n_clicks > 0; the
        # last one in the list wins under Dash's pattern-matching event order.
        for n, ident in zip(reversed(n_clicks_list), reversed(ids_list)):
            if n and n > 0:
                return _handle_pill_click(ident["kind"])
        return dash.no_update, dash.no_update


# -- pure helpers (testable without the Dash callback graph) ----------------------


def _compute_header_children(data: DataLayer, dispatch_id: str):
    payload = data.load_dispatch(dispatch_id)
    return render_header(payload)


def _compute_fleet_children(data: DataLayer, dispatch_id: str):
    payload = data.load_dispatch(dispatch_id)
    hardware = data.load_hardware(dispatch_id)
    return render_fleet_table(payload, hardware, data.lookup_hardware)


def _compute_jobs_children(
    data: DataLayer,
    *,
    dispatch_id: str,
    status_filter: list[str] | None,
    kind_filter: list[str] | None,
    task_text: str,
    failure_filter: str | None,
    expanded_run_ids: list[str],
    ssh_tail_store: dict[str, list[str]],
):
    payload = data.load_dispatch(dispatch_id)
    effective_kind = list(kind_filter or [])
    if failure_filter and failure_filter not in effective_kind:
        effective_kind.append(failure_filter)
    return render_jobs_section(
        payload,
        status_filter=status_filter or None,
        kind_filter=effective_kind or None,
        task_text=task_text or "",
        expanded_run_ids=set(expanded_run_ids or []),
        ssh_tail_store=ssh_tail_store or {},
    )


def _handle_pill_click(kind: str) -> tuple[str, list[str]]:
    """Return (failure-filter store value, kind-dropdown value) for a pill click."""
    return kind, [kind]


def _error_banner(message: str, exc: Exception):
    from dash import html

    return html.Div(
        className="tab-a-error-banner",
        children=[html.Strong(message), f": {type(exc).__name__}: {exc}"],
    )
```

- [ ] **Step 9.4: Run each test individually, verify all PASS**

- [ ] **Step 9.5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/tabs/dispatch_fleet/callbacks.py tools/odin/valhalla/dashboard/tests/test_tab_a_callbacks.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Tab A callbacks (1/2): header / fleet / jobs / failure-pill

register_callbacks wires the dcc.Interval-driven update_header,
update_fleet, update_jobs callbacks plus the pattern-matching
failure-pill click handler. Each callback's body is a thin wrapper
around a pure helper (_compute_*_children, _handle_pill_click) so
tests can call them directly without the Dash callback graph.
Errors are caught at the boundary, logged to stderr, and rendered
as inline error banners. Pattern-matching for expand-toggle and
ssh-tail-load lands in the next commit.
EOF
)"
```

---

## Task 10: `callbacks.py` — expand-toggle + ssh-tail-load (pattern-matching)

**Files:**
- Modify: `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/callbacks.py`
- Modify: `tools/odin/valhalla/dashboard/tests/test_tab_a_callbacks.py`

- [ ] **Step 10.1: Append failing tests** to `tools/odin/valhalla/dashboard/tests/test_tab_a_callbacks.py`

```python
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


def test_load_ssh_tail_callback_writes_store(tmp_path, monkeypatch):
    from tools.odin.valhalla.dashboard.tabs.dispatch_fleet import callbacks as cb_mod

    log_dir = tmp_path / "d" / "Y" / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "ssh-tail.log").write_text("a\nb\n")

    class _Data:
        _runs_root = tmp_path

    out = cb_mod._compute_ssh_tail_store(_Data(), "d", "Y", current_store={})
    assert out == {"Y": ["a", "b"]}


def test_load_ssh_tail_callback_ignores_phantom_click(tmp_path):
    """n_clicks=0 list (no clicks yet) returns dash.no_update."""
    from tools.odin.valhalla.dashboard.tabs.dispatch_fleet import callbacks as cb_mod
    import dash

    out = cb_mod._on_ssh_tail_handler([], [], data=None, current_store={})
    assert out is dash.no_update
```

- [ ] **Step 10.2: Run new tests, verify they FAIL**

- [ ] **Step 10.3: Modify `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/callbacks.py`** to add the two pattern-matching callbacks

Add these inside `register_callbacks` (after the existing `_on_failure_pill` callback):

```python
    @app.callback(
        Output("tab-a-expanded-run-ids", "data"),
        Input({"type": "tab-a-expand-toggle", "run_id": ALL}, "n_clicks"),
        State({"type": "tab-a-expand-toggle", "run_id": ALL}, "id"),
        State("tab-a-expanded-run-ids", "data"),
    )
    def _on_expand_toggle(n_clicks_list, ids_list, current):
        return _on_expand_toggle_handler(n_clicks_list, ids_list, current=current)

    @app.callback(
        Output("tab-a-ssh-tail-store", "data"),
        Input({"type": "tab-a-ssh-tail-button", "run_id": ALL}, "n_clicks"),
        State({"type": "tab-a-ssh-tail-button", "run_id": ALL}, "id"),
        State("tab-a-dispatch-id", "data"),
        State("tab-a-ssh-tail-store", "data"),
    )
    def _on_ssh_tail(n_clicks_list, ids_list, dispatch_id, store):
        return _on_ssh_tail_handler(n_clicks_list, ids_list, data=dispatch_id, current_store=store, runs_root=data._runs_root)
```

(Note the trailing argument on `_on_ssh_tail_handler` — we extract `runs_root` from the `DataLayer` via the closed-over `data`.)

Add the helper functions at module-level:

```python
def _on_expand_toggle_handler(n_clicks_list, ids_list, *, current):
    if not n_clicks_list or not any(n_clicks_list):
        return dash.no_update
    # Find the latest non-zero click; toggle that run_id.
    for n, ident in zip(reversed(n_clicks_list), reversed(ids_list)):
        if n and n > 0:
            return _toggle_run_id(current or [], ident["run_id"])
    return dash.no_update


def _toggle_run_id(current: list[str], run_id: str) -> list[str]:
    """Add ``run_id`` to ``current`` (a list) if absent; remove if present."""
    s = set(current)
    if run_id in s:
        s.remove(run_id)
    else:
        s.add(run_id)
    return sorted(s)


def _on_ssh_tail_handler(n_clicks_list, ids_list, *, data, current_store, runs_root=None):
    if not n_clicks_list or not any(n_clicks_list):
        return dash.no_update
    for n, ident in zip(reversed(n_clicks_list), reversed(ids_list)):
        if n and n > 0:
            run_id = ident["run_id"]
            # Read fresh; current_store may be partial.
            from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.ssh_tail import load_ssh_tail

            new_store = dict(current_store or {})
            if runs_root is None:
                # Test-mode shortcut — caller provided runs_root via _compute_ssh_tail_store helper.
                return new_store
            new_store[run_id] = load_ssh_tail(runs_root, data, run_id)
            return new_store
    return dash.no_update


def _compute_ssh_tail_store(data, dispatch_id: str, run_id: str, *, current_store: dict):
    """Test-friendly helper: load the tail and return the new store dict."""
    from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.ssh_tail import load_ssh_tail

    new_store = dict(current_store or {})
    new_store[run_id] = load_ssh_tail(data._runs_root, dispatch_id, run_id)
    return new_store
```

- [ ] **Step 10.4: Run each test individually, verify all PASS**

- [ ] **Step 10.5: Run the full Tab A test suite** to confirm everything still green

```
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/ --tb=line --noconftest -p no:cacheprovider
```

Expected: all dashboard tests pass (Spec 0's 49 + Spec 1's ~50 = ~100 total).

- [ ] **Step 10.6: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/tabs/dispatch_fleet/callbacks.py tools/odin/valhalla/dashboard/tests/test_tab_a_callbacks.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Tab A callbacks (2/2): expand-toggle + ssh-tail load

Two pattern-matching callbacks: _on_expand_toggle adds/removes a
run_id from the expanded-run-ids store; _on_ssh_tail loads the
ssh-tail.log for the clicked button and writes the lines into the
keyed ssh-tail store. Both gate on n_clicks > 0 to skip Dash's
startup phantom fires. Helpers (_toggle_run_id, _on_*_handler,
_compute_ssh_tail_store) are pure functions tested directly.
EOF
)"
```

---

## Task 11: Architecture-doc change-log entry + manual smoke test

**Files:**
- Modify: `docs/odin/architecture.md`

- [ ] **Step 11.1: Manual smoke test against real dispatches**

Launch the dashboard against the live and historical dispatches and verify Tab A renders correctly. This is a manual visual check — not a pytest test.

```bash
PYTHONPATH=. python3 tools/odin/valhalla/dashboard/cli.py --runs-root odin_runs --no-browser
```

Open `http://127.0.0.1:8050/`, click on `20260428-133931` (live, 153-job), confirm:
- Header shows live pill, totals match `dispatch.json` totals.
- Failure pills row hidden (no failures yet) OR shows the right kinds + counts.
- Fleet table shows two hosts with L40 GPUs, status pills correct.
- Jobs table renders all 153 rows; status filter works.
- Wait 5 s; pill counts update if any job state changed.

Click on `20260424-160119` (failure dispatch), confirm:
- Header shows done pill.
- Failure pills row shows `hugin_crash: 6` (or whatever the actual breakdown is post-preset-handling fix).
- Click `hugin_crash: 6` pill → kind-dropdown reflects `["hugin_crash"]`, jobs table filters to those rows.
- Click on a failed row's `▸` → expand row appears with kind/attempts/message.
- Click `▸ Show ssh-tail.log` → `<pre>` block appears with last 50 lines.

If anything looks broken, fix in a small follow-up commit before T11.2.

- [ ] **Step 11.2: Add change-log entry** to `docs/odin/architecture.md`

Update the "Last updated" line at the top of the doc:

```diff
-**Last updated:** 2026-04-27 (Odin dashboard skeleton)
+**Last updated:** 2026-04-27 (Odin dashboard Tab A)
```

Add a new row at the END of the change-log table (after the dashboard skeleton row):

```markdown
| 2026-04-27 | Odin dashboard Tab A — Dispatch & Fleet (Spec 1 of 4) landed (`docs/superpowers/specs/2026-04-27-odin-dashboard-tab-a-dispatch-fleet-design.md`). New sub-package `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/` (7 files: layout, header, fleet_table, jobs_table, filters, ssh_tail, callbacks). The tab renders a per-dispatch health view: header strip (live/done pill, totals row, click-to-filter per-kind failure pills), fleet table (one row per host, 8 inline columns: Host / Hostname / Status pill / Current run / GPU / CPU / RAM / Last event), and a jobs section (filter row with status + failure-kind dropdowns + free-text task search; 7-column table with pill-coded status and failure-kind cells; failed rows expand inline to show the full failure.message and a button that loads the last 50 lines of ssh-tail.log). Auto-polls dispatch.json every 5s via dcc.Interval; callbacks no-op once the dispatch is done. The Spec 0 registry was extended (~10 lines in app.py) to also call tab_module.register(app, data) at app startup so the tab can wire its dcc.Interval and pattern-matching callbacks. ~50 pure-Python tests across 6 files under dashboard/tests/test_tab_a_*.py. No browser-based tests; visual layout source-of-truth lives in `.superpowers/brainstorm/2312694-1777385893/content/` (fleet-row-vs-card.html, jobs-table.html). | Odin dashboard Tab A |
```

- [ ] **Step 11.3: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add docs/odin/architecture.md
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Architecture doc: change-log entry for dashboard Tab A

Records Spec 1 landing (tools/odin/valhalla/dashboard/tabs/
dispatch_fleet/), the 7-file module split, and the Spec 0 registry
enhancement that lets tab modules wire callbacks at app startup.
EOF
)"
```

---

## Self-review

**Spec coverage:**

| Spec section | Plan task |
|---|---|
| § Architecture / module layout (7 files) | T2 (skeleton) + T3-T10 (one file per task) |
| § Spec 0 registry enhancement | T1 |
| § Components → header.py | T3 |
| § Components → fleet_table.py | T4 |
| § Components → jobs_table.py (rendering only) | T6 |
| § Components → jobs_table.py (expand row + ssh-tail render) | T7 |
| § Components → filters.py | T5 |
| § Components → ssh_tail.py | T8 |
| § Components → callbacks.py (header / fleet / jobs / failure-pill) | T9 |
| § Components → callbacks.py (expand-toggle / ssh-tail-load) | T10 |
| § Data flow Flow 1 (cold mount) | T9 (callbacks fire on mount) |
| § Data flow Flow 2 (live tick) | T9 (Interval-driven) |
| § Data flow Flow 3 (filter change) | T9 (filter Inputs) |
| § Data flow Flow 4 (failure pill click) | T9 (_on_failure_pill) |
| § Data flow Flow 5 (expand row) | T10 (_on_expand_toggle) |
| § Data flow Flow 6 (ssh-tail click) | T10 (_on_ssh_tail) |
| § Data flow Flow 7 (header dropdown switches dispatches) | Already handled by Spec 0 routing |
| § Error handling matrix | T9 / T10 (try/except in each callback wrapper; banner helper) |
| § Testing strategy → test_tab_a_header.py (8 cases) | T3 |
| § Testing strategy → test_tab_a_fleet_table.py (11 cases) | T4 |
| § Testing strategy → test_tab_a_jobs_table.py (16 cases) | T6 (6) + T7 (10) |
| § Testing strategy → test_tab_a_filters.py (8 cases) | T5 |
| § Testing strategy → test_tab_a_ssh_tail.py (6 cases) | T8 |
| § Testing strategy → test_tab_a_callbacks.py (9 cases) | T9 (5) + T10 (4) |
| § Implementation order preview | Tasks 1-11 (close one-to-one match with the spec's preview, with the manual smoke test folded into T11) |

**Placeholder scan:** searched for "TBD", "TODO", "fill in", "<...>" — none. Every code step has concrete code; every test step has actual assertions.

**Type / signature consistency:**
- `render_header(dispatch_payload: dict) -> html.Div` — consistent across spec, plan, and tests.
- `render_fleet_table(dispatch_payload, hardware_payload, fallback_lookup)` — three positional args; `fallback_lookup: Callable[[str], HardwareInfo | None]` matches the `DataLayer.lookup_hardware` signature.
- `filter_jobs(jobs, *, status_filter, kind_filter, task_text)` — keyword-only args; default values consistent across uses in callbacks + tests.
- `render_jobs_section(payload, *, status_filter, kind_filter, task_text, expanded_run_ids, ssh_tail_store)` — six keyword-only args; `expanded_run_ids: set[str]` and `ssh_tail_store: dict[str, list[str]]`.
- `load_ssh_tail(runs_root, dispatch_id, run_id, lines=50)` — four args; `lines` default = `SSH_TAIL_DEFAULT_LINES = 50`.
- `_compute_*` helpers in callbacks.py — all take `data: DataLayer` (or stub) as first arg.
- `_handle_pill_click(kind) -> (str, list[str])` — returns `(store_value, dropdown_value)` tuple.
- `_toggle_run_id(current: list[str], run_id: str) -> list[str]` — list in, list out (sorted) so Dash Stores serialize cleanly.
- Pattern-matching ids: `{"type": "tab-a-failure-pill", "kind": "<kind>"}`, `{"type": "tab-a-expand-toggle", "run_id": "<rid>"}`, `{"type": "tab-a-ssh-tail-button", "run_id": "<rid>"}`. All three shapes appear consistently in spec / plan / tests.

Plan is internally consistent and spec-complete.

---

## Execution

**Plan complete and saved to `docs/superpowers/plans/2026-04-27-odin-dashboard-tab-a-dispatch-fleet.md`.** Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, two-stage review (spec compliance then code quality), fast iteration. Same shape as Spec 0.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
