# Odin Dashboard — Tab B (Spec 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Tab B — Task drill-down — combining per-seed reward + ep_length curves (overlay all seeds), aggregate stats card + per-seed table, and a cross-commit metric trend (line+ribbon by default, bar+whiskers as toggle). Driven by a searchable single-dropdown picker and deep-linkable via URL query string.

**Architecture:** New `tools/odin/valhalla/dashboard/tabs/task_drilldown/` package with 7 files (layout / picker / curves / stats / trend / url_state / callbacks). Mirrors Tab A's structure. No new DataLayer methods; reuses Spec 0's `load_aggregate`, `load_training`, `trend_dispatches_for`. Bundles a one-line Spec 1 extension: Tab A's Task cell becomes a `dcc.Link` to Tab B.

**Tech Stack:** Python 3.10+, Plotly Dash 4.x (`dcc.Dropdown` with `searchable=True`, `dcc.Graph` with plotly figures, pattern-matching ids unused in Tab B), pure-Python pytest. No new pip deps.

**Branch:** `antoiner/feat/odin` (continues from spec commit `83c65bcfc0e`).

**Spec:** `docs/superpowers/specs/2026-04-29-odin-dashboard-tab-b-task-drilldown-design.md`.

**Visual reference (read-only, do not edit):**
- `.superpowers/brainstorm/3762718-1777450164/content/curve-overlay.html` — Option A locked.
- `.superpowers/brainstorm/3762718-1777450164/content/stats-panel.html` — Option C locked.
- `.superpowers/brainstorm/3762718-1777450164/content/trend-shape.html` — Ribbon/bars side-by-side.

---

## Conventions used in every task

- **Test runner:** `PYTHONPATH=. python3 -m pytest <test> -v --tb=short --noconftest -p no:cacheprovider`. Pure-Python; sub-second per test.
- **Run tests one at a time** — never batch.
- **Pre-commit:** `./isaaclab.sh -f` BEFORE `git commit`. Restage and rerun until clean.
- **Commit message:** Imperative subject ≤ 50 chars; body wrapped at 72 chars; **NO** AI co-authorship lines.
- **No new pip deps.** Plotly already a hard dep from Spec 0.

## File map — what gets created or modified

| File | Owner task | Responsibility |
|---|---|---|
| `tools/odin/valhalla/dashboard/tabs/task_drilldown/__init__.py` | T1 | Package marker; re-exports `render` + `register`. |
| `tools/odin/valhalla/dashboard/tabs/task_drilldown/url_state.py` | T1 | `TaskSelection` dataclass + `parse_query_string` + `serialize`. |
| `tools/odin/valhalla/dashboard/tests/test_tab_b_url_state.py` | T1 | Round-trip + edge cases. |
| `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py` | T2 | `_data_row` Task cell becomes a `dcc.Link` to Tab B. |
| `tools/odin/valhalla/dashboard/tests/test_tab_a_jobs_table.py` | T2 | New test for the Task→Tab B link. |
| `tools/odin/valhalla/dashboard/tabs/task_drilldown/picker.py` | T3 | `list_row_options` + `render_picker`. |
| `tools/odin/valhalla/dashboard/tests/test_tab_b_picker.py` | T3 | 8 picker tests. |
| `tools/odin/valhalla/dashboard/tabs/task_drilldown/stats.py` | T4 | `render_aggregate_card` + `render_seeds_table` + `render_stats_panel`. |
| `tools/odin/valhalla/dashboard/tests/test_tab_b_stats.py` | T4 | 13 stats tests. |
| `tools/odin/valhalla/dashboard/tabs/task_drilldown/curves.py` | T5 | `render_curves` + `_SEED_PALETTE`. |
| `tools/odin/valhalla/dashboard/tests/test_tab_b_curves.py` | T5 | 8 curves tests. |
| `tools/odin/valhalla/dashboard/tabs/task_drilldown/trend.py` (compute) | T6 | `_TREND_METRICS` + `compute_trend_points` + `render_trend_chart`. |
| `tools/odin/valhalla/dashboard/tests/test_tab_b_trend.py` (first half) | T6 | 9 trend compute + chart tests. |
| `tools/odin/valhalla/dashboard/tabs/task_drilldown/trend.py` (section) | T7 | `render_metric_selector` + `render_trend_section` w/ empty states. |
| `tools/odin/valhalla/dashboard/tests/test_tab_b_trend.py` (rest) | T7 | 4 trend section tests. |
| `tools/odin/valhalla/dashboard/tabs/task_drilldown/layout.py` | T8 | `build_layout` with stores + slots. |
| `tools/odin/valhalla/dashboard/tabs/task_drilldown/callbacks.py` (1/2) | T9 | 3 standard callbacks: init_picker, sync_url, picker_to_url. |
| `tools/odin/valhalla/dashboard/tests/test_tab_b_callbacks.py` (first half) | T9 | 4 callback tests. |
| `tools/odin/valhalla/dashboard/tabs/task_drilldown/callbacks.py` (2/2) | T10 | 2 update callbacks: curves_and_stats, trend. |
| `tools/odin/valhalla/dashboard/tests/test_tab_b_callbacks.py` (rest) | T10 | 5 update callback tests. |
| `tools/odin/valhalla/dashboard/assets/style.css` | T11 | Tab B CSS classes. |
| `docs/odin/architecture.md` | T12 | Change-log entry + "Last updated" bump. |

---

## Task 1: `url_state.py` — query-string round-trip

**Files:**
- Create: `tools/odin/valhalla/dashboard/tabs/task_drilldown/__init__.py`
- Create: `tools/odin/valhalla/dashboard/tabs/task_drilldown/url_state.py`
- Create: `tools/odin/valhalla/dashboard/tests/test_tab_b_url_state.py`

This task delivers the URL state primitive and the package marker. The `__init__.py` includes `register()` lazy-imported but T9 hasn't landed `callbacks.py` yet — wrap the import in try/except (matching Spec 1 T2's approach) so the registry doesn't crash at app startup. T9 removes the guard.

- [ ] **Step 1.1: Write the failing tests** — `tools/odin/valhalla/dashboard/tests/test_tab_b_url_state.py`

```python
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
```

- [ ] **Step 1.2: Run new tests, verify they FAIL**

Run each individually:

```
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_tab_b_url_state.py::test_parse_empty_string_returns_empty_selection -v --tb=short --noconftest -p no:cacheprovider
```

Expected: each → FAIL with `ModuleNotFoundError: No module named 'tools.odin.valhalla.dashboard.tabs.task_drilldown'`.

- [ ] **Step 1.3: Create `__init__.py`** — `tools/odin/valhalla/dashboard/tabs/task_drilldown/__init__.py`

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tab B — Task drill-down — for the Odin dashboard."""

__all__ = ["render", "register"]


def render(dispatch_id: str, tab_id: str):
    """Spec 0 registry hook — return the static layout for this tab."""
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.layout import build_layout

    return build_layout(dispatch_id)


def register(app, data):
    """Spec 0 registry hook — wire Tab B's callbacks at app startup.

    Until T9 lands the callbacks module, this is a no-op so the
    Spec 0 registry's startup walk doesn't crash.
    """
    try:
        from tools.odin.valhalla.dashboard.tabs.task_drilldown.callbacks import (
            register_callbacks,
        )
    except ModuleNotFoundError:
        return
    register_callbacks(app, data)
```

- [ ] **Step 1.4: Create `url_state.py`** — `tools/odin/valhalla/dashboard/tabs/task_drilldown/url_state.py`

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""URL query-string parsing for Tab B's deep-linked picker state.

The query string carries ``?task=<task_id>&framework=<rsl_rl|skrl>
&backend=<physx|newton>``. Empty fields are omitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import parse_qs, urlencode

__all__ = ["TaskSelection", "parse_query_string", "serialize"]


@dataclass(frozen=True)
class TaskSelection:
    """One picker selection — pinned by URL state."""

    task: str | None
    framework: str | None
    backend: str | None


def parse_query_string(search: str) -> TaskSelection:
    """Parse Dash's ``dcc.Location.search`` into a :class:`TaskSelection`.

    ``search`` may start with ``'?'`` (Dash convention); both forms accepted.
    Duplicate keys take the last value (standard query-string semantics).
    """
    raw = search.lstrip("?")
    pairs = parse_qs(raw, keep_blank_values=False)
    return TaskSelection(
        task=_last(pairs, "task"),
        framework=_last(pairs, "framework"),
        backend=_last(pairs, "backend"),
    )


def serialize(selection: TaskSelection) -> str:
    """Return a query string starting with '?'.

    Empty / None fields are omitted; if all three are None, returns ''.
    """
    fields = [
        ("task", selection.task),
        ("framework", selection.framework),
        ("backend", selection.backend),
    ]
    populated = [(k, v) for k, v in fields if v is not None and v != ""]
    if not populated:
        return ""
    return "?" + urlencode(populated)


def _last(pairs: dict[str, list[str]], key: str) -> str | None:
    values = pairs.get(key)
    if not values:
        return None
    return values[-1]
```

- [ ] **Step 1.5: Run each test individually, verify all PASS** (9 tests)

- [ ] **Step 1.6: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add -f tools/odin/valhalla/dashboard/tabs/task_drilldown/__init__.py
git add tools/odin/valhalla/dashboard/tabs/task_drilldown/url_state.py tools/odin/valhalla/dashboard/tests/test_tab_b_url_state.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Tab B skeleton: package + url_state

Package marker exposes render() (lazy-imports build_layout) and
register() (lazy-imports callbacks; wrapped in try/except until T9
lands callbacks.py). url_state.py provides TaskSelection +
parse_query_string + serialize for the URL-driven picker
(?task=&framework=&backend=).
EOF
)"
```

---

## Task 2: Tab A regression — Task cell links to Tab B

**Files:**
- Modify: `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py` (`_data_row`, `render_jobs_rows`)
- Modify: `tools/odin/valhalla/dashboard/tests/test_tab_a_jobs_table.py`

`_data_row(job)` currently renders the Task cell as `html.Td(job.get("task_id", ""))`. Bundle the spec's Spec 1 extension: make it a `dcc.Link` whose href is `/<dispatch_id>/task-drilldown?task=…&framework=…&backend=…`. `_data_row` doesn't have `dispatch_id` today; thread it through `render_jobs_rows` (one new param) → `_data_row`.

- [ ] **Step 2.1: Append failing test** to `tools/odin/valhalla/dashboard/tests/test_tab_a_jobs_table.py`

```python
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
    expected = (
        "/20260427-141302/task-drilldown"
        "?task=Isaac-Ant-Direct-v0&framework=rsl_rl&backend=physx"
    )
    assert expected in hrefs
```

- [ ] **Step 2.2: Run new test, verify it FAILS**

Run:
```
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_tab_a_jobs_table.py::test_jobs_task_cell_links_to_tab_b -v --tb=short --noconftest -p no:cacheprovider
```
Expected: FAIL — Task cell is plain text, no `Link` component.

- [ ] **Step 2.3: Modify `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py`**

Add `dcc` import if not already present (it is — used for Dropdown / Input). Then thread `dispatch_id` through `render_jobs_rows` and `_data_row`.

In `render_jobs_section`, extract `dispatch_id` from `dispatch_payload`:

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
    """Build the jobs section (filter row + table or empty state)."""
    jobs = dispatch_payload.get("jobs", []) or []
    expanded_run_ids = expanded_run_ids or set()
    ssh_tail_store = ssh_tail_store or {}
    dispatch_id = str(dispatch_payload.get("dispatch_id", "") or "")
    # ...
```

In the visible-rows path (and the empty-jobs and empty-filters branches stay unchanged), pass `dispatch_id` to row builder. Then in `render_jobs_rows`:

```python
def render_jobs_rows(
    dispatch_payload: dict,
    *,
    status_filter: list[str] | None = None,
    kind_filter: list[str] | None = None,
    task_text: str = "",
    expanded_run_ids: set[str] | None = None,
    ssh_tail_store: dict[str, list[str]] | None = None,
) -> html.Div | html.Table:
    """Same as render_jobs_section but returns just the rows portion."""
    jobs = dispatch_payload.get("jobs", []) or []
    expanded_run_ids = expanded_run_ids or set()
    ssh_tail_store = ssh_tail_store or {}
    dispatch_id = str(dispatch_payload.get("dispatch_id", "") or "")
    # ... existing filter logic ...
    body_rows: list = []
    for j in visible:
        body_rows.append(_data_row(j, dispatch_id))
        if j.get("status") == "failed" and j.get("run_id") in expanded_run_ids:
            body_rows.append(_expand_row(j, ssh_tail_store.get(j.get("run_id"))))
    return html.Table(
        className="tab-a-jobs-table",
        children=[
            html.Thead(children=[header]),
            html.Tbody(id="tab-a-jobs-rows", children=body_rows),
        ],
    )
```

(Apply the same `_data_row(j, dispatch_id)` change in `render_jobs_section`'s visible-rows branch.)

Then update `_data_row`:

```python
def _data_row(job: dict, dispatch_id: str) -> html.Tr:
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

    if kind:
        failure_cell = [
            html.Span(kind, className=f"tab-a-kind-pill tab-a-kind-pill-{kind}"),
            html.Button(
                "▸",
                id={"type": "tab-a-expand-toggle", "run_id": job.get("run_id", "")},
                n_clicks=0,
                className="tab-a-expand-toggle",
                title="Show / hide failure details",
            ),
        ]
    else:
        failure_cell = "—"

    started_ended_text = (
        f"{started} · {ended}"
        if ended
        else (f"{started} · —" if started != "—" else "— · —")
    )

    task_id = job.get("task_id", "")
    framework = job.get("framework", "")
    backend = job.get("backend", "")
    task_link = dcc.Link(
        task_id,
        href=f"/{dispatch_id}/task-drilldown?task={task_id}&framework={framework}&backend={backend}",
        className="tab-a-task-link",
    )

    return html.Tr(
        children=[
            html.Td(task_link),
            html.Td(f"{framework} × {backend}", className="tab-a-mono"),
            html.Td(str(job.get("seed", ""))),
            html.Td(status_children),
            html.Td(failure_cell),
            html.Td(host, className="tab-a-mono"),
            html.Td(started_ended_text, className="tab-a-muted"),
        ]
    )
```

- [ ] **Step 2.4: Run the new test + all existing Tab A tests**

```
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/test_tab_a_jobs_table.py --tb=line --noconftest -p no:cacheprovider
```

Expected: all pass (15 existing + 1 new = 16 total).

- [ ] **Step 2.5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py tools/odin/valhalla/dashboard/tests/test_tab_a_jobs_table.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Tab A: Task cell links to Tab B drill-down

_data_row gains a dispatch_id parameter (threaded from
render_jobs_rows / render_jobs_section); the Task cell renders as a
dcc.Link with href /<dispatch_id>/task-drilldown?task=&framework=
&backend=. Spec 1 extension bundled into Spec 2 (Tab B is the
landing target).
EOF
)"
```

---

## Task 3: `picker.py` — searchable row dropdown

**Files:**
- Create: `tools/odin/valhalla/dashboard/tabs/task_drilldown/picker.py`
- Create: `tools/odin/valhalla/dashboard/tests/test_tab_b_picker.py`

- [ ] **Step 3.1: Write failing tests** — `tools/odin/valhalla/dashboard/tests/test_tab_b_picker.py`

```python
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
```

- [ ] **Step 3.2: Run new tests, verify they FAIL** (each individually)

- [ ] **Step 3.3: Implement `tools/odin/valhalla/dashboard/tabs/task_drilldown/picker.py`**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tab B picker — searchable single dropdown of (task, framework, backend) rows."""

from __future__ import annotations

from dash import dcc, html

from tools.odin.valhalla.dashboard.tabs.task_drilldown.url_state import TaskSelection

__all__ = ["list_row_options", "render_picker"]


def list_row_options(aggregate_payload: dict) -> list[dict]:
    """Return ``dcc.Dropdown`` options for every (task, framework, backend) row.

    Sorted A-Z by task name.
    """
    rows = aggregate_payload.get("rows", []) or []
    options = []
    for row in rows:
        task = str(row.get("task", ""))
        framework = str(row.get("framework", ""))
        backend = str(row.get("backend", ""))
        options.append(
            {
                "label": f"{task} · {framework} × {backend}",
                "value": f"{task}|{framework}|{backend}",
            }
        )
    options.sort(key=lambda o: o["label"])
    return options


def render_picker(aggregate_payload: dict, selected: TaskSelection | None) -> html.Div:
    """Build the picker Div.

    Returns Div(id='tab-b-picker') containing a searchable dcc.Dropdown
    'tab-b-row-select'. The dropdown's value is the pipe-separated row key;
    `selected` is resolved to a value if present, else left as None.
    """
    options = list_row_options(aggregate_payload)
    selected_value: str | None = None
    if selected is not None and selected.task and selected.framework and selected.backend:
        candidate = f"{selected.task}|{selected.framework}|{selected.backend}"
        if any(o["value"] == candidate for o in options):
            selected_value = candidate
    return html.Div(
        id="tab-b-picker",
        className="tab-b-picker-row",
        children=[
            html.Span("Row", className="tab-b-picker-label"),
            dcc.Dropdown(
                id="tab-b-row-select",
                options=options,
                value=selected_value,
                searchable=True,
                placeholder="Pick a (task × framework × backend) row…",
                className="tab-b-picker-dropdown",
            ),
        ],
    )
```

- [ ] **Step 3.4: Run each test individually, verify all PASS** (8 tests)

- [ ] **Step 3.5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/tabs/task_drilldown/picker.py tools/odin/valhalla/dashboard/tests/test_tab_b_picker.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Tab B picker: searchable row dropdown

list_row_options returns Dropdown options for every (task, fw,
backend) row in aggregate.json (label = 'task · fw × be',
value = 'task|fw|be' pipe-separated). render_picker wraps it in a
searchable dcc.Dropdown (id='tab-b-row-select') with a 'Row' label
and an empty-aggregate placeholder. Rows sorted A-Z by task name.
EOF
)"
```

---

## Task 4: `stats.py` — aggregate card + per-seed table

**Files:**
- Create: `tools/odin/valhalla/dashboard/tabs/task_drilldown/stats.py`
- Create: `tools/odin/valhalla/dashboard/tests/test_tab_b_stats.py`

- [ ] **Step 4.1: Write failing tests** — `tools/odin/valhalla/dashboard/tests/test_tab_b_stats.py`

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for tab_b.stats."""

from __future__ import annotations

from tools.odin.valhalla.dashboard.tabs.task_drilldown.stats import (
    render_aggregate_card,
    render_seeds_table,
    render_stats_panel,
)


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


def _has_class(component, cls: str) -> bool:
    for c in _walk(component):
        c_cls = getattr(c, "className", "") or ""
        if cls in c_cls.split():
            return True
    return False


def _has_id(component, target_id: str) -> bool:
    return any(getattr(c, "id", None) == target_id for c in _walk(component))


def _aggregate_block(cv_pct: float = 3.2) -> dict:
    return {
        "n_seeds_completed": 3,
        "n_seeds_failed": 0,
        "reward_final_ema":     {"mean": 7991.51, "std": 257.68, "min": 7795.39, "max": 8283.34, "cv_pct": cv_pct},
        "ep_length_final_ema":  {"mean": 839.76,  "std": 16.36,  "min": 822.41,  "max": 854.91,  "cv_pct": 1.95},
        "iter_time_s_mean":     {"mean": 1.828,   "std": 0.033,  "min": 1.803,   "max": 1.865,   "cv_pct": 1.78},
        "env_steps_per_s_mean": {"mean": 72670.28, "std": 1680.0, "min": 70762.97, "max": 73930.63, "cv_pct": 2.31},
        "ram_gb_peak":          {"mean": 4.57,   "std": 0.02,   "min": 4.55,    "max": 4.58,    "cv_pct": 0.33},
        "gpu_mem_gb_peak":      {"mean": 4.24,   "std": 0.0,    "min": 4.24,    "max": 4.24,    "cv_pct": 0.0},
    }


def _seed(*, status: str = "completed", **overrides) -> dict:
    base = {
        "run_id": "rsl-rl_physx_X_seed42",
        "status": status,
        "assigned_to": "10.0.0.1",
        "reward_final_ema": 7795.39,
        "ep_length_final_ema": 822.41,
        "iter_time_s_mean": 1.815,
        "iter_time_s_std": 0.217,
        "env_steps_per_s_mean": 73317.24,
        "iterations_completed": 1000,
        "total_wall_time_s": 1815.11,
        "ram_gb_peak": 4.57,
        "gpu_mem_gb_peak": 4.24,
        "startup_app_launch_s": 3.52,
        "startup_env_creation_s": 13.87,
        "startup_first_step_s": 0.002,
    }
    base.update(overrides)
    return base


def test_aggregate_card_renders_one_line_per_metric():
    component = render_aggregate_card(_aggregate_block(), divergent_seeds=[])
    text = _text_blob(component)
    for label in ("Reward", "Ep length", "Iter time", "env_steps/s", "RAM peak", "GPU mem peak"):
        assert label in text, f"missing label {label!r}"


def test_aggregate_card_formats_mean_pm_std():
    text = _text_blob(render_aggregate_card(_aggregate_block(), divergent_seeds=[]))
    assert "7991.51" in text
    assert "257.68" in text
    assert "cv 3.2%" in text


def test_aggregate_card_cv_color_green_below_5pct():
    component = render_aggregate_card(_aggregate_block(cv_pct=3.2), divergent_seeds=[])
    assert _has_class(component, "tab-b-cv-good")


def test_aggregate_card_cv_color_orange_5_to_15pct():
    component = render_aggregate_card(_aggregate_block(cv_pct=8.0), divergent_seeds=[])
    assert _has_class(component, "tab-b-cv-warn")


def test_aggregate_card_cv_color_red_above_15pct():
    component = render_aggregate_card(_aggregate_block(cv_pct=20.0), divergent_seeds=[])
    assert _has_class(component, "tab-b-cv-bad")


def test_aggregate_card_lists_divergent_seeds():
    component = render_aggregate_card(_aggregate_block(), divergent_seeds=["43"])
    assert "seed 43" in _text_blob(component)


def test_aggregate_card_no_divergent_seeds_renders_dash():
    component = render_aggregate_card(_aggregate_block(), divergent_seeds=[])
    assert "—" in _text_blob(component)


def test_seeds_table_one_row_per_seed():
    seeds = {"42": _seed(), "43": _seed(), "44": _seed()}
    component = render_seeds_table(seeds)
    rows = [c for c in _walk(component) if type(c).__name__ == "Tr"]
    assert len(rows) == 4  # 1 header + 3 seeds


def test_seeds_table_column_set():
    seeds = {"42": _seed()}
    component = render_seeds_table(seeds)
    headers = [c for c in _walk(component) if type(c).__name__ == "Th"]
    expected = ["Seed", "Status", "Reward", "Ep length", "Iter time",
                "env_steps/s", "RAM peak", "GPU mem", "Wall time", "Startup", "Host"]
    actual = [getattr(h, "children", "") for h in headers]
    assert actual == expected


def test_seeds_table_status_pill_for_completed():
    seeds = {"42": _seed(status="completed")}
    component = render_seeds_table(seeds)
    assert _has_class(component, "tab-b-seed-status-completed")


def test_seeds_table_status_pill_for_failed():
    seeds = {"42": _seed(status="failed")}
    component = render_seeds_table(seeds)
    assert _has_class(component, "tab-b-seed-status-failed")


def test_seeds_table_dashes_when_metric_missing():
    seed = {"run_id": "x", "status": "failed", "assigned_to": "10.0.0.1"}
    component = render_seeds_table({"42": seed})
    text = _text_blob(component)
    # Many dashes because no metrics present.
    assert text.count("—") >= 6


def test_render_stats_panel_contains_both_cards():
    row = {
        "task": "X", "framework": "rsl_rl", "backend": "physx",
        "aggregate": _aggregate_block(),
        "seeds": {"42": _seed()},
        "divergent_seeds": [],
    }
    component = render_stats_panel(row)
    assert _has_id(component, "tab-b-aggregate-card")
    assert _has_id(component, "tab-b-seeds-table")
```

- [ ] **Step 4.2: Run new tests, verify they FAIL** (each individually)

- [ ] **Step 4.3: Implement `tools/odin/valhalla/dashboard/tabs/task_drilldown/stats.py`**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render Tab B's stats panel: aggregate card + per-seed table."""

from __future__ import annotations

from dash import html

__all__ = ["render_aggregate_card", "render_seeds_table", "render_stats_panel"]


_METRIC_LABELS = [
    ("reward_final_ema",     "Reward",        ""),
    ("ep_length_final_ema",  "Ep length",     ""),
    ("iter_time_s_mean",     "Iter time",     " s"),
    ("env_steps_per_s_mean", "env_steps/s",   ""),
    ("ram_gb_peak",          "RAM peak",      " GB"),
    ("gpu_mem_gb_peak",      "GPU mem peak",  " GB"),
]


def render_aggregate_card(aggregate_block: dict, divergent_seeds: list[str]) -> html.Div:
    """Build the left-side aggregate card.

    aggregate_block: dict matching aggregate.json.rows[].aggregate.
    Renders one stat-line per metric ('label: mean ± std cv X.X%') plus a
    closing divergent-seeds line.
    """
    lines: list = []
    for key, label, unit in _METRIC_LABELS:
        block = aggregate_block.get(key)
        if not isinstance(block, dict):
            continue
        mean = float(block.get("mean", 0.0))
        std = float(block.get("std", 0.0))
        cv = float(block.get("cv_pct", 0.0))
        cv_class = _cv_class(cv)
        lines.append(
            html.Div(
                className="tab-b-stat-line",
                children=[
                    html.Span(label, className="tab-b-stat-label"),
                    html.Span(
                        children=[
                            html.Strong(_fmt_num(mean) + unit),
                            f"  ± {_fmt_num(std)}{unit}",
                            html.Span(f"  cv {cv:.1f}%", className=f"tab-b-cv {cv_class}"),
                        ],
                        className="tab-b-stat-value",
                    ),
                ],
            )
        )
    div_text = (
        "—" if not divergent_seeds else ", ".join(f"seed {s}" for s in divergent_seeds)
    )
    lines.append(
        html.Div(
            className="tab-b-stat-line tab-b-divergent-line",
            children=[
                html.Span("Divergent seeds", className="tab-b-stat-label"),
                html.Span(div_text, className="tab-b-stat-value"),
            ],
        )
    )
    return html.Div(
        id="tab-b-aggregate-card",
        className="tab-b-stats-card",
        children=[
            html.Div("Aggregate", className="tab-b-stats-card-title"),
            *lines,
        ],
    )


def render_seeds_table(seeds_block: dict) -> html.Div:
    """Build the right-side per-seed table (11 columns)."""
    headers = ["Seed", "Status", "Reward", "Ep length", "Iter time",
               "env_steps/s", "RAM peak", "GPU mem", "Wall time", "Startup", "Host"]
    header_row = html.Tr(children=[html.Th(h) for h in headers])
    body_rows: list = []
    for seed_key in sorted(seeds_block.keys(), key=lambda k: int(k) if str(k).isdigit() else 0):
        seed = seeds_block[seed_key]
        body_rows.append(_seed_row(seed_key, seed))
    return html.Div(
        id="tab-b-seeds-table",
        className="tab-b-seeds-table-wrapper",
        children=[
            html.Table(
                children=[html.Thead(children=[header_row]), html.Tbody(children=body_rows)],
                className="tab-b-seeds-table",
            )
        ],
    )


def render_stats_panel(aggregate_payload_row: dict) -> html.Div:
    """Combine aggregate card + per-seed table in a two-column row."""
    aggregate_block = aggregate_payload_row.get("aggregate", {}) or {}
    seeds_block = aggregate_payload_row.get("seeds", {}) or {}
    divergent = aggregate_payload_row.get("divergent_seeds", []) or []
    return html.Div(
        id="tab-b-stats-content",
        className="tab-b-stats-row",
        children=[
            render_aggregate_card(aggregate_block, divergent),
            render_seeds_table(seeds_block),
        ],
    )


def _seed_row(seed_key: str, seed: dict) -> html.Tr:
    status = str(seed.get("status", "unknown"))
    pill_label = {"completed": "OK", "failed": "Failed"}.get(status, status.capitalize())
    pill = html.Span(
        pill_label,
        className=f"tab-b-pill tab-b-seed-status-{status}",
    )
    return html.Tr(
        children=[
            html.Td(seed_key, className="tab-b-seed-id"),
            html.Td(pill),
            html.Td(_fmt_or_dash(seed.get("reward_final_ema")), className="tab-b-mono"),
            html.Td(_fmt_or_dash(seed.get("ep_length_final_ema")), className="tab-b-mono"),
            html.Td(_fmt_or_dash(seed.get("iter_time_s_mean"), suffix=" s"), className="tab-b-mono"),
            html.Td(_fmt_or_dash(seed.get("env_steps_per_s_mean"), int_fmt=True), className="tab-b-mono"),
            html.Td(_fmt_or_dash(seed.get("ram_gb_peak"), suffix=" GB"), className="tab-b-mono"),
            html.Td(_fmt_or_dash(seed.get("gpu_mem_gb_peak"), suffix=" GB"), className="tab-b-mono"),
            html.Td(_fmt_wall_time(seed.get("total_wall_time_s")), className="tab-b-mono"),
            html.Td(_fmt_startup_phases(seed), className="tab-b-mono"),
            html.Td(str(seed.get("assigned_to") or "—"), className="tab-b-mono tab-b-muted"),
        ]
    )


def _cv_class(cv_pct: float) -> str:
    if cv_pct <= 5.0:
        return "tab-b-cv-good"
    if cv_pct <= 15.0:
        return "tab-b-cv-warn"
    return "tab-b-cv-bad"


def _fmt_num(value) -> str:
    if value is None:
        return "—"
    if abs(value) >= 1000:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _fmt_or_dash(value, *, suffix: str = "", int_fmt: bool = False) -> str:
    if value is None:
        return "—"
    if int_fmt:
        return f"{int(value)}{suffix}"
    return f"{_fmt_num(float(value))}{suffix}"


def _fmt_wall_time(seconds) -> str:
    if seconds is None:
        return "—"
    seconds = int(float(seconds))
    minutes, sec = divmod(seconds, 60)
    return f"{minutes}:{sec:02d}"


def _fmt_startup_phases(seed: dict) -> str:
    parts = [seed.get("startup_app_launch_s"), seed.get("startup_env_creation_s"),
             seed.get("startup_first_step_s")]
    if all(p is None for p in parts):
        return "—"
    rendered = []
    for p in parts:
        if p is None:
            rendered.append("—")
        elif p < 1:
            rendered.append(f"{p:.3f}")
        else:
            rendered.append(f"{p:.1f}")
    return " / ".join(rendered)
```

- [ ] **Step 4.4: Run each test individually, verify all PASS** (13 tests)

- [ ] **Step 4.5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/tabs/task_drilldown/stats.py tools/odin/valhalla/dashboard/tests/test_tab_b_stats.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Tab B stats: aggregate card + per-seed table

render_aggregate_card builds one stat-line per metric in the
aggregate block (mean ± std + cv%; cv color-codes green ≤5,
orange 5-15, red >15) plus a divergent-seeds line. render_seeds_table
builds the 11-column per-seed table with status pills (green OK /
red Failed) and dash placeholders for missing metrics.
render_stats_panel wraps both into a two-column layout.
EOF
)"
```

---

## Task 5: `curves.py` — reward + ep_length overlay charts

**Files:**
- Create: `tools/odin/valhalla/dashboard/tabs/task_drilldown/curves.py`
- Create: `tools/odin/valhalla/dashboard/tests/test_tab_b_curves.py`

- [ ] **Step 5.1: Write failing tests** — `tools/odin/valhalla/dashboard/tests/test_tab_b_curves.py`

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for tab_b.curves."""

from __future__ import annotations

from tools.odin.valhalla.dashboard.tabs.task_drilldown.curves import (
    _SEED_PALETTE,
    render_curves,
)


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


def _bundle(reward: list[float] | None = None, ep_length: list[float] | None = None) -> dict:
    payload: dict = {"schema_version": "1.0", "learning": {}}
    if reward is not None:
        payload["learning"]["reward"] = {"series_per_iter": reward, "final_ema": reward[-1]}
    if ep_length is not None:
        payload["learning"]["ep_length"] = {"series_per_iter": ep_length, "final_ema": ep_length[-1]}
    return payload


def _series(start: float, n: int = 10, slope: float = 5.0) -> list[float]:
    return [start + i * slope for i in range(n)]


def test_render_curves_returns_two_graph_components():
    bundles = {
        "42": _bundle(reward=_series(10), ep_length=_series(100)),
        "43": _bundle(reward=_series(15), ep_length=_series(110)),
    }
    component = render_curves(bundles, divergent_seeds=[])
    graphs = [c for c in _walk(component) if type(c).__name__ == "Graph"]
    assert len(graphs) == 2


def test_render_curves_one_trace_per_seed():
    bundles = {f"4{i}": _bundle(reward=_series(10 + i)) for i in range(3)}
    component = render_curves(bundles, divergent_seeds=[])
    graph = next(c for c in _walk(component) if type(c).__name__ == "Graph")
    fig = graph.figure
    assert len(fig.data) == 3


def test_render_curves_seed_color_assignment_deterministic():
    bundles = {
        "42": _bundle(reward=_series(10)),
        "43": _bundle(reward=_series(15)),
        "44": _bundle(reward=_series(20)),
    }
    component = render_curves(bundles, divergent_seeds=[])
    graph = next(c for c in _walk(component) if type(c).__name__ == "Graph")
    fig = graph.figure
    colors = [trace.line.color for trace in fig.data]
    assert colors[:3] == _SEED_PALETTE[:3]


def test_render_curves_divergent_seed_styled_differently():
    bundles = {
        "42": _bundle(reward=_series(10)),
        "43": _bundle(reward=_series(15)),
    }
    component = render_curves(bundles, divergent_seeds=["43"])
    graph = next(c for c in _walk(component) if type(c).__name__ == "Graph")
    fig = graph.figure
    seed_43_trace = next(t for t in fig.data if t.name == "seed 43")
    assert seed_43_trace.line.color == "#e16868"
    assert seed_43_trace.line.dash == "dash"


def test_render_curves_handles_missing_seed_series():
    bundles = {
        "42": _bundle(reward=_series(10)),
        "43": _bundle(),  # no series
    }
    component = render_curves(bundles, divergent_seeds=[])
    graph = next(c for c in _walk(component) if type(c).__name__ == "Graph")
    fig = graph.figure
    # Only seed 42 should have a trace.
    assert len(fig.data) == 1
    assert fig.data[0].name == "seed 42"


def test_render_curves_empty_bundles_renders_empty_state():
    component = render_curves({}, divergent_seeds=[])
    graphs = [c for c in _walk(component) if type(c).__name__ == "Graph"]
    assert len(graphs) == 0
    text_parts = [getattr(c, "children", "") for c in _walk(component) if isinstance(getattr(c, "children", None), str)]
    text = " ".join(text_parts)
    assert "No training.json bundles" in text


def test_render_curves_x_axis_label():
    bundles = {"42": _bundle(reward=_series(10))}
    component = render_curves(bundles, divergent_seeds=[])
    graph = next(c for c in _walk(component) if type(c).__name__ == "Graph")
    fig = graph.figure
    assert "iteration" in fig.layout.xaxis.title.text.lower()


def test_render_curves_reward_y_axis_label():
    bundles = {"42": _bundle(reward=_series(10))}
    component = render_curves(bundles, divergent_seeds=[])
    graphs = [c for c in _walk(component) if type(c).__name__ == "Graph"]
    reward_graph = graphs[0]
    fig = reward_graph.figure
    assert "reward" in fig.layout.yaxis.title.text.lower()
```

- [ ] **Step 5.2: Run new tests, verify they FAIL** (each individually)

- [ ] **Step 5.3: Implement `tools/odin/valhalla/dashboard/tabs/task_drilldown/curves.py`**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render Tab B's reward + ep_length overlay charts."""

from __future__ import annotations

import plotly.graph_objects as go
from dash import dcc, html

__all__ = ["render_curves", "_SEED_PALETTE"]


_SEED_PALETTE = ["#66b6ff", "#50c88c", "#ffa500", "#beA0ff", "#e16868"]
_DIVERGENT_COLOR = "#e16868"


def render_curves(
    bundles: dict[str, dict],
    *,
    divergent_seeds: list[str],
) -> html.Div:
    """Build two plotly figures (reward + ep_length) overlaying all seeds.

    Args:
        bundles: ``{seed_str: training_payload}``. Missing series → seed skipped.
        divergent_seeds: seeds drawn with red dashed stroke instead of palette.

    Returns:
        Div(id='tab-b-curves-content') with up to 2 ``dcc.Graph`` components.
    """
    if not bundles:
        return html.Div(
            id="tab-b-curves-content",
            className="tab-b-empty-state",
            children=[html.P("No training.json bundles available for this row.")],
        )

    sorted_seeds = sorted(bundles.keys(), key=lambda k: int(k) if str(k).isdigit() else 0)
    divergent_set = set(str(s) for s in divergent_seeds)

    reward_fig = _build_overlay_figure(
        sorted_seeds, bundles, "reward", divergent_set, y_label="reward (final EMA)"
    )
    ep_fig = _build_overlay_figure(
        sorted_seeds, bundles, "ep_length", divergent_set, y_label="ep_length (final EMA)"
    )
    return html.Div(
        id="tab-b-curves-content",
        children=[
            dcc.Graph(id="tab-b-curve-reward", figure=reward_fig, config={"displayModeBar": False}),
            dcc.Graph(id="tab-b-curve-ep-length", figure=ep_fig, config={"displayModeBar": False}),
        ],
    )


def _build_overlay_figure(
    sorted_seeds: list[str],
    bundles: dict[str, dict],
    series_key: str,
    divergent_set: set[str],
    y_label: str,
) -> go.Figure:
    fig = go.Figure()
    for idx, seed in enumerate(sorted_seeds):
        learning = bundles[seed].get("learning", {}) or {}
        block = learning.get(series_key) or {}
        series = block.get("series_per_iter")
        if not series:
            continue
        is_divergent = seed in divergent_set
        color = _DIVERGENT_COLOR if is_divergent else _SEED_PALETTE[idx % len(_SEED_PALETTE)]
        dash = "dash" if is_divergent else "solid"
        fig.add_trace(
            go.Scatter(
                x=list(range(len(series))),
                y=series,
                mode="lines",
                name=f"seed {seed}",
                line={"color": color, "dash": dash, "width": 1.6},
                hovertemplate=f"seed {seed}<br>iter %{{x}}<br>{series_key} %{{y:.2f}}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        margin={"l": 50, "r": 20, "t": 30, "b": 40},
        xaxis_title="iterations",
        yaxis_title=y_label,
        height=240,
        legend={"orientation": "h", "y": -0.18},
    )
    return fig
```

- [ ] **Step 5.4: Run each test individually, verify all PASS** (8 tests)

- [ ] **Step 5.5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/tabs/task_drilldown/curves.py tools/odin/valhalla/dashboard/tests/test_tab_b_curves.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Tab B curves: overlay reward + ep_length per seed

render_curves builds two plotly figures (reward EMA + ep_length EMA)
with one trace per seed using the shared _SEED_PALETTE (blue, green,
orange, purple, red). Divergent seeds draw with a red dashed stroke
on top. Plotly's legend toggles individual seeds. Empty bundles or
missing series_per_iter fields fall back to an empty-state message.
EOF
)"
```

---

## Task 6: `trend.py` — compute_trend_points + render_trend_chart

**Files:**
- Create: `tools/odin/valhalla/dashboard/tabs/task_drilldown/trend.py`
- Create: `tools/odin/valhalla/dashboard/tests/test_tab_b_trend.py`

This task delivers the metric universe constant, the trend-points compute helper, and the chart renderer. T7 adds `render_trend_section` (which wraps these with empty/single-point states) and `render_metric_selector`.

- [ ] **Step 6.1: Write the first 9 failing tests** — `tools/odin/valhalla/dashboard/tests/test_tab_b_trend.py`

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for tab_b.trend."""

from __future__ import annotations

from tools.odin.valhalla.dashboard.tabs.task_drilldown.trend import (
    compute_trend_points,
    render_trend_chart,
)


def _row(*, task: str = "X", framework: str = "rsl_rl", backend: str = "physx",
         agg: dict | None = None, seeds: dict | None = None) -> dict:
    return {
        "task": task, "framework": framework, "backend": backend,
        "aggregate": agg or {},
        "seeds": seeds or {},
        "divergent_seeds": [],
    }


def _aggregate_block(reward_mean: float = 7991.51, reward_std: float = 257.68) -> dict:
    return {
        "n_seeds_completed": 3,
        "reward_final_ema":     {"mean": reward_mean, "std": reward_std, "min": 0.0, "max": 0.0, "cv_pct": 3.2},
        "ep_length_final_ema":  {"mean": 839.76, "std": 16.36, "min": 0, "max": 0, "cv_pct": 1.95},
        "iter_time_s_mean":     {"mean": 1.828,  "std": 0.033, "min": 0, "max": 0, "cv_pct": 1.78},
        "env_steps_per_s_mean": {"mean": 72670.28, "std": 1680.0, "min": 0, "max": 0, "cv_pct": 2.31},
        "ram_gb_peak":          {"mean": 4.57,   "std": 0.02,  "min": 0, "max": 0, "cv_pct": 0.33},
        "gpu_mem_gb_peak":      {"mean": 4.24,   "std": 0.0,   "min": 0, "max": 0, "cv_pct": 0.0},
    }


def _seed(*, startup_app_launch_s: float = 3.5) -> dict:
    return {
        "run_id": "rsl-rl_physx_X_seed42",
        "status": "completed",
        "reward_final_ema": 7795.39,
        "iter_time_s_mean": 1.815,
        "ram_gb_peak": 4.57,
        "startup_app_launch_s": startup_app_launch_s,
        "startup_env_creation_s": 13.87,
        "startup_first_step_s": 0.002,
    }


class _StubData:
    """DataLayer drop-in for trend tests."""

    def __init__(self):
        self._aggregates: dict[str, dict | None] = {}
        self._dispatches: dict[str, dict] = {}
        self.load_aggregate_calls: list[str] = []

    def add_dispatch(self, dispatch_id: str, *, commit: str = "abc1234",
                     row_kwargs: dict | None = None, no_aggregate: bool = False):
        self._dispatches[dispatch_id] = {
            "schema_version": "1.3",
            "dispatch_id": dispatch_id,
            "commit_sha": commit,
            "fleet": [],
            "jobs": [],
            "skipped": [],
        }
        if no_aggregate:
            self._aggregates[dispatch_id] = None
        elif row_kwargs is None:
            self._aggregates[dispatch_id] = {"schema_version": "1.0", "rows": []}
        else:
            self._aggregates[dispatch_id] = {
                "schema_version": "1.0",
                "rows": [_row(**row_kwargs)],
            }

    def load_aggregate(self, dispatch_id: str):
        self.load_aggregate_calls.append(dispatch_id)
        return self._aggregates.get(dispatch_id)

    def load_dispatch(self, dispatch_id: str):
        return self._dispatches[dispatch_id]


def test_compute_points_returns_one_per_dispatch():
    data = _StubData()
    for did in ["d1", "d2", "d3"]:
        data.add_dispatch(did, commit=did, row_kwargs={"agg": _aggregate_block()})
    points = compute_trend_points(data, ["d1", "d2", "d3"], "X", "rsl_rl", "physx", "reward_final_ema")
    assert len(points) == 3


def test_compute_points_skips_missing_aggregate():
    data = _StubData()
    data.add_dispatch("d1", row_kwargs={"agg": _aggregate_block()})
    data.add_dispatch("d2", no_aggregate=True)
    data.add_dispatch("d3", row_kwargs={"agg": _aggregate_block()})
    points = compute_trend_points(data, ["d1", "d2", "d3"], "X", "rsl_rl", "physx", "reward_final_ema")
    assert len(points) == 2
    assert [p["dispatch_id"] for p in points] == ["d1", "d3"]


def test_compute_points_skips_dispatch_missing_row():
    data = _StubData()
    data.add_dispatch("d1", row_kwargs={"agg": _aggregate_block()})
    data.add_dispatch("d2")  # row_kwargs=None → empty rows
    points = compute_trend_points(data, ["d1", "d2"], "X", "rsl_rl", "physx", "reward_final_ema")
    assert len(points) == 1
    assert points[0]["dispatch_id"] == "d1"


def test_compute_points_uses_aggregate_for_known_metric():
    data = _StubData()
    data.add_dispatch("d1", row_kwargs={"agg": _aggregate_block(reward_mean=7000, reward_std=300)})
    points = compute_trend_points(data, ["d1"], "X", "rsl_rl", "physx", "reward_final_ema")
    assert points[0]["mean"] == 7000.0
    assert points[0]["std"] == 300.0


def test_compute_points_computes_from_seeds_for_startup_metric():
    data = _StubData()
    data.add_dispatch("d1", row_kwargs={
        "agg": _aggregate_block(),
        "seeds": {
            "42": _seed(startup_app_launch_s=3.0),
            "43": _seed(startup_app_launch_s=4.0),
            "44": _seed(startup_app_launch_s=5.0),
        },
    })
    points = compute_trend_points(data, ["d1"], "X", "rsl_rl", "physx", "startup_app_launch_s")
    assert points[0]["mean"] == 4.0
    # population std of [3, 4, 5] = sqrt(2/3) ≈ 0.8165
    assert abs(points[0]["std"] - 0.8164965809277260) < 1e-6


def test_compute_points_carries_commit_sha():
    data = _StubData()
    data.add_dispatch("d1", commit="abc1234567890", row_kwargs={"agg": _aggregate_block()})
    points = compute_trend_points(data, ["d1"], "X", "rsl_rl", "physx", "reward_final_ema")
    assert points[0]["commit_sha"] == "abc1234567890"


def test_render_trend_chart_ribbon_mode():
    points = [
        {"dispatch_id": "d1", "commit_sha": "aaa1111", "mean": 7000.0, "std": 300.0, "n_seeds_completed": 3},
        {"dispatch_id": "d2", "commit_sha": "bbb2222", "mean": 7500.0, "std": 280.0, "n_seeds_completed": 3},
        {"dispatch_id": "d3", "commit_sha": "ccc3333", "mean": 7991.5, "std": 257.7, "n_seeds_completed": 3},
    ]
    graph = render_trend_chart(points, "Reward (final EMA)", mode="ribbon")
    fig = graph.figure
    # ribbon mode → 2 traces (mean line + fill band).
    assert len(fig.data) == 2


def test_render_trend_chart_bars_mode():
    points = [
        {"dispatch_id": "d1", "commit_sha": "aaa1111", "mean": 7000.0, "std": 300.0, "n_seeds_completed": 3},
        {"dispatch_id": "d2", "commit_sha": "bbb2222", "mean": 7500.0, "std": 280.0, "n_seeds_completed": 3},
    ]
    graph = render_trend_chart(points, "Reward (final EMA)", mode="bars")
    fig = graph.figure
    # bars mode → 1 trace with error_y populated.
    assert len(fig.data) == 1
    assert fig.data[0].error_y is not None


def test_render_trend_chart_x_labels_short_sha():
    points = [
        {"dispatch_id": "d1", "commit_sha": "abcdef1234567", "mean": 1.0, "std": 0.1, "n_seeds_completed": 3},
    ]
    graph = render_trend_chart(points, "Iter time", mode="ribbon")
    fig = graph.figure
    # Tick labels should use the short SHA (first 7).
    tick_text = list(fig.layout.xaxis.ticktext or [])
    assert "abcdef1" in tick_text
```

- [ ] **Step 6.2: Run new tests, verify they FAIL** (each individually)

- [ ] **Step 6.3: Implement `tools/odin/valhalla/dashboard/tabs/task_drilldown/trend.py`** (compute + chart only)

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render Tab B's metric trend section: selector + line/bar chart over N dispatches."""

from __future__ import annotations

import statistics
import sys

import plotly.graph_objects as go
from dash import dcc

__all__ = [
    "_TREND_METRICS",
    "compute_trend_points",
    "render_trend_chart",
]


_TREND_METRICS = [
    {"value": "reward_final_ema",      "label": "Reward (final EMA)",          "source": "aggregate"},
    {"value": "ep_length_final_ema",   "label": "Episode length (final EMA)",  "source": "aggregate"},
    {"value": "iter_time_s_mean",      "label": "Iter time",                   "source": "aggregate"},
    {"value": "env_steps_per_s_mean",  "label": "Env steps / s",               "source": "aggregate"},
    {"value": "ram_gb_peak",           "label": "RAM peak",                    "source": "aggregate"},
    {"value": "gpu_mem_gb_peak",       "label": "GPU mem peak",                "source": "aggregate"},
    {"value": "startup_app_launch_s",  "label": "Startup: app launch",         "source": "seeds"},
    {"value": "startup_env_creation_s","label": "Startup: env creation",       "source": "seeds"},
    {"value": "startup_first_step_s",  "label": "Startup: first step",         "source": "seeds"},
]
_METRIC_SOURCES = {m["value"]: m["source"] for m in _TREND_METRICS}


def compute_trend_points(
    data,
    dispatch_ids: list[str],
    task: str,
    framework: str,
    backend: str,
    metric: str,
) -> list[dict]:
    """For each dispatch_id, compute one point for the metric.

    Each point: {dispatch_id, commit_sha, mean, std, n_seeds_completed}.

    Skips dispatches whose aggregate.json is missing or whose rows[] don't
    include the requested (task, framework, backend) tuple. Logs a [WARNING]
    for any dispatch whose aggregate read raises.
    """
    source = _METRIC_SOURCES.get(metric, "aggregate")
    out: list[dict] = []
    for did in dispatch_ids:
        try:
            agg = data.load_aggregate(did)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] tab-b trend: load_aggregate({did}) raised: {exc}", file=sys.stderr)
            continue
        if agg is None:
            continue
        row = _find_row(agg, task, framework, backend)
        if row is None:
            continue
        try:
            mean, std = _metric_value(row, metric, source)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] tab-b trend: metric extract failed for {did}: {exc}", file=sys.stderr)
            continue
        commit_sha = ""
        try:
            commit_sha = str(data.load_dispatch(did).get("commit_sha", "") or "")
        except Exception:  # noqa: BLE001
            pass
        n_seeds = (row.get("aggregate") or {}).get("n_seeds_completed", 0)
        out.append({
            "dispatch_id": did,
            "commit_sha": commit_sha,
            "mean": mean,
            "std": std,
            "n_seeds_completed": int(n_seeds or 0),
        })
    return out


def render_trend_chart(points: list[dict], metric_label: str, *, mode: str = "ribbon") -> dcc.Graph:
    """Render the trend chart.

    mode='ribbon' → mean line + fill band of (mean ± std).
    mode='bars'   → bar chart with vertical error_y whiskers.
    """
    n = len(points)
    if n == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_dark", height=240)
        return dcc.Graph(id="tab-b-trend-chart", figure=empty_fig, config={"displayModeBar": False})

    xs = list(range(n))
    means = [p["mean"] for p in points]
    stds = [p["std"] for p in points]
    short_shas = [(p["commit_sha"][:7] or "—") for p in points]
    labels = [
        f"{p['commit_sha'][:7] or '—'}<br>n={p['n_seeds_completed']}<br>{p['dispatch_id']}"
        for p in points
    ]
    # Mark the rightmost (newest) point as "current".
    short_shas[-1] = short_shas[-1] + " ▲"

    fig = go.Figure()
    if mode == "bars":
        marker_colors = ["#66b6ff"] * (n - 1) + ["#76b900"]
        fig.add_trace(
            go.Bar(
                x=xs,
                y=means,
                error_y={"type": "data", "array": stds, "color": "#aaa"},
                marker={"color": marker_colors},
                hovertext=labels,
                hovertemplate="%{hovertext}<br>" + metric_label + ": %{y:.2f}<extra></extra>",
                name=metric_label,
            )
        )
    else:
        # Ribbon: fill band (upper-then-lower) drawn first, then mean line on top.
        upper = [m + s for m, s in zip(means, stds)]
        lower = [m - s for m, s in zip(means, stds)]
        fig.add_trace(
            go.Scatter(
                x=xs + xs[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor="rgba(102,182,255,0.18)",
                line={"color": "rgba(0,0,0,0)"},
                name="±std",
                hoverinfo="skip",
                showlegend=False,
            )
        )
        marker_colors = ["#66b6ff"] * (n - 1) + ["#76b900"]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=means,
                mode="lines+markers",
                line={"color": "#66b6ff", "width": 2},
                marker={"color": marker_colors, "size": 8, "line": {"color": "#fff", "width": 1}},
                hovertext=labels,
                hovertemplate="%{hovertext}<br>" + metric_label + ": %{y:.2f}<extra></extra>",
                name=metric_label,
            )
        )

    fig.update_layout(
        template="plotly_dark",
        margin={"l": 60, "r": 20, "t": 30, "b": 60},
        xaxis={
            "tickmode": "array",
            "tickvals": xs,
            "ticktext": short_shas,
        },
        yaxis_title=metric_label,
        height=260,
        showlegend=False,
    )
    return dcc.Graph(id="tab-b-trend-chart", figure=fig, config={"displayModeBar": False})


def _find_row(aggregate: dict, task: str, framework: str, backend: str) -> dict | None:
    for row in aggregate.get("rows", []) or []:
        if row.get("task") == task and row.get("framework") == framework and row.get("backend") == backend:
            return row
    return None


def _metric_value(row: dict, metric: str, source: str) -> tuple[float, float]:
    if source == "aggregate":
        block = (row.get("aggregate") or {}).get(metric)
        if not isinstance(block, dict):
            raise KeyError(metric)
        return float(block.get("mean", 0.0)), float(block.get("std", 0.0))
    seeds = row.get("seeds") or {}
    values = []
    for seed in seeds.values():
        v = seed.get(metric)
        if v is not None:
            values.append(float(v))
    if not values:
        raise ValueError(f"no per-seed values for {metric}")
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std
```

- [ ] **Step 6.4: Run each test individually, verify all PASS** (9 tests)

- [ ] **Step 6.5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/tabs/task_drilldown/trend.py tools/odin/valhalla/dashboard/tests/test_tab_b_trend.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Tab B trend (1/2): _TREND_METRICS + compute_trend_points + chart

_TREND_METRICS exposes 9 metrics across two sources: 6 read from
aggregate.rows[].aggregate (reward, ep_length, iter_time,
env_steps/s, RAM, GPU mem) and 3 computed on the fly from seeds[]
(startup app/env/first). compute_trend_points walks dispatch_ids,
extracts (mean, std) per dispatch, skips missing aggregates / rows.
render_trend_chart handles both ribbon (line + fill band) and bars
(error_y whiskers) modes; rightmost point drawn in NVIDIA green.
T7 wraps these with empty/single-point states.
EOF
)"
```

---

## Task 7: `trend.py` — render_metric_selector + render_trend_section

**Files:**
- Modify: `tools/odin/valhalla/dashboard/tabs/task_drilldown/trend.py`
- Modify: `tools/odin/valhalla/dashboard/tests/test_tab_b_trend.py`

- [ ] **Step 7.1: Append failing tests** to `tools/odin/valhalla/dashboard/tests/test_tab_b_trend.py`

```python
def test_render_trend_chart_current_marker_highlighted():
    points = [
        {"dispatch_id": "d1", "commit_sha": "aaa1111", "mean": 7000.0, "std": 300.0, "n_seeds_completed": 3},
        {"dispatch_id": "d2", "commit_sha": "bbb2222", "mean": 7500.0, "std": 280.0, "n_seeds_completed": 3},
        {"dispatch_id": "d3", "commit_sha": "ccc3333", "mean": 7991.5, "std": 257.7, "n_seeds_completed": 3},
    ]
    graph = render_trend_chart(points, "Reward (final EMA)", mode="ribbon")
    fig = graph.figure
    # Locate the line+marker trace (the second one in ribbon mode).
    line_trace = fig.data[1]
    marker_colors = list(line_trace.marker.color)
    # Last marker color is NVIDIA green (#76b900); others are #66b6ff.
    assert marker_colors[-1] == "#76b900"
    assert all(c == "#66b6ff" for c in marker_colors[:-1])


def test_render_trend_section_empty_state():
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.trend import render_trend_section
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.url_state import TaskSelection

    class _Data:
        def trend_dispatches_for(self, current, task, fw, be, n=10):
            return []
        def load_hardware(self, dispatch_id):
            return {"fingerprint": "gpu:NVIDIA-L40"}

    selection = TaskSelection("Isaac-Ant-Direct-v0", "rsl_rl", "physx")
    component = render_trend_section(
        _Data(), current_dispatch_id="d-now", selection=selection,
        metric="reward_final_ema", mode="ribbon",
    )
    text_parts = [getattr(c, "children", "") for c in _walk(component) if isinstance(getattr(c, "children", None), str)]
    text = " ".join(text_parts)
    assert "Trend needs at least 1 prior" in text


def test_render_trend_section_single_point_state():
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.trend import render_trend_section
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.url_state import TaskSelection

    data = _StubData()
    data.add_dispatch("d-now", row_kwargs={"agg": _aggregate_block()})

    class _DataWrapper:
        def trend_dispatches_for(self, current, task, fw, be, n=10):
            return ["d-now"]
        def load_aggregate(self, dispatch_id):
            return data.load_aggregate(dispatch_id)
        def load_dispatch(self, dispatch_id):
            return data.load_dispatch(dispatch_id)
        def load_hardware(self, dispatch_id):
            return {"fingerprint": "gpu:NVIDIA-L40"}

    selection = TaskSelection("X", "rsl_rl", "physx")
    component = render_trend_section(
        _DataWrapper(), current_dispatch_id="d-now", selection=selection,
        metric="reward_final_ema", mode="ribbon",
    )
    text_parts = [getattr(c, "children", "") for c in _walk(component) if isinstance(getattr(c, "children", None), str)]
    text = " ".join(text_parts)
    assert "First matching dispatch" in text


def test_render_trend_section_normal_render():
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.trend import render_trend_section
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.url_state import TaskSelection

    data = _StubData()
    for did in ["d1", "d2", "d3"]:
        data.add_dispatch(did, commit=did, row_kwargs={"agg": _aggregate_block()})

    class _DataWrapper:
        def trend_dispatches_for(self, current, task, fw, be, n=10):
            return ["d1", "d2", "d3"]
        def load_aggregate(self, dispatch_id):
            return data.load_aggregate(dispatch_id)
        def load_dispatch(self, dispatch_id):
            return data.load_dispatch(dispatch_id)
        def load_hardware(self, dispatch_id):
            return {"fingerprint": "gpu:NVIDIA-L40"}

    selection = TaskSelection("X", "rsl_rl", "physx")
    component = render_trend_section(
        _DataWrapper(), current_dispatch_id="d3", selection=selection,
        metric="reward_final_ema", mode="ribbon",
    )
    graphs = [c for c in _walk(component) if type(c).__name__ == "Graph"]
    assert len(graphs) == 1
```

- [ ] **Step 7.2: Run new tests, verify they FAIL** (each individually)

- [ ] **Step 7.3: Append `render_metric_selector` and `render_trend_section`** to `tools/odin/valhalla/dashboard/tabs/task_drilldown/trend.py`

Update `__all__`:
```python
__all__ = [
    "_TREND_METRICS",
    "compute_trend_points",
    "render_metric_selector",
    "render_trend_chart",
    "render_trend_section",
]
```

Append after `render_trend_chart`:

```python
def render_metric_selector(default_metric: str = "reward_final_ema",
                           default_mode: str = "ribbon"):
    """Build the metric dropdown + view-mode toggle.

    Returns a Div with two child controls:
    - tab-b-trend-metric-select (dcc.Dropdown)
    - tab-b-trend-mode-toggle (dcc.RadioItems for Ribbon/Bars)
    """
    from dash import dcc, html

    return html.Div(
        className="tab-b-trend-controls",
        children=[
            html.Span("Metric", className="tab-b-trend-label"),
            dcc.Dropdown(
                id="tab-b-trend-metric-select",
                options=[{"label": m["label"], "value": m["value"]} for m in _TREND_METRICS],
                value=default_metric,
                clearable=False,
                className="tab-b-trend-metric-dropdown",
            ),
            html.Span("View", className="tab-b-trend-label"),
            dcc.RadioItems(
                id="tab-b-trend-mode-toggle",
                options=[
                    {"label": "Ribbon", "value": "ribbon"},
                    {"label": "Bars", "value": "bars"},
                ],
                value=default_mode,
                inline=True,
                className="tab-b-trend-mode-toggle",
            ),
        ],
    )


def render_trend_section(
    data,
    *,
    current_dispatch_id: str,
    selection,
    metric: str,
    mode: str,
):
    """Top-level: selector + chart wrapped in Div(id='tab-b-trend-content').

    Handles three states:
    - 0 matches → empty-state banner.
    - 1 match (only current) → single-point chart + note.
    - N>1 matches → normal chart.
    """
    from dash import html

    metric_label = next((m["label"] for m in _TREND_METRICS if m["value"] == metric),
                        metric)
    selector = render_metric_selector(default_metric=metric, default_mode=mode)

    dispatch_ids = data.trend_dispatches_for(
        current_dispatch_id, selection.task, selection.framework, selection.backend, n=10,
    )
    fingerprint = ""
    try:
        hw = data.load_hardware(current_dispatch_id) or {}
        fingerprint = hw.get("fingerprint", "")
    except Exception:  # noqa: BLE001
        fingerprint = ""

    if not dispatch_ids:
        return html.Div(
            id="tab-b-trend-content",
            children=[
                selector,
                html.Div(
                    className="tab-b-empty-state",
                    children=[
                        html.P(
                            f"Trend needs at least 1 prior dispatch matching "
                            f"{selection.task} · {selection.framework} × {selection.backend}"
                            + (f" on the same hardware ({fingerprint})." if fingerprint else "."),
                        ),
                    ],
                ),
            ],
        )

    points = compute_trend_points(
        data, dispatch_ids, selection.task, selection.framework, selection.backend, metric,
    )
    chart = render_trend_chart(points, metric_label, mode=mode)

    if len(dispatch_ids) == 1 and dispatch_ids[0] == current_dispatch_id:
        return html.Div(
            id="tab-b-trend-content",
            children=[
                selector,
                chart,
                html.P(
                    "First matching dispatch — trend will populate as more land.",
                    className="tab-b-trend-note",
                ),
            ],
        )

    return html.Div(id="tab-b-trend-content", children=[selector, chart])
```

- [ ] **Step 7.4: Run each new test individually, verify all PASS** (4 tests)

- [ ] **Step 7.5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/tabs/task_drilldown/trend.py tools/odin/valhalla/dashboard/tests/test_tab_b_trend.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Tab B trend (2/2): selector + section with empty/single-point states

render_metric_selector builds the metric Dropdown + Ribbon/Bars
RadioItems toggle. render_trend_section wires it together: walks
trend_dispatches_for, computes points, renders the chart, handles
the empty-state ('Trend needs at least 1 prior dispatch …') and the
single-point state ('First matching dispatch — trend will populate
as more land.') paths.
EOF
)"
```

---

## Task 8: `layout.py` — static slots + stores

**Files:**
- Create: `tools/odin/valhalla/dashboard/tabs/task_drilldown/layout.py`

No new test file — coverage comes via the callback tests in T9/T10. The layout's only behavior is "returns the right component IDs"; that's verified end-to-end when callbacks update the slots.

- [ ] **Step 8.1: Create `tools/odin/valhalla/dashboard/tabs/task_drilldown/layout.py`**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static layout for /<id>/task-drilldown."""

from __future__ import annotations

from dash import dcc, html

__all__ = ["build_layout"]


def build_layout(dispatch_id: str) -> html.Div:
    """Return Tab B's static layout.

    Stores carry per-page state (URL selection, trend metric, trend mode).
    Slots are empty Divs that callbacks populate after the URL has been
    parsed and the picker initialised.
    """
    return html.Div(
        id="tab-b-root",
        children=[
            dcc.Store(id="tab-b-dispatch-id", storage_type="memory", data=dispatch_id),
            dcc.Store(id="tab-b-selection", storage_type="memory", data=None),
            dcc.Store(id="tab-b-trend-metric", storage_type="memory", data="reward_final_ema"),
            dcc.Store(id="tab-b-trend-mode", storage_type="memory", data="ribbon"),
            html.Div(id="tab-b-picker"),
            html.Div(id="tab-b-curves"),
            html.Div(id="tab-b-stats"),
            html.Div(id="tab-b-trend"),
        ],
    )
```

- [ ] **Step 8.2: Smoke-test the layout against Spec 0's routing**

```
PYTHONPATH=. python3 -c "
from pathlib import Path
import json, tempfile
with tempfile.TemporaryDirectory() as td:
    runs_root = Path(td)
    (runs_root / 'd1').mkdir()
    (runs_root / 'd1' / 'dispatch.json').write_text(json.dumps({
        'schema_version': '1.3', 'dispatch_id': 'd1', 'started_at': 'x', 'ended_at': None,
        'seeds': [], 'commit_sha': '', 'fleet': [], 'jobs': [], 'skipped': [],
    }))
    from tools.odin.valhalla.dashboard.app import route_pathname
    from tools.odin.valhalla.dashboard.data import DataLayer
    component = route_pathname('/d1/task-drilldown', DataLayer(runs_root))
    ids = []
    def walk(c):
        ids.append(getattr(c, 'id', None))
        kids = getattr(c, 'children', None) or []
        if isinstance(kids, list):
            for k in kids:
                if k is not None and not isinstance(k, str):
                    walk(k)
        elif kids is not None and not isinstance(kids, str):
            walk(kids)
    walk(component)
    expected = {'tab-b-root', 'tab-b-dispatch-id', 'tab-b-selection',
                'tab-b-trend-metric', 'tab-b-trend-mode', 'tab-b-picker',
                'tab-b-curves', 'tab-b-stats', 'tab-b-trend'}
    missing = expected - set(filter(None, ids))
    assert not missing, f'missing slots: {missing}'
    print('layout OK')
"
```

Expected output: `layout OK`.

- [ ] **Step 8.3: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/tabs/task_drilldown/layout.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Tab B layout: stores + slots

build_layout returns the Tab B static layout: 4 memory-backed stores
(dispatch-id, selection, trend-metric, trend-mode) and 4 empty
content slots (picker, curves, stats, trend). Callbacks fill the
slots after the URL has been parsed.
EOF
)"
```

---

## Task 9: `callbacks.py` — init_picker + sync_url + picker_to_url

**Files:**
- Create: `tools/odin/valhalla/dashboard/tabs/task_drilldown/callbacks.py`
- Create: `tools/odin/valhalla/dashboard/tests/test_tab_b_callbacks.py`
- Modify: `tools/odin/valhalla/dashboard/tabs/task_drilldown/__init__.py` (remove the T1 try/except guard)

T1 stubbed `register()` with a try/except guard until callbacks.py existed. T9 removes the guard.

- [ ] **Step 9.1: Write the first 4 failing tests** — `tools/odin/valhalla/dashboard/tests/test_tab_b_callbacks.py`

```python
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
        "task": task, "framework": framework, "backend": backend,
        "aggregate": {}, "seeds": {}, "divergent_seeds": [],
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
```

- [ ] **Step 9.2: Run new tests, verify they FAIL** (each individually)

- [ ] **Step 9.3: Implement `tools/odin/valhalla/dashboard/tabs/task_drilldown/callbacks.py`**

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wire Tab B's callbacks against the layout."""

from __future__ import annotations

import sys

import dash
from dash import Input, Output, State

from tools.odin.valhalla.dashboard.data import DataLayer
from tools.odin.valhalla.dashboard.tabs.task_drilldown.picker import render_picker
from tools.odin.valhalla.dashboard.tabs.task_drilldown.url_state import (
    TaskSelection,
    parse_query_string,
    serialize,
)

__all__ = ["register_callbacks"]


def register_callbacks(app: dash.Dash, data: DataLayer) -> None:
    """Register Tab B's callbacks (3 standard + 2 update; update lands in T10)."""

    @app.callback(
        Output("tab-b-picker", "children"),
        Input("tab-b-dispatch-id", "data"),
        State("url", "search"),
    )
    def _init_picker(dispatch_id, search):
        if not dispatch_id:
            return dash.no_update
        try:
            return _compute_picker_children(data, dispatch_id, search or "")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] tab-b init_picker: {type(exc).__name__}: {exc}", file=sys.stderr)
            return _error_banner("Failed to render picker", exc)

    @app.callback(
        Output("tab-b-selection", "data"),
        Input("url", "search"),
    )
    def _sync_url(search):
        return _sync_url_to_selection(search or "")

    @app.callback(
        Output("url", "search"),
        Input("tab-b-row-select", "value"),
        State("url", "search"),
    )
    def _picker_to_url(value, current_search):
        new_search = _serialize_to_url(value)
        if new_search == (current_search or ""):
            return dash.no_update
        return new_search


# -- pure helpers -----------------------------------------------------------


def _compute_picker_children(data: DataLayer, dispatch_id: str, search: str):
    aggregate = data.load_aggregate(dispatch_id)
    if aggregate is None:
        from dash import html

        return html.Div(
            id="tab-b-picker",
            className="tab-b-error-banner",
            children=[
                html.Strong("Aggregate not yet generated for this dispatch"),
                " — Tab B is empty until aggregation completes.",
            ],
        )
    selected = parse_query_string(search)
    return render_picker(aggregate, selected)


def _sync_url_to_selection(search: str) -> str | None:
    sel = parse_query_string(search)
    if sel.task and sel.framework and sel.backend:
        return f"{sel.task}|{sel.framework}|{sel.backend}"
    return None


def _serialize_to_url(value: str | None) -> str:
    if not value:
        return ""
    parts = value.split("|", 2)
    if len(parts) != 3:
        return ""
    sel = TaskSelection(parts[0] or None, parts[1] or None, parts[2] or None)
    return serialize(sel)


def _error_banner(message: str, exc: Exception):
    from dash import html

    return html.Div(
        className="tab-b-error-banner",
        children=[html.Strong(message), f": {type(exc).__name__}: {exc}"],
    )
```

- [ ] **Step 9.4: Update `__init__.py`** — remove the T1 try/except guard

```python
def register(app, data):
    """Spec 0 registry hook — wire Tab B's callbacks at app startup."""
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.callbacks import register_callbacks

    register_callbacks(app, data)
```

- [ ] **Step 9.5: Run each test individually, verify all PASS** (4 tests)

Also run the full dashboard test suite to confirm Spec 0 + Spec 1 + Tab B still green:

```
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/ --tb=line --noconftest -p no:cacheprovider
```

Expected: all green (existing 111 + Tab B's so-far ~50).

- [ ] **Step 9.6: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/tabs/task_drilldown/callbacks.py tools/odin/valhalla/dashboard/tabs/task_drilldown/__init__.py tools/odin/valhalla/dashboard/tests/test_tab_b_callbacks.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Tab B callbacks (1/2): init_picker / sync_url / picker_to_url

Three standard callbacks: init_picker reads aggregate.json, parses
the URL's query string, renders the searchable dropdown with the
URL's selection pre-applied; sync_url_to_selection mirrors the URL
into a memory store so other callbacks read it; picker_to_url
serialises a dropdown value back into the URL search string. Each
helper is a free function tested directly. T1's try/except guard
in __init__ register() is removed now that callbacks.py exists.
EOF
)"
```

---

## Task 10: `callbacks.py` — update_curves_and_stats + update_trend

**Files:**
- Modify: `tools/odin/valhalla/dashboard/tabs/task_drilldown/callbacks.py`
- Modify: `tools/odin/valhalla/dashboard/tests/test_tab_b_callbacks.py`

- [ ] **Step 10.1: Append failing tests**

```python
def test_update_curves_and_stats_loads_three_seed_bundles():
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.callbacks import (
        _compute_curves_and_stats,
    )

    aggregate = _aggregate([
        {
            "task": "X", "framework": "rsl_rl", "backend": "physx",
            "aggregate": {"reward_final_ema": {"mean": 1, "std": 0, "min": 0, "max": 0, "cv_pct": 0}},
            "seeds": {
                "42": {"run_id": "rsl-rl_physx_X_seed42", "status": "completed"},
                "43": {"run_id": "rsl-rl_physx_X_seed43", "status": "completed"},
                "44": {"run_id": "rsl-rl_physx_X_seed44", "status": "completed"},
            },
            "divergent_seeds": [],
        },
    ])

    class _Data:
        def __init__(self):
            self.load_training_calls: list = []

        def load_aggregate(self, dispatch_id):
            return aggregate

        def load_training(self, dispatch_id, run_id):
            self.load_training_calls.append((dispatch_id, run_id))
            return None

    data = _Data()
    curves, stats = _compute_curves_and_stats(
        data, dispatch_id="d-1", selection_value="X|rsl_rl|physx",
    )
    assert len(data.load_training_calls) == 3
    assert getattr(curves, "id", None) is not None
    assert getattr(stats, "id", None) is not None


def test_update_curves_and_stats_returns_curves_and_stats_divs():
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.callbacks import (
        _compute_curves_and_stats,
    )

    aggregate = _aggregate([_row("X")])

    class _Data:
        def load_aggregate(self, dispatch_id):
            return aggregate

        def load_training(self, dispatch_id, run_id):
            return None

    curves, stats = _compute_curves_and_stats(
        _Data(), dispatch_id="d-1", selection_value="X|rsl_rl|physx",
    )
    # The IDs are 'tab-b-curves-content' and 'tab-b-stats-content'.
    assert getattr(curves, "id", None) == "tab-b-curves-content"
    assert getattr(stats, "id", None) == "tab-b-stats-content"


def test_update_curves_and_stats_renders_banner_when_row_missing():
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.callbacks import (
        _compute_curves_and_stats,
    )

    aggregate = _aggregate([_row("Y")])  # row "X" not in aggregate

    class _Data:
        def load_aggregate(self, dispatch_id):
            return aggregate

        def load_training(self, dispatch_id, run_id):
            return None

    curves, stats = _compute_curves_and_stats(
        _Data(), dispatch_id="d-1", selection_value="X|rsl_rl|physx",
    )
    # When row missing, both slots render an empty/banner div.
    def _walk(c):
        yield c
        kids = getattr(c, "children", None)
        if isinstance(kids, list):
            for k in kids:
                if k is not None and not isinstance(k, str):
                    yield from _walk(k)
        elif kids is not None and not isinstance(kids, str):
            yield from _walk(kids)

    text = " ".join(
        str(getattr(c, "children", "") or "")
        for c in _walk(curves)
        if isinstance(getattr(c, "children", None), str)
    ) + " " + " ".join(
        str(getattr(c, "children", "") or "")
        for c in _walk(stats)
        if isinstance(getattr(c, "children", None), str)
    )
    assert "Row not found" in text


def test_update_trend_returns_trend_div():
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.callbacks import (
        _compute_trend_children,
    )

    class _Data:
        def trend_dispatches_for(self, current, task, fw, be, n=10):
            return []

        def load_hardware(self, dispatch_id):
            return {"fingerprint": "gpu:NVIDIA-L40"}

    out = _compute_trend_children(
        _Data(), dispatch_id="d-1", selection_value="X|rsl_rl|physx",
        metric="reward_final_ema", mode="ribbon",
    )
    assert getattr(out, "id", None) == "tab-b-trend-content"


def test_update_trend_ignores_phantom_initial_calls():
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.callbacks import (
        _compute_trend_children,
    )

    class _Data:
        pass

    # No selection_value → returns the empty-state div with no Graph.
    out = _compute_trend_children(
        _Data(), dispatch_id="d-1", selection_value=None,
        metric="reward_final_ema", mode="ribbon",
    )
    graphs = []
    def _walk(c):
        yield c
        kids = getattr(c, "children", None)
        if isinstance(kids, list):
            for k in kids:
                if k is not None and not isinstance(k, str):
                    yield from _walk(k)
        elif kids is not None and not isinstance(kids, str):
            yield from _walk(kids)
    for c in _walk(out):
        if type(c).__name__ == "Graph":
            graphs.append(c)
    assert len(graphs) == 0
```

- [ ] **Step 10.2: Run new tests, verify they FAIL** (each individually)

- [ ] **Step 10.3: Modify `tools/odin/valhalla/dashboard/tabs/task_drilldown/callbacks.py`**

Add two more callbacks inside `register_callbacks`:

```python
    @app.callback(
        Output("tab-b-curves", "children"),
        Output("tab-b-stats", "children"),
        Input("tab-b-selection", "data"),
        Input("tab-b-dispatch-id", "data"),
    )
    def _update_curves_and_stats(selection_value, dispatch_id):
        if not dispatch_id:
            return dash.no_update, dash.no_update
        try:
            return _compute_curves_and_stats(data, dispatch_id, selection_value)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] tab-b curves/stats: {type(exc).__name__}: {exc}", file=sys.stderr)
            banner = _error_banner("Failed to render curves/stats", exc)
            return banner, banner

    @app.callback(
        Output("tab-b-trend", "children"),
        Input("tab-b-selection", "data"),
        Input("tab-b-dispatch-id", "data"),
        Input("tab-b-trend-metric", "data"),
        Input("tab-b-trend-mode", "data"),
    )
    def _update_trend(selection_value, dispatch_id, metric, mode):
        if not dispatch_id:
            return dash.no_update
        try:
            return _compute_trend_children(data, dispatch_id, selection_value, metric, mode)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] tab-b trend: {type(exc).__name__}: {exc}", file=sys.stderr)
            return _error_banner("Failed to render trend section", exc)

    # Two tiny store-writers (metric dropdown → store, mode toggle → store).
    @app.callback(
        Output("tab-b-trend-metric", "data"),
        Input("tab-b-trend-metric-select", "value"),
    )
    def _on_trend_metric(value):
        return value or dash.no_update

    @app.callback(
        Output("tab-b-trend-mode", "data"),
        Input("tab-b-trend-mode-toggle", "value"),
    )
    def _on_trend_mode(value):
        return value or dash.no_update
```

Also add the helper functions at module level (after `_serialize_to_url`):

```python
def _compute_curves_and_stats(data, dispatch_id: str, selection_value: str | None):
    from dash import html

    from tools.odin.valhalla.dashboard.tabs.task_drilldown.curves import render_curves
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.stats import render_stats_panel

    aggregate = data.load_aggregate(dispatch_id)
    if aggregate is None:
        banner = _banner_div("Aggregate not yet generated for this dispatch — Tab B is empty until aggregation completes.")
        return banner, banner

    if not selection_value:
        empty = _banner_div("Pick a row from the dropdown above.")
        return empty, empty

    parts = selection_value.split("|", 2)
    if len(parts) != 3:
        empty = _banner_div("Selection malformed.")
        return empty, empty
    task, framework, backend = parts

    row = next(
        (r for r in aggregate.get("rows", []) or []
         if r.get("task") == task and r.get("framework") == framework and r.get("backend") == backend),
        None,
    )
    if row is None:
        banner = _banner_div(
            f"Row not found in this dispatch: {task} · {framework} × {backend}. "
            "Pick another row from the dropdown.",
        )
        return banner, banner

    seeds = row.get("seeds") or {}
    bundles: dict[str, dict] = {}
    for seed_key, seed in seeds.items():
        run_id = seed.get("run_id")
        if not run_id:
            continue
        training = data.load_training(dispatch_id, run_id)
        if training is not None:
            bundles[seed_key] = training

    divergent = row.get("divergent_seeds", []) or []
    return render_curves(bundles, divergent_seeds=divergent), render_stats_panel(row)


def _compute_trend_children(data, dispatch_id: str, selection_value: str | None,
                            metric: str, mode: str):
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.trend import render_trend_section

    if not selection_value:
        return _banner_div("Pick a row from the dropdown to populate the trend.")
    parts = selection_value.split("|", 2)
    if len(parts) != 3:
        return _banner_div("Selection malformed.")
    task, framework, backend = parts
    selection = TaskSelection(task or None, framework or None, backend or None)
    return render_trend_section(
        data,
        current_dispatch_id=dispatch_id,
        selection=selection,
        metric=metric,
        mode=mode,
    )


def _banner_div(message: str):
    from dash import html

    return html.Div(
        id="tab-b-trend-content",
        className="tab-b-empty-state",
        children=[html.P(message)],
    )
```

> **Note:** `_banner_div` is reused for the trend slot AND as a placeholder for curves/stats; using `id="tab-b-trend-content"` for both is fine because Dash assigns the id to whichever slot the callback writes — Dash's HTML rendering deduplicates IDs only within a single rendered tree, and the curves and stats slots would only ever contain ONE banner at a time. To be safe, the test `test_update_curves_and_stats_returns_curves_and_stats_divs` already asserts the IDs `tab-b-curves-content` / `tab-b-stats-content` — so update `_compute_curves_and_stats` to wrap its result components in those specific IDs:
>
> Use `render_curves(...)` (returns `Div(id="tab-b-curves-content")`) and `render_stats_panel(...)` (returns `Div(id="tab-b-stats-content")`) directly. The banner case can use `id="tab-b-curves-content"` / `id="tab-b-stats-content"` for the two empty banners. Adjust `_banner_div` to accept an `id_` argument:

Replace `_banner_div` with this version:

```python
def _banner_div(message: str, *, id_: str = "tab-b-banner"):
    from dash import html

    return html.Div(
        id=id_,
        className="tab-b-empty-state",
        children=[html.P(message)],
    )
```

And update the curves/stats banner returns:

```python
    if aggregate is None:
        return (
            _banner_div("Aggregate not yet generated for this dispatch — Tab B is empty until aggregation completes.", id_="tab-b-curves-content"),
            _banner_div("", id_="tab-b-stats-content"),
        )
    if not selection_value:
        return (
            _banner_div("Pick a row from the dropdown above.", id_="tab-b-curves-content"),
            _banner_div("", id_="tab-b-stats-content"),
        )
    if len(parts) != 3:
        return (
            _banner_div("Selection malformed.", id_="tab-b-curves-content"),
            _banner_div("", id_="tab-b-stats-content"),
        )
    if row is None:
        return (
            _banner_div(
                f"Row not found in this dispatch: {task} · {framework} × {backend}. "
                "Pick another row from the dropdown.",
                id_="tab-b-curves-content",
            ),
            _banner_div("", id_="tab-b-stats-content"),
        )
```

And the trend-side use:

```python
    if not selection_value:
        return _banner_div("Pick a row from the dropdown to populate the trend.", id_="tab-b-trend-content")
    if len(parts) != 3:
        return _banner_div("Selection malformed.", id_="tab-b-trend-content")
```

- [ ] **Step 10.4: Run each new test individually, verify all PASS** (5 tests)

- [ ] **Step 10.5: Run the full dashboard test suite** to confirm everything green

```
PYTHONPATH=. python3 -m pytest tools/odin/valhalla/dashboard/tests/ --tb=line --noconftest -p no:cacheprovider
```

Expected: all dashboard tests pass (existing 111 + Spec 2 ~55 = ~166 total).

- [ ] **Step 10.6: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/tabs/task_drilldown/callbacks.py tools/odin/valhalla/dashboard/tests/test_tab_b_callbacks.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Tab B callbacks (2/2): curves+stats / trend / metric+mode store-writers

update_curves_and_stats: reads aggregate.json, finds the row from
the selection-store value, loads training.json bundles per seed,
calls render_curves + render_stats_panel. Banner-with-target-id
fallback when aggregate / row / selection is missing. update_trend:
calls render_trend_section with the trend-metric and trend-mode
store values. Plus two thin store-writers wiring the trend metric
dropdown / Ribbon-Bars toggle into their respective stores.
EOF
)"
```

---

## Task 11: CSS additions for Tab B

**Files:**
- Modify: `tools/odin/valhalla/dashboard/assets/style.css`

Tab B uses several new class names that don't have rules yet (`tab-b-picker-row`, `tab-b-stats-card`, `tab-b-stat-line`, `tab-b-cv-good`, `tab-b-trend-controls`, etc.). Without rules they render unstyled. Add the dark-theme rules.

- [ ] **Step 11.1: Append CSS rules** to `tools/odin/valhalla/dashboard/assets/style.css`

Add at the bottom of the file:

```css
/* ---------- Tab B — Task Drill-down ---------- */

#tab-b-root {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

/* picker row */

.tab-b-picker-row {
  display: flex;
  gap: 12px;
  align-items: center;
  background: #15191e;
  border: 1px solid #2a2e35;
  border-radius: 8px;
  padding: 12px 16px;
}
.tab-b-picker-label {
  color: #888;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.tab-b-picker-dropdown {
  flex: 1;
  max-width: 540px;
}
.tab-b-picker-dropdown .Select-control,
.tab-b-picker-dropdown .Select-menu-outer,
.tab-b-picker-dropdown .Select.is-focused > .Select-control {
  background: #0e1115;
  border-color: #2a2e35;
  color: #d0d0d0;
}
.tab-b-picker-dropdown .Select-value-label,
.tab-b-picker-dropdown .Select-input > input,
.tab-b-picker-dropdown .Select-placeholder {
  color: #d0d0d0 !important;
}
.tab-b-picker-dropdown .Select-menu-outer {
  background: #15191e;
  border-color: #2a2e35;
}
.tab-b-picker-dropdown .VirtualizedSelectOption {
  background: #15191e;
  color: #d0d0d0;
}
.tab-b-picker-dropdown .VirtualizedSelectFocusedOption {
  background: #1f242a;
}

/* curves */

#tab-b-curves-content {
  background: #15191e;
  border: 1px solid #2a2e35;
  border-radius: 8px;
  padding: 14px 18px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

/* stats */

#tab-b-stats-content {
  display: grid;
  grid-template-columns: 320px 1fr;
  gap: 16px;
}
.tab-b-stats-card {
  background: #15191e;
  border: 1px solid #2a2e35;
  border-radius: 8px;
  padding: 14px 16px;
}
.tab-b-stats-card-title {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #888;
  margin-bottom: 10px;
  padding-bottom: 8px;
  border-bottom: 1px solid #2a2e35;
}
.tab-b-stat-line {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding: 4px 0;
  font-size: 12px;
}
.tab-b-stat-label { color: #888; }
.tab-b-stat-value {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  color: #d0d0d0;
}
.tab-b-stat-value strong {
  color: #e0e0e0;
  font-weight: 600;
}
.tab-b-divergent-line {
  margin-top: 6px;
  padding-top: 8px;
  border-top: 1px solid #2a2e35;
}
.tab-b-cv {
  margin-left: 6px;
  font-size: 11px;
}
.tab-b-cv-good { color: #76b900; }
.tab-b-cv-warn { color: #ffa500; }
.tab-b-cv-bad  { color: #e16868; }

.tab-b-seeds-table-wrapper {
  background: #15191e;
  border: 1px solid #2a2e35;
  border-radius: 8px;
  overflow: hidden;
}
.tab-b-seeds-table { width: 100%; }
.tab-b-seeds-table th, .tab-b-seeds-table td {
  padding: 8px 10px;
  text-align: right;
  border-bottom: 1px solid #2a2e35;
  font-size: 11.5px;
}
.tab-b-seeds-table th:first-child,
.tab-b-seeds-table td:first-child {
  text-align: left;
}
.tab-b-seeds-table tr:last-child td { border-bottom: none; }
.tab-b-seeds-table tr:hover td { background: #1f242a; }
.tab-b-seed-id {
  font-family: ui-monospace, SFMono-Regular, monospace;
  color: #66b6ff;
}
.tab-b-mono { font-family: ui-monospace, SFMono-Regular, monospace; }
.tab-b-muted { color: #888; }

.tab-b-pill {
  display: inline-block;
  padding: 1px 6px;
  border-radius: 8px;
  font-size: 9.5px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.tab-b-seed-status-completed {
  background: rgba(80, 200, 140, 0.18);
  color: #50c88c;
  border: 1px solid rgba(80, 200, 140, 0.4);
}
.tab-b-seed-status-failed {
  background: rgba(220, 80, 80, 0.18);
  color: #e16868;
  border: 1px solid rgba(220, 80, 80, 0.4);
}

/* trend */

#tab-b-trend-content {
  background: #15191e;
  border: 1px solid #2a2e35;
  border-radius: 8px;
  padding: 14px 18px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.tab-b-trend-controls {
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
}
.tab-b-trend-label {
  color: #888;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.tab-b-trend-metric-dropdown {
  min-width: 240px;
}
.tab-b-trend-mode-toggle label {
  margin-right: 12px;
  color: #d0d0d0;
  font-size: 12px;
}
.tab-b-trend-note {
  color: #888;
  font-size: 11px;
  font-style: italic;
  margin: 0;
}

/* empty / error states */

.tab-b-empty-state {
  background: #15191e;
  border: 1px solid #2a2e35;
  border-radius: 8px;
  padding: 24px 28px;
  color: #888;
  text-align: center;
}
.tab-b-error-banner {
  background: rgba(220, 180, 80, 0.12);
  border: 1px solid rgba(220, 180, 80, 0.4);
  border-radius: 4px;
  padding: 10px 14px;
  color: #e8c068;
  font-size: 12.5px;
}
.tab-b-error-banner strong {
  color: #e8c068;
  margin-right: 6px;
}

/* Tab A → Tab B Task cell link styling */

.tab-a-task-link {
  color: #66b6ff;
  text-decoration: none;
}
.tab-a-task-link:hover {
  text-decoration: underline;
}
```

- [ ] **Step 11.2: Smoke-test the CSS load**

Restart the dashboard and verify the CSS is served:

```bash
pkill -f "valhalla/dashboard/cli.py" 2>/dev/null; true
PYTHONPATH=. python3 tools/odin/valhalla/dashboard/cli.py --runs-root odin_runs --no-browser &
sleep 3
curl -sS -o /dev/null -w "HTTP %{http_code} bytes=%{size_download}\n" http://127.0.0.1:8050/assets/style.css
```

Expected: `HTTP 200 bytes=>15000` (style.css is now larger after the Tab B additions).

- [ ] **Step 11.3: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/valhalla/dashboard/assets/style.css
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Tab B: dark-theme stylesheet additions

CSS rules for the new Tab B class names: picker row, stats card +
per-seed table (status pills, cv color codes), trend controls (metric
dropdown + ribbon/bars toggle), empty / error banner states, and the
.tab-a-task-link styling for the Task→Tab B drill-in.
EOF
)"
```

---

## Task 12: Architecture-doc change-log entry + manual smoke test

**Files:**
- Modify: `docs/odin/architecture.md`

- [ ] **Step 12.1: Manual smoke test against `20260427-141302`**

```bash
pkill -f "valhalla/dashboard/cli.py" 2>/dev/null; true
PYTHONPATH=. python3 tools/odin/valhalla/dashboard/cli.py --runs-root odin_runs --no-browser &
sleep 3
```

Open `http://127.0.0.1:8050/20260427-141302/task-drilldown` (or click into Tab A's Task cell from `http://127.0.0.1:8050/20260427-141302/dispatch-fleet`).

Confirm:
- Picker dropdown lists 5 rows (Ant / Cartpole / Humanoid / Quadcopter / Anymal-C Flat × rsl_rl × physx).
- Picking Ant: reward + ep_length curves overlay 3 seeds with the seed-palette colors (blue / green / orange).
- Stats panel: aggregate card lists 6 metrics with cv color codes + "Divergent seeds: —"; per-seed table has 3 rows × 11 columns.
- Trend section: empty-state ("Trend needs at least 1 prior dispatch matching … on the same hardware (gpu:NVIDIA-L40)") because `trend_dispatches_for` returns `[]` for this row today.
- Switch dispatch via the header dropdown to `20260424-160119` (different hardware fleet). Tab B re-routes to `/20260424-160119/task-drilldown?…` and either re-renders or shows "Row not found" banner depending on whether the same row exists.

Fix any visible issues in a small follow-up commit before T12.2.

- [ ] **Step 12.2: Update `docs/odin/architecture.md`**

Update the "Last updated" line near the top of the doc:

```diff
-**Last updated:** 2026-04-27 (Odin dashboard Tab A)
+**Last updated:** 2026-04-29 (Odin dashboard Tab B)
```

Append a new row at the END of the change-log table:

```markdown
| 2026-04-29 | Odin dashboard Tab B — Task drill-down (Spec 2 of 4) landed (`docs/superpowers/specs/2026-04-29-odin-dashboard-tab-b-task-drilldown-design.md`). New sub-package `tools/odin/valhalla/dashboard/tabs/task_drilldown/` (7 files: layout / picker / curves / stats / trend / url_state / callbacks). The tab combines per-seed reward + ep_length curve overlays (one trace per seed; divergent seeds drawn as a red dashed stroke), an aggregate stats card (mean ± std with cv% color codes) + per-seed table (11 columns including startup phase breakdown), and a cross-commit metric trend (line + ribbon by default, bar + whiskers toggle) over the last N=10 dispatches matching the row's hardware fingerprint via `data.trend_dispatches_for`. Picker is a searchable single dropdown over `aggregate.rows[]`; URL deep-link via `?task=&framework=&backend=`. Tab A's Task cell becomes a `dcc.Link` to the matching Tab B view (Spec 1 extension bundled). 9 trend metrics: 6 read directly from `aggregate.rows[].aggregate`, 3 startup phases computed on the fly from `seeds[]` via `statistics.mean / pstdev`. ~55 pure-Python tests across 6 files; total dashboard suite ~166 passing tests. No browser-based tests; visual layout source-of-truth lives in `.superpowers/brainstorm/3762718-1777450164/content/` (curve-overlay.html, stats-panel.html, trend-shape.html). | Odin dashboard Tab B |
```

- [ ] **Step 12.3: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add docs/odin/architecture.md
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Architecture doc: change-log entry for dashboard Tab B

Records Spec 2 landing (tools/odin/valhalla/dashboard/tabs/
task_drilldown/), the 7-file split (layout / picker / curves /
stats / trend / url_state / callbacks), the URL deep-link query
string, and the Spec 1 extension that turns Tab A's Task cell into
a drill-in link.
EOF
)"
```

---

## Self-review

**Spec coverage:**

| Spec section | Plan task |
|---|---|
| § Architecture / 7-file module layout | T1 (skeleton + url_state) + T3 (picker) + T4 (stats) + T5 (curves) + T6/T7 (trend) + T8 (layout) + T9/T10 (callbacks) |
| § Module boundary contract (`__init__.py`) | T1 (with try/except guard); T9 removes guard |
| § Tab A → Tab B drill-in link | T2 |
| § No new DataLayer methods | confirmed (uses Spec 0 `load_aggregate` / `load_training` / `trend_dispatches_for`) |
| § Components → url_state.py | T1 |
| § Components → picker.py | T3 |
| § Components → curves.py + _SEED_PALETTE | T5 |
| § Components → stats.py | T4 |
| § Components → trend.py (compute + chart) | T6 |
| § Components → trend.py (selector + section + empty/single-point) | T7 |
| § Components → layout.py | T8 |
| § Components → callbacks.py (3 standard) | T9 |
| § Components → callbacks.py (2 update + 2 store-writers) | T10 |
| § Data flow Flow 1 (cold mount via direct URL) | T9 + T10 (callbacks fire on mount) |
| § Data flow Flow 2 (cold mount without query string) | T9 (sync_url_to_selection + picker_to_url) |
| § Data flow Flow 3 (user picks different row) | T9 (picker_to_url) + T10 (update_curves_and_stats + update_trend) |
| § Data flow Flow 4 (metric/mode change) | T10 (store-writers + update_trend) |
| § Data flow Flow 5 (Tab A drill-in) | T2 |
| § Data flow Flow 6 (header dispatch dropdown switch) | T9 (init_picker re-runs on dispatch-id change) |
| § Data flow Flow 7 (empty / single-point trend) | T7 |
| § Error handling matrix | T9/T10 (try/except wrappers + banner helpers) |
| § Testing strategy → test_tab_b_url_state.py (9 tests) | T1 |
| § Testing strategy → test_tab_b_picker.py (8 tests) | T3 |
| § Testing strategy → test_tab_b_curves.py (8 tests) | T5 |
| § Testing strategy → test_tab_b_stats.py (13 tests) | T4 |
| § Testing strategy → test_tab_b_trend.py (13 tests) | T6 (9) + T7 (4) |
| § Testing strategy → test_tab_b_callbacks.py (9 tests) | T9 (4) + T10 (5) |
| § Tab A regression (1 new test) | T2 |
| § Implementation order preview | Tasks 1-12 closely match the spec's preview |

**Placeholder scan:** searched for "TBD", "TODO", "fill in", "<...>" — none. Every code step has concrete code; every test step has actual assertions.

**Type / signature consistency:**
- `TaskSelection(task, framework, backend)` — frozen dataclass, used identically across `url_state.py`, `picker.py`, `callbacks.py`, tests.
- `parse_query_string(search) -> TaskSelection` and `serialize(TaskSelection) -> str` — round-trip pair.
- `list_row_options(aggregate_payload) -> list[dict]` — option dicts have `label` + `value` keys.
- `render_picker(aggregate_payload, selected: TaskSelection | None)` — second arg may be `None` or a TaskSelection.
- `render_curves(bundles: dict[str, dict], *, divergent_seeds: list[str])` — bundles keyed by seed string.
- `_SEED_PALETTE` constant uses 5 colors; tests read its first 3 entries.
- `render_aggregate_card(aggregate_block, divergent_seeds)` — block matches `aggregate.rows[].aggregate` shape.
- `render_seeds_table(seeds_block)` — block matches `aggregate.rows[].seeds` shape (keyed by seed string).
- `render_stats_panel(aggregate_payload_row)` — takes the full row.
- `compute_trend_points(data, dispatch_ids, task, framework, backend, metric)` — returns `list[dict]` with `dispatch_id` / `commit_sha` / `mean` / `std` / `n_seeds_completed`.
- `render_trend_chart(points, metric_label, *, mode='ribbon')` — returns `dcc.Graph`.
- `render_trend_section(data, *, current_dispatch_id, selection, metric, mode)` — handles empty/single-point states.
- `_compute_picker_children(data, dispatch_id, search)` / `_compute_curves_and_stats(data, dispatch_id, selection_value)` / `_compute_trend_children(data, dispatch_id, selection_value, metric, mode)` — callback helpers; tested directly.
- Pipe-separated value format `"task|framework|backend"` consistent across `picker.list_row_options`, `_sync_url_to_selection`, `_serialize_to_url`, `_compute_curves_and_stats` parsing.

Plan is internally consistent and spec-complete.

---

## Execution

**Plan complete and saved to `docs/superpowers/plans/2026-04-29-odin-dashboard-tab-b-task-drilldown.md`.** Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, two-stage review, fast iteration. Same shape as Spec 0 + Spec 1.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
