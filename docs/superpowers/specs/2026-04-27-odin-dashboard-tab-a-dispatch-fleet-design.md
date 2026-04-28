# Odin Dashboard — Tab A: Dispatch & Fleet (Spec 1) Design

**Status:** approved (pending user review of this written form)
**Branch:** `antoiner/feat/odin`
**Series:** Spec 1 of 4. Spec 0 (skeleton) landed; this fills in the first real tab.

## Background

Spec 0 shipped the dashboard skeleton: `odin-dashboard` CLI, multi-dispatch landing page, header dispatch picker, three placeholder tabs (`dispatch-fleet` / `task-drilldown` / `startup`) routed via `tools/odin/valhalla/dashboard/app.py`'s registry. The placeholder text says "Coming in Spec N" and that's all you see when you click a tab.

Tab A — "Dispatch & Fleet" — is the most-used view: it answers "did this dispatch work, what's running, what failed, which hosts are alive". The user's initial requirements (locked during brainstorming):

- **Tasks queued / running / succeeded / failed**, live or static.
- **Fleet alive + hardware** (which hosts are up, what GPUs / CPUs / RAM).
- **Failures** filterable by `failure_kind`, with `failure_message` + `assigned_to` + a link to the bundle's `ssh-tail.log`.
- **Recovered + host_down events** surfaced from `dispatch.json.fleet[].last_error`.
- **Auto-poll every 5–10 s** when the dispatch is live (`ended_at == null`).

This spec turns that into a concrete implementation.

## Goal

Ship a working Tab A: when the user opens `/<id>/dispatch-fleet` they see a per-dispatch health view that updates every 5 s while the dispatch is live, lets them scan the fleet + jobs at a glance, drill into individual failures inline, and load the last 50 lines of `ssh-tail.log` on demand.

## Non-goals

- Tabs B (Task drill-down) or C (Startup deep-dive) — Specs 2/3.
- Cross-dispatch comparison views inside Tab A — Tabs B/C have the trend axes; Tab A is single-dispatch.
- Live tail of `ssh-tail.log` — file is loaded once per click, not auto-refreshed.
- Browser-based E2E tests — layout-tree + pure-function tests cover the surface.
- Authentication, persistence beyond `dcc.Store` (memory storage), shareable URLs beyond what Spec 0 already provides.

## Locked decisions (from brainstorming Q1–Q7)

| Q | Decision |
|---|---|
| Q1 — Layout | Vertical stack: header strip → fleet table → jobs section. Single column. |
| Q2 — Polling | Always-on `dcc.Interval` at 5 s; callbacks no-op when `ended_at != null`. |
| Q3 — Failure detail | Inline expansion under failed rows; `<pre>` block for ssh-tail. No modal. |
| Q4 — Jobs table | Plain HTML table, scrolling. No Dash DataTable, no pagination. |
| Q5 — Fleet table | One row per host, 8 inline columns. Layout matches `fleet-row-vs-card.html` mockup option A. |
| Q6 — Jobs filters | B+C: status dropdown + failure-kind dropdown + free-text task-id search. Failure-kind also visible as a column on each row. |
| Q7 — Failures section | Drop the dedicated table. Add a click-to-filter pill strip in the header (`Failures: 6 (hugin_crash: 2, gpu_lost: 1, …)`). Pill click sets the kind filter on the jobs table AND updates the kind-dropdown's display value. |

**Visual reference:** `.superpowers/brainstorm/2312694-1777385893/content/fleet-row-vs-card.html` and `jobs-table.html` are the visual source of truth — the implementer should match their structure (status pill colors, kind pill colors, table styling, dark theme).

## Architecture

Tab A is one new sub-package under Spec 0's `tabs/`. **No edits to `app.py`, `data.py`, or `cli.py`** — Spec 0's registry pattern picks it up via `importlib.import_module` when `tab_id == "dispatch-fleet"`.

**One small Spec 0 enhancement:** the registry's `_discover_tabs()` (in `app.py`) currently only calls `tab_module.render(dispatch_id, tab_id)`. Tab A also needs callbacks registered at app-startup time. We extend the registry by ~5 lines: if `tab_module` has a `register(app, data)` function, call it once during `create_app`. This is additive and backward-compatible (Spec 0's placeholder doesn't have `register`, so it's never called).

### Module layout

```
tools/odin/valhalla/dashboard/tabs/dispatch_fleet/
├── __init__.py            # re-exports `register` + `render` for the registry
├── layout.py              # builds the static layout (header, fleet, jobs sections)
├── header.py              # totals strip + failure-pill ribbon (pure render functions)
├── fleet_table.py         # fleet row builder (pure render function)
├── jobs_table.py          # jobs row builder + inline-expand row builder
├── filters.py             # status/kind/task-text filter logic (pure functions)
├── callbacks.py           # registers the 6 callbacks against the layout
└── ssh_tail.py            # lazy-load + render ssh-tail.log helper

tools/odin/valhalla/dashboard/tests/
├── test_tab_a_header.py            # totals + failure-pill rendering
├── test_tab_a_fleet_table.py       # row builder + hardware lookup integration
├── test_tab_a_jobs_table.py        # row builder + filtered rendering + expand row
├── test_tab_a_filters.py           # status/kind/task-text filtering, edge cases
├── test_tab_a_ssh_tail.py          # ssh-tail file loader (truncation, missing file)
└── test_tab_a_callbacks.py         # callback wiring (one test per callback, calls helpers directly)
```

**Why split this way:**
1. `layout.py` is the only file that calls `dash.html.*` and friends.
2. `header.py` / `fleet_table.py` / `jobs_table.py` each take a parsed `dispatch.json` dict (+ optional hardware) and return a row-list. Trivially unit-testable.
3. `filters.py` is pure data logic — no Dash imports.
4. `callbacks.py` registers the callbacks; tests exercise the callback functions directly.
5. `ssh_tail.py` is pure I/O; tested with tmp_path.

### Module boundary contract

```python
# tabs/dispatch_fleet/__init__.py

from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.layout import build_layout
from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.callbacks import register_callbacks


def render(dispatch_id: str, tab_id: str):
    """Spec 0 registry hook — return the static layout for this tab."""
    return build_layout(dispatch_id)


def register(app, data):
    """Spec 0 registry hook — wire the dcc.Interval callbacks at app startup."""
    register_callbacks(app, data)
```

## Components

### `layout.py`

```python
def build_layout(dispatch_id: str) -> html.Div:
    """Static layout for /<id>/dispatch-fleet.

    All dynamic content lives in slots (id="tab-a-..."); callbacks fill them.
    """
```

Returns:

```python
Div(id="tab-a-root", children=[
    dcc.Interval(id="tab-a-tick", interval=5_000, n_intervals=0),
    dcc.Store(id="tab-a-dispatch-id", data=dispatch_id),
    dcc.Store(id="tab-a-failure-filter", data=None),
    dcc.Store(id="tab-a-expanded-run-ids", data=[]),
    dcc.Store(id="tab-a-ssh-tail-store", data={}),
    Div(id="tab-a-header"),
    Div(id="tab-a-fleet-table"),
    Div(id="tab-a-jobs-section", children=[
        # filter row (status dropdown, kind dropdown, task input) + tab-a-jobs-rows slot
    ]),
])
```

### `header.py`

```python
def render_header(dispatch_payload: dict) -> html.Div:
    """Build the header strip: dispatch metadata, live pill, totals, failure pills."""
```

Reads `dispatch_id`, `commit_sha`, `started_at`, `ended_at`, `fleet[]`, `jobs[]`. Returns `Div(id="tab-a-header-content")` containing:

- **Title row:** `"Dispatch <id> · commit <short_sha> · <N> hosts"` (short_sha = first 7 chars; omit segment if empty).
- **Live/Done pill:** `Span(className="tab-a-live-pill", text="● Live")` (orange) when `ended_at is None`; `Span(className="tab-a-done-pill", text="✓ Done")` (grey) otherwise.
- **Totals row:** `"<total> total · <completed> completed · <failed> failed · <pending> pending · <skipped> skipped"`.
- **Failure-pills row** (only when there's at least one failure): `"Failures: <N>  "` followed by per-kind clickable pills. Each pill has `id={"type": "tab-a-failure-pill", "kind": "<kind>"}` (Dash pattern-matching id) and text `"<kind>: <count>"`.

### `fleet_table.py`

```python
def render_fleet_table(dispatch_payload: dict, hardware_payload: dict | None,
                       fallback_lookup: Callable[[str], HardwareInfo | None]) -> html.Div:
    """Build the fleet table — one row per host, 8 inline columns."""
```

Columns (in order): **Host** (IP, monospace), **Hostname**, **Status** (pill: idle=grey, busy=orange, down=red), **Current run** (truncated run_id link to `…` for now — link target lands when Tab B is implemented), **GPU** (`name + " · " + mem_gb + " GB"`), **CPU** (`cpu_name + " ×" + cpu_count`), **RAM** (`ram_gb + " GB"`), **Last event** (recovery pill or `—`).

Hardware lookup order per host:
1. `hardware_payload["hosts"][host]` (per-dispatch `hardware.json`, written by Spec 0 Task 7).
2. `fallback_lookup(host)` → `data.lookup_hardware(host)` (cross-dispatch fallback).
3. `—` for the GPU/CPU/RAM cells.

`fallback_lookup` is invoked at most once per host per render (cached in a local dict). Status pill / current run / last event are independent of hardware.

`Last event` rules:
- `last_error == "gpu_lost: recovered"` → green recovered pill, label `"gpu_lost: recovered"`.
- `last_error` starts with `"gpu_lost: recovery_failed"` → red recovery-failed pill; full string in `title=` (hover tooltip).
- `last_error is None` → `—`.
- Anything else → render the string as plain text (forward-compat for future event types).

### `jobs_table.py`

```python
def render_jobs_section(dispatch_payload: dict, *,
                        status_filter: list[str] | None = None,
                        kind_filter: list[str] | None = None,
                        task_text: str = "",
                        expanded_run_ids: set[str] | None = None,
                        ssh_tail_store: dict[str, list[str]] | None = None) -> html.Div:
    """Build the jobs section: filter row + table + inline expand rows for failed jobs."""
```

Returns `Div(id="tab-a-jobs-section", children=[filter_row, table])`.

**Filter row** (rendered once during initial layout, replaced on filter changes via callbacks):
- Status dropdown (`dcc.Dropdown`, multi=True, options: pending / running / completed / failed).
- Failure-kind dropdown (multi=True, options: hugin_crash / gpu_lost / hugin_malformed_bundle / timeout / preset_unsupported / infrastructure).
- Task-text input (`dcc.Input`, type=text, placeholder="filter task…").

**Columns** (Q6 = B+C):
1. Task (job.task_id).
2. Framework × Backend (e.g., `"rsl_rl × physx"`, monospace).
3. Seed.
4. Status (pill).
5. Failure (kind pill if failed, `—` otherwise).
6. Host (assigned_to, monospace; `—` if pending).
7. Started / Ended (relative time, e.g. `"3m ago"`; both fields when `ended_at` set, else just started).

**Attempts badge:** when `attempts > 1`, append a small `×<N>` badge to the status pill (orange).

**Failed rows expand** (when run_id is in `expanded_run_ids`):
- Insert `<tr class="tab-a-expand-row">` directly after the data row.
- Contents: `failure.kind` pill, `Attempts: <N>`, full `failure.message` (preserving newlines via `<pre>`), and a button `id={"type":"tab-a-ssh-tail-button","run_id":"<run_id>"}` labeled `"▸ Show ssh-tail.log (last 50 lines)"`.
- After ssh-tail click, if `ssh_tail_store[run_id]` is set, render those lines as a `<pre>` block under the button. (`ssh_tail_store` is passed in from the callback's input.)
- The expand-toggle button has `id={"type":"tab-a-expand-toggle","run_id":"<run_id>"}` and lives on the data row (e.g., a chevron icon at the start of the row).

**Empty-state messages:**
- All filters set + no rows match → `Div(id="tab-a-jobs-empty")` with text `"No jobs match the current filters."` + a `Clear` button that resets all three filter inputs and the failure-filter store.
- `dispatch.json.jobs == []` (no jobs queued at all) → `Div(id="tab-a-jobs-empty-zero")` with text `"No jobs queued for this dispatch yet."` and no Clear button.

### `filters.py`

```python
def filter_jobs(jobs: list[dict], *,
                status_filter: list[str] | None = None,
                kind_filter: list[str] | None = None,
                task_text: str = "") -> list[dict]:
    """Apply the three filters in sequence; return the filtered job list.

    Empty / None filter = pass-through.
    task_text matches case-insensitive substring on job["task_id"].
    """
```

Pure data function; zero Dash imports. Used by both initial render and the on-tick `update_jobs` callback.

### `ssh_tail.py`

```python
SSH_TAIL_DEFAULT_LINES = 50
SSH_TAIL_MAX_BYTES = 64 * 1024


def load_ssh_tail(runs_root: Path, dispatch_id: str, run_id: str,
                  lines: int = SSH_TAIL_DEFAULT_LINES) -> list[str]:
    """Read the last N lines of <runs_root>/<dispatch_id>/<run_id>/logs/ssh-tail.log.

    Returns:
        The lines (without trailing newlines). Empty list if file missing.
        At most SSH_TAIL_MAX_BYTES bytes are read; truncation marker prepended if so.
    """
```

Implementation:
- Path: `runs_root / dispatch_id / run_id / "logs" / "ssh-tail.log"`.
- Missing file → return `[]`.
- File ≤ `SSH_TAIL_MAX_BYTES` → `open(path).readlines()[-lines:]`.
- File > `SSH_TAIL_MAX_BYTES` → seek to `-SSH_TAIL_MAX_BYTES` from end; decode (errors="ignore"); split on `\n`; drop the first partial line; take last `lines`; prepend a sentinel string `"… (truncated to last 64 KB) …"`.
- `OSError` (e.g., permission denied) → return `[]`.

### `callbacks.py`

```python
def register_callbacks(app: dash.Dash, data: DataLayer) -> None:
    """Register the 6 callbacks for Tab A.

    All callbacks gate on the URL implicitly: their outputs are tab-a-* IDs that
    only exist when the tab is mounted, so Dash skips them when the user is on
    a different tab.
    """
```

Each callback's body is implemented as a free function (e.g., `_compute_header_children`) so tests can call it directly without spinning up Dash.

| # | Callback | Inputs | Output | Helper |
|---|---|---|---|---|
| 1 | `update_header` | tick, dispatch-id | `tab-a-header.children` | `_compute_header_children(payload)` → `header.render_header(...)` |
| 2 | `update_fleet` | tick, dispatch-id | `tab-a-fleet-table.children` | `_compute_fleet_children(payload, hw, fallback)` → `fleet_table.render_fleet_table(...)` |
| 3 | `update_jobs` | tick, dispatch-id, status-filter, kind-filter, task-text, failure-filter store, expanded-run-ids store, ssh-tail store | `tab-a-jobs-rows.children` | `_compute_jobs_children(payload, **filters_and_state)` → `filters.filter_jobs(...)` + `jobs_table.render_jobs_section(...)` |
| 4 | `toggle_expand_row` | pattern-matched click on `{"type":"tab-a-expand-toggle","run_id":ALL}` | `tab-a-expanded-run-ids.data` | `_toggle_run_id(current_set, run_id)` |
| 5 | `load_ssh_tail` | pattern-matched click on `{"type":"tab-a-ssh-tail-button","run_id":ALL}` | `tab-a-ssh-tail-store.data` | `_load_tail(store, runs_root, dispatch_id, run_id)` → calls `ssh_tail.load_ssh_tail` |
| 6 | `failure_pill_click` | pattern-matched click on `{"type":"tab-a-failure-pill","kind":ALL}` | `tab-a-failure-filter.data` AND `tab-a-kind-dropdown.value` | `_handle_pill_click(kind)` returns `(kind, [kind])` |

**Phantom-click guard.** Dash fires pattern-matched callbacks at app startup with `n_clicks=0` for every matched element. Callbacks 4/5/6 check `n_clicks and n_clicks > 0`; otherwise return `dash.no_update`.

**`update_fleet` workflow:**
1. `payload = data.load_dispatch(dispatch_id)` (always-fresh per Spec 0 caching policy).
2. `hw = data.load_hardware(dispatch_id)` (per-dispatch hardware.json; `None` for pre-feature dispatches).
3. `fallback = data.lookup_hardware` (cross-dispatch fallback function).
4. `fleet_table.render_fleet_table(payload, hw, fallback)`.

**`update_jobs` workflow:**
1. Read `dispatch.json` (uncached, per Spec 0).
2. If `failure-filter store` is set, prepend its value to `kind_filter`.
3. `filtered = filters.filter_jobs(jobs, status_filter, kind_filter, task_text)`.
4. `jobs_table.render_jobs_section(filtered, expanded_run_ids, ssh_tail_store)`.

## Data flow

### Flow 1 — Cold mount (user navigates to `/<id>/dispatch-fleet`)

```
app.py route → registry → tabs.dispatch_fleet.render(dispatch_id, "dispatch-fleet")
  → layout.build_layout(dispatch_id)
  → returns layout with empty slots + dcc.Stores + dcc.Interval(interval=5000, n_intervals=0)
Dash mounts layout → all six Inputs fire once → all six callbacks run
  → update_header reads dispatch.json once → fills tab-a-header
  → update_fleet reads dispatch.json + hardware.json (+ fallback per host) → fills tab-a-fleet-table
  → update_jobs reads dispatch.json + filters (default: empty) → fills tab-a-jobs-rows
  → toggle_expand_row / load_ssh_tail / failure_pill_click: no-op on first fire (n_clicks=0)
```

Three reads of `dispatch.json` in cold path; intentional, microsecond cost, Spec 0 caching policy is "always fresh".

### Flow 2 — Live tick (every 5 s)

```
dcc.Interval fires → all 3 "update_*" callbacks re-fire (filter stores unchanged → no extra fan-out)
  → re-read dispatch.json
  → re-render header / fleet / jobs-rows
  → DOM diff is tiny in steady state
```

When `ended_at` flips from `None` to a value:
- Header re-renders the live pill grey ("✓ Done"), totals settle.
- Subsequent ticks still fire; output is stable.
- `data.invalidate(dispatch_id)` called once on the transition (Spec 0 stub; future caching specs make it meaningful).

### Flow 3 — User changes a filter

```
User picks "Failed" in Status dropdown
  → tab-a-status-filter.value updates
  → update_jobs re-fires (other Inputs unchanged)
  → filters.filter_jobs(jobs, status_filter=["failed"], ...) returns subset
  → jobs_table.render_jobs_section(...) builds new rows
  → tab-a-jobs-rows.children replaced
Other callbacks (header, fleet) do not re-render — their Inputs didn't change.
```

### Flow 4 — User clicks a failure pill in the header

```
Click on Pill(id={"type":"tab-a-failure-pill","kind":"hugin_crash"})
  → failure_pill_click fires
  → reads pattern-matched Input.id["kind"] = "hugin_crash"
  → returns ("hugin_crash", ["hugin_crash"])
  → tab-a-failure-filter.data <- "hugin_crash"
  → tab-a-kind-dropdown.value <- ["hugin_crash"]  (UI now reflects the filter)
  → both are Inputs to update_jobs → jobs filter
```

### Flow 5 — User expands a failed row

```
Click on Button(id={"type":"tab-a-expand-toggle","run_id":"rsl-rl_..."})
  → toggle_expand_row fires
  → reads tab-a-expanded-run-ids Store (set of run_ids)
  → toggles the run_id (add if missing, remove if present)
  → writes back to Store
  → update_jobs re-fires (Store is one of its Inputs)
  → jobs_table re-renders rows; for each run_id in expanded_run_ids of status="failed",
    inserts an expand <tr> after the data <tr>
ssh-tail not loaded yet — that's a separate click.
```

### Flow 6 — User clicks "Show ssh-tail" inside an expanded row

```
Click on Button(id={"type":"tab-a-ssh-tail-button","run_id":"rsl-rl_..."})
  → load_ssh_tail fires
  → ssh_tail.load_ssh_tail(runs_root, dispatch_id, run_id, lines=50)
  → returns list[str]
  → writes {run_id: lines} into tab-a-ssh-tail-store.data (keyed dict)
  → update_jobs re-fires; jobs_table reads ssh_tail_store
    and renders <pre> block inside the expand <tr>
File missing → empty list → render shows "ssh-tail.log not found at <relative-path>".
```

Tail content cached in store for the page session (Spec 0 lives in memory; reload clears it). Failed jobs are terminal so the file is frozen — re-reading on each tick would always return the same content. A "Refresh tail" button can land later if needed.

### Flow 7 — Header dropdown switches dispatches (Spec 0 already handles)

```
User picks a different dispatch in Spec 0's header dropdown
  → URL changes to /<other-id>/dispatch-fleet
  → app.py routing rebuilds the page-content tree
  → tabs.dispatch_fleet.render() called fresh with new dispatch_id
  → all stores reset; cold-mount flow re-runs (Flow 1)
```

Tab A doesn't need to handle dispatch switching internally.

## Error handling

| Failure mode | Where | Behavior |
|---|---|---|
| `dispatch.json` malformed (`json.JSONDecodeError`) | `data.load_dispatch` raises during a callback | Callback catches; renders an inline error banner in the relevant slot. Other slots unaffected. Next tick retries. |
| `dispatch.json` truncated (race with runner's atomic-write) | Same as above | Same: catch, banner, retry. The runner uses temp+rename so the race window is tiny. |
| `hardware.json` absent (pre-feature dispatch) | `data.load_hardware` returns `None` | Fleet table calls `data.lookup_hardware(host)` for each host as fall-back. Both miss → GPU/CPU/RAM cells render `—`. Status / current run / last event still render. |
| `hardware.json` malformed | `data.load_hardware` raises | Caught at the callback boundary; banner; fall-back lookup proceeds. |
| `ssh-tail.log` missing | `ssh_tail.load_ssh_tail` returns `[]` | Expand-row renders `"ssh-tail.log not found at <relative-path>"` instead of the `<pre>` block. |
| `ssh-tail.log` huge (> 64 KB) | `ssh_tail.load_ssh_tail` truncates | Renders truncation marker line prepended to the lines. |
| `ssh-tail.log` permission denied | `OSError` | Caught; returns `[]`; renders `"ssh-tail.log: permission denied"`. |
| User clicks failure pill for a kind that no longer has matches (e.g., the only `gpu_lost` row was retried successfully on next poll) | Filter is set; `update_jobs` filters to zero rows | Empty-state message in jobs slot: `"No jobs match the current filters."` + Clear button. |
| `dispatch.json` exists but `jobs[]` is empty | All cold | Header renders totals as 0. Fleet renders normally. Jobs section renders the no-jobs empty-state. |
| Tick fires while page is unmounted | `dcc.Interval` keeps firing per Dash semantics, but page-content has no `tab-a-*` ids | Dash silently drops these (output IDs not in DOM). No work, no error. |
| Failed-row expansion clicked but `failure.message` is `None` | Pure-render edge case | Expand row renders `"(no failure message recorded)"` and shows the kind pill + attempts count + ssh-tail button only. |
| Header dispatch dropdown switches mid-tick | URL changes; new path triggers full page-content re-render | All Stores reset (memory storage); fresh cold mount of Tab A for the new dispatch. No state bleeds between dispatches. |
| Phantom click at app startup (`n_clicks=0`) | Pattern-matched callbacks fire | Callbacks 4/5/6 gate on `n_clicks > 0` and return `dash.no_update`. |
| `data.load_hardware` raises non-`json.JSONDecodeError` (disk error, etc.) | Bubbles up to callback | Caught; banner; fall-back lookup proceeds. |

**Banner shape:** `Div(className="tab-a-error-banner", children=[...])`. Plain CSS (yellow border, dark yellow background). Text format: `"<friendly description>: <error class>: <message>"`. At most one banner per slot.

**No silent failures.** Anything caught at the callback boundary surfaces a banner. The user always knows when something went wrong.

**Logging:** Each caught exception writes a `[WARNING]` line to `sys.stderr` (the Dash dev server's terminal). No file-based dashboard log for v1.

## Testing strategy

Six pure-Python pytest files. Run with `PYTHONPATH=. python3 -m pytest --noconftest -p no:cacheprovider`. Tests are fast (sub-second total) — no Dash server, no Isaac Sim, no browser.

### `test_tab_a_header.py`

| Test | Behavior |
|---|---|
| `test_header_live_pill_when_ended_at_null` | `ended_at=None` → output contains a child with class `tab-a-live-pill` and text containing `"Live"` |
| `test_header_done_pill_when_ended_at_set` | `ended_at="2026-04-27T..."` → child with class `tab-a-done-pill` and text containing `"Done"` |
| `test_header_totals_match_jobs_array` | Fabricated jobs (3 completed, 2 failed, 1 pending, skipped=2) → totals strings match exactly |
| `test_header_failure_pills_grouped_by_kind` | Jobs with kinds `[hugin_crash, hugin_crash, gpu_lost, preset_unsupported]` → renders three pills with text containing `"hugin_crash: 2"`, `"gpu_lost: 1"`, `"preset_unsupported: 1"` |
| `test_header_failure_pill_ids_use_pattern_matching` | Each failure pill has id of shape `{"type": "tab-a-failure-pill", "kind": "<kind>"}` |
| `test_header_no_failure_pills_when_no_failures` | All-completed dispatch → no failure-pills row rendered |
| `test_header_short_commit_sha` | Full SHA `"abc123def456"` → header text shows `"abc123d"` (first 7 chars) |
| `test_header_handles_missing_commit_sha` | `commit_sha=""` → header omits the commit segment |

### `test_tab_a_fleet_table.py`

| Test | Behavior |
|---|---|
| `test_fleet_renders_one_row_per_host` | Two-host fleet → tbody has exactly 2 `<tr>` children |
| `test_fleet_status_pill_idle_busy_down` | Each status maps to expected pill class |
| `test_fleet_current_run_link_for_busy_host` | `current_run_id="rsl-rl_..."` → cell contains `<a href="…">` with truncated text |
| `test_fleet_current_run_dash_when_idle` | `current_run_id=None` → cell renders `"—"` |
| `test_fleet_hardware_from_hardware_json` | `hardware_payload.hosts[host]` populated → GPU/CPU/RAM cells use those values |
| `test_fleet_hardware_falls_back_to_lookup` | `hardware_payload=None` → `fallback_lookup` called once per host; cells use returned `HardwareInfo` |
| `test_fleet_hardware_dash_when_unknown` | Both miss → GPU/CPU/RAM cells = `"—"` |
| `test_fleet_last_event_recovered_pill` | `last_error="gpu_lost: recovered"` → renders green recovered-pill |
| `test_fleet_last_event_recovery_failed_pill` | `last_error="gpu_lost: recovery_failed (...)"` → renders red recovery-failed-pill, hover-title contains the detail |
| `test_fleet_last_event_dash_when_no_error` | `last_error=None` → cell = `"—"` |
| `test_fleet_fallback_lookup_called_at_most_once_per_host` | Fabricated dispatch with 5 jobs assigned to 2 hosts; lookup called twice |

### `test_tab_a_jobs_table.py`

| Test | Behavior |
|---|---|
| `test_jobs_renders_filter_row_with_three_controls` | Output contains the status dropdown, kind dropdown, and task-text input |
| `test_jobs_renders_one_row_per_job` | 5 jobs → 5 data `<tr>`s + 1 header `<tr>` |
| `test_jobs_status_pill_per_status` | Each status → expected pill class |
| `test_jobs_failure_kind_column_filled_for_failed_only` | Failed row → kind pill in Failure column; running/pending/completed → `"—"` |
| `test_jobs_relative_started_at` | `started_at` set → cell text matches `r"\d+[mhsd] ago"` (loose pattern; we just check there's a relative-time string) |
| `test_jobs_attempts_badge_only_when_gt_1` | attempts=1 → no badge; attempts=2 → `"×2"` badge after status pill |
| `test_jobs_filter_status` | `filter_jobs(status_filter=["failed"])` → only failed pass |
| `test_jobs_filter_kind` | `kind_filter=["gpu_lost"]` → only failed rows whose `failure.kind == "gpu_lost"` pass |
| `test_jobs_filter_task_text_substring_case_insensitive` | task_text="anymal" → passes any task_id containing `"Anymal"` |
| `test_jobs_filters_compose` | All three filters together → AND-combined |
| `test_jobs_expanded_row_for_failed_in_expanded_set` | `expanded_run_ids={run_id}` → expand `<tr>` follows that data row with the failure message |
| `test_jobs_expanded_row_not_rendered_when_collapsed` | `expanded_run_ids=set()` → no expand row |
| `test_jobs_expanded_row_ssh_tail_button_present` | Expanded failed row contains a button with id `{"type":"tab-a-ssh-tail-button","run_id":"…"}` |
| `test_jobs_expanded_row_ssh_tail_lines_rendered` | `ssh_tail_store={"<rid>": [...]}` → expand row contains a `<pre>` with each line |
| `test_jobs_empty_state_when_filters_match_nothing` | All jobs filtered out → empty-state Div with `id="tab-a-jobs-empty"`, text `"No jobs match"`, Clear button |
| `test_jobs_empty_state_when_dispatch_has_no_jobs` | `jobs=[]` → different empty-state text: `"No jobs queued for this dispatch yet."`, no Clear button |

### `test_tab_a_filters.py`

| Test | Behavior |
|---|---|
| `test_no_filters_returns_all` | All filter args None/empty → output == input |
| `test_status_filter_single` | `status_filter=["failed"]` → only failed jobs |
| `test_status_filter_multi` | `status_filter=["completed", "failed"]` → both kinds pass |
| `test_kind_filter_passes_failed_only` | `kind_filter=["hugin_crash"]` → completed/running rows are filtered out (their `failure` is None) |
| `test_task_text_substring` | `task_text="ant"` matches `"Isaac-Ant-Direct-v0"` and `"Isaac-Velocity-Flat-Anymal-..."` |
| `test_task_text_empty_string` | `task_text=""` → no filtering on task |
| `test_task_text_case_insensitive` | `task_text="ANT"` → same matches as `"ant"` |
| `test_combined_filters_intersect` | All three set; only rows matching all three pass |

### `test_tab_a_ssh_tail.py`

| Test | Behavior |
|---|---|
| `test_load_ssh_tail_full_file_under_threshold` | Write 10-line file → `load_ssh_tail(lines=50)` returns all 10 lines |
| `test_load_ssh_tail_returns_last_n_lines` | Write 100-line file → `load_ssh_tail(lines=10)` returns the last 10 |
| `test_load_ssh_tail_returns_empty_when_file_missing` | No file → returns `[]` |
| `test_load_ssh_tail_truncates_huge_file` | Write 200 KB file → returns lines from the last 64 KB only; first element of returned list is the truncation marker line |
| `test_load_ssh_tail_handles_partial_first_line_when_seeking` | Truncated read starts mid-line → first partial line dropped |
| `test_load_ssh_tail_returns_empty_on_permission_error` | Patch `open()` to raise `PermissionError` → returns `[]` |

### `test_tab_a_callbacks.py`

| Test | Behavior |
|---|---|
| `test_update_header_callback_returns_header_div` | Pass dispatch_payload; assert returned Div has id `"tab-a-header-content"` |
| `test_update_fleet_callback_invokes_data_layer` | Pass DataLayer-stub; callback calls `load_dispatch`, `load_hardware`, then `lookup_hardware` for hosts missing from hardware_payload |
| `test_update_jobs_callback_applies_filters` | Pass status="failed", kind="hugin_crash", task="anymal" → filtered rows match `filter_jobs(...)` |
| `test_update_jobs_callback_uses_failure_filter_store` | Failure-filter store data="gpu_lost" → kind_filter computed = `["gpu_lost"]` |
| `test_toggle_expand_row_adds_then_removes` | Initial set empty + click run_id "X" → set is `["X"]`; click again → set is `[]` |
| `test_toggle_expand_row_ignores_phantom_click` | n_clicks=0 → returns `dash.no_update` |
| `test_load_ssh_tail_callback_writes_store` | Click on run_id "Y"; ssh_tail returns `["a", "b"]` → output store has `{"Y": ["a", "b"]}` |
| `test_load_ssh_tail_callback_ignores_phantom_click` | n_clicks=0 → no_update |
| `test_failure_pill_click_writes_store_and_dropdown` | Click on pattern-matched id with kind="gpu_lost" → returns `("gpu_lost", ["gpu_lost"])` (the store value AND the kind-dropdown value) |

### Coverage targets

- `header.py` ≥ 95% (pure render).
- `fleet_table.py` ≥ 95% (pure render).
- `jobs_table.py` ≥ 90% (denser branching for expansions).
- `filters.py` ≥ 100% (small + pure).
- `ssh_tail.py` ≥ 95%.
- `callbacks.py` ≥ 85% (some `dash.callback` plumbing isn't worth covering).

**Total new tests:** ~50 across 6 files. All run in well under a second combined. Plus the existing 49 Spec 0 tests remain green — regression check.

**No browser-based tests.** Layout-tree assertions + pure-function tests cover behavior; visual rendering is verified against the existing mockup HTML files (`fleet-row-vs-card.html`, `jobs-table.html`) — those are the visual-design source of truth.

## Implementation order (preview — full plan in writing-plans phase)

Approximate task chain:

1. Spec 0 registry enhancement: `_discover_tabs` also calls `tab_module.register(app, data)` if present. Tiny change, 1 test added.
2. `__init__.py` + `layout.py` skeleton (just the slots; tests assert the slot ids exist).
3. `header.py` + `test_tab_a_header.py`.
4. `fleet_table.py` + `test_tab_a_fleet_table.py`.
5. `filters.py` + `test_tab_a_filters.py`.
6. `jobs_table.py` (rendering only, no expand) + half of `test_tab_a_jobs_table.py`.
7. `jobs_table.py` expand-row support + remaining `test_tab_a_jobs_table.py`.
8. `ssh_tail.py` + `test_tab_a_ssh_tail.py`.
9. `callbacks.py` (header / fleet / jobs / failure-pill) + corresponding tests.
10. `callbacks.py` (toggle_expand_row / load_ssh_tail) + corresponding tests.
11. Architecture-doc change-log entry.
12. Visual smoke test: launch dashboard against `odin_runs/`, verify Tab A renders both the live `20260428-133931` dispatch and the failed `20260424-160119` dispatch.

## Non-functional requirements

- **No new system or pip dependencies.** Tab A uses dash/plotly/pandas (already hard deps from Spec 0).
- **Sub-second total test time.** Six pytest files, ~50 tests, all pure-Python.
- **Five-second poll cadence is the only background work.** No threads, no timers, no background processes.
- **No browser automation.** Visual verification is via the saved mockup HTML files + manual smoke test in step 12.
- **Schema:** No `dispatch.json` / `hardware.json` schema changes. Reads existing fields only.
