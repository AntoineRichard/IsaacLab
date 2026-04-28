# Odin Dashboard — Skeleton (Spec 0) Design

**Status:** approved (pending user review of this written form)
**Branch:** `antoiner/feat/odin`
**Series:** Spec 0 of 4 (skeleton). Specs 1/2/3 add Tabs A/B/C respectively.

## Background

Odin's dispatcher (`odin-dispatch`) and aggregator (`odin-aggregate`) produce a tree under `odin_runs/<dispatch_id>/` containing:

- `dispatch.json` — schema 1.3, job + fleet state, includes `gpu_lost` failures and `recovered` / `host_down` `last_error` strings.
- `aggregate.json` — schema 1.0, per-`(task, framework, backend)` rows with seed-level metric blocks; top-level `failures[]` and `totals`.
- Per-bundle `<run_id>/training.json` — 1000-pt `series_per_iter` for reward/ep_length/iter_time, plus `hardware`, `versions`, `runtime`.
- Per-bundle `<run_id>/startup.json` — phase timings, hardware, versions.
- `preflight.json` — pre-dispatch host checks (no hardware).
- `fleet.yaml.snapshot` — the fleet config in effect.

Today, reading any of this requires `jq` or a Python REPL. The user wants a browser-based dashboard to read dispatch results: per-dispatch health view, per-task drill-down with training curves, per-startup attribution, all with a cross-commit trend axis filtered to matching hardware.

## Decomposition

The full feature splits into four sequential specs:

- **Spec 0 (this doc):** dashboard skeleton — `odin-dashboard` CLI, Dash app shell, multi-dispatch routing, header + dispatch picker, shared data layer, plus a small change to the aggregator that emits `hardware.json` per dispatch. Lands a usable dashboard with a multi-dispatch landing page; tabs render placeholders.
- **Spec 1:** Tab A — Dispatch & Fleet (jobs / fleet / failures / live auto-poll).
- **Spec 2:** Tab B — Task drill-down + cross-commit trend.
- **Spec 3:** Tab C — Startup deep-dive + cross-commit trend.

Specs 1/2/3 are independent additive changes once Spec 0 lands.

## Goal (Spec 0)

Ship a usable Dash app that:

1. Discovers dispatches under `odin_runs/`.
2. Lets the user browse them via a multi-dispatch landing table and a header dropdown.
3. Routes to a per-dispatch view with three placeholder tabs (Dispatch & Fleet / Task drill-down / Startup).
4. Reads dispatch state through a shared, well-tested `DataLayer` so future tab specs only ever touch UI concerns.
5. Causes the aggregator to write a per-dispatch `hardware.json` so future trend axes can filter to matching hardware.

## Non-goals (Spec 0)

- Actual tab content for any of A/B/C — those are Specs 1/2/3.
- Auto-poll of `dispatch.json` for live updates — only Tab A needs it (Spec 1).
- Reading or rendering `training.json` 1000-pt series — Tabs B/C only (Specs 2/3).
- Dash browser-based E2E tests (require Chrome; slow). Layout-tree + callback unit tests are sufficient for skeleton work.
- Multi-user / shareable URLs / static-HTML export.
- Authentication.
- Hardware fingerprint richer than GPU model. (`fingerprint` is intentionally `"gpu:<gpu_name>"`; can grow later.)

## Architecture

A new sub-module `tools/odin/valhalla/dashboard/` sits next to the existing aggregator/writer/stats. Three responsibilities cleanly separated:

```
tools/odin/valhalla/dashboard/
├── __init__.py
├── cli.py                # `odin-dashboard` entry point
├── app.py                # Dash app factory + header + tab registry
├── data.py               # Pure data-layer (no Dash imports)
├── tabs/
│   ├── __init__.py
│   └── _placeholder.py   # Spec 0 only — placeholder for unimplemented tabs
└── tests/
    ├── test_cli.py
    ├── test_data.py
    ├── test_app.py
    ├── test_app_landing.py
    └── test_aggregator_hardware.py
```

Plus one change outside the new module:

- `tools/odin/valhalla/aggregator.py` — gains a `_write_hardware_json(dispatch_dir, rows, …)` helper called alongside `aggregate.json` writing. Reads hardware from any bundle's `training.json.hardware` and emits `hardware.json` per the schema below.

`dash`, `plotly`, and `pandas` become **hard dependencies** of the Odin tooling (added to `pyproject.toml`'s install_requires for the package, not as an optional extras group).

### Tab registry pattern

Each future tab spec adds one module under `tabs/` (e.g., `tabs/dispatch_fleet.py` for Spec 1) that exports `register(app, data)`. Spec 0 adds a small `_discover_tabs()` helper in `app.py` that imports the three known module names if they exist, calling `register` on each. This means Specs 1/2/3 are pure additive changes — no `app.py` edits.

## Components

### `cli.py`

`odin-dashboard` CLI. Same shape as `tools/odin/asgard/cli.py`.

**Invocations:**
```
odin-dashboard                              # multi-dispatch landing
odin-dashboard 20260427-141302              # jumps to that dispatch (Tab A is default landing)
odin-dashboard --dispatch 20260427-141302   # same, explicit
odin-dashboard LATEST                       # newest dispatch by sort order
```

**Flags:**

| Flag | Default | Purpose |
|---|---|---|
| `--runs-root` | `Path("odin_runs")` | Root scanned for dispatches. |
| `--port` | `8050` | Dash default. |
| `--host` | `127.0.0.1` | Loopback by default. |
| `--debug` | `False` | Forwards to `app.run_server(debug=...)` for hot reload. |
| `--no-browser` | `False` | Suppress the auto `webbrowser.open(url)` on startup. |

**Startup:**
1. Parse args; resolve `runs_root` to absolute path.
2. If `runs_root` doesn't exist → print error to stderr, exit 2.
3. If positional dispatch arg given → resolve via the existing `tools.odin.asgard.runner.resolve_dispatch_dir`. Bad ID → exit 2.
4. Build app via `app.create_app(runs_root, initial_dispatch=…)`.
5. Print one line: `odin-dashboard: serving http://<host>:<port>/ runs_root=<path>`.
6. Unless `--no-browser`, `webbrowser.open(url)` after a short sleep.
7. `app.run_server(host=…, port=…, debug=…)` — blocks until Ctrl-C.

**Exit codes:**
- 0 — clean shutdown.
- 2 — invalid args (bad runs_root or unknown dispatch_id).
- 3 — `dash` not installed (caught at top-of-file import; friendly message).
- 4 — port in use.
- 130 — SIGINT (Python default).

### `app.py`

Builds the Dash app. Pure factory function plus callback registration. No global state.

```python
def create_app(runs_root: Path, initial_dispatch: Path | None = None) -> Dash:
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.title = "Odin"
    data = DataLayer(runs_root)
    app.layout = _build_layout(data, initial_dispatch)
    _register_callbacks(app, data)
    for tab_module in _discover_tabs():
        tab_module.register(app, data)
    return app
```

**Layout (top-down):**

```
┌─────────────────────────────────────────────────────────────────┐
│ Header (hidden on landing)                                      │
│  Odin   Dispatch: [20260427-141302 ▼]   [● Live | ✓ Done]      │
│         commit: 0a3d72e426d   2 hosts                           │
├─────────────────────────────────────────────────────────────────┤
│ Tab strip (hidden on landing)                                   │
│ [ A — Dispatch & Fleet ] [ B — Task drill-down ] [ C — Startup ]│
├─────────────────────────────────────────────────────────────────┤
│ <page content>                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**URL scheme:**
- `/` — multi-dispatch landing (real table; click row → navigate to `/<id>/`).
- `/<dispatch_id>/` — redirects to `/<dispatch_id>/dispatch-fleet`.
- `/<dispatch_id>/<tab_id>` — tab content. Valid `tab_id ∈ {dispatch-fleet, task-drilldown, startup}`.
- Anything else — 404 panel with link back to `/`.

**Routing:**
- `dcc.Location(id="url")` reads the URL.
- One root callback `Input("url", "pathname") → Output("page-content", "children")` parses path → renders the matching component.
- `dcc.Store(id="active-dispatch", storage_type="memory")` caches the parsed `dispatch_id` for tab modules to read.

**Header behaviors:**
- Dispatch dropdown populated from `data.list_dispatches()`. Newest-first.
- Selecting an entry updates URL to `/<id>/<current_tab>`.
- "Live | Done" pill driven by `dispatch.json.ended_at` — `null` → orange "● Live", non-null → grey "✓ Done".
- Header hidden on `/` (no dispatch context).

**Tab strip behaviors:**
- Three buttons; one is `.active` based on URL path segment.
- `dcc.Link` for navigation (no full page reloads).
- Hidden on `/`.

**Spec 0 content:**
- `/` → real landing table built from `data.list_dispatches()`. Columns: dispatch_id, started_at, ended_at, jobs total/completed/failed, hostnames. Row-click navigates.
- `/<id>/<tab>` → renders `tabs._placeholder` ("Tab implemented in Spec N — coming soon"). When Specs 1/2/3 add their `tabs/<name>.py` modules with `register()` callbacks, those replace the placeholder.

**`initial_dispatch` from CLI:** if set, the app's first `pathname` is rewritten to `/<id>/dispatch-fleet` so navigating from CLI takes you straight to Tab A. Implemented via a tiny client-side `assets/init.js` script that runs once on page load and sets `window.location.pathname` if it's `/`.

### `data.py`

Pure-Python data layer. Zero Dash imports. Tested independently.

```python
@dataclass(frozen=True)
class DispatchSummary:
    dispatch_id: str
    started_at: str
    ended_at: str | None
    jobs_total: int
    jobs_completed: int
    jobs_failed: int
    jobs_pending: int
    skipped_total: int
    hostnames: list[str]


@dataclass(frozen=True)
class HardwareInfo:
    hostname: str
    gpu_devices: list[dict]   # [{name, mem_gb, compute_cap}, ...]
    cpu_name: str
    cpu_count: int
    ram_gb: float
    sourced_from: str         # "<dispatch_id>/<run_id>"


class DataLayer:
    def __init__(self, runs_root: Path, cache_size: int = 64):
        self._runs_root = runs_root.resolve()
        # decorator-applied caches on the load_* methods below

    def list_dispatches(self) -> list[DispatchSummary]: ...
    def load_dispatch(self, dispatch_id: str) -> dict: ...      # raw dispatch.json
    def load_aggregate(self, dispatch_id: str) -> dict | None:
    def load_hardware(self, dispatch_id: str) -> dict | None:   # the new hardware.json
    def load_training(self, dispatch_id: str, run_id: str) -> dict | None: ...
    def load_startup(self, dispatch_id: str, run_id: str) -> dict | None: ...
    def lookup_hardware(self, host: str) -> HardwareInfo | None: ...
    def trend_dispatches_for(
        self, current_dispatch_id: str, task: str, framework: str, backend: str, n: int = 10,
    ) -> list[str]: ...
    def invalidate(self, dispatch_id: str | None = None) -> None: ...
```

**Caching policy:**

| Method | Cache | Why |
|---|---|---|
| `list_dispatches` | 5 s soft TTL | Multi-dispatch picker re-renders frequently. |
| `load_dispatch` | None — always re-read | Tab A's auto-poll relies on always-fresh reads. |
| `load_aggregate` | LRU 64 | Static after dispatch ends. |
| `load_hardware` | LRU 64 | Static. |
| `load_training` | LRU 64 | Big — 1000-pt series × N seeds. |
| `load_startup` | LRU 64 | Tiny. |
| `lookup_hardware` | LRU 32 (per host) | Walks `odin_runs/`; first hit per session pays the cost. |
| `trend_dispatches_for` | LRU 32 (per task tuple) | Reads `hardware.json` from many dispatches. |

**`invalidate(dispatch_id=None)`** drops the relevant cache entries. Spec 1's Tab A poll calls it when `dispatch.json.ended_at` flips from `null` to a value so the just-written aggregate is picked up.

**`list_dispatches` algorithm:**
1. List entries under `runs_root/`.
2. Filter to directories whose name matches the `YYYYMMDD-HHMMSS` pattern AND that contain a `dispatch.json`. (Excludes loose pre-T3.1 bundles like `rsl-rl_physx_..._seed42`.)
3. Read each `dispatch.json`; build `DispatchSummary` from the totals + fleet.
4. Sort newest-first by directory name.

**`lookup_hardware(host)` fall-back algorithm** (used when current dispatch's `hardware.json` is absent or doesn't list the host):
1. List dispatches newest-first.
2. For each, check `dispatch.json.jobs[].assigned_to`. If any job ran on `host`, open that bundle's `training.json` and read `.hardware`.
3. Return the first hit, with `sourced_from` set.
4. If nothing matches, return `None`.

**`trend_dispatches_for(current_dispatch_id, task, framework, backend, n=10)` algorithm:**
1. Open the current dispatch's `hardware.json`; extract its `fingerprint`.
2. Walk all dispatches newest-first.
3. For each: open its `hardware.json` (skip if absent — pre-feature dispatches are excluded from trend); skip if its fingerprint doesn't match.
4. Open its `aggregate.json`; skip if it doesn't have a row matching `(task, framework, backend)`.
5. Append the dispatch_id to the result list. Stop after `n` matches.
6. Return the list (most recent first).

### Aggregator change: `hardware.json`

Inside `tools/odin/valhalla/aggregator.py`'s `aggregate_dispatch()`, after computing `rows` and before writing `aggregate.json`, also write `hardware.json` next to it.

**Generation logic:**
1. Walk `dispatch.json.jobs[].assigned_to` (only completed jobs — failed jobs may not have a `training.json`).
2. For each unique host, find the first bundle where `assigned_to == host` AND `training.json` exists with a `.hardware` block.
3. Collect into a `hosts` dict keyed by host address.
4. Compute `fingerprint`: take the first host's first GPU's `name`, normalise (strip "NVIDIA " prefix, replace spaces with `-`), prefix with `gpu:`. E.g. `"NVIDIA L40"` → `"gpu:NVIDIA-L40"`.
5. Write to `<dispatch_dir>/hardware.json` atomically (temp-file + rename, mirroring `write_dispatch_state`).

**Schema (1.0):**

```json
{
  "schema_version": "1.0",
  "dispatch_id": "20260427-141302",
  "generated_at": "2026-04-27T14:30:00Z",
  "hosts": {
    "10.176.214.169": {
      "hostname": "Odin-Runner-5",
      "gpu_devices": [
        {"name": "NVIDIA L40", "mem_gb": 44.32, "compute_cap": "8.9"}
      ],
      "cpu_name": "Intel Xeon Processor (Icelake)",
      "cpu_count": 16,
      "ram_gb": 62.79,
      "sourced_from": "rsl-rl_physx_Isaac-Ant-Direct-v0_20260427-141302_seed42"
    },
    "10.176.215.13": { ... }
  },
  "fingerprint": "gpu:NVIDIA-L40"
}
```

**Failure handling:** If hardware-collection raises (no completed bundles, all `training.json` missing the `.hardware` block, etc.), the aggregator logs a `[WARNING]` line and skips the file. `aggregate.json` writing proceeds normally. The dashboard's `lookup_hardware` fall-back covers the missing-file case.

**Heterogeneous fleets:** If hosts in one dispatch ran with different GPUs (rare but possible), the dispatch's `fingerprint` reflects the first host only. Trend filtering is single-fingerprint and won't include cross-fleet comparisons; this is correct — comparing one mixed-fleet dispatch to another is unsafe to do automatically. A future enhancement could promote `fingerprint` to `set(fingerprints)`; not in scope for Spec 0.

## Data flow

**Cold start (no positional arg):**
```
odin-dashboard --runs-root odin_runs
  → cli.main parses args
  → cli imports app.create_app(runs_root, initial_dispatch=None)
  → DataLayer instantiates with empty caches
  → Dash layout built; landing page is the default
  → app.run_server() blocks
  → Browser opens http://127.0.0.1:8050/
  → Routing callback fires for path "/"
  → data.list_dispatches() reads odin_runs/
  → Landing table rendered
```

**With positional arg:**
```
odin-dashboard 20260427-141302
  → cli resolves to absolute dispatch_dir
  → cli passes initial_dispatch=<path> to create_app
  → app's assets/init.js fires once on page load: if path == "/", redirect to "/20260427-141302/dispatch-fleet"
  → Routing callback then handles the new path → renders header + tab strip + placeholder content
```

**Header dispatch dropdown change:**
```
User selects "20260424-160119" in header dropdown
  → Dropdown callback fires: parses URL for current tab, computes new URL "/20260424-160119/<current_tab>"
  → Sets dcc.Location.href
  → Routing callback fires for new path
  → Header rebuilt for new dispatch (commit_sha, hostnames re-read)
  → Tab content refreshed
```

**Aggregator hardware-file write (inside `aggregate_dispatch`):**
```
aggregator iterates rows, computes per-seed metrics
  → after rows complete, _write_hardware_json(dispatch_dir, dispatch_json)
  → reads training.json.hardware from any one completed bundle per host
  → writes hardware.json atomically next to aggregate.json
  → on exception, logs warning and continues
```

## Error handling

| Failure mode | Where | Behavior |
|---|---|---|
| `runs_root` doesn't exist | CLI startup | Print error, exit 2. |
| `dispatch_id` not found | CLI startup OR header dropdown | CLI: exit 2. Dropdown: redirect to `/` with a flash banner. |
| `dispatch.json` malformed | `data.load_dispatch` | Raise; CLI / app shell catch and render an error page with the file path. |
| `aggregate.json` absent | `data.load_aggregate` | Returns `None`. Landing page row shows "Pending" status. Tabs (when implemented) handle `None` per their own contract. |
| `hardware.json` absent for current dispatch | `data.load_hardware` | Returns `None`. `lookup_hardware` falls back to the cross-bundle scan. Trend axis treats this dispatch as fingerprint-unknown (excludes from comparison). |
| `hardware.json` writer fails inside aggregator | `aggregator.py` | Log a `[WARNING]` line; do not fail the aggregator. Trend will fall back as above. |
| Port already in use | `app.run_server()` | Werkzeug raises; CLI catches and prints `odin-dashboard: port {port} is in use; try --port`, exit 4. |
| `dash` not installed | top-of-`cli.py` import | Caught `ModuleNotFoundError` at boundary; print install hint, exit 3. |
| 404 (unknown URL path) | Routing callback | Renders a "Not found" panel with link back to `/`. |
| Concurrent file writes during live poll | `data.load_dispatch` | Reads a snapshot; on `JSONDecodeError`, callers log and reuse the previous value. Self-correcting on the next tick. |

## Testing strategy

Six test files. All pure-Python.

### `tools/odin/valhalla/dashboard/tests/test_data.py`

| Test | Behavior |
|---|---|
| `test_empty_runs_root_returns_empty_list` | `list_dispatches` on a non-existent or empty dir → `[]`. |
| `test_list_dispatches_excludes_loose_bundles` | Fabricated tree with one dispatch dir + one loose `rsl-rl_physx_…_seed42` dir → only the dispatch is returned. |
| `test_list_dispatches_sorted_newest_first` | Three dispatches with timestamp names → returned in descending order. |
| `test_load_dispatch_round_trip` | Reads dispatch.json from a fabricated tree; returns the dict. |
| `test_load_aggregate_returns_none_when_missing` | Dispatch dir without aggregate.json → `None`. |
| `test_load_hardware_returns_none_when_missing` | Dispatch dir without hardware.json → `None`. |
| `test_lookup_hardware_walks_dispatches_newest_first` | Two dispatches with overlapping hosts → `lookup_hardware` returns the newest hit. |
| `test_lookup_hardware_returns_none_when_unknown_host` | No bundles ever ran on this host → `None`. |
| `test_trend_dispatches_for_filters_by_fingerprint` | Three dispatches: two with `gpu:NVIDIA-L40`, one with `gpu:NVIDIA-A100`. Current is L40. Trend list excludes the A100 dispatch. |
| `test_trend_dispatches_for_filters_by_task` | Two L40 dispatches, only one ran the (task, framework, backend) row. Trend list has one entry. |
| `test_trend_dispatches_for_excludes_pre_feature` | A dispatch without hardware.json is excluded from trend results. |
| `test_invalidate_drops_specific_dispatch_caches` | Cache populated; `invalidate("foo")` clears load_dispatch / load_aggregate / load_hardware for "foo" but leaves others. |

### `tools/odin/valhalla/dashboard/tests/test_cli.py`

| Test | Behavior |
|---|---|
| `test_parse_args_defaults` | `--port`, `--host`, `--runs-root` all match documented defaults. |
| `test_parse_args_explicit` | All flags accepted; positional arg parsed as dispatch_id. |
| `test_main_invalid_runs_root_exits_2` | Non-existent `--runs-root` → exit 2; stderr mentions the path. |
| `test_main_unknown_dispatch_exits_2` | Bogus dispatch_id positional → exit 2. |
| `test_main_latest_resolves_via_resolve_dispatch_dir` | `LATEST` resolves to the newest dispatch dir; passed as `initial_dispatch` to `create_app`. |
| `test_main_no_browser_suppresses_open` | `--no-browser` → `webbrowser.open` not called. |
| `test_main_dash_not_installed_exits_3` | Monkeypatch import to raise; CLI exits 3 with friendly message. |
| `test_main_port_in_use_exits_4` | `app.run_server` raises `OSError(EADDRINUSE)`; CLI exits 4. |

### `tools/odin/valhalla/dashboard/tests/test_app.py`

| Test | Behavior |
|---|---|
| `test_create_app_returns_dash_instance` | `create_app(runs_root)` returns a `Dash` whose `layout` is non-empty. |
| `test_layout_contains_dispatch_dropdown` | Walks the layout tree; finds a `dcc.Dropdown` with id `"dispatch-dropdown"`. |
| `test_routing_callback_landing` | Calling the routing callback with `pathname="/"` returns the landing component. |
| `test_routing_callback_dispatch_redirects_to_tab_a` | `pathname="/<id>/"` returns a redirect to `/<id>/dispatch-fleet`. |
| `test_routing_callback_unknown_path_returns_404` | `pathname="/garbage/"` returns the 404 panel. |
| `test_tab_strip_hidden_on_landing` | Layout for `/` doesn't render the tab strip component. |
| `test_placeholder_renders_for_unimplemented_tab` | Path `/<id>/dispatch-fleet` returns the `_placeholder` content (since Spec 0 doesn't ship Tab A content). |

### `tools/odin/valhalla/dashboard/tests/test_app_landing.py`

| Test | Behavior |
|---|---|
| `test_landing_table_lists_all_dispatches` | Fabricate three dispatches; landing component contains a row for each. |
| `test_landing_table_columns` | Component has the documented columns: dispatch_id, started_at, ended_at, jobs total/completed/failed, hostnames. |
| `test_landing_row_link_points_to_dispatch` | Row's link href is `/<dispatch_id>/`. |

### `tools/odin/valhalla/dashboard/tests/test_aggregator_hardware.py`

| Test | Behavior |
|---|---|
| `test_hardware_json_written_alongside_aggregate` | After `aggregate_dispatch`, `hardware.json` exists in the dispatch dir. |
| `test_hardware_json_schema` | Has top-level fields `schema_version`, `dispatch_id`, `generated_at`, `hosts`, `fingerprint`. |
| `test_hardware_json_fingerprint_format` | `fingerprint` is `"gpu:NVIDIA-L40"` for an L40 fleet. |
| `test_hardware_json_per_host_block` | Each host entry has `hostname`, `gpu_devices`, `cpu_name`, `cpu_count`, `ram_gb`, `sourced_from`. |
| `test_hardware_json_write_failure_warns_does_not_raise` | Force a bundle without `.hardware` (corrupt training.json); aggregator logs warning, doesn't raise, `aggregate.json` still written. |

### `tools/odin/tests/test_asgard_integration.py` (extension)

| Test | Behavior |
|---|---|
| `test_loopback_dispatch_writes_hardware_json` | The existing slow-marked loopback test gains an assertion: `<dispatch_dir>/hardware.json` exists after `run_dispatch`. Confirms end-to-end wiring. |

**Coverage targets:** `data.py` ≥ 95%; `cli.py` ≥ 90%; `app.py` ≥ 80%; aggregator hardware code ≥ 95% on the new lines.

**Performance check:** `data.list_dispatches` against a fabricated 50-dir tree completes under 100 ms. Test asserts a wall-clock ceiling.

**No browser-based E2E tests in Spec 0.** Layout-tree assertions + callback unit tests cover behaviors. Tabs 1/2/3 may add browser tests for chart rendering — not Spec 0.

## Implementation order (preview — full plan in writing-plans phase)

Approximate task chain, each one its own commit:

1. Add `dash`, `plotly`, `pandas` to `pyproject.toml` install_requires.
2. `data.py` — `DispatchSummary`, `HardwareInfo`, `DataLayer.list_dispatches` + tests.
3. `data.py` — `load_dispatch`, `load_aggregate`, `load_hardware` + tests.
4. `data.py` — `lookup_hardware` (cross-dispatch fallback) + tests.
5. `data.py` — `trend_dispatches_for` + tests.
6. `data.py` — `load_training`, `load_startup`, `invalidate` + tests.
7. Aggregator: `_write_hardware_json` + tests in `test_aggregator_hardware.py`.
8. `app.py` — `create_app` factory, layout skeleton, routing callback, 404 handler + tests.
9. `app.py` — landing table component + tests in `test_app_landing.py`.
10. `tabs/_placeholder.py` + tab registry helper.
11. `cli.py` — argparse, exit codes, friendly error messages + tests.
12. `assets/init.js` — `initial_dispatch` redirect-on-first-load.
13. Integration test extension: `test_loopback_dispatch_writes_hardware_json`.
14. Architecture-doc change-log entry.

## Non-functional requirements

- **No new required system dependencies** (no Chrome, no Selenium). Pip-installable Python packages only.
- **Single-process**, single-user, loopback-bound by default.
- **Sub-100 ms response** for landing-page load against ≤ 50 dispatches.
- **Telemetry:** none. Local-only tool.
- **Schema:** `hardware.json` schema 1.0; major-match validation in the dashboard reader (consistent with `dispatch.json` / `aggregate.json` precedent).
