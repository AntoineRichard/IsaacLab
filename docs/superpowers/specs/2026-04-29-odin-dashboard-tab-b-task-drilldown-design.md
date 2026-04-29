# Odin Dashboard — Tab B: Task Drill-down (Spec 2) Design

**Status:** approved (pending user review of this written form)
**Branch:** `antoiner/feat/odin`
**Series:** Spec 2 of 4. Spec 0 (skeleton) + Spec 1 (Tab A — Dispatch & Fleet) landed; this fills in the second tab.

## Background

Spec 0 shipped the dashboard skeleton with three placeholder tabs routed via `tools/odin/valhalla/dashboard/app.py`'s registry. Spec 1 filled in Tab A (Dispatch & Fleet — per-dispatch health). Tab B fills in the second placeholder: a per-(task × framework × backend) drill-down view that combines:

- **Per-seed reward / ep-length curves** for the current dispatch (the 1000-pt `series_per_iter` data already produced by Hugin/Munin).
- **Aggregate + per-seed stats** for the row (read from `aggregate.json`).
- **Cross-commit trend** showing how a chosen metric evolves across the last N dispatches that ran the same row on matching hardware (uses Spec 0's `data.trend_dispatches_for`).

Driver dispatch for the brainstorm: `odin_runs/20260427-141302/` — 15 completed jobs across 5 tasks × 3 seeds, all on L40 hardware.

## Goal

Ship a working Tab B: open `/<dispatch_id>/task-drilldown?task=<…>&framework=<…>&backend=<…>` and see (a) reward + ep_length curves with all seeds overlaid, (b) aggregate stats card + per-seed table, (c) a metric trend across the last N matching dispatches.  Tab A's job rows get a "drill in →" link that opens Tab B pre-pinned to that row.

## Non-goals

- Auto-poll. Tab B does not poll on a `dcc.Interval`; reads happen on user actions (picker, metric change, dispatch switch).
- Curves for metrics that aren't series. Iter-time, env-steps/s, RAM/GPU utilization are scalar `{mean, std}` aggregates — they appear in the stats panel, not in curves.
- Browser-based E2E tests.
- Authentication, persistence beyond `dcc.Store` memory storage.
- Aggregating startup metrics into `aggregate.rows[].aggregate` (the aggregator change isn't in scope here — Tab B computes startup mean/std on the fly from `seeds[]`).

## Locked decisions (from brainstorming Q1–Q5)

| Q | Decision |
|---|---|
| Q1 — Picker URL model | Query string: `?task=…&framework=…&backend=…`. Bookmark-able, deep-linkable. Tab A's job rows get a drill-in link to Tab B with these params filled in. |
| Q2 — Curves layout | Overlay all seeds on one chart, color-coded; legend toggles individual seeds (Plotly native). Two charts side-by-stacked: reward EMA + ep_length EMA. Divergent seeds get red dashed stroke. |
| Q3 — Picker shape | Searchable single dropdown with `task · framework × backend` rows. Dash's `dcc.Dropdown(searchable=True)` handles the substring filter. |
| Q4 — Stats panel | Two-column row below the curves: aggregate card (320 px fixed) on the left, per-seed table (rest of width) on the right. Aggregate card shows mean ± std (cv%); table shows 11 columns including startup phases. |
| Q5 — Trend chart | Default = line + ±std ribbon (filled band). Toggle to bar + vertical error-bar whiskers. Metric universe = 9 entries (reward / ep_length / iter_time / env_steps_per_s / ram / gpu / 3 startup phases). |

**Visual reference:** mockup files under `.superpowers/brainstorm/3762718-1777450164/content/` (kept on disk, not committed):
- `curve-overlay.html` — Option A locked.
- `stats-panel.html` — Option C locked, with real Ant numbers.
- `trend-shape.html` — Ribbon vs. bars side-by-side.

These are the visual source of truth for color palette, spacing, pill styles.

## Architecture

### Module layout

Tab B mirrors Tab A's package shape. Picked up by Spec 0's tab registry via `tab_id == "task-drilldown"` — no `app.py` edits required.

```
tools/odin/valhalla/dashboard/tabs/task_drilldown/
├── __init__.py            # re-exports `render` + `register`
├── layout.py              # static layout: picker + slots + stores
├── picker.py              # builds the searchable row dropdown
├── curves.py              # builds the reward + ep_length overlay charts (plotly)
├── stats.py               # builds aggregate card + per-seed table
├── trend.py               # builds metric selector + line/bar chart of N dispatches
├── url_state.py           # parse / serialize ?task=&framework=&backend= query string
└── callbacks.py           # registers the 5 callbacks against the layout

tools/odin/valhalla/dashboard/tests/
├── test_tab_b_url_state.py        # round-trip parse/serialize
├── test_tab_b_picker.py           # row list extraction + dropdown shape
├── test_tab_b_curves.py           # 2 plotly figures, 3 traces each, real series
├── test_tab_b_stats.py            # aggregate card lines + per-seed table rows
├── test_tab_b_trend.py            # ribbon + bar mode + empty state
└── test_tab_b_callbacks.py        # callback helpers (called directly)
```

### Module entry point (`__init__.py`)

```python
from tools.odin.valhalla.dashboard.tabs.task_drilldown.layout import build_layout

__all__ = ["render", "register"]


def render(dispatch_id: str, tab_id: str):
    return build_layout(dispatch_id)


def register(app, data):
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.callbacks import register_callbacks

    register_callbacks(app, data)
```

### Tab A → Tab B drill-in (Spec 1 extension)

`tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py` `_data_row` is updated so the Task cell becomes a `dcc.Link` whose href is `/<dispatch_id>/task-drilldown?task=…&framework=…&backend=…`. One cell renders as a link instead of plain text. One new test (`test_jobs_task_cell_links_to_tab_b`) under `tests/test_tab_a_jobs_table.py`. No other Spec 1 changes.

### No new DataLayer methods required

Tab B reads through Spec 0's `DataLayer`: `load_aggregate`, `load_training`, `trend_dispatches_for`. All exist.

## Components

### `url_state.py`

```python
@dataclass(frozen=True)
class TaskSelection:
    task: str | None
    framework: str | None
    backend: str | None


def parse_query_string(search: str) -> TaskSelection:
    """Parse Dash's ``dcc.Location.search`` value (e.g. '?task=A&framework=rsl_rl&backend=physx')."""


def serialize(selection: TaskSelection) -> str:
    """Return a query string starting with '?'.

    Empty / None fields are omitted; if all three are None, returns ''.
    """
```

Pure-data; zero Dash imports.

### `picker.py`

```python
def list_row_options(aggregate_payload: dict) -> list[dict]:
    """Return ``dcc.Dropdown`` options for every (task, framework, backend) row.

    Each option:
        {"label": "Isaac-Ant-Direct-v0 · rsl_rl × physx",
         "value": "Isaac-Ant-Direct-v0|rsl_rl|physx"}
    """


def render_picker(aggregate_payload: dict, selected: TaskSelection | None) -> html.Div:
    """Build the picker Div.

    Returns Div(id='tab-b-picker') containing a ``dcc.Dropdown`` with
    id 'tab-b-row-select', searchable=True. The dropdown's value is
    the pipe-separated row key from list_row_options; selected is
    resolved to a value if present, else left None.
    """
```

The pipe-separated value (`task|framework|backend`) is the simplest serialization through Dash — splits back to a `TaskSelection` in `callbacks.py`.

### `curves.py`

```python
def render_curves(
    bundles: dict[str, dict],
    *,
    divergent_seeds: list[str],
) -> html.Div:
    """Build the curves panel: two plotly figures (reward + ep_length).

    Args:
        bundles: ``{seed: training_payload}``. Each training_payload has
            ``learning.reward.series_per_iter`` and
            ``learning.ep_length.series_per_iter`` (both 1000-pt lists).
        divergent_seeds: seed strings that ``aggregate.divergent_seeds``
            flagged. Drawn with red dashed stroke instead of normal palette.

    Returns:
        Div(id='tab-b-curves') with two ``dcc.Graph`` components inside.
    """


_SEED_PALETTE = ["#66b6ff", "#50c88c", "#ffa500", "#beA0ff", "#e16868"]
```

Per-seed traces use the palette in seed-sorted order so seed 42 is always blue, 43 green, 44 orange, etc. Divergent seeds get red dashed stroke drawn on top.

### `stats.py`

```python
def render_aggregate_card(aggregate_block: dict, divergent_seeds: list[str]) -> html.Div:
    """Build the left-side aggregate card.

    aggregate_block is ``aggregate.json.rows[].aggregate``. Renders one
    line per metric: 'reward EMA: 7991.51 ± 257.68 cv 3.2%'.
    Closing line: divergent seeds list, or '—'.
    cv_pct color-codes: green ≤5%, orange 5-15%, red >15%.
    """


def render_seeds_table(seeds_block: dict) -> html.Div:
    """Build the right-side per-seed table.

    Columns: Seed · Status · Reward · Ep length · Iter time · env_steps/s
             · RAM peak · GPU mem · Wall time · Startup (app/env/first) · Host.
    11 columns total. Status pill: green 'OK' / red 'Failed' / grey '—'.
    """


def render_stats_panel(aggregate_payload_row: dict) -> html.Div:
    """Combine aggregate card + per-seed table into a two-column row.

    Returns Div(id='tab-b-stats') with both child cards.
    """
```

### `trend.py`

```python
_TREND_METRICS = [
    {"value": "reward_final_ema",      "label": "Reward (final EMA)",     "source": "aggregate"},
    {"value": "ep_length_final_ema",   "label": "Episode length (final EMA)", "source": "aggregate"},
    {"value": "iter_time_s_mean",      "label": "Iter time",              "source": "aggregate"},
    {"value": "env_steps_per_s_mean",  "label": "Env steps / s",          "source": "aggregate"},
    {"value": "ram_gb_peak",           "label": "RAM peak",               "source": "aggregate"},
    {"value": "gpu_mem_gb_peak",       "label": "GPU mem peak",           "source": "aggregate"},
    {"value": "startup_app_launch_s",  "label": "Startup: app launch",    "source": "seeds"},
    {"value": "startup_env_creation_s","label": "Startup: env creation",  "source": "seeds"},
    {"value": "startup_first_step_s",  "label": "Startup: first step",    "source": "seeds"},
]


def render_metric_selector(default_metric: str = "reward_final_ema",
                          default_mode: str = "ribbon") -> html.Div:
    """Metric dropdown + view-mode toggle (Ribbon / Bars)."""


def compute_trend_points(
    data,
    dispatch_ids: list[str],
    task: str,
    framework: str,
    backend: str,
    metric: str,
) -> list[dict]:
    """For each dispatch_id, compute {dispatch_id, commit_sha, mean, std,
    n_seeds_completed} for the metric.

    For metrics with source='aggregate', reads ``aggregate.rows[].aggregate.<metric>``.
    For metrics with source='seeds', computes mean/std on the fly from
    ``seeds[].<metric>`` (uses statistics.mean / statistics.pstdev).

    Skips dispatches whose aggregate.json is malformed or missing the row;
    logs a [WARNING] to stderr per skip.

    Returns dispatches in newest-first order.
    """


def render_trend_chart(points: list[dict], metric_label: str, *,
                       mode: str = "ribbon") -> dcc.Graph:
    """Render the trend chart.

    mode='ribbon' → line + ±std fill-between trace (default).
    mode='bars'   → bar chart with vertical error-bar whiskers.

    X-axis: commit_sha[:7] tick labels (left = oldest, right = newest).
            Hover tooltip shows full dispatch_id + n_seeds_completed.
    Y-axis: metric value, label = metric's pretty name.
    Current dispatch's marker (rightmost) drawn in NVIDIA green with
    white stroke; other points in seed-palette blue.
    """


def render_trend_section(
    data,
    *,
    current_dispatch_id: str,
    selection: TaskSelection,
    metric: str,
    mode: str,
) -> html.Div:
    """Top-level: selector + chart wrapped in Div(id='tab-b-trend').

    Calls trend_dispatches_for, compute_trend_points, render_trend_chart.
    Empty / single-point states (Section 3, Flow 7) handled here.
    """
```

### `layout.py`

```python
def build_layout(dispatch_id: str) -> html.Div:
    """Static layout for /<id>/task-drilldown.

    Slots:
      - tab-b-picker      (filled by init_picker callback)
      - tab-b-curves      (filled by update_curves_and_stats)
      - tab-b-stats       (filled by update_curves_and_stats)
      - tab-b-trend       (filled by update_trend)

    Stores: tab-b-dispatch-id, tab-b-selection, tab-b-trend-metric, tab-b-trend-mode.
    """
```

### `callbacks.py`

Five callbacks registered against the layout:

| # | Callback | Trigger | Output | Helper |
|---|---|---|---|---|
| 1 | `init_picker` | `tab-b-dispatch-id.data` | `tab-b-picker.children` | `_compute_picker_children` |
| 2 | `sync_url_to_selection` | `url.search` | `tab-b-selection.data` | `url_state.parse_query_string` |
| 3 | `picker_to_url` | `tab-b-row-select.value` | `url.search` | `url_state.serialize` |
| 4 | `update_curves_and_stats` | `tab-b-selection.data` + `tab-b-dispatch-id.data` | `tab-b-curves.children` + `tab-b-stats.children` | `_compute_curves_and_stats` |
| 5 | `update_trend` | `tab-b-selection.data` + `tab-b-dispatch-id.data` + `tab-b-trend-metric.data` + `tab-b-trend-mode.data` | `tab-b-trend.children` | `_compute_trend_children` |

Plus two thin store-writers (not in the count):
- Metric dropdown `value` → `tab-b-trend-metric` store.
- View-mode toggle `value` → `tab-b-trend-mode` store.

Each callback's body is a free function (`_compute_*`) tested directly without spinning up Dash — the same pattern Tab A uses.

**Bidirectional URL sync (callbacks 2 + 3):** URL → store on load; store → URL when picker changes. Loop guard via `dash.no_update` when value is unchanged.

## Data flow

### Flow 1 — Cold mount via direct URL

```
User opens /<id>/task-drilldown?task=Isaac-Ant-Direct-v0&framework=rsl_rl&backend=physx
  → app.py route → registry → tabs.task_drilldown.render(dispatch_id, "task-drilldown")
  → layout.build_layout(dispatch_id) returns layout with empty slots + stores
Dash mounts → Inputs fire → callbacks run:
  → sync_url_to_selection reads url.search → writes "Isaac-Ant-Direct-v0|rsl_rl|physx" to tab-b-selection store
  → init_picker reads aggregate.json → renders picker with that value pre-selected
  → update_curves_and_stats reads aggregate.json[row] + 3 training.json files → renders curves + stats
  → update_trend reads aggregate.json + walks trend_dispatches_for → renders trend chart
```

### Flow 2 — Cold mount without query string

```
User opens /<id>/task-drilldown (no params)
  → sync_url_to_selection: search='' → tab-b-selection store = empty TaskSelection
  → init_picker: aggregate has rows; default to first row (sorted by task name)
  → picker_to_url fires: serializes the default selection back to URL
  → URL becomes /<id>/task-drilldown?task=…&framework=…&backend=…
  → URL change re-fires sync_url_to_selection (same value, no-op via dash.no_update guard)
```

### Flow 3 — User picks a different row

```
User selects "Isaac-Cartpole-Direct-v0 · rsl_rl × physx" in the dropdown
  → tab-b-row-select.value = "Isaac-Cartpole-Direct-v0|rsl_rl|physx"
  → picker_to_url callback fires → writes ?task=…&framework=…&backend=… to URL
  → sync_url_to_selection re-fires → updates tab-b-selection store
  → update_curves_and_stats fires:
      reads aggregate.rows[] for new row, reads 3 new training.json bundles, re-renders
  → update_trend fires:
      walks trend_dispatches_for for new (task, fw, backend) tuple, re-renders chart
```

### Flow 4 — User changes metric or chart mode

```
User picks "Iter time" in metric dropdown
  → tab-b-trend-metric store updates
  → update_trend re-fires
  → reads same trend dispatch list, but reads different metric from each
  → re-renders chart with new y-axis label and units

User toggles Ribbon → Bars
  → tab-b-trend-mode store updates
  → update_trend re-fires
  → re-renders chart with mode='bars'
```

### Flow 5 — Tab A → Tab B drill-in

```
User on Tab A, clicks the Task cell on a row
  → dcc.Link href = "/<id>/task-drilldown?task=…&framework=…&backend=…"
  → URL changes; app.py routing rebuilds page-content for Tab B
  → Tab B cold mount with the link's params (Flow 1)
```

### Flow 6 — Header dispatch dropdown switches dispatches

```
Header dropdown: user picks 20260424-160119
  → URL = /20260424-160119/task-drilldown?task=Isaac-Ant-Direct-v0&framework=rsl_rl&backend=physx
  → Spec 0 routing rebuilds page-content for new dispatch
  → Tab B cold mount; init_picker reads new dispatch's aggregate.json
  → If "Isaac-Ant-Direct-v0 · rsl_rl × physx" exists in new dispatch → keep selection.
  → If row doesn't exist → render banner "Row not found in this dispatch — pick another row from the dropdown."
     Curves / stats / trend slots empty. User picks a row from the populated dropdown to continue.
```

### Flow 7 — Empty / single-point trend

```
trend_dispatches_for returns [] (no matching prior dispatch)
  → render_trend_section returns the empty-state Div:
      "Trend needs at least 1 prior dispatch matching <task> · <fw> × <be>
       on the same hardware (gpu:NVIDIA-L40)."
trend_dispatches_for returns [current_dispatch_id] (only the current one)
  → render_trend_section renders a single-point chart + a note:
      "First matching dispatch — trend will populate as more land."
trend_dispatches_for returns [N>1 dispatch_ids]
  → normal chart, default mode='ribbon'.
```

## Error handling

| Failure mode | Where | Behavior |
|---|---|---|
| `aggregate.json` absent (live dispatch, not yet aggregated) | `data.load_aggregate` returns `None` | Banner: "Aggregate not yet generated for this dispatch — Tab B is empty until aggregation completes." Picker / curves / stats / trend slots empty. |
| `aggregate.json` malformed | `data.load_aggregate` raises | Caught at callback boundary; render `tab-b-error-banner` with file path + exception class. Other slots empty. |
| Row not in current dispatch (after dispatch switch) | `aggregate.rows[]` filter returns `[]` | Banner "Row not found in this dispatch — pick another row from the dropdown." Curves / stats / trend slots empty. Picker still populated. |
| `training.json` missing for one seed | `data.load_training` returns `None` | Curves render the OTHER seeds normally; missing seed gets a small note in the legend "seed 43 (no series)". Stats table shows that seed's row with "—" in metric columns and red "Failed" status. |
| `training.json` missing for ALL seeds | All `load_training` calls return `None` | Curves panel renders empty-state "No training.json bundles available for this row." Stats card still works (reads aggregate, not training). |
| `training.json` lacks `learning.reward.series_per_iter` | Pure-render edge case | Curves panel renders "Reward series unavailable" text in place of that one chart; ep_length still tries normally. |
| `aggregate.json.rows[].seeds[]` empty | Stats panel, picker | Picker still lists the row. Stats shows "No completed seeds yet" in the aggregate card; per-seed table renders the failed/pending seeds with "—" metrics + status pills. |
| `data.trend_dispatches_for` returns `[]` | Empty-state path | Per Flow 7: banner-style note. |
| `data.trend_dispatches_for` returns `[current]` | Single-point path | Per Flow 7: single point + note. |
| `compute_trend_points` raises (one of the dispatch's aggregate.json malformed) | Caught per dispatch | Skip that dispatch, log `[WARNING]` to stderr, continue. Tooltip shows count of skipped dispatches if any. |
| `current_dispatch_id` has no `hardware.json` | `trend_dispatches_for` returns `[]` (Spec 0 behavior) | Empty-state Flow 7 path; banner: "Trend requires hardware.json for the current dispatch (run odin-aggregate)." |
| User-picked metric not in aggregate or seeds[] | `compute_trend_points` returns empty | Trend chart renders empty-state with the metric name: "Metric `<name>` not present in aggregate.json for this row." |
| Bad query string (e.g. `?task=foo` with foo not in any row) | `sync_url_to_selection` parses; row lookup fails | Banner: "Row 'foo' not found in this dispatch." Picker stays unselected (user picks valid row). |
| Duplicate query keys (`?task=a&task=b`) | `parse_query_string` | Take the last value; standard query-string semantics. |
| URL-encoded values | `parse_query_string` | `urllib.parse.parse_qs` handles decode; round-trip lossless. |

**Banner shape:** `Div(className="tab-b-error-banner")` (yellow border, dark yellow background — same family as Tab A's banner, different ID prefix).

**No silent failures.** Anything caught at the callback boundary surfaces a banner.

**Logging:** Each caught exception writes a `[WARNING]` line to `sys.stderr`.

**Concurrency:** Tab B does not auto-poll. All reads happen on user action (picker change, metric change, dispatch switch). No race window with on-disk writers.

## Testing strategy

Six pure-Python pytest files. Run with `PYTHONPATH=. python3 -m pytest --noconftest -p no:cacheprovider`. Sub-second total — no Dash server, no Isaac Sim.

### `test_tab_b_url_state.py`

| Test | Behavior |
|---|---|
| `test_parse_empty_string_returns_empty_selection` | `parse_query_string("")` → `TaskSelection(None, None, None)` |
| `test_parse_full_query_string` | `?task=A&framework=rsl_rl&backend=physx` → `TaskSelection("A", "rsl_rl", "physx")` |
| `test_parse_partial_query_string` | `?task=A` → `TaskSelection("A", None, None)` |
| `test_parse_url_encoded_task` | `?task=Isaac-Repose-Cube-Allegro-Direct-v0` round-trips correctly |
| `test_parse_duplicate_keys_takes_last` | `?task=a&task=b` → `task="b"` |
| `test_serialize_full_selection` | All three fields → `?task=A&framework=rsl_rl&backend=physx` |
| `test_serialize_omits_none_fields` | `TaskSelection("A", None, None)` → `?task=A` |
| `test_serialize_empty_selection_returns_empty_string` | All None → `""` |
| `test_round_trip_preserves_special_characters` | `serialize → parse → serialize` is idempotent |

### `test_tab_b_picker.py`

| Test | Behavior |
|---|---|
| `test_list_row_options_one_per_row` | 5 rows → 5 options |
| `test_list_row_options_label_format` | First option's label is "Isaac-Ant-Direct-v0 · rsl_rl × physx" |
| `test_list_row_options_value_format` | First option's value is "Isaac-Ant-Direct-v0\|rsl_rl\|physx" |
| `test_list_row_options_sorted_by_task_name` | Output sorted A-Z by task_id |
| `test_render_picker_contains_dropdown` | Output has `dcc.Dropdown` with `id="tab-b-row-select"` and `searchable=True` |
| `test_render_picker_preselects_value_when_in_options` | TaskSelection matching a row → dropdown's `value` set |
| `test_render_picker_no_preselection_when_selection_missing` | TaskSelection for a row not in options → `value` stays `None` |
| `test_render_picker_handles_empty_aggregate` | aggregate.rows = [] → dropdown rendered with empty options + placeholder |

### `test_tab_b_curves.py`

| Test | Behavior |
|---|---|
| `test_render_curves_returns_two_graph_components` | Output contains 2 `dcc.Graph` (reward + ep_length) |
| `test_render_curves_one_trace_per_seed` | 3 bundles → reward graph has 3 traces |
| `test_render_curves_seed_color_assignment_deterministic` | seed 42 always gets the first palette color, 43 the second, 44 the third |
| `test_render_curves_divergent_seed_styled_differently` | divergent_seeds=["43"] → seed 43's trace has dashed stroke + red color |
| `test_render_curves_handles_missing_seed_series` | one bundle has no `series_per_iter` → trace omitted, others render |
| `test_render_curves_empty_bundles_renders_empty_state` | bundles={} → renders "No training.json bundles available for this row." |
| `test_render_curves_x_axis_label` | x-axis title = "iterations" |
| `test_render_curves_reward_y_axis_label` | reward graph y-axis title = "reward (final EMA)" |

### `test_tab_b_stats.py`

| Test | Behavior |
|---|---|
| `test_aggregate_card_renders_one_line_per_metric` | 6 metrics → 6 stat lines + a divergent-seeds line |
| `test_aggregate_card_formats_mean_pm_std` | `{mean: 7991.51, std: 257.68, cv_pct: 3.22}` → text contains "7991.51", "± 257.68", "cv 3.2%" |
| `test_aggregate_card_cv_color_green_below_5pct` | cv_pct=3.2 → cv span has class `tab-b-cv-good` |
| `test_aggregate_card_cv_color_orange_5_to_15pct` | cv_pct=8.0 → class `tab-b-cv-warn` |
| `test_aggregate_card_cv_color_red_above_15pct` | cv_pct=20.0 → class `tab-b-cv-bad` |
| `test_aggregate_card_lists_divergent_seeds` | divergent_seeds=["43"] → text contains "seed 43" |
| `test_aggregate_card_no_divergent_seeds_renders_dash` | divergent_seeds=[] → text contains "—" on that line |
| `test_seeds_table_one_row_per_seed` | 3 seeds → 3 data rows + 1 header |
| `test_seeds_table_column_set` | Header has 11 columns (Seed / Status / Reward / Ep length / Iter time / env_steps/s / RAM peak / GPU mem / Wall time / Startup / Host) |
| `test_seeds_table_status_pill_for_completed` | seed.status="completed" → green "OK" pill |
| `test_seeds_table_status_pill_for_failed` | seed.status="failed" → red "Failed" pill |
| `test_seeds_table_dashes_when_metric_missing` | seed has no `iter_time_s_mean` → cell shows "—" |
| `test_render_stats_panel_contains_both_cards` | Output has both `tab-b-aggregate-card` and `tab-b-seeds-table` ids |

### `test_tab_b_trend.py`

| Test | Behavior |
|---|---|
| `test_compute_points_returns_one_per_dispatch` | 3 dispatch_ids, all with the row → 3 points |
| `test_compute_points_skips_missing_aggregate` | One dispatch lacks aggregate.json → skipped, 2 points returned |
| `test_compute_points_skips_dispatch_missing_row` | One dispatch's aggregate has no matching row → skipped |
| `test_compute_points_uses_aggregate_for_known_metric` | metric=`reward_final_ema` → reads `aggregate.rows[].aggregate.reward_final_ema` |
| `test_compute_points_computes_from_seeds_for_startup_metric` | metric=`startup_app_launch_s` → computes mean+std from `seeds[].startup_app_launch_s` |
| `test_compute_points_carries_commit_sha` | Each point has `commit_sha` from dispatch.json |
| `test_render_trend_chart_ribbon_mode` | mode='ribbon' → 2 traces (mean line + fill band) |
| `test_render_trend_chart_bars_mode` | mode='bars' → 1 bar trace with error_y |
| `test_render_trend_chart_current_marker_highlighted` | rightmost point has different color (NVIDIA green) |
| `test_render_trend_chart_x_labels_short_sha` | x-axis tick labels are commit_sha[:7] |
| `test_render_trend_section_empty_state` | trend_dispatches_for returns [] → renders empty-state banner |
| `test_render_trend_section_single_point_state` | trend_dispatches_for returns [current] → single-point chart + note |
| `test_render_trend_section_normal_render` | 3+ matching dispatches → renders metric selector + chart |

### `test_tab_b_callbacks.py`

| Test | Behavior |
|---|---|
| `test_init_picker_returns_picker_div` | Pass `_StubData` with aggregate; picker contains the dropdown |
| `test_sync_url_to_selection_parses_full` | `?task=A&framework=rsl_rl&backend=physx` → store value `"A\|rsl_rl\|physx"` |
| `test_sync_url_to_selection_handles_empty` | `""` → store value `None` |
| `test_picker_to_url_serializes_value` | Selection store value `"A\|rsl_rl\|physx"` → URL search `"?task=A&framework=rsl_rl&backend=physx"` |
| `test_update_curves_and_stats_loads_three_seed_bundles` | Selection set + 3 seeds in aggregate → `_StubData.load_training` called 3 times |
| `test_update_curves_and_stats_returns_curves_and_stats_divs` | Output is a 2-tuple of components, each with the expected outer id |
| `test_update_curves_and_stats_renders_banner_when_row_missing` | Selection points to a row not in aggregate → banner Div, empty curves/stats |
| `test_update_trend_returns_trend_div` | Default metric + ribbon mode → Div with trend chart inside |
| `test_update_trend_ignores_phantom_initial_calls` | All inputs None → `dash.no_update` |

### Tab A regression / extension

One new test under `tools/odin/valhalla/dashboard/tests/test_tab_a_jobs_table.py`:

| Test | Behavior |
|---|---|
| `test_jobs_task_cell_links_to_tab_b` | The Task cell renders as a `dcc.Link` with `href=f"/{dispatch_id}/task-drilldown?task={task}&framework={fw}&backend={be}"` |

### Coverage targets

- `url_state.py` ≥ 100% (small + pure).
- `picker.py` ≥ 95%.
- `curves.py` ≥ 90%.
- `stats.py` ≥ 95%.
- `trend.py` ≥ 90%.
- `callbacks.py` ≥ 85%.

**Total new tests:** ~55 across 6 files + 1 regression test in Tab A. Plus the existing 111 dashboard tests must stay green.

**No browser-based tests.** Layout-tree assertions + pure-function tests cover behavior; visual rendering is verified against the mockup HTML files.

## Implementation order (preview — full plan in writing-plans phase)

Approximate task chain:

1. `url_state.py` + tests (pure data; fastest TDD cycle).
2. Tab A regression: Task cell → `dcc.Link`. One-line edit + one new test.
3. `picker.py` + tests.
4. `stats.py` + tests.
5. `curves.py` + tests.
6. `trend.py` (compute_trend_points + render_trend_chart) + first half of tests.
7. `trend.py` (render_trend_section + empty/single-point states) + remaining tests.
8. `layout.py` skeleton with stores + slots.
9. `callbacks.py` (3 standard callbacks: init_picker, sync_url, picker_to_url) + tests.
10. `callbacks.py` (2 update callbacks: curves_and_stats, trend) + remaining tests.
11. CSS additions to `assets/style.css` for Tab B classes.
12. Architecture-doc change-log entry + manual smoke test against `20260427-141302`.

## Non-functional requirements

- **No new pip dependencies.** Tab B uses dash/plotly/pandas (already hard deps from Spec 0).
- **Sub-second total test time.** Six pytest files, ~55 tests, all pure-Python.
- **No background work.** Tab B does not auto-poll. Reads happen on user action.
- **Schema:** No `dispatch.json` / `aggregate.json` / `hardware.json` schema changes. Reads existing fields only.
- **Visual layout source-of-truth** lives in `.superpowers/brainstorm/3762718-1777450164/content/` mockup files.
