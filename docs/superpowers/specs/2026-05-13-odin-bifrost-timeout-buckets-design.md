# Bifrost per-task timeouts via timeout-class buckets — Design

**Status:** draft
**Date:** 2026-05-13
**Task covered:** Give Bifrost dispatches per-task timeout budgets (e.g.
Cartpole 15m, Shadow-Vision 8h) instead of one workflow-wide value, by
splitting a dispatch into N OSMO workflows keyed on a `timeout_class`.

## 1. Motivation

Asgard varies `per_job_timeout_s` per task class so a wedged Cartpole
gets reaped in 15 min while a legitimate Shadow-Vision run can use its
full 8 h budget. Bifrost today emits one OSMO workflow with a single
`workflow.timeout.exec_timeout` value — every task in the dispatch
shares it. Two failure modes:

- **Tight global timeout** (sized for short tasks): long tasks get
  reaped mid-training as `FAILED_EXEC_TIMEOUT`.
- **Loose global timeout** (sized for the longest task): a wedged
  short task burns its entire node-slot for hours before OSMO reaps it.

A 200-job real dispatch would routinely hit one or both. We need
per-task budgets.

## 2. Why a single OSMO workflow can't do it

OSMO's YAML schema admits `workflow.timeout: TimeoutSpec` only.
`TaskGroupSpec` and `TaskSpec` have no `timeout` field and pydantic's
`extra='forbid'` on both means we can't smuggle one in. Source:
`OSMO/src/utils/job/task.py` (no `timeout`), `OSMO/src/utils/job/workflow.py`
(`extra='forbid'` on `TimeoutSpec`).

OSMO auto-wraps top-level `tasks:` into one-group-per-task
(`workflow.py:validate_tasks_groups`), so each task already has an
independent clock — but the value is workflow-wide. There's no
per-group override path.

The only way to vary the value: **submit multiple workflows**, one per
timeout class, each with its own `workflow.timeout`.

## 3. Approach

Bifrost groups planned rows by a `timeout_class` (a short string keyed
on the curated env YAML), chunks each group into batches of `<=
chunk_size` (default 25), and emits one OSMO workflow per chunk.
Dispatch state tracks all workflow IDs and which run_ids live in which
workflow; the poller walks all of them.

### 3.1 Bucket strategy

Two-step grouping per dispatch:

1. **Bucket by `timeout_class`.** A class is a short string (`short`,
   `medium`, `long`, `very_long` is a fine default lexicon, but it's a
   free-form string in the YAML so users can name e.g. `shadow_vision`
   if they want a one-off). The class maps to an `exec_timeout` value
   declared in `bifrost-osmo.yaml`.
2. **Chunk by size.** Within each bucket, split into batches of at
   most `chunk_size` rows (config field, default 25).

Each chunk submits as one OSMO workflow named
`odin-disp-{dispatch_id}-{timeout_class}-{chunk_index}` with the
class's `exec_timeout`.

### 3.2 Why 25 (default chunk_size)

- **Polling efficiency.** `osmo workflow query` cost scales with task
  count; smaller workflows refresh faster.
- **Failure blast radius.** A workflow-level infrastructure failure
  affects at most `chunk_size` runs.
- **Pool fairness.** Smaller workflows are less likely to monopolize
  a pool's quota.

Not a hard OSMO limit — we've seen 24-task workflows submit fine. The
value is tunable in `bifrost-osmo.yaml`.

## 4. Schema changes

### 4.1 Curated env YAML

Each kept env gains a `timeout_class` field (free-form string;
required). The schema today (`physx_envs.yaml`,
`newton_envs.yaml`) is:

```yaml
groups:
  direct/cartpole:
    - task_id: Isaac-Cartpole-Direct-v0
      framework: rsl_rl
      num_envs: 4096
      max_iterations: 150
      keep: true
      # NEW:
      timeout_class: short
```

The class names are not enforced by the curated YAML; they're matched
against `bifrost-osmo.yaml`'s `timeout_classes` table at planner time.
Unknown classes cause planner failure with a clear error.

Backwards compatibility: a missing `timeout_class` falls back to the
config's `default_timeout_class` (also a config field, default
`"medium"`).

### 4.2 `bifrost-osmo.yaml`

```yaml
timeout_classes:
  short:      "30m"
  medium:     "2h"
  long:       "8h"
  very_long:  "24h"

default_timeout_class: medium
chunk_size: 25            # max tasks per OSMO workflow
queue_timeout: "2h"       # workflow-level, applied to every chunk
```

Existing `defaults.exec_timeout` / `defaults.queue_timeout` become
deprecated and ignored once `timeout_classes` is present. We keep
parsing them so old configs don't crash, but emit a warning.

### 4.3 `JobEntry`

Two new optional fields:

- `timeout_class: str | None` — populated at planning, used for
  bucketing.
- `osmo_workflow_id: str | None` — already exists; semantics change to
  "the specific chunk-workflow this run lives in" rather than the
  dispatch-wide one.

### 4.4 `DispatchState`

- `osmo_workflow_id: str | None` → **removed** (replaced by per-job
  field; the existing single-value field becomes
  `osmo_workflow_ids: list[str]` for the dispatch-level inventory,
  used for resume).

Old `dispatch.json` migration: on load, if `osmo_workflow_id` is set
and `osmo_workflow_ids` is absent, treat as a 1-workflow legacy
dispatch and populate `osmo_workflow_ids = [osmo_workflow_id]` and
backfill each job's per-job field. Asgard-only dispatches (no OSMO
fields) are unaffected.

## 5. CLI / planner changes

### 5.1 `_build_rows`

Adds reading of `timeout_class` from each env. Fallback to
`cfg.default_timeout_class` if missing. Unknown class → raise.

### 5.2 New `_bucket_and_chunk` planner step

Pure function `(rows, chunk_size) -> list[(timeout_class, chunk_index, rows)]`.
Deterministic ordering: sort rows by `(timeout_class, task_id,
backend, seed)` before chunking so reruns are stable.

### 5.3 Workflow rendering

The Jinja template gains a parameter for `exec_timeout` (resolved
per-chunk by the planner) instead of pulling from
`cfg.defaults.exec_timeout`. Each chunk gets its class's value.

### 5.4 `main()` orchestration

Pseudo-flow:

```python
buckets = _bucket_and_chunk(rows, cfg.chunk_size)
state = build_initial_state(rows, dispatch_id)
write_dispatch_state(dispatch_dir, state)

for timeout_class, chunk_idx, chunk_rows in buckets:
    yaml = render(chunk_rows, exec_timeout=cfg.timeout_classes[timeout_class])
    wf_id = client.submit(yaml, pool=cfg.pool)
    state.osmo_workflow_ids.append(wf_id)
    for r in chunk_rows:
        get_job(state, r.run_id).osmo_workflow_id = wf_id
    write_dispatch_state(dispatch_dir, state)

poll_all(state, client, on_completed)
_aggregate_at_end(dispatch_dir)
```

Failure handling: if a `client.submit()` raises mid-way, the workflows
already submitted keep running. The dispatch.json captures which are
live; `--resume` picks them up.

## 6. Poller changes (`poller.py`)

`poll_until_terminal` walks the list of workflow IDs each tick. For
efficiency:

- One `client.status(wf_id)` per workflow per tick (serial; fan-out
  isn't worth thread complexity for `chunk_size = 25` and N workflows
  in single-digits).
- The merged `snap.tasks` from all workflows feeds the existing
  per-task status loop — no change to the per-task transition logic.
- Terminal check: `_all_terminal(state)` is unchanged; it already
  walks `state.jobs`, which spans all workflows.

The poll cost is `O(N_workflows)` per tick. For a 200-job dispatch at
chunk_size=25 across 4 classes, that's ~10 workflows → 10 polls per
tick. With `--poll-interval 15s` that's 0.67 polls/sec — fine.

## 7. Resume path

`--resume <dispatch_id>` reads `osmo_workflow_ids` from dispatch.json
and re-attaches the poller to all of them. Same behavior as the new
forward path, just skipping the submit step.

If `--retry-failed` is used post-resume, the retried rows get
re-bucketed (new chunks of failed-only rows, new workflow_ids
appended).

## 8. Bundle download

Per-task dataset names are unchanged
(`{prefix}-{dispatch_id}-{run_id}`). Each task uploads its dataset
regardless of which workflow it lived in. `download_and_validate_bundle`
behavior is unchanged.

## 9. Aggregate

`aggregate_dispatch` already operates on `dispatch.json` + per-`run_id`
bundle dirs, with no awareness of OSMO workflows. The
end-of-dispatch hook in `cli.py` (already landed today) runs after
the multi-workflow poll loop terminates. No changes.

## 10. Test plan

Unit:

- `_bucket_and_chunk`: deterministic ordering, chunk-size respected,
  buckets keyed correctly, empty input handled.
- Curated YAML loader: missing `timeout_class` → fallback to default,
  unknown class → raise.
- Config loader: `timeout_classes` parsing, deprecation warning when
  `defaults.exec_timeout` is also present.
- `JobEntry`: per-job `osmo_workflow_id` set after submit.
- `DispatchState` migration: old single-`osmo_workflow_id` dispatches
  load cleanly into the new `osmo_workflow_ids` list.

Integration (with mock `OsmoClient`):

- 3 rows, 2 classes → 2 workflows submitted, each with the right
  `exec_timeout`.
- 50 rows, 1 class, `chunk_size=25` → 2 workflows of 25 each.
- Poll loop with 2 workflows: per-task transitions land in
  dispatch.json across workflows.
- Submit failure on workflow 2 of 3: workflow 1 keeps running, dispatch
  resumable.

E2E (against real OSMO, slow-marked):

- 1 dispatch with mixed `short` (cartpole) and `medium` (ant) classes,
  6 tasks total in 2 workflows. Verify both timeout values land in
  OSMO and bundles aggregate correctly.

## 11. Out of scope

- Per-task overrides beyond the class system (would need a free-form
  numeric field per env — adds complexity without clear value vs.
  reusing a class).
- Client-side wedge detection (Heimdall-style for OSMO). Tracked
  separately; complements the timeout-class system but doesn't replace
  it.
- Asgard changes. The asgard `per_job_timeout_s` per-task field
  remains, populated from the same curated-YAML field (if we add the
  field to the YAML we should backfill asgard too — but that's a
  separate task).

## 12. Migration / rollout

- The curated YAML field is added incrementally: PR adds the field +
  populates it for the ~50 kept envs. CI fails if a kept env lacks
  `timeout_class` (after we land the parser change).
- The config field `timeout_classes` is required for any dispatch
  that uses the new path. Old configs lacking it get a deprecation
  warning and fall back to single-workflow mode (one workflow,
  `defaults.exec_timeout` for all tasks).
- Asgard-only users see no change.

---

## Revision (2026-05-13): drop `timeout_class`, reuse `job_budgets.yaml`

The original design above introduced a `timeout_class` indirection on
each curated env (`short`, `medium`, ...), mapped via a
`timeout_classes:` table in `bifrost-osmo.yaml` to OSMO `exec_timeout`
strings. After landing 10 commits implementing that, the user observed
that Asgard already maintains a per-task numeric source of truth at
`tools/odin/config/job_budgets.yaml`, loaded by
`tools.odin.asgard.budgets.load_budgets`. The class system duplicated
this in a coarser, less discoverable form: every new env needed both a
budget entry (for Asgard) and a class label (for Bifrost), and the
class names had to be kept in sync with the YAML's
`timeout_classes` keys.

### What changed

- `_PlannedRow.timeout_class: str` → `_PlannedRow.per_task_timeout_s: int`.
- New `_resolve_per_task_timeout_s(task_id, framework, budgets)` helper
  pulls the seconds value straight from
  `budgets.budgets[task_id][framework]`, with fallback to
  `budgets.defaults[framework]`. Missing both → `BifrostConfigError` at
  plan time.
- `_build_rows` takes a `budgets: Budgets` kwarg (the loaded table).
- `_bucket_and_chunk` no longer groups by class. It now:
  1. Sorts rows ascending by `(per_task_timeout_s, task_id, backend, seed)`.
  2. Chunks at `chunk_size`.
  3. Emits `(chunk_index, max_timeout_s, rows)` where
     `max_timeout_s = max(r.per_task_timeout_s for r in chunk)`.
- The CLI renders one workflow YAML per chunk
  (`workflow.<idx>.yaml`) with `exec_timeout = f"{max_timeout_s}s"`.
- `BifrostConfig.timeout_classes` and `default_timeout_class` are
  removed from the dataclass. The loader still parses
  `bifrost-osmo.yaml` files that carry these fields (and
  `defaults.exec_timeout`) but warns "no longer used" so operators
  spot the dead field.
- A new `--budgets-yaml` CLI flag points at the budgets table
  (default `tools/odin/config/job_budgets.yaml`).
- Curated YAMLs (`physx_envs.yaml`, `newton_envs.yaml`) revert to the
  pre-class shape; the `timeout_class:` lines added on each kept env
  are dropped (54 + 23 lines).

### Why

One source of truth. Asgard and Bifrost now look up the same numeric
budget for the same `(task, framework)` pair. A budget bump is one
file edit, no class table to keep in sync. The chunked-by-budget
ordering is also strictly more flexible — the workflow `exec_timeout`
is sized to its tightest-fitting chunk rather than to a coarse class.

### Behavior change worth noting

- Bifrost no longer warns / errors on "unknown class" because there
  are no classes. Instead it errors at plan time if a task is missing
  from `job_budgets.yaml` **and** `defaults.<framework>` is absent in
  that file. With `defaults.rsl_rl` and `defaults.skrl` populated (the
  HEAD state), this only fires on a brand-new framework key.
- Chunk filenames change from `workflow.<class>.<idx>.yaml` to
  `workflow.<idx>.yaml`. The single-workflow path
  (`workflow.yaml`) is also gone; even a 1-row dispatch produces
  `workflow.0.yaml`.
- Curated YAMLs no longer carry the `timeout_class` field. The
  enumerator scripts that generated those files (if any) should not
  re-emit it.

### Migration

- Local `bifrost-osmo.yaml`: remove `timeout_classes:` and
  `default_timeout_class:` lines. Keep `chunk_size`. `defaults.exec_timeout`
  can stay (warning) until a follow-up makes it optional in the schema.
- No-op for `job_budgets.yaml`; it already has the right shape.
- Asgard: untouched.
