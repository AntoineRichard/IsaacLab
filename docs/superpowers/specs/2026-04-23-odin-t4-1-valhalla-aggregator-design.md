# Odin T4.1 — Valhalla Aggregator — Design

**Status:** approved
**Date:** 2026-04-23
**Task covered:** T4.1 of the Odin eval plan — per-dispatch aggregation of
bundle results into a single `aggregate.json` artifact.

## 1. Motivation

T3.1's dispatcher fans out training runs and rsyncs back a bundle per
`(task, framework, backend, seed)` tuple under
`odin_runs/<dispatch_id>/<run_id>/`. Each bundle carries three JSON files
(`manifest.json`, `training.json`, `startup.json`) plus raw
`training_data/`. Downstream consumption today requires walking the
directory, opening dozens of files, and re-deriving multi-seed stats
by hand.

T4.1 delivers the **first consumable layer**: one aggregated JSON per
dispatch that rolls every bundle into a structured per-row summary
(nested per-seed, with cross-seed aggregates) ready for the dashboard
(T4.2) or for CLI-level diff across commits.

## 2. Goals

- One atomically-written `odin_runs/<dispatch_id>/aggregate.json` per
  dispatch, schema-versioned.
- Nested shape: one row per `(task, framework, backend)`, with per-seed
  metrics under `seeds: {<seed_int>: {...}}` and cross-seed stats under
  `aggregate: {mean, std, min, max, cv_pct}`.
- Divergent-seed detection via z-score threshold (default `2.0`,
  CLI-tunable).
- Failed runs listed in a top-level `failures[]` with the
  `failure.kind` / `failure.message` copied from `dispatch.json`.
- Both auto-generation (at the end of `run_dispatch`) and manual
  re-generation (`odin-aggregate` CLI).
- No upstream IsaacLab changes — lives entirely in `tools/odin/valhalla/`.

## 3. Non-goals

- Cross-dispatch comparison or index files (T4.2 dashboard).
- Baseline thresholds or pass/fail indicators per task (T4.3).
- Parquet / SQLite / DuckDB output — JSON only for T4.1.
- cProfile aggregation from `startup.json` — the three coarse
  startup phase timings come from `training.json:runtime.startup_phase_times_s`,
  which is sufficient for T4.1. Rich profiles stay in `startup.json`
  and may be surfaced by T4.2's detail views.
- Partial-data pulling from non-zero-exit bundles (see §8 failure policy).
- Any UI.

## 4. Overview of changes

| Area | Change | File(s) |
|---|---|---|
| Aggregator core | `aggregate_dispatch(dispatch_dir, options)` | new `tools/odin/valhalla/aggregator.py` |
| Stats helpers | mean/std/min/max/cv_pct + divergence-z | new `tools/odin/valhalla/stats.py` |
| Atomic writer | tempfile + `os.replace` for `aggregate.json` | new `tools/odin/valhalla/writer.py` |
| CLI entry | `odin-aggregate` wrapper | new `tools/odin/valhalla/cli.py` |
| Package exports | Re-exports | new `tools/odin/valhalla/__init__.py` |
| T3.1 hook | Auto-aggregate at end of `run_dispatch`; `--skip-aggregate` opt-out | modify `tools/odin/asgard/runner.py`, `tools/odin/asgard/cli.py` |
| Tests | unit + integration | new `tools/odin/tests/test_valhalla_*.py` |

## 5. Aggregate JSON schema (v1.0)

### 5.1 Top-level shape

```json
{
  "schema_version": "1.0",
  "dispatch_id":    "20260423-114242",
  "generated_at":   "2026-04-23T12:00:00Z",
  "commit_sha":     "abc123",
  "hostnames":      ["valkyrie-01.internal", "valkyrie-02.internal"],
  "totals":         {"tasks": 54, "runs": 108, "completed": 105, "failed": 3},
  "rows":           [ /* see §5.2 */ ],
  "failures":       [ /* see §5.3 */ ]
}
```

- `commit_sha` — majority SHA across completed bundles' manifests.
  If bundles disagree, the majority wins and a warning line is emitted
  to stdout listing the minority SHAs. Partial re-dispatches after a
  commit change produce mixed SHAs; erroring out would be too strict.
- `hostnames` — union of `manifest.machine.hostname` across completed
  bundles, sorted.
- `totals.tasks` = distinct `(task, framework, backend)` triples seen.
  `totals.runs` = sum of seed entries across all rows. `totals.completed`
  and `totals.failed` are mutually exclusive over `totals.runs`.

### 5.2 Row shape (one per `(task, framework, backend)`)

```json
{
  "task":      "Isaac-Ant-Direct-v0",
  "framework": "rsl_rl",
  "backend":   "physx",
  "seeds": {
    "42": {
      "run_id":                  "rsl-rl_physx_Isaac-Ant-Direct-v0_...seed42",
      "status":                  "completed",
      "assigned_to":             "valkyrie-02.internal",
      "reward_final_ema":        6283.1,
      "ep_length_final_ema":     967.3,
      "iter_time_s_mean":        0.455,
      "iter_time_s_std":         0.110,
      "env_steps_per_s_mean":    288500.0,
      "iterations_completed":    300,
      "total_wall_time_s":       136.4,
      "ram_gb_peak":             8.3,
      "gpu_mem_gb_peak":         4.2,
      "startup_app_launch_s":    4.5,
      "startup_env_creation_s":  12.4,
      "startup_first_step_s":    0.0063
    },
    "43": { /* same shape */ }
  },
  "aggregate": {
    "n_seeds_completed": 2,
    "n_seeds_failed":    0,
    "reward_final_ema":     {"mean": 6200,   "std": 120,  "min": 6080,  "max": 6320,  "cv_pct": 1.94},
    "ep_length_final_ema":  {"mean": 950,    "std": 30,   "min": 920,   "max": 980,   "cv_pct": 3.16},
    "iter_time_s_mean":     {"mean": 0.46,   "std": 0.02, "min": 0.44,  "max": 0.48,  "cv_pct": 4.35},
    "env_steps_per_s_mean": {"mean": 290000, "std": 5000, "min": 285000, "max": 295000, "cv_pct": 1.72},
    "ram_gb_peak":          {"mean": 8.3,    "std": 0.2,  "min": 8.1,   "max": 8.5,   "cv_pct": 2.41},
    "gpu_mem_gb_peak":      {"mean": 4.2,    "std": 0.05, "min": 4.15,  "max": 4.25,  "cv_pct": 1.19}
  },
  "divergent_seeds": []
}
```

The aggregate block carries Stats blobs for **six** metrics only —
`reward_final_ema`, `ep_length_final_ema`, `iter_time_s_mean`,
`env_steps_per_s_mean`, `ram_gb_peak`, `gpu_mem_gb_peak`. Per-seed
fields that are not aggregated (`iter_time_s_std`,
`iterations_completed`, `total_wall_time_s`, `startup_*_s`) are either
derived quantities whose cross-seed aggregation has no obvious
interpretation (the std of per-seed stds) or are constant by design
(`iterations_completed == max_iterations` on success). The six
aggregated metrics are the ones the T4.2 dashboard will want for
cross-backend / cross-commit comparisons.

Per-seed field sources:

| Field | Source |
|---|---|
| `run_id` | bundle directory name |
| `status` | always `"completed"` in `seeds` (failed seeds go to `failures[]`) |
| `assigned_to` | `dispatch.json:jobs[<run_id>].assigned_to` |
| `reward_final_ema` | `training.json:learning.reward.final_ema` |
| `ep_length_final_ema` | `training.json:learning.ep_length.final_ema` |
| `iter_time_s_{mean,std}` | `training.json:runtime.iteration_time_s.{mean,std}` |
| `env_steps_per_s_mean` | `training.json:runtime.env_steps_per_s.mean` |
| `iterations_completed` | `training.json:runtime.iterations_completed` |
| `total_wall_time_s` | `training.json:runtime.total_wall_time_s` |
| `ram_gb_peak` | `training.json:resources.ram_gb.peak` |
| `gpu_mem_gb_peak` | `training.json:resources.gpu_mem_gb.peak` |
| `startup_*_s` | `training.json:runtime.startup_phase_times_s.{app_launch,env_creation,first_step}` |

Aggregate-block metrics are computed over *completed* seeds only:

- `mean` — arithmetic mean.
- `std` — population std (ddof=0) for n=2, sample std (ddof=1) for n≥3.
  For n=1 the field is `std: 0.0, cv_pct: 0.0` (no variance).
- `min`, `max` — element-wise.
- `cv_pct` — `100 * std / abs(mean)` when `mean != 0`, else `0.0`.

`divergent_seeds` — list of seed-string keys that fired the divergence
check on `reward_final_ema` specifically. A seed is divergent when
`|seed_value − mean| > z * std` (default `z = 2.0`). No seeds flag
when n < 3 (two-sample tests are not meaningful). The check runs
only on `reward_final_ema` for now; extending to other metrics is a
T4.2 or T4.3 concern.

### 5.3 Failures array

```json
"failures": [
  {
    "run_id":       "rsl-rl_newton_Isaac-Humanoid-Direct-v0_...seed42",
    "task":         "Isaac-Humanoid-Direct-v0",
    "framework":    "rsl_rl",
    "backend":      "newton",
    "seed":         42,
    "assigned_to":  "valkyrie-02.internal",
    "failure_kind": "hugin_crash",
    "failure_message": "RSL-RL subprocess exited 1 after 45s\n(last 16 KB of stderr: ...)"
  }
]
```

The `failure_kind` enum is T3.1's four-way classification:
`infrastructure | hugin_crash | hugin_malformed_bundle | timeout`. A
bundle that's missing from disk entirely (but expected per
`dispatch.json`) carries `failure_kind: "missing_bundle"` — a fifth
value synthesized by the aggregator.

## 6. Aggregator internals

### 6.1 `tools/odin/valhalla/stats.py`

```python
@dataclass
class Stats:
    mean: float
    std: float
    min: float
    max: float
    cv_pct: float

def stats_over(values: list[float]) -> Stats: ...
def is_divergent(values: list[float], z: float) -> list[int]:
    """Return the indices of values that are > z * std from mean.

    Returns [] when len(values) < 3. Uses ddof=1 (sample std) for n ≥ 3.
    """
```

### 6.2 `tools/odin/valhalla/aggregator.py`

```python
@dataclass
class AggregateOptions:
    divergence_z: float = 2.0

def aggregate_dispatch(
    dispatch_dir: Path,
    options: AggregateOptions = AggregateOptions(),
) -> dict:
    """Read dispatch.json + every <run_id>/ bundle, return aggregate dict.

    Does NOT write to disk — see :func:`valhalla.writer.write_aggregate`.
    Raises :class:`FileNotFoundError` if ``dispatch_dir/dispatch.json`` is
    absent.
    """
```

Execution flow:
1. Load `dispatch_dir/dispatch.json` — source of truth for which jobs
   existed.
2. For each `job` in `dispatch.jobs`:
   - Derive `(task, framework, backend, seed)` from `job.task_id`,
     `job.framework`, `job.backend`, `job.seed`.
   - Attempt to open `dispatch_dir/<run_id>/manifest.json` +
     `training.json`.
   - If successful (status=completed, exit_code=0, schema_version ok):
     append to the `seeds` dict of the matching row.
   - Otherwise: append to `failures[]` with the classification from
     `dispatch.json:jobs[...].failure` (or `missing_bundle` if the
     dispatch.json job entry itself shows status=completed but the
     bundle dir is missing — an anomaly worth reporting).
3. For each row: compute `aggregate.*` using `stats_over`; compute
   `divergent_seeds` via `is_divergent` on `reward_final_ema`.
4. Tally `totals`.
5. Return dict.

### 6.3 `tools/odin/valhalla/writer.py`

```python
def write_aggregate(
    dispatch_dir: Path,
    aggregate: dict,
    *,
    overwrite: bool = True,
) -> Path:
    """Atomically write ``<dispatch_dir>/aggregate.json`` and return its path.

    Uses tempfile + ``os.replace`` (same pattern as T3.1's dispatch.json
    writer). ``overwrite=False`` raises ``FileExistsError`` if the file
    already exists.
    """
```

### 6.4 `tools/odin/valhalla/cli.py`

`odin-aggregate <dispatch_id|LATEST>` entry point:

```bash
PYTHONPATH=. ./isaaclab.sh -p tools/odin/valhalla/cli.py <dispatch_id|LATEST> \
    [--runs-root odin_runs/] \
    [--divergence-z 2.0] \
    [--overwrite] \
    [--quiet]
```

- `<dispatch_id|LATEST>` — required positional. `LATEST` resolves to the
  most recent subdirectory of `runs-root` (mirrors T3.1's `resolve_dispatch_dir`).
- `--runs-root` (default `odin_runs/`) — where dispatch dirs live.
- `--divergence-z` (default `2.0`) — threshold for `divergent_seeds`.
- `--overwrite` — default true; `--no-overwrite` raises on pre-existing
  `aggregate.json`.
- `--quiet` — suppresses the "wrote N rows / M failures" summary line.

## 7. T3.1 auto-aggregate hook

Modify `tools/odin/asgard/runner.py:run_dispatch` and
`tools/odin/asgard/cli.py`:

- Add `DispatchOptions.skip_aggregate: bool = False`.
- At the end of `run_dispatch`, after `state.ended_at = _utc_now_iso()`
  and `write_dispatch_state(...)`, if `not options.skip_aggregate`:
  call `valhalla.aggregator.aggregate_dispatch(dispatch_dir)` and
  `valhalla.writer.write_aggregate(dispatch_dir, agg)`. Any exception
  from the aggregator is caught, logged to stdout as
  `[WARNING] aggregate step failed: ...`, and does not mask the
  dispatch's own return value.
- Expose `--skip-aggregate` on the Asgard CLI as a passthrough.

The aggregation step is observationally idempotent and fast (no
training, just file parsing) — running it by default is the right trade.

## 8. Failure handling policy

Per decision in brainstorming Q6: **strict whitelist**. A seed is
`"completed"` in the `seeds` dict *only if* all three conditions hold:

1. `<run_id>/` directory exists under the dispatch dir.
2. `manifest.json` parses with `phases.training.status == "completed"`
   and `phases.training.exit_code == 0`.
3. `training.json` parses with `schema_version` matching `"1.*"`
   (minor/patch bumps accepted, major bumps rejected).

Anything else routes the seed to `failures[]`. The `failure_kind` comes
from:

- `dispatch.json:jobs[...].failure.kind` when that job is marked
  `status=failed` in the dispatch state.
- `"missing_bundle"` when the dispatch job is marked `completed` but
  no directory exists (anomaly — logged).
- `"malformed_bundle"` when the directory exists but (2) or (3) fails
  — overrides whatever the dispatch state thought.

Cross-seed aggregates compute over completed seeds only. Rows with
zero completed seeds emit `aggregate: null` and
`n_seeds_completed: 0`. Such rows still appear in `rows[]` so the
dashboard can surface "every seed failed for this task" prominently.

## 9. Testing strategy

### 9.1 Unit tests in `tools/odin/tests/test_valhalla_stats.py`

- `stats_over` on known lists: `[1,2,3,4,5]` → mean=3, std≈1.58 (sample),
  min=1, max=5, cv_pct≈52.7.
- `stats_over` on `[]`, `[5]`, `[5, 5]` — edge cases; std=0 for n=1,
  population std for n=2.
- `is_divergent([1,1,1,1,1,10], z=2.0)` → `[5]`; with `z=3.0` → `[]`.
  (The `(n-1)/sqrt(n-1)` bound on a single-outlier z-score requires n ≥ 6
  to clear `z=2.0` under strict `>` comparison.)
- `is_divergent` on n < 3 always returns `[]`.
- `cv_pct` when mean=0 returns 0.0 (not NaN).

### 9.2 Unit tests in `tools/odin/tests/test_valhalla_aggregator.py`

Build synthetic dispatch directories under `tmp_path` with fake
`dispatch.json`, `manifest.json`, `training.json` files:

- **Happy path**: two completed seeds for one row. Assert:
  - `seeds["42"]` / `seeds["43"]` populated from training.json.
  - `aggregate.reward_final_ema.mean` equals the expected value.
  - `divergent_seeds == []`.
- **Divergent seed**: three seeds with one outlier. Assert the
  outlier appears in `divergent_seeds`.
- **Mixed completed + failed**: two completed, one failed. Assert
  failed seed lands in `failures[]` with correct `failure_kind`,
  aggregate computes over the 2 completed.
- **All seeds failed**: row has `aggregate: null`,
  `n_seeds_completed: 0`, and all three entries land in `failures[]`.
- **Missing bundle dir**: `dispatch.json` says completed but
  no directory — synthesized `failure_kind: "missing_bundle"`.
- **Malformed training.json** (bad schema_version): overrides
  dispatch.json's completed status, failure_kind=`"malformed_bundle"`.
- **Commit SHA mismatch across bundles**: majority wins, warning to
  stdout (captured with `capsys`).
- **Empty dispatch** (no jobs): aggregator returns
  `rows=[], failures=[], totals.tasks=0`.

### 9.3 Unit tests in `tools/odin/tests/test_valhalla_writer.py`

- Atomic write creates the file and no leftover tempfile.
- `overwrite=False` on pre-existing file raises `FileExistsError`.
- Concurrent-write simulation (writer called twice in rapid succession)
  doesn't corrupt; final file is valid JSON.

### 9.4 Unit tests in `tools/odin/tests/test_valhalla_cli.py`

Mirror-parser pattern for argparse; subprocess-level tests for the
`LATEST` resolution and `--quiet`/`--overwrite` behavior.

### 9.5 Integration test in `tools/odin/tests/test_valhalla_integration.py`

Drive T3.1's `run_dispatch` with fake SSH + rsync runners and fake
Hugin/Munin output that writes realistic manifest + training JSON
shapes. Assert `aggregate.json` lands in the dispatch dir with the
expected row/failure counts.

### 9.6 Regression-guarantee discipline

For every test that verifies the fix of a failure-handling branch,
temporarily revert the handling code and confirm the test fails —
per IsaacLab's `AGENTS.md` "Always verify regression tests fail
without the fix" rule.

## 10. Real-fleet validation plan

T4.1 is **not considered done** until the aggregator has been run
against real bundles produced on the user's runner fleet. This is
also the first end-to-end exercise of T3.1's dispatcher against a
real (non-loopback) fleet.

### 10.1 Validation steps

1. **Pick a curated subset**: ~10-15 tasks across `direct/` and
   `manager_based/` categories, covering both PhysX and Newton where
   available. A rough shape: a handful of locomotion tasks
   (Ant, Humanoid, Cartpole, Quadcopter, Cassie), a handful of
   manipulation tasks (Franka-Lift, Franka-Cabinet, Allegro-Hand),
   and a few more locomotion-on-terrain tasks (Anymal-C-Flat,
   Spot-Flat, Unitree-Go1-Flat). Exact list chosen by the operator
   at dispatch time. Filter the T2.1 `physx_envs.yaml` /
   `newton_envs.yaml` via `--include` on the dispatch CLI.
2. **Pick seeds**: `--seeds 42,43,44` (three for meaningful
   cross-seed stats including divergence detection).
3. **Pick fleet**: user-provided runner machines, configured in a
   `fleet.yaml` that lists them all.
4. **Dispatch**: `PYTHONPATH=. ./isaaclab.sh -p tools/odin/asgard/cli.py
   --fleet fleet.yaml --physx-yaml ... --newton-yaml ... --seeds 42,43,44
   --include '<patterns>' --verbose`. Auto-aggregate fires at end.
5. **Inspect** `odin_runs/<dispatch_id>/aggregate.json` — visually
   scan for shape correctness, sanity of numeric values, expected
   row count, expected failure handling.
6. **Debug loop**: any T3.1 or T4.1 issue found gets fixed, the
   relevant bundle(s) re-generated (via `--retry-failed` or a full
   re-dispatch), and aggregator re-run via `odin-aggregate`.

### 10.2 Success criteria

- Dispatch completes with all jobs either `completed` or `failed`
  (no in-flight left over).
- `aggregate.json` is produced automatically, parses, and has one
  row per unique `(task, framework, backend)` and matching seed
  counts.
- Any task where all seeds failed shows `aggregate: null` but still
  appears in `rows[]` (no silent drop).
- `divergent_seeds` fires on at least one expected noisy task or
  returns empty for everything (both are valid — the point is the
  code path runs).
- Hostnames across runs match the fleet.
- Aggregator wall time under 10 seconds for the ~30-60 bundle sample.

### 10.3 What this surfaces

This pass jointly exercises:

- T3.1 SSH transport on real remote hosts (prior integration test
  was loopback-only).
- T3.1 rsync provisioning against real `isaaclab_path` + docker
  containers.
- T3.1 real docker exec invocation of Hugin/Munin.
- T4.1 aggregator on non-synthetic bundle shapes.
- The full JSON schemas end-to-end on real training data.

Expected outcome: a small list of real bugs to fix before starting
T4.2. The validation is run interactively — not as part of the plan's
automated TDD flow.

## 11. Out of scope

Re-stating the non-goals as explicit non-actions:

- No cross-dispatch aggregation or index file.
- No dashboard UI.
- No baseline thresholds.
- No Parquet / SQLite / DuckDB output.
- No upstream IsaacLab changes.
- No changes to the bundle layout or the T1 v1.0 schemas —
  `aggregate.json` is a derived artifact.
- No commit_sha reconciliation beyond "majority wins + warn".
- No partial-data pulling from non-zero-exit bundles.
