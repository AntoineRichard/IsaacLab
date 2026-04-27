# Odin Preset-Handling — Design

**Status:** approved
**Date:** 2026-04-27
**Task covered:** Stop Odin from dispatching `(task, backend)` pairs the task
doesn't actually support. Surfaced during T4.1's real-fleet validation
(2026-04-24): six of fifteen jobs failed with `ValueError: Unknown
preset(s): physx` because `Isaac-Velocity-Flat-Anymal-C-Direct-v0` and
`Isaac-Quadcopter-Direct-v0` don't have a `physx` preset registered
in their hydra config. The benchmark scripts' unconditional
`presets=<backend>` injection turned an upstream metadata gap into a
runtime crash. This spec adds Odin-side awareness of which presets
each task supports and routes unsupported pairs into a new
`skipped[]` array on `dispatch.json`.

## 1. Motivation

On 2026-04-24 the first physx-only T4.1 sweep ran 5 tasks × 3 seeds × 2
hosts. Three tasks (Ant, Cartpole, Humanoid) completed all 9 of their
seeds cleanly with full `aggregate.json` populated. Two tasks failed
all 6 of their seeds with the same error:

```
ValueError: Unknown preset(s): physx
  File "isaaclab_tasks/utils/hydra.py", line 387, in register_task
    raise ValueError(_format_unknown_presets_error(unknown, display))
```

Both `Isaac-Velocity-Flat-Anymal-C-Direct-v0` and
`Isaac-Quadcopter-Direct-v0` are pre-preset-system tasks: their
`env_cfg.sim.physics` is a plain `PhysxCfg`, not a `PresetCfg`
wrapper, so they have no named presets at all. They run perfectly
when invoked without `presets=physx` — the task default is physx.

But `scripts/benchmarks/benchmark_rsl_rl.py:117-122` (and the SKRL
twin) unconditionally prepend `presets=<args_cli.backend>` to
`hydra_args` whenever `--backend` is set. This was added as part of
the T1 bundle-fix to force the requested backend on dual-preset
tasks; on no-preset tasks it crashes with the generic Hydra error
above.

The T4.1 failures are real (the dispatcher made the operator wait
~25 minutes for those six jobs to fail), but they're also entirely
preventable: Odin already enumerates every task's preset support via
`has_physics_preset(raw_cfg, name)` at T2.1 enumeration time. We
simply don't surface or use that information at dispatch time.

## 2. Goals

- **Asgard skips** `(task, backend)` pairs the task doesn't support
  — they never reach the worker queue, so no GPU time is wasted and
  no `hugin_crash` failures pollute `dispatch.json`.
- **Yaml is the source of truth.** `presets_available: list[str]`
  becomes a populated field on every `EnvEntry`, written by the
  enumerator and consumed by the dispatcher.
- **Visible telemetry.** Filtered jobs land in a new top-level
  `skipped[]` array on `dispatch.json`; the dispatch's stdout summary
  includes a one-line block listing what was skipped and why.
- **Defense in depth.** When yaml drifts from runtime reality
  (re-enumeration is stale, a task got a preset added/removed
  between yaml-stamp and dispatch), the benchmark scripts fail fast
  with an actionable `preset_unsupported:` error and the worker
  classifier maps that into a distinct failure kind — not a
  catch-all `hugin_crash`.
- **Backward compatible.** Yaml without the new field still loads
  (treated as "unknown — pass through"), `dispatch.json`
  schema_version bumps minor (1.0 → 1.1, additive), `failure_kind`
  enum is extended (existing readers ignore unknown values per
  Odin's schema rules in `docs/odin/architecture.md` §5).

## 3. Non-goals

- **No live re-enumeration on dispatch.** Asgard does not import
  `isaaclab_tasks.utils.parse_cfg` to query presets at queue time.
  The yaml is the contract; if it's stale, the runtime safety net
  catches it.
- **No silent backend swap.** When a task has no requested preset,
  Odin does not fall back to the task's default backend "to be
  helpful" — the operator asked for physx and physx isn't there;
  that's a `skipped` (queue-time) or `preset_unsupported` (runtime)
  outcome, never a silent reroute.
- **No flag-gating.** This is a one-shot landing, not a behind-flag
  rollout. The schema bumps are additive; the benchmark scripts'
  new failure mode supersedes a worse failure mode.
- **No T2.1 yaml hand-edits.** The two affected rows
  (Anymal-C Flat + Quadcopter) get their `presets_available: []`
  populated by re-running the enumerator, not by manual yaml edit.
- **Dashboard rendering of `skipped[]` is out of scope here.** That
  belongs to T4.2 (Valhalla dashboard).

## 4. Architecture overview

Four layers of change, one consistent direction:

```
[Enumeration]      enumerate_{physx,newton}_envs.py
                   → calls has_physics_preset(raw_cfg, name) per task
                   → stamps presets_available: [...] into yaml row

[YAML schema]      EnvEntry gains a new field; physx_envs.yaml +
                   newton_envs.yaml are re-enumerated once

[Asgard queue]     _expand_env_list filters out (task, backend) pairs
                   where backend ∉ row.presets_available;
                   filtered jobs go into a new top-level
                   dispatch.json `skipped[]` array (not jobs[])

[Benchmark
 scripts]          benchmark_{rsl_rl,skrl}.py learns to fail fast
                   with an actionable error when --backend X is
                   passed but the task lacks an X preset

[Worker
 classification]   ValkyrieWorker._classify recognises the new
                   error pattern → failure_kind=preset_unsupported
                   (instead of catch-all hugin_crash)
```

Yaml is the **primary** filter — almost all `(task, backend)` gaps
get caught here. Benchmark-script + worker classifier are the
**safety net** — drift between yaml and reality (stale yaml, new
preset added since enumeration, etc.) gets caught at runtime with a
clear classification, not a generic crash.

## 5. Schema changes

### 5.1 `EnvEntry` (`tools/odin/common/env_list.py`)

One new field:

```python
@dataclass
class EnvEntry:
    ...
    presets_available: list[str] = field(default_factory=list)
    # e.g. ["physx", "newton"], or [] if the task has no preset system
```

`list[str]` over two booleans because: (a) it scales to a third
backend without another schema bump, (b) it captures the "no preset
system at all" case naturally as `[]`, (c) it's exactly the shape
the enumerator's `has_physics_preset` calls produce.

**Backward compat**: `default_factory=list` so old yaml without the
field deserializes with `presets_available=[]`. The Asgard filter
treats `presets_available == []` as **"unknown — let it through, the
runtime safety net catches if missing"**, so no behaviour change for
yaml that hasn't been re-enumerated yet.

### 5.2 Yaml row

One new key per task, written by the enumerator:

```yaml
- task_id: Isaac-Velocity-Flat-Anymal-C-Direct-v0
  ...
  notes: ''
  presets_available: []     # ← new
```

### 5.3 `dispatch.json` (`tools/odin/asgard/state.py`, `runner.py`)

`schema_version` bumps `1.0 → 1.1` (additive). New top-level
`skipped[]` array alongside `jobs[]`:

```jsonc
{
  "schema_version": "1.1",
  "dispatch_id": "...",
  "jobs": [...],
  "skipped": [
    {
      "task_id": "Isaac-Velocity-Flat-Anymal-C-Direct-v0",
      "framework": "rsl_rl",
      "backend": "physx",
      "seed": 42,
      "reason": "preset_unsupported",
      "presets_available": []
    },
    ...
  ],
  ...
}
```

**Reader compatibility**: `tools/odin/asgard/state.py:142-144` today
checks `got_schema != SCHEMA_VERSION` (strict equality) and raises
`ValueError("Unsupported dispatch.json schema_version ...")` on
mismatch. Bumping `SCHEMA_VERSION = "1.1"` without softening that
check would make every pre-1.1 `dispatch.json` unreadable on
`--resume`. Two coupled changes are needed:

1. `state.py:SCHEMA_VERSION` → `"1.1"`.
2. The validator changes from strict equality to "accept any
   `schema_version` whose major matches `SCHEMA_VERSION`'s major"
   — i.e. parse on `"."` and compare the leading int. This matches
   the architecture-doc rule (additive minor bumps must be
   tolerated by readers) and keeps T3.1's resume-from-1.0 dispatches
   readable. The new field defaults to `[]` when missing on read,
   so 1.0 files load cleanly with `skipped == []`.

T4.1 aggregator currently ignores `dispatch.json.schema_version`
entirely — it reads `jobs[]` and `failures[]` defensively — so it's
already tolerant. Other unknown top-level keys are passed through.

### 5.4 `FailureInfo.kind` (`tools/odin/asgard/jobs.py`)

Enum extends with one new value:

```
"infrastructure" | "hugin_crash" | "hugin_malformed_bundle"
| "timeout" | "preset_unsupported"
```

The aggregator's failure-kind whitelist accepts the new value; one
line change in `tools/odin/valhalla/aggregator.py`.

### 5.5 New `SkippedEntry` dataclass

```python
@dataclass
class SkippedEntry:
    task_id: str
    framework: str
    backend: str
    seed: int
    reason: str                          # "preset_unsupported" today; extensible
    presets_available: list[str]         # what WAS available; for the operator
```

Lives next to `JobEntry` in `tools/odin/asgard/jobs.py`. JSON
serialised by the same atomic writer that handles `JobEntry`.

## 6. Enumeration update

`tools/odin/common/env_list.py:build_entry_from_task_spec` gains one
helper-call:

```python
def build_entry_from_task_spec(...) -> EnvEntry:
    ...
    raw_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point", ...)
    presets = []
    for name in ("physx", "newton"):
        if has_physics_preset(raw_cfg, name):
            presets.append(name)
    return EnvEntry(
        task_id=task_id,
        ...
        presets_available=presets,
    )
```

Both enumerators (`tools/odin/scripts/enumerate_physx_envs.py` and
`enumerate_newton_envs.py`) flow through this builder, so populating
the field once handles both yaml files. The newton enumerator's
existing `classify_for_newton(raw_cfg)` keeps doing its
narrative-classification job — populating `presets_available` is
independent.

**Migration**: re-run both enumerators once after the implementation
lands. Both enumerators already auto-merge with the existing yaml via
`tools.odin.common.env_list.merge`, so human-edited fields like
`keep`, `notes`, `status`, `suspected_gap` are preserved by default —
no flag needed.

```bash
PYTHONPATH=. ./isaaclab.sh -p tools/odin/scripts/enumerate_physx_envs.py
PYTHONPATH=. ./isaaclab.sh -p tools/odin/scripts/enumerate_newton_envs.py
```

Both scripts ship default `--output` / `--newton-output` paths
pointing at `tools/odin/config/*.yaml`, so no flags are needed for
the standard run.

The yaml diffs are committed in the same PR as the code changes so
the change is self-validating.

## 7. Asgard queue-time filter

`tools/odin/asgard/jobs.py:_expand_env_list` returns a `(jobs,
skipped)` tuple instead of a bare list:

```python
def _expand_env_list(
    yaml_path: Path,
    backend: str,
    seeds: list[int],
    dispatch_id: str,
    include_filter: list[str] | None = None,
) -> tuple[list[JobEntry], list[SkippedEntry]]:
    env_list = load_env_list(yaml_path)
    jobs: list[JobEntry] = []
    skipped: list[SkippedEntry] = []
    for group_rows in env_list.groups.values():
        for row in group_rows:
            if not row.keep or row.status == "stale":
                continue
            if not _apply_include_filter(row.task_id, include_filter):
                continue
            if row.framework is None or row.num_envs is None or row.max_iterations is None:
                continue

            # Preset-support gate. Empty list = unknown → pass through;
            # populated list with backend missing → skip.
            if row.presets_available and backend not in row.presets_available:
                for seed in seeds:
                    skipped.append(SkippedEntry(
                        task_id=row.task_id,
                        framework=row.framework,
                        backend=backend,
                        seed=seed,
                        reason="preset_unsupported",
                        presets_available=list(row.presets_available),
                    ))
                continue

            for seed in seeds:
                jobs.append(JobEntry(...))
    return jobs, skipped
```

Filter ordering is intentional: `--include` filter runs *before* the
preset filter, so rows that didn't match the operator's include
pattern don't pollute `skipped[]` (those are deliberate exclusions,
not capability gaps).

`tools/odin/asgard/runner.py:run_dispatch`:
- Calls `_expand_env_list(...)` once per backend (physx, newton);
  concatenates both `jobs` lists and both `skipped` lists.
- Writes the merged `skipped` into `dispatch.json` alongside `jobs`.
- Pre-dispatch stdout block: when `skipped` is non-empty:

  ```
  [INFO] Skipping 6 (task, backend) pairs with no preset support:
  [INFO]   Isaac-Velocity-Flat-Anymal-C-Direct-v0 × physx (seeds 42, 43, 44) — available: []
  [INFO]   Isaac-Quadcopter-Direct-v0 × physx (seeds 42, 43, 44) — available: []
  ```

- Final dispatch summary line gains a counter:

  ```
  odin-dispatch: 9 completed, 0 failed, 6 skipped (preset_unsupported), 0 pending out of 15 total
  ```

**Resume semantics**: when `--resume` re-loads a `dispatch.json`,
the existing `skipped[]` is preserved verbatim. We never re-run
skipped entries; we don't re-evaluate them on resume even if the
yaml has since changed (the dispatch's identity is fixed at first
write).

## 8. Benchmark-script defense + worker classification

### 8.1 `scripts/benchmarks/benchmark_{rsl_rl,skrl}.py`

The unconditional `presets=` injection guards against missing
presets:

```python
if args_cli.backend is not None:
    existing = [a for a in hydra_args if a.startswith("presets=")]
    if existing:
        print(f"[WARNING] --backend ignored because {existing[0]} explicit.")
    else:
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from isaaclab_tasks.utils.presets import has_physics_preset
        raw_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
        if not has_physics_preset(raw_cfg, args_cli.backend):
            sys.stderr.write(
                f"[ERROR] preset_unsupported: task {args_cli.task!r} has no "
                f"{args_cli.backend!r} preset. Inspect raw_cfg.sim.physics or "
                f"re-enumerate {{physx,newton}}_envs.yaml.\n"
            )
            sys.exit(2)
        hydra_args = [f"presets={args_cli.backend}"] + hydra_args
```

Three intentional choices:

- **Exit 2** (not 1) so the worker classifier can disambiguate via
  exit code if the stderr regex ever fails. Stderr `preset_unsupported:`
  prefix is the primary signal; exit code is the fallback.
- **`preset_unsupported:` prefix** is a stable magic string the
  worker classifier matches on. Specific enough to not collide with
  arbitrary training output, plain text so it survives any
  intermediate logging shenanigans.
- **Same fix in both wrappers** — the duplication between
  `benchmark_rsl_rl.py` and `benchmark_skrl.py` is real but factoring
  the shared logic is out of scope for this spec; just port the same
  shape into both.

### 8.2 `tools/odin/asgard/worker.py:_classify`

One new branch:

```python
def _classify(self, r: SSHResult, job: JobEntry, ssh_tail: Path) -> FailureInfo | None:
    if r.timed_out:
        return FailureInfo(kind="timeout", ...)
    if r.exit_code in _INFRASTRUCTURE_DOCKER_EXIT_CODES:
        return FailureInfo(kind="infrastructure", ...)
    if r.exit_code != 0:
        if "preset_unsupported:" in (r.stderr or ""):
            return FailureInfo(
                kind="preset_unsupported",
                message="benchmark script reported missing preset",
                details={
                    "exit_code": r.exit_code,
                    "log_tail_path": str(ssh_tail.relative_to(self._dispatch_dir)),
                },
            )
        return FailureInfo(kind="hugin_crash", ...)
    return None
```

### 8.3 Aggregator awareness

`tools/odin/valhalla/aggregator.py`'s failure-kind whitelist accepts
the new `preset_unsupported` value; otherwise no aggregator logic
changes. Failures of this kind flow into `failures[]` cleanly.
Whether `aggregate.json` should also surface the dispatch's
`skipped[]` array is deferred to T4.2 (dashboard); the aggregator
already passes through unknown top-level fields, so this spec needs
no change to the aggregator's reading of `dispatch.json`.

## 9. Migration & rollout

The change set is additive. Order of landing matters because the
Asgard filter has to see populated `presets_available` lists or it'll
be a no-op (per §5.1 backward-compat rule, empty list = pass-through).

Single PR, ordered commits:

1. **Schema + helpers** — `EnvEntry.presets_available` field,
   `SkippedEntry` dataclass, `dispatch.json` schema_version → 1.1,
   `FailureInfo` kind extension. No behaviour change yet (filter
   doesn't act on empty lists).
2. **Enumeration** — `build_entry_from_task_spec` calls
   `has_physics_preset`. No yaml diff yet (we haven't re-run the
   enumerators).
3. **Re-enumeration commit** — run both enumerators, commit the
   resulting yaml diffs. This is the only commit that touches yaml
   content. Now every row has `presets_available: [...]`.
4. **Asgard filter + dispatch.json `skipped[]`** — the new filter
   logic in `_expand_env_list` plus the writer change. From this
   commit on, dispatches start dropping unsupported pairs and
   writing them to `skipped[]`.
5. **Benchmark-script defense + worker classifier** — fail-fast in
   benchmark scripts, `preset_unsupported` classification. Safety
   net is live.
6. **Aggregator awareness** — accept `preset_unsupported` in the
   failure-kind whitelist. (One line.)

Steps 5+6 are independent of yaml state and can land before step 3
if execution prefers; the architecture only requires that **step 3
lands before any operator runs a real dispatch and expects skips to
show up**.

No flag-gating, no migration window. Schema bumps are additive (1.0
readers see `skipped: []` and ignore it; 1.1 writers don't break old
aggregators). The "empty list = pass-through" rule means yaml that
hasn't been re-enumerated still works (just falls into the runtime
safety net).

## 10. Test plan

### Unit tests

- **`tools/odin/tests/test_env_list.py`** (extend):
  - Round-trip yaml load + dump preserves `presets_available`
    (incl. empty list).
  - Backward-compat: yaml without the field loads with
    `presets_available=[]`.
  - `build_entry_from_task_spec` populates the field correctly for
    the four cases (`[]`, `[physx]`, `[newton]`,
    `[physx, newton]`) — `has_physics_preset` mocked.

- **`tools/odin/tests/test_asgard_queue.py`** (extend — already
  covers `_expand_env_list`):
  - `presets_available=[physx]` + backend=physx → 1 JobEntry per
    seed, 0 skipped.
  - `presets_available=[physx]` + backend=newton → 0 jobs, 1
    SkippedEntry per seed with `reason="preset_unsupported"` and
    `presets_available=[physx]`.
  - `presets_available=[]` (unknown) + any backend → 1 JobEntry per
    seed (pass-through, runtime catches).
  - `presets_available=[physx, newton]` + either backend → 1
    JobEntry per seed.
  - `--include` filter applies before preset filter (skipped rows
    that didn't match include don't appear in `skipped[]`).

- **`tools/odin/tests/test_asgard_state.py`** (extend — already
  covers `dispatch.json` atomic-write round-trip):
  - `skipped[]` round-trips through atomic write + read.
  - Reading a 1.0 file with no `skipped` key produces an empty
    list (backward-compat).
  - Reading a 1.1 file produces the populated list.
  - `schema_version == "1.1"` written.

- **`tools/odin/tests/test_asgard_runner.py`** (extend):
  - Resume preserves `skipped[]` verbatim (never re-evaluated even
    if the yaml has changed since first write).

- **`tools/odin/tests/test_asgard_worker.py`** (extend):
  - `exit_code != 0` + stderr containing `"preset_unsupported:"`
    → `FailureInfo(kind="preset_unsupported", ...)`.
  - `exit_code != 0` + stderr without that prefix → existing
    `hugin_crash` classification (regression).

- **`scripts/benchmarks/tests/test_benchmark_rsl_rl_cli.py`** +
  **`test_benchmark_skrl_cli.py`** (extend):
  - Mock `has_physics_preset` returning False → script `sys.exit(2)`
    with stderr containing `"preset_unsupported:"`.
  - Mock returning True → original behaviour unchanged (preset
    injected into hydra_args).
  - `presets=` already explicit in hydra_args → validation skipped,
    warning printed (current behaviour preserved).

- **`tools/odin/tests/test_valhalla_aggregator.py`** (extend):
  - Failure with `kind="preset_unsupported"` flows into
    `failures[]` cleanly.

### Integration test

- **`tools/odin/tests/test_asgard_integration.py`** (extend):
  - Build env_list with one (task, backend) unsupported.
  - Run `run_dispatch` end-to-end.
  - Verify final `dispatch.json` has `skipped[]` populated and
    `jobs[]` excludes the unsupported pair.

Roughly **15–20 new tests** plus 2-3 modifications to existing
tests. No live-fleet validation needed — a fresh dispatch with the
re-enumerated yaml against the existing T4.1 fleet will exercise
the path end-to-end.

## 11. Risks & open questions

- **Yaml drift between enumeration and dispatch.** If a task's
  preset structure changes upstream (preset added or removed) and
  the operator forgets to re-enumerate, `presets_available` is
  stale. Two cases: (a) yaml says supported but task no longer is →
  the runtime safety net catches it as `preset_unsupported`. (b)
  yaml says unsupported but task now is → operator never sees the
  job; mitigated by re-enumeration being a one-line CLI invocation
  and reviewer-visible in the PR diff. Acceptable.

- **Tasks that have a preset system but no named presets at all.**
  If `raw_cfg.sim.physics` is a `PresetCfg` wrapper with zero named
  presets defined (theoretical edge case),
  `has_physics_preset(...)` returns False for any name; the row's
  `presets_available` is `[]`; the filter passes the row through.
  Runtime safety net then catches it. No special handling needed.

- **Operator wants to force a backend on a no-preset task.** Today
  this would be `--include` against a task with `presets_available
  == []` plus the operator knowing the task default already matches
  the requested backend. The benchmark script's pre-injection
  `existing_presets` check still honours an explicit
  `presets=<X>` in `hydra_args`, so a hand-injected preset
  bypasses our validation. Out of scope — operator override is by
  design.

- **Schema_version bump risk.** Going 1.0 → 1.1 is additive in
  payload, but `tools/odin/asgard/state.py:143` currently does a
  strict-equality check against `SCHEMA_VERSION` and would reject
  pre-1.1 files. §5.3 covers the coupled change: bump
  `SCHEMA_VERSION = "1.1"` AND switch the validator from strict
  equality to major-version match. Existing 1.0 `dispatch.json`
  files (e.g. `--resume` of an in-flight T3.1 dispatch) load
  cleanly with `skipped == []`. The T4.1 aggregator doesn't check
  `dispatch.json.schema_version` at all, so it's unaffected.

- **Hugin/Munin wrapper layer**: this spec doesn't change Hugin or
  Munin's wrapper scripts at all. They pass `--backend` through;
  the benchmark scripts now do the validation. Choice was to keep
  the wrappers as thin bundle-output shapers, not duplicate the
  preset-resolution logic. The trade-off is that direct invocations
  of Hugin/Munin (rare; dispatcher is the primary caller) get the
  same defence as direct invocations of the benchmark scripts.

## 12. Out of scope

- **Dashboard rendering of `skipped[]`** — T4.2.
- **Re-aggregating prior dispatches** with the new schema — old
  `dispatch.json` files don't get rewritten.
- **Collapsing the duplicated preset-injection logic** between
  `benchmark_rsl_rl.py` and `benchmark_skrl.py` into a shared
  helper — separate refactor.
- **Cross-fleet migration tooling** — n/a, single-tree change.

## Summary

Stamp `presets_available: list[str]` on every yaml row at
enumeration time; gate Asgard's queue-build by it; route skipped
pairs into a new top-level `skipped[]` on `dispatch.json`; harden
the benchmark scripts with a fail-fast safety net keyed on
`preset_unsupported:`; teach the worker to classify that as a
distinct failure kind. Six commits, one re-enumeration, two
schema bumps, no behaviour regressions on yaml that hasn't been
re-stamped yet.
