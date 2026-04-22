# Odin T2.2 — Startup Profiling Survey & Reliability Fixes Design

**Project:** Odin (multi-backend IsaacLab evaluation harness)
**Task:** T2.2 — dense startup profiling survey
**Date:** 2026-04-22
**Branch:** `antoiner/feat/odin`
**Status:** Draft — pending user review

## Context

T1 shipped `benchmark_startup.py` with five cProfile phases (`app_launch`,
`python_imports`, `task_config`, `env_creation`, `first_step`) and a schema
v1.0 `StartupBundle` that Odin wrappers fold into the per-run bundle. It also
shipped `MemoryInfoRecorder` and `GPUInfoRecorder` that track mean / std /
count via Welford's algorithm. The T1 change-log in
`docs/odin/architecture.md` explicitly carried two reliability caveats to
T2.2:

1. `CProfileFunction.calls` is always `0` because the upstream
   `parse_cprofile_stats` helper (`scripts/benchmarks/utils.py`) returns
   `(label, tottime, cumtime)` triples and drops the `ncalls` field that
   CPython's internal `pstats.Stats.stats` already carries.
2. `Resources.*.peak` in `training.json` falls back to `mean` because the
   recorders don't track peak alongside the Welford stats.

T2.2's brief from `eval_plan.md` is to "explore the dense profiling
capabilities in IsaacLab to see what should be reported." Scoping during
brainstorming (2026-04-22) settled on **framing B**: survey the current
profiling surface + close the two T1 caveats + re-tune the
`startup_whitelist.yaml`. No schema bump (consumers just want the fields
populated, not a version gate), no speculative new phases or metrics
(T4 / Valhalla will identify those empirically once it exists).

## Goals & non-goals

**In scope:**

- **Survey doc** `docs/odin/startup_profiling_survey.md` covering all five
  phases, cross-cutting interpretation guidance, and whitelist rationale.
  Content grounded in a fresh local run of `benchmark_startup.py` against
  `Isaac-Ant-Direct-v0` on `antoiner/feat/odin`.
- **Whitelist re-tuning** in `scripts/benchmarks/startup_whitelist.yaml`
  to cover all five phases with recommended patterns (or an explicit
  comment declaring a phase stays on `top_n` fallback, with rationale).
- **Fix 1** — `parse_cprofile_stats` returns 4-tuples
  `(label, tottime_ms, cumtime_ms, ncalls)`. The one call site in
  `benchmark_startup.py` is updated to pass the real `ncalls` into
  `CProfileFunction.calls` instead of `0`.
- **Fix 2** — `MemoryInfoRecorder` and `GPUInfoRecorder` track peak
  alongside Welford mean / std. New runtime-info keys: `rss_peak`,
  `vms_peak`, `uss_peak`, `gpu_mem_peak[i]`, `gpu_util_peak[i]`.
  `benchmark_rsl_rl.py` / `benchmark_skrl.py` populate
  `Resources.*.peak` from the real peak instead of copying `mean`.
- **Tests** — peak coverage bolted onto the existing
  `TestMemoryInfoRecorder` / `TestGPUInfoRecorder` classes in
  `test_recorders.py`; a new `test_parse_cprofile_stats.py` for the
  `ncalls` contract (`parse_cprofile_stats` has no existing test file).
- **IsaacLab CHANGELOG + extension version bump** (patch) noting both
  fixes.

**Out of scope:**

- Schema version bump — stay at `1.0`. Both fixes land as behaviour
  corrections to existing fields, not additive or renamed fields.
- New phases / metrics (GPU memory timeline, import-tree breakdown,
  USD-asset load timing, warp kernel compile time, Kit-subsystem cost).
  Documented as open questions in the survey doc's §5 to seed T4.
- Cross-commit / cross-backend comparison tooling — T4's concern.
- Regenerating the T1 dry-run bundles under `odin_runs/` — they have a
  separate corruption bug tracked elsewhere (stale TB copies + identical
  reward series across backends). T2.2 emits correct bundles on fresh
  runs; the T1 dry-run replay is a different cleanup.
- Any change to `benchmark_rsl_rl.py` / `benchmark_skrl.py` beyond wiring
  the peak values into the `Resources` payload.
- Any change to Odin-side (`tools/odin/**`) code. T2.2 is entirely
  upstream IsaacLab.

**Success criteria:**

1. `docs/odin/startup_profiling_survey.md` exists and every phase has a
   short description + current top-N / whitelist commentary sourced from
   a fresh profile. The "Reading the data" cross-cutting section covers
   cProfile semantics, whitelist vs top_n, comparing across commits /
   backends, resource caveats.
2. `startup_whitelist.yaml` has explicit patterns (or an explicit
   fall-through comment) for all five phases.
3. A fresh `benchmark_startup.py` run against Ant-Direct-v0 produces a
   `startup.json` where every `top_functions[*].calls` is `> 0` for at
   least the non-placeholder entries.
4. A fresh `benchmark_rsl_rl.py` or `benchmark_skrl.py` run produces a
   `training.json` where `Resources.gpu_mem_gb.peak > mean` (or
   `>=` with strict equality only if the sample variance is genuinely
   zero) — confirming `peak` is no longer a copy of `mean`.
5. Unit tests cover both fixes; existing benchmark tests still pass.

## Architecture — four narrow changes

Each change is scoped to a single file (or module), with no cross-coupling
beyond their data contracts.

### A. `scripts/benchmarks/utils.py` — `parse_cprofile_stats`

Return `list[tuple[str, float, float, int]]` adding `ncalls` from
`stats.stats[func_key][1]` (CPython's pstats internal dict already carries
this; the current implementation discards it). The inline comment in the
existing function already documents
``stats.stats[(filename, lineno, funcname)] -> (pcalls, ncalls, tottime, cumtime, callers)``
so no new pstats exploration is needed.

The whitelist-path placeholder rows (patterns that match no function)
stay at `(pattern, 0.0, 0.0, 0)` — zero calls on a zero-time row is
semantically correct.

### B. `scripts/benchmarks/benchmark_startup.py`

Both call sites of `parse_cprofile_stats` (top_n path and whitelist
path — lines ~254 and ~408 today) unpack the new 4-tuple and pass
`calls=ncalls` to `CProfileFunction(...)` instead of the hardcoded `0`.
No other logic changes. The inline comment that currently says

```python
# parse_cprofile_stats does not currently return call counts;
# pass 0 as a placeholder until the upstream fix lands.
```

is removed — the fix IS the upstream fix.

### C. Recorders — peak tracking

**`source/isaaclab/isaaclab/test/benchmark/recorders/record_memory_info.py`:**

Add instance attributes `_rss_peak`, `_vms_peak`, `_uss_peak` (all `float`,
initialized to `0.0`). In `_get_runtime_info()`, after each Welford update:

```python
self._rss_peak = max(self._rss_peak, mem_info.rss)
```

Emit new keys `rss_peak`, `vms_peak`, `uss_peak` in `_memory_runtime_info`.
Welford state is untouched.

**`source/isaaclab/isaaclab/test/benchmark/recorders/record_gpu_info.py`:**

Add per-device lists `_mem_peak` and `_util_peak` (populated alongside
`_mem_mean` and `_util_mean` during `_get_hardware_info()`). On each
sample, `self._mem_peak[i] = max(self._mem_peak[i], mem_bytes)` (and
similarly for util). Emit new keys `gpu_mem_peak[i]` / `gpu_util_peak[i]`
in `_gpu_runtime_info`.

**Peak-before-any-record behaviour.** If `.record()` is never called, all
`*_peak` stay at `0.0`. The recorder docstring documents this explicitly;
the peak test covers it.

### D. `scripts/benchmarks/benchmark_{rsl_rl,skrl}.py`

When building the `Resources.gpu_mem_gb` and `Resources.ram_gb` payload,
read the recorder's new `*_peak` keys instead of copying `mean`. Two
~5-line patches; no other logic changes.

## Data flow consequence

```
Before T2.2                                      After T2.2
──────────────────────────────────────────────   ──────────────────────────────────────────────
parse_cprofile_stats → (label, tot, cum)         parse_cprofile_stats → (label, tot, cum, ncalls)
CProfileFunction.calls = 0                       CProfileFunction.calls = ncalls (real)

Welford mean/std tracked                         Welford mean/std + running peak tracked
peak = mean (placeholder)                        peak = actual observed max
```

**Invariants preserved:**

- Schema v1.0 field names and types unchanged; the fix is populating
  correctly, not reshaping.
- Welford state untouched — peak is additive, never read by Welford.
- No cross-file dependencies change; each change is local to its file.
- `parse_cprofile_stats`'s existing filter logic (IsaacLab source +
  first-level external callers) is unchanged.

## Survey doc structure

`docs/odin/startup_profiling_survey.md`, hybrid layout:

```markdown
# Startup profiling survey

**Scope.** What benchmark_startup.py captures today, what the numbers
mean, and what Valhalla / comparison tooling should look at. Grounded
in a fresh local run on antoiner/feat/odin at <commit sha>.

## 1. Pipeline overview
~10 lines: cProfile per phase, whitelist vs top_n, schema v1.0
reference, where startup.json lives in the bundle.

## 2. Phase reference
Per-phase section, five phases. For each:
  - What it is (one sentence).
  - Typical wall-time range from the fresh run.
  - Current top-N / whitelisted functions, with one-line commentary
    on what each dominant function does.
  - Known caveats / noise sources.

### 2.1 app_launch
### 2.2 python_imports
### 2.3 task_config
### 2.4 env_creation
### 2.5 first_step

## 3. Reading the data (cross-cutting)
~200 words: cProfile semantics (own-time vs cumulative, filter scope,
ncalls interpretation), whitelist vs top_n guidance, stability
across commits / backends, resource-peak semantics post-T2.2.

## 4. Whitelist recommendations
~15 lines: describe the updated startup_whitelist.yaml — which
phases have explicit patterns, which deliberately fall through to
top_n, pointers for adding patterns later.

## 5. Open questions (seeds for T4)
~10 lines: things we noticed but didn't solve. GPU memory delta per
phase, warp kernel compile time, Kit-subsystem cost, USD asset
loading time. Not promises — pointers for future work.
```

**Grounding procedure.** Section 2 numbers come from:

```
PYTHONPATH=. ./isaaclab.sh -p scripts/benchmarks/benchmark_startup.py \
    --task Isaac-Ant-Direct-v0 --schema_v1_output /tmp/t2_2_startup.json
```

after Fixes 1 + 2 land. A one-shot Python snippet reads the JSON and
drops the top-5 functions per phase into the survey doc at author-time.
The doc is a snapshot at the T2.2 commit, not auto-regenerated.

## Whitelist re-tuning procedure

1. Run the fresh `benchmark_startup.py` (no whitelist) to get top-30
   per phase.
2. For each phase, pick ~5 stable, meaningful patterns that match the
   actual observed top functions.
3. If a phase's top functions are too variable or too low-level to
   whitelist usefully, leave its entry as a commented-out block
   explaining why it stays on `top_n`.
4. Re-run with the new whitelist; every explicit pattern must match
   ≥1 function (the existing placeholder-row warning helps catch
   stale patterns).
5. Commit `startup_whitelist.yaml`, the survey doc, and the
   benchmark-script changes together.

The survey doc's §4 describes the final state and rationale per phase.

## Testing approach

**Tier 1 — fast unit tests (CI, no GPU strictly required):**

- `source/isaaclab/test/benchmark/test_parse_cprofile_stats.py` (new).
  Two tests:
  - `test_top_n_returns_ncalls` — builds a small synthetic
    `cProfile.Profile` by actually calling a couple of functions a known
    number of times, then asserts the returned tuples carry correct
    `ncalls` values.
  - `test_whitelist_returns_ncalls` — same setup, calls with a whitelist
    pattern, asserts both matched rows and placeholder rows carry the
    right `ncalls` (placeholder = `0`).
- `source/isaaclab/test/benchmark/test_recorders.py` (modified).
  Extend `TestMemoryInfoRecorder` with:
  - `test_rss_peak_tracks_running_max` — monkey-patch
    `psutil.Process.memory_info` to return scripted RSS values
    `(100, 200, 150)`; call `_get_runtime_info()` three times; assert
    `_memory_runtime_info["rss_peak"] == 200`.
  - Parallel tests for `vms_peak` and `uss_peak`.
  - `test_peak_is_zero_before_any_record` — new recorder, never call
    `.record()`; assert all peaks are `0.0`.
  Extend `TestGPUInfoRecorder` with parallel peak tests for
  `gpu_mem_peak[i]` and `gpu_util_peak[i]`. Reuse whatever
  `torch.cuda.is_available()` mocking the existing tests already do.

**Tier 2 — smoke pass (manual, one-shot at T2.2 delivery):**

- Run `benchmark_startup.py` against `Isaac-Ant-Direct-v0` → eyeball
  the emitted `startup.json`; confirm `top_functions[*].calls > 0`
  for non-placeholder rows.
- Run a ~10-iteration `benchmark_rsl_rl.py` → eyeball `training.json`;
  confirm `Resources.gpu_mem_gb.peak != mean` on a non-trivially-loaded
  run.

Not in CI — the smoke pass is part of the T2.2 acceptance checklist.

**Verification gates**

- `./isaaclab.sh -f` clean before each commit.
- `./isaaclab.sh -p -m pytest source/isaaclab/test/benchmark/ -v` —
  all tests pass, including the new + extended ones.
- Existing benchmark regression tests stay green.

## File-by-file change list

### Lands in IsaacLab

| Path | Change |
|---|---|
| `scripts/benchmarks/utils.py` | `parse_cprofile_stats` returns 4-tuple with `ncalls`. |
| `scripts/benchmarks/benchmark_startup.py` | Unpack 4-tuple; pass `calls=ncalls`; drop placeholder comment. |
| `scripts/benchmarks/benchmark_rsl_rl.py` | `Resources.*.peak` sourced from recorder's `*_peak` keys. |
| `scripts/benchmarks/benchmark_skrl.py` | Same. |
| `scripts/benchmarks/startup_whitelist.yaml` | Extended to cover all five phases (with explicit fall-through comments where applicable). |
| `source/isaaclab/isaaclab/test/benchmark/recorders/record_memory_info.py` | Add `_rss_peak`, `_vms_peak`, `_uss_peak`; emit `*_peak` keys. |
| `source/isaaclab/isaaclab/test/benchmark/recorders/record_gpu_info.py` | Add per-device `_mem_peak[]`, `_util_peak[]`; emit `*_peak` keys. |
| `source/isaaclab/test/benchmark/test_recorders.py` | New peak-tracking tests added to existing `TestMemoryInfoRecorder` / `TestGPUInfoRecorder` classes. |
| `source/isaaclab/test/benchmark/test_parse_cprofile_stats.py` | **New** — `parse_cprofile_stats` has no existing test file. |
| `source/isaaclab/docs/CHANGELOG.rst` | New patch version entry under `Fixed`, citing both fixes. |
| `source/isaaclab/config/extension.toml` | Version bump to match the CHANGELOG entry. |

### Lands in Odin (`docs/odin/`)

| Path | Change |
|---|---|
| `docs/odin/startup_profiling_survey.md` | **New** — the survey doc. |
| `docs/odin/architecture.md` | §9 change-log entry for T2.2; no task-map change (T2.2 was already in the table from T2.1). |

## Open questions (resolved at implementation time)

- **Fall-through phases.** Whether `python_imports` and `task_config`
  genuinely benefit from a whitelist or stay on `top_n`. Decided during
  the fresh profile run: if no stable pattern emerges, the YAML keeps
  an explicit fall-through comment.
- **`Resources.*.peak` unit consistency.** The recorder emits peaks in
  bytes; the bundle's `gpu_mem_gb` is in GB. The benchmark scripts
  already divide bytes → GB for `mean` — the peak conversion uses the
  same divisor. One-line change, verified in the smoke pass.
- **Snapshot docstring tone.** Whether §5 of the survey ("Open
  questions") overlaps uncomfortably with the architecture doc's
  change-log. Resolved by keeping §5 tight (bullet list, no narrative)
  and letting the change-log carry the cross-task pointers.

These are execution-time questions; the architecture and deliverables
don't change based on how they resolve.
