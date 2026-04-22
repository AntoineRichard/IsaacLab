# Odin — Architecture Reference

> **Living document.** Every task (T1 … T5) that refines or extends the system
> **must** update this file in the same commit that introduces the change.
> Treat it as the single source of truth for "how does Odin fit together."
> Per-task design details live in the per-task specs under
> `docs/superpowers/specs/`; this document gives the map.

**Last updated:** 2026-04-22 (end of T1 design)
**Plan:** `eval_plan.md` (repo root)
**Branch during in-tree development:** `antoiner/feat/odin`

## 1. What Odin is

Odin is a multi-backend, large-scale training evaluation harness for IsaacLab.
It runs benchmarked training jobs across physics backends (PhysX, Newton; ovphysx
deferred) and learning frameworks (RSL-RL, SKRL), collects runtime perf,
startup perf, system resource consumption, and reward-reached metrics, and
serves them through a dashboard for comparison across commits, backends, and
machines.

Odin is **not** part of IsaacLab itself. During development it lives at
`tools/odin/` in-tree for convenience; when it graduates it will move to its
own repo. Real improvements to the benchmark toolchain (e.g. EMA smoothing,
standard result schema) land **in IsaacLab** — Odin-specific concerns (run
orchestration, naming, dispatch) live in Odin.

## 2. Naming glossary — Norse mythology

Odin receives information from his two ravens, Hugin and Munin, who fly out
over the world and report back. That metaphor is the architecture.

| Name | Role | Introduced in |
|---|---|---|
| **Odin** | The project itself; the controller that dispatches jobs and aggregates results | T0 |
| **Hugin** | Benchmark runner wrapper for **RSL-RL** | T1 |
| **Munin** | Benchmark runner wrapper for **SKRL** | T1 |
| **Valhalla** | Results archive and dashboard | T4 |
| **Asgard** | The compute cluster — the pool of worker machines | T3 |
| **Valkyries** | Individual worker nodes that run jobs and return results | T3 |
| **Bifrost** | Inter-node communication / SSH transport | T3 |
| **Ratatoskr** | Status / notification bus (optional) | T3 |
| **Yggdrasil** | The IL 2.3.x ↔ IL 3.x bridge for apples-to-apples version comparison | T5 |

When adding a new subsystem, prefer extending this Norse vocabulary. Don't
invent parallel name schemes. Update this table when a new name is committed.

## 3. Layered architecture

```
┌───────────────────────────────────────────────────────────────────┐
│ Layer 4 — Valhalla (T4)                                           │
│   Aggregation + Dash/Plotly dashboard.                            │
│   Reads bundles, compares across commits / backends / machines.   │
└────────────────────────────▲──────────────────────────────────────┘
                             │ reads <run_id>/{manifest,training,startup}.json
┌────────────────────────────┴──────────────────────────────────────┐
│ Layer 3 — Odin controller + Asgard (T3)                           │
│   Dispatches jobs over Bifrost (SSH) to Valkyrie nodes;           │
│   monitors progress; collects bundles back from workers.          │
│   Runs docker setup on each Valkyrie on first contact.            │
└────────────────────────────▲──────────────────────────────────────┘
                             │ launches one Hugin/Munin per (framework, backend, task, seed)
┌────────────────────────────┴──────────────────────────────────────┐
│ Layer 2 — Hugin / Munin runner wrappers (T1)                      │
│   Owns run identity, bundle directory, manifest, log tails.       │
│   Two subprocesses per run: startup profile + training.           │
│   No retry logic — one honest attempt; retries are Layer 3.       │
└────────────────────────────▲──────────────────────────────────────┘
                             │ subprocess: --output-path <bundle>/{startup,training}.json
┌────────────────────────────┴──────────────────────────────────────┐
│ Layer 1 — IsaacLab benchmark scripts (T1)                         │
│   scripts/benchmarks/benchmark_{rsl_rl,skrl,startup}.py           │
│   Independently invocable; each emits one schema-v1 JSON.         │
│   Reuses VersionInfoRecorder / GPUInfoRecorder / CPUInfoRecorder. │
└───────────────────────────────────────────────────────────────────┘
```

**Invariants between layers:**

- Layer 1 scripts write *exactly one* file to the `--output-path` they're given.
  They know nothing about run IDs, bundles, or Odin. They are usable by anyone
  benchmarking IsaacLab.
- Layer 2 owns the `<run_id>/` bundle directory and the `manifest.json` that
  stitches the layer-1 outputs together.
- Layer 3 treats a bundle as an opaque deliverable: any (framework, backend,
  task, seed) is run by launching a Layer-2 wrapper with the right args and
  waiting for the bundle.
- Layer 4 treats the bundle as read-only input. No metric is computed for the
  first time in the dashboard — everything is in the JSON.

If any of these invariants changes, **update this doc in the same commit.**

## 4. Run identity & bundle layout

Every Odin run is uniquely identified by `(framework, backend, task, seed, date)`.

**Run ID format:**

```
<framework>_<backend>_<task>_<date>_seed<seed>
```

- `framework` ∈ `{rsl-rl, skrl}` (hyphen variant in the path; underscored in JSON).
- `backend` ∈ `{physx, newton}`.
- `task` = gym ID verbatim (e.g. `Isaac-Ant-Direct-v0`).
- `date` = `YYYYMMDD-HHMMSS` in UTC at run start.
- `seed` = `seed<integer>`.

**Bundle layout:**

```
<runs_root>/<run_id>/
├── manifest.json       # Odin-side index (Layer 2)
├── startup.json        # Dense cProfile, v1.0 schema (Layer 1)
├── training.json       # Full training metrics, v1.0 schema (Layer 1)
├── tb/                 # Raw TB event files, best-effort copy
└── logs/               # stderr/stdout tails (last 16 KB) on failure only
```

## 5. Schema versioning

- Every JSON file carries `schema_version: "x.y"`.
- **Major bump** (`1.0` → `2.0`): breaking changes (field removed or renamed;
  semantics change). Requires backfilling or explicit deprecation.
- **Minor bump** (`1.0` → `1.1`): additive changes (new optional field).
  Layer-4 consumers must treat unknown fields as optional.
- Layer-1 writers and Layer-2/4 readers advance in lockstep for major bumps.
  Layer-4 should tolerate older minor versions indefinitely.

The canonical schema lives in
`source/isaaclab/isaaclab/test/benchmark/standard_schema.py` (Layer 1) and
`tools/odin/common/manifest.py` (Layer 2). See the T1 spec for field-by-field
details.

## 6. Task map

Status legend: ✅ complete · 🟡 in progress · ⚪ pending

| Task | Title | Spec | Status |
|---|---|---|---|
| T0 | Naming | — (recorded in `eval_plan.md`) | ✅ |
| T1 | Evaluation runner (Layer 1 + 2) | `docs/superpowers/specs/2026-04-22-odin-t1-evaluation-runner-design.md` | ✅ |
| T2.1 | Environment lists + Newton gap doc | — | ⚪ |
| T2.2 | Dense startup profiling survey | — | ⚪ |
| T3 | Distributed dispatcher (Layer 3) + Asgard | — | ⚪ |
| T4 | Reporting + Valhalla dashboard (Layer 4) | — | ⚪ |
| T5 | IL 2.3.x backport (Yggdrasil) | — | ⚪ |

When a spec is written for T2.1+, add its path to the table.

## 7. Scope boundaries between tasks

Tasks should keep to their layer and concerns; this table is the reference for
"does this belong here or there?"

| Concern | Owning task |
|---|---|
| Per-run JSON schema, benchmark scripts, EMA | T1 |
| Reference bundle for Ant Direct | T1 |
| Which environments should run, Newton API gaps | T2.1 |
| What to capture in startup profiles | T2.2 |
| SSH dispatch, Docker bring-up on Valkyries | T3 |
| Live progress monitoring UI for a dispatch run | T3 |
| Cross-run aggregation, failure reporting | T4 |
| Dash/Plotly dashboard, per-machine baselines | T4 |
| Making IL 2.3.x emit v1 schema bundles | T5 |

If you find yourself adding a feature in one task's area while nominally
working on another, either scope-creep check: move it to the right task, or
explicitly expand scope and note it here.

## 8. How to update this document

- Every task that introduces a new subsystem must add it to §2 (glossary) and
  §3 (layer diagram).
- When a spec is written, add its path to §6.
- When a layer's invariants change (§3), update that section and note the
  change in §9.
- When scope boundaries shift between tasks, update §7.
- Commit this update in the same commit as the underlying change — never in
  a separate "docs" commit.

## 9. Change log for this document

| Date | Change | By |
|---|---|---|
| 2026-04-22 | Initial version — created at end of T1 design. | Odin T1 |
| 2026-04-22 | T1 implementation complete: v1.0 schema, three benchmark scripts upgraded/added, Hugin + Munin runners. `startup.json` captures five phases (`app_launch`, `python_imports`, `task_config`, `env_creation`, `first_step`) — the T1 spec originally listed three; the implementation reused `benchmark_startup.py`'s richer existing split. Known v1 limitations: `CProfileFunction.calls` is always `0` (upstream `parse_cprofile_stats` does not return call counts) and `Resources.*.peak` falls back to `mean` because the underlying recorders track Welford online stats but not peak. | Odin T1 |
