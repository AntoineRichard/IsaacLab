# Odin — Architecture Reference

> **Living document.** Every task (T1 … T5) that refines or extends the system
> **must** update this file in the same commit that introduces the change.
> Treat it as the single source of truth for "how does Odin fit together."
> Per-task design details live in the per-task specs under
> `docs/superpowers/specs/`; this document gives the map.

**Last updated:** 2026-04-24 (T3.1 integration fixes landed)
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
├── training_data/      # TB events + params/ + checkpoints, written by the training framework
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
| T2.1 | Environment lists + Newton gap doc | `docs/superpowers/specs/2026-04-22-odin-t2-1-env-lists-design.md` | ✅ |
| T2.2 | Dense startup profiling survey | `docs/superpowers/specs/2026-04-22-odin-t2-2-startup-profiling-design.md` | ✅ |
| T3 | Distributed dispatcher (Layer 3) + Asgard | T3.1 `docs/superpowers/specs/2026-04-22-odin-t3-1-dispatch-design.md`; bootstrap `docs/superpowers/specs/2026-04-23-odin-bootstrap-design.md`; integration fixes `docs/superpowers/specs/2026-04-24-odin-t3-1-integration-fixes-design.md` | 🟡 |
| T4 | Reporting + Valhalla dashboard (Layer 4) | `docs/superpowers/specs/2026-04-23-odin-t4-1-valhalla-aggregator-design.md` | 🟡 |
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
| 2026-04-22 | T2.1 delivered. Three curated artifacts committed: `tools/odin/config/physx_envs.yaml` (188 envs → 54 keep=true), `tools/odin/config/newton_envs.yaml` (49 envs → 21 keep=true), `docs/odin/newton_api_gaps.md` (per-gap narrative + per-env appendix). Upstream IsaacLab change: `has_physics_preset` promoted from `test/env_test_utils._has_physics_preset` to public `isaaclab_tasks.utils.presets.has_physics_preset` (CHANGELOG 1.5.24). Odin side: `tools/odin/common/env_list.py` (EnvEntry dataclass, YAML IO + merge semantics, training-defaults loader, `build_entry_from_task_spec`, `classify_for_newton`, `GAP_VOCABULARY`) plus `tools/odin/scripts/enumerate_{physx,newton}_envs.py` with Play-v0 filtering. Findings surfaced: `has_rl_games` flag on EnvEntry documents the rl_games phase-out (55 rows), Deploy-family duplicates (Deploy-Reach-* mirrors reach/) noted for curation, and `preset_missing` (41 rows) identified as the largest bucket of Newton candidates — pure wiring work, not an API gap. Deferred polish items carried to T2.2: `raw_cfg: object` annotation on the promoted helper, skrl-dataclass test path for training-defaults extraction, module docstring drift in `env_list.py`. Known-broken T1 dry-run bundles under `odin_runs/` (stale TB copies; identical reward series across backends) still await a fix pass; T2.1 does not consume them. | Odin T2.1 |
| 2026-04-22 | T2.2 delivered. Closed both T1-carried reliability caveats: `parse_cprofile_stats` returns `ncalls` so `CProfileFunction.calls` is real (was always `0`); `MemoryInfoRecorder` and `GPUInfoRecorder` track peak so `Resources.*.peak` is the real running max (was copying `mean`). No schema bump — v1.0 fields are now populated correctly. `scripts/benchmarks/startup_whitelist.yaml` re-tuned to cover all five phases (or explicit `top_n` fallback comments). Survey published at `docs/odin/startup_profiling_survey.md` grounded in a fresh `Isaac-Ant-Direct-v0` profile. isaaclab version bumped 4.6.9 → 4.6.10. Open observations carried to T4 / future work: Kit-subsystem cost (not visible to cProfile), warp kernel compile time, per-asset USD load timing, per-phase GPU memory delta. | Odin T2.2 |
| 2026-04-23 | T3.1 delivered. `tools/odin/asgard/` library + thin CLI ingests `fleet.yaml` + T2.1 env YAMLs + CLI `--seeds`, provisions Valkyries via rsync + `./docker/container.py start` (smart-sync with `--fresh` override; rsync-the-working-tree transport avoids the no-push constraint), preflights each host (ssh + docker + container + isaaclab path), dispatches concurrently (one thread per Valkyrie) via `docker exec`, rsyncs bundles back, classifies failures (`infrastructure` retried up to 2×, `hugin_crash` / `hugin_malformed_bundle` / `timeout` never auto-retried). On-disk layout: `odin_runs/<dispatch_id>/<run_id>/` bundles + `dispatch.json` (atomic write) + `fleet.yaml.snapshot` + `preflight.json`. Resume via `--resume <dispatch_id|LATEST>` flips in-flight to `pending`, preserves completed/failed. One design deviation from the spec: infrastructure retry happens inside `_execute` (same-host loop) rather than via queue re-dispatch with `preferred_not` hint — simpler, but drops cross-host routing on infra failures. `preferred_not` fallback still exists for external callers. No upstream IsaacLab changes. Known pre-commit cleanup items (latent): C901 complexity on `run_dispatch`, license-header hook wants to touch YAML config files. T3.2 (local web UI on the T3.1 state) deferred — may fold into T4's Valhalla dashboard. | Odin T3.1 |
| 2026-04-23 | T1 bundle-fix landed. Fixed four bugs in the T1 dry-run bundles (spec: `docs/superpowers/specs/2026-04-23-odin-t1-bundle-fix-design.md`): (1) stale TB files copied via a racy `sorted(os.listdir(...), reverse=True)` glob — replaced by direct `--log_dir <bundle>/training_data` wiring from Hugin/Munin into `benchmark_{rsl_rl,skrl}.py`; `_copy_tb_events` + the glob block removed. (2) `--backend` was just a bundle tag — now injects `presets=<backend>` into `hydra_args` so physics really switches. (3) SKRL per-iter timing was `total / max_iters` flat — new `BenchmarkTrainer(SequentialTrainer)` subclass (`scripts/benchmarks/skrl_benchmark_trainer.py`) mirrors the parent loop and stamps per-iter wall time at every rollout boundary. Wired via a local `_BenchmarkRunner(Runner)` subclass that overrides `_generate_trainer`, keeping `agent.init()` single-firing. (4) SKRL reward series had length 97 (episode-boundary) — trainer now accumulates per-step `rewards.mean()` and emits one float per iter. Bundle layout: `tb/` → `training_data/` (also captures checkpoints + `params/`). SKRL override uses path decomposition (`directory=dirname(abspath(log_dir))`, `experiment_name=basename(abspath(log_dir))`) because SKRL's `BaseAgent` falls back to a synthesized timestamp subdir on empty `experiment_name`. All four bundles regenerated; reward SHAs now differ across backends, SKRL series length = 300, iter_time std > 0. 18 new tests, 139/140 pass. | Odin T1 bundle fix |
| 2026-04-23 | T4.1 delivered. New `tools/odin/valhalla/` package (spec: `docs/superpowers/specs/2026-04-23-odin-t4-1-valhalla-aggregator-design.md`): `stats` (Stats dataclass + `stats_over` + `is_divergent` z-score helper), `aggregator` (`aggregate_dispatch(dispatch_dir, options)` reads `dispatch.json` + per-bundle `manifest.json` / `training.json`, returns a nested `rows[]` × `seeds{}` dict with cross-seed `aggregate` blocks for 6 headline metrics — `reward_final_ema`, `ep_length_final_ema`, `iter_time_s_mean`, `env_steps_per_s_mean`, `ram_gb_peak`, `gpu_mem_gb_peak`), `writer` (atomic `aggregate.json` write via `tempfile.mkstemp`+`os.replace`), `cli` (`odin-aggregate <dispatch_id\|LATEST>` with `--divergence-z`, `--no-overwrite`, `--quiet`). Auto-wired into `run_dispatch` as the post-`write_dispatch_state` tail step (best-effort try/except that logs `[WARNING]` without masking dispatch return), opt-out via `--skip-aggregate` / `DispatchOptions.skip_aggregate`. Strict-whitelist failure policy (§8): a seed is "completed" only if manifest `phases.training.status == "completed"` AND `exit_code == 0` AND training.json `schema_version` matches `1.*`; everything else lands in top-level `failures[]` with `failure_kind` from T3.1's classification plus synthesised `missing_bundle` / `malformed_bundle`. Commit-SHA majority-wins with stdout WARNING on mixed SHAs. Divergent-seed z-score defaults to 2.0 (note: `(n-1)/sqrt(n-1)` bound means single-outlier flagging needs n ≥ 6 at z=2.0 under strict `>`). 31 unit tests + 1 integration test exercising full T3.1 → T4.1 pipeline with fake SSH/Rsync runners. Real-fleet validation pass still pending per spec §10 — T4.1 is not "done" until one real dispatch on the runner machines produces an `aggregate.json` the operator inspects. | Odin T4.1 |
| 2026-04-24 | Odin bootstrap delivered. Closes a T3.1 gap surfaced on the first real-fleet wiring attempt: fresh Valkyries (just SSH + Docker + GPU, no IsaacLab clone, no container) cannot pass T3.1 preflight, and the provisioner's 300-s `_container_start` timeout is too short for a first-time docker image build. New `tools/odin/asgard/bootstrap.py` + `bootstrap_cli.py` (spec: `docs/superpowers/specs/2026-04-23-odin-bootstrap-design.md`) provide a single `odin-bootstrap --fleet fleet.yaml` entry that per-host does: ssh reach → docker daemon probe → `rm -rf {isaaclab_path}` → rsync working tree → `./docker/container.py start` (30-min default timeout, configurable via `--build-timeout`) → `docker inspect` verify. Parallel per-host by default (ThreadPoolExecutor); `--sequential` opt-out. `BootstrapResult` records per-step wall times for the four work steps. Provisioner tweak: `_container_start` gains a keyword-only `timeout_s: int = 300` so the warm path stays unchanged while bootstrap passes 1800. Zero changes to T3.1's `run_dispatch` / preflight / worker / state machine. Tests: 15 for `bootstrap_valkyrie` + `bootstrap_fleet` (incl. 6 failure-path + 2 timing-variance tests), 6 for the CLI, 2 for the provisioner timeout. | Odin T3 bootstrap |
| 2026-04-24 | T3.1 integration fixes landed (spec: `docs/superpowers/specs/2026-04-24-odin-t3-1-integration-fixes-design.md`). Closed three gaps that only surfaced on the first real-fleet dispatch attempt: (1) Hugin/Munin wrappers now accept a `--run_id` CLI override that bypasses `compute_run_id(...)`, so the dispatcher-computed id is used end-to-end. (2) `docker/docker-compose.yaml` bind-mounts `~/IsaacLab/odin_runs` into the isaac-lab-base container at `/workspace/isaaclab/odin_runs`, so bundles written inside the container land on the host where T3.1's rsync-pull expects them; bootstrap gained a `create_odin_runs` step to pre-create the bind-mount source (owned by the SSH user, not root-via-docker). (3) `tools/odin/asgard/worker.py:_build_docker_exec_cmd` appends `--run_id <job.run_id>` to every Hugin/Munin invocation. Three further bugs surfaced and were fixed on top of the plan: (a) bootstrap `rm -rf` is now best-effort (`2>/dev/null; true`) so root-owned `__pycache__/` artifacts left by prior container runs don't block the wipe. (b) `ShellSSHRunner.run()` drains stdout/stderr on reader threads so `timeout_s` actually fires when the remote command produces no output — previously the drain loop could block in `readline()` indefinitely. (c) `ShellRsyncRunner.pull()` now forces a trailing slash on the remote source path, so the bundle contents land in `local_path/` rather than a doubly-nested `local_path/<run_id>/`. Real-fleet smoke on 2 hosts × 2 seeds × rsl_rl + Ant: both `physx` jobs complete end-to-end with fully-populated bundles + `aggregate.json`; both `newton` jobs fail with `Warp CUDA error 900: operation not permitted when stream is capturing` — a mjwarp/newton stack issue on hosts with CUDA driver <12.4 (`Model.opt.graph_conditional should be set to False` per the upstream warning), unrelated to Odin infrastructure. Follow-ups: docker-exec zombies on SSH timeout (remote training continues after local SSH terminates, hogging the GPU for the next job on the same host), newton CUDA-<12.4 workaround or host driver upgrade. | Odin T3.1 integration fixes |
