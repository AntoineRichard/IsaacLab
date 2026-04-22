# Odin T1 — Evaluation Runner Design

**Project:** Odin (multi-backend IsaacLab evaluation harness)
**Task:** T1 — the per-run benchmark pipeline
**Date:** 2026-04-22
**Branch:** `antoiner/feat/odin`
**Status:** Draft — pending user review

## Context

Odin is a multi-backend, large-scale training evaluation harness for IsaacLab (plan:
`eval_plan.md`). T0 picked the Norse-mythology naming: Odin is the controller,
Hugin and Munin are the two benchmark runners (ravens that fly out and report back),
Valhalla is the eventual dashboard (T4), and Asgard / Valkyries / Bifrost cover
distributed execution (T3).

T1's job is to design **one canonical unit of work**: given one
`(framework, backend, task, seed)` tuple on a fixed machine, produce a
self-contained bundle of artifacts — startup profile, training metrics, manifest
— that downstream layers (T3 dispatcher, T4 dashboard) can rely on.

T1 is *not* responsible for orchestration across seeds, tasks, or machines —
that is explicitly deferred to T3.

## Goals & non-goals

**In scope:**

- A single-run benchmark pipeline that captures runtime perf, startup perf,
  system resource consumption, reward reached, and full version/hardware context.
- Upstream improvements to IsaacLab's existing `scripts/benchmarks/` that benefit
  standalone IsaacLab users, not just Odin.
- A versioned JSON schema for run artifacts that downstream Odin layers can
  depend on.
- A concrete dry-run deliverable on `Isaac-Ant-Direct-v0` that exercises both
  backends (PhysX, Newton) and both frameworks (RSL-RL, SKRL).

**Out of scope (deferred to later tasks):**

- Multi-seed / multi-task orchestration (T3).
- SSH dispatch, Docker setup on Valkyrie nodes, the Asgard cluster (T3).
- Aggregation, cross-run comparison, Valhalla dashboard (T4).
- IL 2.3.x backport concerns / Yggdrasil (T5).
- Retry logic at the individual-run level. Hugin/Munin make one honest attempt;
  retries are T3's concern.

## Architecture — three layers

### 1. IsaacLab — single-purpose benchmark scripts (upstream)

Three scripts in `scripts/benchmarks/`, each independently invocable and each
emitting one file of a known schema:

| Script | Role | Output |
|---|---|---|
| `benchmark_rsl_rl.py` (upgraded) | Full RSL-RL training with recorders + inline EMA | `training.json` |
| `benchmark_skrl.py` (new, parallel) | Same, for SKRL | `training.json` |
| `benchmark_startup.py` (lightly extended) | Dense cProfile on cold-start phases | `startup.json` |

Each script accepts a `--output-path` CLI argument and writes exactly one file.
The shared schema envelope is defined in a new module
`source/isaaclab/isaaclab/test/benchmark/standard_schema.py` — neutral name, no
Odin branding upstream — with a small `write_bundle_file()` helper that both
training scripts and the startup script call.

The existing `VersionInfoRecorder`, `GPUInfoRecorder`, and `CPUInfoRecorder` in
`source/isaaclab/isaaclab/test/benchmark/recorders/` already capture nearly all
the `versions` / `hardware` / `resources` content the schema needs. T1 wires them
in, does not re-invent them.

### 2. Odin — runners (`tools/odin/`)

Lives at `tools/odin/` inside IsaacLab today. When Odin graduates to its own
repo, this directory moves out wholesale; the IsaacLab-side scripts stay
independently usable.

- `tools/odin/hugin/run.py` — RSL-RL runner wrapper.
- `tools/odin/munin/run.py` — SKRL runner wrapper.
- `tools/odin/common/manifest.py` — shared helpers (run_id format, manifest
  writer, log-tail utility).
- `tools/odin/README.md` — local invocation docs, bundle structure, graduation plan.
- `tools/odin/.gitignore` — excludes `odin_runs/`.
- `tools/odin/tests/` — unit tests for run_id format, manifest schema, log tail
  tail; one integration test with faked subprocesses.

One Hugin or Munin invocation corresponds to exactly one Odin run. Each wrapper:

1. Computes `run_id` and creates `<runs_root>/<run_id>/`.
2. Subprocess-launches `benchmark_startup.py` → writes `startup.json` into the
   bundle.
3. Subprocess-launches `benchmark_<framework>.py` → writes `training.json` into
   the bundle. Copies the generated TB event file(s) into `<bundle>/tb/`.
4. Writes `<bundle>/manifest.json` (see schema below).
5. If either phase crashes, the manifest is still written with `status:
   "failed"`, partial artifacts remain, and the last 16 KB of each stream's
   stderr / stdout is saved to `<bundle>/logs/`.

### 3. Orchestrator (T3) & Dashboard (T4)

Out of T1 scope. T3 will dispatch many Hugin/Munin invocations across Valkyrie
nodes. T4 (Valhalla) will consume bundles uniformly by reading
`<run_id>/{manifest, training, startup}.json`.

### Data flow for one run

```
Hugin or Munin (run_id = rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-131500_seed42)
  │
  ├─▶ subprocess: benchmark_startup.py  \
  │       --task ... --seed ...          \
  │       --output-path <bundle>/startup.json
  │     (dense cProfile on app_launch / env_creation / first_step)
  │
  ├─▶ subprocess: benchmark_<framework>.py \
  │       --task ... --seed ... --max-iterations ... --num-envs ... \
  │       --output-path <bundle>/training.json
  │     (full training; inline per-iter capture; recorders attached)
  │
  └─▶ writes <bundle>/manifest.json
      copies TB events into <bundle>/tb/
      tails stderr/stdout on failure into <bundle>/logs/
```

## Schema v1.0

Three files per bundle, each self-contained. Every file declares
`schema_version: "1.0"`.

### `training.json` — the main artifact

```json
{
  "schema_version": "1.0",
  "run": {
    "run_id": "rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-131500_seed42",
    "framework": "rsl_rl",
    "backend": "physx",
    "task": "Isaac-Ant-Direct-v0",
    "seed": 42,
    "num_envs": 4096,
    "max_iterations": 500,
    "start_time_utc": "2026-04-22T13:15:00Z",
    "end_time_utc": "2026-04-22T13:47:22Z",
    "duration_s": 1942.1,
    "status": "completed"
  },
  "versions": {
    "isaaclab": "4.6.8", "isaacsim": "5.0.0", "kit": "107.1.0",
    "newton": "0.1.2", "warp": "1.7.3", "mjwarp": "0.0.4",
    "torch": "2.5.1", "rsl_rl": "2.3.0", "skrl": null,
    "git_commit": "3d42b11d513", "git_branch": "antoiner/feat/odin",
    "git_dirty": false
  },
  "hardware": {
    "hostname": "valkyrie-03",
    "gpu_devices": [
      {"name": "NVIDIA H100 80GB", "mem_gb": 80, "compute_cap": "9.0"}
    ],
    "cpu_name": "AMD EPYC 7763",
    "cpu_count": 64,
    "ram_gb": 512
  },
  "runtime": {
    "startup_phase_times_s": {
      "app_launch": 18.4, "env_creation": 22.9, "first_step": 4.1
    },
    "iterations_completed": 500,
    "total_wall_time_s": 1946.0,
    "steps_per_iteration": 24,
    "iteration_time_s":  {"mean": 3.82,    "std": 0.04},
    "env_steps_per_s":   {"mean": 1071780, "std": 11200},
    "iterations_per_s":  {"mean": 0.2618,  "std": 0.0028}
  },
  "resources": {
    "gpu_util_pct":  {"mean": 87.2, "std": 6.1},
    "gpu_mem_gb":    {"mean": 18.4, "std": 0.3, "peak": 19.2},
    "cpu_util_pct":  {"mean": 31.5, "std": 4.8},
    "ram_gb":        {"mean": 22.1, "std": 0.4, "peak": 24.8}
  },
  "learning": {
    "ema_alpha": 0.05,
    "reward": {
      "final_raw": 1823.4,
      "final_ema": 1796.1,
      "series_per_iter": [12.3, 34.5, 58.1, "..."]
    },
    "ep_length": {
      "final_raw": 987.0,
      "final_ema": 962.3,
      "series_per_iter": [4.1, 5.0, 7.2, "..."]
    }
  }
}
```

Field semantics:

- `run.status`: `"completed" | "interrupted" | "crashed"`. `"interrupted"` means
  the process exited cleanly before `max_iterations` (e.g. SIGTERM); `"crashed"`
  means non-zero exit with an error.
- `runtime.env_steps_per_s`: the canonical throughput metric. Defined as
  `num_envs × steps_per_iteration / iteration_time_s`. Aggregated across all
  parallel envs — this is what you compare between frameworks and backends.
- `runtime.iteration_time_s`, `env_steps_per_s`, `iterations_per_s`: all derived
  from the same per-iteration wall-time series. Mean/std computed via Welford's
  algorithm (already provided by `StatisticalMeasurement`).
- `resources.*`: sampled at a fixed interval (default 1 Hz) during training and
  aggregated to `{mean, std, peak}`. Startup period excluded from the sample
  window.
- `learning.ema_alpha`: 0.05 by default (≈ 20-sample effective window). CLI
  override via `--ema-alpha`.
- `learning.reward.series_per_iter` and `ep_length.series_per_iter`: full
  per-iteration mean values (one float per iteration). Enabled by default;
  opt-out via `--no-series`. Payload for a 500-iter run is ≈ 8 KB.
- `versions.skrl: null` when the run used RSL-RL, and vice versa. Keep both keys
  for schema uniformity.

### `startup.json`

Same envelope (`schema_version`, `run`, `versions`, `hardware` blocks as above;
`run.seed` may be absent since startup is seed-independent in practice, but we
include it for bundle consistency). Main body:

```json
{
  "schema_version": "1.0",
  "run": { "run_id": "...", "framework": "rsl_rl", "backend": "physx",
           "task": "Isaac-Ant-Direct-v0", "seed": 42, "status": "completed",
           "start_time_utc": "...", "end_time_utc": "...", "duration_s": 48.7 },
  "versions": { "..." : "..." },
  "hardware": { "..." : "..." },
  "phases": {
    "app_launch": {
      "total_time_s": 18.4,
      "top_functions": [
        {"name": "isaaclab.utils.configclass:_custom_post_init",
         "own_time_s": 1.82, "cum_time_s": 2.41, "calls": 4312},
        "..."
      ]
    },
    "env_creation":  {"total_time_s": 22.9, "top_functions": ["..."]},
    "first_step":    {"total_time_s":  4.1, "top_functions": ["..."]}
  },
  "config": {
    "top_n": 30,
    "whitelist": "startup_whitelist.yaml"
  }
}
```

The `phases` shape mirrors what `benchmark_startup.py` already produces; the
schema envelope is the only new surface.

### `manifest.json` — Odin's bundle index

Thin, navigational. No metric data — that lives in `training.json` and
`startup.json`.

```json
{
  "schema_version": "1.0",
  "run_id": "rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-131500_seed42",
  "run_start_time_utc": "2026-04-22T13:15:00Z",
  "run_end_time_utc":   "2026-04-22T13:47:48Z",
  "run_duration_s": 1968.3,
  "config": {
    "framework": "rsl_rl", "backend": "physx",
    "task": "Isaac-Ant-Direct-v0", "seed": 42,
    "num_envs": 4096, "max_iterations": 500
  },
  "machine": {
    "hostname": "valkyrie-03",
    "git_commit": "3d42b11d513", "git_branch": "antoiner/feat/odin"
  },
  "phases": {
    "startup":  {"file": "startup.json",  "status": "completed", "duration_s": 48.7,
                 "exit_code": 0},
    "training": {"file": "training.json", "status": "completed", "duration_s": 1942.1,
                 "exit_code": 0}
  },
  "artifacts": ["manifest.json", "startup.json", "training.json",
                "tb/events.out.tfevents.1745329500.valkyrie-03"]
}
```

On failure, `phases.<phase>.status` is `"failed"`, `exit_code` is the non-zero
code, and `logs/<phase>.stderr.log` / `logs/<phase>.stdout.log` hold the last
16 KB of each stream.

## Naming & bundle layout

### Run ID format

```
<framework>_<backend>_<task>_<date>_seed<seed>
```

- `framework` ∈ `{rsl-rl, skrl}` — hyphen variant to avoid clashing with the
  `_` separator. Mapped to the underscore form in JSON.
- `backend` ∈ `{physx, newton}`.
- `task` = the gym ID verbatim (e.g. `Isaac-Ant-Direct-v0`). Hyphens inside
  the task are fine — they cannot collide with the `_` separator.
- `date` = `YYYYMMDD-HHMMSS` in UTC, captured at run start.
- `seed` = `seed<integer>` (e.g. `seed42`).

Example: `rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-131500_seed42`

Sorted listings group naturally by (framework, backend, task, time). Works for
`ls`, S3 listings, and Valhalla's filtering.

### Bundle directory layout

```
<runs_root>/
└── rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-131500_seed42/
    ├── manifest.json
    ├── startup.json
    ├── training.json
    ├── tb/                      # raw TB event files, copied verbatim
    │   └── events.out.tfevents.<...>
    └── logs/                    # only written when a phase fails
        ├── startup.stderr.log   # last 16 KB
        ├── startup.stdout.log   # last 16 KB
        ├── training.stderr.log
        └── training.stdout.log
```

### `<runs_root>` default

- Local dev: `./odin_runs/` under the cwd. Excluded from git via
  `tools/odin/.gitignore`.
- Worker machines (T3): `--runs-root` flag passed by the dispatcher.

## Dry-run deliverable

A four-bundle reference set for `Isaac-Ant-Direct-v0`, committed under
`docs/odin/reference_runs/`:

- `rsl-rl_physx_Isaac-Ant-Direct-v0_<date>_seed42/`
- `rsl-rl_newton_Isaac-Ant-Direct-v0_<date>_seed42/`
- `skrl_physx_Isaac-Ant-Direct-v0_<date>_seed42/`
- `skrl_newton_Isaac-Ant-Direct-v0_<date>_seed42/`

Each committed bundle contains `manifest.json`, `training.json`,
`startup.json`, and a `README.md` noting that `tb/` and `logs/` are omitted
for size reasons. Run configuration: `num_envs=4096`, `max_iterations=500`,
`headless=true`, `seed=42`. Same settings as the existing
`run_training_benchmarks.sh`.

Running the four exercises: both Hugin and Munin wrappers, both backends, both
recorder paths (`rsl_rl` vs `skrl` version fields), and the full schema across
four independent invocations.

**Gating.** Each of the four runs is contingent on the corresponding config
existing for Ant Direct:

- `rsl_rl_cfg_entry_point` must be registered (today: yes).
- `skrl_cfg_entry_point` must be registered (verify during T1 execution).
- Newton must support Ant (verify during T1 execution).

Any missing cell is noted in the T1 completion report as a concrete input to
T2.1 (newton gaps doc); T1 does not attempt to fill gaps.

**Machine caveat.** The committed reference bundles capture *one* machine's
numbers. They are the canonical schema example, not golden thresholds for CI.
Per-machine baselines are T4's concern.

## Upstream vs Odin split

### Lands in IsaacLab

| Path | Purpose |
|---|---|
| `scripts/benchmarks/benchmark_rsl_rl.py` | Extended: inline reward/ep_length capture, EMA, `--output-path` / `--no-series` / `--ema-alpha` flags, v1.0 schema output |
| `scripts/benchmarks/benchmark_skrl.py` | New, symmetric to RSL-RL script |
| `scripts/benchmarks/benchmark_startup.py` | Minor: `--output-path`, v1.0 schema envelope |
| `source/isaaclab/isaaclab/test/benchmark/standard_schema.py` | Dataclasses + `write_bundle_file()` helper |
| `source/isaaclab/test/benchmark/test_standard_schema.py` | Schema validation tests against the committed reference bundles |
| `docs/source/features/benchmarking.md` | User-facing docs for the schema and how to invoke the scripts |
| `source/isaaclab/docs/CHANGELOG.rst` | New version entry under "Added" / "Changed" |
| `source/isaaclab/config/extension.toml` | Version bump to match changelog |

Plus CHANGELOG entries for `isaaclab_physx` / `isaaclab_newton` if the recorders
touch those packages' version fields.

### Lands in Odin (`tools/odin/`, moves out when Odin graduates)

| Path | Purpose |
|---|---|
| `tools/odin/hugin/run.py` | RSL-RL runner wrapper |
| `tools/odin/munin/run.py` | SKRL runner wrapper |
| `tools/odin/common/manifest.py` | run_id format, manifest writer, log-tail utility |
| `tools/odin/tests/test_run_id.py` | run_id format tests |
| `tools/odin/tests/test_manifest.py` | manifest schema tests |
| `tools/odin/tests/test_integration.py` | integration test with faked subprocesses |
| `tools/odin/README.md` | local invocation docs, bundle structure, graduation plan |
| `tools/odin/.gitignore` | excludes `odin_runs/` |

### Shared artifact

`docs/odin/reference_runs/` — the four committed Ant Direct reference bundles.
Lives with Odin (moves when it graduates), but committed to IsaacLab today.

## Testing approach

- **IsaacLab side (upstream):**
  - `test_standard_schema.py` loads each committed reference bundle and asserts
    schema validity (required fields, types, `schema_version == "1.0"`).
  - Existing `test_benchmark_*` tests continue to pass.
  - A new small unit test for the inline EMA computation with known inputs.

- **Odin side:**
  - `test_run_id.py` — round-trip and collision properties of the run_id format,
    including edge cases (tasks with multiple hyphens, long seeds, UTC handling).
  - `test_manifest.py` — manifest-writer output matches schema v1.0; handles
    partial phase completion (startup done, training crashed).
  - `test_integration.py` — full Hugin/Munin loop with the two benchmark scripts
    mocked out to fast fakes that emit valid schema-compliant output.

- **Verification gates:**
  - `./isaaclab.sh -f` clean before commit.
  - `./isaaclab.sh -p -m pytest` on both new Odin and IsaacLab test files
    (sequentially — GPU tests not parallel).
  - The dry-run itself verifies the real pipeline end-to-end, with the output
    bundles committed.

## Open questions (to be resolved during implementation planning)

- Exact logger-hook surface in RSL-RL's `OnPolicyRunner` for per-iter reward /
  ep_length capture. Likely a callback, but need to confirm the hook point exists
  and survives across rsl_rl versions.
- SKRL's equivalent hook surface — parallel question for Munin's side.
- Whether `Isaac-Ant-Direct-v0` currently has a registered `skrl_cfg_entry_point`
  and whether Newton supports it. Both are verifiable during implementation.
- Exact resource-sampling mechanism during training (`GPUInfoRecorder` already
  supports periodic sampling; confirm it works alongside the training loop
  without interfering).

These are execution-time questions, not design-time questions — the schema and
architecture don't change based on how they resolve.
