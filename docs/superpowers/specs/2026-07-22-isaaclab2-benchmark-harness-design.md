<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# IsaacLab 2.x Benchmark Harness Backport Design

**Status:** Approved

## Context

IsaacLab 3.0 has a unified benchmark framework with typed schema-v1
artifacts, system recorders, multiple output formatters, and common runtime,
startup, training, and play entrypoints. The maintained IsaacLab 2.x line on
`origin/main` still has independent legacy scripts whose measurements and
outputs are not directly comparable with the 3.0 harness.

The work will port the current 3.0 harness to the 2.x main line, adapting its
integration points to the older launcher, task configuration, and RL APIs.
A shared controller will then run an identical PhysX benchmark matrix against
both isolated revisions and generate reproducible tables and plots from
preserved raw artifacts.

## Pinned Revisions

- IsaacLab 2.x target: `origin/main@858234d06e`.
- IsaacLab 3.0 harness source and comparison reference:
  `origin/develop@cb508381fb`.

The controller records the full commit identifiers used for every run. If
either revision changes, it constitutes a new run set rather than silently
changing an existing comparison.

## Goals

- Make the complete current 3.0 benchmark harness available on IsaacLab 2.x.
- Preserve the 3.0 public CLI and schema-v1 output contract wherever the 2.x
  runtime can support it.
- Adapt implementation details to 2.x instead of importing unrelated 3.0
  architecture changes.
- Run a shared six-workload PhysX matrix on both revisions.
- Measure runtime collection throughput, training collection throughput, GPU
  memory, and GPU utilization.
- Preserve all raw artifacts and make report generation independently
  rerunnable.
- Keep all performance conclusions informational; do not introduce regression
  thresholds.

## Non-goals

- Backport Newton, OVPhysX, OVRTX, typed physics presets, or other 3.0 physics
  backends to IsaacLab 2.x.
- Force both product lines onto an unsupported common Isaac Sim or dependency
  stack.
- Change task implementations to make their performance identical.
- Add benchmark warm-up semantics that are not merged into the pinned 3.0
  harness.
- Remove legacy 2.x benchmark entrypoints before a deprecation cycle.
- Treat GPU utilization percentage as a quality score. It is reported as an
  observed resource metric.

## Architecture

Three isolated areas are used:

1. A feature worktree based on the pinned 2.x main revision. It contains the
   adapted harness and is built into a local GPU-enabled Docker image.
2. A clean detached worktree at the pinned 3.0 revision. It uses its own uv
   environment and the native 3.0 harness without comparison-specific code
   changes.
3. A shared artifact root outside both worktrees. It contains the run-set
   manifest, logs, raw bundles, normalized data, reports, and plots.

The intended persistent layout is:

```text
/home/antoiner/benchmarks/isaaclab2-vs-3/
├── lab2-main/
├── lab3-develop/
└── artifacts/
```

A host-side comparison controller owns matrix expansion, process execution,
validation, resumption, and report generation. It delegates simulator
execution to version-specific executors:

- The 2.x executor runs commands in the locally built Docker image with GPU 0
  exposed and the run output directory mounted.
- The 3.0 executor runs commands through the pinned worktree's uv environment
  with the Isaac Sim/PhysX extra.

Simulator jobs are serialized. The controller never runs two benchmark cells
on the GPU concurrently.

## Ported Benchmark Framework

The 2.x branch gains the 3.0 `isaaclab.test.benchmark` package, including:

- schema-v1 bundle models;
- benchmark builders and capture helpers;
- measurement and metadata models;
- JSON, schema, Osmo, OmniPerf, and summary formatters;
- CPU, GPU, memory, and version recorders;
- background monitoring;
- profiling and stepping utilities;
- serialization and method-benchmark support; and
- public exports and type stubs.

Public import paths, field names, value types, units, and formatter projections
match the pinned 3.0 source. The GPU recorder retains the merged 3.0 sampling
behavior: a one-second monitoring interval, NVML with `nvidia-smi` fallback,
Welford mean and standard deviation, sample count, and peak memory.

The port will not add a required or optional core dependency. Reporting runs
in the 3.0 uv environment, where matplotlib is already a declared dependency.

## Compatibility Boundary

The port keeps benchmark-domain code aligned with 3.0 and isolates 2.x
differences behind small internal adapters.

The adapters cover:

- 3.0 `launch_simulation` behavior using the 2.x `AppLauncher`;
- 3.0 task and agent configuration resolution using 2.x loaders;
- accepted PhysX preset metadata and rejection of unavailable backends;
- 3.0 RL entrypoint helpers using the 2.x training and play APIs;
- RSL-RL, RL-Games, skrl, and Stable-Baselines3 wrapper differences;
- checkpoint discovery and play integration; and
- version/provenance capture for the older package layout.

The port preserves these unified entrypoints:

- `scripts/benchmarks/runtime.py`;
- `scripts/benchmarks/startup.py`;
- `scripts/benchmarks/training.py`;
- `scripts/benchmarks/play.py`;
- all four training adapters;
- all four play adapters;
- early stopping support; and
- Nsight trace configuration.

The comparison uses RSL-RL only, but the complete adapter set remains
functional and tested.

Existing main-branch scripts such as `benchmark_non_rl.py`,
`benchmark_rlgames.py`, and `benchmark_rsl_rl.py` are not removed. Where
appropriate, they become deprecated forwarding wrappers with migration
guidance to the unified entrypoints. Any public helpers in the legacy
`utils.py` remain available for the deprecation period.

The 2.x unified CLI accepts the common 3.0 benchmark arguments. A PhysX
compatibility token may be normalized internally when the 2.x configuration
system has no equivalent preset group. A request for a 3.0-only backend fails
before launching the simulator and names the supported alternative.

## Shared Benchmark Matrix

Every matrix cell uses:

- PhysX;
- GPU 0;
- headless execution;
- schema-v1 output;
- 4,096 environments;
- no benchmark warm-up;
- three paired repetitions with seeds 42, 43, and 44; and
- the same seed on both product versions within a repetition.

The modes are:

| Mode | Work |
|---|---|
| `runtime_100` | 100 measured environment steps |
| `runtime_1000` | 1,000 measured environment steps |
| `training_100` | RSL-RL with `--max_iterations 100` |

The version-specific task aliases are:

| Matrix key | IsaacLab 2.x | IsaacLab 3.0 |
|---|---|---|
| `cartpole` | `Isaac-Cartpole-v0` | `Isaac-Cartpole` |
| `ant` | `Isaac-Ant-v0` | `Isaac-Ant` |
| `anymal_d_flat` | `Isaac-Velocity-Flat-Anymal-D-v0` | `Isaac-Velocity-Flat-AnymalD` |
| `g1_flat` | `Isaac-Velocity-Flat-G1-v0` | `Isaac-Velocity-Flat-G1` |
| `allegro_cube` | `Isaac-Repose-Cube-Allegro-v0` | `Isaac-Reorient-Cube-Allegro` |
| `franka_reach` | `Isaac-Reach-Franka-v0` | `Isaac-Reach-Franka` |

The complete matrix contains 108 runs:

`6 workloads × 3 modes × 3 repetitions × 2 versions`.

Execution is counterbalanced within each task and mode:

- seed 42: 2.x, then 3.0;
- seed 43: 3.0, then 2.x;
- seed 44: 2.x, then 3.0.

This reduces systematic ordering bias without adding cooldown or warm-up
behavior absent from the native harness.

## Controller and Run State

The matrix is defined in a checked-in TOML manifest. TOML keeps the controller
configuration readable while using the standard library parser in the 3.0
Python environment.

Each expanded cell receives a deterministic identity derived from:

- run-set identifier;
- version;
- pinned commit;
- workload key;
- mode;
- environment count;
- seed; and
- executor configuration.

The controller writes the exact command, selected environment variables,
start/end timestamps, exit code, wall time, image or uv identity, and commit
SHA. It captures stdout and stderr separately.

A cell is marked successful only when:

- the subprocess exits with code zero;
- exactly one schema-v1 bundle for the requested workload is identified;
- the bundle matches the requested task, seed, environment count, and mode;
- collection FPS is present;
- mean and peak GPU memory are present;
- mean GPU utilization and its sample count are present; and
- the raw artifact checksum is recorded.

Successful cells are immutable. Resumption skips valid successful cells and
retries failed or incomplete cells into a new attempt directory. Failed
attempts are never deleted automatically.

## Artifact Contract

Each run set uses:

```text
artifacts/<run_set>/
├── manifest.json
├── raw/
│   ├── lab2/<task>/<mode>/seed_<seed>/
│   └── lab3/<task>/<mode>/seed_<seed>/
│       └── attempt_<number>/
│           ├── command.json
│           ├── stdout.log
│           ├── stderr.log
│           ├── validation.json
│           └── *.json
├── normalized/
│   └── results.csv
└── report/
    ├── report.md
    └── plots/
        ├── *.png
        └── *.svg
```

The raw directory preserves native benchmark outputs rather than rewriting
them in place. `validation.json` records the chosen bundle, checksum, schema
checks, and extracted canonical metrics.

The normalized CSV contains one row per successful run and retains provenance
columns sufficient to locate the source bundle. Report generation can be
rerun against a partial or complete run set without executing simulations.

## Reporting

The Markdown report contains:

- both Git revisions and dependency/runtime identities;
- GPU model, driver, CUDA, Isaac Sim, PyTorch, and RL-library versions;
- matrix parameters and task aliases;
- successful, failed, and missing cells;
- per-run values;
- mean and sample standard deviation across repetitions;
- 3.0-versus-2.x percentage deltas; and
- GPU resource sample counts.

The delta formula is:

`(IsaacLab 3.0 - IsaacLab 2.x) / IsaacLab 2.x × 100`.

Positive collection-FPS deltas mean higher throughput. Memory and utilization
deltas are presented without labeling either direction as inherently better.

The rerunnable plotting command generates PNG and SVG versions of:

- collection FPS with individual repeats plus mean/error bars;
- mean GPU memory;
- peak GPU memory;
- mean GPU utilization; and
- a percentage-delta overview separated into runtime-100, runtime-1000, and
  training-100 panels.

Plots read normalized data derived from raw bundles. They do not parse terminal
logs or depend on simulator availability.

## Failure Handling

- A missing required metric invalidates the cell; it is not replaced with
  zero.
- An OOM at 4,096 environments remains a failed cell. The runner does not
  silently reduce the environment count.
- Unsupported task aliases or backends fail before workload execution with a
  diagnostic naming the matrix cell.
- Docker build, image identity, uv synchronization, and GPU visibility are
  preflight checks.
- The controller continues after a cell failure and reports partial results.
- Keyboard interruption leaves completed artifacts and state resumable.
- Low GPU sampling counts remain visible in raw and normalized output.
- Report generation clearly distinguishes zero, unavailable, failed, and
  missing values.

## Validation Strategy

Validation proceeds in layers.

### Baselines

- Run relevant existing 2.x benchmark and task tests before modifying the
  isolated main worktree.
- Run existing 3.0 benchmark unit and smoke tests in the clean uv worktree.
- Record baseline failures and stop for review instead of treating them as
  port regressions.

### Port Tests

- Port schema, measurement, formatter, recorder, capture, builder, stepping,
  and profiling tests before their implementations.
- Use synthetic golden bundles to verify matching keys, types, units, and
  formatter projections across versions.
- Test all four train/play dispatchers and their argument handling.
- Exercise help paths without launching the simulator.
- Run small Cartpole integrations for runtime, startup, training, and play.
- Run end-to-end train-then-play smoke coverage for every installed RL adapter.

### Controller and Report Tests

Synthetic fixtures cover:

- TOML parsing and matrix expansion;
- version-specific task aliases;
- deterministic cell identities;
- counterbalanced ordering;
- command construction for Docker and uv;
- interruption and resumption;
- successful and failed attempt selection;
- schema and provenance validation;
- canonical metric extraction;
- CSV normalization;
- percentage-delta calculations;
- partial reports; and
- PNG/SVG plot generation.

### Performance Acceptance

Before the full matrix, run one canary repetition per workload and mode using
the real 4,096 environment count but reduced runtime frames and training
iterations. The canary validates memory feasibility and integration; it is
stored separately and never mixed into final results.

Final acceptance requires:

- all applicable focused tests passing;
- the complete 108-cell matrix attempted;
- all successful cells validated;
- CSV, Markdown, PNG, and SVG artifacts regenerated solely from raw files;
- public documentation generated with `./isaaclab.sh -d`; and
- `./isaaclab.sh -f` passing after any formatter modifications are reviewed.

## Documentation and Changelog

The 2.x branch gains user documentation for the unified benchmark API and
entrypoints, including the PhysX-only compatibility boundary and migration
from legacy scripts. Public symbols are added to generated API documentation.

Each touched package receives one changelog fragment. Deprecated legacy
entrypoints are documented under `Deprecated` with migration guidance.
Compiled changelogs and extension versions are not edited directly.

New files use the 2026 Isaac Lab copyright header. Python follows PEP 8,
modern type hints, Google-style docstrings, and SI-unit annotations for public
physical quantities.

## Delivery Boundaries

The work is delivered in independently reviewable stages:

1. Core schema and framework port.
2. System recorders and formatters.
3. Unified runtime and startup entrypoints.
4. Unified training and play adapters.
5. Legacy deprecation wrappers and documentation.
6. Shared matrix controller and artifact validation.
7. Normalization, reporting, and plots.
8. Docker/uv preflight, canary execution, and full matrix execution.

Each stage has its own focused tests and commit. The full implementation plan
will define exact files, test commands, expected failures, and commit
boundaries.
