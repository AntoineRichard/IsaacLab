# IsaacLab 2.x Benchmark Harness Backport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Backport the current IsaacLab 3.0 benchmark harness to the pinned IsaacLab 2.x main revision, then run a reproducible RSL-RL comparison that reports collection FPS, GPU memory, and GPU utilization for six matched tasks.

**Architecture:** Keep the upstream benchmark schema and measurement implementation as the compatibility boundary. Add a small 2.x adapter layer around launch, task configuration, Gym environment creation, checkpoint handling, and RL-library dispatch. A separate standard-library controller owns the paired matrix, immutable artifacts, validation, normalization, reporting, and plotting across a locally built 2.x Docker image and a pinned 3.0 uv checkout.

**Tech Stack:** Python 3.11, Isaac Lab, Isaac Sim, Gymnasium, RSL-RL, Docker Compose, uv, pytest, NVIDIA SMI, Matplotlib, reStructuredText, Markdown.

## Fixed inputs and invariants

- Backport base: origin/main at 858234d06e.
- Harness source: origin/develop at cb508381fb.
- Development worktree: /tmp/isaaclab-backport-benchmark-harness.
- Development branch: antoiner/backport-benchmark-harness.
- Persistent comparison root: /home/antoiner/benchmarks/isaaclab2-vs-3.
- Lab 2 execution: locally built Docker image from the backport worktree.
- Lab 3 execution: uv environment from a detached worktree pinned to cb508381fb.
- Physics comparison: PhysX only.
- RL library: RSL-RL only for measured training.
- Environment count: 4096 for every measured cell.
- Repeats and seeds: 42, 43, and 44.
- Modes: runtime at 100 steps, runtime at 1000 steps, and training at 100 iterations.
- Pair order: lab2 then lab3 for seed 42, lab3 then lab2 for seed 43, lab2 then lab3 for seed 44.
- Full matrix size: 6 tasks times 3 modes times 3 seeds times 2 versions equals 108 attempts.
- Do not add benchmark warm-up semantics. Environment smoke and canary results are stored separately from measured results.
- Failed and out-of-memory attempts remain failures. The controller must never silently lower the environment count.
- Successful raw artifacts are immutable. Reports and plots are always regenerable from raw artifacts.
- The report is informational only and has no pass/fail regression threshold.
- Every new source file uses the 2026 Isaac Lab copyright header.
- A measured attempt starts only after the host passes the idle gate. The runner never kills unrelated work to make the machine idle.

## Task 0: Prepare pinned execution checkouts

**Files:**

- Verify: /tmp/isaaclab-backport-benchmark-harness
- Move worktree to: /home/antoiner/benchmarks/isaaclab2-vs-3/lab2-main
- Create worktree: /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop
- Create directory: /home/antoiner/benchmarks/isaaclab2-vs-3/artifacts

- [ ] **Step 1: Verify the development worktree and branch**

Run:

~~~bash
git -C /tmp/isaaclab-backport-benchmark-harness status --short --branch
git -C /tmp/isaaclab-backport-benchmark-harness rev-parse HEAD
git -C /tmp/isaaclab-backport-benchmark-harness rev-parse origin/main
git -C /tmp/isaaclab-backport-benchmark-harness rev-parse origin/develop
~~~

Expected: the branch is antoiner/backport-benchmark-harness, its base contains 858234d06e, and the two remote revisions resolve to the pinned inputs above.

- [ ] **Step 2: Move the feature worktree and create the detached 3.0 worktree**

Move /tmp/isaaclab-backport-benchmark-harness to the persistent lab2-main path with git worktree move, retaining the checked-out feature branch. Create lab3-develop as a detached worktree at cb508381fb. Do not reuse the dirty original checkout and do not try to check out the feature branch in two worktrees.

- [ ] **Step 3: Create and synchronize the 3.0 uv environment**

Run uv sync from lab3-develop with the Isaac Sim and RSL-RL extras required by the pinned checkout. Record the uv version, lockfile hash, Python version, and installed Isaac Sim version in artifacts/preflight/lab3.json.

- [ ] **Step 4: Prove the unmodified 2.x Docker baseline builds**

Use the repository Docker helper and a benchmark suffix to build and start the base image. Record the image ID, Dockerfile inputs, driver version, GPU model, and Isaac Sim version in artifacts/preflight/lab2.json.

- [ ] **Step 5: Record pinned baseline behavior**

Run each existing 2.x legacy benchmark entrypoint with --help, then use 16 environments for a minimal Cartpole runtime in both versions. Run the pinned 3.0 benchmark unit tests before backport changes. Store commands, logs, and any pre-existing failures under artifacts/baseline. This is an environment check only; do not include its output in the comparison report.

## Task 1: Backport the typed benchmark core

**Files:**

- Create: source/isaaclab/isaaclab/test/benchmark/__init__.py
- Create: source/isaaclab/isaaclab/test/benchmark/__init__.pyi
- Create: source/isaaclab/isaaclab/test/benchmark/benchmark_core.py
- Create: source/isaaclab/isaaclab/test/benchmark/benchmark_monitor.py
- Create: source/isaaclab/isaaclab/test/benchmark/builders.py
- Create: source/isaaclab/isaaclab/test/benchmark/capture.py
- Create: source/isaaclab/isaaclab/test/benchmark/formatters.py
- Create: source/isaaclab/isaaclab/test/benchmark/interfaces.py
- Create: source/isaaclab/isaaclab/test/benchmark/measurements.py
- Create: source/isaaclab/isaaclab/test/benchmark/method_benchmark.py
- Create: source/isaaclab/isaaclab/test/benchmark/metrics.py
- Create: source/isaaclab/isaaclab/test/benchmark/profiling.py
- Create: source/isaaclab/isaaclab/test/benchmark/schema.py
- Create: source/isaaclab/isaaclab/test/benchmark/serialize.py
- Create: source/isaaclab/isaaclab/test/benchmark/stepping.py
- Create: source/isaaclab/isaaclab/test/benchmark/recorders/__init__.py
- Create: source/isaaclab/isaaclab/test/benchmark/recorders/__init__.pyi
- Create: source/isaaclab/isaaclab/test/benchmark/recorders/record_cpu_info.py
- Create: source/isaaclab/isaaclab/test/benchmark/recorders/record_gpu_info.py
- Create: source/isaaclab/isaaclab/test/benchmark/recorders/record_memory_info.py
- Create: source/isaaclab/isaaclab/test/benchmark/recorders/record_version_info.py
- Create: source/isaaclab/test/benchmark/test_benchmark_core.py
- Create: source/isaaclab/test/benchmark/test_builders.py
- Create: source/isaaclab/test/benchmark/test_capture.py
- Create: source/isaaclab/test/benchmark/test_formatters.py
- Create: source/isaaclab/test/benchmark/test_metrics.py
- Create: source/isaaclab/test/benchmark/test_play_schema.py
- Create: source/isaaclab/test/benchmark/test_profiling.py
- Create: source/isaaclab/test/benchmark/test_recorders.py
- Create: source/isaaclab/test/benchmark/test_schema.py
- Create: source/isaaclab/test/benchmark/test_stepping.py

- [ ] **Step 1: Restore the pinned upstream tests before implementation**

Restore only the listed tests from cb508381fb, keeping their contents byte-identical initially.

- [ ] **Step 2: Run one focused test and capture the expected failure**

Run:

~~~bash
./isaaclab.sh -p -m pytest source/isaaclab/test/benchmark/test_schema.py -q
~~~

Expected: collection fails because isaaclab.test.benchmark does not exist.

- [ ] **Step 3: Restore the pinned implementation**

Restore the benchmark package and recorder modules from cb508381fb. Preserve all schema names, formatter names, JSON field names, measurement units, and public type signatures.

- [ ] **Step 4: Add the two minimal 2.x compatibility adaptations**

Replace lazy exports with eager imports because 2.x has no isaaclab.utils.module.lazy_export. In capture.py, guard the 3.0-only PresetTarget import and provide stable physics, renderer, and preset metadata for 2.x, mapping Newton and Kamino names without making those solvers runnable.

- [ ] **Step 5: Run the complete benchmark-core test directory**

Run:

~~~bash
./isaaclab.sh -p -m pytest source/isaaclab/test/benchmark -q
~~~

Expected: all ported tests pass.

- [ ] **Step 6: Run formatting twice and commit**

Run ./isaaclab.sh -f, review any changes, stage them, run ./isaaclab.sh -f again, then commit:

~~~text
Add typed benchmark measurement core
~~~

## Task 2: Add the 2.x script compatibility layer and runtime harness

**Files:**

- Create: scripts/benchmarks/_compat.py
- Create: scripts/benchmarks/runtime.py
- Create: scripts/benchmarks/startup.py
- Create: scripts/benchmarks/early_stop.py
- Create: scripts/benchmarks/startup_whitelist.yaml
- Create: scripts/benchmarks/nsys_trace.json
- Create: scripts/benchmarks/test/conftest.py
- Create: scripts/benchmarks/test/test_early_stop.py
- Create: scripts/benchmarks/test/test_nsys_trace.py
- Create: scripts/benchmarks/test/test_runtime_smoke.py
- Create: scripts/benchmarks/test/test_startup_smoke.py

- [ ] **Step 1: Restore pinned script tests and verify failure**

Restore the listed tests from cb508381fb, then run:

~~~bash
./isaaclab.sh -p -m pytest scripts/benchmarks/test/test_runtime_smoke.py scripts/benchmarks/test/test_startup_smoke.py -q
~~~

Expected: imports or CLI construction fail because the new scripts and compatibility layer are absent.

- [ ] **Step 2: Implement _compat.py with a narrow, typed surface**

Provide these operations using 2.x APIs:

- add_launcher_args
- parse_benchmark_args
- launch_app
- resolve_task_config
- apply_env_overrides
- add_common_train_args
- enable_cameras_for_video
- create_isaaclab_env
- wrap_record_video
- dispatch_library_entrypoint
- write_run_manifest
- resolve_play_checkpoint
- resolve_checkpoint_selector

Keep it private to scripts/benchmarks. Do not backport or expose 3.0 public isaaclab_rl entrypoint APIs.

- [ ] **Step 3: Restore runtime, startup, and helper scripts**

Use the pinned scripts as the baseline. Adapt launch ordering so CLI parsing and AppLauncher construction occur before Gym task registration and config lookup. Keep the 3.0 CLI argument names and benchmark formatter choices.

- [ ] **Step 4: Add compatibility tests**

Test that launch arguments are registered once, task lookup happens after launch, schema and JSON formatters can be requested together, run manifests include both Git SHAs, and 2.x rejects unsupported 3.0 solver presets with an actionable error.

- [ ] **Step 5: Run unit and smoke verification**

Run the four script test files under scripts/benchmarks/test that cover runtime, startup, early stopping, and Nsight trace configuration. Then execute an unmeasured 16-environment Cartpole runtime for 10 steps inside the 2.x Docker image with schema and JSON output enabled.

- [ ] **Step 6: Format twice and commit**

Commit:

~~~text
Backport runtime benchmark scripts
~~~

## Task 3: Backport RSL-RL training and play

**Files:**

- Create: scripts/benchmarks/training.py
- Create: scripts/benchmarks/play.py
- Create: scripts/benchmarks/rsl_rl/benchmark_rsl_rl_train.py
- Create: scripts/benchmarks/rsl_rl/benchmark_rsl_rl_play.py
- Create: scripts/benchmarks/test/test_benchmark_smoke.py
- Create: scripts/benchmarks/test/test_training_adapters.py

- [ ] **Step 1: Restore pinned tests and verify failure**

Run the RSL-RL-focused training and play tests before adding the adapter.

Expected: dispatcher or adapter imports fail.

- [ ] **Step 2: Restore the pinned training, play, and RSL-RL sources**

Retain upstream benchmark phase names and measurement boundaries. Adapt only CLI/config resolution, legacy RSL-RL wrapper construction, log directory setup, checkpoint selectors, and AppLauncher lifetime.

- [ ] **Step 3: Verify exact measured CLI behavior**

Ensure training accepts --task, --num_envs, --seed, --max_iterations, --benchmark_formatter schema,json, and a PhysX preset selection. Preserve 3.0 names where 2.x has a direct semantic equivalent.

- [ ] **Step 4: Add tests for legacy dispatch**

Cover task config resolution, seed propagation, max-iteration override, headless operation, checkpoint resolution, manifest creation, and cleanup after exceptions.

- [ ] **Step 5: Run smoke training and play**

Inside Docker, run Cartpole with 16 environments and two RSL-RL iterations. Confirm both schema and generic JSON artifacts parse, then play the generated checkpoint briefly.

- [ ] **Step 6: Format twice and commit**

Commit:

~~~text
Backport RSL-RL benchmark adapters
~~~

## Task 4: Complete the upstream harness surface

**Files:**

- Create: scripts/benchmarks/rl_games/benchmark_rl_games_train.py
- Create: scripts/benchmarks/rl_games/benchmark_rl_games_play.py
- Create: scripts/benchmarks/sb3/benchmark_sb3_train.py
- Create: scripts/benchmarks/sb3/benchmark_sb3_play.py
- Create: scripts/benchmarks/skrl/benchmark_skrl_train.py
- Create: scripts/benchmarks/skrl/benchmark_skrl_play.py
- Modify: scripts/benchmarks/test/test_benchmark_smoke.py
- Modify: scripts/benchmarks/test/test_training_adapters.py

- [ ] **Step 1: Restore the remaining pinned tests and verify failure**

Run the adapter CLI and import tests before copying the implementations.

- [ ] **Step 2: Restore the remaining adapters**

Keep all four current upstream adapter families present even though the comparison controller selects only RSL-RL. Route their shared launch, environment, video, and manifest behavior through _compat.py.

- [ ] **Step 3: Run import, help, and focused unit tests**

Every adapter entrypoint must import under 2.x and render --help without launching Isaac Sim. Run any upstream tests that do not require unavailable external services.

- [ ] **Step 4: Validate the profiler wrapper**

Run the nsys wrapper test and confirm it forwards formatter and task arguments without shell interpolation.

- [ ] **Step 5: Format twice and commit**

Commit:

~~~text
Complete benchmark adapter backport
~~~

## Task 5: Preserve and deprecate legacy benchmark entrypoints

**Files:**

- Modify: scripts/benchmarks/benchmark_non_rl.py
- Modify: scripts/benchmarks/benchmark_rlgames.py
- Modify: scripts/benchmarks/benchmark_rsl_rl.py
- Preserve: scripts/benchmarks/utils.py
- Create: scripts/benchmarks/test_legacy_entrypoints.py
- Create: docs/source/testing/index.rst
- Create: docs/source/testing/benchmarks.rst
- Modify: docs/index.rst
- Modify: docs/source/api/index.rst
- Create: docs/source/api/lab/isaaclab.test.rst
- Create: source/isaaclab/changelog.d/benchmark-harness-backport.minor.rst

- [ ] **Step 1: Add failing translation tests**

Assert that each legacy script still accepts its historical flags, emits a deprecation warning, and forwards to the corresponding new entrypoint. Translate legacy backends as follows: LocalLog to summary, JSON to json, Osmo to osmo, and OmniPerf to omniperf.

- [ ] **Step 2: Convert legacy scripts to thin compatibility wrappers**

Do not remove or rename them. Preserve exit codes and user-facing errors. Keep utils.py available for downstream imports and delegate new behavior to the typed harness.

- [ ] **Step 3: Port and adapt benchmark documentation**

Document runtime, startup, training, play, formatters, raw artifacts, 2.x limitations, and legacy migration. Add the testing section to docs/index.rst and the public isaaclab.test API page to the API autosummary.

- [ ] **Step 4: Add the changelog fragment**

Use Added and Deprecated sections, past tense, and explicit migration guidance from the three legacy scripts to runtime.py and training.py.

- [ ] **Step 5: Verify tests and docs**

Run the legacy tests, benchmark test suite, and:

~~~bash
./isaaclab.sh -d
~~~

- [ ] **Step 6: Format twice and commit**

Commit:

~~~text
Deprecate legacy benchmark entrypoints
~~~

## Task 6: Define the shared comparison matrix

**Files:**

- Create: tools/benchmark_comparison/__init__.py
- Create: tools/benchmark_comparison/models.py
- Create: tools/benchmark_comparison/matrix.py
- Create: tools/benchmark_comparison/matrix.toml
- Create: tools/benchmark_comparison/tests/test_matrix.py

- [ ] **Step 1: Write failing matrix tests**

Assert exact task aliases:

| Logical task | IsaacLab 2.x | IsaacLab 3.0 |
|---|---|---|
| cartpole | Isaac-Cartpole-v0 | Isaac-Cartpole |
| ant | Isaac-Ant-v0 | Isaac-Ant |
| anymal_d_flat | Isaac-Velocity-Flat-Anymal-D-v0 | Isaac-Velocity-Flat-AnymalD |
| g1_flat | Isaac-Velocity-Flat-G1-v0 | Isaac-Velocity-Flat-G1 |
| allegro_cube | Isaac-Repose-Cube-Allegro-v0 | Isaac-Reorient-Cube-Allegro |
| franka_reach | Isaac-Reach-Franka-v0 | Isaac-Reach-Franka |

Also assert 4096 environments, modes runtime-100, runtime-1000, and training-100, seeds 42 through 44, counterbalanced ordering, 54 logical pairs, and 108 total attempts.

- [ ] **Step 2: Implement immutable matrix models**

Use frozen dataclasses and enums. Make task aliases, mode parameters, version identity, seed, repeat index, and pair order explicit. Do not derive task names heuristically.

- [ ] **Step 3: Add a canary selector**

The canary matrix contains all six tasks and three modes at seed 42 for both versions: 36 attempts. Use 10 steps for the runtime-100 canary, 25 steps for the runtime-1000 canary, and 2 iterations for the training canary, always with 4096 environments. It uses the same command generation and validation path as the full matrix but writes to a separate canary run directory.

- [ ] **Step 4: Run tests and commit**

Commit:

~~~text
Define paired benchmark matrix
~~~

## Task 7: Implement immutable artifacts and semantic validation

**Files:**

- Create: tools/benchmark_comparison/artifacts.py
- Create: tools/benchmark_comparison/validate.py
- Create: tools/benchmark_comparison/tests/fixtures/schema_runtime.json
- Create: tools/benchmark_comparison/tests/fixtures/generic_runtime.json
- Create: tools/benchmark_comparison/tests/fixtures/schema_training.json
- Create: tools/benchmark_comparison/tests/fixtures/generic_training.json
- Create: tools/benchmark_comparison/tests/test_artifacts.py
- Create: tools/benchmark_comparison/tests/test_validate.py

- [ ] **Step 1: Write failing artifact-layout tests**

Require one attempt directory per version, task, mode, and seed containing command.json, environment.json, stdout.log, stderr.log, exit.json, schema.json, measurements.json, validation.json, and checksums.sha256.

- [ ] **Step 2: Write failing semantic validation tests**

A success must have exit code zero, matching task and seed metadata, the requested step or iteration bound, nonempty phase timings, collection FPS, mean and peak GPU memory, mean GPU utilization, and GPU utilization sample count. Take semantic values from schema.json and the sample count from measurements.json.

- [ ] **Step 3: Implement atomic artifact finalization**

Write into an attempt-local staging directory, validate, compute hashes, then atomically rename it to the final success directory. Never overwrite a successful directory. Preserve failed attempts with a monotonically increasing attempt number.

- [ ] **Step 4: Implement strict failure classification**

Classify setup, launch, timeout, out-of-memory, nonzero exit, malformed artifact, identity mismatch, and missing metric separately. Store the reason without converting a failure into a numeric zero.

- [ ] **Step 5: Run tests and commit**

Commit:

~~~text
Validate immutable benchmark artifacts
~~~

## Task 8: Implement Docker and uv executors plus the resumable runner

**Files:**

- Create: tools/benchmark_comparison/docker-compose.benchmark.yaml
- Create: tools/benchmark_comparison/executors.py
- Create: tools/benchmark_comparison/runner.py
- Create: tools/benchmark_comparison/cli.py
- Create: tools/benchmark_comparison/tests/test_executors.py
- Create: tools/benchmark_comparison/tests/test_runner.py
- Create: tools/benchmark_comparison/tests/test_idle_gate.py

- [ ] **Step 1: Write failing command-generation tests**

Require argument-vector commands with no shell evaluation. Lab 2 commands run through the locally built benchmark container and mount the artifact root at /benchmark_artifacts. Lab 3 commands use uv run with the pinned project, Isaac Sim extra, RSL-RL entrypoint, and PhysX preset.

- [ ] **Step 2: Add the Docker Compose override**

Reuse the repository base service, add a benchmark suffix, mount only the pinned lab2 worktree and shared artifact root, request all NVIDIA GPUs, and keep host networking and display behavior consistent with existing Docker tooling.

- [ ] **Step 3: Implement executor preflight**

Validate Git SHAs, clean worktrees, Docker image identity, uv lock state, NVIDIA SMI access, free disk space, writable artifact root, task registration, and both formatter names before any measured attempt starts.

- [ ] **Step 4: Implement the host-idle gate**

Before every canary or final attempt, require all of the following:

- no unexpected NVIDIA compute process or GPU-enabled Docker container;
- GPU utilization at or below 5 percent for 60 consecutive one-second samples;
- GPU memory no more than 1024 MiB above the idle baseline recorded during preflight;
- one-minute host load average at or below 25 percent of the logical CPU count; and
- no child process remaining from an earlier benchmark attempt.

Persist the 60 raw NVIDIA SMI samples, process inventory, load average, idle baseline, thresholds, and decision beside the attempt. When the gate fails, wait five minutes and retry without modifying or terminating unrelated processes. After a configurable idle timeout, record a preflight failure and stop the run set before launching another simulator process.

- [ ] **Step 5: Implement the resumable paired runner**

Execute one attempt at a time to avoid GPU contention. Follow the approved version order for each seed. Skip only a semantically valid success. Retry failures only when explicitly requested. Persist runner state after every attempt and handle interruption without losing completed artifacts.

- [ ] **Step 6: Add timeout and signal handling**

Terminate the child cleanly, then the container or process group if needed. Record timeout or interruption distinctly and leave a recoverable failed attempt.

- [ ] **Step 7: Test with fake executors**

Cover complete run, resume, corrupt-success rerun, failure preservation, counterbalancing, interruption, timeout, out-of-memory classification, idle acceptance, busy-GPU rejection, busy-host rejection, retry, and idle timeout.

- [ ] **Step 8: Run tests and commit**

Commit:

~~~text
Add resumable benchmark runner
~~~

## Task 9: Normalize results and generate reports and plots

**Files:**

- Create: tools/benchmark_comparison/normalize.py
- Create: tools/benchmark_comparison/report.py
- Create: tools/benchmark_comparison/plot.py
- Create: tools/benchmark_comparison/tests/test_normalize.py
- Create: tools/benchmark_comparison/tests/test_report.py
- Create: tools/benchmark_comparison/tests/test_plot.py

- [ ] **Step 1: Write failing normalization tests**

Require one row per successful attempt with version SHA, environment identity, logical and concrete task names, mode, bound, seed, environment count, collection FPS, mean and peak GPU memory in MiB, mean GPU utilization in percent, utilization sample count, elapsed time, and artifact path.

- [ ] **Step 2: Implement paired summaries**

Aggregate repeats with mean and standard deviation, retain the individual rows, and compute lab3-versus-lab2 absolute and percent deltas only when both members of a pair are valid. Do not impute missing failures.

- [ ] **Step 3: Generate Markdown and CSV outputs**

Write raw_runs.csv, paired_summary.csv, failures.csv, and report.md. The report includes methodology, pinned SHAs, hardware and software inventory, task mapping, per-mode tables, paired deltas, sample counts, and linked failure artifacts.

- [ ] **Step 4: Generate deterministic plots**

Produce PNG and SVG for collection FPS, mean GPU memory, peak GPU memory, and mean GPU utilization. Group by mode and task, show both versions, include repeat variability, label missing cells, use fixed ordering and colors, and keep plotting rerunnable from normalized CSV alone.

- [ ] **Step 5: Add snapshot and image-structure tests**

Test stable row ordering, delta signs, failure rendering, required Markdown sections, plot filenames, nonempty output, and deterministic dimensions.

- [ ] **Step 6: Run tests and commit**

Commit:

~~~text
Report paired benchmark results
~~~

## Task 10: Verify the backport and execute the comparison

**Files:**

- Read: all files changed above
- Generate outside Git: /home/antoiner/benchmarks/isaaclab2-vs-3/artifacts

- [ ] **Step 1: Run focused unit tests**

Run:

~~~bash
./isaaclab.sh -p -m pytest source/isaaclab/test/benchmark scripts/benchmarks/test tools/benchmark_comparison/tests -q
~~~

- [ ] **Step 2: Pin the persistent Lab 2 worktree and rebuild its final image**

Ensure the feature worktree is at /home/antoiner/benchmarks/isaaclab2-vs-3/lab2-main, clean, and at the exact feature commit that passed tests. Build from that commit and record the commit and image digest. Do not run measured cells from an image built from an uncommitted tree.

- [ ] **Step 3: Run final preflight and task-registration checks**

Resolve all six task aliases in both versions, verify 4096 environments are accepted by CLI/config composition, record the idle GPU-memory baseline, and write a preflight report. This does not execute measured cells.

- [ ] **Step 4: Run the separate 36-attempt canary**

Execute seed 42 across all six tasks, the three reduced canary bounds, and both versions at 4096 environments. Store this under a distinct canary run-set ID. Stop before the full matrix if any task mapping, required metric, artifact identity, or formatter validation fails.

- [ ] **Step 5: Inspect canary artifacts and regenerate its report**

Manually inspect at least one runtime and one training artifact from each version. Regenerate CSV, Markdown, PNG, and SVG from raw data to prove report independence.

- [ ] **Step 6: Run or resume the full 108-attempt matrix**

Use the same artifact root and runner but a new final run-set ID. Never reuse reduced canary results as final measurements. Require and archive a fresh idle-gate decision before every attempt; if the machine becomes busy, pause the matrix until the gate passes again.

- [ ] **Step 7: Validate matrix completeness**

Require 108 terminal attempt records. Report valid successes and failures separately. Check that every successful record has all four requested metrics and a positive GPU sample count.

- [ ] **Step 8: Regenerate the final report twice**

Hash normalized tables and plots after each generation. Expected: identical outputs from identical raw artifacts.

- [ ] **Step 9: Run full formatting and documentation verification**

Run:

~~~bash
./isaaclab.sh -f
./isaaclab.sh -d
git diff --check
git status --short
~~~

If formatting changes files, review and stage them, rerun ./isaaclab.sh -f, and add a focused cleanup commit before any push.

- [ ] **Step 10: Perform final review**

Confirm no public API was removed, legacy scripts remain callable with deprecation warnings, no new required dependency was added, raw results are outside Git, both source SHAs and environment manifests are present, and every report value traces to a raw artifact.

## Completion criteria

- The pinned 3.0 benchmark core, scripts, formatters, schema, adapters, tests, and docs exist on the 2.x feature branch with narrowly scoped compatibility changes.
- Benchmark core and script tests pass under the 2.x environment.
- Legacy public entrypoints remain usable and provide migration guidance.
- The controller expands exactly 108 attempts and resumes without overwriting valid successes.
- Every measured attempt has an accepted idle-gate record proving the host and GPU were not busy immediately before launch.
- Every success reports collection FPS, mean and peak GPU memory, mean GPU utilization, and GPU utilization sample count.
- Raw schema and generic JSON artifacts, logs, manifests, validation records, normalized CSV files, Markdown report, PNG plots, and SVG plots are retained.
- The final report is informational and clearly exposes failures rather than hiding or imputing them.
- ./isaaclab.sh -f passes after all code changes, and documentation builds successfully.
