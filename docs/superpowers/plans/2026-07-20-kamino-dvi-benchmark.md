# Kamino DVI Training Benchmark Implementation Plan

> **For Codex:** Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` to execute this plan task by task. Apply test-driven development to every behavior change and use `superpowers:verification-before-completion` before commits, pushes, or completion claims.

**Goal:** Build and execute a reproducible RSL-RL benchmark that compares current Kamino P-ADMM, Newton PR 3570 P-ADMM and DVI, MJWarp, and PhysX on the approved five-task matrix, then publish a Markdown/PDF report with 95% confidence intervals and stability evidence.

**Architecture:** Add a backward-compatible Kamino solver-selection bridge and a benchmark-local ANYmal-D Kamino preset. Put experiment orchestration, immutable manifests, TensorBoard/schema parsing, statistics, plotting, and reporting in `benchmarks/kamino_dvi/`. Keep large raw runs under ignored `benchmark_artifacts/kamino_dvi/`; track only the matrix, tooling, compact summaries, figures, and final reports.

**Tech Stack:** Python 3.12, IsaacLab unified training benchmark, RSL-RL, Hydra presets, PyYAML, TensorBoard event accumulator, matplotlib, Python Markdown, headless LibreOffice, pytest.

**Design:** `docs/superpowers/specs/2026-07-20-kamino-dvi-benchmark-design.md`

## Global execution rules

- Work only in `/tmp/isaaclab-kamino-dvi-benchmark` on `antoiner/kamino-dvi-benchmark`.
- Preserve the original dirty checkout and the separate schema worktree.
- Use `./isaaclab.sh -p` for Python and pytest commands.
- For each regression test, demonstrate RED before implementation and GREEN afterward. For a bug fix, temporarily revert the fix and prove the regression test fails before reapplying it.
- Run `PATH=/tmp/isaaclab-git-lfs-tool/extracted/usr/bin:$PATH ./isaaclab.sh -f` before every commit and again before any push.
- Never push to `origin`; use the `antoine` fork only after explicit execution approval.
- New files use the 2026 Isaac Lab SPDX header.
- Do not add required or optional dependencies. Use the packages already present in the locked development environment; calculate the five-seed Student-t interval with the fixed `df=4` critical value rather than importing SciPy.
- Never hide missing benchmark fields. Record them as a benchmark-stack bug and stop the affected analysis path.

## Task 1: Add the backward-compatible Kamino dynamics solver bridge

**Files:**

- Modify: `source/isaaclab_newton/isaaclab_newton/physics/kamino_manager_cfg.py`
- Create: `source/isaaclab_newton/test/physics/test_kamino_manager_cfg.py`
- Modify: `docs/source/overview/core-concepts/physical-backends/newton/kamino-solver.rst`
- Create: `source/isaaclab_newton/changelog.d/antoiner-kamino-dynamics-solver.minor.rst`

1. Write focused tests using monkeypatched fake `newton._src.solvers.kamino.config` and `newton.solvers.SolverKamino` modules. Assert that `KaminoSolverCfg().to_solver_config()` does not pass `dynamics_solver`, while `KaminoSolverCfg(dynamics_solver="dvi")` passes exactly `dynamics_solver="dvi"`. Preserve assertions for the existing nested config objects.
2. Run `./isaaclab.sh -p -m pytest source/isaaclab_newton/test/physics/test_kamino_manager_cfg.py -q` and confirm the DVI test fails because the public field is absent.
3. Add `dynamics_solver: str | None = None` with a concise public docstring. Build a keyword dictionary in `to_solver_config()` and insert `dynamics_solver` only when non-`None`; do not pass `None` to current Newton.
4. Rerun the focused test and the existing manager abstraction tests:
   `./isaaclab.sh -p -m pytest source/isaaclab_newton/test/physics/test_kamino_manager_cfg.py source/isaaclab_newton/test/physics/test_newton_manager_abstraction.py -q`.
5. Add the field to the Kamino solver documentation under core integration, stating that `None` preserves Newton's default and `"dvi"` selects PR 3570's experimental solver. Add a minor changelog fragment under `Added`.
6. Run `./isaaclab.sh -d`, review generated documentation changes, then run full pre-commit twice if the first run modifies files.
7. Commit: `git commit -m "Add Kamino dynamics solver selection"`.

## Task 2: Add the benchmark-local ANYmal-D Kamino preset

**Files:**

- Modify: `source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/anymal_d/flat_env_cfg.py`
- Create: `source/isaaclab_tasks/test/core/test_anymal_d_physics_presets.py`
- Create: `source/isaaclab_tasks/changelog.d/antoiner-kamino-dvi-benchmark.skip`

1. Write a config-only test asserting that `PhysicsCfg.newton_kamino` uses `NewtonCfg`, `KaminoSolverCfg(max_contacts_per_world=64)`, one substep, and `debug_mode=False`. Assert that existing `default`, `physx`, `newton_mjwarp`, and `ovphysx` values remain available.
2. Run `./isaaclab.sh -p -m pytest source/isaaclab_tasks/test/core/test_anymal_d_physics_presets.py -q` and confirm it fails because `newton_kamino` is absent.
3. Import `KaminoSolverCfg` and add the exact approved preset without variant-specific tuning.
4. Rerun the focused test, then run the task configuration tests that cover ANYmal-D registration and preset resolution.
5. Add a `.skip` changelog fragment because this preset is benchmark-local support rather than a general support claim.
6. Run full pre-commit and commit: `git commit -m "Add ANYmal-D Kamino benchmark preset"`.

## Task 3: Define typed experiment records and the immutable matrix

**Files:**

- Create: `benchmarks/__init__.py`
- Create: `benchmarks/kamino_dvi/__init__.py`
- Create: `benchmarks/kamino_dvi/models.py`
- Create: `benchmarks/kamino_dvi/matrix.py`
- Create: `benchmarks/kamino_dvi/matrix.yaml`
- Create: `benchmarks/kamino_dvi/tests/__init__.py`
- Create: `benchmarks/kamino_dvi/tests/test_matrix.py`

1. Write tests for exactly five tasks, five variants, seeds 42–46, the `4096, 2048, 1024, 512, 256, 128` ladder, 21 task/variant cells, 105 full-run identities, and five-iteration seed-42 preflights.
2. Write parametrized tests for environment labels and variant applicability: common tasks receive all five variants; Kamino-only tasks receive only the three Kamino variants.
3. Write tests for deterministic counterbalancing: rotate by seed index and reverse base order on alternating tasks.
4. Run the tests and confirm imports/files are absent.
5. Implement frozen dataclasses and string enums for task, variant, environment, phase, terminal state, failure category, run identity, retry lineage, and manifest. Parse `matrix.yaml` with `yaml.safe_load` and validate duplicates, revisions, positive counts/timeouts, and matrix cardinality.
6. Rerun `./isaaclab.sh -p -m pytest benchmarks/kamino_dvi/tests/test_matrix.py -q`, then full pre-commit.
7. Commit: `git commit -m "Define Kamino DVI benchmark matrix"`.

## Task 4: Build commands and validate isolated Newton environments

**Files:**

- Create: `benchmarks/kamino_dvi/commands.py`
- Create: `benchmarks/kamino_dvi/environment.py`
- Create: `benchmarks/kamino_dvi/tests/test_commands.py`
- Create: `benchmarks/kamino_dvi/tests/test_environment.py`

1. Write command tests asserting use of `scripts/benchmarks/training.py`, `--rl_library rsl_rl`, exact task, seed, count, `--max_iterations`, output path, schema formatter, and correct preset. Assert DVI adds only `env.sim.physics.solver_cfg.dynamics_solver=dvi` after `presets=newton_kamino`; PR P-ADMM leaves the field unset.
2. Assert current Kamino/MJWarp/PhysX select `.venv-current/bin/python`, while PR P-ADMM/DVI select `.venv-pr3570/bin/python`. Assert commands are argv lists with no shell interpolation.
3. Write revision-validation tests against synthetic `importlib.metadata`, `newton.__file__`, and Git metadata. Mismatched IsaacLab/Newton revisions must fail before training.
4. Run both tests and observe RED.
5. Implement pure command construction and environment validation. Add version/provenance capture that records all installed distributions plus the importable Newton path and source commit.
6. Rerun focused tests and full pre-commit.
7. Commit: `git commit -m "Build benchmark commands and environments"`.

## Task 5: Implement atomic manifests and resumability

**Files:**

- Create: `benchmarks/kamino_dvi/manifests.py`
- Create: `benchmarks/kamino_dvi/tests/test_manifests.py`

1. Write tests for stable run IDs, canonical command hashes, relative artifact paths, SHA-256 files, atomic temp-file replacement, and state transitions `planned -> running -> completed|failed`.
2. Write resume tests showing a completed run is skipped only when command hash, revisions, schema version, environment count, and terminal success all match. Failed, incomplete, stale, or corrupt manifests must rerun.
3. Run focused tests and observe RED.
4. Implement JSON serialization with `dataclasses.asdict`, sorted keys, `Path.replace`, and fsync before replacement. Validate legal transitions and retain retry ancestry.
5. Rerun focused tests and full pre-commit.
6. Commit: `git commit -m "Add resumable benchmark manifests"`.

## Task 6: Implement failure classification and capacity fallback

**Files:**

- Create: `benchmarks/kamino_dvi/failures.py`
- Create: `benchmarks/kamino_dvi/scheduler.py`
- Create: `benchmarks/kamino_dvi/tests/fixtures/logs/*.txt`
- Create: `benchmarks/kamino_dvi/tests/test_failures.py`
- Create: `benchmarks/kamino_dvi/tests/test_scheduler.py`

1. Add synthetic logs for CUDA OOM, allocation failure, contact-capacity exhaustion, non-finite values, explicit divergence, timeout, generic crash, zero-exit incomplete training, and missing artifacts.
2. Write tests for precedence and required diagnostics: category, return code/signal, parsed exception, completed iterations, last 200 log lines, and retry lineage.
3. Write scheduler tests proving only `capacity` lowers the common task count, all task variants repeat preflight after lowering, full-run capacity invalidates the entire task/count comparison, and the ladder stops cleanly at 128. Numerical failures and timeouts stay at the selected count.
4. Run tests and observe RED.
5. Implement deterministic classifiers and a pure scheduler state machine.
6. Rerun focused tests and full pre-commit.
7. Commit: `git commit -m "Handle benchmark failures and capacity fallback"`.

## Task 7: Implement the single-GPU subprocess runner

**Files:**

- Create: `benchmarks/kamino_dvi/run.py`
- Create: `benchmarks/kamino_dvi/gpu.py`
- Create: `benchmarks/kamino_dvi/tests/test_run.py`
- Create: `benchmarks/kamino_dvi/tests/test_gpu.py`

1. Write tests around a fake `subprocess.Popen` for sequential execution, streamed stdout/stderr files, process-group timeout termination, continue-after-failure, atomic state updates, and `--resume`, `--preflight-only`, `--full-only`, `--dry-run`, task, variant, and seed filters.
2. Write `nvidia-smi` parser tests for visible processes, free/used/total memory, utilization, temperature, and clocks. Missing `nvidia-smi` must fail preflight with a clear message rather than fabricate values.
3. Run focused tests and observe RED.
4. Implement the CLI and runner. Use one process at a time, a new process group, periodic health metadata, explicit 30-minute/4-hour timeouts, and manifests at every state transition. Preserve all failed output.
5. Rerun focused tests, then all `benchmarks/kamino_dvi/tests` and full pre-commit.
6. Commit: `git commit -m "Add Kamino DVI benchmark runner"`.

## Task 8: Parse schema bundles and TensorBoard traces

**Files:**

- Create: `benchmarks/kamino_dvi/parsing.py`
- Create: `benchmarks/kamino_dvi/tests/fixtures/training_v1_1.json`
- Create: `benchmarks/kamino_dvi/tests/fixtures/tensorboard/`
- Create: `benchmarks/kamino_dvi/tests/test_parsing.py`

1. Generate a tiny TensorBoard fixture with `Perf/collection_time`, `Perf/learning_time`, `Perf/total_fps`, reward, episode length, and success values across 300 numbered iterations. Add schema v1.1 fixtures with and without `learning.success_rate`.
2. Test strict schema version/revision/run-identity validation, exact 300-iteration alignment, derived collection FPS, total FPS, iteration time, reward, episode length, success, startup, and resource extraction.
3. Test that missing success yields `N/A`, but missing reward/episode length/runtime tags raises a `MissingBenchmarkFieldError` identifying the benchmark-stack bug.
4. Run tests and observe RED.
5. Implement schema JSON parsing and TensorBoard `EventAccumulator` parsing. Match values by step; never zip unequal series silently. Treat the schema learning curves as authoritative and TensorBoard as authoritative only for runtime series the schema aggregates.
6. Rerun focused tests and full pre-commit.
7. Commit: `git commit -m "Parse benchmark traces and learning series"`.

## Task 9: Implement seed-level statistics and quality warnings

**Files:**

- Create: `benchmarks/kamino_dvi/statistics.py`
- Create: `benchmarks/kamino_dvi/tests/test_statistics.py`

1. Write exact-value tests for exclusion of iterations 1–10, final-20 learning means, ten-iteration rolling means, five-seed arithmetic means, and two-sided 95% intervals using `t_0.975,4 = 2.7764451051977987`.
2. Test seed-paired throughput ratios and absolute differences, mismatched seed rejection, and exclusion of comparison bars unless all five successful seeds exist.
3. Test pointwise learning intervals, `N/A` success propagation, and completed-run `quality_warning` without changing terminal status.
4. Run tests and observe RED.
5. Implement standard-library statistics with explicit sample standard deviation and typed summary records.
6. Rerun focused tests and full pre-commit.
7. Commit: `git commit -m "Add benchmark statistical analysis"`.

## Task 10: Generate figures and compact results

**Files:**

- Create: `benchmarks/kamino_dvi/analyze.py`
- Create: `benchmarks/kamino_dvi/plotting.py`
- Create: `benchmarks/kamino_dvi/tests/test_analysis.py`
- Create: `benchmarks/kamino_dvi/tests/test_plotting.py`
- Create: `benchmarks/kamino_dvi/results/.gitkeep`
- Create: `benchmarks/kamino_dvi/figures/.gitkeep`

1. Write fixture-driven tests that generate `summary.csv`, `summary.json`, `failures.md`, and coverage/provenance tables with stable ordering and explicit actual environment counts.
2. Write plot smoke/metadata tests for Collection FPS bars, paired speedups, Total FPS, peak memory, three learning-curve families, stability heatmap, and coverage view. Require both PNG and SVG, non-empty files, labels, units, and 95% interval legends.
3. Test incomplete cells appear in stability/coverage outputs but not complete-comparison bars.
4. Run tests and observe RED.
5. Implement a headless matplotlib `Agg` pipeline and deterministic analysis CLI. Close every figure and embed generation metadata in compact JSON.
6. Rerun focused tests and full pre-commit.
7. Commit: `git commit -m "Generate benchmark summaries and figures"`.

## Task 11: Generate Markdown, HTML, and PDF reports

**Files:**

- Create: `benchmarks/kamino_dvi/report.py`
- Create: `benchmarks/kamino_dvi/templates/report.html`
- Create: `benchmarks/kamino_dvi/tests/test_report.py`
- Create: `benchmarks/kamino_dvi/report.md`
- Create: `benchmarks/kamino_dvi/report.pdf`

1. Write fixture-based report tests for the executive summary, coverage, runtime tables, final reward/success/episode length, failure appendix, capacity fallback, hardware/software revisions, methods, and embedded figure links.
2. Test Markdown-to-HTML using the installed `markdown` package. Test LibreOffice command construction separately; then run one integration conversion and assert the PDF exists, starts with `%PDF`, and is non-trivial in size.
3. Run tests and observe RED.
4. Implement deterministic Markdown assembly, styled standalone HTML, and headless LibreOffice conversion in a temporary profile directory. Surface conversion stderr on failure.
5. Generate a complete fixture report before any long training run and visually inspect representative PNGs plus the rendered PDF.
6. Rerun focused tests and full pre-commit.
7. Commit: `git commit -m "Add Kamino DVI benchmark reporting"`.

## Task 12: Document setup, raw artifact policy, and exact commands

**Files:**

- Create: `benchmarks/kamino_dvi/README.md`
- Modify: `.gitignore`

1. Add a documentation test or command snapshot asserting the README contains immutable revisions, two-environment setup, `uv pip install --no-deps`, dry-run, preflight, resume, analysis, report, and raw-artifact locations.
2. Add only `/benchmark_artifacts/kamino_dvi/` to `.gitignore`; do not ignore tracked compact results or figures.
3. Document creation of `.venv-current` from the lockfile, cloning it to `.venv-pr3570`, replacing Newton only in the candidate, and verifying both revisions before use.
4. Document all 21 cells, 105 runs, fallback semantics, timeouts, statistics, and failure policy.
5. Run all benchmark-tool tests and full pre-commit.
6. Commit: `git commit -m "Document Kamino DVI benchmark workflow"`.

## Task 13: Verify tooling and create the two locked environments

**Files:**

- Modify only if fixture verification exposes a tested defect.
- Produce untracked environment/provenance manifests under `benchmark_artifacts/kamino_dvi/`.

1. Run all focused suites:
   `./isaaclab.sh -p -m pytest benchmarks/kamino_dvi/tests source/isaaclab_newton/test/physics/test_kamino_manager_cfg.py source/isaaclab_tasks/test/core/test_anymal_d_physics_presets.py scripts/benchmarks/test -q`.
2. Run `./isaaclab.sh -d` and full pre-commit. Confirm `git diff --check` and inspect the complete branch diff.
3. Create `.venv-current` from the frozen lockfile and validate Newton `c7ae7c7648cd0717df39e5c94b95d5a02c997320`.
4. Clone the environment to `.venv-pr3570`, install PR head `7906676b2e5061273db96af179d7081fc6cbbba0` with `uv pip install --no-deps`, and prove every non-Newton distribution version matches the baseline.
5. Run construction tests in both environments: current omits `dynamics_solver`; PR accepts both default P-ADMM and `"dvi"`.
6. Generate and inspect the fixture Markdown/PDF report again in the actual execution environment.
7. Commit only tested corrections, each as a focused conventional commit.

## Task 14: Run functional smokes and select common environment counts

**Files:**

- Produce raw artifacts under `benchmark_artifacts/kamino_dvi/`.
- Later update: `benchmarks/kamino_dvi/results/failures.md` and report provenance.

1. Run `run.py --dry-run` and manually verify the exact commands/order for all 21 cells.
2. Run a small one-iteration functional smoke for every task/variant combination to catch construction or compatibility errors before timed preflight. Record any missing requested field as a benchmark-stack bug immediately.
3. Run the five-iteration seed-42 preflight at 4096 for every task/variant. Apply the approved common-count ladder only for classified capacity failures, rerunning every applicable variant for that task at each lower count.
4. Freeze and record the selected common count for every task. Do not begin full runs until every task has either a common count or a documented irrecoverable preflight failure.
5. Review GPU process, memory, utilization, temperature, clocks, log tails, and manifest completeness.
6. If code changes are needed, reproduce with a failing test, implement the minimal fix, rerun all focused tests and full pre-commit, and commit before resuming.

## Task 15: Execute the 300-iteration, five-seed matrix

**Files:**

- Produce raw artifacts under `benchmark_artifacts/kamino_dvi/`.

1. Execute all full cells sequentially with seeds 42–46 and the selected per-task common environment counts. Use deterministic counterbalanced variant order, no early stop, and four-hour per-run timeouts.
2. Check manifests and artifact hashes after every run. Continue after failures; preserve stdout, stderr, TensorBoard events, schema bundle, checkpoint, resource samples, and failure record.
3. On a full-run capacity failure, invalidate comparisons at that task/count, lower one ladder step, rerun all task preflights, and rerun all full cells for the task. Preserve the earlier failed artifacts and retry lineage.
4. Do not lower counts for numerical instability, poor reward, timeout, crash, incomplete training, or artifact bugs.
5. Resume until every requested run is terminal and no valid matrix cell is accidentally missing.

## Task 16: Analyze results and publish the final report

**Files:**

- Populate: `benchmarks/kamino_dvi/results/summary.csv`
- Populate: `benchmarks/kamino_dvi/results/summary.json`
- Populate: `benchmarks/kamino_dvi/results/failures.md`
- Populate: `benchmarks/kamino_dvi/figures/*.png`
- Populate: `benchmarks/kamino_dvi/figures/*.svg`
- Populate: `benchmarks/kamino_dvi/report.md`
- Populate: `benchmarks/kamino_dvi/report.pdf`

1. Validate every raw artifact against its manifest, revision, schema version, task, seed, variant, iteration count, and environment count.
2. Run `analyze.py`; inspect five-seed coverage, excluded incomplete cells, paired seed alignment, first-ten exclusion, final-20 summaries, 95% intervals, capacity fallbacks, and quality warnings.
3. Inspect every PNG/SVG for clipped labels, misleading axes, missing units, and readable intervals. Keep missing success as `N/A`.
4. Generate Markdown, HTML, and PDF. Visually inspect the PDF and compare headline numbers against `summary.csv`/`summary.json`.
5. Run the complete focused pytest suite, `./isaaclab.sh -d`, full pre-commit, `git diff --check`, and inspect `git status --short` for untracked raw artifacts.
6. Commit tracked results and report: `git commit -m "Report Kamino DVI benchmark results"`.

## Task 17: Independent review and branch handoff

1. Use `superpowers:requesting-code-review` for an independent spec, code, statistical-method, and report review. Require reviewers to run `./isaaclab.sh -f` as mandated by `AGENTS.md`.
2. Address every finding with new focused commits; do not amend review-history commits.
3. Use `superpowers:verification-before-completion` and rerun all final verification commands from fresh state.
4. Use `superpowers:finishing-a-development-branch` to present merge/PR/keep/cleanup choices.
5. If the user chooses a PR, push only to the `antoine` fork and provide the compare link if GitHub API credentials remain unavailable.
