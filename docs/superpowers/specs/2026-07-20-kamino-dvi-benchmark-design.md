# Kamino DVI Training Benchmark Design

## Objective

Measure whether the experimental DVI forward-dynamics solver in Newton PR 3570 improves Isaac Lab training runtime relative to current Kamino P-ADMM, while retaining useful training behavior. Compare the common tasks with MJWarp and PhysX, and evaluate the Kamino-only closed-loop tasks against both P-ADMM controls.

The deliverable is a reproducible experiment bundle plus Markdown and PDF reports that let a reviewer quickly assess throughput, speedup, resource use, learning behavior, and stability.

## Revisions and Isolation

Use the isolated Isaac Lab branch `antoiner/kamino-dvi-benchmark`, based on:

- Isaac Lab `origin/develop`: `79accca281128660a786abb599f40bd335963963`.
- The verified benchmark schema v1.1 prerequisite ending at `47d325124080d36a270daafe3d20e2a3d11f280b`.
- Current Newton baseline: Isaac Lab's pinned commit `c7ae7c7648cd0717df39e5c94b95d5a02c997320`.
- Candidate Newton PR 3570 head: `7906676b2e5061273db96af179d7081fc6cbbba0` from `vastsoun/newton` branch `dev/dvi_solver`, captured on 2026-07-20.

Create two worktree-local Python 3.12 environments from the same frozen Isaac Lab dependency graph. Replace Newton in the candidate with `uv pip install --no-deps` so no other dependency can drift:

- `.venv-current`: the current Newton pin.
- `.venv-pr3570`: the same environment with only Newton replaced by the immutable PR head.

Never update either Newton revision after execution begins. Record installed package versions and the importable Newton source revision in every run manifest. PhysX, MJWarp, and current Kamino use `.venv-current`; PR P-ADMM and PR DVI use `.venv-pr3570`.

## Isaac Lab Compatibility Bridge

Add an optional `dynamics_solver: str | None = None` field to `KaminoSolverCfg`. When the field is `None`, omit the keyword when constructing `SolverKamino.Config`, keeping the current pinned Newton fully compatible. When set, pass the keyword to PR 3570. Use `"dvi"` only for the DVI variant; leaving it unset selects the PR's default P-ADMM control.

The field is benchmark-enabling public configuration and therefore requires a Google-style docstring, focused compatibility tests, an update to the Kamino solver documentation, and an `isaaclab_newton` minor changelog fragment. It does not rename or remove an existing API.

Latest `develop` does not expose a Kamino preset for ANYmal-D Flat. Add this benchmark-local preset to its `PhysicsCfg`:

```python
newton_kamino = NewtonCfg(
    solver_cfg=KaminoSolverCfg(max_contacts_per_world=64),
    num_substeps=1,
    debug_mode=False,
)
```

Use the same preset for current Kamino, PR P-ADMM, and PR DVI. Do not tune solver parameters separately by variant. Document the preset as benchmark-local rather than claiming general task support.

## Benchmark Matrix

The exact task identifiers are:

- `Isaac-Cartpole-Direct`
- `Isaac-Ant-Direct`
- `Isaac-Velocity-Flat-AnymalD`
- `Isaac-DrLegs-Walk-v0`
- `Isaac-Fourbar-Pole-Swingup`

The exact variants are:

- `kamino_current`: current Newton, `presets=newton_kamino`, P-ADMM.
- `kamino_pr_padmm`: PR Newton, `presets=newton_kamino`, PR-default P-ADMM.
- `kamino_pr_dvi`: PR Newton, `presets=newton_kamino`, `dynamics_solver="dvi"`.
- `mjwarp`: current Newton, `presets=newton_mjwarp`.
- `physx`: current environment, `presets=physx`.

Run all five variants on Cartpole, Ant, and ANYmal-D. Run only the three Kamino variants on DR Legs Walk and Four-bar Pole Swingup.

Each full cell uses:

- RSL-RL.
- 300 training iterations (`--max_iterations 300`).
- Seeds 42, 43, 44, 45, and 46.
- Headless execution with no video and no early stopping.
- Full benchmark schema series retained.

This is 105 requested full runs before any capacity retries.

## Environment Count and Preflight

Start every task at 4096 environments. Use the fallback ladder `4096, 2048, 1024, 512, 256, 128`.

For each task, run every applicable variant for five RSL-RL iterations at seed 42 before the full matrix. A capacity failure, including CUDA out-of-memory, allocation failure, or solver contact-capacity exhaustion, lowers the task to the next count. Repeat all task preflights at the new count. The first count where every applicable variant completes is the task's comparison count.

Numerical instability, non-finite values, solver divergence, or poor reward does not trigger count reduction. Record it at the selected count. If a capacity failure first appears in a full run, retain the failed artifacts for diagnosis, exclude all results at that count from comparisons, lower the task one step, preflight every variant again, and rerun every full task cell at the new common count.

Use a 30-minute preflight timeout and a four-hour full-run timeout. Timeout is a recorded execution failure, not an environment-count fallback unless logs identify a capacity cause.

## Execution Ordering

Use one GPU and one training process at a time. Before each run, record visible GPU processes, free memory, utilization, temperature, and clock state. Do not run variants concurrently.

For each task and seed, rotate the variant order by the seed index. Reverse the base order on alternating tasks. This deterministic counterbalancing distributes compilation-cache, thermal, and time-of-day effects without making the run plan irreproducibly random.

Smoke runs compile kernels before full runs. Startup and compilation metrics remain reported separately, but the primary training runtime analysis excludes the first ten full-run iterations.

## Runner and Artifacts

Create a dedicated `benchmarks/kamino_dvi/` experiment package containing:

- `matrix.yaml`: revisions, tasks, variants, seeds, counts, timeouts, and ordering.
- `run.py`: matrix expansion, environment selection, subprocess execution, retry policy, and manifest updates.
- `analyze.py`: trace parsing, validation, statistics, tables, and figures.
- `report.py`: Markdown, styled HTML, and PDF assembly.
- `models.py`: typed run, failure, and summary records.
- `tests/`: synthetic fixtures and unit tests.
- `README.md`: exact setup, resume, run, analysis, and report commands.
- `results/summary.csv`, `results/summary.json`, and `results/failures.md`: compact tracked outputs.
- `figures/`: tracked PNG and SVG visualizations.
- `report.md` and `report.pdf`: final tracked reports.

Place large raw output beneath `benchmark_artifacts/kamino_dvi/`, excluded from git:

- one directory per run identity;
- canonical schema bundle;
- TensorBoard event file;
- stdout and stderr;
- command and environment manifest;
- checkpoint when emitted;
- failure record when applicable.

The tracked summary manifest records the relative raw path, SHA-256 hashes, command, environment label, task, variant, seed, requested and actual environment counts, timestamps, exit status, completed iterations, package versions, git revisions, and hardware.

Write manifests atomically after each state transition so an interrupted experiment resumes without repeating valid completed runs. A `--resume` run skips only artifacts whose command hash, revisions, schema version, and terminal success state match the current request.

## Failure Classification

The runner continues after an individual failure and records one primary category:

- `capacity`: OOM, allocation, or contact-capacity exhaustion.
- `timeout`: subprocess exceeded its configured timeout.
- `numerical`: non-finite values, explicit solver divergence, or CUDA numerical failure.
- `crash`: non-zero exit not classified above.
- `incomplete`: fewer than 300 full iterations despite zero exit.
- `artifact`: missing or unreadable benchmark/TensorBoard output.

Store the return code, signal, last 200 stdout/stderr lines, parsed exception type, completed iterations, and retry lineage. Never substitute a failed value with zero.

A completed variant may separately receive a `quality_warning` when its five-seed final-20-iteration reward confidence interval lies wholly below the appropriate reference interval. Use current Kamino as the Kamino comparison reference and the best available established backend (MJWarp or PhysX) as contextual reference on common tasks. Report the exact values; keep the run status completed and do not convert this warning into an execution failure category.

## Metrics and Statistical Analysis

The primary metric is steady-state Collection FPS. It represents environment rollout throughput including physics, task logic, and inference while excluding policy learning.

Secondary runtime and resource metrics are:

- Total FPS.
- Iteration time [s].
- Training wall time [s].
- Startup phase times [s].
- Peak GPU memory [GB].
- Mean GPU utilization [%].

Read per-iteration runtime values from TensorBoard traces. Exclude iterations 1 through 10 for steady-state summaries and use iterations 11 through 300. Treat each seed-level steady-state mean as one independent observation.

For each task and variant, report the arithmetic mean over five seeds and a two-sided 95% Student-t confidence interval with four degrees of freedom. Do not treat iterations as independent replicates.

For PR P-ADMM and PR DVI speedups against current Kamino, pair equal seeds, compute each seed's throughput ratio, and report the mean ratio and 95% Student-t interval. Also report absolute paired differences.

For learning outcomes:

- Plot reward, success rate, and episode length over all 300 iterations.
- Apply a ten-iteration rolling mean within each seed for visualization.
- Plot the cross-seed mean with pointwise 95% Student-t intervals.
- Summarize each seed by its final 20-iteration mean, then aggregate across seeds.
- Report a missing task-defined success metric as `N/A`, never zero.

The analysis validates exactly five successful seeds before producing a complete comparison bar. Incomplete cells appear in the stability view and tables but not as misleading partial-confidence bars.

## Figures and Report

The report includes:

- Executive summary with headline DVI speedups and stability outcome.
- Task/variant coverage table.
- Collection-FPS grouped bars with 95% intervals.
- Seed-paired P-ADMM and DVI speedup bars.
- Total-FPS and peak-memory comparisons.
- Reward, success-rate, and episode-length learning curves.
- Stability heatmap by task, variant, and seed.
- Summary runtime and final-learning tables.
- Capacity fallback and failure appendix.
- Full hardware, software, commit, command, and statistical-method provenance.

Generate PNG at publication resolution and matching SVG files. `report.md` is the source of truth. Render a styled HTML intermediate using the existing Python Markdown package, then convert HTML to PDF with headless LibreOffice. Validate this conversion using synthetic fixture data before executing the full matrix.

## Testing and Validation

Use test-driven development for benchmark tooling. Unit tests cover:

- exact 21-cell matrix expansion and 105 full-run identities;
- deterministic counterbalanced ordering;
- environment and solver command construction;
- current/PR Newton revision validation;
- atomic manifest writes and resume matching;
- capacity fallback and retry lineage;
- failure classification from synthetic logs;
- TensorBoard and schema parsing;
- first-ten-iteration exclusion;
- Student-t intervals and seed-paired speedups;
- incomplete-cell exclusion;
- Markdown, figures, HTML, and PDF generation from fixtures.

Before long execution:

1. Run focused unit tests and all pre-commit hooks.
2. Validate current Newton omits `dynamics_solver` and remains functional.
3. Validate PR P-ADMM and DVI construct successfully.
4. Run one small functional training smoke for every task/variant combination.
5. Generate a complete fixture-based Markdown and PDF report.

Any missing requested benchmark field is reported as a benchmark-stack bug before adding a workaround. Raw TensorBoard parsing is allowed for existing per-iteration data that the canonical schema intentionally aggregates; it is not used to conceal absent data.

## Delivery Boundaries

This branch produces benchmark tooling and report artifacts. It does not merge Newton PR 3570, alter its solver defaults, tune DVI independently, or claim general ANYmal-D Kamino support. The schema v1.1 prerequisite remains a separate PR. Preserve both worktrees for review and follow-up changes.
