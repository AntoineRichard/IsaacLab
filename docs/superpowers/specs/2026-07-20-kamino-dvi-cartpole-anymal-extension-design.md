# Cartpole and ANYmal-D Kamino DVI Benchmark Extension

## Objective

Extend the tuned Kamino DVI evaluation from Ant to Cartpole and ANYmal-D Flat. Run fresh Cartpole and ANYmal-D data under the same RSL-RL protocol, retain the already validated Ant traces, and regenerate one combined report that makes runtime, learning quality, capacity, and stability trade-offs easy to assess.

## Scope

The comparison covers these tasks:

- `Isaac-Cartpole-Direct`
- `Isaac-Ant-Direct`
- `Isaac-Velocity-Flat-AnymalD`

Each task compares current Kamino, PR 3570 P-ADMM, tuned PR 3570 DVI, MJWarp, and PhysX. Full runs use 300 training iterations and seeds 42–44. Every task starts at 4096 environments. A task may use a lower common count only after an explicit capacity failure, and the report must identify the fallback.

Ant is not rerun because its 15 agreed traces already passed completion, length, finiteness, and schema-field validation. Cartpole is rerun from scratch so its results use the final tuned DVI configuration rather than the earlier pre-tuning candidate.

## Physics presets

Expose `newton_kamino_dvi` as a named task-local physics preset for Cartpole Direct and ANYmal-D Flat.

Cartpole preserves its existing Kamino collision pipeline, collision limits, CUDA-graph selection, and task-specific constraint settings. Its DVI preset uses Moreau integration, sparse Jacobian and dynamics paths, no preconditioning, a CR dynamics solve capped at 9 iterations, and the accepted DVI settings: omega 0.3, 16 block iterations, 2 contact iterations, bilateral update period 2, contact omega 0.45, and contact relaxation 0.9.

ANYmal-D preserves `max_contacts_per_world=64`, one substep, and disabled debug mode. Its DVI preset otherwise uses the same accepted sparse/Moreau/CR/DVI settings as Ant.

The presets are explicit copies instead of a new shared factory. This keeps each task's physical limits visible and avoids introducing an abstraction for only three task configurations.

## Execution

First run a five-iteration tuned-DVI preflight for Cartpole and ANYmal-D at 4096 environments. If both construct and train, run the remaining four preflights for each task at the same count. A capacity-classified failure restarts all variants for that task at the next count in the declared ladder. Numerical divergence, non-finite metrics, missing benchmark fields, or unrelated crashes do not trigger capacity fallback.

After successful preflights, run the five variants sequentially on one GPU for seeds 42–44 and 300 iterations. The runner continues to isolate `PYTHONPATH` and validate the exact current Newton, PR 3570, IsaacLab, and schema revisions before launching jobs. Atomic manifests and `--resume` keep every identity restartable.

## Validation and reporting

For every completed run, require exactly 300 finite samples for iteration time, reward, episode length, and success rate. A missing learning series is a benchmark-schema bug and stops aggregation for that task/variant. Compare the schema success series with its matching TensorBoard series and use TensorBoard success while the known PR 6624 generator issue remains in schema v1.1.

Regenerate the Markdown report, PDF report, compact JSON summary, runtime figure, and three-panel learning figure. Tables and figures use per-task three-seed means with two-sided 95% Student-t confidence intervals. Runtime excludes iterations 1–10; learning metrics average the final 20 iterations. The report calls out environment-count fallbacks, failed runs, schema mismatches, and wide success-rate intervals.

## Verification

Tests must cover both new task-local presets, the 15-cell/45-run three-task matrix, command construction, plot generation, and report quality warnings. Before committing, run the complete focused benchmark, Kamino configuration, Cartpole preset, and ANYmal-D preset tests, followed by the repository pre-commit hooks.
