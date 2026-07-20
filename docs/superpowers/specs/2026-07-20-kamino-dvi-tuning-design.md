# Kamino DVI Ant Tuning Design

## Objective

Tune the DVI option implemented inside Newton PR 3570's Kamino solver on
`Isaac-Ant-Direct` at 4096 environments. Milad's standalone DVI result is a
performance reference only; it is not an additional backend in the benchmark
matrix. The selected configuration must improve runtime without hiding unstable
reward, episode-length, or task-defined success behavior.

## Configuration Surface

Extend `KaminoSolverCfg` with the PR's missing constrained-dynamics and DVI
settings. Keep the existing flat, prefix-grouped API:

- `dynamics_linear_solver_type` and `dynamics_linear_solver_max_iterations`;
- the `dvi_`-prefixed iteration, tolerance, relaxation, preconditioner, and
  warm-start fields represented by `DVISolverConfig`.

`to_solver_config()` will construct `ConstrainedDynamicsConfig` and
`DVISolverConfig` from these values. Existing P-ADMM behavior remains unchanged,
and the optional DVI imports remain inside `to_solver_config()` so the current
pinned Newton remains importable when DVI is not selected.

Add `newton_kamino_dvi` to Ant's `AntPhysicsCfg`. It will copy the task's existing
Kamino contact and stabilization choices while selecting DVI, Moreau integration,
sparse Jacobian and dynamics, disabled dynamics preconditioning, and CR with nine
iterations. This matches PR 3570's intended sparse-DVI defaults and gives Hydra a
named, reproducible starting point. The existing `newton_kamino` preset remains
the P-ADMM control.

## Evaluation

Use seed 42 and 20 training iterations for one-variable screening. The first
candidate is the named sparse-DVI preset. Subsequent candidates may change one
DVI field at a time, prioritizing the block/contact iteration counts and
integrator only after the sparse CR/9 baseline is measured.

For every candidate, retain the schema trace and summarize steady iteration time,
collection FPS, reward, episode length, and success. Reject candidates that
crash, emit non-finite values, or show materially worse short-run learning
behavior. Run the best candidate for 300 iterations with seeds 42, 43, and 44.
Compare it with the existing three-seed current-Kamino, PR-DVI, MJWarp, and PhysX
Ant results. Report Milad's preflight measurement separately as a non-equivalent
reference.

## Testing and Delivery

Write failing tests before implementation for:

- forwarding CR and its iteration budget into the constrained-dynamics config;
- forwarding representative DVI settings into `DVISolverConfig`;
- the Ant `newton_kamino_dvi` preset's effective solver settings;
- unchanged construction of the existing P-ADMM preset.

Run focused tests after each change, then the relevant package tests and full
pre-commit hooks before committing. Update the `isaaclab_newton` public solver
documentation and changelog fragment because the solver configuration API grows.
Update the benchmark report with the tuning sweep, three-seed finalist, confidence
intervals, and explicit stability findings.
