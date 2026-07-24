<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Expanded Benchmark Task List Design

**Status:** Approved

## Context

The IsaacLab 2.x-versus-3.0 comparison harness currently benchmarks six
logical tasks across two runtime modes and one RSL-RL training mode. The next
comparison should cover additional direct and manager-based workflows, a
camera workload using the Kit renderer, another humanoid platform, and both
flat and rough Anymal-D terrain cases.

This change prepares and validates the expanded task matrix. It does not run
the canary or final benchmarks and does not change the collected metrics,
report format, seeds, environment count, or counterbalancing policy.

## Goals

- Add direct Cartpole, Ant, and Humanoid tasks.
- Add the manager-based Humanoid task.
- Add flat-terrain Cassie.
- Preserve flat-terrain Anymal-D and add rough-terrain Anymal-D as a
  complementary workload.
- Add manager-based RGB Cartpole with camera rendering enabled in both
  versions and the Kit renderer selected in IsaacLab 3.0.
- Restrict RGB Cartpole to the two runtime modes because its IsaacLab 2.x
  registration has no RSL-RL configuration.
- Keep every other task in both runtime modes and the RSL-RL training mode.
- Derive and validate the new deterministic matrix counts.

## Non-goals

- Run smoke, canary, or final benchmark attempts.
- Change the existing 4,096-environment setting.
- Add per-task environment-count overrides.
- Backport an RSL-RL agent configuration for IsaacLab 2.x RGB Cartpole.
- Add skipped or expected-failure training attempts for RGB Cartpole.
- Change metric collection, normalization, plotting, or report semantics.
- Add Cassie rough terrain or replace either Anymal-D terrain workload.

## Matrix Model

The checked-in TOML remains the source of truth. Each task may declare an
optional ordered list of supported mode identifiers. Omitting the list means
the task uses every globally configured mode, preserving the existing default
for non-camera tasks.

RGB Cartpole declares only `runtime-100` and `runtime-1000`. Matrix expansion
iterates the configured modes in their existing global order and includes a
task-mode pair only when the task supports that mode. Unsupported pairs are
absent from the expansion; they are not emitted as skips or failures.

The task model also carries the minimum execution metadata required by the
camera workload:

- a camera-enablement flag, translated to `--enable_cameras` for both
  executors; and
- version-specific preset additions, used for IsaacLab 3.0 RGB selection.

IsaacLab 3.0 RGB Cartpole combines the existing PhysX selection with the RGB
camera preset and leaves the camera renderer on the Kit-backed default. The
IsaacLab 2.x task identifier already selects RGB observations, so it only
needs camera enablement in addition to the existing PhysX execution path.

Non-camera tasks do not declare camera or extra-preset metadata and retain
their current command lines.

## Task Mappings

The complete matrix contains these 13 logical tasks:

| Matrix alias | IsaacLab 2.x task | IsaacLab 3.0 task | Supported modes |
|---|---|---|---|
| `cartpole` | `Isaac-Cartpole-v0` | `Isaac-Cartpole` | All |
| `cartpole_rgb_kit` | `Isaac-Cartpole-RGB-v0` | `Isaac-Cartpole-Camera` | Runtime only |
| `cartpole_direct` | `Isaac-Cartpole-Direct-v0` | `Isaac-Cartpole-Direct` | All |
| `ant` | `Isaac-Ant-v0` | `Isaac-Ant` | All |
| `ant_direct` | `Isaac-Ant-Direct-v0` | `Isaac-Ant-Direct` | All |
| `humanoid_manager` | `Isaac-Humanoid-v0` | `Isaac-Humanoid` | All |
| `humanoid_direct` | `Isaac-Humanoid-Direct-v0` | `Isaac-Humanoid-Direct` | All |
| `anymal_d_flat` | `Isaac-Velocity-Flat-Anymal-D-v0` | `Isaac-Velocity-Flat-AnymalD` | All |
| `anymal_d_rough` | `Isaac-Velocity-Rough-Anymal-D-v0` | `Isaac-Velocity-Rough-AnymalD` | All |
| `g1_flat` | `Isaac-Velocity-Flat-G1-v0` | `Isaac-Velocity-Flat-G1` | All |
| `cassie_flat` | `Isaac-Velocity-Flat-Cassie-v0` | `Isaac-Velocity-Flat-Cassie` | All |
| `allegro_cube` | `Isaac-Repose-Cube-Allegro-v0` | `Isaac-Reorient-Cube-Allegro` | All |
| `franka_reach` | `Isaac-Reach-Franka-v0` | `Isaac-Reach-Franka` | All |

“All” means `runtime-100`, `runtime-1000`, and `training-100`. “Runtime only”
means `runtime-100` and `runtime-1000`.

## Expansion Counts

Twelve tasks use all three modes and RGB Cartpole uses two modes:

```text
(12 tasks × 3 modes) + (1 task × 2 modes) = 38 task-mode cells
```

The final run set retains three paired seeds and two version attempts per
logical pair:

```text
38 cells × 3 seeds = 114 logical pairs
114 pairs × 2 versions = 228 attempts
```

The canary retains seed 42 only:

```text
38 cells × 1 seed = 38 logical pairs
38 pairs × 2 versions = 76 attempts
```

The existing version order remains unchanged for each included task-mode
cell: Lab 2 then Lab 3 for seeds 42 and 44, and Lab 3 then Lab 2 for seed 43.

## Preflight and Validation

Matrix validation rejects:

- unknown or duplicate task mode identifiers;
- a task with no supported modes;
- duplicate aliases or concrete task identifiers;
- unexpected task mappings, mode definitions, seeds, or environment count;
  and
- expansions whose derived pair or attempt counts differ from the expected
  114/228 final and 38/76 canary counts.

Registration preflight resolves every configured task and its environment
configuration in both versions. It resolves `rsl_rl_cfg_entry_point` only for
tasks that support `training-100`. This validates normal training tasks while
allowing the runtime-only IsaacLab 2.x RGB registration.

The RGB preflight launches with cameras enabled and resolves the same
version-specific camera settings used by benchmark attempts. It verifies that
the selected configuration is a camera task using PhysX and the intended
IsaacLab 3.0 RGB/Kit path.

## Reporting Order

The canonical task order expands to match the table above. Normalized CSV
rows, report tables, and plots use that order. Because reports derive their
task-mode rows from successful attempts, RGB Cartpole naturally appears only
for `runtime-100` and `runtime-1000`; report generation does not manufacture a
training row.

The plotting code must tolerate the sparse RGB task-mode combination while
keeping axes deterministic. No historical artifact is rewritten.

## Testing

Controller tests cover:

- parsing tasks that inherit all modes and tasks with explicit mode subsets;
- rejection of empty, duplicate, and unknown task mode lists;
- the exact 13 task mappings and canonical ordering;
- complementary flat and rough Anymal-D entries;
- runtime-only RGB expansion and absence of RGB training attempts;
- exact final and canary pair and attempt counts;
- camera flags and IsaacLab 3.0 RGB/Kit preset command construction;
- preflight behavior that requires RSL-RL only for training-capable tasks;
- deterministic ordering and identities for the sparse matrix; and
- normalization, report, and plotting behavior with RGB runtime rows and no
  RGB training row.

The implementation runs the focused controller test suite followed by the
repository-wide pre-commit hooks. Simulator benchmark execution remains out
of scope for this task-list-only change.
