# Core Task Benchmark Expansion Design

## Goal

Extend the Isaac Lab 2.3.2 versus Isaac Lab 3.0 comparison from 23 to 39
logical tasks by adding every canonical, non-Play Isaac Lab 3 Core task with a
clear Isaac Lab 2 counterpart. Execute only the new attempts, preserve the
existing 408-run artifact set unchanged, and fold the runtime-regression
diagnosis into the main Markdown and PDF reports.

## Scope

Add these 16 matched task pairs:

| Alias | Category | Isaac Lab 2 | Isaac Lab 3 | Modes | Environments |
|---|---|---|---|---|---:|
| `cartpole_camera_direct` | Classic | `Isaac-Cartpole-RGB-Camera-Direct-v0` | `Isaac-Cartpole-Camera-Direct` | Runtime only | 512 |
| `pendulum_direct` | Classic | `Isaac-Cart-Double-Pendulum-Direct-v0` | `Isaac-Pendulum-Direct` | Runtime only | 4096 |
| `h1_flat` | Locomotion Flat | `Isaac-Velocity-Flat-H1-v0` | `Isaac-Velocity-Flat-H1` | All | 4096 |
| `spot_flat` | Locomotion Flat | `Isaac-Velocity-Flat-Spot-v0` | `Isaac-Velocity-Flat-Spot` | All | 4096 |
| `cassie_rough` | Locomotion Rough | `Isaac-Velocity-Rough-Cassie-v0` | `Isaac-Velocity-Rough-Cassie` | All | 4096 |
| `h1_rough` | Locomotion Rough | `Isaac-Velocity-Rough-H1-v0` | `Isaac-Velocity-Rough-H1` | All | 4096 |
| `franka_lift_cube` | Manipulation | `Isaac-Lift-Cube-Franka-v0` | `Isaac-Lift-Cube-Franka` | All | 4096 |
| `franka_drawer` | Manipulation | `Isaac-Open-Drawer-Franka-v0` | `Isaac-Open-Drawer-Franka` | All | 4096 |
| `franka_reach_osc` | Manipulation | `Isaac-Reach-Franka-OSC-v0` | `Isaac-Reach-Franka-OSC` | All | 4096 |
| `ur10_reach` | Manipulation | `Isaac-Reach-UR10-v0` | `Isaac-Reach-UR10` | All | 4096 |
| `allegro_cube_direct` | Manipulation | `Isaac-Repose-Cube-Allegro-Direct-v0` | `Isaac-Reorient-Cube-Allegro-Direct` | All | 4096 |
| `shadow_cube_direct` | Manipulation | `Isaac-Repose-Cube-Shadow-Direct-v0` | `Isaac-Reorient-Cube-Shadow-Direct` | All | 4096 |
| `shadow_openai_ff_direct` | Manipulation | `Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-v0` | `Isaac-Reorient-Cube-Shadow-OpenAI-FF-Direct` | All | 4096 |
| `shadow_openai_lstm_direct` | Manipulation | `Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0` | `Isaac-Reorient-Cube-Shadow-OpenAI-LSTM-Direct` | Runtime only | 4096 |
| `shadow_camera_direct` | Manipulation | `Isaac-Repose-Cube-Shadow-Vision-Direct-v0` | `Isaac-Reorient-Cube-Shadow-Camera-Direct` | All | 1225 |
| `shadow_handover_direct` | Manipulation | `Isaac-Shadow-Hand-Over-Direct-v0` | `Isaac-Shadow-Handover-Direct` | Runtime only | 2048 |

“All” means runtime at 100 and 1000 steps plus RSL-RL training at 100
iterations. A task is runtime-only when either version lacks an RSL-RL
configuration. SKRL is deliberately outside scope. Play/deprecated aliases,
Lab 3-only tasks, Shadow Camera Benchmark, and the non-equivalent Newton-IK
pair are excluded.

## Matrix and execution design

Add an optional per-task `num_envs` field. When omitted, the matrix-level 4096
default remains unchanged. Attempt identity, commands, manifests, normalized
CSV, and report tables use the resolved per-task value.

The expanded final set contains 336 paired runs and 672 version attempts:

- Existing final successes imported and skipped: 408.
- New final attempts executed: 264.
- Expanded canary attempts: 224.
- Existing canary successes imported and skipped: 136.
- New canary attempts executed: 88.

Use a new artifact root:

```text
/home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd-core-expanded-672
```

For each run set, use the existing transactional importer with
`--prepare_only`, verify the import audit, then run the ordinary expanded
matrix. The runner skips validated imported successes and invokes executors
only for missing identities. The original
`c8d672a1dd-expanded-408` root remains byte-for-byte unchanged.

Canary may run without waiting for full-machine idleness. Final attempts use
the existing idle gate and wait as necessary.

## Runtime-diagnosis report integration

Copy the validated 2026-07-26 investigation bundle into the fresh final run
set before its first report publication:

```text
final/runtime_investigation_2026-07-26/
```

Add a strict read-only loader that verifies its `SHA256SUMS`, rejects unsafe or
unlisted paths, and validates the pinned Lab revisions and task-count metadata.
The main report labels it as an analysis of the original 408-run snapshot.

Insert a self-contained “Runtime diagnosis” section after aggregate results
and before category detail. It includes:

- The PhysX/Isaac Sim 6 tiny-articulation solver finding and Nsight evidence.
- The indexed target/list-to-Warp conversion overhead.
- Workload dependence, including G1 Flat and Go1 Rough.
- Startup phase movement and the Digit warning storm.
- The non-equivalent Kuka decimation/control-rate comparison.
- Rejected hypotheses and confidence limits.
- An actionable fix-target table for PhysX, Isaac Lab APIs, task
  configuration, and benchmark-equivalence checks.
- Links to the detailed Markdown/PDF, traces, profiles, reproducers, and
  checksums.

Render the validated text and selected PNGs directly through the existing
deterministic PDF pipeline. Do not execute code from the bundle or concatenate
PDF files. Keep the canonical report output inventory at 57 generated files;
the diagnosis remains hashed raw input outside `final/report`.

## Validation and failure handling

Use focused tests only:

- Matrix parsing, resolved per-task environment counts, exact 224/672
  expansion counts, and preserved identities for all imported attempts.
- Executor commands for the three non-default environment counts and camera
  flags.
- Old-subset-to-new-superset import and skip behavior.
- Diagnosis path/checksum validation, Markdown placement, deterministic PDF
  tokens/order, and report atomic rollback.

Run the focused benchmark-comparison suite and required pre-commit hooks. Do
not run the full Isaac Lab simulator or repository test suites.

If a new canary fails, diagnose only that task pair. Do not start final runs
until all new canaries succeed or the task is proven non-equivalent and
removed from the matrix. Retain every raw failure and successful attempt.
