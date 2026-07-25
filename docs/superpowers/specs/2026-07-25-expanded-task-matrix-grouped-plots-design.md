<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Expanded Task Matrix and Grouped Plots Design

**Status:** Approved

## Context

The completed IsaacLab 2.x-versus-3.0 comparison contains 228 immutable
successful final attempts and 76 immutable successful canary attempts. Its
report covers 13 logical tasks, three benchmark modes, three final seeds, two
versions, startup phases, runtime/training throughput, GPU resources, Markdown,
editable plots, and a deterministic PDF.

The next comparison adds ten logical tasks that exist with RSL-RL
configurations in both IsaacLab versions. It must not rerun the existing 228
final attempts. The expanded report should combine the original and new
measurements while making the larger task inventory readable through three
plot categories: Classic, Locomotion, and Manipulation.

## Goals

- Add G1 Rough, Digit Flat/Rough, Go1 Flat/Rough, Go2 Flat/Rough, Franka
  Cabinet Direct, and Kuka Allegro Reorient/Lift.
- Run runtime-100, runtime-1000, and training-100 for every new task.
- Preserve 4,096 environments, seeds 42/43/44, RSL-RL, version ordering, and
  all existing mode bounds.
- Reuse the 228 final and 76 canary successes without changing their bytes.
- Execute only 60 new canaries and 180 new final attempts.
- Split each comparison metric into separate Classic, Locomotion, and
  Manipulation figures.
- Produce one audited 408-attempt report while retaining the original
  228-attempt root and report unchanged.

## Non-goals

- Rerun an existing final or canary attempt.
- Change benchmark bounds, environment count, seeds, framework, task
  semantics, execution order policy, or idle-gate policy.
- Rewrite the original artifact root, its manifest, success directories, raw
  hash manifest, report, diagnostics, or runner state.
- Merge measurements with different Lab 2/Lab 3 source, image, or lock
  identities.
- Add legacy modes, performance pass/fail thresholds, or new dependencies.
- Add camera variants of the Kuka Allegro tasks.
- Add the manager-based Lab 3 Franka Cabinet task, because IsaacLab 2 only
  provides the Direct workflow.

## Task Matrix

The expanded matrix contains 23 logical tasks in this deterministic order.
Every task supports all three modes except `cartpole_rgb_kit`, which remains
runtime-only.

### Classic

| Alias | IsaacLab 2 task | IsaacLab 3 task |
|---|---|---|
| `cartpole` | `Isaac-Cartpole-v0` | `Isaac-Cartpole` |
| `cartpole_rgb_kit` | `Isaac-Cartpole-RGB-v0` | `Isaac-Cartpole-Camera` |
| `cartpole_direct` | `Isaac-Cartpole-Direct-v0` | `Isaac-Cartpole-Direct` |
| `ant` | `Isaac-Ant-v0` | `Isaac-Ant` |
| `ant_direct` | `Isaac-Ant-Direct-v0` | `Isaac-Ant-Direct` |
| `humanoid_manager` | `Isaac-Humanoid-v0` | `Isaac-Humanoid` |
| `humanoid_direct` | `Isaac-Humanoid-Direct-v0` | `Isaac-Humanoid-Direct` |

### Locomotion

| Alias | IsaacLab 2 task | IsaacLab 3 task |
|---|---|---|
| `anymal_d_flat` | `Isaac-Velocity-Flat-Anymal-D-v0` | `Isaac-Velocity-Flat-AnymalD` |
| `anymal_d_rough` | `Isaac-Velocity-Rough-Anymal-D-v0` | `Isaac-Velocity-Rough-AnymalD` |
| `g1_flat` | `Isaac-Velocity-Flat-G1-v0` | `Isaac-Velocity-Flat-G1` |
| `g1_rough` | `Isaac-Velocity-Rough-G1-v0` | `Isaac-Velocity-Rough-G1` |
| `cassie_flat` | `Isaac-Velocity-Flat-Cassie-v0` | `Isaac-Velocity-Flat-Cassie` |
| `digit_flat` | `Isaac-Velocity-Flat-Digit-v0` | `Isaac-Velocity-Flat-Digit` |
| `digit_rough` | `Isaac-Velocity-Rough-Digit-v0` | `Isaac-Velocity-Rough-Digit` |
| `go1_flat` | `Isaac-Velocity-Flat-Unitree-Go1-v0` | `IsaacContrib-Velocity-Flat-UnitreeGo1` |
| `go1_rough` | `Isaac-Velocity-Rough-Unitree-Go1-v0` | `IsaacContrib-Velocity-Rough-UnitreeGo1` |
| `go2_flat` | `Isaac-Velocity-Flat-Unitree-Go2-v0` | `Isaac-Velocity-Flat-UnitreeGo2` |
| `go2_rough` | `Isaac-Velocity-Rough-Unitree-Go2-v0` | `Isaac-Velocity-Rough-UnitreeGo2` |

### Manipulation

| Alias | IsaacLab 2 task | IsaacLab 3 task |
|---|---|---|
| `allegro_cube` | `Isaac-Repose-Cube-Allegro-v0` | `Isaac-Reorient-Cube-Allegro` |
| `franka_reach` | `Isaac-Reach-Franka-v0` | `Isaac-Reach-Franka` |
| `franka_cabinet_direct` | `Isaac-Franka-Cabinet-Direct-v0` | `Isaac-Open-Drawer-Franka-Direct` |
| `kuka_allegro_reorient` | `Isaac-Dexsuite-Kuka-Allegro-Reorient-v0` | `Isaac-Reorient-KukaAllegro` |
| `kuka_allegro_lift` | `Isaac-Dexsuite-Kuka-Allegro-Lift-v0` | `Isaac-Lift-KukaAllegro` |

The Go1 task moved under `IsaacContrib` in IsaacLab 3, but its corresponding
Flat/Rough environment and RSL-RL runner configurations exist in both
versions. Kuka Allegro Reorient and Lift also expose RSL-RL configurations in
both versions.

## Expansion Counts

The two runtime modes cover all 23 tasks:

```text
23 tasks × 2 runtime modes × 3 seeds × 2 versions = 276 final attempts
```

Training covers 22 tasks because Cartpole RGB remains runtime-only:

```text
22 tasks × 1 training mode × 3 seeds × 2 versions = 132 final attempts
```

The expanded final matrix therefore contains 408 attempts. The ten additions
contribute 180 attempts; the original 228 remain complete.

The canary uses seed 42 only:

```text
(23 × 2 × 1 × 2) + (22 × 1 × 1 × 2) = 136 canary attempts
```

The ten additions contribute 60 canaries; the original 76 remain complete.

## Artifact and Provenance Strategy

A new artifact root owns the expanded schema-2 canary and final manifests.
The original artifact root remains read-only and unchanged.

The expanded root is seeded as follows:

1. Read and strictly validate the original canary/final manifests.
2. Verify every original success bundle, its checksums, semantic data, exact
   attempt identity, and execution provenance.
3. Create the new schema-2 manifests with the expanded 136/408 attempt
   identities and the exact original execution provenance.
4. Copy the complete immutable attempt directory for each of the 76 canary and
   228 final successes using independent copies or copy-on-write reflinks. An
   attempt import includes `success/` and any preceding failure or quarantine
   directories needed to preserve its validated attempt number. Do not use
   hardlinks or symlinks.
5. Revalidate every imported success against the expanded manifest and its
   copied attempt history, then prove byte-for-byte equality with its original
   source.
6. Record an import audit containing source/destination roots, source manifest
   digests, imported counts, file counts, and aggregate hashes.

The execution identities remain pinned to the completed comparison:

- the original Lab 2 source SHA and local Docker image ID;
- the original Lab 3 source SHA and `uv.lock` digest; and
- the original hardware/GPU identity.

The controller runs from the updated harness checkout, but executors target
separate worktrees/environments pinned to those historical source identities.
Controller changes do not alter the code under benchmark.

Manifest creation and result import complete before an executor is allowed to
run. The runner then treats the imported successes as completed and schedules
only attempts absent from the expanded root.

## Plot Grouping

Category membership is explicit, deterministic, and validated. Every task in
the configured matrix must belong to exactly one category; unknown,
unassigned, or multiply assigned aliases are rejected before plot generation.

The six metric families remain:

- collection FPS;
- mean GPU memory;
- peak GPU memory;
- mean GPU utilization;
- total startup time; and
- startup phase breakdown.

Each metric produces three separate category figures. Category-first filenames
keep the output grouped and discoverable:

```text
classic_<metric>.png
classic_<metric>.svg
locomotion_<metric>.png
locomotion_<metric>.svg
manipulation_<metric>.png
manipulation_<metric>.svg
```

Each figure retains the existing three mode panels, deterministic dimensions,
fixed metadata/font/color/hatch conventions, seed points, error bars, task
ordering, and missing-value behavior. The Classic training panel omits the
runtime-only Cartpole RGB task rather than showing a training slot.

The six combined figure families are replaced in the expanded report by 18
category figure families, for 36 editable image files. The original report
continues to retain its six combined figures unchanged.

## Markdown, PDF, and Generated Inventory

The expanded Markdown and PDF use the same normalized 408-run model. Their
task order follows the category blocks and the order inside each block.
Paired startup/runtime tables and the individual-run appendix retain their
current metric definitions and signed-delta semantics.

The PDF includes all 18 category figures in this fixed order:

1. all six Classic figures;
2. all six Locomotion figures; and
3. all six Manipulation figures.

Within each category, metric order follows the existing plot family order.
The PDF continues to validate its exact title, metadata, page count, version
SHAs, run-set identity, and extractable text before atomic publication.

The expanded report has exactly 41 generated files:

```text
3 normalized CSVs + report.md + report.pdf + 36 plot files = 41
```

`generated_hashes.sha256` contains exactly those 41 paths. The external audit
summary and both hash manifests remain outside the generated self-hash. Raw
hashing excludes both the selected output directory and the canonical report
directory.

## Execution Workflow

No existing attempt is rerun.

1. Implement and verify the matrix, category plots, import utility, and report
   inventory without launching a simulator.
2. Seed and validate the expanded canary/final roots.
3. Run only the 60 new canary attempts. Canaries may run while the machine is
   otherwise in use, consistent with the prior policy.
4. Resolve any mapping/configuration failure before final execution; do not
   substitute a different task silently.
5. Before real benchmarking, require the existing machine-idle gate and verify
   no unrelated GPU/CPU workload is active.
6. Run only the 180 missing final attempts under the existing balanced version
   order and retry/quarantine policy.
7. Generate the expanded report twice from the completed artifacts and require
   byte-identical generated hashes.

All execution settings remain unchanged:

- RSL-RL only;
- 4,096 environments;
- seeds 42, 43, and 44;
- runtime bounds of 100 and 1,000 steps; and
- training bound `--max_iterations 100`.

## Failure Handling and Integrity

- An imported success that fails checksum, semantic, identity, or provenance
  validation aborts import; it is not marked missing and is not rerun
  automatically.
- A task registration or RSL-RL configuration missing from either version is
  reported as a mapping defect. No cross-version substitute is inferred.
- Original artifacts remain unchanged even if import, canary, final execution,
  plotting, PDF rendering, or publication fails.
- Plot/PDF failure leaves the previously complete expanded report intact.
- Raw artifacts are rehashed immediately before publication.
- The final audit must reconcile manifest attempts, imported attempt counts,
  runner history (`skipped_success` for imports and `success` for new runs),
  success directories, normalized rows, paired rows, raw hashes, and generated
  hashes.

## Testing Strategy

Testing is proportional to this task-list and layout extension:

- focused matrix tests for exact 23-task mappings, supported modes, 136/408
  expansion counts, and stable version ordering;
- focused registration/configuration checks for all ten new mappings in both
  versions;
- focused import tests for independent-copy semantics, checksum/provenance
  rejection, exact imported identities/counts, and original-root immutability;
- focused plot tests for complete/disjoint category membership, exact 18-family
  ordering, 36-file inventory, dimensions, labels, and deterministic bytes;
- focused PDF/report CLI tests for 41 generated files, exact hash inventory,
  category figure order, and atomic rollback;
- one full simulator-free benchmark-comparison suite after implementation;
- repository-wide pre-commit before commits and handoff;
- 60 new canaries, followed by 180 idle-gated final attempts; and
- two report-only regenerations with identical generated-manifest digests.

Separate one-off smoke runs are not required unless a canary exposes a defect
that needs a narrower reproduction. The existing 228 final and 76 canary
attempts are never used as regression executions.

## Deliverables

- expanded matrix and exact task mappings;
- validated task-category model;
- safe result-import command and import audit;
- expanded canary/final artifact root containing 136/408 successes;
- raw normalized CSVs and paired summaries;
- 36 grouped PNG/SVG figures;
- combined Markdown and deterministic PDF report;
- raw/generated hash manifests and audit summary; and
- retained original 228-attempt artifact root and report, unchanged.
