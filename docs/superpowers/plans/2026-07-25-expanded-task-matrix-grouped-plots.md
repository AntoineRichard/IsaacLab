<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Expanded Task Matrix and Grouped Plots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ten cross-version RSL-RL tasks, reuse the 76 canary and 228 final successes without mutation, execute only 60 new canaries and 180 new idle-gated final attempts, and publish one audited 408-attempt report with Classic, Locomotion, and Manipulation plots.

**Architecture:** Keep the updated harness checkout as the controller while both executors remain pinned to the historical source/image/lock identities. Build expanded schema-2 manifests in a new artifact root, transactionally copy every retained completed attempt directory, let immutable resume semantics skip those imports, and derive 18 category plot families plus Markdown/PDF from the combined normalized data.

**Tech Stack:** Python 3.12, standard-library dataclasses/TOML/JSON/CSV/filesystem APIs, Matplotlib Agg/PDF backends, pytest, Docker Compose for IsaacLab 2.3.2, locked `uv` for IsaacLab 3.0, RSL-RL, NVIDIA telemetry, pre-commit.

## Global Constraints

- Do not modify or regenerate `/home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd`.
- Use `/home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd-expanded-408` as the new artifact root.
- Keep Lab 2 SHA `c8d672a1dd662324c490a9887fe0f5c01478a910`, Lab 3 SHA `cb508381fb4874ce7afffeb9197bd91c20db7dad`, Lab 2 image `isaac-lab-base-benchmark-c8d672a1dd:latest`, image ID `sha256:4f6282c2a7b36cb7457c3262e87ea30e1c522e2e57c5ad158dbf94a276f517fb`, and lock digest `911857c6da5cb0b07b96222d54894fd2a90941c142a4c178ee1de729de035e18` exact.
- Keep GPU UUID `GPU-c641346f-f5cc-f5ca-beb8-6a5d4957872f`, 4,096 environments, RSL-RL, seeds 42/43/44, and current counterbalanced version order exact.
- `cartpole_rgb_kit` remains runtime-only. Every other task runs runtime-100, runtime-1000, and training-100; training uses `--max_iterations 100`.
- Copy complete attempt directories, including `success/` and preceding retry/quarantine evidence. Never hardlink, symlink, rewrite, or normalize imported bytes.
- Canary execution may proceed while the host is busy. Final execution may start only after the existing idle gate proves the machine is free.
- Do not add legacy modes, performance thresholds, required dependencies, optional dependencies, or a changelog fragment.
- Use PEP 8, modern Python types, Google-style docstrings, snake_case CLI arguments, and the 2026 SPDX header for new files.

## File Map

- Modify `tools/benchmark_comparison/models.py`: add the task-category enum and category field.
- Modify `tools/benchmark_comparison/matrix.toml`: add categories and the ten approved task mappings.
- Modify `tools/benchmark_comparison/matrix.py`: parse/validate categories and update current schema-2 counts without changing schema-1 compatibility.
- Create `tools/benchmark_comparison/import_results.py`: validate and transactionally import complete successful attempt directories.
- Modify `tools/benchmark_comparison/cli.py`: add preparation/import arguments and stop before executor construction in prepare-only mode.
- Modify `tools/benchmark_comparison/plot.py`: generate six metrics for three category groups.
- Modify `tools/benchmark_comparison/pdf_report.py`: accept and render the 18 fixed category figures.
- Modify `tools/benchmark_comparison/report_cli.py`: publish the 41-file generated inventory.
- Modify `tools/benchmark_comparison/tests/test_matrix.py`: exact mappings, categories, counts, modes, and version order.
- Create `tools/benchmark_comparison/tests/test_import_results.py`: transaction, integrity, retry-history, copy independence, and idempotency coverage.
- Create `tools/benchmark_comparison/tests/test_cli.py`: preparation/import ordering and no-executor guarantees.
- Modify `tools/benchmark_comparison/tests/test_plot.py`: exact 18-family names, membership, labels, dimensions, and deterministic bytes.
- Modify `tools/benchmark_comparison/tests/test_pdf_report.py`: exact category-page ordering and rejection behavior.
- Modify `tools/benchmark_comparison/tests/test_report_cli_success.py`: exact 41-file synthetic report.
- Modify `tools/benchmark_comparison/tests/test_report_cli_atomicity.py`: grouped-plot/PDF rollback coverage.
- Modify `tools/benchmark_comparison/tests/test_report_integrity.py`: exact generated hash inventory.
- Keep `tools/benchmark_comparison/tests/test_actual_report_artifacts.py` explicitly fixed at the retained 228-attempt, 17-generated-file report.

---

### Task 1: Expand the Typed Matrix and Category Model

**Files:**
- Modify: `tools/benchmark_comparison/models.py`
- Modify: `tools/benchmark_comparison/matrix.toml`
- Modify: `tools/benchmark_comparison/matrix.py`
- Modify: `tools/benchmark_comparison/tests/test_matrix.py`

- [ ] **Step 1: Write the failing exact-matrix tests**

Replace `_EXPECTED_TASK_ALIASES` with the exact 23-entry ordered mapping from the approved design. Add `_EXPECTED_CATEGORIES` with these exact groups:

```python
_EXPECTED_CATEGORIES = {
    "classic": (
        "cartpole",
        "cartpole_rgb_kit",
        "cartpole_direct",
        "ant",
        "ant_direct",
        "humanoid_manager",
        "humanoid_direct",
    ),
    "locomotion": (
        "anymal_d_flat",
        "anymal_d_rough",
        "g1_flat",
        "g1_rough",
        "cassie_flat",
        "digit_flat",
        "digit_rough",
        "go1_flat",
        "go1_rough",
        "go2_flat",
        "go2_rough",
    ),
    "manipulation": (
        "allegro_cube",
        "franka_reach",
        "franka_cabinet_direct",
        "kuka_allegro_reorient",
        "kuka_allegro_lift",
    ),
}
```

Assert every configured task has the expected `TaskCategory`, all groups are disjoint, their union is all 23 aliases, and their concatenation is the matrix order. Assert the new mappings exactly:

```python
{
    "g1_rough": ("Isaac-Velocity-Rough-G1-v0", "Isaac-Velocity-Rough-G1"),
    "digit_flat": ("Isaac-Velocity-Flat-Digit-v0", "Isaac-Velocity-Flat-Digit"),
    "digit_rough": ("Isaac-Velocity-Rough-Digit-v0", "Isaac-Velocity-Rough-Digit"),
    "go1_flat": ("Isaac-Velocity-Flat-Unitree-Go1-v0", "IsaacContrib-Velocity-Flat-UnitreeGo1"),
    "go1_rough": ("Isaac-Velocity-Rough-Unitree-Go1-v0", "IsaacContrib-Velocity-Rough-UnitreeGo1"),
    "go2_flat": ("Isaac-Velocity-Flat-Unitree-Go2-v0", "Isaac-Velocity-Flat-UnitreeGo2"),
    "go2_rough": ("Isaac-Velocity-Rough-Unitree-Go2-v0", "Isaac-Velocity-Rough-UnitreeGo2"),
    "franka_cabinet_direct": ("Isaac-Franka-Cabinet-Direct-v0", "Isaac-Open-Drawer-Franka-Direct"),
    "kuka_allegro_reorient": ("Isaac-Dexsuite-Kuka-Allegro-Reorient-v0", "Isaac-Reorient-KukaAllegro"),
    "kuka_allegro_lift": ("Isaac-Dexsuite-Kuka-Allegro-Lift-v0", "Isaac-Lift-KukaAllegro"),
}
```

Update count assertions to 204 logical final pairs, 408 final attempts, 68 canary pairs, and 136 canary attempts. Assert the ten new aliases each expose all three modes and contribute exactly 18 final attempts and 6 canary attempts per alias.

Add invalid TOML cases for a missing category, an unknown category, and duplicate category aliases caused by a duplicate task alias. Keep the existing RGB runtime-only and schema-1 tests unchanged in meaning.

- [ ] **Step 2: Run the focused matrix tests and confirm RED**

Run:

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_matrix.py -v
```

Expected: FAIL because the category type and ten task entries do not exist and the expansion remains 228 attempts.

- [ ] **Step 3: Implement the category model and exact matrix**

Add:

```python
class TaskCategory(str, Enum):
    """Readability group used by benchmark reports."""

    CLASSIC = "classic"
    LOCOMOTION = "locomotion"
    MANIPULATION = "manipulation"
```

Make `BenchmarkTask.category` a required field placed after `lab3_id`. Add `category =` to every checked-in TOML task and add the ten exact mappings in approved category order. Do not add `supported_modes` to any new task.

Update current matrix constants to:

```python
FINAL_LOGICAL_PAIR_COUNT = 204
FINAL_ATTEMPT_COUNT = 408
CANARY_LOGICAL_PAIR_COUNT = 68
CANARY_ATTEMPT_COUNT = 136
```

Extend `_TASK_IDENTIFIERS` with category plus both concrete IDs. Parse `task.category` through `TaskCategory`. Validate exact configured order, category completeness, category block order, task/mode shape, and counts. Give the six schema-1 compatibility tasks explicit categories while leaving `_LEGACY_SCHEMA_1_TASK_IDENTIFIERS`, its 108/36 attempt counts, identities, and manifest parsing unchanged.

Expose a helper that returns configured aliases by category for a supplied expansion. It must preserve `TaskCategory.CLASSIC`, `TaskCategory.LOCOMOTION`, `TaskCategory.MANIPULATION` order, reject expansion aliases absent from the current matrix, reject missing/duplicate assignments, and filter each group to aliases present in that expansion.

- [ ] **Step 4: Run matrix and manifest compatibility tests and confirm GREEN**

Run:

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_matrix.py \
  tools/benchmark_comparison/tests/test_manifest.py \
  tools/benchmark_comparison/tests/test_manifest_normalization.py -v
```

Expected: PASS, including retained schema-1 reconstruction.

- [ ] **Step 5: Commit the matrix unit**

```bash
git add tools/benchmark_comparison/models.py tools/benchmark_comparison/matrix.py \
  tools/benchmark_comparison/matrix.toml tools/benchmark_comparison/tests/test_matrix.py
git commit -m "Expand benchmark task matrix"
```

---

### Task 2: Add Transactional Completed-Attempt Import

**Files:**
- Create: `tools/benchmark_comparison/import_results.py`
- Create: `tools/benchmark_comparison/tests/test_import_results.py`

**Interface:**

```python
def import_completed_attempts(
    source_root: Path,
    destination_root: Path,
    run_set: RunSet,
) -> ImportAudit:
    """Import a validated completed run-set subset into an expanded root."""
```

`ImportAudit` contains source/destination roots, run set, source/destination manifest SHA-256 values, imported attempt count, imported file count, source aggregate SHA-256, and destination aggregate SHA-256. Publish it atomically at `destination_root / run_set.value / "import_audit.json"`.

- [ ] **Step 1: Write a two-attempt import fixture and failing happy-path test**

Build a schema-2 source manifest from the first pair of a canary expansion and a schema-2 destination manifest from its first two pairs. Use `finalize_attempt` to create both source successes with exact synthetic provenance/GPU environment. Before the second source success, finalize one nonzero failure so its successful `validation.json` has attempt number 2.

Call `import_completed_attempts` and assert:

- exactly two attempts and every file below both attempt roots were imported;
- the retrying attempt contains both `attempt-0001-nonzero_exit/` and `success/`;
- `verify_success` accepts both destination successes against destination attempts;
- source and destination aggregate hashes match;
- source and destination files have equal bytes but different inode numbers;
- no symlink and no `.import-staging-` directory remains;
- source tree bytes and metadata-visible file inventory are unchanged; and
- the audit JSON round-trips exactly to the returned frozen dataclass.

- [ ] **Step 2: Run the happy-path test and confirm RED**

Run:

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_import_results.py::test_import_completed_attempts_copies_full_attempt_history_independently -v
```

Expected: FAIL because `tools.benchmark_comparison.import_results` does not exist.

- [ ] **Step 3: Add failing integrity and transaction tests**

Cover each case independently:

- source and destination are equal, nested, or reverse-nested;
- source/destination run sets, phases, provenance, host/GPU, or software identities differ;
- source attempt identity is absent from the destination expansion;
- same directory identity maps to different concrete task, bound, camera flag, preset, or framework;
- source attempt root is absent, has no success, contains a symlink, or has corrupt success bytes;
- source success validation attempt number does not match copied retry history;
- destination has a conflicting attempt directory or import audit;
- injected `copy2`, validation, publication, and audit-write failures leave no published attempt and no staging residue;
- a second identical call is idempotent and changes no bytes; and
- tampering after the first call makes the idempotent call fail closed.

Use monkeypatching at named internal seams instead of patching `os.replace` process-wide. Include one publication-failure test with two staged attempt roots and assert rollback removes the first published root.

- [ ] **Step 4: Implement strict validation, staging, rollback, and audit publication**

Implementation rules:

1. Resolve both roots and reject equality or containment in either direction.
2. Read both manifests with `read_manifest`, resolve expansions with `resolve_manifest_expansion`, and require schema 2.0.
3. Require equal run set, phase, provenance, host, Lab 2 software, Lab 3 software, and CPU power profile.
4. Index attempts by immutable `identity`. Require every source identity in the destination and compare `attempt_identity`, `run_directory`, `enable_cameras`, `extra_presets`, `version_order`, and complete mode bounds. Permit only `pair_order` and `attempt_order` to differ.
5. Reject non-directories and every symlink below an imported attempt root. Verify the source `success/` with destination provenance/GPU and destination attempt before copying.
6. Hash the source import set using sorted attempt identities, relative child paths, file modes, sizes, and bytes. Do not include timestamps or inode numbers.
7. Copy each complete attempt root with `shutil.copytree` and `shutil.copy2` below one `.import-staging-` directory in the destination run-set directory. Do not preserve symlinks.
8. Revalidate staged successes and require their aggregate digest to equal the source digest.
9. Acquire an import lock. Reject destination conflicts, publish attempt roots with `os.replace`, and track every move. On any publication error, move published roots back to staging before raising.
10. Write the deterministic audit through a temporary file, `fsync`, and `os.replace` only after all attempts are published. On audit failure, roll back every newly published attempt root.
11. On an existing identical audit, revalidate all imported destination attempts and hashes, then return without copying.
12. In `finally`, remove only the importer-owned staging directory. Never delete or rewrite the source.

Use `verify_success` for the canonical checksum, semantic, attempt-number, provenance, and selected-GPU check. Keep internal JSON writing deterministic with `sort_keys=True`, `allow_nan=False`, UTF-8, and a trailing newline.

- [ ] **Step 5: Run importer and artifact tests and confirm GREEN**

Run:

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_import_results.py \
  tools/benchmark_comparison/tests/test_artifacts.py -v
```

Expected: PASS with no simulator launch.

- [ ] **Step 6: Commit the import unit**

```bash
git add tools/benchmark_comparison/import_results.py \
  tools/benchmark_comparison/tests/test_import_results.py
git commit -m "Add benchmark result importer"
```

---

### Task 3: Integrate Safe Preparation and Import into the Controller

**Files:**
- Modify: `tools/benchmark_comparison/cli.py`
- Create: `tools/benchmark_comparison/tests/test_cli.py`
- Modify: `tools/benchmark_comparison/tests/test_runner.py`

- [ ] **Step 1: Write failing parser and prepare-only tests**

Add CLI arguments:

```text
--import_from_artifact_root PATH
--prepare_only
```

Test `main` with preflight, manifest writing, and importer replaced by deterministic fakes. Assert this exact order:

1. preflight completes;
2. expanded manifest is written;
3. import completes and returns a validated audit;
4. prepare-only returns zero; and
5. `OwnedProcessGroups`, `ProcessLauncher`, `HostIdleGate`, both executors, and `BenchmarkRunner` are never constructed.

Add failures for `--prepare_only` without `--import_from_artifact_root`, import source equal/nested with destination, and importer rejection. Assert each returns nonzero through the existing exception behavior without constructing an executor.

- [ ] **Step 2: Run the prepare-only test and confirm RED**

Run:

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_cli.py::test_prepare_only_writes_manifest_and_imports_before_executor_construction -v
```

Expected: FAIL because the parser does not recognize the two new arguments.

- [ ] **Step 3: Write a failing resume integration test**

Prepare a four-attempt destination expansion with two imported successes. Run `BenchmarkRunner` with fake executors for the remaining attempts. Assert imported attempts produce `skipped_success`, idle gating and executors are called only twice, new attempts produce `success`, and the latest history event for every identity reconciles to four completed attempts.

- [ ] **Step 4: Implement CLI ordering and early return**

Resolve `--import_from_artifact_root` to a `Path`. In `_run_locked`, perform preflight and expansion, write the destination manifest, call `import_completed_attempts` when configured, and return zero immediately for prepare-only. Construct process groups, launcher, idle gate, executors, and runner only after those steps.

Do not import source `runner-state.json` or idle evidence. During a real controller run, immutable destination successes are visited and recorded as `skipped_success`; only missing attempts reach the idle gate and executor.

- [ ] **Step 5: Run CLI and runner tests and confirm GREEN**

Run:

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_cli.py \
  tools/benchmark_comparison/tests/test_runner.py -v
```

Expected: PASS and fake executor call count equals only missing attempts.

- [ ] **Step 6: Commit the controller unit**

```bash
git add tools/benchmark_comparison/cli.py tools/benchmark_comparison/tests/test_cli.py \
  tools/benchmark_comparison/tests/test_runner.py
git commit -m "Seed expanded benchmark runs safely"
```

---

### Task 4: Generate Three Category Figures per Metric

**Files:**
- Modify: `tools/benchmark_comparison/plot.py`
- Modify: `tools/benchmark_comparison/tests/test_plot.py`

- [ ] **Step 1: Replace the six-family assertions with failing grouped assertions**

Define expected category-first basenames by taking each category in `classic`, `locomotion`, `manipulation` order and each metric in this order:

```python
(
    "collection_fps",
    "gpu_memory_mean_mib",
    "gpu_memory_peak_mib",
    "gpu_utilization_mean_pct",
    "startup_total_s",
    "startup_phase_breakdown",
)
```

Assert `PLOT_BASENAMES` contains exactly 18 names such as `classic_collection_fps`, `locomotion_startup_total_s`, and `manipulation_startup_phase_breakdown`. Assert generation returns exactly 36 PNG/SVG paths and regenerates byte-identically.

Create rows from every category and assert each SVG contains only its group labels. Assert the Classic training panel omits `cartpole_rgb_kit`, while its runtime panels include it. Keep exact PNG dimensions `(1800, 1000)`, missing-label behavior, 45-degree task labels, seed dots, error bars, startup phase labels, version labels/hatches, and deterministic metadata.

- [ ] **Step 2: Run grouped plot tests and confirm RED**

Run:

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_plot.py -v
```

Expected: FAIL because only six ungrouped basenames and twelve files are generated.

- [ ] **Step 3: Factor rendering around explicit category task orders**

Keep `PLOT_METRICS` and all rendering semantics unchanged. Build `PLOT_BASENAMES` category-first. In `generate_plots`, obtain the validated category alias groups from the matrix helper and invoke one scalar-metric renderer and one phase-breakdown renderer per category.

For each mode, intersect the category aliases with `task_order_for_mode(mode, expansion)` so mode capability filtering stays driven by the manifest expansion. Use titles in the form `Classic — Collection FPS` and `Manipulation — Startup Phase Breakdown`. Save atomically using the current temporary-file/`os.replace` path.

- [ ] **Step 4: Run plot and normalization tests and confirm GREEN**

Run:

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_plot.py \
  tools/benchmark_comparison/tests/test_normalize.py -v
```

Expected: PASS with exactly 36 plot files.

- [ ] **Step 5: Commit the plot unit**

```bash
git add tools/benchmark_comparison/plot.py tools/benchmark_comparison/tests/test_plot.py
git commit -m "Group benchmark plots by task category"
```

---

### Task 5: Expand PDF and Report Inventory to 41 Files

**Files:**
- Modify: `tools/benchmark_comparison/pdf_report.py`
- Modify: `tools/benchmark_comparison/report_cli.py`
- Modify: `tools/benchmark_comparison/tests/test_pdf_report.py`
- Modify: `tools/benchmark_comparison/tests/test_report_cli_success.py`
- Modify: `tools/benchmark_comparison/tests/test_report_cli_atomicity.py`
- Modify: `tools/benchmark_comparison/tests/test_report_integrity.py`
- Verify unchanged: `tools/benchmark_comparison/tests/test_actual_report_artifacts.py`

- [ ] **Step 1: Write failing PDF ordering and input-validation tests**

Generate 18 small PNG fixtures named from `PLOT_BASENAMES`, pass them in reverse order, and assert PDF plot pages appear category-first and metric-second. Expected plot-page titles begin with all six Classic titles, then six Locomotion titles, then six Manipulation titles.

Add rejection tests for a missing category plot, duplicate basename, unexpected old ungrouped basename, non-PNG input, and invalid image bytes. Preserve the existing destination and remove the temporary PDF on every failure.

- [ ] **Step 2: Write failing report inventory tests**

Update the synthetic report expectation to exactly:

- 3 normalized CSV files;
- `report.md` and `report.pdf`; and
- 36 category PNG/SVG files.

Assert `generated_hashes.sha256` has exactly those 41 unique relative paths, the audit reports `generated_file_count == 41`, and a second report-only run produces byte-identical generated files and the same generated-manifest digest.

Update atomicity tests so a grouped plot or PDF failure preserves the previously published 41-file report. Update generic integrity fixtures from 17 to 41 only where they model the current report pipeline.

Do not change the retained-root assertions in `test_actual_report_artifacts.py`: they must continue to assert 228 attempts, 17 generated files, the original six combined plot families, and the original hash manifests.

- [ ] **Step 3: Run focused report tests and confirm RED**

Run:

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_pdf_report.py \
  tools/benchmark_comparison/tests/test_report_cli_success.py \
  tools/benchmark_comparison/tests/test_report_cli_atomicity.py \
  tools/benchmark_comparison/tests/test_report_integrity.py -v
```

Expected: FAIL on six-plot ordering and 17-file expectations.

- [ ] **Step 4: Implement fixed 18-page figure ordering and dynamic inventory**

Keep `PLOT_BASENAMES` as the sole allowed image order in `pdf_report.py`. Update page titles to include the category display name and metric display name. Continue to reorder shuffled input, reject missing/duplicate/unexpected inputs, validate the finished PDF, and publish atomically.

In `report_cli.py`, retain the existing staging transaction. Generate all 36 plot files before Markdown/PDF, include all 41 derived files in generated hashes, and keep `audit_summary.json`, `raw_artifact_hashes.sha256`, and `generated_hashes.sha256` outside the generated self-hash.

- [ ] **Step 5: Run the report tests and confirm GREEN**

Run the Step 3 command again.

Expected: PASS with 18 figure pages and exactly 41 generated hash entries.

- [ ] **Step 6: Run the retained-report regression explicitly**

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_actual_report_artifacts.py -v
```

Expected: PASS against the untouched 228-attempt, 17-generated-file original report.

- [ ] **Step 7: Commit the report unit**

```bash
git add tools/benchmark_comparison/pdf_report.py tools/benchmark_comparison/report_cli.py \
  tools/benchmark_comparison/tests/test_pdf_report.py \
  tools/benchmark_comparison/tests/test_report_cli_success.py \
  tools/benchmark_comparison/tests/test_report_cli_atomicity.py \
  tools/benchmark_comparison/tests/test_report_integrity.py
git commit -m "Expand benchmark report figure inventory"
```

---

### Task 6: Complete Simulator-Free Verification and Freeze the Original Root Digest

**Files:**
- Modify only if failures require it: files already listed in Tasks 1–5
- Do not modify: measured artifact roots

- [ ] **Step 1: Run the complete benchmark-comparison suite**

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests -v
```

Expected: all tests pass with no Docker or simulator process launched by the test suite.

- [ ] **Step 2: Run repository-wide pre-commit twice as required**

```bash
./isaaclab.sh -f
./isaaclab.sh -f
```

If the wrapper cannot use the isolated environment, use the established equivalent and record it:

```bash
uvx pre-commit run --all-files
uvx pre-commit run --all-files
```

Review every automatic change before staging it. Commit any formatting-only corrections in a focused commit.

- [ ] **Step 3: Record immutable original-root evidence outside both artifact roots**

Create `/home/antoiner/benchmarks/isaaclab2-vs-3/original-c8d672a1dd-before.sha256` by hashing every regular file below the original root in sorted relative-path order. Include paths and bytes in the aggregate. Verify the retained manifest/report facts before proceeding:

```text
canary manifest attempts: 76
final manifest attempts: 228
final raw_runs.csv rows: 228
final failures.csv data rows: 0
final generated hash entries: 17
```

Also record SHA-256 values for original canary/final manifests, both runner states, both report hash manifests, and the original PDF. This evidence file is operational and must not be committed.

- [ ] **Step 4: Verify clean source state and commits**

```bash
git status --short
git log --oneline --decorate -8
```

Expected: clean worktree and focused commits for matrix, importer, controller, plots, and report inventory.

---

### Task 7: Prepare the Expanded Roots and Run Only the 60 New Canaries

**Files/artifacts:**
- Create operational worktree: `/home/antoiner/benchmarks/isaaclab2-vs-3/lab2-execution-c8d672a1dd`
- Create artifact root: `/home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd-expanded-408`
- Preserve source root: `/home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd`

- [ ] **Step 1: Create and verify the pinned Lab 2 execution worktree**

```bash
git worktree add --detach \
  /home/antoiner/benchmarks/isaaclab2-vs-3/lab2-execution-c8d672a1dd \
  c8d672a1dd662324c490a9887fe0f5c01478a910
git -C /home/antoiner/benchmarks/isaaclab2-vs-3/lab2-execution-c8d672a1dd rev-parse HEAD
git -C /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop rev-parse HEAD
docker image inspect isaac-lab-base-benchmark-c8d672a1dd:latest \
  --format '{{.Id}}'
```

Expected: the two SHAs and image ID equal the pinned values in Global Constraints. If the execution worktree already exists, verify it instead of recreating it.

- [ ] **Step 2: Prepare and import canary and final run sets without executors**

From the updated controller worktree, run the controller module twice with `--prepare_only` and `--import_from_artifact_root /home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd`. Use phase `measured`, the pinned roots and identities, and the new artifact root.

Canary command:

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked \
  python -m tools.benchmark_comparison.cli \
  --run_set canary --phase measured \
  --lab2_root /home/antoiner/benchmarks/isaaclab2-vs-3/lab2-execution-c8d672a1dd \
  --lab3_root /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --artifact_root /home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd-expanded-408 \
  --lab2_sha c8d672a1dd662324c490a9887fe0f5c01478a910 \
  --lab3_sha cb508381fb4874ce7afffeb9197bd91c20db7dad \
  --lab2_image isaac-lab-base-benchmark-c8d672a1dd:latest \
  --lab2_image_id sha256:4f6282c2a7b36cb7457c3262e87ea30e1c522e2e57c5ad158dbf94a276f517fb \
  --import_from_artifact_root /home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd \
  --prepare_only
```

Repeat with `--run_set final`. Expected audits: 76 imported canaries and 228 imported final attempts. No runner state, idle evidence, or output directory is created by preparation.

- [ ] **Step 3: Audit imported roots before any new execution**

Verify:

- expanded manifests contain exactly 136 canary and 408 final attempts;
- success directories count exactly 76 canary and 228 final;
- missing identities are exactly the 60 canary and 180 final attempts belonging to the ten new aliases;
- both import audits match source/destination manifests and aggregate digests;
- the imported Cartpole Direct retry root includes its preceding failure evidence;
- every imported success passes `verify_success` against the expanded manifest; and
- recomputing the original-root digest matches `original-c8d672a1dd-before.sha256`.

Stop on any mismatch. Do not let the controller reinterpret an invalid import as a missing attempt.

- [ ] **Step 4: Execute the canary matrix**

Run the canary command from Step 2 again without `--prepare_only` and without `--import_from_artifact_root`. The runner must skip 76 imported successes and execute only 60 missing attempts.

Canaries may run while the host is otherwise busy, but do not deliberately overlap another GPU simulator job. Monitor runner state and immutable attempt directories. On interruption, resume with the same command; use `--retry_failures` only after classifying a preserved failure.

- [ ] **Step 5: Reconcile canary completion before final execution**

Require:

```text
manifest attempts: 136
success directories: 136
failed latest identities: 0
imported identities: 76
newly executed identities: 60
```

For each identity, use the latest runner-history event when retries or resumes created repeated history rows. Imported identities must end in `skipped_success`; new identities must have a `success` event and corresponding idle evidence. Confirm every new mapping registered and emitted valid runtime/training schema plus resource telemetry in both versions.

---

### Task 8: Wait for an Idle Machine and Run Only the 180 New Final Attempts

**Files/artifacts:**
- Modify only: expanded `final/` operational artifacts
- Do not modify: original root or imported success bytes

- [ ] **Step 1: Prove final-run readiness**

Immediately before launch, verify:

- canary reconciliation is complete;
- current source/image/lock/GPU identities equal the pinned manifest;
- expanded final root contains exactly 228 imported successes and 180 missing identities;
- no unrelated Docker GPU container is running;
- no unrelated compute process owns the selected GPU;
- CPU load and GPU memory/utilization meet the existing `HostIdleGate` thresholds; and
- original-root aggregate digest still matches the before evidence.

If the machine is busy, wait and recheck. Do not relax thresholds or bypass the gate.

- [ ] **Step 2: Launch the final controller with the existing idle gate**

Use the Step 2 command from Task 7 with `--run_set final`, without `--prepare_only` and without `--import_from_artifact_root`. Keep the default per-attempt idle gate and an idle timeout of 3,600 seconds.

Expected controller behavior: 228 imported attempts produce `skipped_success` without reaching the idle gate; only 180 missing attempts wait for idle and launch. Seeds preserve Lab2/Lab3 order for 42, Lab3/Lab2 for 43, and Lab2/Lab3 for 44.

- [ ] **Step 3: Monitor and handle failures conservatively**

Inspect each preserved failure's `validation.json`, stdout/stderr, exit status, schema, measurements, and environment. Correct only confirmed task-registration/configuration or harness compatibility defects. Never substitute a different task. Resume with the identical command and `--retry_failures` only for classified transient or fixed failures.

Do not rerun any imported success or newly successful attempt. The runner's immutable validation must reject corrupt or provenance-mismatched success directories rather than overwrite them.

- [ ] **Step 4: Reconcile final completion**

Require:

```text
manifest attempts: 408
success directories: 408
failed latest identities: 0
imported identities: 228
newly executed identities: 180
```

Check the latest history event per identity, all 180 new idle-evidence documents, exact version order, exact bounds, exact 4,096 environments, and exact RSL-RL framework. Revalidate all 408 successes against the expanded manifest and pinned provenance.

---

### Task 9: Publish and Audit the Expanded Markdown/PDF Report

**Files/artifacts:**
- Create: `/home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd-expanded-408/final/report`
- Preserve: original report and all raw success bundles

- [ ] **Step 1: Generate the report from the completed final root**

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked \
  python -m tools.benchmark_comparison.report_cli \
  --artifact_root /home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd-expanded-408 \
  --run_set final --phase measured \
  --output_dir /home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd-expanded-408/final/report
```

Expected normalized inventory:

```text
raw_runs.csv data rows: 408
failures.csv data rows: 0
paired_summary.csv data rows: 816
generated files in generated_hashes.sha256: 41
plot files: 36
plot families: 18
```

- [ ] **Step 2: Validate report semantics and visual grouping**

Verify raw rows exactly cover every manifest attempt once. Verify paired rows cover 204 logical pairs across the 12 existing summary metrics. Confirm Markdown and PDF include collection FPS, GPU mean/peak memory, GPU utilization, total startup, startup phases, pinned revisions, hardware/software, failures, and raw artifact audit.

Open representative PNGs from every category and metric family. Check that labels are readable, no task appears in the wrong category, Cartpole RGB has no training slot, seed points/error bars are visible, startup stacks match total startup, and no subplot is clipped. Inspect the PDF page sequence and extracted text; all Classic figures must precede Locomotion, which must precede Manipulation.

- [ ] **Step 3: Regenerate and prove deterministic derived output**

Run the same report command again. Require the second `generated_hashes.sha256` bytes and SHA-256 digest to equal the first. Confirm atomic publication left no staging/backup residue and raw artifact hashes were recomputed immediately before publication.

- [ ] **Step 4: Perform the complete cross-layer audit**

Reconcile:

- 408 manifest attempts;
- 228 imported attempts plus 180 newly executed attempts;
- latest runner state for all 408 identities;
- 408 valid success directories;
- 408 unique normalized rows;
- 204 logical pairs, 816 seed-aggregated paired metric rows, and 2,448 seed-metric contributions;
- header-only failures CSV;
- all raw hash entries and `raw_file_count`;
- exact 41 generated hash entries and `generated_file_count`;
- deterministic PDF metadata/title/text validation; and
- import audit source/destination hashes.

Recompute the original-root aggregate digest one last time and require it to match `original-c8d672a1dd-before.sha256`. Also require the original report's generated manifest digest, raw manifest digest, PDF digest, 228 rows, and 17-file inventory to remain unchanged.

---

### Task 10: Final Source Verification and Handoff

**Files:**
- Verify all implementation and test files from Tasks 1–5
- Do not commit operational artifacts or digest evidence

- [ ] **Step 1: Run the complete simulator-free suite after all operational fixes**

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests -v
```

Expected: all tests pass.

- [ ] **Step 2: Run pre-commit before the final source commit**

```bash
./isaaclab.sh -f
```

If it modifies files, review, stage, rerun the complete focused tests, and run `./isaaclab.sh -f` again before committing.

- [ ] **Step 3: Commit only confirmed operational fixes**

Use a focused conventional commit only if canary/final execution required source corrections. Do not amend earlier commits and do not add artifact roots, generated reports, local worktrees, or digest evidence to Git.

- [ ] **Step 4: Apply verification-before-completion and report concrete evidence**

Report branch/HEAD, test count, pre-commit result, canary/final success counts, imported/new counts, generated/raw manifest digests, PDF path/page count, expanded artifact root, original-root digest equality, and any retained failure/retry evidence. Do not claim completion until all 408 final successes and the deterministic 41-file report are present and audited.
