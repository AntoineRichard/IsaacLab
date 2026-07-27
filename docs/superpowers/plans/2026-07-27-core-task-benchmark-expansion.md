# Core Task Benchmark Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task by task.

**Goal:** Add the 16 approved cross-version Core task matches, execute only
their missing benchmark attempts, and publish an expanded Markdown/PDF report
that incorporates the validated runtime-regression investigation.

**Architecture:** Preserve the existing matrix-driven runner and immutable
artifact model. Resolve an optional environment count per task during matrix
expansion, import the old 408 final and 136 canary successes into a fresh
superset artifact root, and let the ordinary resume logic execute only new
identities. Treat the prior runtime investigation as checksummed raw input;
load it through a strict path/checksum validator and render selected content
through the existing deterministic Markdown/PDF report pipeline.

**Tech Stack:** Python 3.12, dataclasses, TOML, standard-library hashing and
path validation, pytest, Matplotlib/PdfPages, Docker for Isaac Lab 2.3.2,
`uv` for Isaac Lab 3.0.

## Global constraints

- Work only in
  `/home/antoiner/benchmarks/isaaclab2-vs-3/lab2-main`.
- Use `gpt-5.6-sol` with medium reasoning for implementation subagents.
- Do not modify the original
  `artifacts/c8d672a1dd-expanded-408` artifact tree.
- Do not add SKRL, Play aliases, deprecated aliases, or non-equivalent task
  pairs.
- Do not add training attempts for the four runtime-only pairs.
- Preserve every existing attempt identity byte-for-byte.
- Keep the generated report inventory at 57 files.
- Run only focused benchmark-harness tests and pre-commit; do not run the full
  Isaac Lab simulator or repository test suites.
- Final measurements must pass the existing machine-idle gate. Canary probes
  may run immediately.

---

### Task 1: Expand the matrix and resolve per-task environment counts

**Files:**

- Modify: `tools/benchmark_comparison/models.py`
- Modify: `tools/benchmark_comparison/matrix.toml`
- Modify: `tools/benchmark_comparison/matrix.py`
- Modify: `tools/benchmark_comparison/executors.py`
- Modify: `tools/benchmark_comparison/tests/test_matrix.py`
- Modify: `tools/benchmark_comparison/tests/test_executors.py`
- Modify when required by the new count contract:
  `tools/benchmark_comparison/tests/test_import_results.py`
- Modify when required by the new count contract:
  `tools/benchmark_comparison/tests/test_runner.py`

**Step 1: Add failing matrix tests**

Cover:

- Exactly 39 configured tasks.
- Exactly 336 final pairs/672 attempts.
- Exactly 112 canary pairs/224 attempts.
- The 12 added trainable tasks expose all three modes.
- The four runtime-only tasks expose only `runtime_100` and `runtime_1000`.
- `cartpole_camera_direct`, `shadow_camera_direct`, and
  `shadow_handover_direct` resolve to 512, 1225, and 2048 environments.
- Every other task resolves to the matrix default of 4096.
- The complete old 408-attempt identity set is an exact subset of the
  expanded identity set.

Run:

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_matrix.py -v
```

Expected: failures for missing tasks, old counts, and absent per-task
environment resolution.

**Step 2: Add the model and TOML data**

Add `num_envs: int | None = None` to `BenchmarkTask` and a small
`resolved_num_envs(default: int) -> int` helper that rejects non-positive
values. Append the 16 approved task entries in the order specified by the
design document. Set:

- `supported_modes = ["runtime_100", "runtime_1000"]` on the four
  runtime-only tasks.
- `enable_cameras = true`, `lab3_presets = ["rgb"]`, and `num_envs = 512`
  for `cartpole_camera_direct`.
- `enable_cameras = true` and `num_envs = 1225` for
  `shadow_camera_direct`.
- `num_envs = 2048` for `shadow_handover_direct`.

Do not repeat the default 4096 in TOML.

**Step 3: Expand and validate with resolved values**

Update the canonical task tuple and expansion constants. Resolve `num_envs`
once per task, pass it to `BenchmarkAttempt`, and include it in new attempt
identities. Existing identities must remain unchanged because their resolved
value remains 4096. Validate the allowed task capability overrides and reject
invalid/non-positive task-level values.

**Step 4: Make registration probes task-aware**

Pass each task's resolved environment count into both Lab 2 and Lab 3
registration probes. Replace the camera probe's assumption that every camera
is named `scene.tiled_camera` with a narrow inspection that validates the
configured camera sensor(s) for both approved camera task families without
weakening renderer/data-type checks.

Add executor tests for:

- 512-, 1225-, and 2048-environment commands.
- Matching registration-probe environment counts.
- Camera flags and Lab 3 RGB preset.
- Both camera configuration shapes.

Run:

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_matrix.py \
  tools/benchmark_comparison/tests/test_executors.py \
  tools/benchmark_comparison/tests/test_import_results.py \
  tools/benchmark_comparison/tests/test_runner.py -v
```

Expected: all pass.

**Step 5: Commit the focused change**

```bash
git add tools/benchmark_comparison/models.py \
  tools/benchmark_comparison/matrix.toml \
  tools/benchmark_comparison/matrix.py \
  tools/benchmark_comparison/executors.py \
  tools/benchmark_comparison/tests/test_matrix.py \
  tools/benchmark_comparison/tests/test_executors.py \
  tools/benchmark_comparison/tests/test_import_results.py \
  tools/benchmark_comparison/tests/test_runner.py
git commit -m "Expand benchmark Core task matrix"
```

---

### Task 2: Add a strict runtime-investigation loader

**Files:**

- Create: `tools/benchmark_comparison/runtime_investigation.py`
- Create: `tools/benchmark_comparison/tests/test_runtime_investigation.py`

**Step 1: Write failing loader tests**

Build minimal temporary bundles and cover:

- Valid exact-tree SHA-256 verification.
- Required `REPORT.md`, `REPORT.pdf`, and `SHA256SUMS`.
- Rejection of missing, modified, duplicate, absolute, escaping, symlink,
  special, and unlisted paths.
- Rejection when the selected directory is outside the artifact root or run
  set.
- Validation of the pinned Lab 2/Lab 3 revisions and original 23-task,
  408-attempt snapshot metadata.
- Stable ordered discovery of selected PNG evidence and linked raw files.

Run:

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_runtime_investigation.py -v
```

Expected: import failure because the loader does not exist.

**Step 2: Implement the read-only loader**

Create frozen typed records for validated investigation metadata and assets.
Resolve the caller-provided relative path beneath the selected run set,
inspect with `lstat`, parse `SHA256SUMS` without shelling out, and compare the
manifest against the exact regular-file tree. Read only validated UTF-8
Markdown/JSON inputs and expose validated paths; never import or execute
bundle code.

Use the old manifest metadata as the authoritative source for revisions and
the investigation's own summary for the historical 23-task/408-attempt
scope. Error messages must name the rejected invariant and path.

**Step 3: Run and commit**

Re-run the focused test above, then:

```bash
git add tools/benchmark_comparison/runtime_investigation.py \
  tools/benchmark_comparison/tests/test_runtime_investigation.py
git commit -m "Validate runtime investigation artifacts"
```

---

### Task 3: Integrate the diagnosis into Markdown, PDF, and report CLI

**Files:**

- Modify: `tools/benchmark_comparison/report.py`
- Modify: `tools/benchmark_comparison/pdf_report.py`
- Modify: `tools/benchmark_comparison/report_cli.py`
- Modify: `tools/benchmark_comparison/tests/test_report.py`
- Modify: `tools/benchmark_comparison/tests/test_pdf_report.py`
- Modify: `tools/benchmark_comparison/tests/test_report_cli.py`
- Modify: `tools/benchmark_comparison/tests/test_report_cli_success.py`
- Modify: `tools/benchmark_comparison/tests/test_report_cli_atomicity.py`
- Modify: `tools/benchmark_comparison/tests/test_report_integrity.py`
- Modify: `tools/benchmark_comparison/tests/test_report_path_safety.py`

**Step 1: Add failing report tests**

Require an explicit CLI option:

```text
--runtime_investigation final/runtime_investigation_2026-07-26
```

Cover:

- Diagnosis appears after aggregate results and before category detail.
- It is clearly labeled as analysis of the original 23-task/408-attempt
  snapshot.
- Markdown includes the PhysX solver, indexed target conversion, workload
  dependence, startup/Digit warning, Kuka equivalence caveat, rejected
  hypotheses, confidence limits, and actionable fix-target table.
- Markdown links point only to validated relative bundle assets.
- PDF contains the same narrative headings and selected validated PNGs.
- Generated output remains exactly 57 files.
- Diagnosis raw files are included in raw-input hashing, not generated-output
  inventory.
- Any loader, Markdown, image, or PDF error leaves the old report untouched.

Run:

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_report.py \
  tools/benchmark_comparison/tests/test_pdf_report.py \
  tools/benchmark_comparison/tests/test_report_cli.py \
  tools/benchmark_comparison/tests/test_report_cli_success.py \
  tools/benchmark_comparison/tests/test_report_cli_atomicity.py \
  tools/benchmark_comparison/tests/test_report_integrity.py \
  tools/benchmark_comparison/tests/test_report_path_safety.py -v
```

Expected: failures for the absent CLI/data flow and diagnosis section.

**Step 2: Thread validated investigation data through reporting**

Load the bundle before report staging. Pass one immutable validated object to
the Markdown and PDF renderers. Keep normal report generation valid when the
option is omitted so canary reporting and unit fixtures remain lightweight.
When the option is present, validation failure must happen before replacing
the published report directory.

Render concise findings and an actionable fix table in the main report while
linking the detailed retained evidence. Reuse only selected validated PNGs in
the PDF and preserve canonical plot ordering/inventory.

**Step 3: Run and commit**

Re-run the focused report tests above, then:

```bash
git add tools/benchmark_comparison/report.py \
  tools/benchmark_comparison/pdf_report.py \
  tools/benchmark_comparison/report_cli.py \
  tools/benchmark_comparison/tests/test_report.py \
  tools/benchmark_comparison/tests/test_pdf_report.py \
  tools/benchmark_comparison/tests/test_report_cli.py \
  tools/benchmark_comparison/tests/test_report_cli_success.py \
  tools/benchmark_comparison/tests/test_report_cli_atomicity.py \
  tools/benchmark_comparison/tests/test_report_integrity.py \
  tools/benchmark_comparison/tests/test_report_path_safety.py
git commit -m "Integrate runtime diagnosis into reports"
```

---

### Task 4: Focused harness verification and review

**Files:**

- Review all files changed in Tasks 1-3.

**Step 1: Run the benchmark-harness test directory**

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests -q
```

Expected: all pass.

**Step 2: Run pre-commit twice if it rewrites files**

Attempt the repository command first:

```bash
./isaaclab.sh -f
```

If host Isaac Sim Python is unavailable, use the established tooling-only
fallback:

```bash
uvx pre-commit run --all-files
```

Review any rewrite, stage it, and re-run until clean.

**Step 3: Independent reviews**

Run a specification-compliance review against the approved design, then a
code-quality review. Fix only concrete findings and re-run the affected
focused tests. Confirm `git diff --check` and a clean worktree.

---

### Task 5: Prepare the fresh root and execute new canaries only

**Inputs:**

- Old root:
  `/home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd-expanded-408`
- New root:
  `/home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd-core-expanded-672`

**Step 1: Transactionally import old canary successes**

Use the benchmark CLI's existing `--import_from_artifact_root` and
`--prepare_only` flow. Verify:

- 136 imported attempts.
- Every imported checksum and semantic identity validates.
- Exactly 88 canary attempts remain missing.

Do not manually copy or merge result directories.

**Step 2: Run the ordinary canary matrix**

Start the normal canary command against the new root. Verify the manifest
records 136 `skipped_success` decisions and invokes executors for only 88 new
attempts. Diagnose only failed new task pairs. Retain all failure artifacts
and do not begin final measurements until the new canary set succeeds or a
pair is removed as non-equivalent with documented evidence.

---

### Task 6: Execute new final attempts and publish the expanded report

**Step 1: Import and audit final successes**

Prepare the final run set from the old root and verify:

- 408 imported final successes.
- Exactly 264 final attempts remain missing.
- The original artifact tree is unchanged.

**Step 2: Stage and validate the diagnosis bundle**

Copy the existing investigation tree into:

```text
final/runtime_investigation_2026-07-26/
```

Use a metadata-preserving copy into the fresh root, then validate the exact
tree through the new loader and its internal `SHA256SUMS` before any report
publication.

**Step 3: Run final measurements**

Run the ordinary final matrix. Let the idle gate wait until the machine is
free. Verify 408 imported successes are skipped and only 264 new attempts are
executed. Do not bypass the idle gate.

**Step 4: Generate and audit the report**

Invoke report generation with the explicit investigation option. Verify:

- 672/672 successful attempts and 336/336 complete pairs.
- 39 logical tasks and four stable categories.
- 57 generated report files: 3 CSV, Markdown, PDF, 26 PNG, and 26 SVG.
- The main Markdown and PDF contain the runtime diagnosis and actionable fix
  targets.
- Raw hashes include the diagnosis bundle.
- Internal and outer SHA-256 manifests validate.
- The original 408-attempt root remains byte-for-byte unchanged.

Record final test output, manifest digest, report/PDF digests and page count,
attempt counts, import/skip counts, and artifact/report paths in the handoff.
