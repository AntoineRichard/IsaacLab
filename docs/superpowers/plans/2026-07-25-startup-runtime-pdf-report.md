<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Startup and Runtime PDF Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing report-only pipeline to normalize startup timings from the retained 228 immutable successes, visualize total and phased startup, produce a large combined Markdown/PDF report, and regenerate all derived artifacts without running a benchmark.

**Architecture:** Keep immutable success bundles as the only source of startup measurements and project their five validated phases plus a computed total into `NormalizedRun` and `raw_runs.csv`. Make CSVs the shared input for Markdown, PNG/SVG, and a new Matplotlib PDF renderer; assemble every output in the existing staging directory before one atomic publish and hash audit.

**Tech Stack:** Python 3.12, standard library CSV/JSON/dataclasses, Matplotlib Agg/PDF backends, pytest, pre-commit, existing IsaacLab benchmark artifact schemas.

## Global Constraints

- Do not launch Docker, Isaac Sim, RSL-RL training, runtime benchmarks, smoke runs, canaries, or matrix attempts.
- Do not rewrite `manifest.json`, any `success` directory, diagnostics, idle evidence, runner state, or another raw artifact.
- Consume exactly the five phases `app_launch`, `python_imports`, `task_config`, `env_creation`, and `first_step`; do not impute missing values.
- Define `startup_total_s` as the sum of those five phases and keep it distinct from controller `elapsed_time_s`.
- Keep 4,096 environments, seeds 42/43/44, task mappings, mode bounds, version ordering, and provenance unchanged.
- Add no required or optional dependency; use the Matplotlib already present in the locked Lab 3 reporting environment.
- Keep the report informational only and preserve raw CSV plus PNG/SVG as editable sources.
- Keep all derived writes staged and atomic, and include the PDF and both startup plot formats in generated hashes.
- Use PEP 8, modern Python type syntax, Google-style docstrings, the existing 2026 SPDX header, and snake_case CLI/API names.
- No changelog fragment is required: this is an experimental report-tool change, not a release deliverable.

## File Map

- Modify `tools/benchmark_comparison/normalize.py`: canonical startup fields, `NormalizedRun`, CSV projection, artifact extraction, paired summaries.
- Modify `tools/benchmark_comparison/report.py`: audit projection, startup/runtime table separation, expanded individual rows.
- Modify `tools/benchmark_comparison/plot.py`: total-startup plot and stacked phase-breakdown plot.
- Create `tools/benchmark_comparison/pdf_report.py`: deterministic paginated PDF rendering and validation.
- Modify `tools/benchmark_comparison/report_cli.py`: output ordering, audit projection, PDF generation, hashing, atomic publication.
- Modify `tools/benchmark_comparison/tests/test_normalize.py`: startup extraction, rejection, CSV, and paired-summary coverage.
- Modify `tools/benchmark_comparison/tests/test_report.py`: large combined Markdown content and non-circular audit appendix.
- Modify `tools/benchmark_comparison/tests/test_plot.py`: six plot families, stack semantics, determinism.
- Create `tools/benchmark_comparison/tests/test_pdf_report.py`: PDF content, pagination, validation, determinism.
- Modify `tools/benchmark_comparison/tests/test_report_cli_success.py`: end-to-end report/PDF artifact inventory.
- Modify `tools/benchmark_comparison/tests/test_report_cli_atomicity.py`: PDF failure rollback.
- Modify `tools/benchmark_comparison/tests/test_report_integrity.py`: audit/hash expectations.
- Modify `tools/benchmark_comparison/tests/test_actual_report_artifacts.py`: retained expanded dataset regeneration checks.
- Update test `NormalizedRun` factories in `test_report_path_safety.py` and any other constructor located by `rg "NormalizedRun\\(" tools/benchmark_comparison/tests`.

---

### Task 1: Normalize Canonical Startup Metrics

**Files:**
- Modify: `tools/benchmark_comparison/normalize.py`
- Modify: `tools/benchmark_comparison/tests/test_normalize.py`
- Modify: every test helper returned by `rg -l "NormalizedRun\\(" tools/benchmark_comparison/tests`

**Interfaces:**
- Consumes: `SemanticMetrics.phase_timings_s: dict[str, float]` from `validate_attempt_directory`.
- Produces: `STARTUP_PHASES`, six new `NormalizedRun` float attributes, six new `raw_runs.csv` fields, and six additional `SUMMARY_METRICS` names.

- [ ] **Step 1: Add failing startup extraction and round-trip tests**

Add constants in `test_normalize.py` and assert exact phase projection from the fixture:

```python
STARTUP = {
    "app_launch": 2.5,
    "python_imports": 0.2,
    "task_config": 0.4,
    "env_creation": 1.3,
    "first_step": 0.01,
}


def test_normalization_preserves_startup_components_and_computed_total(tmp_path: Path) -> None:
    attempt = _attempts().attempts[0]
    expansion = replace(_attempts(), attempts=(attempt,), pairs=())
    payloads = _payloads(attempt, collection_fps=100.0, utilization=40.0)
    payloads["schema"]["runtime"]["startup_time_s"] = STARTUP
    finalize_attempt(tmp_path, attempt, **payloads)

    runs, failures = normalize_run_set(tmp_path, expansion, _manifest())

    assert failures == ()
    assert len(runs) == 1
    run = runs[0]
    assert run.startup_app_launch_s == 2.5
    assert run.startup_python_imports_s == 0.2
    assert run.startup_task_config_s == 0.4
    assert run.startup_env_creation_s == 1.3
    assert run.startup_first_step_s == 0.01
    assert run.startup_total_s == pytest.approx(4.41)
    path = write_raw_runs_csv(tmp_path / "raw_runs.csv", runs)
    assert read_raw_runs_csv(path) == runs
```

Update every `_run()` helper to pass these stable synthetic values:

```python
startup_total_s=4.41,
startup_app_launch_s=2.5,
startup_python_imports_s=0.2,
startup_task_config_s=0.4,
startup_env_creation_s=1.3,
startup_first_step_s=0.01,
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_normalize.py::test_normalization_preserves_startup_components_and_computed_total -v
```

Expected: FAIL because `NormalizedRun` has no `startup_total_s` attribute.

- [ ] **Step 3: Add failing exact-phase validation cases**

Add:

```python
@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("missing", "startup phases do not match canonical set"),
        ("unexpected", "startup phases do not match canonical set"),
        ("negative", "startup phase app_launch must be non-negative"),
    ],
)
def test_normalization_rejects_noncanonical_startup_metrics(
    tmp_path: Path, mutation: str, reason: str
) -> None:
    attempt = _attempts().attempts[0]
    expansion = replace(_attempts(), attempts=(attempt,), pairs=())
    payloads = _payloads(attempt, collection_fps=100.0, utilization=40.0)
    startup = payloads["schema"]["runtime"]["startup_time_s"]
    if mutation == "missing":
        startup.pop("first_step")
    elif mutation == "unexpected":
        startup["other"] = 1.0
    else:
        startup["app_launch"] = -1.0
    finalize_attempt(tmp_path, attempt, **payloads)

    runs, failures = normalize_run_set(tmp_path, expansion, _manifest())

    assert runs == ()
    assert len(failures) == 1
    assert failures[0].failure_kind == "invalid_success"
    assert reason in failures[0].reason


def test_startup_metrics_rejects_non_finite_phase() -> None:
    startup = dict(STARTUP)
    startup["app_launch"] = math.inf

    with pytest.raises(ValueError, match="startup phase app_launch must be finite"):
        _startup_metrics(startup)


def test_startup_metrics_rejects_non_finite_total() -> None:
    startup = {phase: 1e308 for phase in STARTUP}

    with pytest.raises(ValueError, match="startup total must be finite"):
        _startup_metrics(startup)
```

Import `_startup_metrics` from `normalize.py` for the focused unit test. Keep
the non-finite case out of `finalize_attempt`, whose strict JSON writer rejects
non-finite JSON before normalization is reached.

- [ ] **Step 4: Implement canonical extraction and CSV fields**

In `normalize.py`, define:

```python
STARTUP_PHASES = (
    ("app_launch", "startup_app_launch_s"),
    ("python_imports", "startup_python_imports_s"),
    ("task_config", "startup_task_config_s"),
    ("env_creation", "startup_env_creation_s"),
    ("first_step", "startup_first_step_s"),
)
STARTUP_METRICS = ("startup_total_s", *(attribute for _, attribute in STARTUP_PHASES))
```

Add `STARTUP_METRICS` to `SUMMARY_METRICS`, add the six fields to
`RAW_RUN_FIELDS`, and make the six fields required `NormalizedRun` floats.
Project them in `to_csv_row()` and `_run_from_csv()`.

Add the extractor:

```python
def _startup_metrics(phase_timings_s: Mapping[str, float]) -> dict[str, float]:
    expected = {phase for phase, _ in STARTUP_PHASES}
    if set(phase_timings_s) != expected:
        raise ValueError("startup phases do not match canonical set")
    values: dict[str, float] = {}
    for phase, attribute in STARTUP_PHASES:
        value = _finite_number(phase_timings_s[phase], f"startup phase {phase}")
        if value < 0:
            raise ValueError(f"startup phase {phase} must be non-negative")
        values[attribute] = value
    total = sum(values.values())
    if not math.isfinite(total):
        raise ValueError("startup total must be finite")
    values["startup_total_s"] = total
    return values
```

Call it inside `_read_success()` before constructing `NormalizedRun`; pass
`**startup` into the existing `NormalizedRun` keyword construction immediately
before `elapsed_time_s=wall_time` and leave every existing argument unchanged.

When reading CSV, reject negative values and verify:

```python
if not math.isclose(
    run.startup_total_s,
    sum(getattr(run, attribute) for _, attribute in STARTUP_PHASES),
    rel_tol=1e-10,
    abs_tol=1e-10,
):
    raise ValueError("startup_total_s does not equal serialized startup phases")
```

- [ ] **Step 5: Run normalization tests and confirm GREEN**

Run:

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_normalize.py -v
```

Expected: all normalization tests PASS.

- [ ] **Step 6: Commit the normalization unit**

```bash
git add tools/benchmark_comparison/normalize.py tools/benchmark_comparison/tests
git commit -m "Add startup metrics to benchmark normalization"
```

---

### Task 2: Expand Paired Statistics and Markdown Reporting

**Files:**
- Modify: `tools/benchmark_comparison/report.py`
- Modify: `tools/benchmark_comparison/tests/test_report.py`
- Modify: `tools/benchmark_comparison/tests/test_report_integrity.py`

**Interfaces:**
- Consumes: the six `STARTUP_METRICS` rows in `paired_summary.csv` and startup attributes in `NormalizedRun`.
- Produces: `ReportAudit`, separate startup/runtime paired tables, expanded individual rows, and a non-circular integrity appendix.

- [ ] **Step 1: Write failing combined-report tests**

Add this public audit projection in test imports and construct it in fixtures:

```python
audit = ReportAudit(
    successful_attempts=2,
    failed_or_missing_attempts=1,
    raw_file_count=25,
    generated_file_count=17,
    raw_hash_manifest_sha256="e" * 64,
)
```

Extend the main report test with:

```python
report_path = write_markdown_report(
    normalized["raw_runs"],
    normalized["paired_summary"],
    normalized["failures"],
    tmp_path / "report" / "report.md",
    manifest=_manifest(),
    audit=audit,
)
text = report_path.read_text(encoding="utf-8")
for expected in (
    "### Startup comparison",
    "Total startup [s]",
    "App launch [s]",
    "Python imports [s]",
    "Task configuration [s]",
    "Environment creation [s]",
    "First step [s]",
    "### Runtime and resource comparison",
    "## Artifact integrity",
    "Raw files | 25",
    f"`{'e' * 64}`",
):
    assert expected in text
assert "generated hash manifest SHA" not in text
```

Assert the individual-run header contains all six startup columns and that
`4.410` appears for each synthetic run.

- [ ] **Step 2: Run the report test and confirm RED**

Run the focused test module with the Task 1 pytest command pattern.

Expected: FAIL because `ReportAudit` and startup sections do not exist.

- [ ] **Step 3: Implement audit and table partitioning**

In `report.py`, add:

```python
@dataclass(frozen=True)
class ReportAudit:
    """Non-circular integrity values rendered inside human-readable reports."""

    successful_attempts: int
    failed_or_missing_attempts: int
    raw_file_count: int
    generated_file_count: int
    raw_hash_manifest_sha256: str
```

Extend `_METRIC_LABELS` with all six startup names. Change
the `write_markdown_report` API with `audit: ReportAudit | None = None` and partition
each mode's paired rows using `metric in STARTUP_METRICS`.

Emit these headings in order:

```python
lines.extend(["", "### Startup comparison", ""])
lines.extend(_paired_summary_table(startup_summaries))
lines.extend(["", "### Runtime and resource comparison", ""])
lines.extend(_paired_summary_table(runtime_summaries))
```

Extract the existing paired table formatting into
`_paired_summary_table(rows: Sequence[Mapping[str, str]]) -> list[str]` so the
delta semantics remain identical.

Expand the individual table with startup total and phases before FPS. Append
the integrity section only when `audit is not None`; render counts and the raw
manifest digest, but never the generated manifest digest.

- [ ] **Step 4: Verify summary row counts and Markdown tests**

Add to `test_normalize.py`; the Lab 2 helper retains the Task 1 default startup
of 4.41 seconds and the Lab 3 row uses a five-second startup:

```python
lab3 = _run(
    version="lab3",
    version_sha=LAB3_SHA,
    environment_identity=f"uv-lock:{LAB3_LOCK}",
    concrete_task="Isaac-Cartpole",
    startup_total_s=5.0,
    startup_app_launch_s=3.0,
    startup_python_imports_s=0.25,
    startup_task_config_s=0.5,
    startup_env_creation_s=1.2,
    startup_first_step_s=0.05,
)
summaries = summarize_pairs((_run(), lab3))
assert {row.metric for row in summaries} == set(SUMMARY_METRICS)
startup = next(row for row in summaries if row.metric == "startup_total_s")
assert startup.absolute_delta == pytest.approx(0.59)
```

Run:

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_normalize.py \
  tools/benchmark_comparison/tests/test_report.py \
  tools/benchmark_comparison/tests/test_report_integrity.py -v
```

Expected: all selected tests PASS.

- [ ] **Step 5: Commit the combined Markdown unit**

```bash
git add tools/benchmark_comparison/report.py tools/benchmark_comparison/tests/test_normalize.py \
  tools/benchmark_comparison/tests/test_report.py tools/benchmark_comparison/tests/test_report_integrity.py
git commit -m "Add startup analysis to benchmark report"
```

---

### Task 3: Generate Total and Stacked Startup Plots

**Files:**
- Modify: `tools/benchmark_comparison/plot.py`
- Modify: `tools/benchmark_comparison/tests/test_plot.py`

**Interfaces:**
- Consumes: normalized startup columns from `raw_runs.csv`.
- Produces: `startup_total_s.{png,svg}`, `startup_phase_breakdown.{png,svg}`, and an expanded `PLOT_BASENAMES` with six names.

- [ ] **Step 1: Write failing plot-inventory and stack tests**

Update the test factory with Task 1 startup fields and change expected names:

```python
assert PLOT_BASENAMES == (
    "collection_fps",
    "gpu_memory_mean_mib",
    "gpu_memory_peak_mib",
    "gpu_utilization_mean_pct",
    "startup_total_s",
    "startup_phase_breakdown",
)
```

Add:

```python
def test_startup_phase_means_sum_to_total_bar_height() -> None:
    runs = (
        replace(
            _run("cartpole", "runtime-100", 42, "lab2", 100.0),
            startup_total_s=15.0,
            startup_app_launch_s=1.0,
            startup_python_imports_s=2.0,
            startup_task_config_s=3.0,
            startup_env_creation_s=4.0,
            startup_first_step_s=5.0,
        ),
        replace(
            _run("cartpole", "runtime-100", 43, "lab2", 110.0),
            startup_total_s=20.0,
            startup_app_launch_s=2.0,
            startup_python_imports_s=3.0,
            startup_task_config_s=4.0,
            startup_env_creation_s=5.0,
            startup_first_step_s=6.0,
        ),
    )
    phase_means = _startup_phase_means(runs, "runtime-100", "cartpole", "lab2")
    assert tuple(phase_means) == tuple(attribute for _, attribute in STARTUP_PHASES)
    assert sum(phase_means.values()) == pytest.approx(
        statistics.fmean(run.startup_total_s for run in runs if run.version == "lab2")
    )
```

Import `statistics`, `replace`, `pytest`, `STARTUP_PHASES`, and
`_startup_phase_means` in the test module.

Assert every generated SVG is deterministic, the total SVG contains
`Total startup time [s]`, and the breakdown SVG contains all five human phase
labels plus `Isaac Lab 2` and `Isaac Lab 3`.

- [ ] **Step 2: Run plot tests and confirm RED**

Run `test_plot.py` with the Task 1 pytest command pattern.

Expected: FAIL because both startup basenames and stack helper are missing.

- [ ] **Step 3: Add total-startup plot and phase constants**

Add to `PLOT_METRICS`:

```python
"startup_total_s": ("startup_total_s", "Total Startup Time", "Total startup time [s]"),
```

Define fixed phase colors/labels:

```python
_PHASE_COLORS = {
    "startup_app_launch_s": "#4C78A8",
    "startup_python_imports_s": "#72B7B2",
    "startup_task_config_s": "#F2CF5B",
    "startup_env_creation_s": "#F58518",
    "startup_first_step_s": "#E45756",
}
```

- [ ] **Step 4: Implement stacked breakdown rendering**

Add:

```python
def _startup_phase_means(
    runs: Sequence[NormalizedRun], mode: str, task: str, version: str
) -> dict[str, float]:
    selected = tuple(
        run for run in runs if run.mode == mode and run.logical_task == task and run.version == version
    )
    return {
        attribute: statistics.fmean(getattr(run, attribute) for run in selected)
        for _, attribute in STARTUP_PHASES
    }
```

Implement `_generate_startup_phase_breakdown(plt, matplotlib, runs,
output_directory, mode_order, expansion) -> tuple[Path, Path]`. Reuse the
existing task order, 45-degree labels, fixed 1800x1000 dimensions, temporary
files, metadata, and `os.replace`. Draw paired bars with `bottom` accumulated
in canonical phase order. Use hatch `/` for Lab 2 and `\\` for Lab 3 while
phase colors identify components.

Append both returned paths to `generate_plots()` and define
`PLOT_BASENAMES = (*tuple(PLOT_METRICS), "startup_phase_breakdown")`.

- [ ] **Step 5: Run plot tests and visually inspect synthetic outputs**

Run `test_plot.py`; expect PASS. Open both synthetic PNGs with the local image
viewer and confirm labels, phase legend, version hatch legend, and stacks are
readable.

- [ ] **Step 6: Commit the startup plots**

```bash
git add tools/benchmark_comparison/plot.py tools/benchmark_comparison/tests/test_plot.py
git commit -m "Plot benchmark startup timings"
```

---

### Task 4: Add the Deterministic Large PDF Renderer

**Files:**
- Create: `tools/benchmark_comparison/pdf_report.py`
- Create: `tools/benchmark_comparison/tests/test_pdf_report.py`

**Interfaces:**
- Consumes: `raw_runs.csv`, `paired_summary.csv`, `failures.csv`, `RunSetManifest`, `ReportAudit`, and the six PNG plot paths.
- Produces: a `write_pdf_report` function returning `Path`, an atomically written `report.pdf`, and `validate_pdf(path, expected_tokens) -> None`.

- [ ] **Step 1: Write a failing deterministic PDF test**

Create `test_pdf_report.py` with synthetic paired runs from the report test,
normalized CSVs, six minimal plot PNGs, and:

```python
def test_pdf_contains_large_report_and_regenerates_byte_identically(tmp_path: Path) -> None:
    normalized, manifest, audit, plots = _inputs(tmp_path)
    first = write_pdf_report(
        normalized["raw_runs"],
        normalized["paired_summary"],
        normalized["failures"],
        plots,
        tmp_path / "first.pdf",
        manifest=manifest,
        audit=audit,
    )
    second = write_pdf_report(
        normalized["raw_runs"],
        normalized["paired_summary"],
        normalized["failures"],
        plots,
        tmp_path / "second.pdf",
        manifest=manifest,
        audit=audit,
    )
    assert first.read_bytes().startswith(b"%PDF-")
    assert first.stat().st_size > 10_000
    assert first.read_bytes() == second.read_bytes()
    validate_pdf(first, ("final", "a" * 40, "b" * 40, "Startup"))
```

Add a second test with 40 synthetic runs and assert `pdfinfo` reports more
than one page and `pdftotext` contains the first and last attempt identities.

- [ ] **Step 2: Run the PDF tests and confirm RED**

Run `test_pdf_report.py` with the Task 1 pytest command pattern.

Expected: collection ERROR because `tools.benchmark_comparison.pdf_report`
does not exist.

- [ ] **Step 3: Implement fixed PDF infrastructure**

Create `pdf_report.py` with the 2026 header and:

```python
def write_pdf_report(
    raw_runs_path: Path,
    paired_summary_path: Path,
    failures_path: Path,
    plot_paths: Sequence[Path],
    output_path: Path,
    *,
    manifest: RunSetManifest,
    audit: ReportAudit,
) -> Path:
    """Write the deterministic paginated benchmark report PDF."""
```

Use `matplotlib.backends.backend_pdf.PdfPages` and fixed metadata:

```python
_PDF_METADATA = {
    "Title": "Isaac Lab Startup and Runtime Benchmark Report",
    "Author": "The Isaac Lab Project Developers",
    "Creator": "Isaac Lab benchmark comparison",
    "Producer": "Isaac Lab benchmark comparison",
    "CreationDate": datetime(2026, 7, 25, tzinfo=timezone.utc),
    "ModDate": datetime(2026, 7, 25, tzinfo=timezone.utc),
}
```

Write to `output_path.with_suffix(".pdf.tmp")`, then validate and replace.
Use fixed US-letter landscape pages for tables and portrait pages for title,
methodology, and audit.

- [ ] **Step 4: Implement reusable page and table functions**

Implement these private units with the following exact responsibilities:

```python
def _text_page(pdf: PdfPages, title: str, lines: Sequence[str]) -> None:
    """Render a portrait page with a title and wrapped text lines."""


def _table_pages(
    pdf: PdfPages,
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    rows_per_page: int,
) -> None:
    """Render deterministic landscape table pages with repeated headers."""


def _plot_page(pdf: PdfPages, title: str, path: Path) -> None:
    """Render one PNG on a full landscape page without resampling it."""


def _attempt_identity_from_artifact(path: str) -> str:
    """Return the attempt directory name immediately preceding the artifact leaf."""
```

The implementation bodies must create and close one Matplotlib figure per
page. `_table_pages` slices rows in fixed `rows_per_page` chunks and emits one
header-only page when there are no rows. `_attempt_identity_from_artifact`
uses `PurePosixPath(path).parts[-2]` after requiring at least two components.

Render pages in this fixed order: cover/methodology, pins/inventory, task
mapping, three mode startup tables, three mode runtime/resource tables,
individual-run appendix, failures/audit, then the six PNG figures in
`PLOT_BASENAMES` order. Repeat table headers on every page.

- [ ] **Step 5: Implement PDF validation without a Python dependency**

Implement:

```python
def validate_pdf(path: Path, expected_tokens: Sequence[str]) -> None:
    if not path.read_bytes().startswith(b"%PDF-"):
        raise ValueError("report PDF header is invalid")
    for executable in ("pdfinfo", "pdftotext"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"report PDF validation requires {executable}")
    info = subprocess.run(["pdfinfo", str(path)], check=True, text=True, capture_output=True).stdout
    if not re.search(r"^Pages:\s+[1-9][0-9]*$", info, re.MULTILINE):
        raise ValueError("report PDF has no pages")
    text = subprocess.run(
        ["pdftotext", str(path), "-"], check=True, text=True, capture_output=True
    ).stdout
    missing = [token for token in expected_tokens if token not in text]
    if missing:
        raise ValueError(f"report PDF is missing expected text: {missing}")
```

- [ ] **Step 6: Run PDF tests and confirm GREEN**

Run `test_pdf_report.py`; expect all tests PASS and no warnings.

- [ ] **Step 7: Commit the PDF renderer**

```bash
git add tools/benchmark_comparison/pdf_report.py tools/benchmark_comparison/tests/test_pdf_report.py
git commit -m "Generate benchmark PDF report"
```

---

### Task 5: Integrate Startup and PDF Outputs Atomically

**Files:**
- Modify: `tools/benchmark_comparison/report_cli.py`
- Modify: `tools/benchmark_comparison/tests/test_report_cli_success.py`
- Modify: `tools/benchmark_comparison/tests/test_report_cli_atomicity.py`
- Modify: `tools/benchmark_comparison/tests/test_report_integrity.py`

**Interfaces:**
- Consumes: `generate_plots()`, `ReportAudit`, and `write_pdf_report()`.
- Produces: one staged 17-file generated set, updated hash/audit manifests, and unchanged atomic publication semantics.

- [ ] **Step 1: Write failing end-to-end inventory assertions**

Extend `test_report_cli_success.py`:

```python
expected = {
    "raw_runs.csv",
    "paired_summary.csv",
    "failures.csv",
    "report.md",
    "report.pdf",
    "collection_fps.png",
    "collection_fps.svg",
    "gpu_memory_mean_mib.png",
    "gpu_memory_mean_mib.svg",
    "gpu_memory_peak_mib.png",
    "gpu_memory_peak_mib.svg",
    "gpu_utilization_mean_pct.png",
    "gpu_utilization_mean_pct.svg",
    "startup_total_s.png",
    "startup_total_s.svg",
    "startup_phase_breakdown.png",
    "startup_phase_breakdown.svg",
}
assert expected <= {path.name for path in output.iterdir()}
audit = json.loads((output / "audit_summary.json").read_text(encoding="utf-8"))
assert audit["generated_file_count"] == 17
```

Assert startup CSV values equal the fixture and validate the PDF expected
tokens.

- [ ] **Step 2: Add a failing PDF rollback test**

In `test_report_cli_atomicity.py`, create an existing report directory, patch
`write_pdf_report` to raise `RuntimeError("injected PDF failure")`, invoke
`main()`, and assert the old report bytes and stale sentinel remain unchanged.

- [ ] **Step 3: Run both tests and confirm RED**

Run `test_report_cli_success.py` and `test_report_cli_atomicity.py`.

Expected: inventory assertion FAIL and PDF rollback patch target is missing.

- [ ] **Step 4: Reorder staged report assembly**

In `report_cli.py`, import `write_pdf_report` and `ReportAudit`. After
normalization:

```python
normalized = write_normalized_outputs(staging, runs, failures, expansion=expansion)
plots = generate_plots(normalized["raw_runs"], staging, expansion=expansion)
generated_file_count = len(_NORMALIZED_FILES) + len(plots) + 2  # Markdown + PDF
report_audit = ReportAudit(
    successful_attempts=len(runs),
    failed_or_missing_attempts=len(failures),
    raw_file_count=raw_file_count,
    generated_file_count=generated_file_count,
    raw_hash_manifest_sha256=hashlib.sha256(raw_hash_contents.encode()).hexdigest(),
)
write_markdown_report(
    normalized["raw_runs"],
    normalized["paired_summary"],
    normalized["failures"],
    staging / "report.md",
    manifest=manifest,
    artifact_root=artifact_root,
    audit=report_audit,
)
write_pdf_report(
    normalized["raw_runs"],
    normalized["paired_summary"],
    normalized["failures"],
    tuple(path for path in plots if path.suffix == ".png"),
    staging / "report.pdf",
    manifest=manifest,
    audit=report_audit,
)
```

Add `report.pdf` to `generated`, assert `len(generated) ==
generated_file_count`, and use `report_audit` values to populate the external
`audit_summary.json`. Keep raw rehash and `_publish()` after every derived
file succeeds.

- [ ] **Step 5: Verify CLI integration and hash manifests**

Run:

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_report_cli_success.py \
  tools/benchmark_comparison/tests/test_report_cli_atomicity.py \
  tools/benchmark_comparison/tests/test_report_integrity.py -v
```

Expected: all selected tests PASS; generated hashes validate all 17 files.

- [ ] **Step 6: Commit atomic integration**

```bash
git add tools/benchmark_comparison/report_cli.py \
  tools/benchmark_comparison/tests/test_report_cli_success.py \
  tools/benchmark_comparison/tests/test_report_cli_atomicity.py \
  tools/benchmark_comparison/tests/test_report_integrity.py
git commit -m "Integrate startup PDF report outputs"
```

---

### Task 6: Regenerate and Audit the Retained Expanded Report

**Files:**
- Modify: `tools/benchmark_comparison/tests/test_actual_report_artifacts.py`
- Regenerate outside Git: `/home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd/final/report/`

**Interfaces:**
- Consumes: the existing artifact root `c8d672a1dd`, final manifest, 228 success directories, and the completed report pipeline.
- Produces: retained expanded-dataset coverage plus the final Markdown, PDF, CSVs, PNG/SVG plots, audit summary, and hash manifests.

- [ ] **Step 1: Update the retained-data regression test**

Replace the previous canary/final parameterization with one retained-final
test and point `_ROOT` at:

```python
_ROOT = Path("/home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd")
```

Use manifest expansion rather than old hard-coded matrix counts:

```python
expansion = resolve_manifest_expansion(manifest, _ROOT)
assert len(tuple(csv.DictReader((report / "raw_runs.csv").open()))) == len(expansion.attempts)
assert audit["successful_attempts"] == len(expansion.attempts)
assert audit["failed_or_missing_attempts"] == 0
assert audit["generated_file_count"] == 17
assert (report / "report.pdf").is_file()
```

Assert the final expansion is 228 and the CSV contains all six startup
columns. Verify both hash manifests.

- [ ] **Step 2: Run the retained-data test before regeneration and confirm RED**

Run only the retained final report test with the standard pytest command.

Expected: FAIL because the published report lacks startup columns and PDF.

- [ ] **Step 3: Regenerate the canonical report without executors**

Run exactly:

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked \
  python -m tools.benchmark_comparison.report_cli \
  --artifact_root /home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd \
  --run_set final --phase measured \
  --output_dir /home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd/final/report
```

Run from `/home/antoiner/benchmarks/isaaclab2-vs-3/lab2-main` so the current
harness module is importable. Confirm the process tree never contains Docker,
Isaac Sim, runtime.py, training.py, or RSL-RL.

- [ ] **Step 4: Run retained-data test and confirm GREEN**

Run the updated test. Expected: PASS with 228 successes, zero failures, 17
generated files, six startup columns, and a valid PDF.

- [ ] **Step 5: Verify deterministic regeneration**

Hash `generated_hashes.sha256`, run the exact report command a second time,
and hash it again. Expected: identical SHA-256 values.

Then run:

```bash
(cd /home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd/final/report && \
  sha256sum --quiet -c generated_hashes.sha256)
(cd /home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd && \
  sha256sum --quiet -c final/report/raw_artifact_hashes.sha256)
pdfinfo /home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd/final/report/report.pdf
pdftotext /home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd/final/report/report.pdf - | \
  rg "Isaac Lab Startup and Runtime|c8d672a1dd|cb508381fb|Artifact integrity"
```

Expected: both checksum commands exit 0, `pdfinfo` reports at least one page,
and all four text tokens are found.

- [ ] **Step 6: Visually inspect startup outputs**

Inspect `startup_total_s.png`, `startup_phase_breakdown.png`, and representative
PDF pages. Confirm readable task labels, correct legends, complete tables,
non-clipped content, and visible Lab 2/Lab 3 distinctions.

- [ ] **Step 7: Commit retained-data regression coverage**

```bash
git add tools/benchmark_comparison/tests/test_actual_report_artifacts.py
git commit -m "Verify expanded benchmark report artifacts"
```

---

### Task 7: Final Suite, Formatting, and Handoff

**Files:**
- Verify all modified files and external report artifacts.

**Interfaces:**
- Consumes: Tasks 1-6.
- Produces: clean experimental harness branch and verified final report handoff.

- [ ] **Step 1: Run the full simulator-free comparison suite**

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests
```

Expected: all tests PASS with zero failures or warnings.

- [ ] **Step 2: Run all repository pre-commit hooks**

The isolated Lab 2 worktree has no local Isaac Sim Python, so use the working
pre-commit entry point:

```bash
uvx pre-commit run --all-files
```

Expected: all hooks PASS. If a hook modifies files, inspect the diff, stage the
changes, rerun the focused tests, and rerun pre-commit until clean.

- [ ] **Step 3: Verify source and environment integrity**

```bash
git status --short
git diff --check
git -C /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop status --short
sha256sum /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop/uv.lock
```

Expected: both worktrees are clean; the lock hash remains
`911857c6da5cb0b07b96222d54894fd2a90941c142a4c178ee1de729de035e18`.

- [ ] **Step 4: Verify final matrix and report counts**

Confirm runner status `completed`, history `success=228`, exactly 228 success
directories, audit `successful_attempts=228`, audit failures `0`, raw CSV 228
data rows, paired CSV `38 × len(SUMMARY_METRICS)` data rows, and header-only
failures CSV.

- [ ] **Step 5: Hand off links and findings**

Report the final commit SHA and link `report.md`, `report.pdf`, both startup
plots, `paired_summary.csv`, `raw_runs.csv`, and the raw artifact root. State
explicitly that no benchmark was rerun and no raw artifact changed.
