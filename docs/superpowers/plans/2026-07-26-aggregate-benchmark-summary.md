<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Aggregate Benchmark Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the table-heavy benchmark-report opening with median and arithmetic-mean task-delta heatmaps for Classic, Locomotion Flat, Locomotion Rough, and Manipulation, while preserving every raw artifact and moving all audit tables behind the detailed figures.

**Architecture:** Keep normalization and immutable benchmark artifacts unchanged. Split report grouping metadata into four explicit categories, derive a typed ordered aggregate sequence from validated `paired_summary.csv`, and give both renderers the same aggregate values and plot-order constants. Generate two aggregate heatmap families plus 24 detailed figure families, then atomically publish the 57-file report inventory and regenerate it twice from the completed 408-attempt root.

**Tech Stack:** Python 3.12, standard-library dataclasses/enums/CSV/statistics/math/filesystem APIs, Matplotlib Agg/PDF backends, pytest, Poppler `pdfinfo`/`pdftotext`, locked IsaacLab 3 `uv` environment, and pre-commit.

## Global Constraints

- Work only in `/home/antoiner/benchmarks/isaaclab2-vs-3/lab2-main` on `antoiner/backport-benchmark-harness`.
- This is derived-report work. Do not start Isaac Sim, Docker, RSL-RL, a canary, or a measured benchmark attempt.
- Do not modify any file below the immutable attempt directories in `/home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd-expanded-408/final/`.
- Keep all 408 attempt identities, 204 logical pairs, 23 task mappings, 4,096 environments, seeds 42/43/44, modes, bounds, version order, and provenance exact.
- Keep `raw_runs.csv`, `paired_summary.csv`, and `failures.csv` schemas unchanged. The aggregation layer consumes `paired_summary.csv`; it does not add a fourth normalized CSV.
- Keep `cartpole_rgb_kit` runtime-only. Never create or impute a `training-100` value for it.
- Use equal logical-task weighting. Aggregate `(lab3_mean - lab2_mean) / lab2_mean * 100` values, not raw measurements, pooled seeds, absolute deltas, or weighted metrics.
- Exclude only rows with a zero Lab 2 baseline from that cell. Record `task_count=0` and `value=None` for an empty cell; render it as `N/A`.
- Keep mode order `runtime-100`, `runtime-1000`, `training-100`; metric order collection FPS, total startup, mean GPU memory, peak GPU memory, mean GPU utilization; and group order Classic, Locomotion Flat, Locomotion Rough, Manipulation.
- Heatmap color is descriptive only: use one symmetric range centered at zero and no pass/fail, regression/improvement, good/bad, or threshold language.
- Generated inventory must be exactly 57 hashed files: three normalized CSVs, Markdown, PDF, 48 detailed PNG/SVG files, and four aggregate PNG/SVG files. Metadata files remain outside the generated hash inventory.
- Preserve the completed root's 4,271-entry raw hash inventory byte-for-byte. Preserve `/home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd` and the original repository root byte-for-byte.
- Do not edit `.orig` snapshots, `CHANGELOG.rst`, `config/extension.toml`, or package changelog fragments. This experimental tools/docs change has no public package release entry.
- New Python files use the 2026 SPDX header, modern type hints, PEP 8, and Google-style docstrings.
- Run pre-commit before every commit that closes an implementation unit. Because this worktree's wrapper cannot run with `TERM=dumb`, use the established equivalent `uvx pre-commit run --all-files`, review any edits, stage them, and rerun it before committing.

## File Map

- Modify `tools/benchmark_comparison/models.py`: replace the combined Locomotion report enum with Flat and Rough values.
- Modify `tools/benchmark_comparison/matrix.toml`: assign each locomotion task to its exact Flat or Rough report group.
- Modify `tools/benchmark_comparison/matrix.py`: update exact identifiers and schema-1 category reconstruction without changing attempt identities.
- Create `tools/benchmark_comparison/aggregate.py`: typed aggregate cells, exact ordering, CSV validation, zero-baseline exclusion, median, and mean.
- Modify `tools/benchmark_comparison/plot.py`: generate two annotated heatmaps and 24 four-group detail families.
- Modify `tools/benchmark_comparison/report.py`: put aggregate images and grouped figures first; move all tables into an appendix.
- Modify `tools/benchmark_comparison/pdf_report.py`: put cover/methodology, aggregate images, and grouped figures before every appendix table.
- Modify `tools/benchmark_comparison/report_cli.py`: pass one aggregate sequence to plotting/reporting and publish the exact 57-file inventory.
- Modify `tools/benchmark_comparison/tests/test_matrix.py`: exact four-way group membership and unchanged expansion identities/counts.
- Create `tools/benchmark_comparison/tests/test_aggregate.py`: statistics, ordering, equality checks, RGB omission, zero/missing behavior, and invalid input.
- Modify `tools/benchmark_comparison/tests/test_plot.py`: exact 26-family inventory, deterministic heatmaps, labels, annotations, normalization, and clipping checks.
- Modify `tools/benchmark_comparison/tests/test_report.py`: summary-first Markdown and appendix-last tables.
- Modify `tools/benchmark_comparison/tests/test_pdf_report.py`: exact summary/figure/appendix page order and 26-PNG validation.
- Modify `tools/benchmark_comparison/tests/test_report_cli_success.py`: exact 57-file end-to-end report and byte-identical regeneration.
- Modify `tools/benchmark_comparison/tests/test_report_cli_atomicity.py`: rollback at aggregate plots, detailed plots, and PDF boundaries.
- Modify `tools/benchmark_comparison/tests/test_report_integrity.py`: exact 57-entry generated hash inventory.
- Modify `tools/benchmark_comparison/tests/test_report_cli.py`: deterministic simulator-free plot counts.
- Modify `tools/benchmark_comparison/tests/test_report_path_safety.py`: preserve path-safety coverage with the new plot call.
- Keep `tools/benchmark_comparison/tests/test_actual_report_artifacts.py` unchanged because it intentionally audits the older retained 228-attempt, 17-file report root.

---

### Task 1: Split Locomotion Report Metadata Without Changing Runs

**Files:**
- Modify: `tools/benchmark_comparison/models.py`
- Modify: `tools/benchmark_comparison/matrix.toml`
- Modify: `tools/benchmark_comparison/matrix.py`
- Modify: `tools/benchmark_comparison/tests/test_matrix.py`

- [ ] **Step 1: Write the failing exact-membership assertions**

Replace `_EXPECTED_CATEGORIES` in `test_matrix.py` with:

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
    "locomotion_flat": (
        "anymal_d_flat",
        "g1_flat",
        "cassie_flat",
        "digit_flat",
        "go1_flat",
        "go2_flat",
    ),
    "locomotion_rough": (
        "anymal_d_rough",
        "g1_rough",
        "digit_rough",
        "go1_rough",
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

Update `test_task_aliases_by_category_filters_a_supplied_expansion_in_configured_order` to expect the four enum keys in that order. Add assertions that Flat and Rough are disjoint, their union is the old locomotion set, and each group's aliases retain their relative order from the matrix. Assert separately that the complete matrix task order is unchanged; flattening the four report groups intentionally differs from attempt order because existing Flat and Rough tasks are interleaved. Keep the existing exact 204-pair/408-attempt and 68-pair/136-attempt assertions.

Pin the ordered attempt identities before changing metadata. Add this assertion after expansion:

```python
payload = "\n".join(attempt.identity for attempt in expansion.attempts).encode()
assert hashlib.sha256(payload).hexdigest() == "8aba004dc8d09539e0fab0e8f07eb6a026f12059375a3e37e84c250c5c1c32e7"
```

Also retain the existing exact first/last identities. The digest covers all 408 identities and their order, proving category metadata does not change `BenchmarkAttempt.identity`, `run_directory`, or attempt ordering.

- [ ] **Step 2: Run the focused test and confirm RED**

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_matrix.py -v
```

Expected: FAIL because `TaskCategory.LOCOMOTION_FLAT` and `TaskCategory.LOCOMOTION_ROUGH` do not exist and all locomotion TOML entries still use `locomotion`.

- [ ] **Step 3: Implement the four exact report groups**

Change the enum to:

```python
class TaskCategory(str, Enum):
    """Readability group used by benchmark reports."""

    CLASSIC = "classic"
    LOCOMOTION_FLAT = "locomotion_flat"
    LOCOMOTION_ROUGH = "locomotion_rough"
    MANIPULATION = "manipulation"
```

In `matrix.toml`, replace each locomotion category using the exact aliases in the failing fixture. In `_TASK_IDENTIFIERS`, make the same enum substitutions. In `_LEGACY_SCHEMA_1_CATEGORIES`, classify `anymal_d_flat` and `g1_flat` as `LOCOMOTION_FLAT`; schema-1 counts and attempt identities remain unchanged.

Do not derive category from an alias at runtime and do not reorder `matrix.toml`. Continue parsing the explicit TOML value through `TaskCategory` and retain duplicate/completeness validation. Replace the old contiguous-category-block check with validation that every configured task appears once and that each grouped tuple preserves matrix-relative order. This is necessary because Flat and Rough entries are interleaved and report metadata must not change pair/attempt order.

- [ ] **Step 4: Run matrix and manifest compatibility tests and confirm GREEN**

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_matrix.py \
  tools/benchmark_comparison/tests/test_manifest.py \
  tools/benchmark_comparison/tests/test_manifest_normalization.py -v
```

Expected: PASS with unchanged expansion counts and attempt identities.

- [ ] **Step 5: Run pre-commit, stage, rerun, and commit**

```bash
uvx pre-commit run --all-files
git add tools/benchmark_comparison/models.py tools/benchmark_comparison/matrix.toml \
  tools/benchmark_comparison/matrix.py tools/benchmark_comparison/tests/test_matrix.py
uvx pre-commit run --all-files
git commit -m "Split locomotion benchmark report groups"
```

---

### Task 2: Add the Typed Aggregate-Delta Layer

**Files:**
- Create: `tools/benchmark_comparison/aggregate.py`
- Create: `tools/benchmark_comparison/tests/test_aggregate.py`

**Public interface:**

```python
class AggregateStatistic(str, Enum):
    MEDIAN = "median"
    MEAN = "mean"

AGGREGATE_METRICS: tuple[str, ...]
AGGREGATE_STATISTICS: tuple[AggregateStatistic, ...]

@dataclass(frozen=True)
class AggregateDelta:
    category: TaskCategory
    mode: str
    metric: str
    statistic: AggregateStatistic
    value: float | None
    task_count: int

def aggregate_paired_summary(
    paired_summary_path: Path,
    expansion: MatrixExpansion,
) -> tuple[AggregateDelta, ...]:
    """Aggregate validated task-level percentage deltas in report order."""
```

- [ ] **Step 1: Write RED tests for exact order and equal task weighting**

Create CSV helpers that write the exact `PAIRED_SUMMARY_FIELDS`. Build a small expansion subset from the checked-in final expansion and rows whose task deltas are intentionally different from a pooled/scale-weighted result. Assert:

```python
assert AGGREGATE_METRICS == (
    "collection_fps",
    "startup_total_s",
    "gpu_memory_mean_mib",
    "gpu_memory_peak_mib",
    "gpu_utilization_mean_pct",
)
assert tuple(cell.statistic for cell in cells[:60]) == (AggregateStatistic.MEDIAN,) * 60
assert tuple(cell.statistic for cell in cells[60:]) == (AggregateStatistic.MEAN,) * 60
assert len(cells) == 2 * 4 * 3 * 5 == 120
```

For Classic runtime-100 FPS deltas `(-20.0, 10.0, 70.0)`, assert median `10.0`, mean `20.0`, and `task_count == 3`. This proves equal task weighting rather than pooling Lab 2/Lab 3 magnitudes.

- [ ] **Step 2: Add RED edge-case and input-validation tests**

Cover all of these cases in `test_aggregate.py`:

- Cartpole RGB contributes to both runtime rows and never to training.
- Flat tasks never enter Rough cells and Rough tasks never enter Flat cells.
- `percent_delta_status == "undefined_zero_baseline"`, empty `percent_delta`, and `lab2_mean == 0.0` exclude the row and reduce `task_count`.
- A group/mode/metric with no usable rows returns both statistics as `value is None` and `task_count == 0`.
- A serialized percentage delta that differs from `(lab3_mean - lab2_mean) / lab2_mean * 100` raises `ValueError` naming task, mode, and metric.
- Duplicate task/mode/metric rows, unknown task/mode/metric/status, a row for a task's unsupported mode, malformed numbers, and non-finite input raise `ValueError`.
- Missing or duplicate category metadata and an expansion alias absent from the matrix raise before any result is returned. Missing result rows caused by failed/incomplete pairs are allowed and only reduce the contributing count.
- Two calls over identical bytes return equal immutable tuples.

- [ ] **Step 3: Run the new test module and confirm RED**

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_aggregate.py -v
```

Expected: collection error because `aggregate.py` does not exist.

- [ ] **Step 4: Implement strict parsing and ordered aggregation**

Use the exact CSV header and exact matrix membership. The core implementation must follow this shape:

```python
AGGREGATE_METRICS = (
    "collection_fps",
    "startup_total_s",
    "gpu_memory_mean_mib",
    "gpu_memory_peak_mib",
    "gpu_utilization_mean_pct",
)
AGGREGATE_STATISTICS = (AggregateStatistic.MEDIAN, AggregateStatistic.MEAN)

def _task_percent_delta(row: Mapping[str, str]) -> float | None:
    lab2_mean = _finite_float(row["lab2_mean"], "lab2_mean")
    lab3_mean = _finite_float(row["lab3_mean"], "lab3_mean")
    if lab2_mean == 0.0:
        if row["percent_delta_status"] != "undefined_zero_baseline" or row["percent_delta"]:
            raise ValueError("zero Lab 2 baseline has inconsistent percentage status")
        return None
    expected = (lab3_mean - lab2_mean) / lab2_mean * 100.0
    serialized = _finite_float(row["percent_delta"], "percent_delta")
    if row["percent_delta_status"] != "available" or not math.isclose(
        serialized, expected, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise ValueError("serialized percentage delta disagrees with means")
    return serialized
```

Read with `csv.DictReader`, require `tuple(reader.fieldnames or ()) == PAIRED_SUMMARY_FIELDS`, reject duplicate `(logical_task, mode, metric)` keys, and validate every represented alias against `task_aliases_by_category(expansion)`, every mode against `expansion_orders(expansion)`, and every metric against `SUMMARY_METRICS`. Parse `paired_seed_count` as a positive integer and parse all six floating-point summary columns as finite values, except the intentionally empty zero-baseline percentage. Recompute and validate the serialized percentage for every known summary metric, then retain only `AGGREGATE_METRICS` for aggregate cells. The `1e-9` tolerances account for the existing `.12g` CSV serialization without accepting a materially different percentage.

Build a lookup of usable values, then emit in this exact nesting order:

```python
for statistic in AGGREGATE_STATISTICS:
    for category in TaskCategory:
        for mode in mode_order:
            for metric in AGGREGATE_METRICS:
                values = tuple(
                    task_values[(task, mode, metric)]
                    for task in category_tasks[category]
                    if (task, mode, metric) in task_values
                )
                value = (
                    statistics.median(values)
                    if statistic is AggregateStatistic.MEDIAN and values
                    else statistics.fmean(values)
                    if values
                    else None
                )
                cells.append(AggregateDelta(category, mode, metric, statistic, value, len(values)))
```

Check `math.isfinite(value)` before constructing any non-`None` cell. Do not round stored values; renderers own display rounding.

- [ ] **Step 5: Run aggregate and normalization tests and confirm GREEN**

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_aggregate.py \
  tools/benchmark_comparison/tests/test_normalize.py -v
```

Expected: PASS with normalized CSV bytes/schema unchanged.

- [ ] **Step 6: Run pre-commit, stage, rerun, and commit**

```bash
uvx pre-commit run --all-files
git add tools/benchmark_comparison/aggregate.py tools/benchmark_comparison/tests/test_aggregate.py
uvx pre-commit run --all-files
git commit -m "Add aggregate benchmark deltas"
```

---

### Task 3: Generate Aggregate Heatmaps and Four-Group Detail Figures

**Files:**
- Modify: `tools/benchmark_comparison/plot.py`
- Modify: `tools/benchmark_comparison/tests/test_plot.py`

**Interfaces:**

```python
AGGREGATE_PLOT_BASENAMES = (
    "aggregate_delta_median_pct",
    "aggregate_delta_mean_pct",
)
DETAIL_PLOT_BASENAMES = tuple(
    f"{category.value}_{metric}"
    for category in TaskCategory
    for metric in (*PLOT_METRICS, "startup_phase_breakdown")
)
PLOT_BASENAMES = AGGREGATE_PLOT_BASENAMES + DETAIL_PLOT_BASENAMES

def generate_plots(
    raw_runs_path: Path,
    aggregate_deltas: Sequence[AggregateDelta],
    output_directory: Path,
    *,
    expansion: MatrixExpansion | None = None,
) -> tuple[Path, ...]:
    """Generate 26 fixed PNG/SVG figures from normalized CSV inputs."""
```

- [ ] **Step 1: Update detailed-plot tests for four groups and confirm RED**

Change expected detail families from 18 to 24 and generated files from 36 to 48 before adding heatmaps. Assert exact Flat/Rough membership in every SVG. Update parametrization to exercise `LOCOMOTION_FLAT`, `LOCOMOTION_ROUGH`, and `MANIPULATION`. Keep RGB's absent training label and the white left-boundary regression assertion.

Run:

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_plot.py -v
```

Expected: FAIL on old Locomotion basenames and counts.

- [ ] **Step 2: Add RED heatmap tests**

Add an exact `PAIRED_SUMMARY_FIELDS` CSV helper in `test_plot.py`, derive `aggregate_deltas` with `aggregate_paired_summary`, and call `generate_plots(raw_runs, aggregate_deltas, output, expansion=...)` twice. Assert:

- exact 52-file inventory and `PLOT_BASENAMES` order;
- exact aggregate filenames and byte-identical PNG/SVG regeneration;
- PNG dimensions `(1800, 1200)` for aggregate heatmaps and `(1800, 1000)` for detail figures;
- SVG text contains all 12 ordered `Group — mode` labels, all five metric labels, representative `+10.0%`, `-20.0%`, `(n=3)`, and `N/A` annotations;
- `matplotlib.colors.TwoSlopeNorm.vcenter == 0.0`, with `vmin == -vmax`, by testing a factored `_aggregate_color_norm(cells)` helper;
- both heatmaps share the same absolute color limit computed across all available median and mean cells;
- the first/last image rows and columns stay white where labels would otherwise clip; and
- no Flat alias occurs in a Rough detail SVG and no Rough alias occurs in a Flat detail SVG.

- [ ] **Step 3: Implement heatmap rendering from typed cells only**

Import `AggregateDelta`, `AggregateStatistic`, and `aggregate_paired_summary`. Keep plotting free of statistics and membership logic. After the aggregate layer returns 120 cells, validate exact uniqueness/order before drawing.

Use one shared color limit:

```python
available = [abs(cell.value) for cell in cells if cell.value is not None]
color_limit = max(available, default=1.0)
if color_limit == 0.0:
    color_limit = 1.0
norm = matplotlib.colors.TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)
```

For each statistic, build a 12-by-5 masked array and call:

```python
image = axis.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")
```

Annotate each cell with:

```python
annotation = "N/A" if cell.value is None else f"{cell.value:+.1f}%\n(n={cell.task_count})"
```

Use deterministic labels from explicit display dictionaries, a copied `RdBu_r` colormap with unavailable cells set to `#F2F2F2`, a horizontal colorbar labeled `Isaac Lab 3 - Isaac Lab 2 [%]`, `figsize=(12, 8)`, `dpi=150`, and margins large enough for all labels. Save through `_save_figure`; do not use `bbox_inches="tight"` because it destabilizes fixed dimensions.

Generate aggregate plots first, then detailed plots in `TaskCategory` order. Update docstrings and validation language from 18 to 26 families.

- [ ] **Step 4: Run plot tests and confirm GREEN**

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_aggregate.py \
  tools/benchmark_comparison/tests/test_plot.py -v
```

Expected: PASS with 52 byte-identical plot files.

- [ ] **Step 5: Run pre-commit, stage, rerun, and commit**

```bash
uvx pre-commit run --all-files
git add tools/benchmark_comparison/plot.py tools/benchmark_comparison/tests/test_plot.py
uvx pre-commit run --all-files
git commit -m "Plot aggregate benchmark summaries"
```

---

### Task 4: Reorder the Markdown Report Around Figures

**Files:**
- Modify: `tools/benchmark_comparison/report.py`
- Modify: `tools/benchmark_comparison/tests/test_report.py`

- [ ] **Step 1: Write RED ordering and content tests**

Update the main Markdown test to assert the exact top-level order by comparing string indexes:

```python
ordered_tokens = (
    "# Isaac Lab Paired Benchmark Report",
    "## Methodology",
    "## Median task-level delta",
    "aggregate_delta_median_pct.png",
    "## Mean task-level delta",
    "aggregate_delta_mean_pct.png",
    "## Detailed grouped figures",
    "### Classic",
    "### Locomotion Flat",
    "### Locomotion Rough",
    "### Manipulation",
    "## Appendix",
    "### Pinned revisions and execution identities",
    "### Hardware and software inventory",
    "### Task mapping",
    "### Detailed per-task results",
    "### Failures and missing attempts",
    "### Artifact integrity",
)
assert tuple(text.index(token) for token in ordered_tokens) == tuple(
    sorted(text.index(token) for token in ordered_tokens)
)
```

Assert no Markdown table delimiter (`|---`) occurs before `## Appendix`. Assert all 26 PNG basenames are embedded once, in aggregate-first then detailed-group order. Assert methodology states equal task weighting, median/mean, zero-baseline exclusion, no imputation, and informational-only interpretation. Retain all existing exact delta, startup-phase, successful-run, failure-link, provenance, inventory, and audit assertions, but scope them to the appendix substring.

- [ ] **Step 2: Run the Markdown tests and confirm RED**

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_report.py -v
```

Expected: FAIL because metadata and per-mode tables still precede figures and no aggregate images are embedded.

- [ ] **Step 3: Implement summary-first Markdown with an appendix**

Keep `write_markdown_report`'s validation behavior and signature. After title/methodology, emit the two aggregate image sections using relative filenames. Then emit `## Detailed grouped figures`; for each `TaskCategory`, emit the display heading and its six detail PNGs in `DETAIL_PLOT_BASENAMES` order.

Move, without deleting content:

1. pinned revisions;
2. hardware/software inventory and power-profile note;
3. task mapping;
4. each mode's startup table, runtime/resource table, and successful individual runs;
5. failures/missing attempts; and
6. artifact integrity

below one `## Appendix` heading. Demote their headings one level so Markdown structure remains valid. Keep raw artifact links unchanged.

Factor small helpers `_plot_image(basename)` and `_category_label(category)` so Markdown and PDF use the same human-readable names exported by `plot.py`; do not duplicate grouping order in renderer-local tuples.

- [ ] **Step 4: Run report validation suites and confirm GREEN**

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_report.py \
  tools/benchmark_comparison/tests/test_report_path_safety.py -v
```

Expected: PASS. Exact inventory integration remains covered in Task 6.

- [ ] **Step 5: Run pre-commit, stage, rerun, and commit**

```bash
uvx pre-commit run --all-files
git add tools/benchmark_comparison/report.py tools/benchmark_comparison/tests/test_report.py
uvx pre-commit run --all-files
git commit -m "Reorder benchmark Markdown report"
```

---

### Task 5: Reorder the PDF and Validate All 26 PNGs

**Files:**
- Modify: `tools/benchmark_comparison/pdf_report.py`
- Modify: `tools/benchmark_comparison/tests/test_pdf_report.py`

- [ ] **Step 1: Write RED tests for the new page sequence**

Change `_plot_paths` to create every basename in `PLOT_BASENAMES` and use `generated_file_count=57`. In the fixed-order test, reverse the supplied paths and expect the writer to restore this order:

1. `Isaac Lab Startup and Runtime Benchmark Report` cover containing concise methodology;
2. `Median Task-Level Percentage Delta`;
3. `Mean Task-Level Percentage Delta`;
4. 24 detailed plot pages in Classic, Locomotion Flat, Locomotion Rough, Manipulation order;
5. `Appendix: Pinned revisions and execution identities`;
6. `Appendix: Hardware and software inventory`;
7. `Appendix: Task mapping`;
8. six mode-specific startup/runtime table sequences;
9. successful individual-run appendix pages;
10. failures/missing attempts; and
11. artifact integrity audit.

Assert every table page title occurs after the last detailed figure. Assert the first page extracted text includes informational status, complete-pair semantics, equal task weighting, median/mean, zero-baseline exclusion, and no imputation. Keep byte-identical regeneration, ambient font isolation, pagination, atomic failure, title metadata, and Poppler validation tests.

Update invalid-plot cases to remove `aggregate_delta_median_pct`, duplicate a heatmap, use an unknown basename, pass a non-PNG, and corrupt image bytes. Require errors to say the PDF needs exactly 26 PNG plots.

- [ ] **Step 2: Run PDF tests and confirm RED**

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_pdf_report.py -v
```

Expected: FAIL on 18-plot validation and metadata/table-first page order.

- [ ] **Step 3: Implement aggregate-first PDF order**

Keep `write_pdf_report`'s signature and atomic temporary-file behavior. Update `_ordered_plot_paths` to require `set(PLOT_BASENAMES)` and exactly 26 unique PNGs. Render one cover page using `_REPORT_TITLE` and the concise methodology lines; remove the separate methodology page.

Render the two aggregate plot pages first, then iterate `DETAIL_PLOT_BASENAMES`. Only after the last detail page, render all existing metadata, paired-summary, individual-run, failure, and audit pages. Prefix appendix page titles consistently so ordering is visible in extracted text.

Make `_plot_title` handle aggregate basenames explicitly before splitting detail names. A safe implementation is:

```python
_AGGREGATE_TITLES = {
    "aggregate_delta_median_pct": "Median Task-Level Percentage Delta",
    "aggregate_delta_mean_pct": "Mean Task-Level Percentage Delta",
}

def _plot_title(basename: str) -> str:
    if basename in _AGGREGATE_TITLES:
        return _AGGREGATE_TITLES[basename]
    for category in TaskCategory:
        prefix = f"{category.value}_"
        if basename.startswith(prefix):
            return f"{CATEGORY_LABELS[category]}: {_METRIC_TITLES[basename.removeprefix(prefix)]}"
    raise ValueError(f"unknown benchmark plot basename: {basename}")
```

This avoids incorrectly splitting `locomotion_flat_*` at the first underscore. Keep image bytes unmodified and use the generated PNGs directly.

- [ ] **Step 4: Run PDF and plot tests and confirm GREEN**

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_plot.py \
  tools/benchmark_comparison/tests/test_pdf_report.py -v
```

Expected: PASS, including exact page order and byte-identical PDFs.

- [ ] **Step 5: Run pre-commit, stage, rerun, and commit**

```bash
uvx pre-commit run --all-files
git add tools/benchmark_comparison/pdf_report.py tools/benchmark_comparison/tests/test_pdf_report.py
uvx pre-commit run --all-files
git commit -m "Reorder benchmark PDF report"
```

---

### Task 6: Wire the 57-File Transaction and End-to-End Tests

**Files:**
- Modify: `tools/benchmark_comparison/report_cli.py`
- Modify: `tools/benchmark_comparison/tests/test_report_cli_success.py`
- Modify: `tools/benchmark_comparison/tests/test_report_cli_atomicity.py`
- Modify: `tools/benchmark_comparison/tests/test_report_integrity.py`
- Modify: `tools/benchmark_comparison/tests/test_report_cli.py`
- Modify: `tools/benchmark_comparison/tests/test_report_path_safety.py`

- [ ] **Step 1: Write RED exact-inventory integration assertions**

Replace every local duplicated category list in the listed tests with an import of `PLOT_BASENAMES`. Define the expected generated inventory as:

```python
expected_generated = {
    "raw_runs.csv",
    "paired_summary.csv",
    "failures.csv",
    "report.md",
    "report.pdf",
    *(f"{basename}.{suffix}" for basename in PLOT_BASENAMES for suffix in ("png", "svg")),
}
assert len(PLOT_BASENAMES) == 26
assert len(expected_generated) == 57
```

In success and integrity tests, assert exact filenames, 57 unique generated-hash entries, `audit_summary.json["generated_file_count"] == 57`, correct per-file SHA-256 values, and byte-identical second regeneration. Assert Markdown embeds the two aggregate PNGs and PDF validates with `Median Task-Level Percentage Delta` and `Mean Task-Level Percentage Delta` tokens.

Keep `_write_previous_41_file_report` because that is the report inventory deployed before this change, and add `_write_previous_57_file_report` for subsequent regeneration failures. Parametrize both prior inventories across publication failure at `aggregate_plots`, `detailed_plots`, and `pdf`; after each injected failure, assert the entire prior report and raw fixture bytes are unchanged and no staging/backup directory remains.

Update report-path-safety monkeypatches for the new `generate_plots(raw, aggregate_deltas, staging, ...)` call shape. For the two atomic plot boundaries, make the injected `generate_plots` write both aggregate files before raising for `aggregate_plots`, and both aggregate files plus the first detail file before raising for `detailed_plots`. Do not change `test_actual_report_artifacts.py` or its 17-file retained-root expectation.

- [ ] **Step 2: Run integration tests and confirm RED**

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_report_cli.py \
  tools/benchmark_comparison/tests/test_report_cli_success.py \
  tools/benchmark_comparison/tests/test_report_cli_atomicity.py \
  tools/benchmark_comparison/tests/test_report_integrity.py \
  tools/benchmark_comparison/tests/test_report_path_safety.py -v
```

Expected: FAIL because `report_cli` still calls the old plot signature and reports 41 generated files.

- [ ] **Step 3: Wire one validated aggregate flow into the transaction**

Import `aggregate_paired_summary`. After `write_normalized_outputs`, derive the aggregate sequence once and pass it to plotting:

```python
aggregate_deltas = aggregate_paired_summary(normalized["paired_summary"], expansion)
plots = generate_plots(
    normalized["raw_runs"],
    aggregate_deltas,
    staging,
    expansion=expansion,
)
```

Require `tuple(path.stem for path in plots[::2]) == PLOT_BASENAMES` and paired `.png`, `.svg` outputs before constructing the inventory. Continue passing only PNG paths to `write_pdf_report`. The Markdown writer references the same basenames in the same staging directory. Keep all 57 generated paths sorted by filename for `generated_hashes.sha256` and keep `audit_summary.json`, raw/generated hash manifests, and the manifest outside that generated count.

Do not weaken the existing checks that raw hashes are stable before/after generation, publication replaces the whole directory, failures preserve the prior destination, and staging is always removed.

- [ ] **Step 4: Run all report integration tests and confirm GREEN**

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_report_cli.py \
  tools/benchmark_comparison/tests/test_report_cli_success.py \
  tools/benchmark_comparison/tests/test_report_cli_atomicity.py \
  tools/benchmark_comparison/tests/test_report_integrity.py \
  tools/benchmark_comparison/tests/test_report_path_safety.py -v
```

Expected: PASS with exact 57-file inventories and byte-identical reruns.

- [ ] **Step 5: Run pre-commit, stage, rerun, and commit**

```bash
uvx pre-commit run --all-files
git add tools/benchmark_comparison/report_cli.py \
  tools/benchmark_comparison/tests/test_report_cli.py \
  tools/benchmark_comparison/tests/test_report_cli_success.py \
  tools/benchmark_comparison/tests/test_report_cli_atomicity.py \
  tools/benchmark_comparison/tests/test_report_integrity.py \
  tools/benchmark_comparison/tests/test_report_path_safety.py
uvx pre-commit run --all-files
git commit -m "Publish aggregate benchmark report"
```

---

### Task 7: Verify the Complete Simulator-Free Change

**Files:**
- Verify only: `tools/benchmark_comparison/`
- Verify only: repository-wide formatting/lint state

- [ ] **Step 1: Run the complete benchmark-comparison suite**

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked --with pytest \
  python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests -v
```

Expected: all tests PASS, no simulator launch, and `test_actual_report_artifacts.py` either skips its absent old root or passes its unchanged retained-root assertions.

- [ ] **Step 2: Run pre-commit twice and require a clean tree**

```bash
uvx pre-commit run --all-files
git status --short
uvx pre-commit run --all-files
git status --short
```

Expected: both pre-commit runs PASS and both status checks are empty. If hooks edit files, review and commit a focused fix before rerunning this step.

- [ ] **Step 3: Review the diff against the approved design**

```bash
git diff 7209bba5e7..HEAD --check
git diff 7209bba5e7..HEAD --stat
git log --oneline 7209bba5e7..HEAD
```

Verify manually that no benchmark execution code, attempt artifacts, dependency files, `.orig` snapshots, changelogs, or version files changed.

---

### Task 8: Regenerate the Completed 408-Attempt Report Twice

**Files:**
- Regenerate only: `/home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd-expanded-408/final/report/`
- Preserve: every other path below `/home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd-expanded-408/`

- [ ] **Step 1: Snapshot immutable evidence before publication**

```bash
cp /home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd-expanded-408/final/report/raw_artifact_hashes.sha256 \
  /tmp/c8d672a1dd-expanded-408.raw.before.sha256
cp /home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd-expanded-408/final/report/generated_hashes.sha256 \
  /tmp/c8d672a1dd-expanded-408.generated.old.sha256
sha256sum /home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd-expanded-408/final/manifest.json \
  > /tmp/c8d672a1dd-expanded-408.manifest.before.sha256
```

Also record `git status --short` in the original `/home/antoiner/Documents/IsaacLab` checkout and do not modify it.

- [ ] **Step 2: Regenerate once without running either framework**

```bash
uv run --project /home/antoiner/benchmarks/isaaclab2-vs-3/lab3-develop \
  --extra isaacsim --extra rsl-rl --locked \
  python -m tools.benchmark_comparison.report_cli \
  --artifact_root /home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd-expanded-408 \
  --run_set final --phase measured \
  --output_dir /home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd-expanded-408/final/report
```

Expected: exit 0, no simulator process, no benchmark attempt directory write, and atomic replacement of only `final/report`.

- [ ] **Step 3: Audit the first generated report**

```bash
wc -l /home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd-expanded-408/final/report/raw_artifact_hashes.sha256
wc -l /home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd-expanded-408/final/report/generated_hashes.sha256
cmp /tmp/c8d672a1dd-expanded-408.raw.before.sha256 \
  /home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd-expanded-408/final/report/raw_artifact_hashes.sha256
pdfinfo /home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd-expanded-408/final/report/report.pdf
pdftotext /home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd-expanded-408/final/report/report.pdf - \
  | rg "Median Task-Level Percentage Delta|Mean Task-Level Percentage Delta|Appendix: Pinned revisions"
```

Expected: raw manifest 4,271 lines and byte-identical to the snapshot; generated manifest 57 lines; valid non-empty PDF; extracted section tokens in summary-first order. Inspect `audit_summary.json` for 408 successes, 0 failures/missing, 4,271 raw files, and 57 generated files.

Copy the first new generated manifest for the determinism comparison:

```bash
cp /home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd-expanded-408/final/report/generated_hashes.sha256 \
  /tmp/c8d672a1dd-expanded-408.generated.first.sha256
```

- [ ] **Step 4: Visually inspect the two heatmaps and representative detail plots**

Open or inspect at original resolution:

```text
aggregate_delta_median_pct.png
aggregate_delta_mean_pct.png
classic_collection_fps.png
locomotion_flat_collection_fps.png
locomotion_rough_collection_fps.png
manipulation_collection_fps.png
```

Require readable complete row/column labels, visible signed percentages and `n`, neutral zero-centered colors, no clipped axis/colorbar text, correct group membership, white margins, and no empty RGB training slot.

- [ ] **Step 5: Regenerate a second time and prove byte identity**

Repeat the exact report-only command from Step 2, then run:

```bash
cmp /tmp/c8d672a1dd-expanded-408.generated.first.sha256 \
  /home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd-expanded-408/final/report/generated_hashes.sha256
cmp /tmp/c8d672a1dd-expanded-408.raw.before.sha256 \
  /home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd-expanded-408/final/report/raw_artifact_hashes.sha256
(cd /home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd-expanded-408/final/report && \
  sha256sum -c generated_hashes.sha256)
```

Expected: both `cmp` commands and all 57 SHA-256 checks PASS.

- [ ] **Step 6: Prove no residue or out-of-scope mutation remains**

```bash
find /home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd-expanded-408/final \
  -maxdepth 1 -type d -name '.report.*' -print
sha256sum -c /tmp/c8d672a1dd-expanded-408.manifest.before.sha256
git status --short
```

Expected: no staging/backup directories, unchanged final manifest, clean implementation worktree, and unchanged original checkout status. Remove only the `/tmp/c8d672a1dd-expanded-408.*` snapshots after recording final evidence.

---

### Task 9: Final Review and Handoff

**Files:**
- Review only: implementation commits and regenerated report evidence

- [ ] **Step 1: Apply the verification-before-completion skill**

Re-run any command whose output is no longer fresh. Record the complete pytest result, both pre-commit results, 57/4,271 inventory counts, generated manifest digest, PDF digest/page count, and the clean-tree result. Do not claim completion from earlier logs.

- [ ] **Step 2: Review against every approved requirement**

Confirm all of the following in the actual generated report: no opening tables; separate median and mean heatmaps; separate mode rows; exact four groups; exact five metrics; equal task weighting; RGB runtime-only; zero-baseline exclusion; detailed figures before appendices; audit tables retained; raw data retained; 408 successes; zero failures; and deterministic second generation.

- [ ] **Step 3: Request code review before final handoff**

Apply `superpowers:requesting-code-review` to the complete diff. Address Critical or Important findings with new focused commits, then repeat Task 7 and the relevant Task 8 audits. Do not merge, push, restore experimental files, or rerun benchmarks unless the user separately requests it.
