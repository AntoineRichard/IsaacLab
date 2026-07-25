<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Startup and Runtime PDF Report Design

**Status:** Approved

## Context

The completed IsaacLab 2.x-versus-3.0 final matrix contains 228 immutable
successful attempts: 13 tasks in two runtime modes, 12 training-capable tasks
in one training mode, three paired seeds, and two IsaacLab versions. The
current report normalizes collection FPS, GPU memory, GPU utilization, and
process elapsed time. It does not preserve startup timings in `raw_runs.csv`
or visualize them even though every success artifact already contains the same
five validated startup phases.

The report must add startup analysis without running another simulator
benchmark. It must continue to preserve all raw artifacts, generate editable
CSV and image outputs, expand the Markdown report, and add a large PDF that
combines startup, runtime, training, resource, provenance, and integrity
information.

## Goals

- Reuse the 228 existing checksummed success artifacts without launching
  Isaac Sim, Docker benchmark containers, or RSL-RL training.
- Normalize total startup time and every measured startup phase for each run.
- Add paired three-seed startup statistics to the existing comparison tables.
- Add an editable total-startup plot and a startup-phase breakdown plot.
- Expand the Markdown report to cover startup and steady-state/runtime results
  in one document.
- Generate a large deterministic PDF from the same normalized report model.
- Preserve the existing raw and generated hash audits.

## Non-goals

- Run a dedicated startup benchmark or rerun any existing matrix attempt.
- Change task mappings, seeds, environment count, RSL-RL settings, execution
  order, idle-gate policy, or benchmark commands.
- Rewrite any immutable success, diagnostic, idle-gate, or manifest artifact.
- Infer startup phases that are absent from an artifact.
- Add a required IsaacLab core dependency or a browser-based PDF renderer.
- Define performance thresholds or pass/fail criteria from startup results.

## Source Data and Startup Semantics

The canonical source is `schema.json` inside each immutable `success`
directory. Validation already extracts `runtime.startup_time_s` into
`SemanticMetrics.phase_timings_s`, and `validation.json` contains the same
validated projection. Report normalization continues to validate the success
bundle and its checksums before consuming these metrics.

All 228 final successes contain exactly these phases:

1. `app_launch`
2. `python_imports`
3. `task_config`
4. `env_creation`
5. `first_step`

The normalized total is:

```text
startup_total_s = app_launch + python_imports + task_config
                  + env_creation + first_step
```

Each component and the total are finite, non-negative seconds. Normalization
rejects a success if a required phase is missing, an unexpected phase is
present, or the total is not finite. This keeps startup comparisons identical
across versions and prevents silent schema drift.

`startup_total_s` is distinct from `elapsed_time_s`. Startup is measured by
the benchmark formatter inside the workflow; elapsed time is the controller's
wall-clock duration for the complete process. Both remain in the report and
are labeled explicitly.

## Normalized Data Model

`NormalizedRun` and `raw_runs.csv` gain these ordered fields before
`elapsed_time_s`:

- `startup_total_s`
- `startup_app_launch_s`
- `startup_python_imports_s`
- `startup_task_config_s`
- `startup_env_creation_s`
- `startup_first_step_s`

The CSV remains the complete editable per-run source for downstream tables,
plots, and PDF rendering. Re-reading `raw_runs.csv` validates every new value
as finite and non-negative. The total is checked against the sum of its five
serialized components within the formatter's numeric precision.

The six startup fields join `SUMMARY_METRICS`, so `paired_summary.csv`
contains Lab 2 and Lab 3 means, sample standard deviations, signed deltas, and
percentage deltas for three complete seed pairs. Existing pair invariants and
ordering remain unchanged.

## Plots

The four existing plot families remain:

- collection FPS;
- mean GPU memory;
- peak GPU memory; and
- mean GPU utilization.

Two new plot families are generated as PNG and SVG:

### Total Startup Time

`startup_total_s` uses the existing grouped-bar convention: Lab 2 and Lab 3
means, sample-standard-deviation error bars, and one point per seed. It has the
same task and mode order as the other comparison plots and a seconds axis.

### Startup Phase Breakdown

The phase breakdown uses paired stacked bars for each task and mode. Each
version bar contains the five phases in canonical startup order and uses a
fixed phase color palette. The version distinction remains visible through
paired bar positions and version labels; phase colors do not change between
versions. A phase legend and a version legend are placed outside the data
region. The same readable 45-degree task labels and fixed output dimensions
are used.

The stacked bar height must equal the corresponding normalized
`startup_total_s` mean within floating-point tolerance. Seed-level totals are
not stacked individually; their variability is already visible in the total
startup plot and paired tables.

## Markdown Report

`report.md` becomes a large combined startup and runtime report. It retains
the current informational-only language, methodology, pinned revisions,
hardware/software inventory, task mapping, and failure section.

Each mode contains:

1. a startup paired-summary table containing total startup and all five
   phases;
2. the existing throughput/resource paired-summary table;
3. individual successful runs with startup total, the five startup phases,
   collection FPS, GPU memory, GPU utilization, sample count, elapsed time,
   and artifact link.

An artifact-integrity appendix records successful, failed/missing, raw-file,
and generated-file counts plus the raw hash-manifest digest. The report does
not embed the digest of `generated_hashes.sha256`, because that manifest hashes
the report itself and would create a circular dependency. The external
`audit_summary.json` records the final generated-manifest digest after all
report outputs have been assembled.

## PDF Report

The PDF is a second rendering of the same normalized report model, not a
separate source of metrics. It is generated with Matplotlib's PDF backend,
which is already required for plots, and does not require Chrome, Pandoc,
WeasyPrint, or another package.

The PDF contains:

- title, methodology, and interpretation notes;
- pinned revisions and environment identities;
- hardware/software inventory and task mappings;
- per-mode startup and throughput/resource paired tables;
- all 228 individual successful-run rows in a paginated appendix;
- failure and integrity summaries; and
- full-page versions of all six comparison figures, including both startup
  figures.

Generic table pagination uses fixed column sets, font sizes, row counts, and
page dimensions. Long artifact paths are shown as compact attempt identities
while the Markdown/CSV retain the complete path. Tables repeat headers on each
page.

PDF metadata uses fixed title, author, creator, creation date, and modification
date values. Font family, page size, plot metadata, and ordering are fixed so
regeneration from identical normalized inputs is byte-identical. The PDF is
written to a temporary path and atomically replaced only after successful
rendering and validation.

## Data Flow

```text
immutable success artifacts
        |
        v
checksum + semantic validation
        |
        v
NormalizedRun (runtime, resources, startup total, startup phases)
        |
        +--> raw_runs.csv
        +--> paired_summary.csv
        +--> failures.csv
                  |
                  +--> PNG/SVG plots
                  +--> report.md
                  +--> report.pdf
                  +--> audit_summary.json and hash manifests
```

No report step invokes an IsaacLab executor. A report generation command that
would launch Docker, Isaac Sim, or a training script is a defect.

## Failure Handling and Integrity

- Existing immutable success checksums are verified before normalization.
- Missing or malformed startup phases classify the success as invalid rather
  than substituting zero or dropping a phase.
- Plot or PDF failure leaves the previously complete report directory intact
  through the existing atomic staging/swap behavior.
- PDF validation verifies the PDF header, non-zero page count, expected title,
  and extractable text containing the run-set identity and both version SHAs.
- `generated_hashes.sha256` includes the PDF and both files for each new plot.
- `raw_artifact_hashes.sha256` is regenerated from, and checked against, the
  unchanged raw artifact tree.
- Report generation is run twice; normalized CSVs, Markdown, images, PDF, and
  generated hash manifest must be byte-identical.

## Testing

Test-driven implementation covers:

- extraction of the five canonical phase values and their total;
- rejection of missing, unexpected, negative, or non-finite startup values;
- CSV serialization/deserialization and total/component consistency;
- paired startup statistics using only complete seed pairs;
- deterministic ordering and labeling in Markdown tables;
- total-startup PNG/SVG generation and deterministic regeneration;
- stacked phase order, legend content, and stack-height equality;
- PDF section order, pagination, required text, page count, fixed metadata,
  and byte-identical regeneration;
- generated/raw hash inventory updates;
- atomic behavior when plotting or PDF generation fails; and
- regeneration of the actual 228-run report without invoking an executor.

The focused simulator-free comparison suite runs after every change, followed
by repository pre-commit hooks. Final verification regenerates the report
twice from the existing artifacts, validates all hashes, inspects both startup
plots and representative PDF pages, and confirms that the runner history and
raw artifact tree are unchanged.

## Deliverables

The final report directory contains at least:

- `raw_runs.csv`
- `paired_summary.csv`
- `failures.csv`
- `report.md`
- `report.pdf`
- `collection_fps.{png,svg}`
- `gpu_memory_mean_mib.{png,svg}`
- `gpu_memory_peak_mib.{png,svg}`
- `gpu_utilization_mean_pct.{png,svg}`
- `startup_total_s.{png,svg}`
- `startup_phase_breakdown.{png,svg}`
- `audit_summary.json`
- `raw_artifact_hashes.sha256`
- `generated_hashes.sha256`

The raw benchmark artifacts, Docker image identity, Lab 2 benchmark SHA, Lab 3
SHA, and Lab 3 lock hash remain unchanged.
