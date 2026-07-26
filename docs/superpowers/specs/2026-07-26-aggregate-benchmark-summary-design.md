# Aggregate Benchmark Summary Design

## Goal

Replace the table-heavy opening of the Isaac Lab 2.3.2 versus Isaac Lab 3
benchmark report with an executive comparison of aggregate percentage deltas.
The report must make the four requested workload groups easy to compare:
Classic, Locomotion Flat, Locomotion Rough, and Manipulation.

This is a derived-report change only. It must not modify benchmark attempts,
raw measurements, normalized raw rows, import evidence, or immutable artifact
bundles.

## Executive summary

The report opens with two heatmaps instead of revision, hardware, task-mapping,
or per-task result tables:

1. median task-level percentage delta;
2. arithmetic mean task-level percentage delta.

Each heatmap uses rows for workload group and benchmark mode, in this order:

1. Classic — runtime-100;
2. Classic — runtime-1000;
3. Classic — training-100;
4. Locomotion Flat — runtime-100;
5. Locomotion Flat — runtime-1000;
6. Locomotion Flat — training-100;
7. Locomotion Rough — runtime-100;
8. Locomotion Rough — runtime-1000;
9. Locomotion Rough — training-100;
10. Manipulation — runtime-100;
11. Manipulation — runtime-1000;
12. Manipulation — training-100.

Columns use the following metrics and order:

1. collection FPS;
2. total startup time [s];
3. mean GPU memory [MiB];
4. peak GPU memory [MiB];
5. mean GPU utilization [%].

Every available cell contains the aggregate percentage delta and contributing
logical-task count, for example `+12.4%\n(n=7)`. Unavailable cells display
`N/A`. Colors use a neutral diverging scale centered at zero. The report stays
informational and does not label positive or negative values as inherently
good or bad.

## Aggregation semantics

The aggregate input is the validated seed-aggregated `paired_summary.csv`.
For each logical task, mode, and metric, the existing paired summary defines:

```text
task_delta_pct = (lab3_mean - lab2_mean) / lab2_mean * 100
```

Each logical task contributes at most one percentage delta to a group, mode,
and metric. Logical tasks receive equal weight regardless of absolute metric
scale. The summary reports both:

- the median of the contributing task-level percentage deltas; and
- the arithmetic mean of the contributing task-level percentage deltas.

The aggregation must use the serialized `percent_delta` already validated as
derived from normalized raw runs, or independently recompute and compare it
before use. A zero Lab 2 baseline is excluded from that metric's aggregate and
reduces its displayed `n`. A group, mode, and metric with no valid task delta
is `N/A` with `n=0` represented internally.

The aggregation does not pool raw seeds and does not weight a task by FPS,
memory scale, duration, number of phases, or number of environments. Existing
paired-seed means remain the task-level comparison boundary.

## Workload groups

### Classic

Classic contains the existing Classic category:

- Cartpole manager-based;
- Cartpole RGB Kit renderer;
- Cartpole Direct;
- Ant manager-based;
- Ant Direct;
- Humanoid manager-based;
- Humanoid Direct.

Cartpole RGB contributes only to runtime-100 and runtime-1000. It does not
create a training slot or an imputed training value.

### Locomotion Flat

Locomotion Flat contains every Locomotion task whose logical alias denotes
Flat terrain. This includes Flat-only robots and the Flat member of robots
that also expose Rough terrain.

### Locomotion Rough

Locomotion Rough contains every Locomotion task whose logical alias denotes
Rough terrain. Flat results never contribute to this group, and Rough results
never contribute to Locomotion Flat.

### Manipulation

Manipulation contains the existing Manipulation category, including Allegro,
Franka, and Kuka–Allegro tasks.

Group membership must come from the exact manifest expansion and matrix task
metadata. The report must reject duplicate, missing, or ambiguous membership
rather than silently classify by display labels alone.

The matrix's current three-value report category becomes four explicit values:
`classic`, `locomotion_flat`, `locomotion_rough`, and `manipulation`. Existing
Locomotion task entries move to the appropriate Flat or Rough value. This
metadata affects report grouping only; attempt identities, concrete task
mappings, run ordering, and immutable artifact compatibility remain unchanged.

## Report organization

The Markdown and PDF use the same conceptual order:

1. title and concise methodology statement;
2. median aggregate-delta heatmap;
3. mean aggregate-delta heatmap;
4. detailed grouped figures;
5. appendix.

Detailed figures are ordered:

1. Classic;
2. Locomotion Flat;
3. Locomotion Rough;
4. Manipulation.

The appendix contains:

1. pinned revisions and execution identities;
2. hardware and software inventory;
3. task mapping;
4. detailed per-task startup and runtime tables;
5. failures and missing attempts;
6. raw-artifact integrity audit.

The opening contains no metadata, task-mapping, or detailed per-task tables.
Those tables remain in the appendix for auditability. Normalized CSVs remain
published unchanged.

## Components and data flow

The implementation adds a deterministic aggregate-summary component with one
responsibility: convert validated paired-summary rows and exact matrix group
membership into ordered aggregate cells. Its public output to renderers is a
typed immutable sequence containing group, mode, metric, statistic, value,
and contributing task count.

Data flows as follows:

```text
immutable success artifacts
  -> normalized raw_runs.csv
  -> validated paired_summary.csv
  -> aggregate task-delta summary
  -> median and mean PNG/SVG heatmaps
  -> reordered Markdown and PDF
```

Markdown and PDF rendering consume the same aggregate-summary values and the
same ordering constants. Plotting owns visual presentation only; it must not
reimplement grouping or statistics.

## Output inventory and determinism

Splitting the existing Locomotion detailed figures replaces six combined plot
families with twelve Flat/Rough families. The two heatmap families add four
more generated plot files:

- `aggregate_delta_median_pct.png`;
- `aggregate_delta_median_pct.svg`;
- `aggregate_delta_mean_pct.png`;
- `aggregate_delta_mean_pct.svg`.

The generated-file count therefore increases from 41 to 57. Plot files
increase from 36 to 52 across 26 plot families: 24 detailed group/metric
families plus two aggregate heatmaps, each emitted as PNG and SVG.
`raw_artifact_hashes.sha256` and its 4,271-entry raw inventory must remain
unchanged.

The report must be regenerated twice from the completed 408-attempt root.
Both `generated_hashes.sha256` files must be byte-identical, and publication
must leave no staging, backup, or temporary residue.

## Error handling

Report generation fails before publication when:

- a paired-summary row is not derived from normalized raw runs;
- a task has missing, duplicate, or ambiguous group membership;
- an expected group, mode, metric, or statistic ordering value is unknown;
- a serialized percentage delta disagrees with its Lab 2/Lab 3 means;
- a non-finite aggregate would be emitted;
- heatmap output is missing, duplicated, clipped, or outside the deterministic
  generated inventory; or
- Markdown and PDF use different aggregate values or ordering.

Zero Lab 2 baselines are a defined exclusion, not an error. Missing modes such
as Cartpole RGB training are also defined exclusions and are never imputed.

## Validation

Tests must cover:

- exact Classic, Locomotion Flat, Locomotion Rough, and Manipulation
  membership;
- strict Flat/Rough separation;
- equal logical-task weighting;
- median and arithmetic-mean calculations;
- runtime-only Cartpole RGB behavior;
- zero baselines, unavailable cells, and contributing counts;
- deterministic group, mode, and metric order;
- agreement between serialized and recomputed task percentage deltas;
- matching Markdown, PDF, PNG, and SVG aggregate values;
- zero-centered heatmap color normalization;
- readable, unclipped row/column labels and cell annotations;
- executive-summary-first and appendix-last PDF section order;
- exact 57-entry generated inventory and unchanged 4,271-entry raw inventory;
- atomic publication rollback and absence of residue; and
- byte-identical report regeneration.

The complete simulator-free benchmark-comparison suite and repository-wide
pre-commit hooks must pass. No simulator, Docker container, or measured
benchmark rerun is required for this derived-report change.

## Non-goals

- Changing or rerunning benchmark measurements.
- Adding performance acceptance thresholds or pass/fail labels.
- Combining runtime and training modes into one aggregate.
- Weighting by environment count, task scale, seed count, or runtime length.
- Removing audit tables or normalized CSVs from the report package.
- Changing raw-artifact or original-root contents.
