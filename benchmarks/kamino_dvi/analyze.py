# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate compact results, figures, and reports from completed benchmark runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .analysis import VariantSummary, complete_three_seed_records, load_records, summarize_records
from .matrix import DEFAULT_MATRIX_PATH, load_matrix
from .plotting import VARIANT_LABELS, plot_learning, plot_runtime
from .reporting import write_reports


def quality_issues(records, summaries: list[VariantSummary], artifact_root: Path) -> list[str]:
    """Return explicit schema, capacity, failure, and learning-quality warnings."""
    issues: list[str] = []
    mismatch_count = sum(record.success_schema_mismatch for record in records)
    if mismatch_count:
        issues.append(
            f"Schema v1.1 success series differs from TensorBoard in {mismatch_count} of {len(records)} runs; "
            "report uses TensorBoard success. This is the known generator bug addressed separately in PR 6624."
        )
    for summary in summaries:
        if (
            summary.variant == "kamino_pr_dvi"
            and summary.success_rate is not None
            and summary.success_rate.half_width > 0.25
        ):
            issues.append(
                f"{summary.task} tuned DVI success is seed-sensitive: the three-seed 95% CI half-width is "
                f"{summary.success_rate.half_width:.3f}."
            )
        if summary.num_envs < 4096:
            issues.append(f"{summary.task} used {summary.num_envs} environments after a documented capacity fallback.")
    for manifest_path in sorted(artifact_root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("state") == "failed":
            identity = manifest["identity"]
            issues.append(
                f"Failed {identity['task']} / {identity['variant']} / seed {identity['seed']}: "
                f"{manifest.get('failure_category')}."
            )
    by_task = {summary.task for summary in summaries}
    for task in sorted(by_task):
        rows = {summary.variant: summary for summary in summaries if summary.task == task}
        baseline = rows.get("kamino_current")
        if baseline is None:
            continue
        baseline_floor = baseline.reward.mean - baseline.reward.half_width
        for variant, row in rows.items():
            if variant == "kamino_current":
                continue
            if row.reward.mean + row.reward.half_width < baseline_floor:
                issues.append(
                    f"{task} {VARIANT_LABELS.get(variant, variant)} has materially lower final-window reward than "
                    "current Kamino."
                )
    return issues


def build_parser() -> argparse.ArgumentParser:
    """Build the report generator command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("benchmark_artifacts/kamino_dvi/runs"))
    parser.add_argument("--logs-root", type=Path, default=Path("logs"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/kamino_dvi/results"))
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load validated runs and generate all compact report artifacts."""
    args = build_parser().parse_args(argv)
    matrix = load_matrix(args.matrix)
    records = [record for record in load_records(args.artifact_root, args.logs_root) if record.seed in matrix.seeds]
    summaries = summarize_records(complete_three_seed_records(records))
    if not summaries:
        raise RuntimeError("no complete three-seed task/variant groups are available")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime_figure = args.output_dir / "runtime.png"
    learning_figure = args.output_dir / "learning.png"
    plot_runtime(summaries, runtime_figure)
    plot_learning(summaries, learning_figure)
    issues = quality_issues(records, summaries, args.artifact_root)
    (args.output_dir / "summary.json").write_text(
        json.dumps([asdict(summary) for summary in summaries], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_reports(
        summaries,
        issues,
        [runtime_figure, learning_figure],
        args.output_dir / "kamino_dvi_benchmark.md",
        args.output_dir / "kamino_dvi_benchmark.pdf",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
