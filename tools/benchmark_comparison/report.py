# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate an informational Markdown report from normalized benchmark CSV."""

from __future__ import annotations

import csv
import os
from collections.abc import Mapping
from pathlib import Path

from .normalize import MODE_ORDER, TASK_ORDER, read_raw_runs_csv

_METRIC_LABELS = {
    "collection_fps": "Collection FPS",
    "gpu_memory_mean_mib": "Mean GPU memory [MiB]",
    "gpu_memory_peak_mib": "Peak GPU memory [MiB]",
    "gpu_utilization_mean_pct": "Mean GPU utilization [%]",
    "gpu_utilization_sample_count": "GPU utilization samples",
    "elapsed_time_s": "Elapsed time [s]",
}


def write_markdown_report(
    raw_runs_path: Path,
    paired_summary_path: Path,
    failures_path: Path,
    output_path: Path,
    *,
    inventory: Mapping[str, object],
    artifact_root: Path | None = None,
) -> Path:
    """Write a deterministic report without reading simulator logs or schemas."""
    runs = read_raw_runs_csv(raw_runs_path)
    summaries = _read_csv(paired_summary_path)
    failures = _read_csv(failures_path)
    source_root = artifact_root or _infer_artifact_root(raw_runs_path)
    lines = [
        "# Isaac Lab Paired Benchmark Report",
        "",
        "> This report is informational only; it defines no performance acceptance threshold.",
        "",
        "## Methodology",
        "",
        "Both versions use PhysX, RSL-RL, 4,096 environments, and paired seeds 42, 43, and 44. "
        "Runtime modes collect 100 or 1,000 environment steps; training runs 100 iterations. "
        "The seed order is counterbalanced and no benchmark warm-up semantics are added.",
        "",
        "Only complete Lab 2/Lab 3 seed pairs contribute to paired statistics. Failures and missing "
        "attempts are not imputed. Sample standard deviations describe repeat variability (a single "
        "pair has a displayed standard deviation of zero).",
        "",
        "The signed delta is `Isaac Lab 3 - Isaac Lab 2`; the percentage delta is "
        "`(Lab 3 - Lab 2) / Lab 2 × 100`. A zero Lab 2 baseline is reported as undefined rather "
        "than infinity. Positive collection-FPS deltas mean higher throughput; resource deltas are "
        "not labeled as inherently better or worse.",
        "",
        "## Pinned revisions and execution identities",
        "",
        "| Version | Exact Git SHA | Environment identity |",
        "|---|---|---|",
    ]
    identities = {(run.version, run.version_sha, run.environment_identity) for run in runs}
    for version in ("lab2", "lab3"):
        entries = sorted(entry for entry in identities if entry[0] == version)
        if entries:
            for _, sha, environment in entries:
                lines.append(f"| {version} | `{_escape(sha)}` | `{_escape(environment)}` |")
        else:
            lines.append(f"| {version} | unavailable | unavailable |")

    lines.extend(["", "## Hardware and software inventory", ""])
    if inventory:
        lines.extend(["| Item | Value |", "|---|---|"])
        lines.extend(f"| {_escape(str(name))} | {_escape(str(value))} |" for name, value in inventory.items())
    else:
        lines.append("No external preflight inventory was supplied.")

    lines.extend(["", "## Task mapping", "", "| Logical task | Isaac Lab 2 task | Isaac Lab 3 task |", "|---|---|---|"])
    mappings = _task_mappings(runs, failures)
    for task in _ordered_tasks(set(mappings)):
        versions = mappings[task]
        lines.append(
            f"| `{_escape(task)}` | `{_escape(versions.get('lab2', 'missing'))}` "
            f"| `{_escape(versions.get('lab3', 'missing'))}` |"
        )
    if not mappings:
        lines.append("| unavailable | missing | missing |")

    for mode in MODE_ORDER:
        lines.extend(["", f"## {mode}", ""])
        mode_runs = [run for run in runs if run.mode == mode]
        mode_summaries = [row for row in summaries if row.get("mode") == mode]
        if mode_summaries:
            lines.extend(
                [
                    "| Task | Metric | Paired seeds | Lab 2 mean ± std | Lab 3 mean ± std | "
                    "Lab 3 - Lab 2 | Delta [%] |",
                    "|---|---|---:|---:|---:|---:|---:|",
                ]
            )
            for row in mode_summaries:
                delta_pct = (
                    f"{float(row['percent_delta']):+.3f}%"
                    if row.get("percent_delta_status") == "available" and row.get("percent_delta")
                    else "undefined (zero Lab 2 baseline)"
                )
                lines.append(
                    f"| `{_escape(row['logical_task'])}` | {_METRIC_LABELS.get(row['metric'], row['metric'])} "
                    f"| {row['paired_seed_count']} | {_number(row['lab2_mean'])} ± {_number(row['lab2_std'])} "
                    f"| {_number(row['lab3_mean'])} ± {_number(row['lab3_std'])} "
                    f"| {float(row['absolute_delta']):+.3f} | {delta_pct} |"
                )
        else:
            lines.append("No valid paired results.")

        lines.extend(["", "Successful individual runs:", ""])
        if mode_runs:
            lines.extend(
                [
                    "| Task | Version | Seed | Collection FPS | Mean GPU memory [MiB] | "
                    "Peak GPU memory [MiB] | Mean GPU utilization [%] | GPU utilization samples | "
                    "Elapsed [s] | Artifact |",
                    "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
                ]
            )
            for run in mode_runs:
                lines.append(
                    f"| `{_escape(run.logical_task)}` | {run.version} | {run.seed} "
                    f"| {run.collection_fps:.3f} | {run.gpu_memory_mean_mib:.3f} "
                    f"| {run.gpu_memory_peak_mib:.3f} | {run.gpu_utilization_mean_pct:.3f} "
                    f"| {run.gpu_utilization_sample_count} | {run.elapsed_time_s:.3f} "
                    f"| {_artifact_link(run.artifact_path, source_root, output_path)} |"
                )
        else:
            lines.append("No successful runs.")

    lines.extend(["", "## Failures and missing attempts", ""])
    if failures:
        lines.extend(
            [
                "| Task | Mode | Version | Seed | Classification | Reason | Artifact |",
                "|---|---|---|---:|---|---|---|",
            ]
        )
        for row in failures:
            artifact = row.get("artifact_path", "")
            link = (
                _artifact_link(artifact, source_root, output_path)
                if artifact and row.get("failure_kind") != "missing"
                else "unavailable"
            )
            lines.append(
                f"| `{_escape(row.get('logical_task', ''))}` | `{_escape(row.get('mode', ''))}` "
                f"| {row.get('version', '')} | {row.get('seed', '')} "
                f"| `{_escape(row.get('failure_kind', ''))}` | {_escape(row.get('reason', ''))} | {link} |"
            )
    else:
        lines.append("No failed or missing attempts were recorded.")

    _write_text_atomic(output_path, "\n".join(lines) + "\n")
    return output_path


def _task_mappings(runs, failures: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    mappings: dict[str, dict[str, str]] = {}
    for run in runs:
        mappings.setdefault(run.logical_task, {})[run.version] = run.concrete_task
    for row in failures:
        logical = row.get("logical_task")
        version = row.get("version")
        concrete = row.get("concrete_task")
        if logical and version and concrete:
            mappings.setdefault(logical, {}).setdefault(version, concrete)
    return mappings


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _ordered_tasks(tasks: set[str]) -> list[str]:
    return sorted(tasks, key=lambda task: (TASK_ORDER.index(task) if task in TASK_ORDER else len(TASK_ORDER), task))


def _number(value: str) -> str:
    return f"{float(value):.3f}"


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _link(value: str) -> str:
    return value.replace(" ", "%20").replace("(", "%28").replace(")", "%29")


def _artifact_link(artifact: str, artifact_root: Path, report_path: Path) -> str:
    target = artifact_root / artifact
    relative = os.path.relpath(target, report_path.parent)
    return f"[{_escape(artifact)}]({_link(Path(relative).as_posix())})"


def _infer_artifact_root(raw_runs_path: Path) -> Path:
    normalized_parent = raw_runs_path.parent.parent
    if normalized_parent.name in {"final", "canary"}:
        return normalized_parent.parent
    return normalized_parent


def _write_text_atomic(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents, encoding="utf-8")
    os.replace(temporary, path)
