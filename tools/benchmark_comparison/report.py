# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate an informational Markdown report from normalized benchmark CSV."""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

from .manifest import RunSetManifest, read_manifest, resolve_manifest_expansion, validate_manifest
from .models import ExecutionProvenance, MatrixExpansion
from .normalize import (
    FAILURE_FIELDS,
    PAIRED_SUMMARY_FIELDS,
    expansion_orders,
    read_raw_runs_csv,
    summarize_pairs,
)
from .report_paths import validate_artifact_path

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
    manifest: RunSetManifest | Path,
    artifact_root: Path | None = None,
) -> Path:
    """Write a deterministic report without reading simulator logs or schemas."""
    runs = read_raw_runs_csv(raw_runs_path)
    expected_manifest = read_manifest(manifest) if isinstance(manifest, Path) else validate_manifest(manifest)
    expected_provenance = expected_manifest.provenance
    source_root = (artifact_root or _infer_artifact_root(raw_runs_path)).resolve()
    expansion = resolve_manifest_expansion(expected_manifest, source_root)
    task_order, mode_order, _task_modes = expansion_orders(expansion)
    attempt_directories = _attempt_directory_index(expansion)
    _validate_run_manifest(runs, expected_manifest, source_root, attempt_directories)
    summaries = _read_csv(paired_summary_path, PAIRED_SUMMARY_FIELDS)
    failures = _read_csv(failures_path, FAILURE_FIELDS)
    _validate_summaries(runs, summaries, expansion)
    _validate_failures(failures, expected_manifest, source_root, attempt_directories)
    lines = [
        "# Isaac Lab Paired Benchmark Report",
        "",
        "> This report is informational only; it defines no performance acceptance threshold.",
        "",
        "## Methodology",
        "",
        *_methodology(expected_manifest),
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
    for version in ("lab2", "lab3"):
        lines.append(
            f"| {version} | `{expected_provenance.version_sha(version)}` "
            f"| `{expected_provenance.environment_identity(version)}` |"
        )

    gpu_index = expected_manifest.host.gpu_index
    gpu_index_text = str(gpu_index) if gpu_index is not None else chr(8212)
    lines.extend(
        [
            "",
            "## Hardware and software inventory",
            "",
            "| Host item | Value |",
            "|---|---|",
            f"| Hostname | {_escape(expected_manifest.host.hostname)} |",
            f"| Operating system | {_escape(expected_manifest.host.os)} |",
            f"| CPU | {_escape(expected_manifest.host.cpu_model)} |",
            f"| Logical CPUs | {expected_manifest.host.logical_cpu_count} |",
            f"| GPU | {_escape(expected_manifest.host.gpu_model)} |",
            f"| Physical GPU index | {gpu_index_text} |",
            f"| GPU UUID | {_escape(expected_manifest.host.gpu_uuid or chr(8212))} |",
            f"| NVIDIA driver | {_escape(expected_manifest.host.gpu_driver)} |",
            f"| CUDA | {_escape(expected_manifest.host.cuda_version or chr(8212))} |",
            "",
            "| Version | Isaac Lab | Isaac Sim | Python | PyTorch | RSL-RL |",
            "|---|---|---|---|---|---|",
            _software_row("lab2", expected_manifest.lab2),
            _software_row("lab3", expected_manifest.lab3),
        ]
    )
    if expected_manifest.cpu_power_profile is not None:
        lines.extend(
            [
                "",
                f"> Interpretation note: the manifest records the CPU power profile as "
                f"`{_escape(expected_manifest.cpu_power_profile)}`; absolute throughput reflects that host setting.",
            ]
        )

    lines.extend(["", "## Task mapping", "", "| Logical task | Isaac Lab 2 task | Isaac Lab 3 task |", "|---|---|---|"])
    mappings = _task_mappings(runs, failures)
    for task in _ordered_tasks(set(mappings), task_order):
        versions = mappings[task]
        lines.append(
            f"| `{_escape(task)}` | `{_escape(versions.get('lab2', 'missing'))}` "
            f"| `{_escape(versions.get('lab3', 'missing'))}` |"
        )
    if not mappings:
        lines.append("| unavailable | missing | missing |")

    for mode in mode_order:
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


def write_provenance(path: Path, provenance: ExecutionProvenance) -> Path:
    """Atomically write the exact preflight identities consumed by reports."""
    if path.exists():
        if read_provenance(path) != provenance:
            raise ValueError(f"refusing to replace different benchmark provenance: {path}")
        return path
    contents = json.dumps(provenance.to_json(), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    _write_text_atomic(path, contents)
    return path


def read_provenance(path: Path) -> ExecutionProvenance:
    """Read the exact preflight identities consumed by reports."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read benchmark provenance: {error}") from error
    fields = {"lab2_sha", "lab3_sha", "lab2_image_id", "uv_lock_sha256"}
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or not all(isinstance(value[field], str) for field in fields)
    ):
        raise ValueError("benchmark provenance must contain exactly four string identity fields")
    return ExecutionProvenance(**value)


_FAILED_ARTIFACT_DIRECTORY = re.compile(r"attempt-[0-9]{4,}-[a-z_]+")
_QUARANTINED_SUCCESS_DIRECTORY = re.compile(r"corrupt-success-[0-9]{4,}")


def _attempt_directory_index(expansion: MatrixExpansion) -> dict[tuple[str, ...], str]:
    return {
        _attempt_key(
            attempt.version.value,
            attempt.logical_task,
            attempt.concrete_task,
            attempt.mode.id,
            attempt.bound.value,
            attempt.bound.unit.value,
            attempt.seed,
            attempt.num_envs,
        ): attempt.run_directory
        for attempt in expansion.attempts
    }


def _attempt_key(
    version: str,
    logical_task: str,
    concrete_task: str,
    mode: str,
    bound: int | str,
    bound_unit: str,
    seed: int | str,
    num_envs: int | str,
) -> tuple[str, ...]:
    return (version, logical_task, concrete_task, mode, str(bound), bound_unit, str(seed), str(num_envs))


def _expected_attempt_directory(index: dict[tuple[str, ...], str], key: tuple[str, ...], path: str) -> str:
    try:
        return index[key]
    except KeyError as error:
        raise ValueError(f"artifact path identity does not match manifest run-set identity: {path}") from error


def _validate_run_manifest(
    runs,
    manifest: RunSetManifest,
    artifact_root: Path,
    attempt_directories: dict[tuple[str, ...], str],
) -> None:
    for run in runs:
        expected_sha = manifest.provenance.version_sha(run.version)
        expected_environment = manifest.provenance.environment_identity(run.version)
        software = manifest.software(run.version)
        actual_software = (
            run.isaac_lab_version,
            run.isaac_sim_version,
            run.python_version,
            run.pytorch_version,
            run.rsl_rl_version,
        )
        expected_software = (
            software.isaac_lab,
            software.isaac_sim,
            software.python,
            software.pytorch,
            software.rsl_rl,
        )
        if run.version_sha != expected_sha or run.environment_identity != expected_environment:
            raise ValueError("normalized run provenance does not match manifest: " + run.artifact_path)
        if actual_software != expected_software:
            raise ValueError("normalized run software versions do not match manifest: " + run.artifact_path)
        parts = validate_artifact_path(run.artifact_path, manifest.run_set, artifact_root)
        expected_directory = _expected_attempt_directory(
            attempt_directories,
            _attempt_key(
                run.version,
                run.logical_task,
                run.concrete_task,
                run.mode,
                run.bound,
                run.bound_unit,
                run.seed,
                run.num_envs,
            ),
            run.artifact_path,
        )
        if "/".join(parts[:2]) != expected_directory or parts[2:] != ("success",):
            raise ValueError("artifact path is not the expected immutable success: " + run.artifact_path)


def _validate_summaries(runs, summaries: list[dict[str, str]], expansion: MatrixExpansion) -> None:
    expected = [summary.to_csv_row() for summary in summarize_pairs(runs, expansion=expansion)]
    if summaries != expected:
        raise ValueError("paired summary is not derived from normalized raw runs")


def _validate_failures(
    failures: list[dict[str, str]],
    manifest: RunSetManifest,
    artifact_root: Path,
    attempt_directories: dict[tuple[str, ...], str],
) -> None:
    for row in failures:
        if row["version"] not in {"lab2", "lab3"}:
            raise ValueError("failure version is not recognized")
        artifact_path = row["artifact_path"]
        parts = validate_artifact_path(artifact_path, manifest.run_set, artifact_root)
        expected_directory = _expected_attempt_directory(
            attempt_directories,
            _attempt_key(
                row["version"],
                row["logical_task"],
                row["concrete_task"],
                row["mode"],
                row["bound"],
                row["bound_unit"],
                row["seed"],
                row["num_envs"],
            ),
            artifact_path,
        )
        if "/".join(parts[:2]) != expected_directory:
            raise ValueError("artifact path does not match the expected benchmark attempt: " + artifact_path)
        terminal = parts[2] if len(parts) == 3 else None
        if row["failure_kind"] == "missing":
            valid_terminal = terminal is None
        elif row["failure_kind"] == "invalid_success":
            valid_terminal = terminal == "success" or (
                terminal is not None and _QUARANTINED_SUCCESS_DIRECTORY.fullmatch(terminal) is not None
            )
        else:
            valid_terminal = terminal is not None and _FAILED_ARTIFACT_DIRECTORY.fullmatch(terminal) is not None
        if not valid_terminal:
            raise ValueError("artifact path terminal does not match failure classification: " + artifact_path)


def _methodology(manifest: RunSetManifest) -> tuple[str, ...]:
    identity = f"Run set: `{manifest.run_set.value}`; phase: `{manifest.phase}`."
    if manifest.run_set.value == "canary":
        bounds = (
            "Both versions use PhysX, RSL-RL, 4,096 environments, and paired seed 42. "
            "Runtime modes collect 10 or 25 environment steps; training runs 2 iterations. "
            "No benchmark warm-up semantics are added."
        )
    else:
        bounds = (
            "Both versions use PhysX, RSL-RL, 4,096 environments, and paired seeds 42, 43, and 44. "
            "Runtime modes collect 100 or 1,000 environment steps; training runs 100 iterations. "
            "The seed order is counterbalanced and no benchmark warm-up semantics are added."
        )
    return identity, "", bounds


def _software_row(version, software) -> str:
    return (
        f"| {version} | {_escape(software.isaac_lab)} | {_escape(software.isaac_sim)} "
        f"| {_escape(software.python)} | {_escape(software.pytorch)} | {_escape(software.rsl_rl)} |"
    )


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


def _read_csv(path: Path, expected_fields: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise ValueError(f"unexpected {path.name} columns: {reader.fieldnames}")
        return list(reader)


def _ordered_tasks(tasks: set[str], task_order: tuple[str, ...]) -> list[str]:
    return sorted(tasks, key=lambda task: (task_order.index(task) if task in task_order else len(task_order), task))


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
