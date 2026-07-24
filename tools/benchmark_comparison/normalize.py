# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Normalize immutable benchmark artifacts into deterministic CSV tables."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .artifacts import verify_checksums
from .manifest import RunSetManifest, SoftwareIdentity, validate_manifest
from .matrix import load_matrix
from .models import BenchmarkAttempt, ExecutionProvenance, MatrixExpansion
from .validate import validate_attempt_directory, validation_document

VERSION_ORDER = ("lab2", "lab3")
_MATRIX = load_matrix()
TASK_ORDER = tuple(task.alias for task in _MATRIX.tasks)
MODE_ORDER = tuple(mode.id for mode in _MATRIX.modes)
TASK_MODES = {
    task.alias: tuple(mode.id for mode in _MATRIX.modes if task.supports_mode(mode.id)) for task in _MATRIX.tasks
}
SUMMARY_METRICS = (
    "collection_fps",
    "gpu_memory_mean_mib",
    "gpu_memory_peak_mib",
    "gpu_utilization_mean_pct",
    "gpu_utilization_sample_count",
    "elapsed_time_s",
)

RAW_RUN_FIELDS = (
    "version",
    "version_sha",
    "environment_identity",
    "isaac_lab_version",
    "isaac_sim_version",
    "python_version",
    "pytorch_version",
    "rsl_rl_version",
    "logical_task",
    "concrete_task",
    "mode",
    "bound",
    "bound_unit",
    "seed",
    "num_envs",
    "collection_fps",
    "gpu_memory_mean_mib",
    "gpu_memory_peak_mib",
    "gpu_utilization_mean_pct",
    "gpu_utilization_sample_count",
    "elapsed_time_s",
    "artifact_path",
)
PAIRED_SUMMARY_FIELDS = (
    "logical_task",
    "mode",
    "metric",
    "paired_seed_count",
    "lab2_mean",
    "lab2_std",
    "lab3_mean",
    "lab3_std",
    "absolute_delta",
    "percent_delta",
    "percent_delta_status",
)
FAILURE_FIELDS = (
    "version",
    "logical_task",
    "concrete_task",
    "mode",
    "bound",
    "bound_unit",
    "seed",
    "num_envs",
    "attempt_number",
    "failure_kind",
    "reason",
    "artifact_path",
)

_FAILED_DIRECTORY = re.compile(r"attempt-(?P<number>[0-9]+)-(?P<kind>[a-z_]+)$")
_CORRUPT_SUCCESS_DIRECTORY = re.compile(r"corrupt-success-(?P<number>[0-9]+)$")


def expansion_orders(
    expansion: MatrixExpansion | None,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Return deterministic task, mode, and capability order for an expansion."""
    if expansion is None:
        return TASK_ORDER, MODE_ORDER, TASK_MODES
    ordered_pairs = tuple(sorted(expansion.pairs, key=lambda pair: pair.pair_order))
    task_order = tuple(dict.fromkeys(pair.logical_task for pair in ordered_pairs))
    mode_order = tuple(dict.fromkeys(pair.mode.id for pair in ordered_pairs))
    task_modes = {
        task: tuple(dict.fromkeys(pair.mode.id for pair in ordered_pairs if pair.logical_task == task))
        for task in task_order
    }
    return task_order, mode_order, task_modes


def task_order_for_mode(mode: str, expansion: MatrixExpansion | None = None) -> tuple[str, ...]:
    """Return canonical tasks that support ``mode``."""
    task_order, mode_order, task_modes = expansion_orders(expansion)
    if mode not in mode_order:
        raise ValueError(f"unknown benchmark mode: {mode}")
    return tuple(task for task in task_order if mode in task_modes[task])


@dataclass(frozen=True)
class NormalizedRun:
    """One successful version-specific benchmark attempt."""

    version: str
    version_sha: str
    environment_identity: str
    logical_task: str
    concrete_task: str
    mode: str
    bound: int
    bound_unit: str
    seed: int
    num_envs: int
    collection_fps: float
    gpu_memory_mean_mib: float
    gpu_memory_peak_mib: float
    gpu_utilization_mean_pct: float
    gpu_utilization_sample_count: int
    elapsed_time_s: float
    artifact_path: str
    isaac_lab_version: str = ""
    isaac_sim_version: str = ""
    python_version: str = ""
    pytorch_version: str = ""
    rsl_rl_version: str = ""

    def to_csv_row(self) -> dict[str, str]:
        """Return the stable string projection used by ``raw_runs.csv``."""
        return {
            "version": self.version,
            "version_sha": self.version_sha,
            "environment_identity": self.environment_identity,
            "isaac_lab_version": self.isaac_lab_version,
            "isaac_sim_version": self.isaac_sim_version,
            "python_version": self.python_version,
            "pytorch_version": self.pytorch_version,
            "rsl_rl_version": self.rsl_rl_version,
            "logical_task": self.logical_task,
            "concrete_task": self.concrete_task,
            "mode": self.mode,
            "bound": str(self.bound),
            "bound_unit": self.bound_unit,
            "seed": str(self.seed),
            "num_envs": str(self.num_envs),
            "collection_fps": _format_float(self.collection_fps),
            "gpu_memory_mean_mib": _format_float(self.gpu_memory_mean_mib),
            "gpu_memory_peak_mib": _format_float(self.gpu_memory_peak_mib),
            "gpu_utilization_mean_pct": _format_float(self.gpu_utilization_mean_pct),
            "gpu_utilization_sample_count": str(self.gpu_utilization_sample_count),
            "elapsed_time_s": _format_float(self.elapsed_time_s),
            "artifact_path": self.artifact_path,
        }


@dataclass(frozen=True)
class FailureRow:
    """One failed, invalid, or missing matrix attempt."""

    version: str
    logical_task: str
    concrete_task: str
    mode: str
    bound: int
    bound_unit: str
    seed: int
    num_envs: int
    attempt_number: int | None
    failure_kind: str
    reason: str
    artifact_path: str

    def to_csv_row(self) -> dict[str, str]:
        """Return the stable string projection used by ``failures.csv``."""
        return {
            "version": self.version,
            "logical_task": self.logical_task,
            "concrete_task": self.concrete_task,
            "mode": self.mode,
            "bound": str(self.bound),
            "bound_unit": self.bound_unit,
            "seed": str(self.seed),
            "num_envs": str(self.num_envs),
            "attempt_number": "" if self.attempt_number is None else str(self.attempt_number),
            "failure_kind": self.failure_kind,
            "reason": self.reason,
            "artifact_path": self.artifact_path,
        }


@dataclass(frozen=True)
class PairedSummary:
    """Across-seed statistics computed only from complete version pairs."""

    logical_task: str
    mode: str
    metric: str
    paired_seed_count: int
    lab2_mean: float
    lab2_std: float
    lab3_mean: float
    lab3_std: float
    absolute_delta: float
    percent_delta: float | None
    percent_delta_status: str

    def to_csv_row(self) -> dict[str, str]:
        """Return the stable string projection used by ``paired_summary.csv``."""
        return {
            "logical_task": self.logical_task,
            "mode": self.mode,
            "metric": self.metric,
            "paired_seed_count": str(self.paired_seed_count),
            "lab2_mean": _format_float(self.lab2_mean),
            "lab2_std": _format_float(self.lab2_std),
            "lab3_mean": _format_float(self.lab3_mean),
            "lab3_std": _format_float(self.lab3_std),
            "absolute_delta": _format_float(self.absolute_delta),
            "percent_delta": "" if self.percent_delta is None else _format_float(self.percent_delta),
            "percent_delta_status": self.percent_delta_status,
        }


def normalize_run_set(
    root: Path,
    expansion: MatrixExpansion,
    manifest: RunSetManifest,
) -> tuple[tuple[NormalizedRun, ...], tuple[FailureRow, ...]]:
    """Normalize every terminal artifact selected by an expanded matrix.

    Raw artifact files are only read. Invalid successes and missing attempts
    are preserved as failure rows and never converted to zero-valued runs.
    """
    manifest = validate_manifest(manifest)
    if manifest.run_set is not expansion.run_set:
        raise ValueError("manifest run_set does not match matrix expansion")
    if manifest.expansion is not None and manifest.expansion != expansion:
        raise ValueError("matrix expansion does not match manifest run-set identity")
    task_order, mode_order, _task_modes = expansion_orders(expansion)
    runs: list[NormalizedRun] = []
    failures: list[FailureRow] = []
    for attempt in sorted(
        expansion.attempts,
        key=lambda value: (
            _order(task_order, value.logical_task),
            _order(mode_order, value.mode.id),
            value.seed,
            _order(VERSION_ORDER, value.version.value),
        ),
    ):
        attempt_root = root / attempt.run_directory
        success_path = attempt_root / "success"
        success_error: str | None = None
        if success_path.is_dir():
            run, success_error = _read_success(root, success_path, attempt, manifest)
            if run is not None:
                runs.append(run)

        attempt_failures = _read_failures(root, attempt_root, attempt, manifest)
        quarantine_failures = _read_quarantines(root, attempt_root, attempt, manifest)
        failures.extend((*attempt_failures, *quarantine_failures))
        if success_error is not None:
            failures.append(_failure_from_attempt(root, attempt, "invalid_success", success_error, success_path))
        elif not success_path.is_dir() and not attempt_failures and not quarantine_failures:
            failures.append(_failure_from_attempt(root, attempt, "missing", "no terminal artifact", attempt_root))
    return tuple(runs), tuple(failures)


def summarize_pairs(
    runs: Sequence[NormalizedRun], *, expansion: MatrixExpansion | None = None
) -> tuple[PairedSummary, ...]:
    """Summarize metrics across seeds for complete Lab 2/Lab 3 pairs only."""
    task_order, mode_order, _task_modes = expansion_orders(expansion)
    indexed: dict[tuple[str, str, int, str], NormalizedRun] = {}
    for run in runs:
        key = (run.logical_task, run.mode, run.seed, run.version)
        if key in indexed:
            raise ValueError(f"duplicate normalized run: {key}")
        indexed[key] = run

    summaries: list[PairedSummary] = []
    task_modes = {(run.logical_task, run.mode) for run in runs}
    for logical_task, mode in sorted(
        task_modes, key=lambda value: (_order(task_order, value[0]), _order(mode_order, value[1]))
    ):
        seeds = sorted({key[2] for key in indexed if key[:2] == (logical_task, mode)})
        paired = [
            (indexed[(logical_task, mode, seed, "lab2")], indexed[(logical_task, mode, seed, "lab3")])
            for seed in seeds
            if (logical_task, mode, seed, "lab2") in indexed and (logical_task, mode, seed, "lab3") in indexed
        ]
        if not paired:
            continue
        for lab2, lab3 in paired:
            _validate_pair_invariants(lab2, lab3)
        for metric in SUMMARY_METRICS:
            lab2_values = [float(getattr(pair[0], metric)) for pair in paired]
            lab3_values = [float(getattr(pair[1], metric)) for pair in paired]
            lab2_mean = statistics.fmean(lab2_values)
            lab3_mean = statistics.fmean(lab3_values)
            absolute_delta = lab3_mean - lab2_mean
            if lab2_mean == 0.0:
                percent_delta = None
                percent_status = "undefined_zero_baseline"
            else:
                percent_delta = absolute_delta / lab2_mean * 100.0
                percent_status = "available"
            summaries.append(
                PairedSummary(
                    logical_task=logical_task,
                    mode=mode,
                    metric=metric,
                    paired_seed_count=len(paired),
                    lab2_mean=lab2_mean,
                    lab2_std=statistics.stdev(lab2_values) if len(lab2_values) > 1 else 0.0,
                    lab3_mean=lab3_mean,
                    lab3_std=statistics.stdev(lab3_values) if len(lab3_values) > 1 else 0.0,
                    absolute_delta=absolute_delta,
                    percent_delta=percent_delta,
                    percent_delta_status=percent_status,
                )
            )
    return tuple(summaries)


def write_normalized_outputs(
    output_directory: Path,
    runs: Sequence[NormalizedRun],
    failures: Sequence[FailureRow],
    *,
    expansion: MatrixExpansion | None = None,
) -> dict[str, Path]:
    """Atomically write all three deterministic normalized CSV tables."""
    task_order, mode_order, _task_modes = expansion_orders(expansion)
    output_directory.mkdir(parents=True, exist_ok=True)
    raw_runs_path = write_raw_runs_csv(output_directory / "raw_runs.csv", runs, expansion=expansion)
    serialized_runs = read_raw_runs_csv(raw_runs_path)
    paths = {
        "raw_runs": raw_runs_path,
        "paired_summary": _write_csv(
            output_directory / "paired_summary.csv",
            PAIRED_SUMMARY_FIELDS,
            (summary.to_csv_row() for summary in summarize_pairs(serialized_runs, expansion=expansion)),
        ),
        "failures": _write_csv(
            output_directory / "failures.csv",
            FAILURE_FIELDS,
            (
                failure.to_csv_row()
                for failure in sorted(
                    failures,
                    key=lambda row: (
                        _order(task_order, row.logical_task),
                        _order(mode_order, row.mode),
                        row.seed,
                        _order(VERSION_ORDER, row.version),
                        row.attempt_number or 0,
                    ),
                )
            ),
        ),
    }
    return paths


def write_raw_runs_csv(path: Path, runs: Sequence[NormalizedRun], *, expansion: MatrixExpansion | None = None) -> Path:
    """Atomically write stable successful-attempt rows."""
    task_order, mode_order, _task_modes = expansion_orders(expansion)
    ordered = sorted(
        runs,
        key=lambda run: (
            _order(task_order, run.logical_task),
            _order(mode_order, run.mode),
            run.seed,
            _order(VERSION_ORDER, run.version),
        ),
    )
    return _write_csv(path, RAW_RUN_FIELDS, (run.to_csv_row() for run in ordered))


def read_raw_runs_csv(path: Path) -> tuple[NormalizedRun, ...]:
    """Read normalized successful attempts without simulator dependencies."""
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != RAW_RUN_FIELDS:
            raise ValueError(f"unexpected raw_runs.csv columns: {reader.fieldnames}")
        return tuple(_run_from_csv(row) for row in reader)


def _read_success(
    root: Path,
    path: Path,
    attempt: BenchmarkAttempt,
    manifest: RunSetManifest,
) -> tuple[NormalizedRun | None, str | None]:
    if not verify_checksums(path):
        return None, "success checksum or file layout verification failed"
    result = validate_attempt_directory(path, attempt)
    if not result.succeeded or result.metrics is None:
        return None, result.reason or "success semantic validation failed"
    try:
        environment = _read_mapping(path / "environment.json")
        exit_status = _read_mapping(path / "exit.json")
        validation = _read_mapping(path / "validation.json")
        attempt_number = validation.get("attempt_number")
        if not isinstance(attempt_number, int) or isinstance(attempt_number, bool):
            raise ValueError("validation attempt_number must be an integer")
        if validation != validation_document(result, attempt_number):
            raise ValueError("validation.json does not match source artifacts")
        wall_time = _finite_number(exit_status.get("wall_time_s"), "exit.wall_time_s")
        if wall_time < 0:
            raise ValueError("exit.wall_time_s must be non-negative")
        _validate_environment_provenance(environment, attempt, manifest.provenance)
        _validate_selected_gpu(environment, manifest)
        version_sha = environment.get(f"{attempt.version.value}_sha")
        if not isinstance(version_sha, str) or not re.fullmatch(r"[0-9a-f]{40,64}", version_sha):
            raise ValueError(f"environment is missing exact {attempt.version.value} SHA")
        environment_identity = _environment_identity(environment, attempt.version.value, version_sha)
        schema = _read_mapping(path / "schema.json")
        stdout = (path / "stdout.log").read_text(encoding="utf-8")
        software = _validate_schema_identity(schema, stdout, attempt, manifest)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError) as error:
        return None, str(error)
    metrics = result.metrics
    return (
        NormalizedRun(
            version=attempt.version.value,
            version_sha=version_sha,
            environment_identity=environment_identity,
            isaac_lab_version=software.isaac_lab,
            isaac_sim_version=software.isaac_sim,
            python_version=software.python,
            pytorch_version=software.pytorch,
            rsl_rl_version=software.rsl_rl,
            logical_task=attempt.logical_task,
            concrete_task=attempt.concrete_task,
            mode=attempt.mode.id,
            bound=attempt.bound.value,
            bound_unit=attempt.bound.unit.value,
            seed=attempt.seed,
            num_envs=attempt.num_envs,
            collection_fps=metrics.collection_fps,
            gpu_memory_mean_mib=metrics.gpu_memory_mean_mib,
            gpu_memory_peak_mib=metrics.gpu_memory_peak_mib,
            gpu_utilization_mean_pct=metrics.gpu_utilization_mean_pct,
            gpu_utilization_sample_count=metrics.gpu_utilization_sample_count,
            elapsed_time_s=wall_time,
            artifact_path=_relative_artifact(root, path),
        ),
        None,
    )


def _read_quarantines(
    root: Path,
    attempt_root: Path,
    attempt: BenchmarkAttempt,
    manifest: RunSetManifest,
) -> tuple[FailureRow, ...]:
    if not attempt_root.is_dir():
        return ()
    rows: list[FailureRow] = []
    for path in sorted(attempt_root.iterdir()):
        match = _CORRUPT_SUCCESS_DIRECTORY.fullmatch(path.name)
        if match is None or not path.is_dir():
            continue
        _quarantined_run, error = _read_success(root, path, attempt, manifest)
        reason = error or "quarantined success is not the current immutable success"
        rows.append(
            _failure_from_attempt(
                root,
                attempt,
                "invalid_success",
                reason,
                path,
                attempt_number=int(match.group("number")),
            )
        )
    return tuple(rows)


def _read_failures(
    root: Path,
    attempt_root: Path,
    attempt: BenchmarkAttempt,
    manifest: RunSetManifest,
) -> tuple[FailureRow, ...]:
    if not attempt_root.is_dir():
        return ()
    rows: list[FailureRow] = []
    for path in sorted(attempt_root.iterdir()):
        match = _FAILED_DIRECTORY.fullmatch(path.name)
        if match is None or not path.is_dir():
            continue
        kind = match.group("kind")
        reason = f"artifact directory classified as {kind}"
        if verify_checksums(path):
            try:
                validation = _read_mapping(path / "validation.json")
                attempt_number = int(match.group("number"))
                result = validate_attempt_directory(path, attempt)
                if result.succeeded or result.failure_kind is None:
                    raise ValueError("failure directory semantically validates as success")
                if validation != validation_document(result, attempt_number):
                    raise ValueError("failure validation does not match source artifacts")
                _validate_failure_identity(path, attempt, manifest)
                kind = result.failure_kind.value
                reason = result.reason or f"artifact directory classified as {kind}"
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
                kind = "malformed_artifact"
                reason = str(error) or "failure validation document is inconsistent"
        else:
            kind = "malformed_artifact"
            reason = "failure checksum or file layout verification failed"
        rows.append(
            _failure_from_attempt(
                root,
                attempt,
                kind,
                reason,
                path,
                attempt_number=int(match.group("number")),
            )
        )
    return tuple(rows)


def _validate_failure_identity(path: Path, attempt: BenchmarkAttempt, manifest: RunSetManifest) -> None:
    environment = _read_mapping(path / "environment.json")
    _validate_environment_provenance(environment, attempt, manifest.provenance)
    _validate_selected_gpu(environment, manifest)
    schema_value = json.loads((path / "schema.json").read_text(encoding="utf-8"))
    if isinstance(schema_value, Mapping) and {"versions", "hardware"}.issubset(schema_value):
        stdout = (path / "stdout.log").read_text(encoding="utf-8")
        _validate_schema_identity(schema_value, stdout, attempt, manifest)


def _failure_from_attempt(
    root: Path,
    attempt: BenchmarkAttempt,
    kind: str,
    reason: str,
    path: Path,
    attempt_number: int | None = None,
) -> FailureRow:
    return FailureRow(
        version=attempt.version.value,
        logical_task=attempt.logical_task,
        concrete_task=attempt.concrete_task,
        mode=attempt.mode.id,
        bound=attempt.bound.value,
        bound_unit=attempt.bound.unit.value,
        seed=attempt.seed,
        num_envs=attempt.num_envs,
        attempt_number=attempt_number,
        failure_kind=kind,
        reason=reason,
        artifact_path=_relative_artifact(root, path),
    )


def _environment_identity(environment: Mapping[str, object], version: str, version_sha: str) -> str:
    explicit = environment.get("environment_identity")
    if isinstance(explicit, str) and explicit:
        return explicit
    if version == "lab2":
        image_id = environment.get("lab2_image_id")
        if isinstance(image_id, str) and image_id:
            return image_id
    else:
        lock = environment.get("uv_lock_sha256")
        if isinstance(lock, str) and lock:
            return f"uv-lock:{lock}"
    return f"git:{version_sha}"


def _validate_environment_provenance(
    environment: Mapping[str, object],
    attempt: BenchmarkAttempt,
    provenance: ExecutionProvenance,
) -> None:
    expected = provenance.to_json()
    for field, value in expected.items():
        if environment.get(field) != value:
            raise ValueError(f"environment provenance {field} does not match preflight")
    expected_identity = provenance.environment_identity(attempt.version)
    if environment.get("environment_identity") != expected_identity:
        raise ValueError("environment provenance identity does not match preflight")


def _validate_selected_gpu(environment: Mapping[str, object], manifest: RunSetManifest) -> None:
    """Validate physical GPU 0 attestation for schema-2 artifacts."""
    if manifest.schema_version == "1.0":
        return
    expected_uuid = manifest.host.gpu_uuid
    if expected_uuid is None:
        raise ValueError("manifest selected GPU UUID is missing")
    selected_gpu = environment.get("selected_gpu")
    if not isinstance(selected_gpu, Mapping) or set(selected_gpu) != {"physical_index", "uuid"}:
        raise ValueError("environment selected GPU attestation is missing")
    if selected_gpu.get("physical_index") != 0:
        raise ValueError("environment selected GPU physical index does not match manifest")
    if selected_gpu.get("uuid") != expected_uuid:
        raise ValueError("environment selected GPU UUID does not match manifest")
    values = environment.get("values")
    if not isinstance(values, Mapping):
        raise ValueError("environment selected GPU variables are missing")
    expected_variables = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "0",
        "NVIDIA_VISIBLE_DEVICES": "0",
        "ISAACLAB_BENCHMARK_GPU_INDEX": "0",
        "ISAACLAB_BENCHMARK_GPU_UUID": expected_uuid,
    }
    for name, expected in expected_variables.items():
        if values.get(name) != expected:
            raise ValueError(f"environment selected GPU variable {name} does not match manifest")


def _validate_schema_identity(
    schema: Mapping[str, object],
    stdout: str,
    attempt: BenchmarkAttempt,
    manifest: RunSetManifest,
) -> SoftwareIdentity:
    software = manifest.software(attempt.version.value)
    versions = schema.get("versions")
    if not isinstance(versions, Mapping):
        raise ValueError("schema versions must be an object")
    expected_versions = {
        "isaaclab_release": software.isaac_lab,
        "isaacsim": software.isaac_sim,
        "torch": software.pytorch,
        "rsl_rl": software.rsl_rl,
    }
    labels = {"isaaclab_release": "isaac_lab", "isaacsim": "isaac_sim", "torch": "pytorch", "rsl_rl": "rsl_rl"}
    for field, expected in expected_versions.items():
        if versions.get(field) != expected:
            raise ValueError(f"schema versions.{labels[field]} does not match manifest")
    hardware = schema.get("hardware")
    if not isinstance(hardware, Mapping):
        raise ValueError("schema hardware must be an object")
    expected_hardware = {
        "hostname": manifest.host.hostname,
        "cpu_name": manifest.host.cpu_model,
        "cpu_count": manifest.host.logical_cpu_count,
    }
    for field, expected in expected_hardware.items():
        if hardware.get(field) != expected:
            raise ValueError(f"schema hardware.{field} does not match manifest")
    devices = hardware.get("gpu_devices")
    if not isinstance(devices, list) or len(devices) != 1 or not isinstance(devices[0], Mapping):
        raise ValueError("schema hardware.gpu_devices must identify exactly one GPU")
    if devices[0].get("name") != manifest.host.gpu_model:
        raise ValueError("schema hardware.gpu_model does not match manifest")
    driver_match = re.search(r"Driver Version:\s*([^|\s]+)", stdout)
    if driver_match is not None and driver_match.group(1) != manifest.host.gpu_driver:
        raise ValueError("GPU driver does not match manifest")
    return software


def _validate_pair_invariants(lab2: NormalizedRun, lab3: NormalizedRun) -> None:
    invariants = ("logical_task", "mode", "seed", "bound", "bound_unit", "num_envs")
    for field in invariants:
        if getattr(lab2, field) != getattr(lab3, field):
            raise ValueError(f"paired runs have mismatched {field}")


def _run_from_csv(row: Mapping[str, str]) -> NormalizedRun:
    return NormalizedRun(
        version=row["version"],
        version_sha=row["version_sha"],
        environment_identity=row["environment_identity"],
        isaac_lab_version=row["isaac_lab_version"],
        isaac_sim_version=row["isaac_sim_version"],
        python_version=row["python_version"],
        pytorch_version=row["pytorch_version"],
        rsl_rl_version=row["rsl_rl_version"],
        logical_task=row["logical_task"],
        concrete_task=row["concrete_task"],
        mode=row["mode"],
        bound=int(row["bound"]),
        bound_unit=row["bound_unit"],
        seed=int(row["seed"]),
        num_envs=int(row["num_envs"]),
        collection_fps=_finite_number(row["collection_fps"], "collection_fps"),
        gpu_memory_mean_mib=_finite_number(row["gpu_memory_mean_mib"], "gpu_memory_mean_mib"),
        gpu_memory_peak_mib=_finite_number(row["gpu_memory_peak_mib"], "gpu_memory_peak_mib"),
        gpu_utilization_mean_pct=_finite_number(
            row["gpu_utilization_mean_pct"],
            "gpu_utilization_mean_pct",
        ),
        gpu_utilization_sample_count=int(row["gpu_utilization_sample_count"]),
        elapsed_time_s=_finite_number(row["elapsed_time_s"], "elapsed_time_s"),
        artifact_path=row["artifact_path"],
    )


def _read_mapping(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, str]]) -> Path:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _write_text_atomic(path, output.getvalue())
    return path


def _write_text_atomic(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents, encoding="utf-8")
    os.replace(temporary, path)


def _relative_artifact(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _finite_number(value: object, name: str) -> float:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _format_float(value: float) -> str:
    return format(value, ".12g")


def _order(order: Sequence[str], value: str) -> tuple[int, str]:
    try:
        return order.index(value), value
    except ValueError:
        return len(order), value
