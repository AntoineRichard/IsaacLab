# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the identity and semantic measurements of benchmark artifacts."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .models import BenchmarkAttempt, BoundUnit, Version


class FailureKind(str, Enum):
    """Distinct reasons why a benchmark attempt cannot be used."""

    SETUP = "setup"
    LAUNCH = "launch"
    TIMEOUT = "timeout"
    INTERRUPTED = "interrupted"
    OUT_OF_MEMORY = "out_of_memory"
    NONZERO_EXIT = "nonzero_exit"
    MALFORMED_ARTIFACT = "malformed_artifact"
    IDENTITY_MISMATCH = "identity_mismatch"
    MISSING_METRIC = "missing_metric"


@dataclass(frozen=True)
class SemanticMetrics:
    """Canonical metrics extracted from a successful schema bundle.

    Attributes:
        phase_timings_s: Phase timings [s], keyed by phase name.
        collection_fps: Mean collection throughput [FPS].
        gpu_memory_mean_mib: Mean GPU memory use [MiB].
        gpu_memory_peak_mib: Peak GPU memory use [MiB].
        gpu_utilization_mean_pct: Mean GPU utilization [%].
        gpu_utilization_sample_count: Number of GPU utilization samples.
    """

    phase_timings_s: dict[str, float]
    collection_fps: float
    gpu_memory_mean_mib: float
    gpu_memory_peak_mib: float
    gpu_utilization_mean_pct: float
    gpu_utilization_sample_count: int


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating one immutable matrix attempt."""

    succeeded: bool
    failure_kind: FailureKind | None
    reason: str | None
    metrics: SemanticMetrics | None


class _MalformedArtifact(ValueError):
    """Internal signal for an invalid artifact shape or value type."""


class _MissingMetric(ValueError):
    """Internal signal for an absent required semantic measurement."""


def attempt_identity(attempt: BenchmarkAttempt) -> dict[str, Any]:
    """Return the complete identity manifest expected in attempt artifacts.

    Args:
        attempt: Immutable expanded matrix attempt.

    Returns:
        A JSON-compatible identity manifest.
    """
    return {
        "attempt_identity": attempt.identity,
        "run_set": attempt.run_set.value,
        "version": attempt.version.value,
        "logical_task": attempt.logical_task,
        "concrete_task": attempt.concrete_task,
        "mode": attempt.mode.id,
        "seed": attempt.seed,
        "repeat_index": attempt.repeat_index,
        "num_envs": attempt.num_envs,
        "framework": attempt.framework,
        "bound": {
            "unit": attempt.bound.unit.value,
            "value": attempt.bound.value,
        },
    }


def validate_attempt(
    attempt: BenchmarkAttempt,
    *,
    command: object,
    environment: object,
    exit_status: object,
    schema: object,
    measurements: object,
) -> ValidationResult:
    """Validate one attempt's identity, exit status, and semantic measurements.

    Args:
        attempt: Immutable matrix attempt that the artifacts must represent.
        command: Parsed ``command.json`` object.
        environment: Parsed ``environment.json`` object.
        exit_status: Parsed ``exit.json`` object.
        schema: Parsed canonical ``schema.json`` object, or ``None`` for an execution failure.
        measurements: Parsed generic ``measurements.json`` object, or ``None`` for an execution failure.

    Returns:
        A classified validation outcome with metrics only for success.
    """
    if not isinstance(command, Mapping):
        return _failure(FailureKind.MALFORMED_ARTIFACT, "command.json must contain an object")
    if not isinstance(environment, Mapping):
        return _failure(FailureKind.MALFORMED_ARTIFACT, "environment.json must contain an object")
    if not isinstance(exit_status, Mapping):
        return _failure(FailureKind.MALFORMED_ARTIFACT, "exit.json must contain an object")

    for filename, manifest in (("command.json", command), ("environment.json", environment)):
        identity_result = _validate_identity_manifest(attempt, filename, manifest)
        if identity_result is not None:
            return identity_result

    exit_result = _classify_exit(exit_status)
    if exit_result is not None:
        return exit_result

    if not isinstance(schema, Mapping):
        return _failure(FailureKind.MALFORMED_ARTIFACT, "schema.json must contain an object")
    if not isinstance(measurements, list):
        return _failure(FailureKind.MALFORMED_ARTIFACT, "measurements.json must contain a phase list")

    identity_result = _validate_schema_identity(attempt, schema)
    if identity_result is not None:
        return identity_result

    try:
        metrics = _extract_metrics(schema, measurements)
    except _MissingMetric as error:
        return _failure(FailureKind.MISSING_METRIC, str(error))
    except _MalformedArtifact as error:
        return _failure(FailureKind.MALFORMED_ARTIFACT, str(error))
    return ValidationResult(succeeded=True, failure_kind=None, reason=None, metrics=metrics)


def validate_attempt_directory(directory: Path, attempt: BenchmarkAttempt) -> ValidationResult:
    """Load and validate the semantic source files in an attempt directory.

    Args:
        directory: Staged or finalized attempt directory.
        attempt: Immutable matrix attempt that the artifacts must represent.

    Returns:
        A classified validation result. Missing or invalid JSON is malformed.
    """
    objects: dict[str, object] = {}
    for name in ("command.json", "environment.json", "exit.json", "schema.json", "measurements.json"):
        path = directory / name
        try:
            objects[name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            return _failure(FailureKind.MALFORMED_ARTIFACT, f"cannot read {name}: {error}")
    return validate_attempt(
        attempt,
        command=objects["command.json"],
        environment=objects["environment.json"],
        exit_status=objects["exit.json"],
        schema=objects["schema.json"],
        measurements=objects["measurements.json"],
    )


def validation_document(result: ValidationResult, attempt_number: int) -> dict[str, Any]:
    """Convert a validation result into its deterministic JSON document.

    Args:
        result: Classified attempt result.
        attempt_number: Monotonically increasing attempt number.

    Returns:
        A JSON-compatible validation document.
    """
    metrics = None
    if result.metrics is not None:
        metrics = {
            "phase_timings_s": result.metrics.phase_timings_s,
            "collection_fps": result.metrics.collection_fps,
            "gpu_memory_mean_mib": result.metrics.gpu_memory_mean_mib,
            "gpu_memory_peak_mib": result.metrics.gpu_memory_peak_mib,
            "gpu_utilization_mean_pct": result.metrics.gpu_utilization_mean_pct,
            "gpu_utilization_sample_count": result.metrics.gpu_utilization_sample_count,
        }
    return {
        "attempt_number": attempt_number,
        "status": "success" if result.succeeded else "failure",
        "failure_kind": result.failure_kind.value if result.failure_kind is not None else None,
        "reason": result.reason,
        "metrics": metrics,
    }


def _validate_identity_manifest(
    attempt: BenchmarkAttempt, filename: str, manifest: Mapping[object, object]
) -> ValidationResult | None:
    """Validate the explicit identity embedded in a command or environment artifact."""
    actual = manifest.get("identity")
    if not isinstance(actual, Mapping):
        return _failure(FailureKind.MALFORMED_ARTIFACT, f"{filename} identity must contain an object")
    expected = attempt_identity(attempt)
    for key, expected_value in expected.items():
        if key not in actual:
            return _failure(FailureKind.MALFORMED_ARTIFACT, f"{filename} identity is missing {key}")
        if actual[key] != expected_value:
            return _failure(
                FailureKind.IDENTITY_MISMATCH,
                f"{filename} identity {key} does not match the matrix attempt",
            )
    return None


def _classify_exit(exit_status: Mapping[object, object]) -> ValidationResult | None:
    """Validate exit metadata and classify an execution failure."""
    failure_stage = exit_status.get("failure_stage")
    timed_out = exit_status.get("timed_out")
    interrupted = exit_status.get("interrupted", False)
    out_of_memory = exit_status.get("out_of_memory")
    exit_code = exit_status.get("exit_code")
    if failure_stage not in (None, FailureKind.SETUP.value, FailureKind.LAUNCH.value):
        return _failure(FailureKind.MALFORMED_ARTIFACT, "exit.json failure_stage is invalid")
    if not isinstance(timed_out, bool) or not isinstance(interrupted, bool) or not isinstance(out_of_memory, bool):
        return _failure(FailureKind.MALFORMED_ARTIFACT, "exit.json failure flags must be boolean")
    if exit_code is not None and (not isinstance(exit_code, int) or isinstance(exit_code, bool)):
        return _failure(FailureKind.MALFORMED_ARTIFACT, "exit.json exit_code must be an integer or null")
    if failure_stage is not None:
        return _failure(FailureKind(failure_stage), f"benchmark {failure_stage} failed")
    if timed_out:
        return _failure(FailureKind.TIMEOUT, "benchmark timed out")
    if interrupted:
        return _failure(FailureKind.INTERRUPTED, "benchmark was interrupted")
    if out_of_memory:
        return _failure(FailureKind.OUT_OF_MEMORY, "benchmark ran out of memory")
    if exit_code is None:
        return _failure(FailureKind.MALFORMED_ARTIFACT, "exit.json has no exit code or classified failure")
    if exit_code != 0:
        return _failure(FailureKind.NONZERO_EXIT, f"benchmark exited with code {exit_code}")
    return None


def _validate_schema_identity(attempt: BenchmarkAttempt, schema: Mapping[object, object]) -> ValidationResult | None:
    """Match canonical schema identity and completed bound to the matrix attempt."""
    try:
        run = _mapping_at(schema, "run")
        runtime = _mapping_at(schema, "runtime")
        versions = _mapping_at(schema, "versions")
        hardware = _mapping_at(schema, "hardware")
        gpu_devices = _required_at(hardware, "gpu_devices")
        if not isinstance(gpu_devices, list):
            raise _MalformedArtifact("hardware.gpu_devices must be a list")
        release = _required_at(versions, "isaaclab_release")
        if not isinstance(release, str):
            raise _MalformedArtifact("versions.isaaclab_release must be a string")
        actual_identity = {
            "task": _required_at(run, "task"),
            "seed": _required_at(run, "seed"),
            "num_envs": _required_at(run, "num_envs"),
            "status": _required_at(run, "status"),
            "iterations_completed": _required_at(runtime, "iterations_completed"),
            "version": release.partition(".")[0],
        }
        if attempt.bound.unit is BoundUnit.ITERATIONS:
            actual_identity["max_iterations"] = _required_at(run, "max_iterations")
    except _MalformedArtifact as error:
        return _failure(FailureKind.MALFORMED_ARTIFACT, str(error))

    if len(gpu_devices) != 1:
        return _failure(FailureKind.IDENTITY_MISMATCH, "schema.json must identify exactly one visible GPU")

    expected_identity: dict[str, object] = {
        "task": attempt.concrete_task,
        "seed": attempt.seed,
        "num_envs": attempt.num_envs,
        "status": "completed",
        "iterations_completed": attempt.bound.value,
        "version": "2" if attempt.version is Version.LAB2 else "3",
    }
    if attempt.bound.unit is BoundUnit.ITERATIONS:
        expected_identity["max_iterations"] = attempt.bound.value
    for key, expected_value in expected_identity.items():
        if actual_identity[key] != expected_value:
            return _failure(
                FailureKind.IDENTITY_MISMATCH,
                f"schema.json {key} does not match the matrix attempt",
            )
    return None


def _extract_metrics(schema: Mapping[object, object], measurements: list[object]) -> SemanticMetrics:
    """Extract canonical schema metrics and the generic utilization sample count."""
    runtime = _mapping_at(schema, "runtime")
    resources = _mapping_at(schema, "resources")
    startup = _required_metric_mapping(runtime, "startup_time_s", "runtime.startup_time_s")
    phase_timings: dict[str, float] = {}
    for phase, value in startup.items():
        if value is None:
            continue
        if not isinstance(phase, str):
            raise _MalformedArtifact("runtime.startup_time_s phase names must be strings")
        phase_timings[phase] = _number(value, f"runtime.startup_time_s.{phase}")
    if not phase_timings:
        raise _MissingMetric("runtime.startup_time_s must contain at least one phase timing")

    collection = _required_metric_mapping(runtime, "collection_fps", "runtime.collection_fps")
    gpu_memory = _required_metric_mapping(resources, "gpu_mem_gb", "resources.gpu_mem_gb")
    gpu_utilization = _required_metric_mapping(resources, "gpu_util_pct", "resources.gpu_util_pct")
    collection_fps = _required_number(collection, "mean", "runtime.collection_fps.mean")
    gpu_memory_mean_gb = _required_number(gpu_memory, "mean", "resources.gpu_mem_gb.mean")
    gpu_memory_peak_gb = _required_number(gpu_memory, "peak", "resources.gpu_mem_gb.peak")
    gpu_utilization_mean_pct = _required_number(gpu_utilization, "mean", "resources.gpu_util_pct.mean")
    sample_count = _gpu_utilization_sample_count(measurements)
    gpu_memory_mean_mib = _gigabytes_to_mib(gpu_memory_mean_gb, "resources.gpu_mem_gb.mean")
    gpu_memory_peak_mib = _gigabytes_to_mib(gpu_memory_peak_gb, "resources.gpu_mem_gb.peak")
    return SemanticMetrics(
        phase_timings_s=phase_timings,
        collection_fps=collection_fps,
        gpu_memory_mean_mib=gpu_memory_mean_mib,
        gpu_memory_peak_mib=gpu_memory_peak_mib,
        gpu_utilization_mean_pct=gpu_utilization_mean_pct,
        gpu_utilization_sample_count=sample_count,
    )


def _gpu_utilization_sample_count(measurements: list[object]) -> int:
    """Extract the generic runtime phase's GPU utilization sample count."""
    matches: list[Mapping[object, object]] = []
    for phase in measurements:
        if not isinstance(phase, Mapping):
            raise _MalformedArtifact("measurements.json phases must be objects")
        phase_measurements = phase.get("measurements")
        if not isinstance(phase_measurements, list):
            raise _MalformedArtifact("measurements.json phase measurements must be lists")
        for measurement in phase_measurements:
            if not isinstance(measurement, Mapping):
                raise _MalformedArtifact("measurements.json measurements must be objects")
            name = measurement.get("name")
            if not isinstance(name, str):
                raise _MalformedArtifact("measurements.json measurement names must be strings")
            if re.search(r"(?:^| )GPU(?: 0)? Utilization n$", name) is not None:
                matches.append(measurement)
    if not matches:
        raise _MissingMetric("measurements.json is missing GPU Utilization n")
    if len(matches) != 1:
        raise _MalformedArtifact("measurements.json contains duplicate GPU Utilization n measurements")
    sample_measurement = matches[0]
    if "value" not in sample_measurement or sample_measurement["value"] is None:
        raise _MissingMetric("GPU utilization sample count is required")
    sample_count = sample_measurement["value"]
    if not isinstance(sample_count, int) or isinstance(sample_count, bool):
        raise _MalformedArtifact("GPU utilization sample count must be an integer")
    if sample_count <= 0:
        raise _MissingMetric("GPU utilization sample count must be positive")
    return sample_count


def _mapping_at(mapping: Mapping[object, object], key: str) -> Mapping[object, object]:
    """Return a required nested mapping."""
    value = _required_at(mapping, key)
    if not isinstance(value, Mapping):
        raise _MalformedArtifact(f"{key} must contain an object")
    return value


def _required_at(mapping: Mapping[object, object], key: str) -> object:
    """Return a required value from a mapping."""
    if key not in mapping:
        raise _MalformedArtifact(f"missing required field {key}")
    return mapping[key]


def _required_metric_mapping(mapping: Mapping[object, object], key: str, path: str) -> Mapping[object, object]:
    """Return a required metric mapping while distinguishing absence from bad shape."""
    if key not in mapping or mapping[key] is None:
        raise _MissingMetric(f"{path} is required")
    if not isinstance(mapping[key], Mapping):
        raise _MalformedArtifact(f"{path} must contain an object")
    return mapping[key]


def _required_number(mapping: Mapping[object, object], key: str, path: str) -> float:
    """Return a required finite numeric metric."""
    if key not in mapping or mapping[key] is None:
        raise _MissingMetric(f"{path} is required")
    return _number(mapping[key], path)


def _gigabytes_to_mib(value: float, path: str) -> float:
    """Convert a canonical GPU memory value to finite mebibytes [MiB]."""
    try:
        result = value * 1024.0
    except (OverflowError, ValueError, TypeError):
        raise _MalformedArtifact(f"{path} cannot be converted to MiB") from None
    if not math.isfinite(result):
        raise _MalformedArtifact(f"{path} converted to MiB must be finite")
    return result


def _number(value: object, path: str) -> float:
    """Return a finite JSON number without accepting booleans."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise _MalformedArtifact(f"{path} must be numeric")
    try:
        number = float(value)
    except (OverflowError, ValueError, TypeError):
        raise _MalformedArtifact(f"{path} must be a finite numeric value") from None
    if not math.isfinite(number):
        raise _MalformedArtifact(f"{path} must be finite")
    return number


def _failure(kind: FailureKind, reason: str) -> ValidationResult:
    """Build a failure result without inventing numeric metrics."""
    return ValidationResult(succeeded=False, failure_kind=kind, reason=reason, metrics=None)
