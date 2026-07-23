# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for semantic validation of benchmark attempt artifacts."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from tools.benchmark_comparison.matrix import expand_canary_matrix, load_matrix
from tools.benchmark_comparison.models import BenchmarkAttempt
from tools.benchmark_comparison.validate import FailureKind, attempt_identity, validate_attempt

_FIXTURES = Path(__file__).with_name("fixtures")


def _attempt(mode_id: str) -> BenchmarkAttempt:
    return next(
        attempt
        for attempt in expand_canary_matrix(load_matrix()).attempts
        if attempt.mode.id == mode_id and attempt.version.value == "lab2"
    )


def _load(name: str) -> Any:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


def _valid_payloads(mode_id: str = "runtime-100") -> tuple[BenchmarkAttempt, dict[str, Any]]:
    attempt = _attempt(mode_id)
    suffix = "training" if mode_id == "training-100" else "runtime"
    identity = attempt_identity(attempt)
    return attempt, {
        "command": {"argv": ["benchmark"], "identity": identity},
        "environment": {"hostname": "fixture-host", "identity": identity},
        "exit_status": {
            "exit_code": 0,
            "failure_stage": None,
            "timed_out": False,
            "out_of_memory": False,
        },
        "schema": _load(f"schema_{suffix}.json"),
        "measurements": _load(f"generic_{suffix}.json"),
    }


def _gpu_sample_measurement(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the generic GPU utilization sample-count measurement."""
    return next(
        measurement
        for phase in measurements
        for measurement in phase["measurements"]
        if measurement["name"].endswith("GPU Utilization n")
    )


@pytest.mark.parametrize("mode_id", ["runtime-100", "training-100"])
def test_validate_attempt_extracts_canonical_metrics_for_each_bound_unit(mode_id: str) -> None:
    """Runtime steps and training iterations validate against real schema-shaped fixtures."""
    attempt, payloads = _valid_payloads(mode_id)

    result = validate_attempt(attempt, **payloads)

    assert result.succeeded
    assert result.failure_kind is None
    assert result.metrics is not None
    assert result.metrics.phase_timings_s
    assert result.metrics.collection_fps > 0
    assert result.metrics.gpu_memory_mean_mib == pytest.approx(
        payloads["schema"]["resources"]["gpu_mem_gb"]["mean"] * 1024
    )
    assert result.metrics.gpu_memory_peak_mib == pytest.approx(
        payloads["schema"]["resources"]["gpu_mem_gb"]["peak"] * 1024
    )
    assert result.metrics.gpu_utilization_mean_pct == payloads["schema"]["resources"]["gpu_util_pct"]["mean"]
    assert result.metrics.gpu_utilization_sample_count == 2


def test_validate_attempt_accepts_lab3_schema_identity_and_exact_task_alias() -> None:
    """Lab 3 validation binds major version three and its explicit Task 6 task identifier."""
    attempt = next(
        attempt
        for attempt in expand_canary_matrix(load_matrix()).attempts
        if attempt.mode.id == "runtime-100" and attempt.version.value == "lab3"
    )
    _, payloads = _valid_payloads()
    identity = attempt_identity(attempt)
    payloads["command"]["identity"] = identity
    payloads["environment"]["identity"] = identity
    payloads["schema"]["run"]["task"] = "Isaac-Cartpole"
    payloads["schema"]["versions"]["isaaclab_release"] = "3.0.0"

    result = validate_attempt(attempt, **payloads)

    assert attempt.logical_task == "cartpole"
    assert attempt.concrete_task == "Isaac-Cartpole"
    assert result.succeeded


@pytest.mark.parametrize(
    ("exit_update", "expected"),
    [
        ({"failure_stage": "setup", "exit_code": None}, FailureKind.SETUP),
        ({"failure_stage": "launch", "exit_code": None}, FailureKind.LAUNCH),
        ({"timed_out": True, "exit_code": None}, FailureKind.TIMEOUT),
        ({"out_of_memory": True, "exit_code": 137}, FailureKind.OUT_OF_MEMORY),
        ({"exit_code": 3}, FailureKind.NONZERO_EXIT),
    ],
)
def test_validate_attempt_classifies_execution_failures_distinctly(
    exit_update: dict[str, Any], expected: FailureKind
) -> None:
    """Execution failures retain their specific category without synthetic metrics."""
    attempt, payloads = _valid_payloads()
    payloads["exit_status"].update(exit_update)
    payloads["schema"] = None
    payloads["measurements"] = None

    result = validate_attempt(attempt, **payloads)

    assert not result.succeeded
    assert result.failure_kind is expected
    assert result.metrics is None


@pytest.mark.parametrize(
    ("payload_name", "replacement"),
    [
        ("command", ["not", "an", "object"]),
        ("environment", None),
        ("exit_status", {"exit_code": "zero"}),
        ("schema", []),
        ("measurements", {}),
    ],
)
def test_validate_attempt_classifies_malformed_artifacts(payload_name: str, replacement: object) -> None:
    """Wrong top-level artifact shapes are malformed rather than missing measurements."""
    attempt, payloads = _valid_payloads()
    payloads[payload_name] = replacement

    result = validate_attempt(attempt, **payloads)

    assert result.failure_kind is FailureKind.MALFORMED_ARTIFACT
    assert result.metrics is None


@pytest.mark.parametrize("manifest_name", ["command", "environment"])
def test_validate_attempt_rejects_mismatched_attempt_identity(manifest_name: str) -> None:
    """Command and environment manifests must bind the exact immutable matrix attempt."""
    attempt, payloads = _valid_payloads()
    payloads[manifest_name] = deepcopy(payloads[manifest_name])
    payloads[manifest_name]["identity"]["seed"] = 43

    result = validate_attempt(attempt, **payloads)

    assert result.failure_kind is FailureKind.IDENTITY_MISMATCH
    assert "seed" in result.reason
    assert result.metrics is None


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("run", "task"), "Isaac-Ant-v0"),
        (("run", "seed"), 43),
        (("run", "num_envs"), 16),
        (("versions", "isaaclab_release"), "3.0.0"),
        (("runtime", "iterations_completed"), 9),
    ],
)
def test_validate_attempt_rejects_schema_identity_mismatches(path: tuple[str, str], replacement: object) -> None:
    """Canonical schema identity and completed bound must match the matrix attempt."""
    attempt, payloads = _valid_payloads()
    payloads["schema"][path[0]][path[1]] = replacement

    result = validate_attempt(attempt, **payloads)

    assert result.failure_kind is FailureKind.IDENTITY_MISMATCH
    assert result.metrics is None


def test_validate_attempt_rejects_training_iteration_bound_mismatch() -> None:
    """Training schema max_iterations must match the requested iteration bound."""
    attempt, payloads = _valid_payloads("training-100")
    payloads["schema"]["run"]["max_iterations"] = 3

    result = validate_attempt(attempt, **payloads)

    assert result.failure_kind is FailureKind.IDENTITY_MISMATCH


@pytest.mark.parametrize(
    "mutation",
    [
        "empty_phase_timings",
        "collection_fps",
        "gpu_memory_object",
        "gpu_memory_mean",
        "gpu_memory_peak",
        "gpu_utilization_mean",
        "gpu_utilization_samples",
    ],
)
def test_validate_attempt_rejects_each_missing_required_metric(mutation: str) -> None:
    """Every required semantic metric remains a failure when absent."""
    attempt, payloads = _valid_payloads()
    if mutation == "empty_phase_timings":
        payloads["schema"]["runtime"]["startup_time_s"] = {}
    elif mutation == "collection_fps":
        payloads["schema"]["runtime"]["collection_fps"]["mean"] = None
    elif mutation == "gpu_memory_object":
        del payloads["schema"]["resources"]["gpu_mem_gb"]
    elif mutation == "gpu_memory_mean":
        payloads["schema"]["resources"]["gpu_mem_gb"]["mean"] = None
    elif mutation == "gpu_memory_peak":
        payloads["schema"]["resources"]["gpu_mem_gb"]["peak"] = None
    elif mutation == "gpu_utilization_mean":
        payloads["schema"]["resources"]["gpu_util_pct"]["mean"] = None
    else:
        _gpu_sample_measurement(payloads["measurements"])["value"] = 0

    result = validate_attempt(attempt, **payloads)

    assert result.failure_kind is FailureKind.MISSING_METRIC
    assert result.metrics is None
    assert "0" not in result.reason if mutation != "gpu_utilization_samples" else "positive" in result.reason


@pytest.mark.parametrize(
    ("sample_state", "expected"),
    [
        ("absent", FailureKind.MISSING_METRIC),
        ("null", FailureKind.MISSING_METRIC),
        ("wrong_type", FailureKind.MALFORMED_ARTIFACT),
        ("zero", FailureKind.MISSING_METRIC),
        ("positive", None),
    ],
)
def test_validate_attempt_classifies_gpu_sample_count_values(sample_state: str, expected: FailureKind | None) -> None:
    """Missing and zero sample counts differ from malformed non-null values."""
    attempt, payloads = _valid_payloads()
    sample = _gpu_sample_measurement(payloads["measurements"])
    if sample_state == "absent":
        del sample["value"]
    elif sample_state == "null":
        sample["value"] = None
    elif sample_state == "wrong_type":
        sample["value"] = "two"
    elif sample_state == "zero":
        sample["value"] = 0

    result = validate_attempt(attempt, **payloads)

    assert result.failure_kind is expected
    assert result.succeeded is (expected is None)


def test_validate_attempt_uses_schema_values_instead_of_generic_metric_duplicates() -> None:
    """Generic values cannot silently replace the canonical schema metric values."""
    attempt, payloads = _valid_payloads()
    for phase in payloads["measurements"]:
        for measurement in phase["measurements"]:
            if measurement["name"].endswith("Mean Collection FPS"):
                measurement["value"] = 1.0

    result = validate_attempt(attempt, **payloads)

    assert result.metrics is not None
    assert result.metrics.collection_fps == payloads["schema"]["runtime"]["collection_fps"]["mean"]
