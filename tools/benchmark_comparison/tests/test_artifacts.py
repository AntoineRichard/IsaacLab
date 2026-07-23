# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for immutable benchmark attempt artifact storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.benchmark_comparison.artifacts import (
    REQUIRED_ARTIFACT_FILES,
    SuccessfulArtifactExistsError,
    finalize_attempt,
    verify_checksums,
)
from tools.benchmark_comparison.matrix import expand_canary_matrix, load_matrix
from tools.benchmark_comparison.models import BenchmarkAttempt
from tools.benchmark_comparison.validate import attempt_identity

_FIXTURES = Path(__file__).with_name("fixtures")


def _attempt() -> BenchmarkAttempt:
    return next(
        attempt
        for attempt in expand_canary_matrix(load_matrix()).attempts
        if attempt.mode.id == "runtime-100" and attempt.version.value == "lab2"
    )


def _load(name: str) -> Any:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


def _payloads(attempt: BenchmarkAttempt, exit_code: int | None = 0) -> dict[str, Any]:
    identity = attempt_identity(attempt)
    return {
        "command": {"argv": ["benchmark"], "identity": identity},
        "environment": {"hostname": "fixture-host", "identity": identity},
        "stdout": "benchmark output\n",
        "stderr": "",
        "exit_status": {
            "exit_code": exit_code,
            "failure_stage": None,
            "timed_out": False,
            "out_of_memory": False,
        },
        "schema": _load("schema_runtime.json") if exit_code == 0 else None,
        "measurements": _load("generic_runtime.json") if exit_code == 0 else None,
    }


def test_finalize_attempt_atomically_creates_complete_success_directory(tmp_path: Path) -> None:
    """A validated attempt becomes one complete success directory with no staging residue."""
    attempt = _attempt()

    final_path = finalize_attempt(tmp_path, attempt, **_payloads(attempt))

    assert final_path == tmp_path / attempt.run_directory / "success"
    assert {path.name for path in final_path.iterdir()} == set(REQUIRED_ARTIFACT_FILES)
    assert verify_checksums(final_path)
    validation = json.loads((final_path / "validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "success"
    assert validation["attempt_number"] == 1
    assert not any(path.name.startswith(".staging-") for path in final_path.parent.iterdir())


def test_verify_checksums_detects_corrupt_and_missing_artifacts(tmp_path: Path) -> None:
    """Checksum verification fails after content corruption or required-file removal."""
    attempt = _attempt()
    corrupt_path = finalize_attempt(tmp_path / "corrupt", attempt, **_payloads(attempt))
    (corrupt_path / "stdout.log").write_text("tampered\n", encoding="utf-8")

    missing_path = finalize_attempt(tmp_path / "missing", attempt, **_payloads(attempt))
    (missing_path / "schema.json").unlink()

    assert not verify_checksums(corrupt_path)
    assert not verify_checksums(missing_path)


def test_finalize_attempt_never_overwrites_valid_success(tmp_path: Path) -> None:
    """A duplicate successful write leaves the first immutable artifact untouched."""
    attempt = _attempt()
    original_path = finalize_attempt(tmp_path, attempt, **_payloads(attempt))
    original_manifest = (original_path / "checksums.sha256").read_bytes()

    with pytest.raises(SuccessfulArtifactExistsError):
        finalize_attempt(tmp_path, attempt, **_payloads(attempt))

    assert (original_path / "checksums.sha256").read_bytes() == original_manifest
    assert verify_checksums(original_path)


def test_finalize_attempt_preserves_failures_with_monotonic_retry_numbers(tmp_path: Path) -> None:
    """Failed attempts remain numbered while a later valid retry finalizes as success."""
    attempt = _attempt()
    first = finalize_attempt(tmp_path, attempt, **_payloads(attempt, exit_code=3))
    timed_out = _payloads(attempt, exit_code=None)
    timed_out["exit_status"]["timed_out"] = True
    second = finalize_attempt(tmp_path, attempt, **timed_out)
    success = finalize_attempt(tmp_path, attempt, **_payloads(attempt))

    assert first.name == "attempt-0001-nonzero_exit"
    assert second.name == "attempt-0002-timeout"
    assert success.name == "success"
    assert json.loads((success / "validation.json").read_text(encoding="utf-8"))["attempt_number"] == 3
    assert all(path.exists() and verify_checksums(path) for path in (first, second, success))


def test_finalize_attempt_preserves_malformed_artifact_without_zero_metrics(tmp_path: Path) -> None:
    """Malformed successful output remains diagnostic evidence, never a zero-valued success."""
    attempt = _attempt()
    payloads = _payloads(attempt)
    payloads["schema"]["resources"]["gpu_mem_gb"]["peak"] = None

    failed_path = finalize_attempt(tmp_path, attempt, **payloads)

    validation = json.loads((failed_path / "validation.json").read_text(encoding="utf-8"))
    assert failed_path.name == "attempt-0001-missing_metric"
    assert validation["status"] == "failure"
    assert validation["metrics"] is None
    assert verify_checksums(failed_path)


def test_checksums_are_deterministic_for_identical_attempt_bytes(tmp_path: Path) -> None:
    """Equal source artifacts and validation results produce byte-identical manifests."""
    attempt = _attempt()
    first = finalize_attempt(tmp_path / "one", attempt, **_payloads(attempt))
    second = finalize_attempt(tmp_path / "two", attempt, **_payloads(attempt))

    assert (first / "checksums.sha256").read_bytes() == (second / "checksums.sha256").read_bytes()
