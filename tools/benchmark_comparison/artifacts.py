# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Atomically finalize immutable benchmark attempt artifacts."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .models import BenchmarkAttempt, ExecutionProvenance
from .validate import (
    FailureKind,
    validate_attempt_directory,
    validation_document,
)

REQUIRED_ARTIFACT_FILES = (
    "command.json",
    "environment.json",
    "stdout.log",
    "stderr.log",
    "exit.json",
    "schema.json",
    "measurements.json",
    "validation.json",
    "checksums.sha256",
)

_CHECKSUMMED_FILES = tuple(sorted(set(REQUIRED_ARTIFACT_FILES) - {"checksums.sha256"}))
_FAILED_ATTEMPT_PATTERN = re.compile(r"attempt-(?P<number>[0-9]+)-[a-z_]+$")


class SuccessfulArtifactExistsError(FileExistsError):
    """Raised when an immutable successful artifact already exists."""


class ArtifactIntegrityError(RuntimeError):
    """Raised when a pre-existing successful artifact is not trustworthy."""


def finalize_attempt(
    root: Path,
    attempt: BenchmarkAttempt,
    *,
    command: object,
    environment: object,
    stdout: str,
    stderr: str,
    exit_status: object,
    schema: object,
    measurements: object,
) -> Path:
    """Validate and atomically finalize a complete benchmark attempt.

    Args:
        root: Root directory for all comparison artifacts.
        attempt: Immutable expanded matrix attempt.
        command: JSON-compatible command document with explicit attempt identity.
        environment: JSON-compatible environment document with explicit attempt identity.
        stdout: Captured standard output.
        stderr: Captured standard error.
        exit_status: JSON-compatible exit document.
        schema: Canonical benchmark schema document, or ``None`` if unavailable.
        measurements: Generic benchmark measurement phases, or ``None`` if unavailable.

    Returns:
        Final success or numbered failure directory.

    Raises:
        SuccessfulArtifactExistsError: If this matrix attempt already succeeded.
        ArtifactIntegrityError: If a pre-existing success directory is corrupt.
    """
    if not isinstance(stdout, str) or not isinstance(stderr, str):
        raise TypeError("stdout and stderr must be strings")
    attempt_root = root / attempt.run_directory
    attempt_root.mkdir(parents=True, exist_ok=True)

    with _finalization_lock(attempt_root):
        success_path = attempt_root / "success"
        if success_path.exists():
            _verify_existing_success(success_path, attempt_root, attempt)
            raise SuccessfulArtifactExistsError(f"successful artifact already exists: {success_path}")

        attempt_number = _next_attempt_number(attempt_root)
        staging_path = Path(tempfile.mkdtemp(prefix=f".staging-{attempt_number:04d}-", dir=attempt_root))
        try:
            _write_json(staging_path / "command.json", command)
            _write_json(staging_path / "environment.json", environment)
            (staging_path / "stdout.log").write_text(stdout, encoding="utf-8")
            (staging_path / "stderr.log").write_text(stderr, encoding="utf-8")
            _write_json(staging_path / "exit.json", exit_status)
            _write_json(staging_path / "schema.json", schema)
            _write_json(staging_path / "measurements.json", measurements)

            result = validate_attempt_directory(staging_path, attempt)
            _write_json(
                staging_path / "validation.json",
                validation_document(result, attempt_number),
            )
            _write_checksum_manifest(staging_path)
            if not verify_checksums(staging_path):
                raise ArtifactIntegrityError(f"staged artifact failed verification: {staging_path}")

            if result.succeeded:
                final_path = success_path
            else:
                failure_kind = result.failure_kind
                if failure_kind is None:
                    raise ArtifactIntegrityError("failed validation has no failure classification")
                final_path = attempt_root / _failure_directory_name(attempt_number, failure_kind)
            if final_path.exists():
                raise ArtifactIntegrityError(f"artifact destination already exists: {final_path}")
            os.rename(staging_path, final_path)
            return final_path
        finally:
            if staging_path.exists():
                shutil.rmtree(staging_path)


def verify_checksums(directory: Path) -> bool:
    """Verify the deterministic checksum manifest and complete file layout.

    Args:
        directory: Finalized or staged artifact directory.

    Returns:
        ``True`` only when every required file exists and matches its hash.
    """
    try:
        if {path.name for path in directory.iterdir()} != set(REQUIRED_ARTIFACT_FILES):
            return False
        manifest_lines = (directory / "checksums.sha256").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError):
        return False
    if len(manifest_lines) != len(_CHECKSUMMED_FILES):
        return False

    actual_entries: dict[str, str] = {}
    for line in manifest_lines:
        if len(line) < 67 or line[64:66] != "  ":
            return False
        digest, filename = line[:64], line[66:]
        if filename in actual_entries or not re.fullmatch(r"[0-9a-f]{64}", digest):
            return False
        actual_entries[filename] = digest
    if tuple(sorted(actual_entries)) != _CHECKSUMMED_FILES:
        return False

    for filename in _CHECKSUMMED_FILES:
        try:
            contents = (directory / filename).read_bytes()
        except OSError:
            return False
        if hashlib.sha256(contents).hexdigest() != actual_entries[filename]:
            return False
    return True


def verify_success(
    directory: Path,
    attempt: BenchmarkAttempt,
    *,
    expected_provenance: ExecutionProvenance,
    expected_gpu_uuid: str,
) -> bool:
    """Verify a finalized success's checksums, semantics, and validation document.

    Args:
        directory: Finalized ``success`` artifact directory.
        attempt: Immutable matrix attempt expected in the artifact.
        expected_provenance: Exact preflight identities required in the artifact.
        expected_gpu_uuid: Physical GPU UUID selected by preflight.

    Returns:
        ``True`` only when the complete success artifact is trustworthy.
    """
    try:
        _verify_existing_success(directory, directory.parent, attempt)
        _verify_execution_provenance(
            directory,
            attempt,
            expected_provenance=expected_provenance,
            expected_gpu_uuid=expected_gpu_uuid,
        )
    except ArtifactIntegrityError:
        return False
    return True


def _verify_execution_provenance(
    directory: Path,
    attempt: BenchmarkAttempt,
    *,
    expected_provenance: ExecutionProvenance,
    expected_gpu_uuid: str,
) -> None:
    """Reject a success produced by different preflight execution identities."""
    try:
        environment = json.loads((directory / "environment.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError(f"cannot read success provenance: {error}") from error
    if not isinstance(environment, dict) or not isinstance(environment.get("values"), dict):
        raise ArtifactIntegrityError(f"success provenance must contain environment values: {directory}")
    expected = expected_provenance.to_json()
    for field, value in expected.items():
        if environment.get(field) != value:
            raise ArtifactIntegrityError(f"success provenance {field} does not match: {directory}")
    values = environment["values"]
    if values.get("ISAACLAB_BENCHMARK_LAB2_SHA") != expected_provenance.lab2_sha:
        raise ArtifactIntegrityError(f"success environment Lab 2 SHA does not match: {directory}")
    if values.get("ISAACLAB_BENCHMARK_LAB3_SHA") != expected_provenance.lab3_sha:
        raise ArtifactIntegrityError(f"success environment Lab 3 SHA does not match: {directory}")
    if values.get("ISAACLAB_BENCHMARK_LAB2_IMAGE_ID") != expected_provenance.lab2_image_id:
        raise ArtifactIntegrityError(f"success environment Lab 2 image ID does not match: {directory}")
    if values.get("ISAACLAB_BENCHMARK_UV_LOCK_SHA256") != expected_provenance.uv_lock_sha256:
        raise ArtifactIntegrityError(f"success environment uv lock does not match: {directory}")
    if environment.get("environment_identity") != expected_provenance.environment_identity(attempt.version):
        raise ArtifactIntegrityError(f"success executor identity does not match: {directory}")
    expected_gpu = {"physical_index": 0, "uuid": expected_gpu_uuid}
    if environment.get("selected_gpu") != expected_gpu:
        raise ArtifactIntegrityError(f"success selected GPU does not match: {directory}")
    expected_gpu_environment = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "0",
        "NVIDIA_VISIBLE_DEVICES": "0",
        "ISAACLAB_BENCHMARK_GPU_INDEX": "0",
        "ISAACLAB_BENCHMARK_GPU_UUID": expected_gpu_uuid,
    }
    for name, value in expected_gpu_environment.items():
        if values.get(name) != value:
            raise ArtifactIntegrityError(f"success environment {name} does not match: {directory}")


def _verify_existing_success(success_path: Path, attempt_root: Path, attempt: BenchmarkAttempt) -> None:
    """Revalidate a checksummed success against its matrix attempt and validation document."""
    if not verify_checksums(success_path):
        raise ArtifactIntegrityError(f"existing success artifact failed verification: {success_path}")

    result = validate_attempt_directory(success_path, attempt)
    if not result.succeeded:
        kind = result.failure_kind.value if result.failure_kind is not None else "unclassified"
        raise ArtifactIntegrityError(f"existing success failed semantic validation ({kind}): {success_path}")

    try:
        validation = json.loads((success_path / "validation.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError(f"cannot read existing success validation: {error}") from error
    if not isinstance(validation, dict):
        raise ArtifactIntegrityError(f"existing success validation must be an object: {success_path}")
    attempt_number = validation.get("attempt_number")
    if (
        not isinstance(attempt_number, int)
        or isinstance(attempt_number, bool)
        or attempt_number != _next_attempt_number(attempt_root)
    ):
        raise ArtifactIntegrityError(f"existing success attempt number is inconsistent: {success_path}")
    if validation != validation_document(result, attempt_number):
        raise ArtifactIntegrityError(f"existing success validation is inconsistent: {success_path}")


def _write_json(path: Path, value: object) -> None:
    """Write deterministic UTF-8 JSON."""
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_checksum_manifest(directory: Path) -> None:
    """Hash every artifact except the checksum manifest itself."""
    lines = [
        f"{hashlib.sha256((directory / filename).read_bytes()).hexdigest()}  {filename}"
        for filename in _CHECKSUMMED_FILES
    ]
    (directory / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="ascii")


def _next_attempt_number(attempt_root: Path) -> int:
    """Return the next monotonic attempt number under one matrix identity."""
    numbers: list[int] = []
    for path in attempt_root.iterdir():
        match = _FAILED_ATTEMPT_PATTERN.fullmatch(path.name)
        if match is not None:
            number = int(match.group("number"))
            if number > 0:
                numbers.append(number)
    return max(numbers, default=0) + 1


def _failure_directory_name(attempt_number: int, failure_kind: FailureKind) -> str:
    """Build a stable numbered failure directory name."""
    return f"attempt-{attempt_number:04d}-{failure_kind.value}"


@contextmanager
def _finalization_lock(attempt_root: Path) -> Iterator[None]:
    """Serialize finalization so a valid success can never be replaced."""
    with (attempt_root / ".finalize.lock").open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
