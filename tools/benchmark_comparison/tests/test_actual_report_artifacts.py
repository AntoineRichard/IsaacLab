# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read-only smoke checks for the retained measured benchmark reports."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from tools.benchmark_comparison.manifest import read_manifest, resolve_manifest_expansion
from tools.benchmark_comparison.models import MatrixExpansion

_ROOT = Path("/home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd")
_ATTEMPT_IDENTITY_FIELDS = (
    "version",
    "logical_task",
    "concrete_task",
    "mode",
    "bound",
    "bound_unit",
    "seed",
    "num_envs",
)
_EXPECTED_GENERATED_FILES = frozenset(
    {
        "raw_runs.csv",
        "paired_summary.csv",
        "failures.csv",
        "report.md",
        "report.pdf",
        "collection_fps.png",
        "collection_fps.svg",
        "gpu_memory_mean_mib.png",
        "gpu_memory_mean_mib.svg",
        "gpu_memory_peak_mib.png",
        "gpu_memory_peak_mib.svg",
        "gpu_utilization_mean_pct.png",
        "gpu_utilization_mean_pct.svg",
        "startup_total_s.png",
        "startup_total_s.svg",
        "startup_phase_breakdown.png",
        "startup_phase_breakdown.svg",
    }
)


@pytest.mark.benchmark
def test_exact_attempt_coverage_rejects_duplicate_row_substitution() -> None:
    if not _ROOT.is_dir():
        pytest.skip("measured comparison artifact root is not present")
    manifest = read_manifest(_ROOT / "final" / "manifest.json")
    expansion = resolve_manifest_expansion(manifest, _ROOT)
    with (_ROOT / "final" / "report" / "raw_runs.csv").open(newline="", encoding="utf-8") as file:
        rows = tuple(csv.DictReader(file))
    duplicate_substitution = (*rows[:-1], rows[0])

    with pytest.raises(AssertionError):
        _assert_exact_attempt_coverage(duplicate_substitution, expansion)


def test_hash_manifest_verifier_returns_validated_entries(tmp_path: Path) -> None:
    payload = tmp_path / "payload"
    payload.write_bytes(b"payload")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    manifest = tmp_path / "hashes.sha256"
    manifest.write_text(f"{digest}  payload\n", encoding="utf-8")

    assert _verify_hash_manifest(manifest, tmp_path) == {"payload": digest}


def test_hash_manifest_verifier_rejects_empty_inventory(tmp_path: Path) -> None:
    manifest = tmp_path / "hashes.sha256"
    manifest.write_text("", encoding="utf-8")

    with pytest.raises(AssertionError, match="empty"):
        _verify_hash_manifest(manifest, tmp_path)


def test_hash_manifest_verifier_rejects_duplicate_entries(tmp_path: Path) -> None:
    payload = tmp_path / "payload"
    payload.write_bytes(b"payload")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    line = f"{digest}  payload\n"
    manifest = tmp_path / "hashes.sha256"
    manifest.write_text(line + line, encoding="utf-8")

    with pytest.raises(AssertionError, match="duplicate"):
        _verify_hash_manifest(manifest, tmp_path)


def test_hash_manifest_verifier_rejects_truncated_digest(tmp_path: Path) -> None:
    payload = tmp_path / "payload"
    payload.write_bytes(b"payload")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    manifest = tmp_path / "hashes.sha256"
    manifest.write_text(f"{digest[:-1]}  payload\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="digest"):
        _verify_hash_manifest(manifest, tmp_path)


@pytest.mark.benchmark
def test_retained_final_report_manifest_and_hashes_are_self_consistent() -> None:
    if not _ROOT.is_dir():
        pytest.skip("measured comparison artifact root is not present")
    manifest = read_manifest(_ROOT / "final" / "manifest.json")
    expansion = resolve_manifest_expansion(manifest, _ROOT)
    report = _ROOT / "final" / "report"

    assert manifest.run_set.value == "final"
    assert manifest.phase == "measured"
    assert len(expansion.attempts) == 228
    with (report / "raw_runs.csv").open(newline="", encoding="utf-8") as file:
        rows = csv.DictReader(file)
        assert {
            "startup_total_s",
            "startup_app_launch_s",
            "startup_python_imports_s",
            "startup_task_config_s",
            "startup_env_creation_s",
            "startup_first_step_s",
        } <= set(rows.fieldnames or ())
        normalized_rows = tuple(rows)
    _assert_exact_attempt_coverage(normalized_rows, expansion)
    with (report / "failures.csv").open(newline="", encoding="utf-8") as file:
        assert tuple(csv.DictReader(file)) == ()
    audit = json.loads((report / "audit_summary.json").read_text(encoding="utf-8"))
    assert audit["successful_attempts"] == len(expansion.attempts)
    assert audit["failed_or_missing_attempts"] == 0
    assert audit["generated_file_count"] == 17
    assert (report / "report.pdf").is_file()
    generated_manifest = report / "generated_hashes.sha256"
    generated_entries = _verify_hash_manifest(generated_manifest, report)
    assert len(generated_entries) == audit["generated_file_count"]
    assert set(generated_entries) == _EXPECTED_GENERATED_FILES
    raw_manifest = report / "raw_artifact_hashes.sha256"
    raw_entries = _verify_hash_manifest(raw_manifest, _ROOT)
    assert len(raw_entries) == audit["raw_file_count"]
    assert _sha256(raw_manifest) == audit["raw_hash_manifest_sha256"]
    assert _sha256(generated_manifest) == audit["generated_hash_manifest_sha256"]


def _verify_hash_manifest(path: Path, root: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines, f"{path.name} inventory is empty"
    entries: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        parts = line.split("  ", maxsplit=1)
        assert len(parts) == 2, f"{path.name} line {line_number} is malformed"
        expected, relative = parts
        assert len(expected) == 64 and all(character in "0123456789abcdef" for character in expected), (
            f"{path.name} line {line_number} has an invalid SHA-256 digest"
        )
        assert relative, f"{path.name} line {line_number} has an empty path"
        assert relative not in entries, f"{path.name} contains duplicate entry: {relative}"
        assert _sha256(root / relative) == expected, f"{path.name} digest mismatch: {relative}"
        entries[relative] = expected
    return entries


def _assert_exact_attempt_coverage(
    rows: Sequence[Mapping[str, str]],
    expansion: MatrixExpansion,
) -> None:
    expected = {
        (
            attempt.version.value,
            attempt.logical_task,
            attempt.concrete_task,
            attempt.mode.id,
            str(attempt.bound.value),
            attempt.bound.unit.value,
            str(attempt.seed),
            str(attempt.num_envs),
        )
        for attempt in expansion.attempts
    }
    observed = tuple(tuple(row[field] for field in _ATTEMPT_IDENTITY_FIELDS) for row in rows)

    assert len(expected) == len(expansion.attempts), "manifest attempt identities are not unique"
    assert len(observed) == len(expansion.attempts)
    assert len(set(observed)) == len(observed), "raw_runs.csv contains duplicate attempt identities"
    assert set(observed) == expected, "raw_runs.csv attempt identities do not exactly match the manifest expansion"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
