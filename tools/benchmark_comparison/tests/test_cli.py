# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for safe comparison-controller preparation and result import."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.benchmark_comparison import cli
from tools.benchmark_comparison.import_results import ImportAudit
from tools.benchmark_comparison.models import RunSet

LAB2_SHA = "a" * 40
LAB3_SHA = "b" * 40
LAB2_IMAGE_ID = "sha256:" + "c" * 64


def _arguments(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--run_set",
        "canary",
        "--phase",
        "measured",
        "--lab2_root",
        str(tmp_path / "lab2"),
        "--lab3_root",
        str(tmp_path / "lab3"),
        "--artifact_root",
        str(tmp_path / "destination"),
        "--lab2_sha",
        LAB2_SHA,
        "--lab3_sha",
        LAB3_SHA,
        "--lab2_image",
        "fixture",
        "--lab2_image_id",
        LAB2_IMAGE_ID,
        *extra,
    ]


def _forbid_measured_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args, **_kwargs):
        pytest.fail("measured execution objects must not be constructed")

    for name in (
        "OwnedProcessGroups",
        "ProcessLauncher",
        "HostIdleGate",
        "Lab2DockerExecutor",
        "Lab3UvExecutor",
        "BenchmarkRunner",
    ):
        monkeypatch.setattr(cli, name, forbidden)


def _patch_preparation(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    *,
    importer_error: Exception | None = None,
    expected_source_root: Path | None = None,
) -> None:
    manifest = object()

    class Preflight:
        def manifest(self, run_set, phase, expansion):
            assert run_set is RunSet.CANARY
            assert phase == "measured"
            assert len(expansion.attempts) == 136
            return manifest

    def run_preflight(_config):
        events.append("preflight")
        return Preflight()

    def write_manifest(path, value):
        assert events == ["preflight"]
        assert path.name == "manifest.json"
        assert path.parent.name == "canary"
        assert value is manifest
        events.append("manifest_written")
        return path

    def import_completed_attempts(source_root, destination_root, run_set):
        assert events == ["preflight", "manifest_written"]
        assert source_root.is_absolute()
        assert destination_root.is_absolute()
        assert run_set is RunSet.CANARY
        if expected_source_root is not None:
            assert source_root == expected_source_root.resolve()
        if importer_error is not None:
            raise importer_error
        events.append("import_completed")
        return ImportAudit(
            source_root=source_root,
            destination_root=destination_root,
            run_set=run_set,
            source_manifest_sha256="d" * 64,
            destination_manifest_sha256="e" * 64,
            imported_attempt_count=2,
            imported_file_count=18,
            source_aggregate_sha256="f" * 64,
            destination_aggregate_sha256="f" * 64,
        )

    monkeypatch.setattr(cli, "run_preflight", run_preflight)
    monkeypatch.setattr(cli, "write_manifest", write_manifest)
    monkeypatch.setattr(cli, "import_completed_attempts", import_completed_attempts, raising=False)


def test_prepare_only_writes_manifest_and_imports_before_executor_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.chdir(tmp_path)
    source_root = Path("source")
    _patch_preparation(monkeypatch, events, expected_source_root=tmp_path / source_root)
    _forbid_measured_construction(monkeypatch)

    result = cli.main(
        _arguments(
            tmp_path,
            "--import_from_artifact_root",
            str(source_root),
            "--prepare_only",
        )
    )

    assert result == 0
    assert events == ["preflight", "manifest_written", "import_completed"]


def test_prepare_only_requires_import_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_measured_construction(monkeypatch)

    with pytest.raises(SystemExit) as error:
        cli.main(_arguments(tmp_path, "--prepare_only"))

    assert error.value.code != 0


@pytest.mark.parametrize("relationship", ["equal", "nested"])
def test_prepare_only_rejects_import_source_overlapping_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relationship: str,
) -> None:
    events: list[str] = []
    destination_root = tmp_path / "destination"
    source_root = destination_root if relationship == "equal" else destination_root / "nested"
    _patch_preparation(
        monkeypatch,
        events,
        importer_error=ValueError("source and destination roots overlap"),
        expected_source_root=source_root,
    )
    _forbid_measured_construction(monkeypatch)

    with pytest.raises(ValueError, match="overlap"):
        cli.main(
            _arguments(
                tmp_path,
                "--import_from_artifact_root",
                str(source_root),
                "--prepare_only",
            )
        )

    assert events == ["preflight", "manifest_written"]


def test_prepare_only_propagates_importer_rejection_without_executor_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _patch_preparation(monkeypatch, events, importer_error=RuntimeError("import rejected"))
    _forbid_measured_construction(monkeypatch)

    with pytest.raises(RuntimeError, match="import rejected"):
        cli.main(
            _arguments(
                tmp_path,
                "--import_from_artifact_root",
                str(tmp_path / "source"),
                "--prepare_only",
            )
        )

    assert events == ["preflight", "manifest_written"]
