# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for safe comparison-controller preparation and result import."""

from __future__ import annotations

import stat
from pathlib import Path
from typing import Any

import pytest

from tools.benchmark_comparison import cli
from tools.benchmark_comparison.import_results import ImportAudit
from tools.benchmark_comparison.models import RunSet
from tools.benchmark_comparison.runner import ControllerLock

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


def _inventory(root: Path) -> dict[str, tuple[int, int, int, int, bytes | None]]:
    """Capture bytes and metadata without following directory symlinks."""
    inventory = {}
    for path in (root, *sorted(root.rglob("*"))):
        metadata = path.lstat()
        contents = path.read_bytes() if stat.S_ISREG(metadata.st_mode) else None
        relative = path.relative_to(root).as_posix() or "."
        inventory[relative] = (
            metadata.st_mode,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ino,
            contents,
        )
    return inventory


def _filesystem_arguments(tmp_path: Path, source_root: Path, destination_root: Path) -> list[str]:
    """Return prepare-only arguments targeting real source and destination paths."""
    arguments = _arguments(
        tmp_path,
        "--import_from_artifact_root",
        str(source_root),
        "--prepare_only",
    )
    artifact_index = arguments.index("--artifact_root") + 1
    arguments[artifact_index] = str(destination_root)
    return arguments


def _patch_filesystem_preparation(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> None:
    """Replace expensive preflight work while retaining real destination writes."""

    class Preflight:
        def manifest(self, _run_set, _phase, _expansion):
            return object()

    def run_preflight(config, *, artifact_root_for_writes=None):
        assert config.artifact_root.parent != Path("/proc/self/fd")
        assert artifact_root_for_writes is not None
        assert artifact_root_for_writes.parent == Path("/proc/self/fd")
        events.append("preflight")
        return Preflight()

    def write_manifest(path: Path, _manifest: object) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"controller destination manifest\n")
        events.append("manifest_written")
        return path

    monkeypatch.setattr(cli, "run_preflight", run_preflight)
    monkeypatch.setattr(cli, "write_manifest", write_manifest)


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

    def run_preflight(_config, *, artifact_root_for_writes=None):
        assert artifact_root_for_writes is not None
        events.append("preflight")
        return Preflight()

    def write_manifest(path, value):
        assert events == ["preflight"]
        assert path.name == "manifest.json"
        assert path.parent.resolve().name == "canary"
        assert value is manifest
        events.append("manifest_written")
        return path

    def import_completed_attempts(source_root, destination_root, run_set):
        assert events == ["preflight", "manifest_written"]
        assert source_root.is_absolute()
        assert destination_root.is_absolute()
        assert run_set is RunSet.CANARY
        if expected_source_root is not None:
            assert source_root.resolve() == expected_source_root.resolve()
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


@pytest.mark.parametrize(
    "relationship",
    (
        "equal",
        "destination_nested",
        "source_nested",
        "source_root_symlink",
        "destination_root_symlink",
        "source_run_set_symlink",
        "destination_run_set_symlink",
    ),
)
def test_prepare_only_rejects_invalid_import_topology_before_any_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relationship: str,
) -> None:
    source_root = tmp_path / "source"
    destination_root = tmp_path / "destination"
    source_root.mkdir()
    (source_root / "canary").mkdir()
    (source_root / "canary" / "manifest.json").write_bytes(b"immutable source manifest\n")
    destination_root.mkdir()
    (destination_root / "canary").mkdir()
    protected_roots = [source_root, destination_root]

    if relationship == "equal":
        (destination_root / "canary").rmdir()
        destination_root.rmdir()
        destination_root = source_root
        protected_roots = [source_root]
    elif relationship == "destination_nested":
        (destination_root / "canary").rmdir()
        destination_root.rmdir()
        destination_root = source_root / "nested-destination"
        protected_roots = [source_root]
    elif relationship == "source_nested":
        (destination_root / "canary").rmdir()
        source_root.rename(destination_root / "nested-source")
        source_root = destination_root / "nested-source"
        protected_roots = [destination_root]
    elif relationship == "source_root_symlink":
        source_link = tmp_path / "source-link"
        source_link.symlink_to(source_root, target_is_directory=True)
        source_root = source_link
    elif relationship == "destination_root_symlink":
        (destination_root / "canary").rmdir()
        destination_root.rmdir()
        destination_root.symlink_to(source_root, target_is_directory=True)
        protected_roots = [source_root]
    elif relationship == "source_run_set_symlink":
        source_run_set = source_root / "canary"
        source_run_set.rename(tmp_path / "source-canary")
        source_run_set.symlink_to(tmp_path / "source-canary", target_is_directory=True)
        protected_roots.append(tmp_path / "source-canary")
    elif relationship == "destination_run_set_symlink":
        (destination_root / "canary").rmdir()
        (destination_root / "canary").symlink_to(source_root / "canary", target_is_directory=True)

    snapshots = {root: _inventory(root) for root in protected_roots}
    events: list[str] = []
    _patch_filesystem_preparation(monkeypatch, events)
    _forbid_measured_construction(monkeypatch)

    with pytest.raises(ValueError):
        cli.main(_filesystem_arguments(tmp_path, source_root, destination_root))

    assert {root: _inventory(root) for root in protected_roots} == snapshots
    assert events == []


@pytest.mark.parametrize("swap", ("root", "run_set"))
def test_prepare_only_anchors_destination_writes_against_path_swaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swap: str,
) -> None:
    source_root = tmp_path / "source"
    destination_root = tmp_path / "destination"
    source_run_set = source_root / "canary"
    destination_run_set = destination_root / "canary"
    source_run_set.mkdir(parents=True)
    destination_run_set.mkdir(parents=True)
    (source_run_set / "manifest.json").write_bytes(b"immutable source manifest\n")
    source_before = _inventory(source_root)
    events: list[str] = []
    _patch_filesystem_preparation(monkeypatch, events)
    _forbid_measured_construction(monkeypatch)
    relocated = tmp_path / f"relocated-{swap}"

    class SwapBeforeControllerWrite:
        def __init__(self, artifact_root: Path):
            self.lock = ControllerLock(artifact_root)

        def __enter__(self) -> Any:
            if swap == "root":
                destination_root.rename(relocated)
                destination_root.symlink_to(source_root, target_is_directory=True)
            else:
                destination_run_set.rename(relocated)
                destination_run_set.symlink_to(source_run_set, target_is_directory=True)
            return self.lock.__enter__()

        def __exit__(self, *args: object) -> None:
            self.lock.__exit__(*args)

    monkeypatch.setattr(cli, "ControllerLock", SwapBeforeControllerWrite)

    with pytest.raises(ValueError):
        cli.main(_filesystem_arguments(tmp_path, source_root, destination_root))

    assert _inventory(source_root) == source_before
    assert events == ["preflight", "manifest_written"]


def test_prepare_only_writes_manifest_and_imports_before_executor_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.chdir(tmp_path)
    source_root = Path("source")
    (source_root / "canary").mkdir(parents=True)
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

    assert events == []


def test_prepare_only_propagates_importer_rejection_without_executor_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    (tmp_path / "source" / "canary").mkdir(parents=True)
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
