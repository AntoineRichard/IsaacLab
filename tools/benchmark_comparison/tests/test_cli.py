# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for safe comparison-controller preparation and result import."""

from __future__ import annotations

import json
import os
import re
import stat
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tools.benchmark_comparison import cli
from tools.benchmark_comparison.artifacts import finalize_attempt
from tools.benchmark_comparison.import_results import ImportAudit
from tools.benchmark_comparison.manifest import HostIdentity, RunSetManifest, SoftwareIdentity, write_manifest
from tools.benchmark_comparison.matrix import expand_canary_matrix, load_matrix
from tools.benchmark_comparison.models import BenchmarkAttempt, ExecutionProvenance, MatrixExpansion, RunSet
from tools.benchmark_comparison.runner import ControllerLock, ControllerLockError
from tools.benchmark_comparison.validate import attempt_identity

LAB2_SHA = "a" * 40
LAB3_SHA = "b" * 40
LAB2_IMAGE_ID = "sha256:" + "c" * 64
LAB3_LOCK = "d" * 64
GPU_UUID = "GPU-01234567-89ab-cdef-0123-456789abcdef"
FIXTURES = Path(__file__).with_name("fixtures")
PROVENANCE = ExecutionProvenance(
    lab2_sha=LAB2_SHA,
    lab3_sha=LAB3_SHA,
    lab2_image_id=LAB2_IMAGE_ID,
    uv_lock_sha256=LAB3_LOCK,
)


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


def _single_pair_expansion() -> MatrixExpansion:
    """Return the first complete canary pair with canonical local ordering."""
    pair = expand_canary_matrix(load_matrix()).pairs[0]
    attempts = tuple(
        replace(attempt, pair_order=0, version_order=index, attempt_order=index)
        for index, attempt in enumerate(pair.attempts)
    )
    return MatrixExpansion(
        run_set=RunSet.CANARY,
        pairs=(replace(pair, pair_order=0, attempts=attempts),),
        attempts=attempts,
    )


def _import_manifest(expansion: MatrixExpansion) -> RunSetManifest:
    """Return a valid schema-2 manifest for the retained-destination regression."""
    software = SoftwareIdentity(
        isaac_lab="2.3.2",
        isaac_sim="5.1.0",
        python="3.11.13",
        pytorch="2.7.0+cu128",
        rsl_rl="5.0.1",
    )
    return RunSetManifest(
        schema_version="2.0",
        run_set=RunSet.CANARY,
        phase="measured",
        provenance=PROVENANCE,
        host=HostIdentity(
            hostname="benchmark-host",
            os="Ubuntu 24.04",
            cpu_model="Fixture CPU",
            logical_cpu_count=32,
            gpu_model="Fixture GPU",
            gpu_driver="590.48.01",
            cuda_version="13.0",
            gpu_index=0,
            gpu_uuid=GPU_UUID,
        ),
        lab2=software,
        lab3=replace(software, isaac_lab="3.0.0"),
        cpu_power_profile="performance",
        expansion=expansion,
    )


def _successful_payloads(attempt: BenchmarkAttempt) -> dict[str, Any]:
    """Return valid synthetic source-attempt payloads for a real import."""
    identity = attempt_identity(attempt)
    schema_name = "schema_training.json" if attempt.mode.id.startswith("training") else "schema_runtime.json"
    schema = json.loads((FIXTURES / schema_name).read_text(encoding="utf-8"))
    schema["run"]["task"] = attempt.concrete_task
    schema["run"]["seed"] = attempt.seed
    schema["run"]["num_envs"] = attempt.num_envs
    schema["versions"]["isaaclab_release"] = "2.3.2" if attempt.version.value == "lab2" else "3.0.0"
    schema["runtime"]["iterations_completed"] = attempt.bound.value
    if attempt.bound.unit.value == "iterations":
        schema["run"]["max_iterations"] = attempt.bound.value
    return {
        "command": {"identity": identity, "argv": ["fake-benchmark"]},
        "environment": {
            "identity": identity,
            "values": {
                "ISAACLAB_BENCHMARK_LAB2_SHA": LAB2_SHA,
                "ISAACLAB_BENCHMARK_LAB3_SHA": LAB3_SHA,
                "ISAACLAB_BENCHMARK_LAB2_IMAGE_ID": LAB2_IMAGE_ID,
                "ISAACLAB_BENCHMARK_UV_LOCK_SHA256": LAB3_LOCK,
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": "0",
                "NVIDIA_VISIBLE_DEVICES": "0",
                "ISAACLAB_BENCHMARK_GPU_INDEX": "0",
                "ISAACLAB_BENCHMARK_GPU_UUID": GPU_UUID,
            },
            "selected_gpu": {"physical_index": 0, "uuid": GPU_UUID},
            "environment_identity": PROVENANCE.environment_identity(attempt.version),
            **PROVENANCE.to_json(),
        },
        "stdout": "synthetic benchmark output\n",
        "stderr": "",
        "exit_status": {
            "exit_code": 0,
            "failure_stage": None,
            "timed_out": False,
            "interrupted": False,
            "out_of_memory": False,
        },
        "schema": schema,
        "measurements": json.loads((FIXTURES / "generic_runtime.json").read_text(encoding="utf-8")),
    }


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


@pytest.mark.parametrize("leaf_kind", ("symlink", "fifo"))
def test_controller_lock_rejects_non_regular_leaf_without_touching_source(
    tmp_path: Path,
    leaf_kind: str,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_file = source_root / "immutable.txt"
    source_file.write_bytes(b"immutable source bytes\n")
    source_before = _inventory(source_root)
    destination_root = tmp_path / "destination"
    destination_root.mkdir()
    lock_path = destination_root / ".benchmark-controller.lock"
    if leaf_kind == "symlink":
        lock_path.symlink_to(source_file)
    else:
        os.mkfifo(lock_path)

    started = time.monotonic()
    with pytest.raises(ControllerLockError, match="regular file"):
        with ControllerLock(destination_root):
            pass

    assert time.monotonic() - started < 0.5
    assert _inventory(source_root) == source_before


def test_controller_lock_descriptor_contention_reports_canonical_artifact_root(tmp_path: Path) -> None:
    destination_root = tmp_path / "destination"
    destination_root.mkdir()
    descriptor = os.open(destination_root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    descriptor_path = Path("/proc/self/fd") / str(descriptor)
    try:
        with ControllerLock(descriptor_path):
            with pytest.raises(ControllerLockError, match=re.escape(str(destination_root.resolve()))):
                with ControllerLock(descriptor_path):
                    pass
    finally:
        os.close(descriptor)


def test_prepare_only_import_stays_on_retained_destination_after_post_config_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expansion = _single_pair_expansion()
    expected_manifest = _import_manifest(expansion)
    source_root = tmp_path / "source"
    destination_root = tmp_path / "destination"
    destination_run_set = destination_root / "canary"
    destination_run_set.mkdir(parents=True)
    write_manifest(source_root / "canary" / "manifest.json", expected_manifest)
    for attempt in expansion.attempts:
        finalize_attempt(source_root, attempt, **_successful_payloads(attempt))
    source_before = _inventory(source_root)
    original_inode = destination_root.stat().st_ino
    relocated_destination = tmp_path / "original-destination"
    real_write_manifest = cli.write_manifest

    class Preflight:
        def manifest(self, _run_set, _phase, _expansion):
            return expected_manifest

    def run_preflight(config, *, artifact_root_for_writes=None):
        assert config.artifact_root == destination_root.resolve()
        assert artifact_root_for_writes is not None
        return Preflight()

    def write_manifest_then_swap(path: Path, manifest: RunSetManifest) -> Path:
        written = real_write_manifest(path, manifest)
        destination_root.rename(relocated_destination)
        (destination_root / "canary").mkdir(parents=True)
        real_write_manifest(destination_root / "canary" / "manifest.json", manifest)
        return written

    monkeypatch.setattr(cli, "run_preflight", run_preflight)
    monkeypatch.setattr(cli, "write_manifest", write_manifest_then_swap)
    _forbid_measured_construction(monkeypatch)

    assert cli.main(_filesystem_arguments(tmp_path, source_root, destination_root)) == 0

    assert relocated_destination.stat().st_ino == original_inode
    assert (relocated_destination / "canary" / "import_audit.json").is_file()
    assert all((relocated_destination / attempt.run_directory / "success").is_dir() for attempt in expansion.attempts)
    assert not (destination_root / "canary" / "import_audit.json").exists()
    assert all(not (destination_root / attempt.run_directory).exists() for attempt in expansion.attempts)
    assert _inventory(source_root) == source_before


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
