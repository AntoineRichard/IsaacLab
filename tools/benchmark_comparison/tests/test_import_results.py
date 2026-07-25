# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for transactional import of completed benchmark attempts."""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest

from tools.benchmark_comparison import import_results
from tools.benchmark_comparison.artifacts import finalize_attempt, verify_success
from tools.benchmark_comparison.import_results import import_completed_attempts
from tools.benchmark_comparison.manifest import HostIdentity, RunSetManifest, SoftwareIdentity, write_manifest
from tools.benchmark_comparison.matrix import expand_canary_matrix, expand_final_matrix, load_matrix
from tools.benchmark_comparison.models import BenchmarkAttempt, ExecutionProvenance, MatrixExpansion, RunSet
from tools.benchmark_comparison.validate import attempt_identity

_FIXTURES = Path(__file__).with_name("fixtures")
_LAB2_SHA = "a" * 40
_LAB3_SHA = "b" * 40
_LAB2_IMAGE_ID = "sha256:" + "c" * 64
_LAB3_LOCK_SHA256 = "d" * 64
_GPU_UUID = "GPU-01234567-89ab-cdef-0123-456789abcdef"
_PROVENANCE = ExecutionProvenance(
    lab2_sha=_LAB2_SHA,
    lab3_sha=_LAB3_SHA,
    lab2_image_id=_LAB2_IMAGE_ID,
    uv_lock_sha256=_LAB3_LOCK_SHA256,
)


def _select_pairs(expansion: MatrixExpansion, indices: tuple[int, ...]) -> MatrixExpansion:
    """Return a valid schema-2 subset whose pair ordering follows ``indices``."""
    pairs = []
    attempts = []
    for pair_order, index in enumerate(indices):
        pair = expansion.pairs[index]
        pair_attempts = tuple(
            replace(attempt, pair_order=pair_order, attempt_order=pair_order * 2 + version_order)
            for version_order, attempt in enumerate(pair.attempts)
        )
        pairs.append(replace(pair, pair_order=pair_order, attempts=pair_attempts))
        attempts.extend(pair_attempts)
    return MatrixExpansion(run_set=expansion.run_set, pairs=tuple(pairs), attempts=tuple(attempts))


def _manifest(expansion: MatrixExpansion) -> RunSetManifest:
    """Return exact synthetic schema-2 benchmark provenance."""
    return RunSetManifest(
        schema_version="2.0",
        run_set=expansion.run_set,
        phase="measured",
        provenance=_PROVENANCE,
        host=HostIdentity(
            hostname="benchmark-host",
            os="Ubuntu 24.04",
            cpu_model="Fixture CPU",
            logical_cpu_count=32,
            gpu_model="Fixture GPU",
            gpu_driver="590.48.01",
            cuda_version="13.0",
            gpu_index=0,
            gpu_uuid=_GPU_UUID,
        ),
        lab2=SoftwareIdentity(
            isaac_lab="2.3.2",
            isaac_sim="5.1.0",
            python="3.11.13",
            pytorch="2.7.0+cu128",
            rsl_rl="5.0.1",
        ),
        lab3=SoftwareIdentity(
            isaac_lab="3.0.0",
            isaac_sim="6.0.0",
            python="3.12.13",
            pytorch="2.11.0+cu128",
            rsl_rl="5.4.1",
        ),
        cpu_power_profile="performance",
        expansion=expansion,
    )


def _schema(attempt: BenchmarkAttempt) -> dict[str, Any]:
    """Return semantic benchmark output matching ``attempt``."""
    name = "schema_training.json" if attempt.mode.id.startswith("training") else "schema_runtime.json"
    schema = json.loads((_FIXTURES / name).read_text(encoding="utf-8"))
    schema["run"]["task"] = attempt.concrete_task
    schema["run"]["seed"] = attempt.seed
    schema["run"]["num_envs"] = attempt.num_envs
    schema["versions"]["isaaclab_release"] = "2.3.2" if attempt.version.value == "lab2" else "3.0.0"
    schema["runtime"]["iterations_completed"] = attempt.bound.value
    if attempt.bound.unit.value == "iterations":
        schema["run"]["max_iterations"] = attempt.bound.value
    return schema


def _payloads(attempt: BenchmarkAttempt, *, exit_code: int = 0) -> dict[str, Any]:
    """Return finalization payloads with exact provenance and selected GPU."""
    identity = attempt_identity(attempt)
    succeeded = exit_code == 0
    return {
        "command": {"identity": identity, "argv": ["fake-benchmark"]},
        "environment": {
            "identity": identity,
            "values": {
                "ISAACLAB_BENCHMARK_LAB2_SHA": _LAB2_SHA,
                "ISAACLAB_BENCHMARK_LAB3_SHA": _LAB3_SHA,
                "ISAACLAB_BENCHMARK_LAB2_IMAGE_ID": _LAB2_IMAGE_ID,
                "ISAACLAB_BENCHMARK_UV_LOCK_SHA256": _LAB3_LOCK_SHA256,
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": "0",
                "NVIDIA_VISIBLE_DEVICES": "0",
                "ISAACLAB_BENCHMARK_GPU_INDEX": "0",
                "ISAACLAB_BENCHMARK_GPU_UUID": _GPU_UUID,
            },
            "selected_gpu": {"physical_index": 0, "uuid": _GPU_UUID},
            "environment_identity": _PROVENANCE.environment_identity(attempt.version),
            **_PROVENANCE.to_json(),
        },
        "stdout": "synthetic benchmark output\n",
        "stderr": "" if succeeded else "synthetic failure\n",
        "exit_status": {
            "exit_code": exit_code,
            "failure_stage": None,
            "timed_out": False,
            "interrupted": False,
            "out_of_memory": False,
        },
        "schema": _schema(attempt) if succeeded else None,
        "measurements": (
            json.loads((_FIXTURES / "generic_runtime.json").read_text(encoding="utf-8")) if succeeded else None
        ),
    }


def _inventory(root: Path) -> dict[str, tuple[int, int, int, int, bytes | None]]:
    """Capture source bytes and metadata visible without following symlinks."""
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


def _create_import_fixture(tmp_path: Path) -> tuple[Path, Path, MatrixExpansion, MatrixExpansion]:
    """Create a two-attempt source and reordered four-attempt destination manifest."""
    expansion = expand_canary_matrix(load_matrix())
    source_expansion = _select_pairs(expansion, (0,))
    destination_expansion = _select_pairs(expansion, (1, 0))
    source_root = tmp_path / "source"
    destination_root = tmp_path / "destination"
    write_manifest(source_root / "canary" / "manifest.json", _manifest(source_expansion))
    write_manifest(destination_root / "canary" / "manifest.json", _manifest(destination_expansion))

    first, retrying = source_expansion.attempts
    finalize_attempt(source_root, first, **_payloads(first))
    finalize_attempt(source_root, retrying, **_payloads(retrying, exit_code=7))
    finalize_attempt(source_root, retrying, **_payloads(retrying))
    return source_root, destination_root, source_expansion, destination_expansion


def _write_destination_manifest(destination_root: Path, manifest: RunSetManifest) -> None:
    """Replace the fixture destination manifest with another valid identity."""
    path = destination_root / "canary" / "manifest.json"
    path.unlink()
    write_manifest(path, manifest)


def _map_expansion_attempts(expansion: MatrixExpansion, transform) -> MatrixExpansion:
    """Apply an attempt transform while keeping reconstructed pair values consistent."""
    mapped = {attempt.identity: transform(attempt) for attempt in expansion.attempts}
    pairs = []
    for pair in expansion.pairs:
        pair_attempts = tuple(mapped[attempt.identity] for attempt in pair.attempts)
        pairs.append(replace(pair, mode=pair_attempts[0].mode, bound=pair_attempts[0].bound, attempts=pair_attempts))
    return MatrixExpansion(
        run_set=expansion.run_set,
        pairs=tuple(pairs),
        attempts=tuple(mapped[attempt.identity] for attempt in expansion.attempts),
    )


def _rewrite_manifest_attempt_digest(path: Path, mutation) -> None:
    """Mutate a raw attempt document while preserving its enclosing canonical digest."""
    document = json.loads(path.read_text(encoding="utf-8"))
    mutation(document["run_set_identity"]["attempts"])
    encoded = json.dumps(
        document["run_set_identity"]["attempts"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    document["run_set_identity"]["sha256"] = hashlib.sha256(encoded).hexdigest()
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rewrite_checksums(directory: Path) -> None:
    """Recompute checksums after a deliberate validation-document mutation."""
    lines = []
    for line in (directory / "checksums.sha256").read_text(encoding="ascii").splitlines():
        filename = line[66:]
        digest = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        lines.append(f"{digest}  {filename}")
    (directory / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="ascii")


def _assert_no_import_published(
    destination_root: Path,
    source_expansion: MatrixExpansion,
    *,
    audit_may_exist: bool = False,
) -> None:
    """Assert a failed transaction left neither attempts nor staging residue."""
    assert not any((destination_root / attempt.run_directory).exists() for attempt in source_expansion.attempts)
    assert not tuple((destination_root / "canary").glob(".import-staging-*"))
    if not audit_may_exist:
        assert not (destination_root / "canary" / "import_audit.json").exists()


def test_import_completed_attempts_copies_full_attempt_history_independently(tmp_path: Path) -> None:
    source_root, destination_root, source_expansion, destination_expansion = _create_import_fixture(tmp_path)
    source_before = _inventory(source_root)

    audit = import_completed_attempts(source_root, destination_root, RunSet.CANARY)

    destination_by_identity = {attempt.identity: attempt for attempt in destination_expansion.attempts}
    source_files = {
        (attempt.identity, path.relative_to(source_root / attempt.run_directory).as_posix()): path
        for attempt in source_expansion.attempts
        for path in (source_root / attempt.run_directory).rglob("*")
        if path.is_file()
    }
    destination_files = {
        (attempt.identity, path.relative_to(destination_root / attempt.run_directory).as_posix()): path
        for attempt in source_expansion.attempts
        for path in (destination_root / attempt.run_directory).rglob("*")
        if path.is_file()
    }
    assert audit.imported_attempt_count == 2
    assert audit.imported_file_count == len(source_files)
    assert set(destination_files) == set(source_files)
    retry_root = destination_root / source_expansion.attempts[1].run_directory
    assert {path.name for path in retry_root.iterdir()} >= {"attempt-0001-nonzero_exit", "success"}
    assert all(
        verify_success(
            destination_root / attempt.run_directory / "success",
            destination_by_identity[attempt.identity],
            expected_provenance=_PROVENANCE,
            expected_gpu_uuid=_GPU_UUID,
        )
        for attempt in source_expansion.attempts
    )
    assert audit.source_aggregate_sha256 == audit.destination_aggregate_sha256
    assert all(source_files[key].read_bytes() == destination_files[key].read_bytes() for key in source_files)
    assert all(source_files[key].stat().st_ino != destination_files[key].stat().st_ino for key in source_files)
    assert not any(path.is_symlink() for path in destination_root.rglob("*"))
    assert not tuple((destination_root / "canary").glob(".import-staging-*"))
    assert _inventory(source_root) == source_before
    audit_document = json.loads((destination_root / "canary" / "import_audit.json").read_text(encoding="utf-8"))
    assert audit_document == audit.to_json()
    with pytest.raises(FrozenInstanceError):
        audit.imported_attempt_count = 3  # type: ignore[misc]
    assert (
        audit.source_manifest_sha256
        == hashlib.sha256((source_root / "canary" / "manifest.json").read_bytes()).hexdigest()
    )
    assert (
        audit.destination_manifest_sha256
        == hashlib.sha256((destination_root / "canary" / "manifest.json").read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("relationship", ["equal", "nested", "reverse_nested"])
def test_import_completed_attempts_rejects_overlapping_roots(tmp_path: Path, relationship: str) -> None:
    source_root = tmp_path / "source"
    destination_root = tmp_path / "destination"
    if relationship == "equal":
        destination_root = source_root
    elif relationship == "nested":
        destination_root = source_root / "nested"
    else:
        source_root = destination_root / "nested"
    source_root.mkdir(parents=True, exist_ok=True)
    destination_root.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="overlap"):
        import_completed_attempts(source_root, destination_root, RunSet.CANARY)


@pytest.mark.parametrize("target_kind", ["source", "external"])
def test_import_completed_attempts_rejects_symlinked_destination_run_set_without_writes(
    tmp_path: Path,
    target_kind: str,
) -> None:
    source_root, destination_root, _, _ = _create_import_fixture(tmp_path)
    destination_run_set = destination_root / "canary"
    if target_kind == "source":
        shutil.rmtree(destination_run_set)
        target = source_root / "canary"
    else:
        target = tmp_path / "external" / "canary"
        target.parent.mkdir()
        destination_run_set.rename(target)
    destination_run_set.symlink_to(target, target_is_directory=True)
    source_before = _inventory(source_root)
    target_before = _inventory(target)

    with pytest.raises(ValueError, match="destination.*symlink|contained"):
        import_completed_attempts(source_root, destination_root, RunSet.CANARY)

    assert _inventory(source_root) == source_before
    assert _inventory(target) == target_before
    assert not (target / ".import.lock").exists()
    assert not tuple(target.glob(".import-staging-*"))


@pytest.mark.parametrize(
    "difference",
    ["run_set", "phase", "provenance", "host", "gpu", "lab2", "lab3", "cpu_power_profile"],
)
def test_import_completed_attempts_rejects_manifest_identity_differences(
    tmp_path: Path,
    difference: str,
) -> None:
    source_root, destination_root, _, destination_expansion = _create_import_fixture(tmp_path)
    destination_manifest = _manifest(destination_expansion)
    if difference == "run_set":
        destination_manifest = _manifest(_select_pairs(expand_final_matrix(load_matrix()), (0,)))
    elif difference == "phase":
        destination_manifest = replace(destination_manifest, phase="rerun")
    elif difference == "provenance":
        destination_manifest = replace(
            destination_manifest,
            provenance=replace(destination_manifest.provenance, lab2_sha="e" * 40),
        )
    elif difference == "host":
        destination_manifest = replace(destination_manifest, host=replace(destination_manifest.host, hostname="other"))
    elif difference == "gpu":
        destination_manifest = replace(
            destination_manifest,
            host=replace(destination_manifest.host, gpu_uuid="GPU-DIFFERENT-0000"),
        )
    elif difference == "lab2":
        destination_manifest = replace(
            destination_manifest,
            lab2=replace(destination_manifest.lab2, isaac_lab="2.3.3"),
        )
    elif difference == "lab3":
        destination_manifest = replace(
            destination_manifest,
            lab3=replace(destination_manifest.lab3, isaac_lab="3.0.1"),
        )
    else:
        destination_manifest = replace(destination_manifest, cpu_power_profile="powersave")
    _write_destination_manifest(destination_root, destination_manifest)

    with pytest.raises(ValueError, match="manifest"):
        import_completed_attempts(source_root, destination_root, RunSet.CANARY)


def test_import_completed_attempts_rejects_source_identity_absent_from_destination(tmp_path: Path) -> None:
    source_root, destination_root, source_expansion, _ = _create_import_fixture(tmp_path)
    disjoint_destination = _select_pairs(expand_canary_matrix(load_matrix()), (1,))
    _write_destination_manifest(destination_root, _manifest(disjoint_destination))

    with pytest.raises(ValueError, match="absent"):
        import_completed_attempts(source_root, destination_root, RunSet.CANARY)

    _assert_no_import_published(destination_root, source_expansion)


@pytest.mark.parametrize("difference", ["concrete_task", "bound", "enable_cameras", "extra_presets"])
def test_import_completed_attempts_rejects_changed_semantics_for_same_directory_identity(
    tmp_path: Path,
    difference: str,
) -> None:
    source_root, destination_root, source_expansion, destination_expansion = _create_import_fixture(tmp_path)
    target = source_expansion.attempts[0]

    def transform(attempt: BenchmarkAttempt) -> BenchmarkAttempt:
        if difference == "bound" and attempt.logical_pair_identity == target.logical_pair_identity:
            changed_mode = replace(
                attempt.mode,
                final_bound=replace(attempt.mode.final_bound, value=attempt.mode.final_bound.value + 1),
            )
            return replace(attempt, mode=changed_mode)
        if attempt.identity != target.identity:
            return attempt
        if difference == "concrete_task":
            return replace(attempt, concrete_task="Different-Concrete-Task")
        if difference == "enable_cameras":
            return replace(attempt, enable_cameras=not attempt.enable_cameras)
        return replace(attempt, extra_presets=("different=preset",))

    changed_expansion = _map_expansion_attempts(destination_expansion, transform)
    _write_destination_manifest(destination_root, _manifest(changed_expansion))

    with pytest.raises(ValueError, match="attempt"):
        import_completed_attempts(source_root, destination_root, RunSet.CANARY)

    _assert_no_import_published(destination_root, source_expansion)


def test_import_completed_attempts_rejects_changed_framework_for_same_directory_identity(tmp_path: Path) -> None:
    source_root, destination_root, source_expansion, _ = _create_import_fixture(tmp_path)
    target_identity = source_expansion.attempts[0].identity
    manifest_path = destination_root / "canary" / "manifest.json"

    def mutate(attempts: list[dict[str, Any]]) -> None:
        target = next(attempt for attempt in attempts if attempt["identity"] == target_identity)
        target["framework"] = "different-framework"
        target["mode"]["framework"] = "different-framework"

    _rewrite_manifest_attempt_digest(manifest_path, mutate)

    with pytest.raises(ValueError, match="identity"):
        import_completed_attempts(source_root, destination_root, RunSet.CANARY)

    _assert_no_import_published(destination_root, source_expansion)


@pytest.mark.parametrize("problem", ["absent", "file", "no_success", "symlink", "corrupt_success"])
def test_import_completed_attempts_rejects_untrustworthy_source_roots(tmp_path: Path, problem: str) -> None:
    source_root, destination_root, source_expansion, _ = _create_import_fixture(tmp_path)
    attempt_root = source_root / source_expansion.attempts[0].run_directory
    if problem == "absent":
        shutil.rmtree(attempt_root)
    elif problem == "file":
        shutil.rmtree(attempt_root)
        attempt_root.write_text("not a directory\n", encoding="utf-8")
    elif problem == "no_success":
        shutil.rmtree(attempt_root / "success")
    elif problem == "symlink":
        (attempt_root / "manifest-link").symlink_to(source_root / "canary" / "manifest.json")
    else:
        (attempt_root / "success" / "stdout.log").write_text("corrupt\n", encoding="utf-8")

    with pytest.raises((ValueError, RuntimeError), match="source|success|symlink"):
        import_completed_attempts(source_root, destination_root, RunSet.CANARY)

    _assert_no_import_published(destination_root, source_expansion)


def test_import_completed_attempts_rejects_success_number_mismatching_retry_history(tmp_path: Path) -> None:
    source_root, destination_root, source_expansion, _ = _create_import_fixture(tmp_path)
    success = source_root / source_expansion.attempts[1].run_directory / "success"
    validation_path = success / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["attempt_number"] = 1
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rewrite_checksums(success)

    with pytest.raises((ValueError, RuntimeError), match="success"):
        import_completed_attempts(source_root, destination_root, RunSet.CANARY)

    _assert_no_import_published(destination_root, source_expansion)


@pytest.mark.parametrize("conflict", ["attempt", "audit"])
def test_import_completed_attempts_rejects_destination_conflicts(tmp_path: Path, conflict: str) -> None:
    source_root, destination_root, source_expansion, _ = _create_import_fixture(tmp_path)
    if conflict == "attempt":
        (destination_root / source_expansion.attempts[0].run_directory).mkdir(parents=True)
    else:
        (destination_root / "canary" / "import_audit.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises((FileExistsError, ValueError), match="destination|audit|conflict"):
        import_completed_attempts(source_root, destination_root, RunSet.CANARY)

    if conflict == "audit":
        _assert_no_import_published(destination_root, source_expansion, audit_may_exist=True)
    else:
        assert not (destination_root / source_expansion.attempts[1].run_directory).exists()
        assert not tuple((destination_root / "canary").glob(".import-staging-*"))
        assert not (destination_root / "canary" / "import_audit.json").exists()


def test_import_completed_attempts_rolls_back_copy2_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, destination_root, source_expansion, _ = _create_import_fixture(tmp_path)

    def fail_copy(_source: Path, _destination: Path, *, follow_symlinks: bool = True) -> None:
        raise OSError("injected copy2 failure")

    monkeypatch.setattr(import_results.shutil, "copy2", fail_copy)
    with pytest.raises(OSError, match="copy2"):
        import_completed_attempts(source_root, destination_root, RunSet.CANARY)

    _assert_no_import_published(destination_root, source_expansion)


def test_import_completed_attempts_rolls_back_staged_validation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, destination_root, source_expansion, _ = _create_import_fixture(tmp_path)

    def fail_validation(*_args, **_kwargs) -> None:
        raise RuntimeError("injected staged validation failure")

    monkeypatch.setattr(import_results, "_validate_staged_success", fail_validation)
    with pytest.raises(RuntimeError, match="staged validation"):
        import_completed_attempts(source_root, destination_root, RunSet.CANARY)

    _assert_no_import_published(destination_root, source_expansion)


def test_import_completed_attempts_rolls_back_first_publication_when_second_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, destination_root, source_expansion, _ = _create_import_fixture(tmp_path)
    real_publish = import_results._publish_attempt_root
    publications = 0

    def fail_second_publication(staged: Path, destination: Path) -> None:
        nonlocal publications
        publications += 1
        if publications == 2:
            raise OSError("injected publication failure")
        real_publish(staged, destination)

    monkeypatch.setattr(import_results, "_publish_attempt_root", fail_second_publication)
    with pytest.raises(OSError, match="publication"):
        import_completed_attempts(source_root, destination_root, RunSet.CANARY)

    assert publications == 2
    _assert_no_import_published(destination_root, source_expansion)


def test_import_completed_attempts_rolls_back_audit_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, destination_root, source_expansion, _ = _create_import_fixture(tmp_path)

    def fail_audit(_path: Path, _audit) -> None:
        raise OSError("injected audit write failure")

    monkeypatch.setattr(import_results, "_write_audit_atomic", fail_audit)
    with pytest.raises(OSError, match="audit write"):
        import_completed_attempts(source_root, destination_root, RunSet.CANARY)

    _assert_no_import_published(destination_root, source_expansion)


def test_import_completed_attempts_is_byte_stable_when_repeated_identically(tmp_path: Path) -> None:
    source_root, destination_root, _, _ = _create_import_fixture(tmp_path)
    first = import_completed_attempts(source_root, destination_root, RunSet.CANARY)
    destination_before = _inventory(destination_root)

    second = import_completed_attempts(source_root, destination_root, RunSet.CANARY)

    assert second == first
    assert _inventory(destination_root) == destination_before


def test_import_completed_attempts_fails_closed_when_published_attempt_is_tampered(tmp_path: Path) -> None:
    source_root, destination_root, source_expansion, _ = _create_import_fixture(tmp_path)
    import_completed_attempts(source_root, destination_root, RunSet.CANARY)
    audit_before = (destination_root / "canary" / "import_audit.json").read_bytes()
    tampered = destination_root / source_expansion.attempts[0].run_directory / "success" / "stdout.log"
    tampered.write_text("tampered after import\n", encoding="utf-8")

    with pytest.raises((ValueError, RuntimeError), match="destination|success|aggregate"):
        import_completed_attempts(source_root, destination_root, RunSet.CANARY)

    assert (destination_root / "canary" / "import_audit.json").read_bytes() == audit_before
    assert not tuple((destination_root / "canary").glob(".import-staging-*"))
