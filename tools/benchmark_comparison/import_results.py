# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Transactionally import validated completed benchmark attempts."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .artifacts import verify_success
from .manifest import RunSetManifest, read_manifest, resolve_manifest_expansion
from .models import BenchmarkAttempt, RunSet
from .validate import attempt_identity


@dataclass(frozen=True)
class ImportAudit:
    """Deterministic identity and integrity record for one completed import."""

    source_root: Path
    destination_root: Path
    run_set: RunSet
    source_manifest_sha256: str
    destination_manifest_sha256: str
    imported_attempt_count: int
    imported_file_count: int
    source_aggregate_sha256: str
    destination_aggregate_sha256: str

    def to_json(self) -> dict[str, object]:
        """Return the deterministic JSON representation."""
        return {
            "source_root": str(self.source_root),
            "destination_root": str(self.destination_root),
            "run_set": self.run_set.value,
            "source_manifest_sha256": self.source_manifest_sha256,
            "destination_manifest_sha256": self.destination_manifest_sha256,
            "imported_attempt_count": self.imported_attempt_count,
            "imported_file_count": self.imported_file_count,
            "source_aggregate_sha256": self.source_aggregate_sha256,
            "destination_aggregate_sha256": self.destination_aggregate_sha256,
        }


@dataclass(frozen=True)
class _AttemptImport:
    """Validated source and destination mapping for one attempt identity."""

    source_attempt: BenchmarkAttempt
    destination_attempt: BenchmarkAttempt
    source_path: Path


def import_completed_attempts(
    source_root: Path,
    destination_root: Path,
    run_set: RunSet,
) -> ImportAudit:
    """Import a validated completed run-set subset into an expanded root.

    Args:
        source_root: Artifact root containing completed source attempts.
        destination_root: Independent artifact root already containing its schema-2 manifest.
        run_set: Run-set subset to validate and import.

    Returns:
        The immutable audit published with the imported attempt roots.

    Raises:
        ValueError: If roots, manifests, attempts, or an existing audit conflict.
        RuntimeError: If a completed success or aggregate digest is untrustworthy.
        FileExistsError: If a destination attempt root already exists.
    """
    selected_run_set = RunSet(run_set)
    source, destination = _resolve_independent_roots(source_root, destination_root)
    source_run_set = _validate_run_set_directory(source, selected_run_set, "source")
    destination_run_set = _validate_run_set_directory(destination, selected_run_set, "destination")
    source_manifest_path = source_run_set / "manifest.json"
    destination_manifest_path = destination_run_set / "manifest.json"
    source_manifest = read_manifest(source_manifest_path)
    destination_manifest = read_manifest(destination_manifest_path)
    _validate_manifests(source_manifest, destination_manifest, selected_run_set)

    source_expansion = resolve_manifest_expansion(source_manifest, source)
    destination_expansion = resolve_manifest_expansion(destination_manifest, destination)
    imports = _resolve_attempt_imports(
        source,
        source_expansion.attempts,
        destination_expansion.attempts,
        destination_manifest,
    )
    source_digest, source_file_count = _aggregate_digest(
        tuple((item.source_attempt.identity, item.source_path) for item in imports)
    )
    audit = ImportAudit(
        source_root=source,
        destination_root=destination,
        run_set=selected_run_set,
        source_manifest_sha256=_file_sha256(source_manifest_path),
        destination_manifest_sha256=_file_sha256(destination_manifest_path),
        imported_attempt_count=len(imports),
        imported_file_count=source_file_count,
        source_aggregate_sha256=source_digest,
        destination_aggregate_sha256=source_digest,
    )

    audit_path = destination_run_set / "import_audit.json"
    if _path_exists(audit_path):
        _validate_regular_file(audit_path, "destination import audit")
        with _import_lock(destination_run_set):
            return _validate_existing_import(audit_path, audit, imports, destination_manifest)

    destinations = _destination_attempt_paths(destination, destination_run_set, imports)
    _validate_run_set_directory(destination, selected_run_set, "destination")
    staging_root = Path(tempfile.mkdtemp(prefix=".import-staging-", dir=destination_run_set))
    try:
        staged = _stage_attempts(staging_root, imports, destination_manifest)
        staged_digest, staged_file_count = _aggregate_digest(
            tuple((item.destination_attempt.identity, path) for item, path in staged)
        )
        if staged_digest != source_digest or staged_file_count != source_file_count:
            raise RuntimeError("staged import aggregate does not match the source aggregate")

        with _import_lock(destination_run_set):
            if _path_exists(audit_path):
                _validate_regular_file(audit_path, "destination import audit")
                return _validate_existing_import(audit_path, audit, imports, destination_manifest)
            conflicts = tuple(path for path in destinations if _path_exists(path))
            if conflicts:
                raise FileExistsError(f"destination attempt conflict: {conflicts[0]}")

            published: list[tuple[Path, Path]] = []
            try:
                for (item, staged_path), destination_path in zip(staged, destinations, strict=True):
                    if item.destination_attempt.identity != item.source_attempt.identity:
                        raise RuntimeError("validated attempt mapping changed before publication")
                    _publish_attempt_root(staged_path, destination_path)
                    published.append((destination_path, staged_path))
                _write_audit_atomic(audit_path, audit)
            except BaseException:
                _rollback_published_attempts(published)
                raise
        return audit
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)


def _resolve_independent_roots(source_root: Path, destination_root: Path) -> tuple[Path, Path]:
    """Resolve roots and reject equality or containment in either direction."""
    source = source_root.resolve()
    destination = destination_root.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("source and destination roots overlap")
    return source, destination


def _validate_run_set_directory(root: Path, run_set: RunSet, location: str) -> Path:
    """Require a direct, non-symlinked run-set directory beneath ``root``."""
    run_set_root = root / run_set.value
    try:
        metadata = run_set_root.lstat()
    except OSError as error:
        raise ValueError(f"{location} run-set directory is unavailable: {error}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{location} run-set directory must not be a symlink: {run_set_root}")
    if not stat.S_ISDIR(metadata.st_mode) or run_set_root.resolve() != run_set_root:
        raise ValueError(f"{location} run-set directory is not contained directly beneath its root: {run_set_root}")
    return run_set_root


def _destination_attempt_paths(
    destination_root: Path,
    destination_run_set: Path,
    imports: tuple[_AttemptImport, ...],
) -> tuple[Path, ...]:
    """Resolve publication paths only when every parent is the validated run-set root."""
    destinations = tuple(destination_root / item.destination_attempt.run_directory for item in imports)
    if any(path.parent != destination_run_set for path in destinations):
        raise ValueError("destination attempt path is not contained directly beneath the run-set directory")
    return destinations


def _validate_regular_file(path: Path, name: str) -> None:
    """Reject symlinked or non-regular control files."""
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ValueError(f"{name} is unavailable: {error}") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ValueError(f"{name} must be a regular file: {path}")


def _validate_manifests(
    source: RunSetManifest,
    destination: RunSetManifest,
    run_set: RunSet,
) -> None:
    """Require matching schema-2 execution identities."""
    if source.schema_version != "2.0" or destination.schema_version != "2.0":
        raise ValueError("source and destination manifests must use schema 2.0")
    if source.run_set is not run_set or destination.run_set is not run_set:
        raise ValueError("source and destination manifest run sets must match the requested run set")
    fields = ("phase", "provenance", "host", "lab2", "lab3", "cpu_power_profile")
    for field in fields:
        if getattr(source, field) != getattr(destination, field):
            raise ValueError(f"source and destination manifest {field} differ")


def _resolve_attempt_imports(
    source_root: Path,
    source_attempts: tuple[BenchmarkAttempt, ...],
    destination_attempts: tuple[BenchmarkAttempt, ...],
    destination_manifest: RunSetManifest,
) -> tuple[_AttemptImport, ...]:
    """Validate immutable attempt mappings and every completed source root."""
    destination_by_identity = {attempt.identity: attempt for attempt in destination_attempts}
    imports = []
    for source_attempt in sorted(source_attempts, key=lambda attempt: attempt.identity):
        destination_attempt = destination_by_identity.get(source_attempt.identity)
        if destination_attempt is None:
            raise ValueError(f"source attempt identity is absent from destination expansion: {source_attempt.identity}")
        _validate_attempt_mapping(source_attempt, destination_attempt)
        source_path = source_root / source_attempt.run_directory
        _validate_attempt_tree(source_path, "source")
        _validate_success(
            source_path / "success",
            destination_attempt,
            destination_manifest,
            location="source",
        )
        imports.append(_AttemptImport(source_attempt, destination_attempt, source_path))
    return tuple(imports)


def _validate_attempt_mapping(source: BenchmarkAttempt, destination: BenchmarkAttempt) -> None:
    """Reject semantic changes while permitting pair and global order changes."""
    if attempt_identity(source) != attempt_identity(destination):
        raise ValueError(f"attempt semantic identity differs: {source.identity}")
    compared = (
        "run_directory",
        "enable_cameras",
        "extra_presets",
        "version_order",
        "mode",
    )
    for field in compared:
        if getattr(source, field) != getattr(destination, field):
            raise ValueError(f"attempt {field} differs: {source.identity}")


def _validate_attempt_tree(root: Path, location: str) -> None:
    """Reject absent, non-directory, symlinked, or special-file attempt trees."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{location} attempt root must be a directory: {root}")
    for path in sorted(root.rglob("*")):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{location} attempt root contains a symlink: {path}")
        if not stat.S_ISDIR(metadata.st_mode) and not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{location} attempt root contains a special file: {path}")


def _validate_success(
    success_path: Path,
    attempt: BenchmarkAttempt,
    manifest: RunSetManifest,
    *,
    location: str,
) -> None:
    """Validate one success against destination semantics and execution identity."""
    gpu_uuid = manifest.host.gpu_uuid
    if gpu_uuid is None or not verify_success(
        success_path,
        attempt,
        expected_provenance=manifest.provenance,
        expected_gpu_uuid=gpu_uuid,
    ):
        raise RuntimeError(f"{location} success failed validation: {success_path}")


def _validate_staged_success(
    success_path: Path,
    attempt: BenchmarkAttempt,
    manifest: RunSetManifest,
) -> None:
    """Named validation seam for staged success fault injection."""
    _validate_success(success_path, attempt, manifest, location="staged")


def _stage_attempts(
    staging_root: Path,
    imports: tuple[_AttemptImport, ...],
    destination_manifest: RunSetManifest,
) -> tuple[tuple[_AttemptImport, Path], ...]:
    """Copy complete attempt roots into one importer-owned staging directory."""
    staged = []
    for item in imports:
        staged_path = staging_root / item.destination_attempt.identity
        shutil.copytree(
            item.source_path,
            staged_path,
            symlinks=False,
            copy_function=shutil.copy2,
        )
        _validate_attempt_tree(staged_path, "staged")
        _validate_staged_success(staged_path / "success", item.destination_attempt, destination_manifest)
        staged.append((item, staged_path))
    return tuple(staged)


def _aggregate_digest(attempt_roots: tuple[tuple[str, Path], ...]) -> tuple[str, int]:
    """Hash sorted identities, relative file paths, modes, sizes, and bytes."""
    aggregate = hashlib.sha256()
    file_count = 0
    for identity, root in sorted(attempt_roots):
        _validate_attempt_tree(root, "aggregate")
        for path in sorted(root.rglob("*")):
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                continue
            contents = path.read_bytes()
            header = {
                "identity": identity,
                "path": path.relative_to(root).as_posix(),
                "mode": metadata.st_mode,
                "size": metadata.st_size,
            }
            aggregate.update(
                json.dumps(
                    header,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode()
            )
            aggregate.update(b"\n")
            aggregate.update(contents)
            file_count += 1
    return aggregate.hexdigest(), file_count


def _validate_existing_import(
    audit_path: Path,
    expected_audit: ImportAudit,
    imports: tuple[_AttemptImport, ...],
    destination_manifest: RunSetManifest,
) -> ImportAudit:
    """Return an identical import only after revalidating all destination bytes."""
    try:
        existing_bytes = audit_path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read destination import audit: {error}") from error
    if existing_bytes != _audit_bytes(expected_audit):
        raise ValueError(f"conflicting destination import audit: {audit_path}")

    destination_roots = []
    for item in imports:
        destination_path = expected_audit.destination_root / item.destination_attempt.run_directory
        _validate_attempt_tree(destination_path, "destination")
        _validate_success(
            destination_path / "success",
            item.destination_attempt,
            destination_manifest,
            location="destination",
        )
        destination_roots.append((item.destination_attempt.identity, destination_path))
    digest, file_count = _aggregate_digest(tuple(destination_roots))
    if digest != expected_audit.destination_aggregate_sha256 or file_count != expected_audit.imported_file_count:
        raise RuntimeError("destination aggregate does not match the import audit")
    return expected_audit


def _publish_attempt_root(staged: Path, destination: Path) -> None:
    """Publish one staged attempt root atomically."""
    os.replace(staged, destination)


def _rollback_published_attempts(published: list[tuple[Path, Path]]) -> None:
    """Move newly published roots back to importer-owned staging."""
    rollback_errors = []
    for destination, staged in reversed(published):
        try:
            os.replace(destination, staged)
        except OSError as error:
            rollback_errors.append(error)
    if rollback_errors:
        raise RuntimeError("failed to roll back a partially published import") from rollback_errors[0]


def _write_audit_atomic(path: Path, audit: ImportAudit) -> None:
    """Write and fsync a deterministic audit before atomic publication."""
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(_audit_bytes(audit))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _audit_bytes(audit: ImportAudit) -> bytes:
    """Encode an audit as deterministic UTF-8 JSON with a trailing newline."""
    return (
        json.dumps(
            audit.to_json(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 of one file's exact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path_exists(path: Path) -> bool:
    """Return whether a path exists without ignoring broken symlinks."""
    return os.path.lexists(path)


@contextmanager
def _import_lock(run_set_root: Path) -> Iterator[None]:
    """Serialize import publication and idempotent validation."""
    lock_path = run_set_root / ".import.lock"
    if _path_exists(lock_path):
        _validate_regular_file(lock_path, "destination import lock")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o666)
    except OSError as error:
        raise ValueError(f"cannot safely open destination import lock: {error}") from error
    with os.fdopen(descriptor, "r+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
