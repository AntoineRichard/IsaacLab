# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Transactionally import validated completed benchmark attempts."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import verify_success
from .manifest import RunSetManifest, read_manifest, resolve_manifest_expansion
from .models import BenchmarkAttempt, RunSet
from .validate import attempt_identity

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_LIBC = ctypes.CDLL(None, use_errno=True)
_RENAMEAT2 = getattr(_LIBC, "renameat2", None)
if _RENAMEAT2 is not None:
    _RENAMEAT2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    _RENAMEAT2.restype = ctypes.c_int


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
    source_descriptor: int
    destination_name: str


@dataclass(frozen=True)
class ImportPreflightPaths:
    """Descriptor-anchored paths retained across controller preparation."""

    source_root: Path
    destination_root: Path
    destination_run_set: Path


@contextmanager
def preflight_import_paths(
    source_root: Path,
    destination_root: Path,
    run_set: RunSet,
) -> Iterator[ImportPreflightPaths]:
    """Validate import topology without writes, then anchor preparation paths.

    Args:
        source_root: Existing artifact root containing immutable source attempts.
        destination_root: Independent artifact root to prepare.
        run_set: Run-set subset that will be imported.

    Yields:
        Descriptor paths that remain attached to the validated directories.

    Raises:
        ValueError: If roots overlap or any existing root/run-set entry is unsafe.
    """
    selected_run_set = RunSet(run_set)
    source_path = _absolute_path(source_root)
    destination_path = _absolute_path(destination_root)
    _resolve_independent_roots(source_path, destination_path)
    _validate_root_entry(source_path, "source root")
    destination_ancestor, missing_destination_parts = _nearest_existing_ancestor(destination_path)

    with ExitStack() as stack:
        source_descriptor = stack.enter_context(_open_directory(source_path, "source root"))
        source_run_set_descriptor = stack.enter_context(
            _open_directory_at(source_descriptor, selected_run_set.value, "source run-set directory")
        )
        destination_ancestor_descriptor = stack.enter_context(
            _open_directory(destination_ancestor, "destination ancestor")
        )
        anchored_destination = (
            _descriptor_path(destination_ancestor_descriptor).resolve().joinpath(*missing_destination_parts)
        )
        _reject_overlapping_roots(_descriptor_path(source_descriptor).resolve(), anchored_destination)

        # All checks above are read-only. Create only a destination already proven
        # independent, relative to the retained no-follow ancestor descriptor.
        destination_descriptor = destination_ancestor_descriptor
        for part in missing_destination_parts:
            with suppress(FileExistsError):
                os.mkdir(part, mode=0o755, dir_fd=destination_descriptor)
            destination_descriptor = stack.enter_context(
                _open_directory_at(destination_descriptor, part, "destination root")
            )

        destination_run_set_metadata = _directory_entry_metadata(
            destination_descriptor,
            selected_run_set.value,
            "destination run-set directory",
        )
        if destination_run_set_metadata is None:
            os.mkdir(selected_run_set.value, mode=0o755, dir_fd=destination_descriptor)
        destination_run_set_descriptor = stack.enter_context(
            _open_directory_at(destination_descriptor, selected_run_set.value, "destination run-set directory")
        )
        source_run_set_metadata = os.fstat(source_run_set_descriptor)
        destination_run_set_metadata = os.fstat(destination_run_set_descriptor)
        if (source_run_set_metadata.st_dev, source_run_set_metadata.st_ino) == (
            destination_run_set_metadata.st_dev,
            destination_run_set_metadata.st_ino,
        ):
            raise ValueError("source and destination run-set directories overlap")

        yield ImportPreflightPaths(
            source_root=_descriptor_path(source_descriptor),
            destination_root=_descriptor_path(destination_descriptor),
            destination_run_set=_descriptor_path(destination_run_set_descriptor),
        )


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
    _validate_run_set_directory(source, selected_run_set, "source")
    _validate_run_set_directory(destination, selected_run_set, "destination")

    with ExitStack() as stack:
        source_descriptor = stack.enter_context(_open_directory(source_root, "source root"))
        destination_descriptor = stack.enter_context(_open_directory(destination_root, "destination root"))
        source_run_set_descriptor = stack.enter_context(
            _open_directory_at(source_descriptor, selected_run_set.value, "source run-set directory")
        )
        destination_run_set_descriptor = stack.enter_context(
            _open_directory_at(destination_descriptor, selected_run_set.value, "destination run-set directory")
        )
        destination_run_set_view = _descriptor_path(destination_run_set_descriptor)

        source_manifest, source_manifest_sha256 = _read_manifest_snapshot(
            source_run_set_descriptor,
            "source benchmark manifest",
        )
        destination_manifest, destination_manifest_sha256 = _read_manifest_snapshot(
            destination_run_set_descriptor,
            "destination benchmark manifest",
        )
        _validate_manifests(source_manifest, destination_manifest, selected_run_set)

        source_expansion = resolve_manifest_expansion(source_manifest, source)
        destination_expansion = resolve_manifest_expansion(destination_manifest, destination)
        imports = _resolve_attempt_imports(
            source_run_set_descriptor,
            source_expansion.attempts,
            destination_expansion.attempts,
            destination_manifest,
            selected_run_set,
            stack,
        )
        source_digest, source_file_count = _aggregate_digest(
            tuple((item.source_attempt.identity, _descriptor_path(item.source_descriptor)) for item in imports)
        )
        audit = ImportAudit(
            source_root=source,
            destination_root=destination,
            run_set=selected_run_set,
            source_manifest_sha256=source_manifest_sha256,
            destination_manifest_sha256=destination_manifest_sha256,
            imported_attempt_count=len(imports),
            imported_file_count=source_file_count,
            source_aggregate_sha256=source_digest,
            destination_aggregate_sha256=source_digest,
        )

        audit_path = destination_run_set_view / "import_audit.json"
        if _path_exists(audit_path):
            _validate_regular_file(audit_path, "destination import audit")
            with _import_lock(destination_run_set_view):
                return _validate_existing_import(
                    audit_path,
                    audit,
                    imports,
                    destination_manifest,
                    destination_run_set_view,
                )

        destinations = _destination_attempt_paths(destination_run_set_view, imports)
        staging_root = Path(tempfile.mkdtemp(prefix=".import-staging-", dir=destination_run_set_view))
        try:
            staging_descriptor = stack.enter_context(
                _open_directory_at(
                    destination_run_set_descriptor,
                    staging_root.name,
                    "destination import staging directory",
                )
            )
            staged = _stage_attempts(staging_descriptor, imports, destination_manifest)
            staged_digest, staged_file_count = _aggregate_digest(
                tuple((item.destination_attempt.identity, path) for item, path in staged)
            )
            if staged_digest != source_digest or staged_file_count != source_file_count:
                raise RuntimeError("staged import aggregate does not match the source aggregate")

            with _import_lock(destination_run_set_view):
                if _path_exists(audit_path):
                    _validate_regular_file(audit_path, "destination import audit")
                    return _validate_existing_import(
                        audit_path,
                        audit,
                        imports,
                        destination_manifest,
                        destination_run_set_view,
                    )
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
            if _path_exists(staging_root):
                shutil.rmtree(staging_root)


def _resolve_independent_roots(source_root: Path, destination_root: Path) -> tuple[Path, Path]:
    """Resolve roots and reject equality or containment in either direction."""
    source = source_root.resolve()
    destination = destination_root.resolve()
    _reject_overlapping_roots(source, destination)
    return source, destination


def _reject_overlapping_roots(source: Path, destination: Path) -> None:
    """Reject equality or containment in either direction."""
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("source and destination roots overlap")


def _absolute_path(path: Path) -> Path:
    """Return an absolute lexical path without following symlinks."""
    return Path(os.path.abspath(path))


def _validate_root_entry(path: Path, name: str) -> None:
    """Require an existing non-symlink directory root."""
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"{name} must be an available, symlink-free directory: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{name} must be an available, symlink-free directory: {path}")


def _nearest_existing_ancestor(path: Path) -> tuple[Path, tuple[str, ...]]:
    """Return a safe existing ancestor and missing direct-child names."""
    candidate = path
    missing = []
    while True:
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            if candidate == candidate.parent:
                raise ValueError(f"destination root has no available ancestor: {path}")
            missing.append(candidate.name)
            candidate = candidate.parent
            continue
        except OSError as error:
            raise ValueError(f"destination root is unavailable: {error}") from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"destination root must be below a symlink-free directory: {candidate}")
        return candidate, tuple(reversed(missing))


def _directory_entry_metadata(parent_descriptor: int, name: str, description: str) -> os.stat_result | None:
    """Inspect one direct child without following it."""
    try:
        metadata = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ValueError(f"{description} is unavailable: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{description} must be a symlink-free directory")
    return metadata


def _descriptor_path(descriptor: int) -> Path:
    """Return the Linux procfs view of one open descriptor."""
    return Path("/proc/self/fd") / str(descriptor)


def _is_descriptor_path(path: Path) -> bool:
    """Return whether ``path`` is an importer-created descriptor view."""
    return path.parent == Path("/proc/self/fd") and path.name.isdecimal()


@contextmanager
def _open_directory(path: Path, name: str) -> Iterator[int]:
    """Open and retain one no-follow directory descriptor."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if not _is_descriptor_path(path):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{name} must be an available, symlink-free directory: {error}") from error
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError(f"{name} must be a directory: {path}")
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def _open_directory_at(parent_descriptor: int, child_name: str, name: str) -> Iterator[int]:
    """Open and retain a direct no-follow child directory descriptor."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(child_name, flags, dir_fd=parent_descriptor)
    except OSError as error:
        raise ValueError(f"{name} must be an available, symlink-free directory: {error}") from error
    try:
        yield descriptor
    finally:
        os.close(descriptor)


def _read_regular_file(path: Path, name: str) -> bytes:
    """Read one regular file through a no-follow descriptor."""
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{name} is unavailable or symlinked: {error}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{name} must be a regular file: {path}")
        contents = _read_descriptor(descriptor)
    finally:
        os.close(descriptor)
    if len(contents) != metadata.st_size:
        raise ValueError(f"{name} changed while it was read: {path}")
    return contents


def _read_descriptor(descriptor: int) -> bytes:
    """Read a descriptor completely from its current zero offset."""
    chunks = []
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def _read_manifest_snapshot(run_set_descriptor: int, name: str) -> tuple[RunSetManifest, str]:
    """Parse and hash one exact manifest byte snapshot."""
    contents = _read_regular_file(_descriptor_path(run_set_descriptor) / "manifest.json", name)
    with tempfile.NamedTemporaryFile(prefix=".benchmark-manifest-", suffix=".json") as snapshot:
        snapshot.write(contents)
        snapshot.flush()
        manifest = read_manifest(Path(snapshot.name))
    return manifest, hashlib.sha256(contents).hexdigest()


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
    destination_run_set: Path,
    imports: tuple[_AttemptImport, ...],
) -> tuple[Path, ...]:
    """Resolve publication paths only when every parent is the validated run-set root."""
    return tuple(destination_run_set / item.destination_name for item in imports)


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
    source_run_set_descriptor: int,
    source_attempts: tuple[BenchmarkAttempt, ...],
    destination_attempts: tuple[BenchmarkAttempt, ...],
    destination_manifest: RunSetManifest,
    run_set: RunSet,
    stack: ExitStack,
) -> tuple[_AttemptImport, ...]:
    """Validate immutable attempt mappings and every completed source root."""
    destination_by_identity = {attempt.identity: attempt for attempt in destination_attempts}
    imports = []
    for source_attempt in sorted(source_attempts, key=lambda attempt: attempt.identity):
        destination_attempt = destination_by_identity.get(source_attempt.identity)
        if destination_attempt is None:
            raise ValueError(f"source attempt identity is absent from destination expansion: {source_attempt.identity}")
        _validate_attempt_mapping(source_attempt, destination_attempt)
        source_name = _attempt_child_name(source_attempt, run_set)
        destination_name = _attempt_child_name(destination_attempt, run_set)
        source_descriptor = stack.enter_context(
            _open_directory_at(source_run_set_descriptor, source_name, f"source attempt root {source_attempt.identity}")
        )
        source_view = _descriptor_path(source_descriptor)
        _validate_attempt_tree(source_view, "source")
        _validate_success(
            source_view / "success",
            destination_attempt,
            destination_manifest,
            location="source",
        )
        imports.append(
            _AttemptImport(
                source_attempt,
                destination_attempt,
                source_descriptor,
                destination_name,
            )
        )
    return tuple(imports)


def _attempt_child_name(attempt: BenchmarkAttempt, run_set: RunSet) -> str:
    """Return a validated direct run-set child name for one attempt."""
    relative = Path(attempt.run_directory)
    if relative.parent != Path(run_set.value) or relative.name in {"", ".", ".."}:
        raise ValueError(f"attempt path is not a direct run-set child: {attempt.run_directory}")
    return relative.name


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
    with _open_directory(root, f"{location} attempt root") as descriptor:
        _validate_tree_descriptor(descriptor, location)


def _validate_tree_descriptor(descriptor: int, location: str) -> None:
    """Validate a complete tree relative to one stable directory descriptor."""
    for name in _tree_entry_names(descriptor, location):
        with _open_tree_entry(descriptor, name, location) as (entry_descriptor, metadata):
            if stat.S_ISDIR(metadata.st_mode):
                _validate_tree_descriptor(entry_descriptor, location)


def _tree_entry_names(descriptor: int, location: str) -> tuple[str, ...]:
    """Return sorted child names from one anchored directory."""
    try:
        with os.scandir(descriptor) as entries:
            return tuple(sorted(entry.name for entry in entries))
    except OSError as error:
        raise ValueError(f"{location} attempt root cannot be traversed safely: {error}") from error


@contextmanager
def _open_tree_entry(
    parent_descriptor: int,
    child_name: str,
    location: str,
) -> Iterator[tuple[int, os.stat_result]]:
    """Open one child inode without following a symlink or reopening its name."""
    path_flag = getattr(os, "O_PATH", None)
    if path_flag is None:
        raise RuntimeError("descriptor-anchored import requires Linux O_PATH support")
    try:
        path_descriptor = os.open(
            child_name,
            path_flag | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
    except OSError as error:
        raise ValueError(f"{location} attempt root contains an unavailable entry: {child_name}: {error}") from error
    descriptor = -1
    try:
        metadata = os.fstat(path_descriptor)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{location} attempt root contains a symlink: {child_name}")
        if not stat.S_ISDIR(metadata.st_mode) and not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{location} attempt root contains a special file: {child_name}")
        flags = os.O_RDONLY | os.O_CLOEXEC
        if stat.S_ISDIR(metadata.st_mode):
            flags |= os.O_DIRECTORY
        descriptor = os.open(_descriptor_path(path_descriptor), flags)
        opened_metadata = os.fstat(descriptor)
        if (opened_metadata.st_dev, opened_metadata.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise ValueError(f"{location} attempt entry changed while it was opened: {child_name}")
        yield descriptor, opened_metadata
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(path_descriptor)


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
    staging_descriptor: int,
    imports: tuple[_AttemptImport, ...],
    destination_manifest: RunSetManifest,
) -> tuple[tuple[_AttemptImport, Path], ...]:
    """Copy complete attempt roots into one importer-owned staging directory."""
    staged = []
    for item in imports:
        staged_name = item.destination_attempt.identity
        if Path(staged_name).name != staged_name or staged_name in {"", ".", ".."}:
            raise ValueError(f"attempt identity is not a direct staging child: {staged_name}")
        _validate_attempt_tree(_descriptor_path(item.source_descriptor), "aggregate")
        _copy_attempt_tree(item.source_descriptor, staging_descriptor, staged_name)
        staged_path = _descriptor_path(staging_descriptor) / staged_name
        _validate_attempt_tree(staged_path, "staged")
        _validate_staged_success(staged_path / "success", item.destination_attempt, destination_manifest)
        staged.append((item, staged_path))
    return tuple(staged)


def _copy_attempt_tree(source_descriptor: int, staging_descriptor: int, destination_name: str) -> None:
    """Copy one anchored source tree into importer-owned staging."""
    os.mkdir(destination_name, mode=0o700, dir_fd=staging_descriptor)
    with _open_directory_at(staging_descriptor, destination_name, "staged attempt root") as destination_descriptor:
        _copy_tree_contents(source_descriptor, destination_descriptor)
        shutil.copystat(_descriptor_path(source_descriptor), _descriptor_path(destination_descriptor))


def _copy_tree_contents(source_descriptor: int, destination_descriptor: int) -> None:
    """Recursively copy regular files and directories without following source names."""
    for name in _tree_entry_names(source_descriptor, "source"):
        with _open_tree_entry(source_descriptor, name, "source") as (entry_descriptor, metadata):
            if stat.S_ISDIR(metadata.st_mode):
                os.mkdir(name, mode=0o700, dir_fd=destination_descriptor)
                with _open_directory_at(destination_descriptor, name, "staged directory") as destination_child:
                    _copy_tree_contents(entry_descriptor, destination_child)
                    shutil.copystat(_descriptor_path(entry_descriptor), _descriptor_path(destination_child))
                continue

            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
            destination_file = os.open(name, flags, 0o600, dir_fd=destination_descriptor)
            try:
                shutil.copy2(_descriptor_path(entry_descriptor), _descriptor_path(destination_file))
            finally:
                os.close(destination_file)


def _aggregate_digest(attempt_roots: tuple[tuple[str, Path], ...]) -> tuple[str, int]:
    """Hash sorted identities, relative file paths, modes, sizes, and bytes."""
    aggregate = hashlib.sha256()
    file_count = 0
    for identity, root in sorted(attempt_roots, key=lambda item: item[0]):
        _validate_attempt_tree(root, "aggregate")
        with _open_directory(root, "aggregate attempt root") as descriptor:
            file_count += _aggregate_tree(descriptor, Path(), identity, aggregate)
    return aggregate.hexdigest(), file_count


def _aggregate_tree(
    descriptor: int,
    relative_root: Path,
    identity: str,
    aggregate: Any,
) -> int:
    """Add one anchored directory subtree to an aggregate digest."""
    file_count = 0
    for name in _tree_entry_names(descriptor, "aggregate"):
        relative_path = relative_root / name
        with _open_tree_entry(descriptor, name, "aggregate") as (entry_descriptor, metadata):
            if stat.S_ISDIR(metadata.st_mode):
                file_count += _aggregate_tree(entry_descriptor, relative_path, identity, aggregate)
                continue
            contents = _read_descriptor(entry_descriptor)
            if len(contents) != metadata.st_size:
                raise RuntimeError(f"aggregate source changed while read: {relative_path.as_posix()}")
            header = {
                "identity": identity,
                "path": relative_path.as_posix(),
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
    return file_count


def _validate_existing_import(
    audit_path: Path,
    expected_audit: ImportAudit,
    imports: tuple[_AttemptImport, ...],
    destination_manifest: RunSetManifest,
    destination_run_set: Path,
) -> ImportAudit:
    """Return an identical import only after revalidating all destination bytes."""
    existing_bytes = _read_regular_file(audit_path, "destination import audit")
    if existing_bytes != _audit_bytes(expected_audit):
        raise ValueError(f"conflicting destination import audit: {audit_path}")

    with ExitStack() as stack:
        destination_roots = []
        for item in imports:
            destination_path = destination_run_set / item.destination_name
            destination_descriptor = stack.enter_context(
                _open_directory(destination_path, f"destination attempt root {item.destination_attempt.identity}")
            )
            destination_view = _descriptor_path(destination_descriptor)
            _validate_attempt_tree(destination_view, "destination")
            _validate_success(
                destination_view / "success",
                item.destination_attempt,
                destination_manifest,
                location="destination",
            )
            destination_roots.append((item.destination_attempt.identity, destination_view))
        digest, file_count = _aggregate_digest(tuple(destination_roots))
    if digest != expected_audit.destination_aggregate_sha256 or file_count != expected_audit.imported_file_count:
        raise RuntimeError("destination aggregate does not match the import audit")
    return expected_audit


def _publish_attempt_root(staged: Path, destination: Path) -> None:
    """Publish one staged attempt root atomically without replacing a conflict."""
    _rename_noreplace(staged, destination)


def _rollback_published_attempts(published: list[tuple[Path, Path]]) -> None:
    """Move newly published roots back to importer-owned staging."""
    rollback_errors = []
    for destination, staged in reversed(published):
        try:
            _rename_noreplace(destination, staged)
        except OSError as error:
            rollback_errors.append(error)
    if rollback_errors:
        raise RuntimeError("failed to roll back a partially published import") from rollback_errors[0]


def _write_audit_atomic(path: Path, audit: ImportAudit) -> None:
    """Write, fsync, and publish a deterministic audit without replacement."""
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(_audit_bytes(audit))
            file.flush()
            os.fsync(file.fileno())
        _rename_noreplace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename on Linux while refusing to replace any destination."""
    if _RENAMEAT2 is None:
        raise RuntimeError("atomic no-replace publication requires Linux renameat2 support")
    source_directory, source_name = _rename_operand(source)
    destination_directory, destination_name = _rename_operand(destination)
    result = _RENAMEAT2(
        source_directory,
        source_name,
        destination_directory,
        destination_name,
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
        raise RuntimeError("atomic no-replace publication is unavailable on this Linux filesystem")
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _rename_operand(path: Path) -> tuple[int, bytes]:
    """Return a dirfd and leaf name for an anchored rename operand."""
    parts = path.parts
    if len(parts) == 6 and parts[:4] == ("/", "proc", "self", "fd") and parts[4].isdecimal():
        return int(parts[4]), os.fsencode(parts[5])
    return _AT_FDCWD, os.fsencode(path)


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
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError(f"destination import lock must be a regular file: {lock_path}")
    with os.fdopen(descriptor, "r+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
