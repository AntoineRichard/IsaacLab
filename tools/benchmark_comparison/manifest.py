# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Typed, immutable provenance manifest for one benchmark run set."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .models import (
    BenchmarkAttempt,
    BenchmarkMode,
    BenchmarkPair,
    Bound,
    BoundUnit,
    ExecutionProvenance,
    MatrixExpansion,
    RunSet,
    Version,
)

_MANIFEST_V1_FIELDS = {
    "schema_version",
    "run_set",
    "phase",
    "provenance",
    "host",
    "lab2",
    "lab3",
    "cpu_power_profile",
}
_MANIFEST_V2_FIELDS = {*_MANIFEST_V1_FIELDS, "run_set_identity"}
_PROVENANCE_FIELDS = {"lab2_sha", "lab3_sha", "lab2_image_id", "uv_lock_sha256"}
_HOST_V1_FIELDS = {
    "hostname",
    "os",
    "cpu_model",
    "logical_cpu_count",
    "gpu_model",
    "gpu_driver",
    "cuda_version",
}
_HOST_V2_FIELDS = {*_HOST_V1_FIELDS, "gpu_index", "gpu_uuid"}
_SOFTWARE_FIELDS = {"isaac_lab", "isaac_sim", "python", "pytorch", "rsl_rl"}
_RUN_SET_IDENTITY_FIELDS = {"sha256", "attempts"}
_ATTEMPT_FIELDS = {
    "identity",
    "run_directory",
    "logical_pair_identity",
    "logical_task",
    "concrete_task",
    "mode",
    "bound",
    "seed",
    "repeat_index",
    "num_envs",
    "framework",
    "enable_cameras",
    "extra_presets",
    "pair_order",
    "version",
    "version_order",
    "attempt_order",
}
_MODE_FIELDS = {"id", "framework", "final_bound", "canary_bound"}
_BOUND_FIELDS = {"value", "unit"}


@dataclass(frozen=True)
class HostIdentity:
    """Hardware and operating-system identity shared by both benchmark versions."""

    hostname: str
    os: str
    cpu_model: str
    logical_cpu_count: int
    gpu_model: str
    gpu_driver: str
    cuda_version: str | None
    gpu_index: int | None = None
    gpu_uuid: str | None = None

    def to_json(self, *, include_selected_gpu: bool = False) -> dict[str, object]:
        """Return the strict JSON representation."""
        result = {
            "hostname": self.hostname,
            "os": self.os,
            "cpu_model": self.cpu_model,
            "logical_cpu_count": self.logical_cpu_count,
            "gpu_model": self.gpu_model,
            "gpu_driver": self.gpu_driver,
            "cuda_version": self.cuda_version,
        }
        if include_selected_gpu:
            result.update(gpu_index=self.gpu_index, gpu_uuid=self.gpu_uuid)
        return result


@dataclass(frozen=True)
class SoftwareIdentity:
    """Software versions for one Isaac Lab execution environment."""

    isaac_lab: str
    isaac_sim: str
    python: str
    pytorch: str
    rsl_rl: str

    def to_json(self) -> dict[str, str]:
        """Return the strict JSON representation."""
        return {
            "isaac_lab": self.isaac_lab,
            "isaac_sim": self.isaac_sim,
            "python": self.python,
            "pytorch": self.pytorch,
            "rsl_rl": self.rsl_rl,
        }


@dataclass(frozen=True)
class RunSetManifest:
    """Complete self-contained identity contract for one benchmark run set."""

    schema_version: str
    run_set: RunSet
    phase: str
    provenance: ExecutionProvenance
    host: HostIdentity
    lab2: SoftwareIdentity
    lab3: SoftwareIdentity
    cpu_power_profile: str | None = None
    expansion: MatrixExpansion | None = None

    def software(self, version: str) -> SoftwareIdentity:
        """Return the software identity for ``version``."""
        if version == "lab2":
            return self.lab2
        if version == "lab3":
            return self.lab3
        raise ValueError(f"unknown benchmark version: {version}")

    def to_json(self) -> dict[str, object]:
        """Return the deterministic strict JSON representation."""
        result = {
            "schema_version": self.schema_version,
            "run_set": self.run_set.value,
            "phase": self.phase,
            "provenance": self.provenance.to_json(),
            "host": self.host.to_json(include_selected_gpu=self.schema_version == "2.0"),
            "lab2": self.lab2.to_json(),
            "lab3": self.lab3.to_json(),
            "cpu_power_profile": self.cpu_power_profile,
        }
        if self.schema_version == "2.0":
            if self.expansion is None:
                raise ValueError("manifest schema 2.0 requires an exact run-set expansion")
            result["run_set_identity"] = _expansion_document(self.expansion)
        return result


def manifest_path(artifact_root: Path, run_set: RunSet | str) -> Path:
    """Return the canonical manifest path for ``run_set``."""
    return artifact_root / RunSet(run_set).value / "manifest.json"


def resolve_manifest_expansion(manifest: RunSetManifest, artifact_root: Path) -> MatrixExpansion:
    """Resolve only the exact expansion attested by a manifest and its artifacts.

    Schema 2.0 embeds the canonical expansion. Schema 1.0 is accepted only
    when the artifact directory identities exactly match the retained
    six-task matrix; partial or unknown identity sets are ambiguous.
    """
    validated = validate_manifest(manifest)
    if validated.schema_version == "2.0":
        if validated.expansion is None:
            raise ValueError("manifest schema 2.0 is missing its run-set identity")
        return validated.expansion

    from .matrix import expand_legacy_schema_1_matrix

    expansion = expand_legacy_schema_1_matrix(validated.run_set)
    expected = {Path(attempt.run_directory).name for attempt in expansion.attempts}
    run_set_root = artifact_root.resolve() / validated.run_set.value
    try:
        observed = {
            path.name
            for path in run_set_root.iterdir()
            if path.is_dir() and path.name.startswith(f"{validated.run_set.value}--")
        }
    except OSError as error:
        raise ValueError(f"schema 1.0 artifact identities are ambiguous: {error}") from error
    if observed != expected:
        raise ValueError(
            "schema 1.0 artifact identities are ambiguous: expected the exact retained six-task "
            f"set ({len(expected)} attempts), observed {len(observed)}"
        )
    return expansion


def read_manifest(path: Path) -> RunSetManifest:
    """Read and strictly validate a benchmark manifest."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read benchmark manifest: {error}") from error
    if not isinstance(value, Mapping):
        raise ValueError("benchmark manifest must contain an object")
    schema_version = value.get("schema_version")
    if schema_version == "1.0":
        root = _mapping(value, "benchmark manifest", _MANIFEST_V1_FIELDS)
    elif schema_version == "2.0":
        root = _mapping(value, "benchmark manifest", _MANIFEST_V2_FIELDS)
    else:
        raise ValueError("manifest schema_version must be '1.0' or '2.0'")
    phase = _text(root["phase"], "phase")
    try:
        run_set = RunSet(root["run_set"])
    except (TypeError, ValueError) as error:
        raise ValueError("manifest run_set must be 'canary' or 'final'") from error

    provenance_value = _mapping(root["provenance"], "provenance", _PROVENANCE_FIELDS)
    provenance = ExecutionProvenance(
        lab2_sha=_sha(provenance_value["lab2_sha"], "lab2_sha", lengths=(40, 64)),
        lab3_sha=_sha(provenance_value["lab3_sha"], "lab3_sha", lengths=(40, 64)),
        lab2_image_id=_image_id(provenance_value["lab2_image_id"]),
        uv_lock_sha256=_sha(provenance_value["uv_lock_sha256"], "uv_lock_sha256", lengths=(64,)),
    )

    host_fields = _HOST_V2_FIELDS if schema_version == "2.0" else _HOST_V1_FIELDS
    host_value = _mapping(root["host"], "host", host_fields)
    count = host_value["logical_cpu_count"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("host.logical_cpu_count must be a positive integer")
    cuda = host_value["cuda_version"]
    if cuda is not None:
        cuda = _text(cuda, "host.cuda_version")
    gpu_index = None
    gpu_uuid = None
    if schema_version == "2.0":
        gpu_index = _integer(host_value["gpu_index"], "host.gpu_index", minimum=0)
        if gpu_index != 0:
            raise ValueError("host.gpu_index must select physical GPU 0")
        gpu_uuid = _text(host_value["gpu_uuid"], "host.gpu_uuid")
        if re.fullmatch(r"GPU-[0-9A-Za-z-]+", gpu_uuid) is None:
            raise ValueError("host.gpu_uuid must be an NVIDIA GPU UUID")
    host = HostIdentity(
        hostname=_text(host_value["hostname"], "host.hostname"),
        os=_text(host_value["os"], "host.os"),
        cpu_model=_text(host_value["cpu_model"], "host.cpu_model"),
        logical_cpu_count=count,
        gpu_model=_text(host_value["gpu_model"], "host.gpu_model"),
        gpu_driver=_text(host_value["gpu_driver"], "host.gpu_driver"),
        cuda_version=cuda,
        gpu_index=gpu_index,
        gpu_uuid=gpu_uuid,
    )
    power_profile = root["cpu_power_profile"]
    if power_profile is not None:
        power_profile = _text(power_profile, "cpu_power_profile")
    return RunSetManifest(
        schema_version=schema_version,
        run_set=run_set,
        phase=phase,
        provenance=provenance,
        host=host,
        lab2=_software(root["lab2"], "lab2"),
        lab3=_software(root["lab3"], "lab3"),
        cpu_power_profile=power_profile,
        expansion=(_expansion_from_document(root["run_set_identity"], run_set) if schema_version == "2.0" else None),
    )


def validate_manifest(manifest: RunSetManifest) -> RunSetManifest:
    """Return a strictly validated canonical copy of ``manifest``."""
    return _manifest_from_mapping(manifest.to_json())


def write_manifest(path: Path, manifest: RunSetManifest) -> Path:
    """Publish ``manifest`` once, allowing only byte-identical rewrites."""
    validated = validate_manifest(manifest)
    contents = json.dumps(validated.to_json(), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    parent_descriptor = _open_manifest_parent(path.parent)
    parent_view = Path("/proc/self/fd") / str(parent_descriptor)
    try:
        lock_descriptor = _open_manifest_leaf(
            parent_descriptor,
            path.with_suffix(path.suffix + ".lock").name,
            os.O_NOFOLLOW | os.O_CLOEXEC | os.O_CREAT | os.O_RDWR,
            "benchmark manifest lock",
        )
        with os.fdopen(lock_descriptor, "r+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            existing_descriptor = _open_manifest_leaf(
                parent_descriptor,
                path.name,
                os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
                "existing benchmark manifest",
                allow_missing=True,
            )
            if existing_descriptor is not None:
                with os.fdopen(existing_descriptor, "rb") as existing:
                    if existing.read() != contents.encode():
                        raise ValueError(f"refusing to replace different benchmark manifest: {path}")
                return path
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent_view)
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                    file.write(contents)
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temporary, parent_view / path.name)
            finally:
                temporary.unlink(missing_ok=True)
    finally:
        os.close(parent_descriptor)
    return path


def _open_manifest_parent(path: Path) -> int:
    """Open the manifest parent without following its final directory entry."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if not _is_descriptor_path(path):
        flags |= os.O_NOFOLLOW
    try:
        return os.open(path, flags)
    except OSError as error:
        raise ValueError(f"benchmark manifest parent must be an available, symlink-free directory: {error}") from error


def _open_manifest_leaf(
    parent_descriptor: int,
    name: str,
    flags: int,
    description: str,
    *,
    allow_missing: bool = False,
) -> int | None:
    """Open one no-follow regular-file leaf relative to a retained parent."""
    try:
        descriptor = os.open(name, flags, 0o666, dir_fd=parent_descriptor)
    except FileNotFoundError:
        if allow_missing:
            return None
        raise
    except OSError as error:
        raise ValueError(f"{description} must be an available, symlink-free regular file: {error}") from error
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError(f"{description} must be a regular file")
    return descriptor


def _is_descriptor_path(path: Path) -> bool:
    """Return whether ``path`` is a direct Linux procfs descriptor view."""
    return path.parent == Path("/proc/self/fd") and path.name.isdecimal()


def _manifest_from_mapping(value: Mapping[str, object]) -> RunSetManifest:
    descriptor, name = tempfile.mkstemp()
    path = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(value, file)
        return read_manifest(path)
    finally:
        path.unlink(missing_ok=True)


def _expansion_document(expansion: MatrixExpansion) -> dict[str, object]:
    """Return a canonical, self-validating run-set expansion snapshot."""
    attempts = [_attempt_document(attempt) for attempt in expansion.attempts]
    document = {"sha256": _attempts_digest(attempts), "attempts": attempts}
    if _expansion_from_document(document, expansion.run_set) != expansion:
        raise ValueError("run-set expansion cannot be represented canonically")
    return document


def _attempt_document(attempt: BenchmarkAttempt) -> dict[str, object]:
    """Return the complete canonical identity of one expanded attempt."""
    return {
        "identity": attempt.identity,
        "run_directory": attempt.run_directory,
        "logical_pair_identity": attempt.logical_pair_identity,
        "logical_task": attempt.logical_task,
        "concrete_task": attempt.concrete_task,
        "mode": {
            "id": attempt.mode.id,
            "framework": attempt.mode.framework,
            "final_bound": _bound_document(attempt.mode.final_bound),
            "canary_bound": _bound_document(attempt.mode.canary_bound),
        },
        "bound": _bound_document(attempt.bound),
        "seed": attempt.seed,
        "repeat_index": attempt.repeat_index,
        "num_envs": attempt.num_envs,
        "framework": attempt.framework,
        "enable_cameras": attempt.enable_cameras,
        "extra_presets": list(attempt.extra_presets),
        "pair_order": attempt.pair_order,
        "version": attempt.version.value,
        "version_order": attempt.version_order,
        "attempt_order": attempt.attempt_order,
    }


def _bound_document(bound: Bound) -> dict[str, object]:
    """Return one canonical benchmark bound."""
    return {"value": bound.value, "unit": bound.unit.value}


def _attempts_digest(attempts: object) -> str:
    """Hash the canonical attempt list that defines a run set."""
    encoded = json.dumps(
        attempts,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _expansion_from_document(value: object, run_set: RunSet) -> MatrixExpansion:
    """Parse and structurally validate an exact run-set expansion snapshot."""
    root = _mapping(value, "run_set_identity", _RUN_SET_IDENTITY_FIELDS)
    attempt_values = _list(root["attempts"], "run_set_identity.attempts")
    digest = _digest(root["sha256"], "run_set_identity.sha256")
    if digest != _attempts_digest(attempt_values):
        raise ValueError("run_set_identity.sha256 does not match its canonical attempts")
    attempts = tuple(
        _attempt_from_document(attempt_value, run_set, index) for index, attempt_value in enumerate(attempt_values)
    )
    pairs = _pairs_from_attempts(attempts, run_set)
    return MatrixExpansion(run_set=run_set, pairs=pairs, attempts=attempts)


def _attempt_from_document(value: object, run_set: RunSet, index: int) -> BenchmarkAttempt:
    """Parse one strictly typed attempt from a run-set identity snapshot."""
    attempt = _mapping(value, f"run_set_identity.attempts[{index}]", _ATTEMPT_FIELDS)
    mode_value = _mapping(attempt["mode"], f"run_set_identity.attempts[{index}].mode", _MODE_FIELDS)
    mode = BenchmarkMode(
        id=_text(mode_value["id"], "attempt.mode.id"),
        framework=_text(mode_value["framework"], "attempt.mode.framework"),
        final_bound=_bound(mode_value["final_bound"], "attempt.mode.final_bound"),
        canary_bound=_bound(mode_value["canary_bound"], "attempt.mode.canary_bound"),
    )
    try:
        version = Version(attempt["version"])
    except (TypeError, ValueError) as error:
        raise ValueError("attempt.version must be 'lab2' or 'lab3'") from error
    extra_presets = tuple(
        _text(item, "attempt.extra_presets") for item in _list(attempt["extra_presets"], "attempt.extra_presets")
    )
    if len(extra_presets) != len(set(extra_presets)):
        raise ValueError("attempt.extra_presets must not contain duplicates")
    enable_cameras = attempt["enable_cameras"]
    if not isinstance(enable_cameras, bool):
        raise ValueError("attempt.enable_cameras must be a boolean")
    parsed = BenchmarkAttempt(
        identity=_text(attempt["identity"], "attempt.identity"),
        run_directory=_text(attempt["run_directory"], "attempt.run_directory"),
        logical_pair_identity=_text(attempt["logical_pair_identity"], "attempt.logical_pair_identity"),
        run_set=run_set,
        logical_task=_text(attempt["logical_task"], "attempt.logical_task"),
        concrete_task=_text(attempt["concrete_task"], "attempt.concrete_task"),
        mode=mode,
        bound=_bound(attempt["bound"], "attempt.bound"),
        seed=_integer(attempt["seed"], "attempt.seed"),
        repeat_index=_integer(attempt["repeat_index"], "attempt.repeat_index", minimum=0),
        num_envs=_integer(attempt["num_envs"], "attempt.num_envs", minimum=1),
        framework=_text(attempt["framework"], "attempt.framework"),
        enable_cameras=enable_cameras,
        extra_presets=extra_presets,
        pair_order=_integer(attempt["pair_order"], "attempt.pair_order", minimum=0),
        version=version,
        version_order=_integer(attempt["version_order"], "attempt.version_order", minimum=0),
        attempt_order=_integer(attempt["attempt_order"], "attempt.attempt_order", minimum=0),
    )
    _validate_attempt_snapshot(parsed, index)
    return parsed


def _validate_attempt_snapshot(attempt: BenchmarkAttempt, index: int) -> None:
    """Reject snapshot fields that disagree with their deterministic identity."""
    if attempt.attempt_order != index:
        raise ValueError("attempt.attempt_order must be contiguous and match list order")
    if attempt.framework != attempt.mode.framework:
        raise ValueError("attempt.framework must match attempt.mode.framework")
    if attempt.bound != attempt.mode.bound_for(attempt.run_set):
        raise ValueError("attempt.bound must match the selected run-set mode bound")
    pair_identity = (
        f"{attempt.run_set.value}--{attempt.logical_task}--{attempt.mode.id}"
        f"--{attempt.bound.unit.value}-{attempt.bound.value}--seed-{attempt.seed}"
        f"--repeat-{attempt.repeat_index}--envs-{attempt.num_envs}--{attempt.framework}"
    )
    if attempt.logical_pair_identity != pair_identity:
        raise ValueError("attempt.logical_pair_identity does not match its fields")
    identity = f"{pair_identity}--{attempt.version.value}--version-order-{attempt.version_order}"
    if attempt.identity != identity:
        raise ValueError("attempt.identity does not match its fields")
    if attempt.run_directory != f"{attempt.run_set.value}/{identity}":
        raise ValueError("attempt.run_directory does not match its identity")


def _pairs_from_attempts(attempts: tuple[BenchmarkAttempt, ...], run_set: RunSet) -> tuple[BenchmarkPair, ...]:
    """Reconstruct and validate paired identities from ordered attempts."""
    if not attempts:
        raise ValueError("run_set_identity.attempts must not be empty")
    if len({attempt.identity for attempt in attempts}) != len(attempts):
        raise ValueError("run_set_identity contains duplicate attempt identities")
    if len({attempt.run_directory for attempt in attempts}) != len(attempts):
        raise ValueError("run_set_identity contains duplicate run directories")
    pair_orders = {attempt.pair_order for attempt in attempts}
    if pair_orders != set(range(len(pair_orders))):
        raise ValueError("attempt.pair_order must be contiguous")

    pairs: list[BenchmarkPair] = []
    for pair_order in range(len(pair_orders)):
        pair_attempts = tuple(attempt for attempt in attempts if attempt.pair_order == pair_order)
        if len(pair_attempts) != 2:
            raise ValueError("each logical pair must contain exactly two attempts")
        ordered = tuple(sorted(pair_attempts, key=lambda attempt: attempt.version_order))
        if tuple(attempt.version_order for attempt in ordered) != (0, 1):
            raise ValueError("pair version_order values must be 0 and 1")
        if {attempt.version for attempt in ordered} != {Version.LAB2, Version.LAB3}:
            raise ValueError("each logical pair must contain Lab 2 and Lab 3")
        first = ordered[0]
        shared = (
            "logical_pair_identity",
            "logical_task",
            "mode",
            "bound",
            "seed",
            "repeat_index",
            "num_envs",
            "framework",
            "pair_order",
        )
        if any(getattr(attempt, field) != getattr(first, field) for attempt in ordered[1:] for field in shared):
            raise ValueError("paired attempts contain mismatched shared fields")
        if tuple(attempt.attempt_order for attempt in ordered) != (pair_order * 2, pair_order * 2 + 1):
            raise ValueError("attempt_order must preserve pair and version order")
        pairs.append(
            BenchmarkPair(
                identity=first.logical_pair_identity,
                run_set=run_set,
                logical_task=first.logical_task,
                mode=first.mode,
                bound=first.bound,
                seed=first.seed,
                repeat_index=first.repeat_index,
                pair_order=pair_order,
                attempts=ordered,
            )
        )
    return tuple(pairs)


def _bound(value: object, name: str) -> Bound:
    """Parse one positive run-set bound."""
    fields = _mapping(value, name, _BOUND_FIELDS)
    try:
        unit = BoundUnit(fields["unit"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}.unit must be 'steps' or 'iterations'") from error
    return Bound(value=_integer(fields["value"], f"{name}.value", minimum=1), unit=unit)


def _mapping(value: object, name: str, fields: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} must contain exactly: {', '.join(sorted(fields))}")
    return value


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must contain a list")
    return value


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or (minimum is not None and value < minimum):
        qualifier = f" greater than or equal to {minimum}" if minimum is not None else ""
        raise ValueError(f"{name} must be an integer{qualifier}")
    return value


def _software(value: object, name: str) -> SoftwareIdentity:
    fields = _mapping(value, name, _SOFTWARE_FIELDS)
    return SoftwareIdentity(**{field: _text(fields[field], f"{name}.{field}") for field in _SOFTWARE_FIELDS})


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _sha(value: object, name: str, *, lengths: tuple[int, ...]) -> str:
    text = _text(value, f"provenance.{name}")
    if len(text) not in lengths or re.fullmatch(r"[0-9a-f]+", text) is None:
        raise ValueError(f"provenance.{name} must be a lowercase hexadecimal digest")
    return text


def _digest(value: object, name: str) -> str:
    text = _text(value, name)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _image_id(value: object) -> str:
    text = _text(value, "provenance.lab2_image_id")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", text) is None:
        raise ValueError("provenance.lab2_image_id must be an exact sha256 image ID")
    return text
