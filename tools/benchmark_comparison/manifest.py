# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Typed, immutable provenance manifest for one benchmark run set."""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .models import ExecutionProvenance, RunSet

_MANIFEST_FIELDS = {
    "schema_version",
    "run_set",
    "phase",
    "provenance",
    "host",
    "lab2",
    "lab3",
    "cpu_power_profile",
}
_PROVENANCE_FIELDS = {"lab2_sha", "lab3_sha", "lab2_image_id", "uv_lock_sha256"}
_HOST_FIELDS = {
    "hostname",
    "os",
    "cpu_model",
    "logical_cpu_count",
    "gpu_model",
    "gpu_driver",
    "cuda_version",
}
_SOFTWARE_FIELDS = {"isaac_lab", "isaac_sim", "python", "pytorch", "rsl_rl"}


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

    def to_json(self) -> dict[str, object]:
        """Return the strict JSON representation."""
        return {
            "hostname": self.hostname,
            "os": self.os,
            "cpu_model": self.cpu_model,
            "logical_cpu_count": self.logical_cpu_count,
            "gpu_model": self.gpu_model,
            "gpu_driver": self.gpu_driver,
            "cuda_version": self.cuda_version,
        }


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

    def software(self, version: str) -> SoftwareIdentity:
        """Return the software identity for ``version``."""
        if version == "lab2":
            return self.lab2
        if version == "lab3":
            return self.lab3
        raise ValueError(f"unknown benchmark version: {version}")

    def to_json(self) -> dict[str, object]:
        """Return the deterministic strict JSON representation."""
        return {
            "schema_version": self.schema_version,
            "run_set": self.run_set.value,
            "phase": self.phase,
            "provenance": self.provenance.to_json(),
            "host": self.host.to_json(),
            "lab2": self.lab2.to_json(),
            "lab3": self.lab3.to_json(),
            "cpu_power_profile": self.cpu_power_profile,
        }


def manifest_path(artifact_root: Path, run_set: RunSet | str) -> Path:
    """Return the canonical manifest path for ``run_set``."""
    return artifact_root / RunSet(run_set).value / "manifest.json"


def read_manifest(path: Path) -> RunSetManifest:
    """Read and strictly validate a benchmark manifest."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read benchmark manifest: {error}") from error
    root = _mapping(value, "benchmark manifest", _MANIFEST_FIELDS)
    if root["schema_version"] != "1.0":
        raise ValueError("manifest schema_version must be '1.0'")
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

    host_value = _mapping(root["host"], "host", _HOST_FIELDS)
    count = host_value["logical_cpu_count"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("host.logical_cpu_count must be a positive integer")
    cuda = host_value["cuda_version"]
    if cuda is not None:
        cuda = _text(cuda, "host.cuda_version")
    host = HostIdentity(
        hostname=_text(host_value["hostname"], "host.hostname"),
        os=_text(host_value["os"], "host.os"),
        cpu_model=_text(host_value["cpu_model"], "host.cpu_model"),
        logical_cpu_count=count,
        gpu_model=_text(host_value["gpu_model"], "host.gpu_model"),
        gpu_driver=_text(host_value["gpu_driver"], "host.gpu_driver"),
        cuda_version=cuda,
    )
    power_profile = root["cpu_power_profile"]
    if power_profile is not None:
        power_profile = _text(power_profile, "cpu_power_profile")
    return RunSetManifest(
        schema_version="1.0",
        run_set=run_set,
        phase=phase,
        provenance=provenance,
        host=host,
        lab2=_software(root["lab2"], "lab2"),
        lab3=_software(root["lab3"], "lab3"),
        cpu_power_profile=power_profile,
    )


def validate_manifest(manifest: RunSetManifest) -> RunSetManifest:
    """Return a strictly validated canonical copy of ``manifest``."""
    return _manifest_from_mapping(manifest.to_json())


def write_manifest(path: Path, manifest: RunSetManifest) -> Path:
    """Publish ``manifest`` once, allowing only byte-identical rewrites."""
    validated = validate_manifest(manifest)
    contents = json.dumps(validated.to_json(), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            if path.read_bytes() != contents.encode():
                raise ValueError(f"refusing to replace different benchmark manifest: {path}")
            return path
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                file.write(contents)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def _manifest_from_mapping(value: Mapping[str, object]) -> RunSetManifest:
    descriptor, name = tempfile.mkstemp()
    path = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(value, file)
        return read_manifest(path)
    finally:
        path.unlink(missing_ok=True)


def _mapping(value: object, name: str, fields: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} must contain exactly: {', '.join(sorted(fields))}")
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


def _image_id(value: object) -> str:
    text = _text(value, "provenance.lab2_image_id")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", text) is None:
        raise ValueError("provenance.lab2_image_id must be an exact sha256 image ID")
    return text
