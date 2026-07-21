# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Atomic manifest persistence and resume matching."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .models import (
    FailureCategory,
    Phase,
    RetryLineage,
    Revisions,
    RunIdentity,
    RunManifest,
    TaskName,
    TerminalState,
    Variant,
)


def stable_run_id(identity: RunIdentity) -> str:
    """Build a deterministic filesystem-safe identifier for a run."""
    return (
        f"{identity.phase.value}__{identity.task.value}__{identity.variant.value}__seed{identity.seed}"
        f"__env{identity.num_envs}__iter{identity.max_iterations}"
    )


def command_hash(command: tuple[str, ...] | list[str]) -> str:
    """Hash a subprocess argument vector using canonical JSON."""
    encoded = json.dumps(list(command), ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_artifact_path(path: Path, root: Path) -> str:
    """Return a POSIX artifact path relative to its approved raw root."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"artifact {path} is outside artifact root {root}") from error


def _manifest_data(manifest: RunManifest) -> dict[str, Any]:
    return asdict(manifest)


def write_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    """Atomically write a JSON document after flushing it to stable storage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_manifest(path: Path, manifest: RunManifest) -> None:
    """Atomically write a manifest after flushing it to stable storage."""
    write_json_atomic(path, _manifest_data(manifest))


def read_manifest(path: Path) -> RunManifest:
    """Read a typed manifest from its canonical JSON representation."""
    data = json.loads(path.read_text(encoding="utf-8"))
    identity_data = data["identity"]
    identity = RunIdentity(
        task=TaskName(identity_data["task"]),
        variant=Variant(identity_data["variant"]),
        seed=int(identity_data["seed"]),
        phase=Phase(identity_data["phase"]),
        num_envs=int(identity_data["num_envs"]),
        max_iterations=int(identity_data["max_iterations"]),
    )
    retry_data = data["retry"]
    return RunManifest(
        run_id=data["run_id"],
        identity=identity,
        command=tuple(data["command"]),
        command_hash=data["command_hash"],
        revisions=Revisions(**data["revisions"]),
        schema_version=data["schema_version"],
        artifact_root=data["artifact_root"],
        isaaclab_head=data.get("isaaclab_head"),
        tensorboard_event_path=data.get("tensorboard_event_path"),
        tensorboard_event_hash=data.get("tensorboard_event_hash"),
        artifact_hashes=dict(data["artifact_hashes"]),
        state=TerminalState(data["state"]),
        failure_category=(FailureCategory(data["failure_category"]) if data["failure_category"] else None),
        retry=RetryLineage(attempt=int(retry_data["attempt"]), parent_run_id=retry_data["parent_run_id"]),
    )


_LEGAL_TRANSITIONS = {
    TerminalState.PLANNED: frozenset({TerminalState.RUNNING}),
    TerminalState.RUNNING: frozenset({TerminalState.COMPLETED, TerminalState.FAILED}),
    TerminalState.COMPLETED: frozenset({TerminalState.INVALIDATED}),
    TerminalState.INVALIDATED: frozenset(),
    TerminalState.FAILED: frozenset(),
}


def transition(
    manifest: RunManifest,
    state: TerminalState,
    *,
    failure_category: FailureCategory | None = None,
) -> RunManifest:
    """Return a manifest advanced through one legal state transition."""
    if state not in _LEGAL_TRANSITIONS[manifest.state]:
        raise ValueError(f"illegal manifest transition: {manifest.state} -> {state}")
    if state is TerminalState.FAILED and failure_category is None:
        raise ValueError("failed manifest transition requires a failure category")
    if state is not TerminalState.FAILED and failure_category is not None:
        raise ValueError("failure category is only valid for a failed manifest")
    return replace(manifest, state=state, failure_category=failure_category)


def resume_matches(
    manifest: RunManifest,
    *,
    identity: RunIdentity,
    command: tuple[str, ...] | list[str],
    revisions: Revisions,
    schema_version: str,
    isaaclab_head: str,
) -> bool:
    """Return whether a completed manifest exactly matches a requested run."""
    return (
        manifest.state is TerminalState.COMPLETED
        and manifest.identity == identity
        and manifest.command_hash == command_hash(command)
        and manifest.revisions == revisions
        and manifest.schema_version == schema_version
        and manifest.isaaclab_head == isaaclab_head
    )
