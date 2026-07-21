# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for atomic and resumable benchmark manifests."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.kamino_dvi.manifests import (
    command_hash,
    read_manifest,
    relative_artifact_path,
    resume_matches,
    sha256_file,
    stable_run_id,
    transition,
    write_json_atomic,
    write_manifest,
)
from benchmarks.kamino_dvi.matrix import DEFAULT_MATRIX_PATH, load_matrix
from benchmarks.kamino_dvi.models import Phase, RunIdentity, RunManifest, TaskName, TerminalState, Variant


@pytest.fixture
def matrix():
    return load_matrix(DEFAULT_MATRIX_PATH)


@pytest.fixture
def identity():
    return RunIdentity(TaskName.CARTPOLE, Variant.KAMINO_PR_DVI, 42, Phase.FULL, 4096, 300)


@pytest.fixture
def manifest(matrix, identity):
    command = ("/repo/.venv-pr3570/bin/python", "training.py", "--seed", "42")
    return RunManifest(
        run_id=stable_run_id(identity),
        identity=identity,
        command=command,
        command_hash=command_hash(command),
        revisions=matrix.revisions,
        schema_version="1.1",
        artifact_root="full/cartpole/kamino_pr_dvi/seed42",
        isaaclab_head="f" * 40,
    )


def test_stable_run_id_and_command_hash_are_deterministic(identity):
    """Run and command identities must be stable across process restarts."""
    assert stable_run_id(identity) == "full__Isaac-Cartpole-Direct__kamino_pr_dvi__seed42__env4096__iter300"
    assert command_hash(["python", "train.py", "--seed", "42"]) == command_hash(("python", "train.py", "--seed", "42"))
    assert command_hash(["python", "train.py", "--seed", "43"]) != command_hash(["python", "train.py", "--seed", "42"])


def test_sha256_file_and_relative_artifact_path(tmp_path: Path):
    """Tracked summaries must reference and hash raw artifacts unambiguously."""
    root = tmp_path / "raw"
    artifact = root / "run" / "training.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"abc")

    assert sha256_file(artifact) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert relative_artifact_path(artifact, root) == "run/training.json"
    with pytest.raises(ValueError, match="outside artifact root"):
        relative_artifact_path(tmp_path / "elsewhere.json", root)


def test_manifest_round_trip_is_atomic_and_typed(tmp_path: Path, manifest):
    """Atomic persistence must preserve enums, revisions, and run identity."""
    path = tmp_path / "manifest.json"

    write_manifest(path, manifest)

    assert read_manifest(path) == manifest
    assert not list(tmp_path.glob("*.tmp"))
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_write_json_atomic_replaces_complete_document(tmp_path):
    path = tmp_path / "decision.json"
    write_json_atomic(path, {"state": "old"})
    write_json_atomic(path, {"state": "new", "values": [1, 2]})
    assert json.loads(path.read_text()) == {"state": "new", "values": [1, 2]}
    assert not tuple(tmp_path.glob("*.tmp"))


def test_manifest_reads_legacy_artifact_without_exact_isaaclab_head(tmp_path: Path, manifest):
    """Completed raw artifacts from the original campaign remain analyzable."""
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["isaaclab_head"]
    path.write_text(json.dumps(data), encoding="utf-8")

    assert read_manifest(path).isaaclab_head is None


def test_manifest_allows_only_legal_state_transitions(manifest):
    """Run state must advance monotonically to one terminal outcome."""
    running = transition(manifest, TerminalState.RUNNING)
    completed = transition(running, TerminalState.COMPLETED)

    assert completed.state is TerminalState.COMPLETED
    with pytest.raises(ValueError, match="illegal manifest transition"):
        transition(manifest, TerminalState.COMPLETED)
    with pytest.raises(ValueError, match="illegal manifest transition"):
        transition(completed, TerminalState.RUNNING)


def test_resume_requires_exact_completed_manifest(manifest):
    """Only an exact successful terminal artifact may be skipped on resume."""
    completed = transition(transition(manifest, TerminalState.RUNNING), TerminalState.COMPLETED)

    assert resume_matches(
        completed,
        identity=manifest.identity,
        command=manifest.command,
        revisions=manifest.revisions,
        schema_version="1.1",
        isaaclab_head=manifest.isaaclab_head,
    )
    assert not resume_matches(
        replace(completed, state=TerminalState.FAILED),
        identity=manifest.identity,
        command=manifest.command,
        revisions=manifest.revisions,
        schema_version="1.1",
        isaaclab_head=manifest.isaaclab_head,
    )
    assert not resume_matches(
        completed,
        identity=replace(manifest.identity, num_envs=2048),
        command=manifest.command,
        revisions=manifest.revisions,
        schema_version="1.1",
        isaaclab_head=manifest.isaaclab_head,
    )
    assert not resume_matches(
        completed,
        identity=manifest.identity,
        command=manifest.command + ("--extra",),
        revisions=manifest.revisions,
        schema_version="1.1",
        isaaclab_head=manifest.isaaclab_head,
    )
    assert not resume_matches(
        completed,
        identity=manifest.identity,
        command=manifest.command,
        revisions=replace(manifest.revisions, newton_pr="0" * 40),
        schema_version="1.1",
        isaaclab_head=manifest.isaaclab_head,
    )
    assert not resume_matches(
        completed,
        identity=manifest.identity,
        command=manifest.command,
        revisions=manifest.revisions,
        schema_version="1.0",
        isaaclab_head=manifest.isaaclab_head,
    )
    assert not resume_matches(
        completed,
        identity=manifest.identity,
        command=manifest.command,
        revisions=manifest.revisions,
        schema_version="1.1",
        isaaclab_head="0" * 40,
    )


def test_atomic_json_rejects_nonfinite_values(tmp_path):
    """Canonical JSON persistence refuses NaN and infinity."""
    path = tmp_path / "invalid.json"

    with pytest.raises(ValueError, match="JSON"):
        write_json_atomic(path, {"value": float("nan")})

    assert not path.exists()
