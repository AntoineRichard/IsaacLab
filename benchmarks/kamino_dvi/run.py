# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-GPU subprocess execution for the Kamino DVI benchmark."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from .commands import build_training_command
from .environment import probe_environment, python_executable, validate_environment
from .failures import classify_failure
from .manifests import (
    command_hash,
    read_manifest,
    resume_matches,
    sha256_file,
    stable_run_id,
    transition,
    write_manifest,
)
from .matrix import DEFAULT_MATRIX_PATH, expand_full_runs, expand_preflights, load_matrix
from .models import (
    BenchmarkMatrix,
    Phase,
    RetryLineage,
    RunIdentity,
    RunManifest,
    TaskName,
    TerminalState,
    Variant,
)

SCHEMA_VERSION = "1.1"


@dataclass(frozen=True)
class ProcessOutcome:
    """Terminal subprocess status returned to failure classification."""

    returncode: int | None
    timed_out: bool


@dataclass(frozen=True)
class BundleStatus:
    """Validated completion fields from a schema bundle."""

    path: Path | None
    schema_version: str | None
    completed_iterations: int
    complete: bool


def run_directory(artifact_root: Path, identity: RunIdentity) -> Path:
    """Return the deterministic artifact directory for one run identity."""
    return artifact_root / stable_run_id(identity)


def inspect_bundle(output_path: Path, expected_iterations: int) -> BundleStatus:
    """Inspect the newest schema bundle under an output directory."""
    paths = sorted(output_path.glob("benchmark_training_*.json"), key=lambda path: path.stat().st_mtime)
    if not paths:
        return BundleStatus(None, None, 0, False)
    path = paths[-1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        schema_version = str(data["schema_version"])
        completed_iterations = int(data["runtime"]["iterations_completed"])
        complete = data["run"]["status"] == "completed" and completed_iterations == expected_iterations
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return BundleStatus(path, None, 0, False)
    return BundleStatus(path, schema_version, completed_iterations, complete)


def execute_identity(
    matrix: BenchmarkMatrix,
    identity: RunIdentity,
    repo_root: Path,
    artifact_root: Path,
    *,
    resume: bool,
    executor: Callable[..., ProcessOutcome] | None = None,
) -> TerminalState:
    """Execute one identity with atomic manifests and exact resume matching."""
    if executor is None:
        executor = execute_command
    output_path = run_directory(artifact_root, identity)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "manifest.json"
    command = build_training_command(matrix, identity, repo_root, output_path)
    if resume and manifest_path.exists():
        try:
            existing = read_manifest(manifest_path)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            existing = None
        if existing is not None and resume_matches(
            existing,
            identity=identity,
            command=command,
            revisions=matrix.revisions,
            schema_version=SCHEMA_VERSION,
        ):
            return TerminalState.COMPLETED

    manifest = RunManifest(
        run_id=stable_run_id(identity),
        identity=identity,
        command=tuple(command),
        command_hash=command_hash(command),
        revisions=matrix.revisions,
        schema_version=SCHEMA_VERSION,
        artifact_root=str(output_path),
    )
    write_manifest(manifest_path, manifest)
    manifest = transition(manifest, TerminalState.RUNNING)
    write_manifest(manifest_path, manifest)

    stdout_path = output_path / "stdout.log"
    stderr_path = output_path / "stderr.log"
    timeout_s = matrix.preflight_timeout_s if identity.phase is Phase.PREFLIGHT else matrix.full_timeout_s
    outcome = executor(command, stdout_path, stderr_path, timeout_s=timeout_s)
    bundle = inspect_bundle(output_path, identity.max_iterations)
    success = outcome.returncode == 0 and not outcome.timed_out and bundle.complete

    artifact_hashes = {
        path.name: sha256_file(path)
        for path in (stdout_path, stderr_path, bundle.path)
        if path is not None and path.exists()
    }
    if success:
        manifest = transition(manifest, TerminalState.COMPLETED)
    else:
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
        failure = classify_failure(
            returncode=outcome.returncode,
            timed_out=outcome.timed_out,
            completed_iterations=bundle.completed_iterations,
            expected_iterations=identity.max_iterations,
            artifact_present=bundle.path is not None,
            stdout=stdout,
            stderr=stderr,
            retry=RetryLineage(),
        )
        manifest = transition(manifest, TerminalState.FAILED, failure_category=failure.category)
    manifest = RunManifest(**{**manifest.__dict__, "artifact_hashes": artifact_hashes})
    write_manifest(manifest_path, manifest)
    return manifest.state


def _terminate_process_group(pid: int) -> None:
    os.killpg(pid, signal.SIGTERM)


def execute_command(
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
    *,
    timeout_s: int,
    popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    kill_process_group: Callable[[int], None] = _terminate_process_group,
) -> ProcessOutcome:
    """Execute one training command with streamed logs and a hard timeout."""
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    timed_out = False
    process_env = os.environ.copy()
    process_env.pop("PYTHONPATH", None)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = popen_factory(
            command,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
            env=process_env,
        )
        try:
            process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            kill_process_group(process.pid)
            process.communicate()
    return ProcessOutcome(returncode=process.returncode, timed_out=timed_out)


def execute_sequentially(
    commands: Iterable[list[str]],
    executor: Callable[[list[str]], ProcessOutcome],
) -> tuple[ProcessOutcome, ...]:
    """Execute every command in order, retaining failures without stopping."""
    return tuple(executor(command) for command in commands)


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark-runner command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--preflight-only", action="store_true")
    phase.add_argument("--full-only", action="store_true")
    parser.add_argument("--task", choices=[task.value for task in TaskName])
    parser.add_argument("--variant", choices=[variant.value for variant in Variant])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--artifact-root", type=Path, default=Path("benchmark_artifacts/kamino_dvi/runs"))
    return parser


def select_identities(matrix: BenchmarkMatrix, args: argparse.Namespace) -> tuple[RunIdentity, ...]:
    """Expand and filter run identities without performing side effects."""
    identities: tuple[RunIdentity, ...]
    if args.preflight_only:
        identities = expand_preflights(matrix)
    elif args.full_only:
        identities = expand_full_runs(matrix)
    else:
        identities = expand_preflights(matrix) + expand_full_runs(matrix)
    if args.task is not None:
        task = TaskName(args.task)
        identities = tuple(identity for identity in identities if identity.task is task)
    if args.variant is not None:
        variant = Variant(args.variant)
        identities = tuple(identity for identity in identities if identity.variant is variant)
    if args.seed is not None:
        identities = tuple(identity for identity in identities if identity.seed == args.seed)
    return identities


def main(argv: list[str] | None = None) -> int:
    """Validate environments and execute the selected matrix identities sequentially."""
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    matrix = load_matrix(args.matrix)
    identities = select_identities(matrix, args)
    artifact_root = args.artifact_root if args.artifact_root.is_absolute() else repo_root / args.artifact_root

    if args.dry_run:
        for identity in identities:
            output_path = run_directory(artifact_root, identity)
            print(shlex.join(build_training_command(matrix, identity, repo_root, output_path)))
        return 0

    labels = {matrix.variant(identity.variant).environment for identity in identities}
    for label in labels:
        provenance = probe_environment(python_executable(repo_root, label), repo_root)
        validate_environment(matrix, label, provenance)

    failures = 0
    for index, identity in enumerate(identities, start=1):
        print(f"[{index}/{len(identities)}] {stable_run_id(identity)}", flush=True)
        state = execute_identity(matrix, identity, repo_root, artifact_root, resume=args.resume)
        print(f"[{state.value}] {stable_run_id(identity)}", flush=True)
        failures += state is TerminalState.FAILED
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
