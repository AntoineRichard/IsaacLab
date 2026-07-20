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
from dataclasses import dataclass, replace
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
    FailureCategory,
    Phase,
    RetryLineage,
    RunIdentity,
    RunManifest,
    TaskName,
    TerminalState,
    Variant,
)
from .parsing import MissingBenchmarkFieldError, locate_rsl_rl_events
from .scheduler import capacity_retry, next_environment_count

SCHEMA_VERSION = "1.1"


@dataclass(frozen=True)
class ProcessOutcome:
    """Terminal subprocess status returned to failure classification."""

    returncode: int | None
    timed_out: bool


@dataclass(frozen=True)
class ResumeFallback:
    """Persisted task fallback reconstructed from capacity-failure manifests."""

    num_envs: int
    retry: RetryLineage


@dataclass(frozen=True)
class ScheduledRun:
    """One queued identity and its persisted capacity-retry lineage."""

    identity: RunIdentity
    retry: RetryLineage = RetryLineage()


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


def _resume_manifest_matches(
    matrix: BenchmarkMatrix,
    manifest_path: Path,
    manifest: RunManifest,
    tasks: set[TaskName],
    repo_root: Path,
    isaaclab_head: str,
) -> bool:
    """Return whether a capacity manifest belongs to the exact current campaign."""
    identity = manifest.identity
    try:
        task = matrix.task(identity.task)
        expected_iterations = (
            matrix.preflight_iterations if identity.phase is Phase.PREFLIGHT else matrix.full_iterations
        )
        expected_command = tuple(build_training_command(matrix, identity, repo_root, manifest_path.parent))
    except (KeyError, ValueError):
        return False
    return (
        identity.task in tasks
        and identity.variant in task.variants
        and identity.seed in matrix.seeds
        and identity.num_envs in matrix.environment_counts
        and identity.max_iterations == expected_iterations
        and manifest.state is TerminalState.FAILED
        and manifest.failure_category is FailureCategory.CAPACITY
        and manifest.revisions == matrix.revisions
        and manifest.schema_version == SCHEMA_VERSION
        and manifest.isaaclab_head == isaaclab_head
        and manifest.run_id == stable_run_id(identity)
        and manifest_path.parent.name == manifest.run_id
        and Path(manifest.artifact_root).resolve() == manifest_path.parent.resolve()
        and manifest.command_hash == command_hash(manifest.command)
        and manifest.command == expected_command
    )


def derive_resume_fallbacks(
    matrix: BenchmarkMatrix, artifact_root: Path, tasks: set[TaskName], repo_root: Path, isaaclab_head: str
) -> dict[TaskName, ResumeFallback]:
    """Derive trusted fallbacks and preserve a persisted exhausted ladder."""
    selected: dict[TaskName, tuple[int, float, ResumeFallback]] = {}
    for manifest_path in sorted(artifact_root.glob("*/manifest.json")):
        try:
            manifest = read_manifest(manifest_path)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if not _resume_manifest_matches(matrix, manifest_path, manifest, tasks, repo_root, isaaclab_head):
            continue
        identity = manifest.identity
        next_count = next_environment_count(matrix, identity.num_envs, FailureCategory.CAPACITY)
        if next_count is None:
            raise RuntimeError(
                f"capacity ladder exhausted for {identity.task.value} at {identity.num_envs} environments"
            )
        rank = matrix.environment_counts.index(next_count)
        fallback = ResumeFallback(
            num_envs=next_count,
            retry=RetryLineage(attempt=manifest.retry.attempt + 1, parent_run_id=manifest.run_id),
        )
        candidate = (rank, manifest_path.stat().st_mtime, fallback)
        if identity.task not in selected or candidate[:2] > selected[identity.task][:2]:
            selected[identity.task] = candidate
    return {task: candidate[2] for task, candidate in selected.items()}


def rebuild_resume_schedule(
    matrix: BenchmarkMatrix, identities: tuple[RunIdentity, ...], fallbacks: dict[TaskName, ResumeFallback]
) -> list[ScheduledRun]:
    """Rebuild lower-count all-variant preflights before requested full runs."""
    selected_tasks = {identity.task for identity in identities}
    fallback_tasks = selected_tasks & fallbacks.keys()
    scheduled: list[ScheduledRun] = []
    emitted_preflights: set[TaskName] = set()
    for identity in identities:
        if identity.phase is not Phase.PREFLIGHT:
            continue
        fallback = fallbacks.get(identity.task)
        if fallback is None:
            scheduled.append(ScheduledRun(identity))
        elif identity.task not in emitted_preflights:
            scheduled.extend(
                ScheduledRun(run, fallback.retry)
                for run in expand_preflights(matrix, fallback.num_envs)
                if run.task is identity.task
            )
            emitted_preflights.add(identity.task)
    for task in (task.name for task in matrix.tasks):
        if task in fallback_tasks and task not in emitted_preflights:
            fallback = fallbacks[task]
            scheduled.extend(
                ScheduledRun(run, fallback.retry)
                for run in expand_preflights(matrix, fallback.num_envs)
                if run.task is task
            )
    for identity in identities:
        if identity.phase is not Phase.FULL:
            continue
        fallback = fallbacks.get(identity.task)
        if fallback is None:
            scheduled.append(ScheduledRun(identity))
        else:
            scheduled.append(ScheduledRun(replace(identity, num_envs=fallback.num_envs), fallback.retry))
    return scheduled


def invalidate_completed_full_results(artifact_root: Path, task: TaskName, num_envs: int) -> int:
    """Mark completed full results at an invalid task count as ineligible.

    Raw logs, bundles, and hashes remain in their original run directories.
    """
    invalidated = 0
    for manifest_path in sorted(artifact_root.glob("full__*/manifest.json")):
        manifest = read_manifest(manifest_path)
        identity = manifest.identity
        if (
            manifest.state is TerminalState.COMPLETED
            and identity.phase is Phase.FULL
            and identity.task is task
            and identity.num_envs == num_envs
        ):
            write_manifest(manifest_path, transition(manifest, TerminalState.INVALIDATED))
            invalidated += 1
    return invalidated


def execute_identity(
    matrix: BenchmarkMatrix,
    identity: RunIdentity,
    repo_root: Path,
    artifact_root: Path,
    *,
    isaaclab_head: str,
    resume: bool,
    retry: RetryLineage = RetryLineage(),
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
        if (
            existing is not None
            and existing.retry == retry
            and resume_matches(
                existing,
                identity=identity,
                command=command,
                revisions=matrix.revisions,
                schema_version=SCHEMA_VERSION,
                isaaclab_head=isaaclab_head,
            )
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
        isaaclab_head=isaaclab_head,
        retry=retry,
    )
    write_manifest(manifest_path, manifest)
    manifest = transition(manifest, TerminalState.RUNNING)
    write_manifest(manifest_path, manifest)

    stdout_path = output_path / "stdout.log"
    stderr_path = output_path / "stderr.log"
    timeout_s = matrix.preflight_timeout_s if identity.phase is Phase.PREFLIGHT else matrix.full_timeout_s
    outcome = executor(command, stdout_path, stderr_path, timeout_s=timeout_s)
    bundle = inspect_bundle(output_path, identity.max_iterations)
    process_succeeded = outcome.returncode == 0 and not outcome.timed_out and bundle.complete
    event_path: Path | None = None
    if process_succeeded and bundle.path is not None:
        try:
            event_path = locate_rsl_rl_events(bundle.path, repo_root / "logs")
        except MissingBenchmarkFieldError:
            event_path = None
    success = process_succeeded and event_path is not None

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
            artifact_present=bundle.path is not None and event_path is not None,
            stdout=stdout,
            stderr=stderr,
            retry=retry,
        )
        manifest = transition(manifest, TerminalState.FAILED, failure_category=failure.category)
    manifest = replace(
        manifest,
        artifact_hashes=artifact_hashes,
        tensorboard_event_path=str(event_path.resolve()) if event_path is not None else None,
        tensorboard_event_hash=sha256_file(event_path) if event_path is not None else None,
    )
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
    if args.dry_run and not args.resume:
        for scheduled in rebuild_resume_schedule(matrix, identities, {}):
            identity = scheduled.identity
            output_path = run_directory(artifact_root, identity)
            print(shlex.join(build_training_command(matrix, identity, repo_root, output_path)))
        return 0

    selected_tasks = {identity.task for identity in identities}
    labels = {matrix.variant(variant).environment for task in selected_tasks for variant in matrix.task(task).variants}
    provenances = {label: probe_environment(python_executable(repo_root, label), repo_root) for label in labels}
    for label, provenance in provenances.items():
        validate_environment(matrix, label, provenance)
    isaaclab_heads = {provenance.isaaclab.head for provenance in provenances.values()}
    if len(isaaclab_heads) != 1:
        raise ValueError("benchmark environments observed different IsaacLab revisions")
    isaaclab_head = isaaclab_heads.pop()
    fallbacks = (
        derive_resume_fallbacks(matrix, artifact_root, {run.task for run in identities}, repo_root, isaaclab_head)
        if args.resume
        else {}
    )
    schedule = rebuild_resume_schedule(matrix, identities, fallbacks)
    if args.dry_run:
        for scheduled in schedule:
            identity = scheduled.identity
            output_path = run_directory(artifact_root, identity)
            print(shlex.join(build_training_command(matrix, identity, repo_root, output_path)))
        return 0

    requested_full_runs = tuple(scheduled.identity for scheduled in schedule if scheduled.identity.phase is Phase.FULL)
    pending = list(schedule)
    failures = 0
    completed = 0
    while pending:
        scheduled = pending.pop(0)
        identity = scheduled.identity
        completed += 1
        print(f"[{completed}/pending] {stable_run_id(identity)}", flush=True)
        state = execute_identity(
            matrix,
            identity,
            repo_root,
            artifact_root,
            isaaclab_head=isaaclab_head,
            resume=args.resume,
            retry=scheduled.retry,
        )
        print(f"[{state.value}] {stable_run_id(identity)}", flush=True)
        if state is not TerminalState.FAILED:
            continue

        manifest = read_manifest(run_directory(artifact_root, identity) / "manifest.json")
        if manifest.failure_category is not FailureCategory.CAPACITY:
            failures += 1
            if identity.phase is Phase.PREFLIGHT:
                pending = [run for run in pending if run.identity.task is not identity.task]
            continue

        decision = capacity_retry(
            matrix,
            identity.task,
            identity.phase,
            identity.num_envs,
            failed_run_id=stable_run_id(identity),
        )
        if decision.full_results_invalidated:
            invalidate_completed_full_results(artifact_root, identity.task, decision.invalidated_count)
        retry = RetryLineage(attempt=manifest.retry.attempt + 1, parent_run_id=manifest.run_id)
        retried_full_runs = tuple(
            replace(run, num_envs=decision.next_count) for run in requested_full_runs if run.task is identity.task
        )
        pending = [run for run in pending if run.identity.task is not identity.task]
        pending[:0] = [
            *(ScheduledRun(run, retry) for run in decision.preflights),
            *(ScheduledRun(run, retry) for run in retried_full_runs),
        ]
        print(
            f"[capacity fallback] {identity.task.value}: "
            f"{decision.invalidated_count} -> {decision.next_count} environments",
            flush=True,
        )
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
