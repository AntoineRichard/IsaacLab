# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run provenance-safe ANYmal-D Kamino DVI tuning stages."""

from __future__ import annotations

import argparse
import json
import shlex
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .commands import build_training_command
from .environment import probe_environment, python_executable, validate_environment
from .failures import classify_failure
from .manifests import command_hash, sha256_file, write_json_atomic
from .matrix import DEFAULT_MATRIX_PATH, load_matrix
from .models import (
    BenchmarkMatrix,
    FailureCategory,
    Phase,
    RetryLineage,
    Revisions,
    RunIdentity,
    TaskName,
    TerminalState,
    Variant,
)
from .parsing import locate_rsl_rl_events, parse_training_trace
from .run import BundleStatus, ProcessOutcome, execute_command, inspect_bundle
from .tuning import (
    DEFAULT_TUNING_MATRIX_PATH,
    SolverValue,
    TuningCandidate,
    TuningMatrix,
    config_hash,
    hydra_overrides,
    load_tuning_matrix,
    resolve_config,
)

SCHEMA_VERSION = "1.1"
_CANONICAL_CANDIDATE = "canonical_winner"


@dataclass(frozen=True)
class TuningIdentity:
    """One immutable tuning-stage training-run identity.

    Attributes:
        stage: Campaign stage that determines command semantics.
        candidate: Candidate configuration name.
        seed: Training random seed.
        num_envs: Number of parallel simulation environments.
        max_iterations: Number of training iterations.
        attempt: Monotonic raw-evidence attempt number.
    """

    stage: str
    candidate: str
    seed: int
    num_envs: int
    max_iterations: int
    attempt: int = 0

    @property
    def run_id(self) -> str:
        """Return the stable filesystem-safe run identifier."""
        return (
            f"{self.stage}__{self.candidate}__seed{self.seed}__env{self.num_envs}"
            f"__iter{self.max_iterations}__attempt{self.attempt}"
        )


@dataclass(frozen=True)
class TuningManifest:
    """Persistent state and provenance for one tuning attempt.

    Attributes:
        run_id: Stable identifier for this exact attempt.
        identity: Immutable tuning identity.
        config_hash: SHA-256 of :attr:`resolved_config`.
        resolved_config: Literal complete solver configuration.
        command: Exact subprocess argument tuple.
        command_hash: SHA-256 of :attr:`command`.
        revisions: Locked Isaac Lab, schema, and Newton revisions.
        schema_version: Required benchmark bundle schema version.
        isaaclab_head: Exact source HEAD used by the run.
        artifact_root: Absolute attempt artifact directory.
        tensorboard_event_path: Matched RSL-RL event path, if available.
        tensorboard_event_hash: SHA-256 of the matched event, if available.
        artifact_hashes: SHA-256 values for stdout, stderr, and bundle artifacts.
        state: Persisted attempt lifecycle state.
        failure_category: Terminal failure classification, if failed.
        retry: Attempt number and readable parent lineage.
    """

    run_id: str
    identity: TuningIdentity
    config_hash: str
    resolved_config: dict[str, SolverValue]
    command: tuple[str, ...]
    command_hash: str
    revisions: Revisions
    schema_version: str
    isaaclab_head: str
    artifact_root: str
    tensorboard_event_path: str | None = None
    tensorboard_event_hash: str | None = None
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    state: TerminalState = TerminalState.PLANNED
    failure_category: FailureCategory | None = None
    retry: RetryLineage = field(default_factory=RetryLineage)


@dataclass(frozen=True)
class ResolvedTuningCandidate:
    """A named candidate with literal resolved configuration provenance.

    Attributes:
        name: Candidate name.
        overrides: Declared solver overrides.
        resolved_config: Complete solver configuration after applying overrides.
        config_hash: SHA-256 of :attr:`resolved_config`.
    """

    name: str
    overrides: dict[str, SolverValue]
    resolved_config: dict[str, SolverValue]
    config_hash: str


def _run_identity(identity: TuningIdentity) -> RunIdentity:
    phase = Phase.PREFLIGHT if identity.max_iterations == 5 else Phase.FULL
    return RunIdentity(
        task=TaskName.ANYMAL_D,
        variant=Variant.KAMINO_PR_DVI,
        seed=identity.seed,
        phase=phase,
        num_envs=identity.num_envs,
        max_iterations=identity.max_iterations,
    )


def build_tuning_command(
    matrix: BenchmarkMatrix,
    tuning_matrix: TuningMatrix,
    candidate: TuningCandidate,
    identity: TuningIdentity,
    repo_root: Path,
    output_path: Path,
) -> list[str]:
    """Build the locked PR3570 DVI command plus declared tuning overrides.

    Args:
        matrix: Locked five-variant comparison matrix.
        tuning_matrix: Validated ANYmal-D tuning protocol.
        candidate: Declared non-canonical tuning candidate.
        identity: Exact stage, seed, environment count, iterations, and attempt.
        repo_root: Isaac Lab repository root.
        output_path: Attempt artifact directory.

    Returns:
        Shell-free subprocess argument vector.

    Raises:
        ValueError: If the candidate is reserved, the stage is canonical, or
            the candidate/environment count does not match the identity and
            locked protocol.
    """
    if candidate.name == _CANONICAL_CANDIDATE:
        raise ValueError("canonical_winner is reserved for the canonical stage")
    if identity.stage == "canonical":
        raise ValueError("canonical stage commands must use the committed preset")
    if identity.candidate != candidate.name:
        raise ValueError("tuning identity candidate does not match command candidate")
    if identity.num_envs != tuning_matrix.num_envs:
        raise ValueError(f"tuning commands require {tuning_matrix.num_envs} environments")
    command = build_training_command(matrix, _run_identity(identity), repo_root, output_path)
    command.extend(hydra_overrides(tuning_matrix, candidate))
    return command


def validate_tuning_command(
    matrix: BenchmarkMatrix,
    tuning_matrix: TuningMatrix,
    candidate: TuningCandidate,
    identity: TuningIdentity,
    repo_root: Path,
    output_path: Path,
    command: tuple[str, ...] | list[str],
) -> None:
    """Require an exact base command and declared override sequence."""
    expected = tuple(build_tuning_command(matrix, tuning_matrix, candidate, identity, repo_root, output_path))
    if tuple(command) != expected:
        raise ValueError("tuning command must exactly match the reconstructed command")


def tuning_resume_matches(
    manifest: TuningManifest,
    identity: TuningIdentity,
    command: tuple[str, ...] | list[str],
    expected_config_hash: str,
    isaaclab_head: str,
) -> bool:
    """Return whether a completed tuning manifest exactly matches provenance."""
    expected_command = tuple(command)
    return (
        manifest.state is TerminalState.COMPLETED
        and manifest.identity == identity
        and manifest.run_id == identity.run_id
        and manifest.command == expected_command
        and manifest.command_hash == command_hash(expected_command)
        and manifest.config_hash == expected_config_hash
        and manifest.isaaclab_head == isaaclab_head
        and manifest.tensorboard_event_path is not None
        and manifest.tensorboard_event_hash is not None
    )


def _resolved_candidate(tuning_matrix: TuningMatrix, candidate: TuningCandidate) -> ResolvedTuningCandidate:
    resolved = resolve_config(tuning_matrix, candidate)
    return ResolvedTuningCandidate(candidate.name, dict(candidate.overrides), resolved, config_hash(resolved))


def build_canonical_command(
    matrix: BenchmarkMatrix,
    tuning_matrix: TuningMatrix,
    winner: dict[str, Any],
    repo_root: Path,
) -> tuple[TuningIdentity, ResolvedTuningCandidate, list[str]]:
    """Build a committed-preset command while retaining literal winner provenance."""
    resolved_config = dict(winner["resolved_config"])
    expected_hash = config_hash(resolved_config)
    if winner["config_hash"] != expected_hash:
        raise ValueError("winner resolved configuration hash mismatch")
    candidate = ResolvedTuningCandidate(
        name=_CANONICAL_CANDIDATE,
        overrides={},
        resolved_config=resolved_config,
        config_hash=expected_hash,
    )
    identity = TuningIdentity("canonical", candidate.name, 42, tuning_matrix.num_envs, tuning_matrix.final_iterations)
    output_path = repo_root / identity.run_id
    command = build_training_command(matrix, _run_identity(identity), repo_root, output_path)
    return identity, candidate, command


def write_tuning_manifest(path: Path, manifest: TuningManifest) -> None:
    """Atomically persist a tuning manifest."""
    write_json_atomic(path, asdict(manifest))


def read_tuning_manifest(path: Path) -> TuningManifest:
    """Read a tuning manifest without coercing identity or retry types."""
    data = json.loads(path.read_text(encoding="utf-8"))
    identity_data = data["identity"]
    retry_data = data["retry"]
    if not isinstance(identity_data, dict):
        raise ValueError("typed identity must be a mapping")
    if not isinstance(identity_data.get("stage"), str) or not isinstance(identity_data.get("candidate"), str):
        raise ValueError("typed identity stage and candidate must be strings")
    for field_name in ("seed", "num_envs", "max_iterations", "attempt"):
        value = identity_data.get(field_name)
        if type(value) is not int:
            raise ValueError(f"typed identity {field_name} must be an integer")
    identity = TuningIdentity(**identity_data)
    if not isinstance(retry_data, dict) or type(retry_data.get("attempt")) is not int:
        raise ValueError("retry attempt must be an integer")
    parent_run_id = retry_data.get("parent_run_id")
    if parent_run_id is not None and not isinstance(parent_run_id, str):
        raise ValueError("retry parent_run_id must be a string or null")
    retry = RetryLineage(attempt=retry_data["attempt"], parent_run_id=parent_run_id)
    if retry.attempt != identity.attempt:
        raise ValueError("retry attempt does not match identity attempt")
    return TuningManifest(
        run_id=data["run_id"],
        identity=identity,
        config_hash=data["config_hash"],
        resolved_config=dict(data["resolved_config"]),
        command=tuple(data["command"]),
        command_hash=data["command_hash"],
        revisions=Revisions(**data["revisions"]),
        schema_version=data["schema_version"],
        isaaclab_head=data["isaaclab_head"],
        artifact_root=data["artifact_root"],
        tensorboard_event_path=data.get("tensorboard_event_path"),
        tensorboard_event_hash=data.get("tensorboard_event_hash"),
        artifact_hashes=dict(data["artifact_hashes"]),
        state=TerminalState(data["state"]),
        failure_category=(FailureCategory(data["failure_category"]) if data["failure_category"] else None),
        retry=retry,
    )


def _candidate_provenance(
    tuning_matrix: TuningMatrix, candidate: TuningCandidate | ResolvedTuningCandidate
) -> ResolvedTuningCandidate:
    if isinstance(candidate, ResolvedTuningCandidate):
        if candidate.config_hash != config_hash(candidate.resolved_config):
            raise ValueError(f"candidate {candidate.name} has mismatched configuration hash")
        if candidate.name != _CANONICAL_CANDIDATE:
            declared = TuningCandidate(candidate.name, candidate.overrides)
            if candidate.resolved_config != resolve_config(tuning_matrix, declared):
                raise ValueError(f"candidate {candidate.name} has mismatched resolved configuration")
        return candidate
    return _resolved_candidate(tuning_matrix, candidate)


def _command_for_candidate(
    matrix: BenchmarkMatrix,
    tuning_matrix: TuningMatrix,
    candidate: ResolvedTuningCandidate,
    identity: TuningIdentity,
    repo_root: Path,
    output_path: Path,
) -> list[str]:
    if identity.candidate != candidate.name:
        raise ValueError("tuning identity candidate does not match command candidate")
    if identity.stage == "canonical":
        if candidate.name != _CANONICAL_CANDIDATE:
            raise ValueError("canonical stage requires the reserved canonical_winner candidate")
        return build_training_command(matrix, _run_identity(identity), repo_root, output_path)
    if candidate.name == _CANONICAL_CANDIDATE:
        raise ValueError("canonical_winner is reserved for the canonical stage")
    declared = TuningCandidate(candidate.name, candidate.overrides)
    command = build_tuning_command(matrix, tuning_matrix, declared, identity, repo_root, output_path)
    validate_tuning_command(matrix, tuning_matrix, declared, identity, repo_root, output_path, command)
    return command


def _same_base_identity(left: TuningIdentity, right: TuningIdentity) -> bool:
    return (
        left.stage,
        left.candidate,
        left.seed,
        left.num_envs,
        left.max_iterations,
    ) == (
        right.stage,
        right.candidate,
        right.seed,
        right.num_envs,
        right.max_iterations,
    )


def _occupied_attempts(artifact_root: Path, identity: TuningIdentity) -> tuple[int, ...]:
    prefix = (
        f"{identity.stage}__{identity.candidate}__seed{identity.seed}__env{identity.num_envs}"
        f"__iter{identity.max_iterations}__attempt"
    )
    attempts: list[int] = []
    for path in artifact_root.glob(f"{prefix}*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix(prefix)
        if suffix.isdigit():
            attempts.append(int(suffix))
    return tuple(sorted(attempts))


def _existing_attempts(artifact_root: Path, identity: TuningIdentity) -> list[TuningManifest]:
    manifests: list[TuningManifest] = []
    pattern = (
        f"{identity.stage}__{identity.candidate}__seed{identity.seed}__env{identity.num_envs}"
        f"__iter{identity.max_iterations}__attempt*/manifest.json"
    )
    for path in artifact_root.glob(pattern):
        try:
            manifest = read_tuning_manifest(path)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if (
            _same_base_identity(manifest.identity, identity)
            and manifest.run_id == manifest.identity.run_id
            and path.parent.name == manifest.run_id
            and Path(manifest.artifact_root).resolve() == path.parent.resolve()
        ):
            manifests.append(manifest)
    return sorted(manifests, key=lambda manifest: manifest.identity.attempt)


def _recorded_artifacts_intact(manifest: TuningManifest) -> bool:
    event_path = Path(manifest.tensorboard_event_path) if manifest.tensorboard_event_path is not None else None
    if event_path is None or not event_path.is_file() or sha256_file(event_path) != manifest.tensorboard_event_hash:
        return False
    output_path = Path(manifest.artifact_root)
    bundle_keys = tuple(
        name for name in manifest.artifact_hashes if name.startswith("benchmark_training_") and name.endswith(".json")
    )
    mandatory = {"stdout.log", "stderr.log"}
    if not mandatory.issubset(manifest.artifact_hashes) or len(bundle_keys) != 1:
        return False
    return all(
        (output_path / name).is_file() and sha256_file(output_path / name) == digest
        for name, digest in manifest.artifact_hashes.items()
    )


def _append_exception(path: Path, error: Exception) -> None:
    try:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(f"{type(error).__name__}: {error}\n")
    except OSError:
        pass


def _read_log(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def execute_tuning_identity(
    matrix: BenchmarkMatrix,
    tuning_matrix: TuningMatrix,
    candidate: TuningCandidate | ResolvedTuningCandidate,
    identity: TuningIdentity,
    repo_root: Path,
    artifact_root: Path,
    *,
    isaaclab_head: str,
    resume: bool,
    executor: Callable[..., ProcessOutcome] | None = None,
) -> TerminalState:
    """Execute one tuning identity without overwriting any prior attempt.

    Args:
        matrix: Locked five-variant comparison matrix.
        tuning_matrix: Validated ANYmal-D tuning protocol.
        candidate: Candidate with declared or resolved configuration provenance.
        identity: Requested tuning run identity.
        repo_root: Isaac Lab repository root.
        artifact_root: Root containing immutable attempt directories.
        isaaclab_head: Exact source HEAD used for provenance and resume matching.
        resume: Whether an exact latest completed attempt may be skipped.
        executor: Optional subprocess executor used by tests.

    Returns:
        Completed or failed terminal state.

    Raises:
        ValueError: If candidate provenance, reserved naming, or stage semantics
            do not match the requested identity.
    """
    if executor is None:
        executor = execute_command
    resolved_candidate = _candidate_provenance(tuning_matrix, candidate)
    if identity.stage == "canonical" and resolved_candidate.name != _CANONICAL_CANDIDATE:
        raise ValueError("canonical stage requires the reserved canonical_winner candidate")
    if identity.stage != "canonical" and resolved_candidate.name == _CANONICAL_CANDIDATE:
        raise ValueError("canonical_winner is reserved for the canonical stage")
    if resolved_candidate.name != identity.candidate:
        raise ValueError("tuning identity candidate does not match resolved candidate")
    occupied = _occupied_attempts(artifact_root, identity)
    attempts = _existing_attempts(artifact_root, identity)
    previous = attempts[-1] if attempts else None
    if resume and previous is not None and occupied and previous.identity.attempt == occupied[-1]:
        output_path = Path(previous.artifact_root)
        expected_command = _command_for_candidate(
            matrix, tuning_matrix, resolved_candidate, previous.identity, repo_root, output_path
        )
        if (
            previous.revisions == matrix.revisions
            and previous.schema_version == SCHEMA_VERSION
            and previous.resolved_config == resolved_candidate.resolved_config
            and tuning_resume_matches(
                previous,
                previous.identity,
                expected_command,
                resolved_candidate.config_hash,
                isaaclab_head,
            )
            and _recorded_artifacts_intact(previous)
        ):
            return TerminalState.COMPLETED

    attempt = occupied[-1] + 1 if occupied else identity.attempt
    current_identity = replace(identity, attempt=attempt)
    output_path = artifact_root / current_identity.run_id
    output_path.mkdir(parents=True, exist_ok=False)
    command = _command_for_candidate(
        matrix, tuning_matrix, resolved_candidate, current_identity, repo_root, output_path
    )
    retry = RetryLineage(attempt=attempt, parent_run_id=previous.run_id if previous is not None else None)
    manifest = TuningManifest(
        run_id=current_identity.run_id,
        identity=current_identity,
        config_hash=resolved_candidate.config_hash,
        resolved_config=resolved_candidate.resolved_config,
        command=tuple(command),
        command_hash=command_hash(command),
        revisions=matrix.revisions,
        schema_version=SCHEMA_VERSION,
        isaaclab_head=isaaclab_head,
        artifact_root=str(output_path.resolve()),
        retry=retry,
    )
    manifest_path = output_path / "manifest.json"
    write_tuning_manifest(manifest_path, manifest)
    manifest = replace(manifest, state=TerminalState.RUNNING)
    write_tuning_manifest(manifest_path, manifest)

    stdout_path = output_path / "stdout.log"
    stderr_path = output_path / "stderr.log"
    timeout_s = matrix.preflight_timeout_s if current_identity.max_iterations == 5 else matrix.full_timeout_s
    outcome = ProcessOutcome(returncode=1, timed_out=False)
    bundle = BundleStatus(path=None, schema_version=None, completed_iterations=0, complete=False)
    event_path: Path | None = None
    event_hash: str | None = None
    trace_valid = False
    forced_failure: FailureCategory | None = None

    try:
        outcome = executor(command, stdout_path, stderr_path, timeout_s=timeout_s)
    except Exception as error:
        forced_failure = FailureCategory.CRASH
        _append_exception(stderr_path, error)

    if forced_failure is None:
        try:
            bundle = inspect_bundle(output_path, current_identity.max_iterations)
        except Exception as error:
            forced_failure = FailureCategory.ARTIFACT
            _append_exception(stderr_path, error)

    process_succeeded = outcome.returncode == 0 and not outcome.timed_out and bundle.complete
    if forced_failure is None and process_succeeded and bundle.path is not None:
        try:
            event_path = locate_rsl_rl_events(bundle.path, repo_root / "logs")
            trace = parse_training_trace(bundle.path, event_path)
            trace_valid = (
                trace.task == TaskName.ANYMAL_D.value
                and trace.seed == current_identity.seed
                and trace.num_envs == current_identity.num_envs
                and trace.iterations == current_identity.max_iterations
            )
            if not trace_valid:
                forced_failure = FailureCategory.ARTIFACT
        except Exception as error:
            forced_failure = FailureCategory.ARTIFACT
            _append_exception(stderr_path, error)

    artifacts = (stdout_path, stderr_path, bundle.path)
    artifact_hashes: dict[str, str] = {}
    for path in artifacts:
        if path is None or not path.is_file():
            continue
        try:
            artifact_hashes[path.name] = sha256_file(path)
        except Exception as error:
            forced_failure = FailureCategory.ARTIFACT
            _append_exception(stderr_path, error)
    if event_path is not None and event_path.is_file():
        try:
            event_hash = sha256_file(event_path)
        except Exception as error:
            forced_failure = FailureCategory.ARTIFACT
            _append_exception(stderr_path, error)

    success = forced_failure is None and process_succeeded and trace_valid
    if success:
        manifest = replace(manifest, state=TerminalState.COMPLETED)
    else:
        failure_category = forced_failure
        if failure_category is None:
            failure = classify_failure(
                returncode=outcome.returncode,
                timed_out=outcome.timed_out,
                completed_iterations=bundle.completed_iterations,
                expected_iterations=current_identity.max_iterations,
                artifact_present=bundle.path is not None and trace_valid,
                stdout=_read_log(stdout_path),
                stderr=_read_log(stderr_path),
                retry=retry,
            )
            failure_category = failure.category
        manifest = replace(manifest, state=TerminalState.FAILED, failure_category=failure_category)
    manifest = replace(
        manifest,
        artifact_hashes=artifact_hashes,
        tensorboard_event_path=str(event_path.resolve()) if event_path is not None else None,
        tensorboard_event_hash=event_hash,
    )
    write_tuning_manifest(manifest_path, manifest)
    return manifest.state


def _load_resolved_candidates(
    tuning_matrix: TuningMatrix,
    path: Path,
    expected_action: str,
    minimum_count: int,
    maximum_count: int,
) -> tuple[ResolvedTuningCandidate, ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "1.0" or data.get("action") != expected_action:
        raise ValueError(f"{path.name} decision schema/action mismatch")
    records = data.get("resolved_candidates")
    if not isinstance(records, list) or not minimum_count <= len(records) <= maximum_count:
        raise ValueError(f"{path.name} must contain between {minimum_count} and {maximum_count} candidates")
    candidates: list[ResolvedTuningCandidate] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"{path.name} candidate records must be mappings")
        declared = TuningCandidate(str(record["name"]), dict(record["overrides"]))
        if declared.name == _CANONICAL_CANDIDATE:
            raise ValueError("canonical_winner is a reserved candidate name")
        resolved = resolve_config(tuning_matrix, declared)
        literal = dict(record["resolved_config"])
        digest = config_hash(literal)
        if literal != resolved or record["config_hash"] != digest:
            raise ValueError(f"{path.name} candidate {declared.name} has mismatched configuration provenance")
        candidates.append(ResolvedTuningCandidate(declared.name, declared.overrides, literal, digest))
    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError(f"{path.name} contains duplicate candidate names")
    return tuple(candidates)


def _canonical_candidate(tuning_matrix: TuningMatrix, decision_root: Path) -> ResolvedTuningCandidate:
    winner = json.loads((decision_root / "winner.json").read_text(encoding="utf-8"))
    if winner.get("schema_version") != "1.0" or winner.get("action") != "select-winner":
        raise ValueError("winner decision schema/action mismatch")
    resolved = dict(winner["resolved_config"])
    digest = config_hash(resolved)
    if winner["config_hash"] != digest or set(resolved) != set(tuning_matrix.baseline):
        raise ValueError("winner decision has mismatched configuration provenance")
    return ResolvedTuningCandidate(_CANONICAL_CANDIDATE, {}, resolved, digest)


def stage_candidates(
    tuning_matrix: TuningMatrix, stage: str, decision_root: Path
) -> tuple[ResolvedTuningCandidate, ...]:
    """Load and validate the exact candidate set for a tuning stage."""
    if stage == "baseline":
        baseline = TuningCandidate("baseline", {})
        return (_resolved_candidate(tuning_matrix, baseline),)
    if stage == "wave1":
        return tuple(_resolved_candidate(tuning_matrix, candidate) for candidate in tuning_matrix.wave1)
    if stage == "wave2":
        return _load_resolved_candidates(tuning_matrix, decision_root / "wave2.json", "resolve-wave2", 6, 6)
    if stage == "halve":
        return _load_resolved_candidates(tuning_matrix, decision_root / "stage2.json", "promote-stage2", 1, 8)
    if stage == "final":
        return _load_resolved_candidates(tuning_matrix, decision_root / "finalists.json", "promote-finalists", 1, 3)
    if stage == "canonical":
        return (_canonical_candidate(tuning_matrix, decision_root),)
    raise ValueError(f"unsupported tuning stage: {stage}")


def build_parser() -> argparse.ArgumentParser:
    """Build the tuning-stage command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("baseline", "wave1", "wave2", "halve", "final", "canonical"))
    parser.add_argument("--candidate")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--artifact-root", type=Path, default=Path("benchmark_artifacts/kamino_dvi/tuning"))
    parser.add_argument("--decision-root", type=Path, default=Path("benchmark_artifacts/kamino_dvi/decisions"))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--measured-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _measured_identities(
    tuning_matrix: TuningMatrix, stage: str, candidates: tuple[ResolvedTuningCandidate, ...]
) -> tuple[TuningIdentity, ...]:
    if stage in {"baseline", "final", "canonical"}:
        seeds = tuning_matrix.seeds
        iterations = tuning_matrix.final_iterations
    elif stage in {"wave1", "wave2"}:
        seeds = (42,)
        iterations = tuning_matrix.screen_iterations
    elif stage == "halve":
        seeds = (42, 43)
        iterations = tuning_matrix.halve_iterations
    else:
        raise ValueError(f"unsupported tuning stage: {stage}")
    return tuple(
        TuningIdentity(stage, candidate.name, seed, tuning_matrix.num_envs, iterations)
        for candidate in candidates
        for seed in seeds
    )


def select_tuning_identities(tuning_matrix: TuningMatrix, args: argparse.Namespace) -> tuple[TuningIdentity, ...]:
    """Expand and filter exact preflight and measured tuning identities.

    Args:
        tuning_matrix: Validated ANYmal-D tuning protocol.
        args: Parsed stage, candidate, seed, and phase filters.

    Returns:
        Ordered immutable identities selected for execution.

    Raises:
        ValueError: If the requested candidate does not belong to the stage.
    """
    candidates = stage_candidates(tuning_matrix, args.stage, args.decision_root)
    if args.candidate is not None:
        candidates = tuple(candidate for candidate in candidates if candidate.name == args.candidate)
        if not candidates:
            raise ValueError(f"candidate {args.candidate!r} is not part of stage {args.stage}")
    preflights = tuple(
        TuningIdentity(
            "canonical" if args.stage == "canonical" else "preflight",
            candidate.name,
            42,
            tuning_matrix.num_envs,
            tuning_matrix.preflight_iterations,
        )
        for candidate in candidates
    )
    measured = _measured_identities(tuning_matrix, args.stage, candidates)
    if args.preflight_only:
        identities = preflights
    elif args.measured_only:
        identities = measured
    else:
        identities = preflights + measured
    if args.seed is not None:
        identities = tuple(identity for identity in identities if identity.seed == args.seed)
    return identities


def preflight_completed(
    matrix: BenchmarkMatrix,
    tuning_matrix: TuningMatrix,
    candidate: TuningCandidate | ResolvedTuningCandidate,
    repo_root: Path,
    artifact_root: Path,
    isaaclab_head: str,
) -> bool:
    """Return whether exact intact preflight evidence already exists."""
    resolved_candidate = _candidate_provenance(tuning_matrix, candidate)
    identity = TuningIdentity(
        "canonical" if resolved_candidate.name == _CANONICAL_CANDIDATE else "preflight",
        resolved_candidate.name,
        42,
        tuning_matrix.num_envs,
        tuning_matrix.preflight_iterations,
    )
    for manifest in reversed(_existing_attempts(artifact_root, identity)):
        output_path = Path(manifest.artifact_root)
        command = _command_for_candidate(
            matrix,
            tuning_matrix,
            resolved_candidate,
            manifest.identity,
            repo_root,
            output_path,
        )
        if (
            manifest.revisions == matrix.revisions
            and manifest.schema_version == SCHEMA_VERSION
            and manifest.resolved_config == resolved_candidate.resolved_config
            and tuning_resume_matches(
                manifest,
                manifest.identity,
                command,
                resolved_candidate.config_hash,
                isaaclab_head,
            )
            and _recorded_artifacts_intact(manifest)
        ):
            return True
    return False


def _validated_source_head(matrix: BenchmarkMatrix, repo_root: Path) -> str:
    label = matrix.variant(Variant.KAMINO_PR_DVI).environment
    provenance = probe_environment(python_executable(repo_root, label), repo_root)
    validate_environment(matrix, label, provenance)
    return provenance.isaaclab.head


def _selected_candidates(tuning_matrix: TuningMatrix, args: argparse.Namespace) -> tuple[ResolvedTuningCandidate, ...]:
    candidates = stage_candidates(tuning_matrix, args.stage, args.decision_root)
    if args.candidate is not None:
        candidates = tuple(candidate for candidate in candidates if candidate.name == args.candidate)
        if not candidates:
            raise ValueError(f"candidate {args.candidate!r} is not part of stage {args.stage}")
    return candidates


def main(argv: list[str] | None = None) -> int:
    """Validate provenance and execute the selected tuning stage sequentially."""
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    tuning_matrix = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    args.artifact_root = args.artifact_root if args.artifact_root.is_absolute() else repo_root / args.artifact_root
    args.decision_root = args.decision_root if args.decision_root.is_absolute() else repo_root / args.decision_root
    candidates = _selected_candidates(tuning_matrix, args)
    candidates_by_name = {candidate.name: candidate for candidate in candidates}
    identities = select_tuning_identities(tuning_matrix, args)

    if args.dry_run:
        for identity in identities:
            candidate = candidates_by_name[identity.candidate]
            output_path = args.artifact_root / identity.run_id
            command = _command_for_candidate(
                matrix,
                tuning_matrix,
                candidate,
                identity,
                repo_root,
                output_path,
            )
            print(shlex.join(command))
        return 0

    isaaclab_head = _validated_source_head(matrix, repo_root)
    failures = 0
    for identity in identities:
        candidate = candidates_by_name[identity.candidate]
        has_preflight = preflight_completed(
            matrix,
            tuning_matrix,
            candidate,
            repo_root,
            args.artifact_root,
            isaaclab_head,
        )
        is_preflight = identity.max_iterations == tuning_matrix.preflight_iterations
        if is_preflight and has_preflight:
            continue
        if not is_preflight and not has_preflight:
            message = (
                f"candidate {candidate.name} requires a completed exact preflight "
                f"for config {candidate.config_hash} and source HEAD {isaaclab_head}"
            )
            if args.measured_only:
                raise RuntimeError(message)
            print(f"REJECTED: {message}")
            failures += 1
            continue
        state = execute_tuning_identity(
            matrix,
            tuning_matrix,
            candidate,
            identity,
            repo_root,
            args.artifact_root,
            isaaclab_head=isaaclab_head,
            resume=args.resume,
        )
        if state is not TerminalState.COMPLETED:
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
