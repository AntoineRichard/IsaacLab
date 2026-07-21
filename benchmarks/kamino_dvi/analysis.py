# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Aggregate validated benchmark traces into three-seed summaries."""

from __future__ import annotations

import json
import os
import statistics
from collections import Counter
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

from .manifests import command_hash, read_manifest, sha256_file, stable_run_id
from .models import BenchmarkMatrix, Phase, RunIdentity
from .parsing import locate_rsl_rl_events, parse_training_trace
from .statistics import Estimate, final_window_mean, mean_ci95


@dataclass(frozen=True)
class RunMetrics:
    """Per-iteration metrics for one task, variant, and seed."""

    task: str
    variant: str
    seed: int
    num_envs: int
    iteration_time_s: tuple[float, ...]
    total_fps: tuple[float, ...]
    reward: tuple[float, ...]
    ep_length: tuple[float, ...]
    success_rate: tuple[float, ...] | None
    success_schema_mismatch: bool = False
    success_schema_mismatch_points: int = 0


@dataclass(frozen=True)
class VariantSummary:
    """Three-seed estimates for one task and physics variant."""

    task: str
    variant: str
    num_envs: int
    iteration_time_s: Estimate
    total_fps: Estimate
    reward: Estimate
    ep_length: Estimate
    success_rate: Estimate | None


def _steady_mean(values: tuple[float, ...]) -> float:
    if len(values) <= 10:
        raise ValueError("runtime series must contain more than ten warmup iterations")
    return statistics.mean(values[10:])


def summarize_records(records: list[RunMetrics]) -> list[VariantSummary]:
    """Reduce complete three-seed records with the approved runtime and learning windows."""
    summaries: list[VariantSummary] = []
    ordered = sorted(records, key=lambda record: (record.task, record.variant, record.seed))
    for (task, variant), grouped in groupby(ordered, key=lambda record: (record.task, record.variant)):
        runs = list(grouped)
        seeds = {run.seed for run in runs}
        counts = {run.num_envs for run in runs}
        if len(runs) != 3 or len(seeds) != 3:
            raise ValueError(f"{task}/{variant} requires three unique successful seeds")
        if len(counts) != 1:
            raise ValueError(f"{task}/{variant} mixes environment counts")
        success = None
        if all(run.success_rate is not None for run in runs):
            success = mean_ci95([final_window_mean(run.success_rate or ()) for run in runs])
        summaries.append(
            VariantSummary(
                task=task,
                variant=variant,
                num_envs=counts.pop(),
                iteration_time_s=mean_ci95([_steady_mean(run.iteration_time_s) for run in runs]),
                total_fps=mean_ci95([_steady_mean(run.total_fps) for run in runs]),
                reward=mean_ci95([final_window_mean(run.reward) for run in runs]),
                ep_length=mean_ci95([final_window_mean(run.ep_length) for run in runs]),
                success_rate=success,
            )
        )
    return summaries


def _identity_text(identity: tuple[str, str, int]) -> str:
    """Format a task, variant, and seed identity for validation errors."""
    task, variant, seed = identity
    return f"task={task}, variant={variant}, seed={seed}"


def validate_record_matrix(
    records: list[RunMetrics],
    matrix: BenchmarkMatrix,
    *,
    omitted_cells: set[tuple[str, str]] | frozenset[tuple[str, str]] = frozenset(),
) -> None:
    """Validate exact task, variant, seed, and capacity coverage against the matrix."""
    expected_cells = {(task.name.value, variant.value) for task in matrix.tasks for variant in task.variants}
    unexpected_omissions = sorted(set(omitted_cells) - expected_cells)
    if unexpected_omissions:
        raise ValueError(f"unexpected omitted benchmark cell: {unexpected_omissions[0]}")
    expected_all = {
        (task.name.value, variant.value, seed)
        for task in matrix.tasks
        for variant in task.variants
        for seed in matrix.seeds
    }
    expected_required = {identity for identity in expected_all if (identity[0], identity[1]) not in omitted_cells}
    identities = [(record.task, record.variant, record.seed) for record in records]
    counts = Counter(identities)
    unexpected = sorted(set(identities) - expected_all)
    if unexpected:
        raise ValueError(f"unexpected benchmark record: {_identity_text(unexpected[0])}")
    duplicates = sorted(identity for identity, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate benchmark record: {_identity_text(duplicates[0])}")
    missing = sorted(expected_required - set(identities))
    if missing:
        raise ValueError(f"missing benchmark record: {_identity_text(missing[0])}")
    for cell in omitted_cells:
        present_seeds = {seed for task, variant, seed in identities if (task, variant) == cell}
        if present_seeds == set(matrix.seeds):
            raise ValueError(f"omitted benchmark cell is complete: task={cell[0]}, variant={cell[1]}")

    for task in matrix.tasks:
        environment_counts = {record.num_envs for record in records if record.task == task.name.value}
        if len(environment_counts) != 1:
            raise ValueError(f"{task.name.value} mixes environment counts")
        num_envs = next(iter(environment_counts))
        if num_envs not in matrix.environment_counts:
            raise ValueError(f"{task.name.value} uses unapproved environment count {num_envs}")


def complete_three_seed_records(records: list[RunMetrics]) -> list[RunMetrics]:
    """Return only task/variant groups with all three unique approved seeds."""
    complete: list[RunMetrics] = []
    ordered = sorted(records, key=lambda record: (record.task, record.variant, record.seed))
    for _, grouped in groupby(ordered, key=lambda record: (record.task, record.variant)):
        runs = list(grouped)
        if len(runs) == 3 and len({run.seed for run in runs}) == 3:
            complete.extend(runs)
    return complete


def _validate_command_semantics(manifest_path: Path, matrix: BenchmarkMatrix) -> None:
    """Validate that a self-hashed command has one exact supported experiment shape."""
    manifest = read_manifest(manifest_path)
    identity = manifest.identity
    command = manifest.command
    variant = matrix.variant(identity.variant)
    try:
        if len(command) < 20 or command[0] != "/usr/bin/env":
            raise ValueError("command must start with /usr/bin/env and contain the full benchmark protocol")
        environment = command[1]
        if not environment.startswith("VIRTUAL_ENV="):
            raise ValueError("VIRTUAL_ENV must immediately follow /usr/bin/env")

        cursor = 2
        python_path: str | None = None
        if command[cursor].startswith("PYTHONPATH="):
            python_path = command[cursor].partition("=")[2]
            cursor += 1
        isaaclab_script = Path(command[cursor])
        if isaaclab_script.name != "isaaclab.sh":
            raise ValueError("command must invoke isaaclab.sh")
        repo_root = isaaclab_script.parent
        if (
            Path(environment.partition("=")[2]).resolve()
            != (repo_root / f".venv-{variant.environment.value}").resolve()
        ):
            raise ValueError("VIRTUAL_ENV does not match the variant")
        if python_path is not None:
            expected_tasks = (repo_root / "source" / "isaaclab_tasks").resolve()
            expected_current = (
                (repo_root / "source" / "isaaclab_newton").resolve(),
                expected_tasks,
            )
            resolved_paths = tuple(Path(path).resolve() for path in python_path.split(os.pathsep))
            if resolved_paths not in ((expected_tasks,), expected_current):
                raise ValueError("PYTHONPATH does not select the worktree's exact benchmark sources")
        if command[cursor + 1] != "-p":
            raise ValueError("isaaclab.sh must use -p")
        if Path(command[cursor + 2]).resolve() != (repo_root / "scripts" / "benchmarks" / "training.py").resolve():
            raise ValueError("command must invoke the worktree's benchmark training script")

        expected_tail = [
            "--rl_library",
            "rsl_rl",
            "--task",
            identity.task.value,
            "--num_envs",
            str(identity.num_envs),
            "--seed",
            str(identity.seed),
            "--max_iterations",
            str(identity.max_iterations),
            "--output_path",
            str(manifest_path.parent),
            "--benchmark_formatter",
            "schema",
            "--headless",
            f"presets={variant.preset}",
        ]
        if variant.dynamics_solver is not None:
            expected_tail.extend(
                [
                    f"env.sim.physics.solver_cfg.dynamics_solver={variant.dynamics_solver}",
                    "env.sim.physics.solver_cfg.dynamics_preconditioning=False",
                ]
            )
        if list(command[cursor + 3 :]) != expected_tail:
            raise ValueError("arguments or Hydra overrides do not exactly match the configured protocol")
    except (IndexError, ValueError) as error:
        raise ValueError(f"{manifest_path}: command semantics mismatch: {error}") from error


def _validate_event_integrity(manifest_path: Path, event_path: Path) -> None:
    """Validate an optional future-run TensorBoard event path and hash."""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded_path = data.get("tensorboard_event_path")
    recorded_hash = data.get("tensorboard_event_hash")
    if recorded_path is None and recorded_hash is None:
        return
    if not isinstance(recorded_path, str) or not isinstance(recorded_hash, str):
        raise ValueError(f"{manifest_path}: TensorBoard event integrity fields are incomplete")
    if Path(recorded_path).resolve() != event_path.resolve():
        raise ValueError(f"{manifest_path}: TensorBoard event path does not match the parsed event")
    if sha256_file(event_path) != recorded_hash:
        raise ValueError(f"{manifest_path}: TensorBoard event hash mismatch")


def _validate_manifest(manifest_path: Path, bundle: Path, matrix: BenchmarkMatrix) -> None:
    """Validate a completed manifest's protocol identity and retained provenance."""
    manifest = read_manifest(manifest_path)
    identity = manifest.identity
    expected_cells = {(task.name, variant) for task in matrix.tasks for variant in task.variants}
    if (identity.task, identity.variant) not in expected_cells or identity.seed not in matrix.seeds:
        raise ValueError(f"{manifest_path}: unexpected matrix identity {identity}")
    if identity.phase is not Phase.FULL:
        raise ValueError(f"{manifest_path}: identity phase must be full")
    if identity.max_iterations != matrix.full_iterations:
        raise ValueError(f"{manifest_path}: max_iterations {identity.max_iterations} != {matrix.full_iterations}")
    if identity.num_envs not in matrix.environment_counts:
        raise ValueError(f"{manifest_path}: unapproved environment count {identity.num_envs}")
    if manifest.revisions != matrix.revisions:
        raise ValueError(f"{manifest_path}: revisions do not match the benchmark matrix")
    if manifest.schema_version != "1.1":
        raise ValueError(f"{manifest_path}: schema version is not 1.1")
    if manifest.command_hash != command_hash(manifest.command):
        raise ValueError(f"{manifest_path}: command hash does not match the recorded command")
    _validate_command_semantics(manifest_path, matrix)
    if manifest.run_id != stable_run_id(identity) or manifest_path.parent.name != manifest.run_id:
        raise ValueError(f"{manifest_path}: run identity does not match its directory")
    if Path(manifest.artifact_root).resolve() != manifest_path.parent.resolve():
        raise ValueError(f"{manifest_path}: artifact root does not match its directory")

    for name, expected_hash in manifest.artifact_hashes.items():
        artifact = manifest_path.parent / name
        try:
            artifact.resolve().relative_to(manifest_path.parent.resolve())
        except ValueError as error:
            raise ValueError(f"{manifest_path}: artifact path escapes its run directory") from error
        if not artifact.is_file() or sha256_file(artifact) != expected_hash:
            raise ValueError(f"{manifest_path}: artifact hash mismatch for {name}")
    if bundle.name not in manifest.artifact_hashes:
        raise ValueError(f"{manifest_path}: schema bundle has no recorded artifact hash")


def _validate_failed_manifest(
    manifest_path: Path,
    expected: RunIdentity,
    matrix: BenchmarkMatrix,
    *,
    label: str,
) -> None:
    """Validate one terminal failed run that excuses incomplete matrix coverage."""
    manifest = read_manifest(manifest_path)
    if manifest.state.value != "failed" or manifest.failure_category is None:
        raise ValueError(f"{manifest_path}: expected a terminal failed {label}")
    if manifest.identity != expected:
        raise ValueError(f"{manifest_path}: identity does not match the expected failed {label}")
    if manifest.revisions != matrix.revisions:
        raise ValueError(f"{manifest_path}: revisions do not match the benchmark matrix")
    if manifest.schema_version != "1.1":
        raise ValueError(f"{manifest_path}: schema version is not 1.1")
    if manifest.command_hash != command_hash(manifest.command):
        raise ValueError(f"{manifest_path}: command hash does not match the recorded command")
    _validate_command_semantics(manifest_path, matrix)
    if manifest.run_id != stable_run_id(expected) or manifest_path.parent.name != manifest.run_id:
        raise ValueError(f"{manifest_path}: run identity does not match its directory")
    if Path(manifest.artifact_root).resolve() != manifest_path.parent.resolve():
        raise ValueError(f"{manifest_path}: artifact root does not match its directory")
    if not {"stdout.log", "stderr.log"}.issubset(manifest.artifact_hashes):
        raise ValueError(f"{manifest_path}: failed {label} requires retained stdout and stderr")
    for name, expected_hash in manifest.artifact_hashes.items():
        artifact = manifest_path.parent / name
        try:
            artifact.resolve().relative_to(manifest_path.parent.resolve())
        except ValueError as error:
            raise ValueError(f"{manifest_path}: artifact path escapes its run directory") from error
        if not artifact.is_file() or sha256_file(artifact) != expected_hash:
            raise ValueError(f"{manifest_path}: artifact hash mismatch for {name}")


def validate_failure_omissions(
    records: list[RunMetrics], artifact_root: Path, matrix: BenchmarkMatrix
) -> set[tuple[str, str]]:
    """Return incomplete cells authorized by exact terminal preflight or full-run failures."""
    present = {(record.task, record.variant, record.seed) for record in records}
    omissions: set[tuple[str, str]] = set()
    for task in matrix.tasks:
        for variant in task.variants:
            cell = (task.name.value, variant.value)
            present_seeds = {seed for seed in matrix.seeds if (*cell, seed) in present}
            if present_seeds == set(matrix.seeds):
                continue
            environment_counts = {record.num_envs for record in records if record.task == task.name.value}
            if len(environment_counts) != 1:
                raise ValueError(f"{task.name.value} cannot resolve one expected failure environment count")
            num_envs = next(iter(environment_counts))
            if not present_seeds:
                expected = RunIdentity(
                    task=task.name,
                    variant=variant,
                    seed=matrix.preflight_seed,
                    phase=Phase.PREFLIGHT,
                    num_envs=num_envs,
                    max_iterations=matrix.preflight_iterations,
                )
                manifest_path = artifact_root / stable_run_id(expected) / "manifest.json"
                if not manifest_path.is_file():
                    raise ValueError(f"missing failed preflight for task={cell[0]}, variant={cell[1]}")
                _validate_failed_manifest(manifest_path, expected, matrix, label="preflight")
            else:
                for seed in sorted(set(matrix.seeds) - present_seeds):
                    expected = RunIdentity(
                        task=task.name,
                        variant=variant,
                        seed=seed,
                        phase=Phase.FULL,
                        num_envs=num_envs,
                        max_iterations=matrix.full_iterations,
                    )
                    manifest_path = artifact_root / stable_run_id(expected) / "manifest.json"
                    if not manifest_path.is_file():
                        raise ValueError(f"missing failed full run for task={cell[0]}, variant={cell[1]}, seed={seed}")
                    _validate_failed_manifest(manifest_path, expected, matrix, label="full run")
            omissions.add(cell)
    return omissions


def validate_failed_preflight_omissions(
    records: list[RunMetrics], artifact_root: Path, matrix: BenchmarkMatrix
) -> set[tuple[str, str]]:
    """Return incomplete cells authorized by exact terminal failure evidence.

    This compatibility name is retained for callers of the original preflight-only resolver.
    """
    return validate_failure_omissions(records, artifact_root, matrix)


def load_records(
    artifact_root: Path,
    logs_root: Path,
    task: str | None = None,
    *,
    matrix: BenchmarkMatrix | None = None,
) -> list[RunMetrics]:
    """Load every completed full-run manifest and its matched TensorBoard trace."""
    records: list[RunMetrics] = []
    for manifest_path in sorted(artifact_root.glob("full__*/manifest.json")):
        manifest = read_manifest(manifest_path)
        identity = manifest.identity
        if manifest.state.value != "completed" or (task is not None and identity.task.value != task):
            continue
        bundles = tuple(manifest_path.parent.glob("benchmark_training_*.json"))
        if not bundles:
            raise ValueError(f"{manifest_path.parent} contains no schema bundle")
        bundle = max(bundles, key=lambda path: path.stat().st_mtime)
        if matrix is not None:
            _validate_manifest(manifest_path, bundle, matrix)
        event_path = locate_rsl_rl_events(bundle, logs_root)
        _validate_event_integrity(manifest_path, event_path)
        trace = parse_training_trace(bundle, event_path)
        if (
            trace.task,
            trace.seed,
            trace.num_envs,
            trace.iterations,
        ) != (
            identity.task.value,
            identity.seed,
            identity.num_envs,
            identity.max_iterations,
        ):
            raise ValueError(f"{bundle}: bundle identity does not match its manifest")
        records.append(
            RunMetrics(
                task=trace.task,
                variant=identity.variant.value,
                seed=trace.seed,
                num_envs=trace.num_envs,
                iteration_time_s=trace.iteration_time_s,
                total_fps=trace.total_fps,
                reward=trace.reward,
                ep_length=trace.ep_length,
                success_rate=trace.success_rate,
                success_schema_mismatch=trace.success_schema_mismatch,
                success_schema_mismatch_points=trace.success_schema_mismatch_points,
            )
        )
    return records
