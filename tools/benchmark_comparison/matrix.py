# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parse and expand the deterministic Isaac Lab comparison matrix."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tomllib

from .models import (
    BenchmarkAttempt,
    BenchmarkMatrix,
    BenchmarkMode,
    BenchmarkPair,
    BenchmarkTask,
    Bound,
    BoundUnit,
    MatrixExpansion,
    RunSet,
    Version,
)

FINAL_LOGICAL_PAIR_COUNT = 54
FINAL_ATTEMPT_COUNT = 108
CANARY_LOGICAL_PAIR_COUNT = 18
CANARY_ATTEMPT_COUNT = 36

_MATRIX_PATH = Path(__file__).with_name("matrix.toml")
_FINAL_SEEDS = (42, 43, 44)
_CANARY_SEEDS = (42,)
_TASK_IDENTIFIERS = (
    ("cartpole", "Isaac-Cartpole-v0", "Isaac-Cartpole"),
    ("ant", "Isaac-Ant-v0", "Isaac-Ant"),
    ("anymal_d_flat", "Isaac-Velocity-Flat-Anymal-D-v0", "Isaac-Velocity-Flat-AnymalD"),
    ("g1_flat", "Isaac-Velocity-Flat-G1-v0", "Isaac-Velocity-Flat-G1"),
    ("allegro_cube", "Isaac-Repose-Cube-Allegro-v0", "Isaac-Reorient-Cube-Allegro"),
    ("franka_reach", "Isaac-Reach-Franka-v0", "Isaac-Reach-Franka"),
)
_MODE_IDS = ("runtime-100", "runtime-1000", "training-100")
_VERSION_ORDERS = {
    42: (Version.LAB2, Version.LAB3),
    43: (Version.LAB3, Version.LAB2),
    44: (Version.LAB2, Version.LAB3),
}


def load_matrix(path: Path | None = None) -> BenchmarkMatrix:
    """Load and validate a comparison matrix from TOML.

    Args:
        path: TOML configuration to read. The checked-in matrix is used when omitted.

    Returns:
        An immutable, validated benchmark matrix.
    """
    matrix_path = _MATRIX_PATH if path is None else path
    with matrix_path.open("rb") as file:
        data = tomllib.load(file)

    matrix_data = _as_dict(data.get("matrix"), "matrix")
    tasks = tuple(_parse_task(task_data) for task_data in _as_list(data.get("task"), "task"))
    modes = tuple(_parse_mode(mode_data) for mode_data in _as_list(data.get("mode"), "mode"))
    seeds = tuple(_as_int(seed, "matrix.seeds") for seed in _as_list(matrix_data.get("seeds"), "matrix.seeds"))
    matrix = BenchmarkMatrix(
        tasks=tasks,
        modes=modes,
        seeds=seeds,
        num_envs=_as_int(matrix_data.get("num_envs"), "matrix.num_envs"),
    )
    _validate_matrix(matrix)
    return matrix


def expand_final_matrix(matrix: BenchmarkMatrix) -> MatrixExpansion:
    """Expand every configured task, mode, and repeat into final benchmark attempts."""
    return _expand_matrix(matrix, RunSet.FINAL, _FINAL_SEEDS)


def expand_canary_matrix(matrix: BenchmarkMatrix) -> MatrixExpansion:
    """Expand the bounded seed-42 canary attempts from ``matrix``."""
    return _expand_matrix(matrix, RunSet.CANARY, _CANARY_SEEDS)


def _parse_task(data: Any) -> BenchmarkTask:
    task = _as_dict(data, "task entry")
    return BenchmarkTask(
        alias=_as_str(task.get("alias"), "task.alias"),
        lab2_id=_as_str(task.get("lab2_id"), "task.lab2_id"),
        lab3_id=_as_str(task.get("lab3_id"), "task.lab3_id"),
    )


def _parse_mode(data: Any) -> BenchmarkMode:
    mode = _as_dict(data, "mode entry")
    unit = BoundUnit(_as_str(mode.get("unit"), "mode.unit"))
    return BenchmarkMode(
        id=_as_str(mode.get("id"), "mode.id"),
        framework=_as_str(mode.get("framework"), "mode.framework"),
        final_bound=Bound(_as_int(mode.get("final_bound"), "mode.final_bound"), unit),
        canary_bound=Bound(_as_int(mode.get("canary_bound"), "mode.canary_bound"), unit),
    )


def _expand_matrix(matrix: BenchmarkMatrix, run_set: RunSet, selected_seeds: tuple[int, ...]) -> MatrixExpansion:
    """Expand one run-set selection through the shared deterministic path."""
    _validate_matrix(matrix)
    if any(seed not in matrix.seeds for seed in selected_seeds):
        raise ValueError(f"{run_set.value} seeds are not present in the configured matrix")

    pairs: list[BenchmarkPair] = []
    attempts: list[BenchmarkAttempt] = []
    for seed in selected_seeds:
        repeat_index = matrix.seeds.index(seed)
        versions = _VERSION_ORDERS[seed]
        for task in matrix.tasks:
            for mode in matrix.modes:
                pair_order = len(pairs)
                bound = mode.bound_for(run_set)
                pair_identity = _pair_identity(run_set, task, mode, bound, seed, repeat_index)
                pair_attempts = tuple(
                    _make_attempt(
                        run_set=run_set,
                        task=task,
                        mode=mode,
                        bound=bound,
                        seed=seed,
                        repeat_index=repeat_index,
                        pair_order=pair_order,
                        version=version,
                        version_order=version_order,
                        attempt_order=len(attempts) + version_order,
                        pair_identity=pair_identity,
                        num_envs=matrix.num_envs,
                    )
                    for version_order, version in enumerate(versions)
                )
                pairs.append(
                    BenchmarkPair(
                        identity=pair_identity,
                        run_set=run_set,
                        logical_task=task.alias,
                        mode=mode,
                        bound=bound,
                        seed=seed,
                        repeat_index=repeat_index,
                        pair_order=pair_order,
                        attempts=pair_attempts,
                    )
                )
                attempts.extend(pair_attempts)

    expansion = MatrixExpansion(run_set=run_set, pairs=tuple(pairs), attempts=tuple(attempts))
    _validate_expansion(expansion)
    return expansion


def _make_attempt(
    run_set: RunSet,
    task: BenchmarkTask,
    mode: BenchmarkMode,
    bound: Bound,
    seed: int,
    repeat_index: int,
    pair_order: int,
    version: Version,
    version_order: int,
    attempt_order: int,
    pair_identity: str,
    num_envs: int,
) -> BenchmarkAttempt:
    """Create the deterministic concrete attempt for one version in a pair."""
    identity = f"{pair_identity}--{version.value}--version-order-{version_order}"
    return BenchmarkAttempt(
        identity=identity,
        run_directory=f"{run_set.value}/{identity}",
        logical_pair_identity=pair_identity,
        run_set=run_set,
        logical_task=task.alias,
        concrete_task=task.concrete_id(version),
        mode=mode,
        bound=bound,
        seed=seed,
        repeat_index=repeat_index,
        num_envs=num_envs,
        framework=mode.framework,
        pair_order=pair_order,
        version=version,
        version_order=version_order,
        attempt_order=attempt_order,
    )


def _pair_identity(
    run_set: RunSet,
    task: BenchmarkTask,
    mode: BenchmarkMode,
    bound: Bound,
    seed: int,
    repeat_index: int,
) -> str:
    """Build an identity that keeps final and canary outputs disjoint."""
    return (
        f"{run_set.value}--{task.alias}--{mode.id}--{bound.unit.value}-{bound.value}"
        f"--seed-{seed}--repeat-{repeat_index}--envs-4096--{mode.framework}"
    )


def _validate_matrix(matrix: BenchmarkMatrix) -> None:
    """Reject configurations that cannot produce the required paired experiment."""
    aliases = tuple(task.alias for task in matrix.tasks)
    if len(aliases) != len(set(aliases)):
        raise ValueError("duplicate task alias")
    if len(matrix.tasks) != 6 or len(matrix.modes) != 3:
        raise ValueError("expected 6 tasks and 3 modes")

    for version in Version:
        task_ids = tuple(task.concrete_id(version) for task in matrix.tasks)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("duplicate concrete task ID")
    if tuple((task.alias, task.lab2_id, task.lab3_id) for task in matrix.tasks) != _TASK_IDENTIFIERS:
        raise ValueError("unexpected task aliases or concrete task IDs")

    mode_ids = tuple(mode.id for mode in matrix.modes)
    if len(mode_ids) != len(set(mode_ids)):
        raise ValueError("duplicate mode ID")
    if mode_ids != _MODE_IDS:
        raise ValueError("unexpected mode IDs")
    if matrix.num_envs != 4096:
        raise ValueError("matrix.num_envs must be 4096")
    if matrix.seeds != _FINAL_SEEDS:
        raise ValueError(f"expected {FINAL_LOGICAL_PAIR_COUNT} logical pairs from seeds {_FINAL_SEEDS}")
    if len(matrix.tasks) * len(matrix.modes) * len(matrix.seeds) != FINAL_LOGICAL_PAIR_COUNT:
        raise ValueError(f"expected {FINAL_LOGICAL_PAIR_COUNT} logical pairs")


def _validate_expansion(expansion: MatrixExpansion) -> None:
    """Ensure expanded identities and required run-set counts remain deterministic."""
    expected_pair_count, expected_attempt_count = (
        (FINAL_LOGICAL_PAIR_COUNT, FINAL_ATTEMPT_COUNT)
        if expansion.run_set is RunSet.FINAL
        else (CANARY_LOGICAL_PAIR_COUNT, CANARY_ATTEMPT_COUNT)
    )
    if len(expansion.pairs) != expected_pair_count:
        raise ValueError(f"expected {expected_pair_count} logical pairs")
    if len(expansion.attempts) != expected_attempt_count:
        raise ValueError(f"expected {expected_attempt_count} attempts")

    identities = tuple(attempt.identity for attempt in expansion.attempts)
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate attempt identity")
    run_directories = tuple(attempt.run_directory for attempt in expansion.attempts)
    if len(run_directories) != len(set(run_directories)):
        raise ValueError("duplicate attempt run directory")


def _as_dict(value: Any, name: str) -> dict[str, Any]:
    """Return a TOML table or raise a focused validation error."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a TOML table")
    return value


def _as_list(value: Any, name: str) -> list[Any]:
    """Return a TOML array or raise a focused validation error."""
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a TOML array")
    return value


def _as_str(value: Any, name: str) -> str:
    """Return a TOML string or raise a focused validation error."""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _as_int(value: Any, name: str) -> int:
    """Return a TOML integer or raise a focused validation error."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value
