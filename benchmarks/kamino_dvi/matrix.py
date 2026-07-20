# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load, validate, and expand the Kamino DVI benchmark matrix."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from .models import (
    BenchmarkCell,
    BenchmarkMatrix,
    EnvironmentLabel,
    Phase,
    Revisions,
    RunIdentity,
    TaskName,
    TaskSpec,
    Variant,
    VariantSpec,
)

DEFAULT_MATRIX_PATH = Path(__file__).with_name("matrix.yaml")


def _require_unique(values: Sequence[object], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label}")


def _require_positive(values: Sequence[int], label: str) -> None:
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{label} must contain positive values")


def load_matrix(path: Path = DEFAULT_MATRIX_PATH) -> BenchmarkMatrix:
    """Load and validate a benchmark matrix from YAML.

    Args:
        path: Matrix YAML file to load.

    Returns:
        A validated immutable benchmark matrix.

    Raises:
        ValueError: If a required experiment dimension is missing or invalid.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("matrix root must be a mapping")

    revision_data = data["revisions"]
    revisions = Revisions(**revision_data)
    for label, revision in revision_data.items():
        if not isinstance(revision, str) or len(revision) != 40:
            raise ValueError(f"revision {label} must be a 40-character commit hash")

    variants = tuple(
        VariantSpec(
            name=Variant(item["name"]),
            environment=EnvironmentLabel(item["environment"]),
            preset=item["preset"],
            dynamics_solver=item.get("dynamics_solver"),
        )
        for item in data["variants"]
    )
    _require_unique(tuple(item.name for item in variants), "variants")
    variant_names = {item.name for item in variants}

    tasks = tuple(
        TaskSpec(name=TaskName(item["name"]), variants=tuple(Variant(name) for name in item["variants"]))
        for item in data["tasks"]
    )
    _require_unique(tuple(item.name for item in tasks), "tasks")
    for task in tasks:
        _require_unique(task.variants, f"variants for {task.name}")
        unknown = set(task.variants) - variant_names
        if unknown:
            raise ValueError(f"unknown variants for {task.name}: {sorted(unknown)}")

    seeds = tuple(int(value) for value in data["seeds"])
    environment_counts = tuple(int(value) for value in data["environment_counts"])
    _require_unique(seeds, "seeds")
    _require_unique(environment_counts, "environment counts")
    _require_positive(environment_counts, "environment counts")

    iterations = data["iterations"]
    timeouts = data["timeouts_s"]
    positive_protocol_values = (
        int(iterations["preflight"]),
        int(iterations["full"]),
        int(timeouts["preflight"]),
        int(timeouts["full"]),
    )
    _require_positive(positive_protocol_values, "iterations and timeouts")

    matrix = BenchmarkMatrix(
        revisions=revisions,
        tasks=tasks,
        variants=variants,
        seeds=seeds,
        environment_counts=environment_counts,
        preflight_seed=int(data["preflight_seed"]),
        preflight_iterations=positive_protocol_values[0],
        full_iterations=positive_protocol_values[1],
        preflight_timeout_s=positive_protocol_values[2],
        full_timeout_s=positive_protocol_values[3],
    )
    if matrix.preflight_seed not in matrix.seeds:
        raise ValueError("preflight seed must be one of the full-run seeds")
    if len(expand_cells(matrix)) != 15:
        raise ValueError("approved matrix must contain exactly 15 task/variant cells")
    if len(expand_full_runs(matrix)) != 45:
        raise ValueError("approved matrix must contain exactly 45 full runs")
    return matrix


def expand_cells(matrix: BenchmarkMatrix) -> tuple[BenchmarkCell, ...]:
    """Expand every applicable task/variant pair in declaration order."""
    return tuple(BenchmarkCell(task=task.name, variant=variant) for task in matrix.tasks for variant in task.variants)


def expand_full_runs(
    matrix: BenchmarkMatrix,
    environment_counts: Mapping[TaskName, int] | None = None,
) -> tuple[RunIdentity, ...]:
    """Expand all three-seed full-run identities.

    Args:
        matrix: Validated experiment matrix.
        environment_counts: Optional selected common count per task. The first
            count in the fallback ladder is used when omitted.

    Returns:
        Full-run identities in deterministic task, seed, and variant order.
    """
    selected_counts = environment_counts or {}
    runs: list[RunIdentity] = []
    for task in matrix.tasks:
        num_envs = selected_counts.get(task.name, matrix.environment_counts[0])
        for seed in matrix.seeds:
            for variant in ordered_variants(matrix, task.name, seed):
                runs.append(
                    RunIdentity(
                        task=task.name,
                        variant=variant,
                        seed=seed,
                        phase=Phase.FULL,
                        num_envs=num_envs,
                        max_iterations=matrix.full_iterations,
                    )
                )
    return tuple(runs)


def expand_preflights(matrix: BenchmarkMatrix, num_envs: int | None = None) -> tuple[RunIdentity, ...]:
    """Expand one five-iteration preflight per task/variant cell."""
    selected_count = num_envs if num_envs is not None else matrix.environment_counts[0]
    return tuple(
        RunIdentity(
            task=task.name,
            variant=variant,
            seed=matrix.preflight_seed,
            phase=Phase.PREFLIGHT,
            num_envs=selected_count,
            max_iterations=matrix.preflight_iterations,
        )
        for task in matrix.tasks
        for variant in ordered_variants(matrix, task.name, matrix.preflight_seed)
    )


def ordered_variants(matrix: BenchmarkMatrix, task: TaskName, seed: int) -> tuple[Variant, ...]:
    """Return deterministically counterbalanced variants for one task and seed."""
    if seed not in matrix.seeds:
        raise ValueError(f"seed {seed} is not in the benchmark matrix")
    task_index = next(index for index, item in enumerate(matrix.tasks) if item.name is task)
    base = matrix.task(task).variants
    if task_index % 2:
        base = tuple(reversed(base))
    rotation = matrix.seeds.index(seed) % len(base)
    return base[rotation:] + base[:rotation]
