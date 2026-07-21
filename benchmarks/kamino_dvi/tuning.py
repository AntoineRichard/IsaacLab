# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load and resolve the declarative ANYmal-D Kamino DVI tuning matrix."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from .matrix import DEFAULT_MATRIX_PATH, load_matrix
from .models import TaskName, Variant

SolverValue = str | int | float | bool
DEFAULT_TUNING_MATRIX_PATH = Path(__file__).with_name("tuning_matrix.yaml")
HYDRA_PREFIX = "env.sim.physics.solver_cfg."

_BASELINE_KEYS = {
    "integrator",
    "dynamics_linear_solver_max_iterations",
    "dvi_block_iterations",
    "dvi_contact_iterations",
    "dvi_bilateral_solve_period",
    "dvi_omega",
    "dvi_contact_jacobi_omega",
    "dvi_contact_jacobi_relaxation",
    "dynamics_preconditioning",
    "dvi_contact_block_preconditioner",
    "dvi_warmstart_mode",
}


@dataclass(frozen=True)
class TuningCandidate:
    """One named solver configuration change to screen."""

    name: str
    overrides: dict[str, SolverValue]


@dataclass(frozen=True)
class TuningMatrix:
    """Validated immutable configuration for the ANYmal-D tuning campaign."""

    task: str
    variant: Variant
    preset: str
    num_envs: int
    seeds: tuple[int, ...]
    preflight_iterations: int
    screen_iterations: int
    halve_iterations: int
    final_iterations: int
    warmup_iterations: int
    learning_window: int
    baseline: dict[str, SolverValue]
    wave1: tuple[TuningCandidate, ...]

    def candidate(self, name: str) -> TuningCandidate:
        """Return a Wave 1 candidate by name.

        Args:
            name: Candidate name to find.

        Returns:
            The matching tuning candidate.

        Raises:
            KeyError: If no Wave 1 candidate has the requested name.
        """
        for candidate in self.wave1:
            if candidate.name == name:
                return candidate
        raise KeyError(name)


def config_hash(config: Mapping[str, SolverValue]) -> str:
    """Return the canonical SHA-256 hash of a resolved solver configuration.

    Args:
        config: Resolved solver configuration to hash.

    Returns:
        The lowercase hexadecimal SHA-256 digest.
    """
    payload = json.dumps(dict(sorted(config.items())), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def resolve_config(matrix: TuningMatrix, candidate: TuningCandidate) -> dict[str, SolverValue]:
    """Overlay one candidate on the baseline solver configuration.

    Args:
        matrix: Tuning matrix containing the baseline configuration.
        candidate: Candidate overrides to apply.

    Returns:
        A resolved copy of the solver configuration.

    Raises:
        ValueError: If the candidate contains fields outside the baseline schema.
    """
    unknown = candidate.overrides.keys() - matrix.baseline.keys()
    if unknown:
        raise ValueError(f"candidate {candidate.name} has unknown fields: {sorted(unknown)}")
    return {**matrix.baseline, **candidate.overrides}


def _hydra_value(value: SolverValue) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def hydra_overrides(matrix: TuningMatrix, candidate: TuningCandidate) -> tuple[str, ...]:
    """Return deterministic Hydra overrides for a tuning candidate.

    Args:
        matrix: Tuning matrix containing the baseline schema.
        candidate: Candidate overrides to serialize.

    Returns:
        Canonically ordered Hydra override strings.

    Raises:
        ValueError: If the candidate contains fields outside the baseline schema.
    """
    resolve_config(matrix, candidate)
    return tuple(f"{HYDRA_PREFIX}{name}={_hydra_value(value)}" for name, value in sorted(candidate.overrides.items()))


def load_tuning_matrix(path: Path = DEFAULT_TUNING_MATRIX_PATH) -> TuningMatrix:
    """Load and validate the declarative ANYmal-D tuning matrix.

    Args:
        path: Tuning matrix YAML file to load.

    Returns:
        The validated tuning matrix.

    Raises:
        ValueError: If the matrix violates the locked tuning protocol.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("tuning matrix root must be a mapping")

    benchmark_matrix = load_matrix(DEFAULT_MATRIX_PATH)
    if data["task"] != TaskName.ANYMAL_D.value:
        raise ValueError("tuning task must be the ANYmal-D velocity task")
    variant = Variant(data["variant"])
    if variant is not Variant.KAMINO_PR_DVI:
        raise ValueError("tuning variant must be kamino_pr_dvi")
    if data["preset"] != benchmark_matrix.variant(variant).preset:
        raise ValueError("tuning preset must match the locked benchmark matrix")

    num_envs = data["num_envs"]
    if not isinstance(num_envs, int) or isinstance(num_envs, bool):
        raise ValueError("tuning environment count must be an integer")
    if num_envs != 4096:
        raise ValueError("tuning matrix must use exactly 4096 environments")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in data["seeds"]):
        raise ValueError("tuning matrix seeds must be integers")
    seeds = tuple(data["seeds"])
    if seeds != (42, 43, 44):
        raise ValueError("tuning matrix seeds must be exactly (42, 43, 44)")

    iterations = data["iterations"]
    protocol_values = (
        int(iterations["preflight"]),
        int(iterations["screen"]),
        int(iterations["halve"]),
        int(iterations["final"]),
        int(data["warmup_iterations"]),
        int(data["learning_window"]),
    )
    if any(value <= 0 for value in protocol_values):
        raise ValueError("iteration and window values must be positive")

    baseline = dict(data["baseline"])
    missing_baseline_keys = _BASELINE_KEYS - baseline.keys()
    if missing_baseline_keys:
        raise ValueError(f"baseline is missing required keys: {sorted(missing_baseline_keys)}")

    wave1 = tuple(TuningCandidate(name=item["name"], overrides=dict(item["overrides"])) for item in data["wave1"])
    names = tuple(candidate.name for candidate in wave1)
    if len(names) != len(set(names)):
        raise ValueError("duplicate candidate names")
    if len(wave1) != 18:
        raise ValueError("Wave 1 must contain exactly 18 candidates")
    if any(len(candidate.overrides) != 1 for candidate in wave1):
        raise ValueError("each Wave 1 candidate must override exactly one field")

    matrix = TuningMatrix(
        task=data["task"],
        variant=variant,
        preset=data["preset"],
        num_envs=num_envs,
        seeds=seeds,
        preflight_iterations=protocol_values[0],
        screen_iterations=protocol_values[1],
        halve_iterations=protocol_values[2],
        final_iterations=protocol_values[3],
        warmup_iterations=protocol_values[4],
        learning_window=protocol_values[5],
        baseline=baseline,
        wave1=wave1,
    )
    for candidate in matrix.wave1:
        resolve_config(matrix, candidate)
    return matrix
