# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable data models for version-comparison benchmark matrices."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Version(str, Enum):
    """Isaac Lab version participating in a benchmark attempt."""

    LAB2 = "lab2"
    LAB3 = "lab3"


@dataclass(frozen=True)
class ExecutionProvenance:
    """Immutable execution identities validated before measured attempts."""

    lab2_sha: str
    lab3_sha: str
    lab2_image_id: str
    uv_lock_sha256: str

    def version_sha(self, version: Version | str) -> str:
        """Return the expected Git revision for one product version."""
        value = Version(version)
        return self.lab2_sha if value is Version.LAB2 else self.lab3_sha

    def environment_identity(self, version: Version | str) -> str:
        """Return the immutable executor identity for one product version."""
        value = Version(version)
        if value is Version.LAB2:
            return self.lab2_image_id
        return f"uv-lock:{self.uv_lock_sha256}"

    def to_json(self) -> dict[str, str]:
        """Return the deterministic JSON representation used by reporting."""
        return {
            "lab2_sha": self.lab2_sha,
            "lab3_sha": self.lab3_sha,
            "lab2_image_id": self.lab2_image_id,
            "uv_lock_sha256": self.uv_lock_sha256,
        }


class RunSet(str, Enum):
    """Selector for the full final matrix or its bounded canary subset."""

    FINAL = "final"
    CANARY = "canary"


class TaskCategory(str, Enum):
    """Readability group used by benchmark reports."""

    CLASSIC = "classic"
    LOCOMOTION = "locomotion"
    MANIPULATION = "manipulation"


class BoundUnit(str, Enum):
    """Unit used to bound a benchmark mode."""

    STEPS = "steps"
    ITERATIONS = "iterations"


@dataclass(frozen=True)
class Bound:
    """Execution bound for one benchmark mode."""

    value: int
    unit: BoundUnit


@dataclass(frozen=True)
class BenchmarkTask:
    """Logical task alias, version-specific IDs, and execution capabilities."""

    alias: str
    lab2_id: str
    lab3_id: str
    category: TaskCategory
    supported_modes: tuple[str, ...] | None = None
    enable_cameras: bool = False
    lab3_presets: tuple[str, ...] = ()

    def concrete_id(self, version: Version) -> str:
        """Return the configured task identifier for ``version``."""
        if version is Version.LAB2:
            return self.lab2_id
        return self.lab3_id

    def supports_mode(self, mode_id: str) -> bool:
        """Return whether this task participates in ``mode_id``."""
        return self.supported_modes is None or mode_id in self.supported_modes

    def presets_for(self, version: Version) -> tuple[str, ...]:
        """Return task-specific preset additions for ``version``."""
        if version is Version.LAB3:
            return self.lab3_presets
        return ()


@dataclass(frozen=True)
class BenchmarkMode:
    """Benchmark mode and the bounded execution parameters for each run set."""

    id: str
    framework: str
    final_bound: Bound
    canary_bound: Bound

    def bound_for(self, run_set: RunSet) -> Bound:
        """Return the execution bound selected by ``run_set``."""
        if run_set is RunSet.FINAL:
            return self.final_bound
        return self.canary_bound


@dataclass(frozen=True)
class BenchmarkMatrix:
    """Parsed static configuration shared by all comparison controllers."""

    tasks: tuple[BenchmarkTask, ...]
    modes: tuple[BenchmarkMode, ...]
    seeds: tuple[int, ...]
    num_envs: int


@dataclass(frozen=True)
class BenchmarkAttempt:
    """One concrete, version-specific attempt in a logical benchmark pair."""

    identity: str
    run_directory: str
    logical_pair_identity: str
    run_set: RunSet
    logical_task: str
    concrete_task: str
    mode: BenchmarkMode
    bound: Bound
    seed: int
    repeat_index: int
    num_envs: int
    framework: str
    enable_cameras: bool
    extra_presets: tuple[str, ...]
    pair_order: int
    version: Version
    version_order: int
    attempt_order: int


@dataclass(frozen=True)
class BenchmarkPair:
    """Counterbalanced attempts of one logical task, mode, and repeat."""

    identity: str
    run_set: RunSet
    logical_task: str
    mode: BenchmarkMode
    bound: Bound
    seed: int
    repeat_index: int
    pair_order: int
    attempts: tuple[BenchmarkAttempt, BenchmarkAttempt]


@dataclass(frozen=True)
class MatrixExpansion:
    """Deterministic pairs and attempts selected from a benchmark matrix."""

    run_set: RunSet
    pairs: tuple[BenchmarkPair, ...]
    attempts: tuple[BenchmarkAttempt, ...]
