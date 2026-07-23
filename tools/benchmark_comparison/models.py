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


class RunSet(str, Enum):
    """Selector for the full final matrix or its bounded canary subset."""

    FINAL = "final"
    CANARY = "canary"


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
    """Logical task alias and its explicit version-specific task identifiers."""

    alias: str
    lab2_id: str
    lab3_id: str

    def concrete_id(self, version: Version) -> str:
        """Return the configured task identifier for ``version``."""
        if version is Version.LAB2:
            return self.lab2_id
        return self.lab3_id


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
