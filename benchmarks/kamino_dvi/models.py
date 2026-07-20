# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Typed records shared by the Kamino DVI benchmark tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class TaskName(StrEnum):
    """Gym task identifiers in the benchmark matrix."""

    CARTPOLE = "Isaac-Cartpole-Direct"
    ANT = "Isaac-Ant-Direct"
    ANYMAL_D = "Isaac-Velocity-Flat-AnymalD"
    DR_LEGS = "Isaac-DrLegs-Walk-v0"
    FOURBAR_POLE = "Isaac-Fourbar-Pole-Swingup"


class Variant(StrEnum):
    """Physics variants compared by the benchmark."""

    KAMINO_CURRENT = "kamino_current"
    KAMINO_PR_PADMM = "kamino_pr_padmm"
    KAMINO_PR_DVI = "kamino_pr_dvi"
    MJWARP = "mjwarp"
    PHYSX = "physx"


class EnvironmentLabel(StrEnum):
    """Locked Python environments used for benchmark execution."""

    CURRENT = "current"
    PR3570 = "pr3570"


class Phase(StrEnum):
    """Execution phases in the benchmark protocol."""

    PREFLIGHT = "preflight"
    FULL = "full"


class TerminalState(StrEnum):
    """Persistent lifecycle states for a benchmark run."""

    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    INVALIDATED = "invalidated"
    FAILED = "failed"


class FailureCategory(StrEnum):
    """Primary failure categories retained by the runner."""

    CAPACITY = "capacity"
    TIMEOUT = "timeout"
    NUMERICAL = "numerical"
    CRASH = "crash"
    INCOMPLETE = "incomplete"
    ARTIFACT = "artifact"


@dataclass(frozen=True)
class Revisions:
    """Immutable source revisions used by the experiment."""

    isaaclab: str
    schema: str
    newton_current: str
    newton_pr: str


@dataclass(frozen=True)
class VariantSpec:
    """Execution environment and configuration for one physics variant."""

    name: Variant
    environment: EnvironmentLabel
    preset: str
    dynamics_solver: str | None = None


@dataclass(frozen=True)
class TaskSpec:
    """One task and the physics variants that apply to it."""

    name: TaskName
    variants: tuple[Variant, ...]


@dataclass(frozen=True)
class BenchmarkMatrix:
    """Validated immutable experiment configuration."""

    revisions: Revisions
    tasks: tuple[TaskSpec, ...]
    variants: tuple[VariantSpec, ...]
    seeds: tuple[int, ...]
    environment_counts: tuple[int, ...]
    preflight_seed: int
    preflight_iterations: int
    full_iterations: int
    preflight_timeout_s: int
    full_timeout_s: int

    def task(self, name: TaskName) -> TaskSpec:
        """Return the specification for a task.

        Args:
            name: Task identifier to find.

        Returns:
            The matching task specification.

        Raises:
            KeyError: If the task is not part of this matrix.
        """
        for task in self.tasks:
            if task.name is name:
                return task
        raise KeyError(name)

    def variant(self, name: Variant) -> VariantSpec:
        """Return the specification for a physics variant.

        Args:
            name: Variant identifier to find.

        Returns:
            The matching variant specification.

        Raises:
            KeyError: If the variant is not part of this matrix.
        """
        for variant in self.variants:
            if variant.name is name:
                return variant
        raise KeyError(name)


@dataclass(frozen=True)
class BenchmarkCell:
    """One applicable task and physics variant pair."""

    task: TaskName
    variant: Variant


@dataclass(frozen=True)
class RunIdentity:
    """One immutable preflight or full training-run identity."""

    task: TaskName
    variant: Variant
    seed: int
    phase: Phase
    num_envs: int
    max_iterations: int


@dataclass(frozen=True)
class RetryLineage:
    """Links a capacity retry to the run that preceded it."""

    attempt: int = 0
    parent_run_id: str | None = None


@dataclass(frozen=True)
class RunManifest:
    """Persistent state shared by runner and analysis tooling."""

    run_id: str
    identity: RunIdentity
    command: tuple[str, ...]
    command_hash: str
    revisions: Revisions
    schema_version: str
    artifact_root: str
    isaaclab_head: str | None = None
    tensorboard_event_path: str | None = None
    tensorboard_event_hash: str | None = None
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    state: TerminalState = TerminalState.PLANNED
    failure_category: FailureCategory | None = None
    retry: RetryLineage = field(default_factory=RetryLineage)
