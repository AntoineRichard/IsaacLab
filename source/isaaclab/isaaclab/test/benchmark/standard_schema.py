# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Odin standard benchmark schema (v1.0).

Defines the on-disk JSON schema shared by IsaacLab's training and startup
benchmark scripts. Producers populate one of the ``*Bundle`` dataclasses and
call :func:`write_bundle_file` to emit schema-compliant JSON. Consumers
(dashboards, validators) read the same file and reconstruct the dataclasses.

The schema is designed so each file is self-contained: every ``*Bundle``
carries its own ``versions`` and ``hardware`` metadata so a reader need not
cross-reference other files in the bundle directory.

Current version: 1.0
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from typing import Any, Literal

SCHEMA_VERSION = "1.0"

Framework = Literal["rsl_rl", "skrl"]
Backend = Literal["physx", "newton"]
RunStatus = Literal["completed", "interrupted", "crashed"]


@dataclass(frozen=True)
class MeanStd:
    """Scalar with mean and standard deviation."""

    mean: float
    std: float


@dataclass(frozen=True)
class MeanStdPeak:
    """Scalar with mean, standard deviation, and peak."""

    mean: float
    std: float
    peak: float


@dataclass(frozen=True)
class GpuDeviceInfo:
    """Information about a single GPU device."""

    name: str
    mem_gb: float
    compute_cap: str


@dataclass(frozen=True)
class Hardware:
    """Host hardware snapshot captured at run time."""

    hostname: str
    gpu_devices: list[GpuDeviceInfo]
    cpu_name: str
    cpu_count: int
    ram_gb: float


@dataclass(frozen=True)
class Versions:
    """Software versions captured at run time.

    Framework-specific fields (``rsl_rl``, ``skrl``) are ``None`` when the
    corresponding framework is not used by the run.
    """

    isaaclab: str
    isaacsim: str | None
    kit: str | None
    newton: str | None
    warp: str | None
    mjwarp: str | None
    torch: str
    rsl_rl: str | None
    skrl: str | None
    git_commit: str | None
    git_branch: str | None
    git_dirty: bool


@dataclass(frozen=True)
class RunIdentity:
    """Identity of a training run."""

    run_id: str
    framework: Framework
    backend: Backend
    task: str
    seed: int
    num_envs: int
    max_iterations: int
    start_time_utc: str
    end_time_utc: str
    duration_s: float
    status: RunStatus


@dataclass(frozen=True)
class StartupPhaseTimes:
    """Wall-clock duration of each startup phase, in seconds."""

    app_launch: float
    env_creation: float
    first_step: float
    python_imports: float | None = None
    task_config: float | None = None


@dataclass(frozen=True)
class Runtime:
    """Aggregated runtime metrics for a training run."""

    startup_phase_times_s: StartupPhaseTimes
    iterations_completed: int
    total_wall_time_s: float
    steps_per_iteration: int
    iteration_time_s: MeanStd
    env_steps_per_s: MeanStd
    iterations_per_s: MeanStd


@dataclass(frozen=True)
class Resources:
    """Aggregated resource utilisation metrics for a training run."""

    gpu_util_pct: MeanStd
    gpu_mem_gb: MeanStdPeak
    cpu_util_pct: MeanStd
    ram_gb: MeanStdPeak


@dataclass(frozen=True)
class LearningCurve:
    """One learning curve (reward or episode length)."""

    final_raw: float
    final_ema: float
    series_per_iter: list[float] | None


@dataclass(frozen=True)
class Learning:
    """Learning curves for a training run, plus their EMA smoothing factor."""

    ema_alpha: float
    reward: LearningCurve
    ep_length: LearningCurve


@dataclass(frozen=True)
class TrainingBundle:
    """Top-level shape of ``training.json``."""

    run: RunIdentity
    versions: Versions
    hardware: Hardware
    runtime: Runtime
    resources: Resources
    learning: Learning
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class CProfileFunction:
    """One entry from a cProfile top-N table."""

    name: str
    own_time_s: float
    cum_time_s: float
    calls: int


@dataclass(frozen=True)
class StartupPhase:
    """Wall-clock total plus top cProfile functions for one startup phase."""

    total_time_s: float
    top_functions: list[CProfileFunction]


@dataclass(frozen=True)
class StartupConfig:
    """CLI configuration captured in a :class:`StartupBundle`."""

    top_n: int
    whitelist: str | None


@dataclass(frozen=True)
class StartupRunIdentity:
    """Startup runs omit ``num_envs`` / ``max_iterations`` (not meaningful)."""

    run_id: str
    framework: Framework
    backend: Backend
    task: str
    seed: int
    start_time_utc: str
    end_time_utc: str
    duration_s: float
    status: RunStatus


@dataclass(frozen=True)
class StartupBundle:
    """Top-level shape of ``startup.json``."""

    run: StartupRunIdentity
    versions: Versions
    hardware: Hardware
    phases: dict[str, StartupPhase]
    config: StartupConfig
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class ManifestPhase:
    """One phase entry in a :class:`Manifest`."""

    file: str
    status: RunStatus | Literal["failed"]
    duration_s: float
    exit_code: int


@dataclass(frozen=True)
class ManifestConfig:
    """Run configuration captured in a :class:`Manifest`."""

    framework: Framework
    backend: Backend
    task: str
    seed: int
    num_envs: int
    max_iterations: int


@dataclass(frozen=True)
class ManifestMachine:
    """Host and repository identity captured in a :class:`Manifest`."""

    hostname: str
    git_commit: str | None
    git_branch: str | None


@dataclass(frozen=True)
class Manifest:
    """Top-level shape of ``manifest.json`` (Odin-side)."""

    run_id: str
    run_start_time_utc: str
    run_end_time_utc: str
    run_duration_s: float
    config: ManifestConfig
    machine: ManifestMachine
    phases: dict[str, ManifestPhase]
    artifacts: list[str]
    schema_version: str = SCHEMA_VERSION


def _to_plain(obj: Any) -> Any:
    """Recursively convert dataclass instances to plain dicts/lists."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, list):
        return [_to_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    return obj


def write_bundle_file(bundle: TrainingBundle | StartupBundle | Manifest, path: str) -> None:
    """Write a bundle dataclass to disk as schema-v1 JSON.

    Creates the parent directory if missing. Uses ``indent=2`` for readability;
    payloads are small (~10 KB training.json, ~50 KB startup.json).

    Args:
        bundle: One of :class:`TrainingBundle`, :class:`StartupBundle`, or :class:`Manifest`.
        path: Absolute or relative path to the output file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_plain(bundle), f, indent=2, sort_keys=False)
        f.write("\n")
