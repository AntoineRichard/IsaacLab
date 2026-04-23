# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Queue construction — expand curated env lists across seeds into JobEntry rows."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path

from tools.odin.common.env_list import load_env_list

__all__ = ["JobEntry", "FailureInfo", "build_queue_from_env_lists"]


@dataclass
class FailureInfo:
    """Classified failure attached to a :class:`JobEntry` when ``status == 'failed'``."""

    kind: str  # "infrastructure" | "hugin_crash" | "hugin_malformed_bundle" | "timeout"
    message: str
    details: dict[str, object] = field(default_factory=dict)


@dataclass
class JobEntry:
    """One row in the dispatch queue — the smallest unit of work."""

    run_id: str
    task_id: str
    framework: str  # "rsl_rl" | "skrl"
    backend: str  # "physx" | "newton"
    num_envs: int
    max_iterations: int
    seed: int
    bundle_dir_name: str
    status: str = "pending"  # pending | assigned | running | completed | failed
    assigned_to: str | None = None
    attempts: int = 0
    failure: FailureInfo | None = None
    preferred_not: set[str] = field(default_factory=set)
    started_at: str | None = None
    ended_at: str | None = None


def _framework_slug(framework: str) -> str:
    """rsl_rl -> rsl-rl, skrl -> skrl (hyphen variant used in run_id paths)."""
    return framework.replace("_", "-")


def _make_run_id(framework: str, backend: str, task_id: str, dispatch_id: str, seed: int) -> str:
    return f"{_framework_slug(framework)}_{backend}_{task_id}_{dispatch_id}_seed{seed}"


def _apply_include_filter(task_id: str, include_filter: list[str] | None) -> bool:
    if not include_filter:
        return True
    return any(fnmatch.fnmatch(task_id, pat) for pat in include_filter)


def _expand_env_list(
    yaml_path: Path,
    backend: str,
    seeds: list[int],
    dispatch_id: str,
    include_filter: list[str] | None,
) -> list[JobEntry]:
    env_list = load_env_list(yaml_path)
    jobs: list[JobEntry] = []
    for group_rows in env_list.groups.values():
        for row in group_rows:
            if not row.keep or row.status == "stale":
                continue
            if not _apply_include_filter(row.task_id, include_filter):
                continue
            if row.framework is None or row.num_envs is None or row.max_iterations is None:
                continue
            for seed in seeds:
                run_id = _make_run_id(row.framework, backend, row.task_id, dispatch_id, seed)
                jobs.append(
                    JobEntry(
                        run_id=run_id,
                        task_id=row.task_id,
                        framework=row.framework,
                        backend=backend,
                        num_envs=row.num_envs,
                        max_iterations=row.max_iterations,
                        seed=seed,
                        bundle_dir_name=run_id,
                    )
                )
    return jobs


def build_queue_from_env_lists(
    physx_yaml: Path | None,
    newton_yaml: Path | None,
    seeds: list[int],
    dispatch_id: str,
    include_filter: list[str] | None = None,
) -> list[JobEntry]:
    """Expand curated env YAMLs across seeds into a flat job list.

    Args:
        physx_yaml: Path to ``physx_envs.yaml`` (T2.1); ``None`` to skip PhysX.
        newton_yaml: Path to ``newton_envs.yaml`` (T2.1); ``None`` to skip Newton.
        seeds: Seeds to expand each kept row across. Must be non-empty.
        dispatch_id: UTC timestamp (``YYYYMMDD-HHMMSS``) shared by all run_ids
            in this dispatch.
        include_filter: Optional list of fnmatch patterns on ``task_id``; a row
            must match at least one pattern to be queued. Unset = keep all.

    Returns:
        List of :class:`JobEntry` rows in insertion order (PhysX first, then
        Newton; within a backend, YAML-group order; within a group,
        insertion order; within a row, seed order).

    Raises:
        ValueError: If neither YAML is provided or seeds is empty.
    """
    if physx_yaml is None and newton_yaml is None:
        raise ValueError("build_queue_from_env_lists needs at least one env list (physx_yaml or newton_yaml)")
    if not seeds:
        raise ValueError("build_queue_from_env_lists needs a non-empty seed list")

    jobs: list[JobEntry] = []
    if physx_yaml is not None:
        jobs.extend(_expand_env_list(physx_yaml, "physx", seeds, dispatch_id, include_filter))
    if newton_yaml is not None:
        jobs.extend(_expand_env_list(newton_yaml, "newton", seeds, dispatch_id, include_filter))
    return jobs
