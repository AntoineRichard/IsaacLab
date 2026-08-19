# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the dispatch task list from the Gym registry.

Discovery is the default row source; a hand-written list remains possible as an
override via ``--tasks_yaml``. Expansion is total and filtering is the only way
to narrow, so there is one vocabulary rather than two.

The registry walk and preset resolution live in :mod:`tools.task_discovery`, which
Isaac Lab also uses to generate its environment tables. This module only applies
Odin's dispatch policy on top and reshapes the result into rows; everything here
is pure and testable offline.

The emitted file is the same shape :func:`~tools.odin.plan.load_task_rows`
already consumes, so nothing downstream of ``PlannedRow`` changes.

Sizing is deliberately never invented here. ``num_envs``, ``max_iterations`` and
``timeout_s`` come from the harvested ``task_metadata.yaml`` overlay, measured
from a real run.
"""

from __future__ import annotations

import dataclasses
import fnmatch
from pathlib import Path
from typing import Any

import yaml

from tools.task_discovery import DiscoveredTask, DiscoveryError
from tools.task_discovery import discover_tasks as _discover_all_tasks

__all__ = [
    "RL_LIBRARY_PRIORITY",
    "DiscoveredTask",
    "DiscoveryError",
    "discover_tasks",
    "filter_rows",
    "expand_rows",
    "write_task_list",
]

# Stable ordering for the RL library axis.
RL_LIBRARY_PRIORITY: tuple[str, ...] = ("rsl_rl", "rl_games", "skrl", "sb3")

# Physics presets never dispatched. ``newton_mjwarp_vbd_proxy`` is a proxy
# variant rather than a backend under test.
_SKIP_PHYSICS = frozenset({"newton_mjwarp_vbd_proxy"})


def expand_rows(tasks: list[DiscoveredTask]) -> list[dict[str, Any]]:
    """Expand discovered tasks into every dispatchable row.

    Expansion is total across the RL library axis and the task's legal modes.
    Narrowing is :func:`filter_rows`' job, so there is one mechanism for "what to
    keep" rather than a second vocabulary of named row-set shapes.

    Args:
        tasks: Discovered tasks.

    Returns:
        Row dicts sorted by ``(task_id, rl_library, physics, renderer)``. The
        ``physics`` and ``renderer`` keys are omitted when the mode leaves them
        unset, because an empty ``physics=`` token is rejected by the task.
    """
    rows: list[dict[str, Any]] = []
    for task in tasks:
        for library in task.rl_libraries:
            for mode in task.modes:
                row: dict[str, Any] = {
                    "task_id": task.task_id,
                    "scope": task.scope,
                    "rl_library": library,
                }
                if mode.physics is not None:
                    row["physics"] = mode.physics
                if mode.renderer is not None:
                    row["renderer"] = mode.renderer
                if mode.presets is not None:
                    row["presets"] = mode.presets
                rows.append(row)

    rows.sort(key=lambda row: (row["task_id"], row["rl_library"], row.get("physics") or "", row.get("renderer") or ""))
    return rows


def filter_rows(
    rows: list[dict[str, Any]],
    *,
    include: str | None = None,
    exclude: str | None = None,
    libraries: list[str] | None = None,
    physics: list[str] | None = None,
    renderers: list[str] | None = None,
    presets: list[str] | None = None,
    scope: str | None = None,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Narrow expanded rows.

    Args:
        rows: Rows from :func:`expand_rows`.
        include: Glob a ``task_id`` must match.
        exclude: Glob a ``task_id`` must not match.
        libraries: Restrict to these RL libraries.
        physics: Restrict to these physics presets. Rows with no physics token
            are kept only when ``"default"`` is listed.
        renderers: Restrict to these renderer presets. Rows are headless unless
            a renderer was requested, so ``"none"`` keeps headless rows.
        presets: Restrict to these domain presets. ``"default"`` keeps rows that
            select none.
        scope: ``core``, ``contrib``, or ``all``.
        max_rows: Deterministic head of the sorted order, as a cost valve.

    Returns:
        The filtered rows, order preserved.
    """
    result = rows
    if include is not None:
        result = [row for row in result if fnmatch.fnmatch(row["task_id"], include)]
    if exclude is not None:
        result = [row for row in result if not fnmatch.fnmatch(row["task_id"], exclude)]
    if libraries:
        result = [row for row in result if row["rl_library"] in libraries]
    if physics:
        wanted = set(physics)
        result = [row for row in result if (row.get("physics") or "default") in wanted]
    if renderers:
        wanted_renderers = set(renderers)
        result = [row for row in result if (row.get("renderer") or "none") in wanted_renderers]
    if presets:
        wanted_presets = set(presets)
        result = [row for row in result if (row.get("presets") or "default") in wanted_presets]
    if scope and scope != "all":
        result = [row for row in result if row.get("scope") == scope]
    if max_rows is not None:
        result = result[:max_rows]
    return result


def discover_tasks() -> list[DiscoveredTask]:
    """Return every task Odin can dispatch, with its legal modes.

    Thin policy layer over :func:`tools.task_discovery.discover_tasks`, which owns
    the registry walk and the runtime validation. Two narrowings are applied here
    because they are dispatch policy rather than facts about a task:

    - modes whose physics preset Odin never runs are dropped (see
      :data:`_SKIP_PHYSICS`);
    - RL libraries Odin has no benchmark entrypoint for are dropped, so a task
      carrying only those disappears rather than expanding into rows that cannot run.

    ``collapse`` is left on, so spellings resolving to the same run arrive already
    reduced -- that is what makes ``physx`` and ``ovphysx`` one row rather than two
    identical ones.

    Returns:
        Dispatchable tasks sorted by ``task_id``, each with at least one mode and
        one RL library.

    Raises:
        DiscoveryError: If the Isaac Lab task packages cannot be imported.
    """
    supported = set(RL_LIBRARY_PRIORITY)
    dispatchable: list[DiscoveredTask] = []
    for task in _discover_all_tasks(resolve=True, collapse=True):
        libraries = tuple(lib for lib in task.rl_libraries if lib in supported)
        modes = tuple(mode for mode in task.modes if mode.physics not in _SKIP_PHYSICS)
        if not libraries or not modes:
            continue
        dispatchable.append(dataclasses.replace(task, rl_libraries=libraries, modes=modes))
    return dispatchable


def write_task_list(path: Path, rows: list[dict[str, Any]], *, meta: dict[str, Any]) -> None:
    """Write discovered rows as a task list.

    Args:
        path: Destination file; parent directories are created.
        rows: Rows to write.
        meta: Provenance recorded under a ``discovery`` header, which
            :func:`~tools.odin.plan.load_task_rows` ignores. Keep it free of
            values that change per run, or every regeneration is a diff.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Generated by tools/odin/discover.py. Do not edit by hand -- regenerate,\n"
        "# or pass a hand-written file to `dispatch --tasks_yaml` to override.\n"
        "#\n"
        "# Sizing is absent by design: it is harvested from a real run into\n"
        "# task_metadata.yaml and applied as an overlay.\n"
    )
    path.write_text(header + yaml.safe_dump({"discovery": meta, "tasks": rows}, sort_keys=False))
