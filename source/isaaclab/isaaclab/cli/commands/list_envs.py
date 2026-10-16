# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""List registered Isaac Lab environments."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import re
import sys
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import tomllib

from ..registry import DEPENDENCY_GROUP, normalize_name
from ..utils import print_debug, print_warning


def load_task_entry_point(entry_point: importlib.metadata.EntryPoint) -> None:
    """Import one ``isaaclab.tasks`` entry point, warning instead of failing when its package is broken.

    One broken third-party task package must not stop listing, training, or playing every other task. While it
    loads, it cannot replace a task id that is already registered (by Isaac Lab or by another package): gymnasium
    would otherwise let it run its own code under that task's name.

    Args:
        entry_point: Entry point of the ``isaaclab.tasks`` group.
    """
    distribution = entry_point.dist.name if entry_point.dist is not None else "an unknown distribution"
    try:
        with _keep_registered_tasks(distribution):
            entry_point.load()
    except Exception as error:
        print_warning(
            f"Skipped the '{entry_point.name}' task entry point of {distribution}: {type(error).__name__}: {error}"
            " (set DEBUG=1 for the traceback)",
            stream=sys.stderr,
        )
        print_debug(traceback.format_exc())


@contextlib.contextmanager
def _keep_registered_tasks(distribution: str) -> Iterator[None]:
    """Turn ``gymnasium.register`` calls for already registered task ids into warnings while a package loads.

    Args:
        distribution: Name of the distribution being loaded, for the warning.
    """
    try:
        import gymnasium
        from gymnasium.envs import registration
    except ImportError:
        yield
        return
    original = registration.register

    def register(*args: Any, **kwargs: Any) -> None:
        task_id = kwargs.get("id", args[0] if args else None)
        if task_id in registration.registry:
            existing = registration.registry[task_id].entry_point
            print_warning(
                f"Ignored {distribution}'s registration of task '{task_id}', which {existing} already registers.",
                stream=sys.stderr,
            )
            return None
        return original(*args, **kwargs)

    modules = [module for module in (gymnasium, gymnasium.envs, registration) if module.register is original]
    for module in modules:
        module.register = register
    try:
        yield
    finally:
        for module in modules:
            module.register = original


def command_list_envs(args: list[str] | None = None) -> None:
    """List registered environments and their optional presets.

    When invoked from a project that declares an ``isaaclab.tasks`` entry point,
    the command shows that project's tasks by default, including the registry
    packages in its ``isaaclab-registry`` dependency group. Pass ``--keyword`` to use
    an explicit task-id filter or ``--all`` to show every registered task.

    Args:
        args: Command-line arguments excluding the executable name.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument("--keyword", help="Substring used to filter task ids.")
    filter_group.add_argument("--all", action="store_true", help="Show every registered Isaac Lab task.")
    parser.add_argument(
        "--show_presets",
        action="store_true",
        help="Show the physics, renderer, and domain preset combinations that can launch headless.",
    )
    parsed_args = parser.parse_args(args)

    import gymnasium as gym
    from prettytable import PrettyTable

    import isaaclab_tasks  # noqa: F401

    # PLACEHOLDER: Extension template (do not remove this comment)
    with contextlib.suppress(ImportError):
        import isaaclab_tasks_experimental  # noqa: F401

    for entry_point in importlib.metadata.entry_points(group="isaaclab.tasks"):
        load_task_entry_point(entry_point)

    project_modules = () if parsed_args.all or parsed_args.keyword else _find_project_task_modules(Path.cwd())
    task_specs = [
        spec
        for spec in gym.registry.values()
        if "env_cfg_entry_point" in spec.kwargs
        and not spec.kwargs.get("deprecated")
        and (parsed_args.keyword is None or parsed_args.keyword in spec.id)
        and (not project_modules or _belongs_to_project(spec, project_modules))
    ]

    columns = ["S. No.", "Task Name", "Entry Point", "Config", "RL Libraries"]
    if parsed_args.show_presets:
        columns.append("Presets")
    table = PrettyTable(columns)
    table.title = "Available Environments in Isaac Lab"
    for column in columns[1:]:
        table.align[column] = "l"

    if parsed_args.show_presets:
        from isaaclab_tasks.utils.hydra import resolve_task_runs, task_run_overrides

    for index, spec in enumerate(task_specs, start=1):
        libraries = [
            library
            for library in ("rl_games", "rlinf", "rsl_rl", "sb3", "skrl", "torchrl")
            if spec.kwargs.get(f"{library}_cfg_entry_point") is not None
        ]
        row = [index, spec.id, spec.entry_point, spec.kwargs["env_cfg_entry_point"], ", ".join(libraries) or "(none)"]
        if parsed_args.show_presets:
            runs = resolve_task_runs(spec.id)
            if runs is None:
                row.append("(unavailable)")
            else:
                lines = [" ".join(task_run_overrides(run)) or "(defaults)" for run in runs]
                row.append("\n".join([f"{lines[0]} [default]", *lines[1:]]) if lines else "(none)")
        table.add_row(row)
    print(table)


def _find_project_task_modules(start: Path) -> tuple[str, ...]:
    """Return task modules declared by the nearest parent ``pyproject.toml`` and by its registry packages.

    Packages that ``isaaclab registry install`` added to the project's ``isaaclab-registry`` dependency group
    belong to the project, so their installed task modules count as well.
    """
    for directory in (start, *start.parents):
        pyproject_path = directory / "pyproject.toml"
        if not pyproject_path.is_file():
            continue
        with pyproject_path.open("rb") as file:
            data = tomllib.load(file)
        entry_points = data.get("project", {}).get("entry-points", {}).get("isaaclab.tasks", {})
        if not isinstance(entry_points, dict) or not entry_points:
            return ()
        modules = [value for value in entry_points.values() if isinstance(value, str)]
        for requirement in data.get("dependency-groups", {}).get(DEPENDENCY_GROUP, []):
            match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement) if isinstance(requirement, str) else None
            with contextlib.suppress(importlib.metadata.PackageNotFoundError):
                if match:
                    distribution = importlib.metadata.distribution(normalize_name(match.group(0)))
                    modules += [ep.value for ep in distribution.entry_points if ep.group == "isaaclab.tasks"]
        return tuple(modules)
    return ()


def _belongs_to_project(spec: Any, task_modules: tuple[str, ...]) -> bool:
    """Return whether a Gym specification uses one of the project's task modules."""
    references = (spec.entry_point, spec.kwargs.get("env_cfg_entry_point"))
    module_prefixes = tuple(f"{module}." for module in task_modules)
    return any(
        isinstance(reference, str)
        and ((module := reference.partition(":")[0]) in task_modules or module.startswith(module_prefixes))
        for reference in references
    )
