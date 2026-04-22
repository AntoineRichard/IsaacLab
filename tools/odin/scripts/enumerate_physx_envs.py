# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Enumerate IsaacLab PhysX-capable training environments into a YAML manifest.

Writes ``tools/odin/config/physx_envs.yaml`` (by default) with one row per
registered ``Isaac*`` gym task, grouped by directory-derived type. Preserves
user edits (``keep``, ``framework``, ``num_envs``, ``max_iterations``,
``notes``) on re-run via :func:`tools.odin.common.env_list.merge`.

Usage (from the repo root; PYTHONPATH=. is required so ``tools.odin.*`` is
importable):

    PYTHONPATH=. ./isaaclab.sh -p tools/odin/scripts/enumerate_physx_envs.py \\
        [--output-path PATH] [--dry-run] [--regenerate [--force]]
"""

from __future__ import annotations

"""Launch Isaac Sim simulator first."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

_DEFAULT_OUTPUT = Path("tools/odin/config/physx_envs.yaml")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enumerate IsaacLab PhysX envs into a YAML manifest.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Output YAML path (default: {_DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary, write nothing.",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Discard existing YAML and start fresh. Destructive.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the --regenerate confirmation prompt.",
    )
    return parser.parse_args()


args_cli = _parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import sys

import gymnasium as gym

import isaaclab_tasks  # noqa: F401  (populates the gym registry)

from tools.odin.common.env_list import (
    EnvList,
    build_entry_from_task_spec,
    load_env_list,
    merge,
    write_env_list,
)


def _confirm_regenerate() -> bool:
    if args_cli.force:
        return True
    print(
        f"--regenerate will overwrite {args_cli.output_path}, losing any manual edits. Continue? [y/N]",
        end=" ",
        flush=True,
    )
    response = sys.stdin.readline().strip().lower()
    return response == "y"


def main() -> int:
    output_path: Path = args_cli.output_path

    if args_cli.regenerate:
        if not _confirm_regenerate():
            print("Aborted.")
            return 1
        existing = EnvList()
    else:
        existing = load_env_list(output_path)

    discovered: list = []
    errors = 0
    for task_spec in gym.registry.values():
        if "Isaac" not in task_spec.id:
            continue
        try:
            discovered.append(build_entry_from_task_spec(task_spec))
        except Exception as exc:  # noqa: BLE001 — isolate per-task failure
            errors += 1
            print(
                f"WARNING enum: {task_spec.id}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    merged = merge(existing, discovered)

    # Summary: count per status bucket.
    totals = {"current": 0, "new": 0, "stale": 0}
    frameworkless = 0
    for rows in merged.groups.values():
        for e in rows:
            totals[e.status] = totals.get(e.status, 0) + 1
            if e.framework is None:
                frameworkless += 1

    print(
        f"physx envs: {sum(totals.values())} total "
        f"({totals.get('new', 0)} new, {totals.get('stale', 0)} stale, "
        f"{totals.get('current', 0)} current), "
        f"{frameworkless} frameworkless, {errors} enumeration errors."
    )

    if args_cli.dry_run:
        print("--dry-run: not writing.")
        return 0

    write_env_list(output_path, merged, generator="enumerate_physx_envs.py")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        simulation_app.close()
