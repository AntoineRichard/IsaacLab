# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Derive Newton env list and gap-candidates from a filtered PhysX list.

Reads ``tools/odin/config/physx_envs.yaml`` (filtered by the user), visits
every ``keep: true`` row, and partitions them by Newton preset presence:

- Rows whose raw env cfg has a ``newton`` preset → ``newton_envs.yaml``.
- Rows without a ``newton`` preset → ``newton_gap_candidates.yaml`` with
  ``suspected_gap: "tbd"`` for the user to categorize.

Both outputs merge with existing files so prior categorization survives
re-runs.

Usage (from the repo root; PYTHONPATH=. is required so ``tools.odin.*`` is
importable):

    PYTHONPATH=. ./isaaclab.sh -p tools/odin/scripts/enumerate_newton_envs.py \\
        [--physx-input PATH] [--newton-output PATH] [--gap-output PATH] \\
        [--dry-run] [--regenerate [--force]]
"""

from __future__ import annotations

"""Launch Isaac Sim simulator first."""

import argparse
import copy
from pathlib import Path

from isaaclab.app import AppLauncher

_DEFAULT_PHYSX_INPUT = Path("tools/odin/config/physx_envs.yaml")
_DEFAULT_NEWTON_OUTPUT = Path("tools/odin/config/newton_envs.yaml")
_DEFAULT_GAP_OUTPUT = Path("tools/odin/config/newton_gap_candidates.yaml")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enumerate Newton-supported IsaacLab envs and gap candidates.",
    )
    parser.add_argument("--physx-input", type=Path, default=_DEFAULT_PHYSX_INPUT)
    parser.add_argument("--newton-output", type=Path, default=_DEFAULT_NEWTON_OUTPUT)
    parser.add_argument("--gap-output", type=Path, default=_DEFAULT_GAP_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


args_cli = _parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import sys

import isaaclab_tasks  # noqa: F401  (populates the gym registry)
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from tools.odin.common.env_list import (
    EnvList,
    classify_for_newton,
    load_env_list,
    merge,
    write_env_list,
)


def _confirm_regenerate() -> bool:
    if args_cli.force:
        return True
    print(
        f"--regenerate will overwrite {args_cli.newton_output} and "
        f"{args_cli.gap_output}, losing any manual edits. Continue? [y/N]",
        end=" ",
        flush=True,
    )
    return sys.stdin.readline().strip().lower() == "y"


def main() -> int:
    physx_path: Path = args_cli.physx_input
    physx = load_env_list(physx_path)
    if not physx.groups:
        print(
            f"No PhysX env list at {physx_path}. Run "
            f"tools/odin/scripts/enumerate_physx_envs.py first.",
            file=sys.stderr,
        )
        return 1

    kept = [
        e
        for rows in physx.groups.values()
        for e in rows
        if e.keep and e.status != "stale"
    ]
    print(f"PhysX input: {sum(len(v) for v in physx.groups.values())} total, {len(kept)} kept.")

    if args_cli.regenerate:
        if not _confirm_regenerate():
            print("Aborted.")
            return 1
        existing_newton = EnvList()
        existing_gaps = EnvList()
    else:
        existing_newton = load_env_list(args_cli.newton_output)
        existing_gaps = load_env_list(args_cli.gap_output)

    newton_discovered: list = []
    gap_discovered: list = []
    errors = 0
    for e in kept:
        try:
            raw_cfg = load_cfg_from_registry(e.task_id, "env_cfg_entry_point")
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(
                f"WARNING enum: {e.task_id}: cfg load failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue

        verdict = classify_for_newton(raw_cfg)
        if verdict == "supported":
            newton_discovered.append(copy.deepcopy(e))
        else:
            gap_entry = copy.deepcopy(e)
            gap_entry.suspected_gap = "tbd"
            gap_discovered.append(gap_entry)

    newton_merged = merge(existing_newton, newton_discovered)
    gaps_merged = merge(existing_gaps, gap_discovered)

    print(
        f"newton envs:   {sum(len(v) for v in newton_merged.groups.values())} "
        f"({len(newton_discovered)} from this run)"
    )
    print(
        f"gap candidates:{sum(len(v) for v in gaps_merged.groups.values())} "
        f"({len(gap_discovered)} from this run)"
    )
    print(f"load errors:   {errors}")

    if args_cli.dry_run:
        print("--dry-run: not writing.")
        return 0

    write_env_list(args_cli.newton_output, newton_merged,
                   generator="enumerate_newton_envs.py")
    write_env_list(args_cli.gap_output, gaps_merged,
                   generator="enumerate_newton_envs.py")
    print(f"Wrote {args_cli.newton_output}")
    print(f"Wrote {args_cli.gap_output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        simulation_app.close()
