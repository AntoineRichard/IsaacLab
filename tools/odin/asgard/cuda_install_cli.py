# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""odin-cuda CLI — fleet-wide CUDA detection + driver/toolkit upgrade.

Usage::

    PYTHONPATH=. ./isaaclab.sh -p tools/odin/asgard/cuda_install_cli.py check \\
        --fleet fleet.yaml [--floor 12.4] [--verbose]

    PYTHONPATH=. ./isaaclab.sh -p tools/odin/asgard/cuda_install_cli.py install \\
        --fleet fleet.yaml [--floor 12.4] [--target 12.9] \\
        [--sequential] [--yes] [--force] [--reboot-timeout 600] \\
        [--runs-root ./odin_runs] [--verbose]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.odin.asgard.cuda_install import (
    check_fleet,
)
from tools.odin.asgard.fleet import load_fleet
from tools.odin.asgard.transport import ShellSSHRunner

__all__ = ["main", "parse_args"]


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the odin-cuda CLI args. Factored out for unit testing."""
    parser = argparse.ArgumentParser(
        prog="odin-cuda",
        description=(
            "Fleet-wide CUDA detection and driver/toolkit upgrade for Odin "
            "Valkyries. 'check' is read-only; 'install' reboots hosts."
        ),
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    check = sub.add_parser("check", help="Read-only CUDA check across the fleet.")
    check.add_argument("--fleet", required=True, type=Path, help="Path to fleet.yaml.")
    check.add_argument("--floor", default="12.4", help="CUDA floor (default: 12.4).")
    check.add_argument("--verbose", action="store_true")

    install = sub.add_parser(
        "install",
        help="Detect + upgrade hosts below floor (apt + reboot).",
    )
    install.add_argument("--fleet", required=True, type=Path, help="Path to fleet.yaml.")
    install.add_argument("--floor", default="12.4", help="CUDA floor (default: 12.4).")
    install.add_argument(
        "--target",
        default="12.9",
        help="Apt meta-package version key (default: 12.9 -> cuda-12-9).",
    )
    install.add_argument(
        "--sequential",
        action="store_true",
        help="Upgrade hosts one at a time instead of in parallel.",
    )
    install.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt (for CI).",
    )
    install.add_argument(
        "--force",
        action="store_true",
        help="Override the active-dispatch guard.",
    )
    install.add_argument(
        "--reboot-timeout",
        type=int,
        default=600,
        help="Per-host SSH-reachability wait after reboot (default: 600 s).",
    )
    install.add_argument(
        "--runs-root",
        type=Path,
        default=Path("odin_runs"),
        help="Root scanned for in-flight dispatches (default: ./odin_runs).",
    )
    install.add_argument("--verbose", action="store_true")

    return parser.parse_args(argv)


def _print_check_table(results) -> None:
    """Print per-host CUDA status as a fixed-width table."""
    width = max(len(r.host) for r in results) if results else 4
    header = f"{'host':<{width}}  {'driver':<14}  {'cuda':<6}  status"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.host:<{width}}  {(r.driver or '-'):<14}  {(r.cuda or '-'):<6}  "
            f"{r.status}{('  — ' + r.message) if r.message else ''}"
        )


def _run_check(args: argparse.Namespace) -> int:
    fleet = load_fleet(args.fleet)
    results = check_fleet(fleet, ssh=ShellSSHRunner(), floor=args.floor, parallel=True)
    _print_check_table(results)
    return 0 if all(r.status == "ok" for r in results) else 1


def main(argv: list[str] | None = None) -> int:
    """Run the odin-cuda CLI; return 0 on success."""
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if args.subcommand == "check":
        return _run_check(args)
    # install handler arrives in Task 9.
    raise NotImplementedError(args.subcommand)


if __name__ == "__main__":
    sys.exit(main())
