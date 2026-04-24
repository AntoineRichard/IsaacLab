# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""odin-bootstrap — bring a fresh fleet to T3.1-preflight-ready state.

Usage::

    PYTHONPATH=. ./isaaclab.sh -p tools/odin/asgard/bootstrap_cli.py \\
        --fleet fleet.yaml \\
        [--build-timeout 1800] \\
        [--sequential] \\
        [--verbose]
"""

from __future__ import annotations

# When Python runs this file as a script (``./isaaclab.sh -p tools/odin/asgard/bootstrap_cli.py``),
# it prepends this file's directory to ``sys.path[0]``. That directory also
# contains ``queue.py`` (our :mod:`tools.odin.asgard.queue` job-queue module),
# which then shadows the stdlib ``queue`` — and ``concurrent.futures.thread``
# imports ``queue`` to use ``queue.SimpleQueue``. De-prepend the script dir
# before any other import so stdlib ``queue`` resolves correctly. A proper
# fix is to rename ``queue.py`` (tracked as a latent cleanup item).
import os as _os
import sys as _sys

_script_dir = _os.path.dirname(_os.path.abspath(__file__))
if _sys.path and _os.path.abspath(_sys.path[0]) == _script_dir:
    _sys.path.pop(0)

import argparse
import sys
from pathlib import Path

from tools.odin.asgard.bootstrap import bootstrap_fleet
from tools.odin.asgard.fleet import load_fleet
from tools.odin.asgard.transport import ShellRsyncRunner, ShellSSHRunner

__all__ = ["main", "parse_args"]


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the bootstrap CLI args. Factored out for unit testing."""
    parser = argparse.ArgumentParser(
        prog="odin-bootstrap",
        description=(
            "Bring a fresh Odin fleet to T3.1-preflight-ready state: wipe + "
            "rsync the working tree + start the isaac-lab-base container on "
            "every host. Idempotent by design (always wipe + re-rsync)."
        ),
    )
    parser.add_argument("--fleet", required=True, type=Path, help="Path to fleet.yaml.")
    parser.add_argument(
        "--build-timeout",
        type=int,
        default=1800,
        help=(
            "Per-host timeout [s] for `./docker/container.py start`. "
            "Default 1800 (30 min) covers a first-time docker image build."
        ),
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help=(
            "Bootstrap hosts one at a time instead of the default parallel "
            "(one thread per host). Useful when shared network bandwidth "
            "would be saturated by simultaneous rsyncs."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print a per-host ok/fail summary line as each host finishes.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Bootstrap the fleet; return 0 iff every host reported ok."""
    args = parse_args(argv if argv is not None else sys.argv[1:])
    fleet = load_fleet(args.fleet)

    results = bootstrap_fleet(
        fleet,
        Path.cwd(),
        ssh=ShellSSHRunner(),
        rsync=ShellRsyncRunner(),
        build_timeout_s=args.build_timeout,
        parallel=not args.sequential,
        verbose=args.verbose,
    )

    ok_count = sum(1 for r in results if r.ok)
    total = len(results)
    print(f"bootstrap complete: {ok_count}/{total} hosts ok")
    if ok_count < total:
        for r in results:
            if not r.ok:
                print(f"  {r.host}: {r.message}")
    return 0 if ok_count == total else 1


if __name__ == "__main__":
    sys.exit(main())
