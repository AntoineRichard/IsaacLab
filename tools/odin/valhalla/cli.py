# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""odin-aggregate — manual entry point for Valhalla aggregation.

Usage::

    PYTHONPATH=. ./isaaclab.sh -p tools/odin/valhalla/cli.py <dispatch_id|LATEST> \\
        [--runs-root odin_runs/] \\
        [--divergence-z 2.0] \\
        [--no-overwrite] \\
        [--quiet]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.odin.valhalla.aggregator import AggregateOptions, aggregate_dispatch
from tools.odin.valhalla.writer import write_aggregate


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI args. Factored out so unit tests can exercise just the argparse layer."""
    parser = argparse.ArgumentParser(
        prog="odin-aggregate",
        description="Roll an odin_runs/<dispatch_id>/ directory into a single aggregate.json.",
    )
    parser.add_argument(
        "dispatch_id",
        help="Dispatch id (matches odin_runs/<id>/), or LATEST to auto-pick the newest.",
    )
    parser.add_argument("--runs-root", type=Path, default=Path("odin_runs"))
    parser.add_argument("--divergence-z", type=float, default=2.0)
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        default=True,
        help="Refuse to overwrite an existing aggregate.json (default: overwrite).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress the summary line.")
    return parser.parse_args(argv)


def resolve_dispatch_dir(runs_root: Path, dispatch_id: str) -> Path:
    """Resolve ``dispatch_id`` (or ``LATEST``) to a concrete directory under ``runs_root``.

    Raises:
        FileNotFoundError: If ``LATEST`` is used but ``runs_root`` holds no
            subdirectories, or if ``dispatch_id`` names a non-existent directory.
    """
    if dispatch_id == "LATEST":
        children = sorted(p for p in runs_root.iterdir() if p.is_dir())
        if not children:
            raise FileNotFoundError(f"No prior dispatch directories under {runs_root}")
        return children[-1]
    candidate = runs_root / dispatch_id
    if not candidate.exists():
        raise FileNotFoundError(f"{candidate} does not exist")
    return candidate


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        ``0`` on success, non-zero on argparse or resolution errors.
    """
    args = parse_args(argv if argv is not None else [])
    dispatch_dir = resolve_dispatch_dir(args.runs_root, args.dispatch_id)
    agg = aggregate_dispatch(
        dispatch_dir,
        options=AggregateOptions(divergence_z=args.divergence_z),
    )
    path = write_aggregate(dispatch_dir, agg, overwrite=args.overwrite)
    if not args.quiet:
        totals = agg["totals"]
        print(
            f"Wrote {path}: {totals['tasks']} tasks, {totals['runs']} runs, "
            f"{totals['completed']} completed, {totals['failed']} failed."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
