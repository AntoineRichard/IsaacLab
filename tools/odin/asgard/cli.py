# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""odin-dispatch CLI — thin wrapper over :func:`run_dispatch`.

Invoke from the repo root with ``PYTHONPATH=.``:

    PYTHONPATH=. ./isaaclab.sh -p tools/odin/asgard/cli.py \\
        --fleet fleet.yaml \\
        --physx-yaml tools/odin/config/physx_envs.yaml \\
        --seeds 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.odin.asgard.fleet import load_fleet
from tools.odin.asgard.runner import DispatchOptions, resolve_dispatch_dir, run_dispatch

__all__ = ["main", "parse_args", "parse_seed_list"]


def parse_seed_list(spec: str) -> list[int]:
    """Parse a comma-separated seed spec like "42" or "42,43,44"."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    return [int(p) for p in parts]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="odin-dispatch",
        description="Distributed Hugin/Munin dispatch across Asgard (Odin T3.1).",
    )
    parser.add_argument("--fleet", required=True, type=Path, help="Path to fleet.yaml.")
    parser.add_argument(
        "--physx-yaml",
        type=Path,
        default=None,
        help="Path to curated physx_envs.yaml (T2.1). At least one of --physx-yaml / --newton-yaml required.",
    )
    parser.add_argument("--newton-yaml", type=Path, default=None, help="Path to curated newton_envs.yaml (T2.1).")
    parser.add_argument("--seeds", required=True, help="Comma-separated seed list, e.g. '42' or '42,43,44'.")
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Optional fnmatch patterns on task_id; a row must match at least one to be queued.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume target: 'LATEST' or a specific dispatch_id. Default: start a new dispatch.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("odin_runs"),
        help="Root directory for dispatch bundles (default: ./odin_runs).",
    )
    parser.add_argument("--fresh", action="store_true", help="Wipe remote IsaacLab + restart docker container.")
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Continue when some hosts fail preflight (they are marked 'down').",
    )
    parser.add_argument(
        "--per-job-timeout",
        type=int,
        default=43200,
        help="Per-job wall-clock timeout in seconds (default: 43200 = 12h).",
    )
    parser.add_argument(
        "--max-infrastructure-retries",
        type=int,
        default=2,
        help="Max retries for SSH/docker failures before Hugin starts (default: 2).",
    )
    parser.add_argument(
        "--retry-failed",
        default=None,
        help="Comma-separated list of run_ids (from a prior failed dispatch) to re-attempt on resume.",
    )
    parser.add_argument(
        "--retry-all-failed",
        action="store_true",
        help="On --resume, flip every prior failed job back to pending. Mutually exclusive with --retry-failed.",
    )
    parser.add_argument(
        "--live_retry_poll_s",
        "--live-retry-poll-s",
        dest="live_retry_poll_s",
        type=float,
        default=5.0,
        help="Poll period [s] for live retry queue ingestion (default: 5.0).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-transition status lines as jobs progress.",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Skip the end-of-dispatch call to valhalla.aggregate_dispatch.",
    )
    parser.add_argument(
        "--no-circuit-breaker",
        action="store_true",
        help="Disable per-host consecutive-failure quarantine.",
    )
    parser.add_argument(
        "--no-preflight-recover",
        action="store_true",
        help="Skip auto-restart on NVML wedge during preflight.",
    )
    args = parser.parse_args(argv)

    if args.physx_yaml is None and args.newton_yaml is None:
        parser.error("at least one of --physx-yaml / --newton-yaml is required")

    args.seeds = parse_seed_list(args.seeds)
    if args.retry_failed and args.retry_all_failed:
        parser.error("--retry-failed and --retry-all-failed are mutually exclusive")
    if args.retry_failed:
        args.retry_failed = [s.strip() for s in args.retry_failed.split(",") if s.strip()]
    if args.live_retry_poll_s <= 0:
        parser.error("--live_retry_poll_s must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    fleet = load_fleet(args.fleet)
    dispatch_dir = resolve_dispatch_dir(args.runs_root, resume=args.resume)

    options = DispatchOptions(
        seeds=args.seeds,
        max_infrastructure_retries=args.max_infrastructure_retries,
        per_job_timeout_s=args.per_job_timeout,
        fresh=args.fresh,
        skip_preflight=args.skip_preflight,
        include_filter=args.include,
        verbose=args.verbose,
        retry_failed=args.retry_failed,
        retry_all_failed=args.retry_all_failed,
        skip_aggregate=args.skip_aggregate,
        consecutive_failure_quarantine=0 if args.no_circuit_breaker else 3,
        preflight_auto_restart=not args.no_preflight_recover,
        live_retry_poll_s=args.live_retry_poll_s,
    )

    print(f"odin-dispatch: dispatch_id={dispatch_dir.name} fleet={fleet.fleet_name} hosts={len(fleet.hosts)}")
    state = run_dispatch(
        fleet=fleet,
        physx_yaml=args.physx_yaml,
        newton_yaml=args.newton_yaml,
        dispatch_dir=dispatch_dir,
        options=options,
    )

    total = len(state.jobs)
    completed = sum(1 for j in state.jobs if j.status == "completed")
    failed = sum(1 for j in state.jobs if j.status == "failed")
    pending = sum(1 for j in state.jobs if j.status == "pending")
    failed_by_kind: dict[str, int] = {}
    for j in state.jobs:
        if j.status == "failed" and j.failure is not None:
            failed_by_kind[j.failure.kind] = failed_by_kind.get(j.failure.kind, 0) + 1
    summary = f"{completed} completed, {failed} failed"
    if failed_by_kind:
        summary += " (" + ", ".join(f"{n} {k}" for k, n in sorted(failed_by_kind.items())) + ")"
    summary += f", {pending} pending"
    print(f"odin-dispatch: {summary} out of {total} total")
    return 0 if failed == 0 and pending == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
