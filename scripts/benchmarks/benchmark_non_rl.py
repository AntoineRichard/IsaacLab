# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated compatibility entry point for non-RL benchmarks."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

_BACKEND_FORMATTERS = {
    "LocalLogMetrics": "summary",
    "JSONFileMetrics": "json",
    "OsmoKPIFile": "osmo",
    "OmniPerfKPIFile": "omniperf",
}


def _translate_legacy_args(argv: list[str]) -> list[str]:
    """Translate the legacy benchmark backend and retain all other arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--benchmark_backend",
        default="OmniPerfKPIFile",
        choices=tuple(_BACKEND_FORMATTERS),
    )
    args, forwarded = parser.parse_known_args(argv)
    return [*forwarded, "--benchmark_formatter", _BACKEND_FORMATTERS[args.benchmark_backend]]


def main(argv: list[str] | None = None) -> int:
    """Forward the legacy non-RL command to the unified runtime benchmark."""
    if argv is None:
        argv = sys.argv[1:]
    forwarded = _translate_legacy_args(argv)
    print(
        "WARNING: scripts/benchmarks/benchmark_non_rl.py is deprecated; use scripts/benchmarks/runtime.py instead.",
        file=sys.stderr,
    )
    runtime = importlib.import_module("scripts.benchmarks.runtime")
    if "-h" in forwarded or "--help" in forwarded:
        # AppLauncher validates required parser arguments against ``sys.argv``
        # before the runtime entry point parses its explicit argument list.
        # Supply a temporary task so that validation does not preempt help.
        original_argv = sys.argv
        try:
            sys.argv = [sys.argv[0], "--task", "__help__"]
            runtime.run(forwarded)
        finally:
            sys.argv = original_argv
    else:
        runtime.run(forwarded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
