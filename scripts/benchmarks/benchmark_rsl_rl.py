# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated compatibility entry point for RSL-RL training benchmarks."""

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


def _argument_is_set(argv: list[str], argument: str) -> bool:
    """Return whether an argument is present in separated or assigned form."""
    return any(item == argument or item.startswith(f"{argument}=") for item in argv)


def _translate_legacy_args(argv: list[str]) -> list[str]:
    """Translate the legacy benchmark backend and retain all other arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--benchmark_backend",
        default="OmniPerfKPIFile",
        choices=tuple(_BACKEND_FORMATTERS),
    )
    args, forwarded = parser.parse_known_args(argv)
    defaults = {
        "--num_envs": "4096",
        "--seed": "42",
        "--max_iterations": "10",
    }
    for argument, value in defaults.items():
        if not _argument_is_set(forwarded, argument):
            forwarded.extend([argument, value])
    return [*forwarded, "--benchmark_formatter", _BACKEND_FORMATTERS[args.benchmark_backend]]


def main(argv: list[str] | None = None) -> int:
    """Forward the legacy RSL-RL command to the unified training benchmark."""
    if argv is None:
        argv = sys.argv[1:]
    forwarded = _translate_legacy_args(argv)
    print(
        "WARNING: scripts/benchmarks/benchmark_rsl_rl.py is deprecated; use "
        "scripts/benchmarks/training.py --rl_library rsl_rl instead.",
        file=sys.stderr,
    )
    training = importlib.import_module("scripts.benchmarks.training")
    return training.main(["--rl_library", "rsl_rl", *forwarded])


if __name__ == "__main__":
    raise SystemExit(main())
