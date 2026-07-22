# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dispatch play benchmarks to the selected RL library adapter."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.benchmarks._compat import dispatch_library_entrypoint

LIBRARY_ENTRYPOINTS = {
    "rsl_rl": _SCRIPT_DIR / "rsl_rl" / "benchmark_rsl_rl_play.py",
}


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the selected RL library's play-benchmark adapter."""
    return dispatch_library_entrypoint(
        argv,
        LIBRARY_ENTRYPOINTS,
        action="bench_play",
        description="Benchmark RL inference (play) with a selected reinforcement learning library.",
        library_help="Inference library to benchmark.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
