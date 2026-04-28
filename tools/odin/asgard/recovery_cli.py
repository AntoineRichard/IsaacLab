# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""odin-recover — ad-hoc GPU-loss recovery wrapper around :func:`recover_valkyrie_gpu`.

Invoke from the repo root with ``PYTHONPATH=.``:

    PYTHONPATH=. ./isaaclab.sh -p tools/odin/asgard/recovery_cli.py \\
        --fleet fleet.yaml \\
        --host 10.176.214.169

Exits 0 iff the host's container restarts and ``nvidia-smi -L`` lists at
least one GPU. Exits 1 otherwise. Exits 2 if the host is not in the
fleet.yaml.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.odin.asgard.fleet import load_fleet
from tools.odin.asgard.recovery import recover_valkyrie_gpu
from tools.odin.asgard.transport import ShellSSHRunner

__all__ = ["main", "parse_args"]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="odin-recover",
        description="Restart a Valkyrie's container and verify the GPU is visible.",
    )
    parser.add_argument("--fleet", required=True, type=Path, help="Path to fleet.yaml.")
    parser.add_argument("--host", required=True, help="Host address as it appears in fleet.yaml.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv if argv is not None else sys.argv[1:])
    fleet = load_fleet(ns.fleet)
    matches = [h for h in fleet.hosts if h.host == ns.host]
    if not matches:
        print(f"odin-recover: host {ns.host!r} not found in {ns.fleet}", file=sys.stderr)
        sys.exit(2)
    host = matches[0]
    result = recover_valkyrie_gpu(host, ssh=ShellSSHRunner())
    print(f"odin-recover: host={host.host} container={host.container_name} ", end="")
    print(f"recovered={result.recovered} duration_s={result.duration_s:.1f} message={result.message}")
    return 0 if result.recovered else 1


if __name__ == "__main__":
    sys.exit(main())
