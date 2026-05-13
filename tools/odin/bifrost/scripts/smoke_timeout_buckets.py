#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Live OSMO smoke for the per-task timeout-bucket dispatcher.

Submits a small mixed-class dispatch (3 short Cartpole + 3 medium Ant)
against the configured pool, verifying that the planner produces two
OSMO workflows with different ``exec_timeout`` values and that all
bundles aggregate correctly.

Manual runbook — not invoked by CI. Requires:

- ``osmo`` CLI on ``$PATH`` and a working profile.
- The image referenced in ``tools/odin/config/bifrost-osmo.yaml`` must
  be reachable from the target pool.

Usage::

    ./isaaclab.sh -p tools/odin/bifrost/scripts/smoke_timeout_buckets.py

Verification checklist (post-run):

1. ``odin_runs/<dispatch_id>/dispatch.json`` shows ``osmo_workflow_ids``
   with two entries.
2. ``osmo workflow query <wf-id>`` for each shows the right
   ``timeout.exec_timeout`` (``30m`` for short, ``2h`` for medium).
3. ``aggregate.json`` reports ``runs == 6`` and ``completed == 6``.
4. Per-run bundles exist under ``odin_runs/<dispatch_id>/<run_id>/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from tools.odin.bifrost import cli as bifrost_cli

_REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    osmo_config = _REPO_ROOT / "tools/odin/config/bifrost-osmo.yaml"
    physx_yaml = _REPO_ROOT / "tools/odin/config/physx_envs.yaml"
    runs_root = _REPO_ROOT / "odin_runs"
    return bifrost_cli.main(
        [
            "--osmo-config",
            str(osmo_config),
            "--physx-yaml",
            str(physx_yaml),
            # Cartpole has timeout_class=short, Ant has timeout_class=medium
            # so the dispatcher emits two workflows with 30m / 2h budgets.
            "--include",
            "Isaac-Cartpole-Direct-v0,Isaac-Ant-Direct-v0",
            "--seeds",
            "42,43,44",
            "--runs-root",
            str(runs_root),
            "--poll-interval",
            "30",
        ]
    )


if __name__ == "__main__":
    sys.exit(main())
