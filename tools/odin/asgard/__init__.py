# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asgard — Odin's distributed dispatch library.

Public API for dispatching Hugin/Munin jobs across a fleet of Valkyrie
machines (SSH + docker). The CLI :mod:`tools.odin.asgard.cli` is a thin
wrapper over :func:`run_dispatch`; a future T3.2 web UI would consume the
same public surface.
"""

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig, load_fleet
from tools.odin.asgard.queue import FailureInfo, JobEntry, build_queue_from_env_lists

__all__ = [
    "Fleet",
    "ValkyrieConfig",
    "load_fleet",
    "JobEntry",
    "FailureInfo",
    "build_queue_from_env_lists",
]
