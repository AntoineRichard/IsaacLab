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
from tools.odin.asgard.state import (
    SCHEMA_VERSION,
    DispatchState,
    FleetSnapshot,
    read_dispatch_state,
    reset_in_flight_to_pending,
    write_dispatch_state,
)
from tools.odin.asgard.preflight import PreflightResult, preflight_valkyrie
from tools.odin.asgard.provisioner import ProvisionResult, provision_valkyrie
from tools.odin.asgard.transport import (
    RsyncResult,
    RsyncRunner,
    ShellRsyncRunner,
    ShellSSHRunner,
    SSHResult,
    SSHRunner,
)
from tools.odin.asgard.worker import StateEvent, ValkyrieWorker, WorkerOptions
from tools.odin.asgard.runner import DispatchOptions, resolve_dispatch_dir, run_dispatch

__all__ = [
    "Fleet",
    "ValkyrieConfig",
    "load_fleet",
    "JobEntry",
    "FailureInfo",
    "build_queue_from_env_lists",
    "DispatchState",
    "FleetSnapshot",
    "SCHEMA_VERSION",
    "read_dispatch_state",
    "reset_in_flight_to_pending",
    "write_dispatch_state",
    "RsyncResult",
    "RsyncRunner",
    "ShellRsyncRunner",
    "ShellSSHRunner",
    "SSHResult",
    "SSHRunner",
    "PreflightResult",
    "preflight_valkyrie",
    "ProvisionResult",
    "provision_valkyrie",
    "StateEvent",
    "ValkyrieWorker",
    "WorkerOptions",
    "DispatchOptions",
    "resolve_dispatch_dir",
    "run_dispatch",
]
