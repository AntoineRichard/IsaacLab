# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Container-level GPU-loss recovery for a single Valkyrie.

Used by :class:`tools.odin.asgard.worker.ValkyrieWorker` mid-dispatch and by
:mod:`tools.odin.asgard.recovery_cli` for ad-hoc operator recovery. Restarts
the named container, waits for it to reach ``running`` state, then probes
``nvidia-smi -L`` from inside it as the acceptance test.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from tools.odin.asgard.fleet import ValkyrieConfig
from tools.odin.asgard.transport import SSHRunner

__all__ = ["RecoveryResult", "recover_valkyrie_gpu"]


_DOCKER_RESTART_TIMEOUT_S = 60.0
_CONTAINER_UP_POLL_INTERVAL_S = 2.0
_CONTAINER_UP_MAX_POLLS = 15  # 15 * 2 s = 30 s budget
_GPU_PROBE_TIMEOUT_S = 15.0


@dataclass
class RecoveryResult:
    """Outcome of one ``recover_valkyrie_gpu`` invocation."""

    host: str
    container_name: str
    attempted: bool
    recovered: bool
    duration_s: float
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def recover_valkyrie_gpu(host: ValkyrieConfig, *, ssh: SSHRunner) -> RecoveryResult:
    """Restart container, wait for ``running``, probe ``nvidia-smi -L``.

    Returns a :class:`RecoveryResult` whose ``recovered`` is True iff every
    phase succeeded. On failure, ``message`` is one of the well-known
    short strings used by the worker for telemetry.
    """
    started = time.monotonic()
    details: dict[str, Any] = {}

    # Phase 1: docker restart
    r = ssh.run(host, f"docker restart {host.container_name}", timeout_s=_DOCKER_RESTART_TIMEOUT_S)
    if r.exit_code == 255:
        return RecoveryResult(
            host=host.host,
            container_name=host.container_name,
            attempted=True,
            recovered=False,
            duration_s=time.monotonic() - started,
            message="ssh_unreachable",
            details={"docker_restart": "ssh_unreachable"},
        )
    if r.exit_code != 0:
        return RecoveryResult(
            host=host.host,
            container_name=host.container_name,
            attempted=True,
            recovered=False,
            duration_s=time.monotonic() - started,
            message=f"docker_restart_failed: {r.stderr.strip() or 'non-zero exit'}",
            details={"docker_restart": "failed", "stderr": r.stderr.strip()},
        )
    details["docker_restart"] = "ok"

    # Phase 2: poll docker inspect for State.Status == running
    inspect_cmd = f"docker inspect -f '{{{{.State.Status}}}}' {host.container_name}"
    container_running = False
    for _ in range(_CONTAINER_UP_MAX_POLLS):
        r = ssh.run(host, inspect_cmd, timeout_s=10.0)
        if r.exit_code == 0 and r.stdout.strip() == "running":
            container_running = True
            break
        time.sleep(_CONTAINER_UP_POLL_INTERVAL_S)
    if not container_running:
        details["container_up"] = "timeout"
        return RecoveryResult(
            host=host.host,
            container_name=host.container_name,
            attempted=True,
            recovered=False,
            duration_s=time.monotonic() - started,
            message="container_not_running_after_restart",
            details=details,
        )
    details["container_up"] = "ok"

    # Phase 3: GPU probe
    probe_cmd = f"docker exec {host.container_name} nvidia-smi -L"
    r = ssh.run(host, probe_cmd, timeout_s=_GPU_PROBE_TIMEOUT_S)
    if r.exit_code != 0 or not r.stdout.strip():
        details["gpu_probe"] = "failed"
        return RecoveryResult(
            host=host.host,
            container_name=host.container_name,
            attempted=True,
            recovered=False,
            duration_s=time.monotonic() - started,
            message=f"gpu_probe_failed: {r.stderr.strip() or 'empty stdout'}",
            details=details,
        )
    details["gpu_probe"] = "ok"
    details["gpu_probe_stdout"] = r.stdout.strip()

    return RecoveryResult(
        host=host.host,
        container_name=host.container_name,
        attempted=True,
        recovered=True,
        duration_s=time.monotonic() - started,
        message="recovered_via_container_restart",
        details=details,
    )
