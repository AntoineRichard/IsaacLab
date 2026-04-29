# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Preflight — one-shot health check per Valkyrie before any job dispatches."""

from __future__ import annotations

from dataclasses import dataclass, field

from tools.odin.asgard.fleet import ValkyrieConfig
from tools.odin.asgard.recovery import recover_valkyrie_gpu
from tools.odin.asgard.transport import SSHRunner

__all__ = ["PreflightResult", "preflight_valkyrie"]

# Subset of worker._GPU_LOST_SIGNATURES that are fixable via container restart.
# "Vulkan ERROR_INCOMPATIBLE_DRIVER" is excluded: that requires a host-level
# driver fix that docker restart cannot resolve.
_NVML_WEDGE_SIGNATURES = (
    "Failed to initialize NVML",
    "CUDA error: no CUDA-capable device is detected",
)


def _looks_wedged(stderr: str) -> bool:
    return any(s in stderr for s in _NVML_WEDGE_SIGNATURES)


@dataclass
class PreflightResult:
    host: str
    ok: bool
    checks: dict[str, bool] = field(default_factory=dict)
    message: str = ""


def preflight_valkyrie(host: ValkyrieConfig, *, ssh: SSHRunner, auto_restart: bool = True) -> PreflightResult:
    """Run SSH + docker + container + IsaacLab-directory + GPU checks on one host.

    Returns a :class:`PreflightResult` with ``ok=True`` iff all five checks
    pass. Later checks short-circuit: if SSH is unreachable, downstream
    checks are reported as ``False`` and the first failing check's diagnostic
    lands in ``message``.

    When ``auto_restart`` is ``True`` (the default) and the ``gpu_present``
    check fails, :func:`~tools.odin.asgard.recovery.recover_valkyrie_gpu` is
    called to restart the container and re-probe.  If recovery succeeds the
    host is marked healthy; otherwise it is marked down with a
    ``recovery_failed`` marker in ``message``.

    Args:
        host: Target Valkyrie.
        ssh: :class:`SSHRunner` implementation (``ShellSSHRunner`` in prod,
            fake in tests).
        auto_restart: When ``True``, attempt one container restart on NVML
            wedge or empty ``nvidia-smi`` output before failing the host.

    Returns:
        Aggregated :class:`PreflightResult`.
    """
    checks = {
        "ssh_reach": False,
        "docker_running": False,
        "container_up": False,
        "isaaclab_present": False,
        "gpu_present": False,
    }

    # 1. ssh_reach — single round-trip echo.
    r = ssh.run(host, "echo preflight-ok", timeout_s=15.0)
    if r.exit_code != 0:
        return PreflightResult(
            host=host.host,
            ok=False,
            checks=checks,
            message=f"ssh unreachable: {r.stderr.strip() or r.stdout.strip() or 'non-zero exit'}",
        )
    checks["ssh_reach"] = True

    # 2. docker_running — daemon responsive.
    r = ssh.run(host, "docker ps --format '{{.Names}}' 2>&1", timeout_s=15.0)
    if r.exit_code != 0:
        return PreflightResult(
            host=host.host,
            ok=False,
            checks=checks,
            message=f"docker daemon not responding: {r.stderr.strip() or r.stdout.strip()}",
        )
    checks["docker_running"] = True

    # 3. container_up — named container is in "running" state.
    r = ssh.run(
        host,
        f"docker inspect -f '{{{{.State.Status}}}}' {host.container_name}",
        timeout_s=15.0,
    )
    container_status = r.stdout.strip()
    if r.exit_code != 0 or container_status != "running":
        return PreflightResult(
            host=host.host,
            ok=False,
            checks=checks,
            message=f"container {host.container_name!r} not running (status={container_status!r})",
        )
    checks["container_up"] = True

    # 4. isaaclab_present — repo dir exists on the host.
    r = ssh.run(host, f"test -d {host.isaaclab_path}", timeout_s=10.0)
    if r.exit_code != 0:
        return PreflightResult(
            host=host.host,
            ok=False,
            checks=checks,
            message=f"IsaacLab path {host.isaaclab_path!r} missing on host",
        )
    checks["isaaclab_present"] = True

    # 5. gpu_present — at least one GPU visible inside the running container.
    r = ssh.run(host, f"docker exec {host.container_name} nvidia-smi -L", timeout_s=15.0)
    if r.exit_code == 0 and r.stdout.strip():
        checks["gpu_present"] = True
        return PreflightResult(host=host.host, ok=True, checks=checks, message="")

    # GPU absent — try one container-restart recovery if allowed.
    if auto_restart and (_looks_wedged(r.stderr) or not r.stdout.strip()):
        rec = recover_valkyrie_gpu(host, ssh=ssh)
        if rec.recovered:
            checks["gpu_present"] = True
            return PreflightResult(
                host=host.host,
                ok=True,
                checks=checks,
                message="recovered: container restarted to clear NVML wedge",
            )
        return PreflightResult(
            host=host.host,
            ok=False,
            checks=checks,
            message=f"GPU absent in container, recovery_failed: {rec.message}",
        )

    return PreflightResult(
        host=host.host,
        ok=False,
        checks=checks,
        message=f"GPU absent in container: {r.stderr.strip() or 'empty stdout'}",
    )
