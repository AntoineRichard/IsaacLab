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


def _parse_driver_to_cuda(driver_version: str) -> tuple[int, int] | None:
    """Map an NVIDIA driver version → max supported CUDA toolkit (major, minor).

    Conservative table derived from NVIDIA's published compatibility list.
    Returns ``None`` for unparsable input; ``(0, 0)`` for parseable but
    older-than-listed driver versions, so callers can tell parsed-but-too-old
    apart from unparsable.

    Args:
        driver_version: NVIDIA driver version string from
            ``nvidia-smi --query-gpu=driver_version`` (e.g. ``"535.161.07"``).

    Returns:
        ``(major, minor)`` tuple of the highest CUDA toolkit the driver
        supports, ``(0, 0)`` for unrecognized old drivers, or ``None`` for
        unparsable input.
    """
    try:
        major = int(driver_version.split(".", 1)[0])
    except (ValueError, IndexError):
        return None
    # Driver major → max CUDA toolkit version it supports.
    # Newer drivers come first so the first-match wins for forward compat.
    table = [
        (570, (12, 8)),
        (560, (12, 6)),
        (555, (12, 5)),
        (550, (12, 4)),
        (545, (12, 3)),
        (535, (12, 2)),
        (525, (12, 0)),
        (520, (11, 8)),
    ]
    for min_driver, cuda in table:
        if major >= min_driver:
            return cuda
    return (0, 0)


@dataclass
class PreflightResult:
    host: str
    ok: bool
    checks: dict[str, bool] = field(default_factory=dict)
    message: str = ""
    cuda_version: tuple[int, int] | None = None
    newton_available: bool = True
    recovery_attempted: bool = False
    recovery_succeeded: bool = False


def _resolve_newton_availability(
    *,
    host: ValkyrieConfig,
    ssh: SSHRunner,
    newton_cuda_floor: tuple[int, int] | None,
) -> tuple[tuple[int, int] | None, bool, str]:
    """Probe the host's CUDA version and decide newton availability.

    Args:
        host: Target host.
        ssh: SSH runner.
        newton_cuda_floor: Minimum (major, minor) CUDA needed for Newton.
            ``None`` skips the check (newton always available).

    Returns:
        ``(cuda_version, newton_available, message)``. ``message`` is empty
        when newton is available, or carries a hint about how to upgrade
        when not.
    """
    drv = ssh.run(
        host,
        f"docker exec {host.container_name} nvidia-smi --query-gpu=driver_version --format=csv,noheader",
        timeout_s=15.0,
    )
    cuda_version: tuple[int, int] | None = None
    if drv.exit_code == 0 and drv.stdout.strip():
        cuda_version = _parse_driver_to_cuda(drv.stdout.strip().splitlines()[0])

    if newton_cuda_floor is None or cuda_version is None:
        return cuda_version, True, ""
    if cuda_version >= newton_cuda_floor:
        return cuda_version, True, ""
    msg = (
        f"newton unavailable: host CUDA "
        f"{cuda_version[0]}.{cuda_version[1]} < newton floor "
        f"{newton_cuda_floor[0]}.{newton_cuda_floor[1]}; "
        f"run `odin-cuda install --target "
        f"{newton_cuda_floor[0]}.{newton_cuda_floor[1]}`"
    )
    return cuda_version, False, msg


def preflight_valkyrie(
    host: ValkyrieConfig,
    *,
    ssh: SSHRunner,
    auto_restart: bool = True,
    newton_cuda_floor: tuple[int, int] | None = None,
) -> PreflightResult:
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

    When ``newton_cuda_floor`` is set, the host driver version is queried after
    GPU presence is confirmed and mapped to a CUDA toolkit version. Hosts below
    the floor have ``newton_available=False`` in the result but remain
    ``ok=True`` (PhysX jobs can still run on them).

    Args:
        host: Target Valkyrie.
        ssh: :class:`SSHRunner` implementation (``ShellSSHRunner`` in prod,
            fake in tests).
        auto_restart: When ``True``, attempt one container restart on NVML
            wedge or empty ``nvidia-smi`` output before failing the host.
        newton_cuda_floor: Minimum ``(major, minor)`` CUDA toolkit version
            required for Newton (warp) workloads. ``None`` skips the check
            and marks every healthy host as newton-capable.

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
        cuda_version, newton_available, msg = _resolve_newton_availability(
            host=host, ssh=ssh, newton_cuda_floor=newton_cuda_floor
        )
        return PreflightResult(
            host=host.host,
            ok=True,
            checks=checks,
            message=msg,
            cuda_version=cuda_version,
            newton_available=newton_available,
        )

    # GPU absent — try one container-restart recovery if allowed.
    if auto_restart and (_looks_wedged(r.stderr) or not r.stdout.strip()):
        rec = recover_valkyrie_gpu(host, ssh=ssh)
        if rec.recovered:
            checks["gpu_present"] = True
            cuda_version, newton_available, post_recover_msg = _resolve_newton_availability(
                host=host, ssh=ssh, newton_cuda_floor=newton_cuda_floor
            )
            recover_msg = "recovered: container restarted to clear NVML wedge"
            message = f"{recover_msg}; {post_recover_msg}" if post_recover_msg else recover_msg
            return PreflightResult(
                host=host.host,
                ok=True,
                checks=checks,
                message=message,
                cuda_version=cuda_version,
                newton_available=newton_available,
                recovery_attempted=True,
                recovery_succeeded=True,
            )
        return PreflightResult(
            host=host.host,
            ok=False,
            checks=checks,
            message=f"GPU absent in container, recovery_failed: {rec.message}",
            recovery_attempted=True,
            recovery_succeeded=False,
        )

    return PreflightResult(
        host=host.host,
        ok=False,
        checks=checks,
        message=f"GPU absent in container: {r.stderr.strip() or 'empty stdout'}",
    )
