# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Valkyrie provisioning — rsync working tree + docker container bringup."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from tools.odin.asgard.fleet import ValkyrieConfig
from tools.odin.asgard.transport import RsyncRunner, SSHRunner

__all__ = ["ProvisionResult", "container_start", "container_stop", "provision_valkyrie"]


@dataclass
class ProvisionResult:
    host: str
    ok: bool
    message: str = ""
    commit_sha: str = ""


def _resolve_local_sha(working_tree: Path) -> str:
    """Return the controller's current git HEAD SHA, suffixed -dirty if uncommitted."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=working_tree,
            text=True,
        ).strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD"],
            cwd=working_tree,
        ).returncode
        if dirty != 0:
            sha = f"{sha}-dirty"
        return sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _container_status(host: ValkyrieConfig, ssh: SSHRunner) -> str:
    """Return 'running' / 'exited' / ... or '' if docker inspect fails."""
    r = ssh.run(
        host,
        f"docker inspect -f '{{{{.State.Status}}}}' {host.container_name}",
        timeout_s=15.0,
    )
    if r.exit_code != 0:
        return ""
    return r.stdout.strip()


def container_start(host: ValkyrieConfig, ssh: SSHRunner, *, timeout_s: int = 300) -> bool:
    """Run ``./docker/container.py start`` on ``host`` and return True on success.

    The warm-path default of 300 s suits subsequent dispatches where the
    container image is already built. First-time bootstrap must pass a
    longer ``timeout_s`` (see :mod:`tools.odin.asgard.bootstrap`) because the
    first-time docker build takes 15-30 minutes.
    """
    r = ssh.run(
        host,
        f"cd {host.isaaclab_path} && ./docker/container.py start",
        timeout_s=timeout_s,
    )
    return r.exit_code == 0


def container_stop(host: ValkyrieConfig, ssh: SSHRunner) -> bool:
    """Run ``./docker/container.py stop`` on ``host`` and return True on success."""
    r = ssh.run(
        host,
        f"cd {host.isaaclab_path} && ./docker/container.py stop",
        timeout_s=120.0,
    )
    return r.exit_code == 0


def provision_valkyrie(
    host: ValkyrieConfig,
    working_tree: Path,
    *,
    fresh: bool,
    ssh: SSHRunner,
    rsync: RsyncRunner,
) -> ProvisionResult:
    """Bring a Valkyrie up to the controller's current working tree.

    Flow:

    1. If ``fresh=True``: SSH ``rm -rf {isaaclab_path}`` on the host.
    2. Rsync push ``working_tree`` → ``{isaaclab_path}``.
    3. Container state:
       - ``fresh=True``: stop + start.
       - ``fresh=False``: query ``docker inspect`` status; start if not running.
    4. Return a :class:`ProvisionResult` with ``commit_sha`` from the local
       working tree (suffixed ``-dirty`` if uncommitted changes).

    Args:
        host: Target Valkyrie.
        working_tree: Controller-side IsaacLab path to push from (typically
            the repo root).
        fresh: When ``True``, wipe + full re-sync + container restart.
        ssh: SSH runner.
        rsync: Rsync runner.

    Returns:
        :class:`ProvisionResult` with ``ok=False`` on any step failure and
        a descriptive ``message``.
    """
    commit_sha = _resolve_local_sha(working_tree)

    if fresh:
        r = ssh.run(host, f"rm -rf {host.isaaclab_path}", timeout_s=60.0)
        if r.exit_code != 0:
            return ProvisionResult(
                host=host.host,
                ok=False,
                message=f"fresh wipe failed: {r.stderr.strip() or 'non-zero exit'}",
                commit_sha=commit_sha,
            )

    rr = rsync.push(host, working_tree, host.isaaclab_path)
    if rr.exit_code != 0:
        return ProvisionResult(
            host=host.host,
            ok=False,
            message=f"rsync push failed: {rr.stderr.strip() or 'non-zero exit'}",
            commit_sha=commit_sha,
        )

    if fresh:
        # Best-effort stop (may fail if container didn't exist yet — that's fine).
        container_stop(host, ssh)
        if not container_start(host, ssh):
            return ProvisionResult(
                host=host.host,
                ok=False,
                message="container.py start failed after fresh wipe",
                commit_sha=commit_sha,
            )
    else:
        status = _container_status(host, ssh)
        if status != "running":
            if not container_start(host, ssh):
                return ProvisionResult(
                    host=host.host,
                    ok=False,
                    message=f"container.py start failed (prior status={status!r})",
                    commit_sha=commit_sha,
                )

    return ProvisionResult(host=host.host, ok=True, commit_sha=commit_sha)
