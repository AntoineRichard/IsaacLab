# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fresh-Valkyrie bootstrap — bring a naked host to T3.1-preflight-ready state.

Unlike :mod:`tools.odin.asgard.provisioner` (the warm-path refresher used
inside :func:`~tools.odin.asgard.runner.run_dispatch`), bootstrap assumes the
remote has *only* SSH + Docker + a GPU — no IsaacLab clone, no container.
It wipes any prior tree, pushes the working tree, boots the Isaac Lab
container with a long enough timeout to survive first-time image build,
and verifies the container ended up in ``"running"`` state.
"""

from __future__ import annotations

import concurrent.futures as _cf
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
from tools.odin.asgard.provisioner import _container_start
from tools.odin.asgard.transport import RsyncRunner, SSHRunner

__all__ = ["BootstrapResult", "bootstrap_valkyrie", "bootstrap_fleet"]


@dataclass
class BootstrapResult:
    """Outcome of bootstrapping a single Valkyrie."""

    host: str
    ok: bool
    message: str = ""
    commit_sha: str = ""
    step_durations_s: dict[str, float] = field(default_factory=dict)


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


def _time_step() -> float:
    return time.perf_counter()


def bootstrap_valkyrie(
    host: ValkyrieConfig,
    working_tree: Path,
    *,
    ssh: SSHRunner,
    rsync: RsyncRunner,
    build_timeout_s: int = 1800,
) -> BootstrapResult:
    """Bring a fresh Valkyrie to T3.1-preflight-ready state.

    Pipeline (short-circuits on any step failure):

      1. SSH reach — a 15 s ``echo bootstrap-ok`` probe.
      2. Docker daemon reach — a 15 s ``docker ps`` probe.
      3. Wipe — ``rm -rf {isaaclab_path}`` (always, for idempotent re-runs).
      4. Rsync — push ``working_tree`` to ``{isaaclab_path}``.
      5. Container start — ``./docker/container.py start`` with
         ``build_timeout_s``.
      6. Container verify — ``docker inspect`` must report ``"running"``.

    Args:
        host: Target Valkyrie.
        working_tree: Controller-side IsaacLab path to push.
        ssh: SSH runner.
        rsync: Rsync runner.
        build_timeout_s: Timeout [s] for ``./docker/container.py start``
            (default 1800 = 30 min; covers a cold first-time docker build).

    Returns:
        :class:`BootstrapResult` with ``ok=True`` iff all six steps passed.
        ``step_durations_s`` records wall-clock seconds for steps 3-6 (the
        ones that actually do work). Probe steps 1-2 are not included.
    """
    commit_sha = _resolve_local_sha(working_tree)
    step_durations_s: dict[str, float] = {}

    # 1. SSH reach.
    r = ssh.run(host, "echo bootstrap-ok", timeout_s=15)
    if r.exit_code != 0:
        return BootstrapResult(
            host=host.host,
            ok=False,
            message=f"ssh unreachable: {r.stderr.strip() or r.stdout.strip() or 'non-zero exit'}",
            commit_sha=commit_sha,
        )

    # 2. Docker daemon reach.
    r = ssh.run(host, "docker ps --format '{{.Names}}' 2>&1", timeout_s=15)
    if r.exit_code != 0:
        return BootstrapResult(
            host=host.host,
            ok=False,
            message=f"docker daemon not responding: {r.stderr.strip() or r.stdout.strip()}",
            commit_sha=commit_sha,
        )

    # 3. Wipe.
    t0 = _time_step()
    r = ssh.run(host, f"rm -rf {host.isaaclab_path}", timeout_s=60)
    step_durations_s["wipe"] = _time_step() - t0
    if r.exit_code != 0:
        return BootstrapResult(
            host=host.host,
            ok=False,
            message=f"failed to wipe {host.isaaclab_path!r}: {r.stderr.strip() or 'non-zero exit'}",
            commit_sha=commit_sha,
            step_durations_s=step_durations_s,
        )

    # 4. Rsync.
    t0 = _time_step()
    rr = rsync.push(host, working_tree, host.isaaclab_path)
    step_durations_s["rsync"] = _time_step() - t0
    if rr.exit_code != 0:
        return BootstrapResult(
            host=host.host,
            ok=False,
            message=f"rsync push failed: {rr.stderr.strip() or 'non-zero exit'}",
            commit_sha=commit_sha,
            step_durations_s=step_durations_s,
        )

    # 4b. Write headless docker/.container.cfg. Valkyries have no DISPLAY;
    # container.py's x11_check would either crash on missing DISPLAY or prompt
    # interactively on the SSH session (no TTY → EOFError). The statefile is
    # an INI file with ``[X11] X11_FORWARDING_ENABLED = 0`` — write it on the
    # remote before starting the container. .container.cfg itself is excluded
    # from the rsync push (see transport.py _PUSH_EXCLUDES).
    t0 = _time_step()
    cfg_body = "[X11]\\nX11_FORWARDING_ENABLED = 0\\n"
    r = ssh.run(
        host,
        f'printf "{cfg_body}" > {host.isaaclab_path}/docker/.container.cfg',
        timeout_s=15,
    )
    step_durations_s["configure_headless"] = _time_step() - t0
    if r.exit_code != 0:
        return BootstrapResult(
            host=host.host,
            ok=False,
            message=f"failed to write headless .container.cfg: {r.stderr.strip() or 'non-zero exit'}",
            commit_sha=commit_sha,
            step_durations_s=step_durations_s,
        )

    # 4c. Pre-create the bundle output directory on the host.
    # docker-compose's isaac-lab-base service bind-mounts ``~/IsaacLab/odin_runs``
    # to ``/workspace/isaaclab/odin_runs``, so Hugin/Munin bundles written inside
    # the container land on the host and can be rsync-pulled by the dispatcher.
    # compose's ``create_host_path: true`` would auto-create it, but doing it
    # explicitly here means the directory is owned by the SSH user
    # (not root, as docker would do it).
    t0 = _time_step()
    r = ssh.run(
        host,
        f"mkdir -p {host.isaaclab_path}/odin_runs",
        timeout_s=15,
    )
    step_durations_s["create_odin_runs"] = _time_step() - t0
    if r.exit_code != 0:
        return BootstrapResult(
            host=host.host,
            ok=False,
            message=f"failed to create {host.isaaclab_path}/odin_runs: {r.stderr.strip() or 'non-zero exit'}",
            commit_sha=commit_sha,
            step_durations_s=step_durations_s,
        )

    # 5. Container start.
    t0 = _time_step()
    started = _container_start(host, ssh, timeout_s=build_timeout_s)
    step_durations_s["container_start"] = _time_step() - t0
    if not started:
        return BootstrapResult(
            host=host.host,
            ok=False,
            message=f"container.py start failed (timeout={build_timeout_s}s)",
            commit_sha=commit_sha,
            step_durations_s=step_durations_s,
        )

    # 5b. Retarget ``_isaac_sim`` inside the container. Dockerfile.base:70
    # creates ``_isaac_sim → /isaac-sim`` during image build, but its final
    # bulk ``COPY ../ ${ISAACLAB_PATH}/`` (line 121) overwrites it when the
    # build context has a pre-existing ``_isaac_sim`` symlink pointing
    # elsewhere (rsync -a preserves whatever the dev machine has).
    # Excluding ``_isaac_sim`` from rsync prevents it reaching the remote
    # tree; this docker-exec step guarantees the running container's
    # symlink is correct even when the image was built with a broken one.
    t0 = _time_step()
    r = ssh.run(
        host,
        f"docker exec {host.container_name} ln -sf /isaac-sim /workspace/isaaclab/_isaac_sim",
        timeout_s=15,
    )
    step_durations_s["fix_isaac_sim_symlink"] = _time_step() - t0
    if r.exit_code != 0:
        return BootstrapResult(
            host=host.host,
            ok=False,
            message=f"failed to retarget _isaac_sim symlink inside container: {r.stderr.strip() or 'non-zero exit'}",
            commit_sha=commit_sha,
            step_durations_s=step_durations_s,
        )

    # 6. Container verify.
    t0 = _time_step()
    r = ssh.run(
        host,
        f"docker inspect -f '{{{{.State.Status}}}}' {host.container_name}",
        timeout_s=15,
    )
    step_durations_s["container_verify"] = _time_step() - t0
    status = r.stdout.strip()
    if r.exit_code != 0:
        return BootstrapResult(
            host=host.host,
            ok=False,
            message=(f"docker inspect failed for {host.container_name!r}: {r.stderr.strip() or 'non-zero exit'}"),
            commit_sha=commit_sha,
            step_durations_s=step_durations_s,
        )
    if status != "running":
        return BootstrapResult(
            host=host.host,
            ok=False,
            message=f"container {host.container_name!r} not running after start (status={status!r})",
            commit_sha=commit_sha,
            step_durations_s=step_durations_s,
        )

    return BootstrapResult(
        host=host.host,
        ok=True,
        commit_sha=commit_sha,
        step_durations_s=step_durations_s,
    )


def bootstrap_fleet(
    fleet: Fleet,
    working_tree: Path,
    *,
    ssh: SSHRunner,
    rsync: RsyncRunner,
    build_timeout_s: int = 1800,
    parallel: bool = True,
    verbose: bool = False,
) -> list[BootstrapResult]:
    """Bootstrap every host in ``fleet``.

    Args:
        fleet: Loaded fleet (via :func:`~tools.odin.asgard.fleet.load_fleet`).
        working_tree: Controller-side IsaacLab path to push.
        ssh: SSH runner shared across hosts.
        rsync: Rsync runner shared across hosts.
        build_timeout_s: Per-host ``container.py start`` timeout [s].
        parallel: When ``True`` (default), use a thread pool with
            ``max_workers = len(fleet.hosts)`` so hosts bootstrap
            concurrently. When ``False``, bootstrap sequentially —
            useful when shared network bandwidth would be saturated
            by simultaneous rsyncs.
        verbose: When ``True``, print a per-host summary line as each
            host finishes.

    Returns:
        :class:`BootstrapResult` list, one per host, in fleet order.
    """
    if parallel and len(fleet.hosts) > 1:
        with _cf.ThreadPoolExecutor(max_workers=len(fleet.hosts)) as pool:
            futures = [
                pool.submit(
                    bootstrap_valkyrie,
                    h,
                    working_tree,
                    ssh=ssh,
                    rsync=rsync,
                    build_timeout_s=build_timeout_s,
                )
                for h in fleet.hosts
            ]
            results = [f.result() for f in futures]
    else:
        results = [
            bootstrap_valkyrie(
                h,
                working_tree,
                ssh=ssh,
                rsync=rsync,
                build_timeout_s=build_timeout_s,
            )
            for h in fleet.hosts
        ]

    if verbose:
        for r in results:
            status = "ok" if r.ok else f"FAILED: {r.message}"
            print(f"[{r.host}] {status}")

    return results
