# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Transport layer — SSH / rsync Protocols + shell-out default implementations."""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tools.odin.asgard.fleet import ValkyrieConfig

__all__ = [
    "SSHResult",
    "RsyncResult",
    "SSHRunner",
    "RsyncRunner",
    "ShellSSHRunner",
    "ShellRsyncRunner",
]


@dataclass
class SSHResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False


@dataclass
class RsyncResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    bytes_transferred: int | None = None


class SSHRunner(Protocol):
    def run(
        self,
        host: ValkyrieConfig,
        cmd: str,
        *,
        timeout_s: float | None = None,
        stdout_tee: Path | None = None,
    ) -> SSHResult: ...


class RsyncRunner(Protocol):
    def pull(self, host: ValkyrieConfig, remote_path: str, local_path: Path) -> RsyncResult: ...

    def push(self, host: ValkyrieConfig, local_path: Path, remote_path: str) -> RsyncResult: ...


# --- Default ssh implementation ---------------------------------------------


_DEFAULT_SSH_OPTS = [
    "-tt",  # force PTY: ssh client death → SIGHUP to remote process group
    "-o",
    "StrictHostKeyChecking=accept-new",
    "-o",
    "ServerAliveInterval=30",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "BatchMode=yes",  # no interactive password prompts in a dispatch
]


class ShellSSHRunner:
    """SSH runner that shells out to the ``ssh`` command."""

    def _build_ssh_argv(self, host: ValkyrieConfig, cmd: str, timeout_s: float | None) -> list[str]:
        argv: list[str] = ["ssh", *_DEFAULT_SSH_OPTS]
        if host.ssh_key is not None:
            argv += ["-i", str(host.ssh_key)]
        argv += [f"{host.ssh_user}@{host.host}", cmd]
        return argv

    def run(
        self,
        host: ValkyrieConfig,
        cmd: str,
        *,
        timeout_s: float | None = None,
        stdout_tee: Path | None = None,
    ) -> SSHResult:
        """Run ``cmd`` on ``host`` and return an :class:`SSHResult`.

        Streams stdout line-by-line to ``stdout_tee`` (if given) via a reader
        thread so ``timeout_s`` is honoured even when the remote command
        produces no output — a single-thread drain-then-wait approach would
        block in ``readline()`` indefinitely on a hung no-output command. On
        timeout the child process is terminated (then killed after 10 s
        grace); ``timed_out`` is set on the returned result and ``exit_code``
        is whatever the terminated process reported (typically negative).

        Args:
            host: Target host configuration.
            cmd: Shell command to execute on the remote host.
            timeout_s: Optional wall-clock timeout in seconds. ``None`` means no
                timeout.
            stdout_tee: Optional path to append stdout lines as they arrive.

        Returns:
            :class:`SSHResult` capturing exit code, captured output, duration,
            and timeout flag.
        """
        argv = self._build_ssh_argv(host, cmd, timeout_s)
        t0 = time.monotonic()
        tee_fh = None
        stdout_buf: list[str] = []
        stderr_buf: list[str] = []
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if stdout_tee is not None:
            stdout_tee.parent.mkdir(parents=True, exist_ok=True)
            tee_fh = stdout_tee.open("a", encoding="utf-8")

        stdout_stream = proc.stdout
        stderr_stream = proc.stderr

        def _drain_stdout() -> None:
            if stdout_stream is None:
                return
            while True:
                line = stdout_stream.readline()
                if not line:
                    return
                stdout_buf.append(line)
                if tee_fh is not None:
                    tee_fh.write(line)
                    tee_fh.flush()

        def _drain_stderr() -> None:
            if stderr_stream is None:
                return
            while True:
                line = stderr_stream.readline()
                if not line:
                    return
                stderr_buf.append(line)

        stdout_thread = threading.Thread(target=_drain_stdout, daemon=True)
        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        timed_out = False
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.terminate()
            try:
                rc = proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = proc.wait()

        # Reader threads exit naturally when pipes close (which happens once the
        # child is fully reaped). A small join bound guards against pathological
        # cases where the OS has not yet EOF'd the pipe.
        stdout_thread.join(timeout=5.0)
        stderr_thread.join(timeout=5.0)

        if tee_fh is not None:
            tee_fh.close()
        if stdout_stream is not None:
            stdout_stream.close()
        if stderr_stream is not None:
            stderr_stream.close()

        duration = time.monotonic() - t0
        return SSHResult(
            exit_code=int(rc if rc is not None else -1),
            stdout="".join(stdout_buf),
            stderr="".join(stderr_buf),
            duration_s=duration,
            timed_out=timed_out,
        )


# --- Default rsync implementation -------------------------------------------


_PUSH_EXCLUDES = [
    "--exclude=__pycache__/",
    "--exclude=.git/",
    "--exclude=odin_runs/",
    "--exclude=benchmark_*_Isaac-*.json",
    "--exclude=*.swp",
    "--exclude=.claude/",
    # tacsl_sensor asset data is sometimes installed as root:0600 on dev machines,
    # which makes rsync return exit 23 ("partial transfer"). Exclude the whole
    # data directory — these assets are regenerated inside the container on use
    # and no Odin-curated task needs them.
    "--exclude=source/isaaclab/isaaclab/sensors/tacsl_sensor/gelsight_r15_data/",
    # logs/ is the controller-side scratch dir for local training runs (TB
    # events, checkpoints, params) — typically multi-GB and never needed on a
    # Valkyrie (Hugin/Munin write their training output into the bundle via
    # --log_dir, not here).
    "--exclude=logs/",
    # docs/ hosts the sphinx sources + any built HTML / PDF / images. Not
    # needed at runtime on a Valkyrie; can grow to several hundred MB with
    # the full built site.
    "--exclude=docs/",
    # docker/.container.cfg is dev-machine-specific (X11 forwarding state,
    # tmp xauth paths). Bootstrap writes a headless version on the remote
    # after rsync so Valkyries (which have no DISPLAY) can start the
    # container cleanly.
    "--exclude=docker/.container.cfg",
    # _isaac_sim is a symlink to the dev-machine's local Isaac Sim install
    # (e.g. ``../IsaacSim6/_build/linux-x86_64/release``) — meaningless on
    # the remote. The Dockerfile creates a correct symlink
    # ``_isaac_sim → /isaac-sim`` during image build; if the rsync'd broken
    # symlink lands in the source tree the Dockerfile's subsequent bulk
    # ``COPY ../ ${ISAACLAB_PATH}/`` overwrites it with the broken one.
    # Excluding here lets the Dockerfile's symlink win, and bootstrap's
    # post-start fix inside the container also retargets it explicitly.
    "--exclude=_isaac_sim",
    # Each git worktree under .worktrees/ is a full second IsaacLab checkout
    # (often with its own venv and partial cache). Single-digit GB → 10+ GB
    # of dead weight on every push.
    "--exclude=.worktrees/",
    # outputs/ is the dev-machine's scratch dir for local script outputs
    # (rendered frames, generated USDs, debug dumps) — never needed on a
    # Valkyrie.
    "--exclude=outputs/",
    # `_isaaclab_install_ci_*/` are uv venvs created by IsaacLab's
    # install_ci test suite (source/isaaclab/test/install_ci/). When the
    # test crashes mid-install (e.g. flaky pypi.nvidia.com timeout) the
    # cleanup hook misses and the partial venv stays — multi-GB of stale
    # CUDA wheels per leftover.
    "--exclude=_isaaclab_install_ci_*/",
    # Tool / editor caches — local-only, regenerated on demand.
    "--exclude=.ruff_cache/",
    "--exclude=.pytest_cache/",
    "--exclude=.mypy_cache/",
    "--exclude=.vscode/",
    "--exclude=.cursor/",
    # Workflow/skill metadata that lives in the dev tree but is irrelevant
    # to a remote training job.
    "--exclude=.superpowers/",
    "--exclude=.github/",
]


class ShellRsyncRunner:
    """Rsync runner that shells out to the ``rsync`` command."""

    def _build_ssh_transport_opt(self, host: ValkyrieConfig) -> str | None:
        """Return the value for rsync's ``-e`` flag when an ssh_key is set."""
        if host.ssh_key is None:
            return None
        return f"ssh -i {host.ssh_key} -o StrictHostKeyChecking=accept-new"

    def _run_rsync(self, argv: list[str]) -> RsyncResult:
        t0 = time.monotonic()
        proc = subprocess.run(argv, capture_output=True, text=True)
        duration = time.monotonic() - t0
        return RsyncResult(
            exit_code=int(proc.returncode),
            stdout=str(proc.stdout or ""),
            stderr=str(proc.stderr or ""),
            duration_s=duration,
        )

    def push(
        self,
        host: ValkyrieConfig,
        local_path: Path,
        remote_path: str,
    ) -> RsyncResult:
        """Push ``local_path`` (controller side) to ``remote_path`` on the Valkyrie.

        Includes ``--delete`` so the remote tree matches the local tree
        exactly (minus excludes), and a fixed exclude list for noise.

        Args:
            host: Target host configuration.
            local_path: Local directory to sync from.
            remote_path: Destination path on the remote host.

        Returns:
            :class:`RsyncResult` capturing exit code, output, and duration.
        """
        argv: list[str] = ["rsync", "-avz", "--delete", *_PUSH_EXCLUDES]
        transport = self._build_ssh_transport_opt(host)
        if transport is not None:
            argv += ["-e", transport]
        argv += [f"{str(local_path).rstrip('/')}/", f"{host.ssh_user}@{host.host}:{remote_path}"]
        return self._run_rsync(argv)

    def pull(
        self,
        host: ValkyrieConfig,
        remote_path: str,
        local_path: Path,
    ) -> RsyncResult:
        """Pull ``remote_path`` on the Valkyrie to ``local_path`` on the controller.

        NO ``--delete`` — we don't want to prune prior bundles on the
        controller's side when fetching a new bundle.

        The source is forced to end in ``/`` so rsync copies the remote
        directory's *contents* into ``local_path`` rather than creating a
        nested ``local_path/<basename>/``. This matches the symmetric
        behaviour of :meth:`push` and keeps the caller's mental model
        simple: "pull THIS remote dir INTO THIS local dir".

        Args:
            host: Target host configuration.
            remote_path: Source path on the remote host.
            local_path: Local destination path.

        Returns:
            :class:`RsyncResult` capturing exit code, output, and duration.
        """
        local_path.parent.mkdir(parents=True, exist_ok=True)
        argv: list[str] = ["rsync", "-avz"]
        transport = self._build_ssh_transport_opt(host)
        if transport is not None:
            argv += ["-e", transport]
        argv += [f"{host.ssh_user}@{host.host}:{remote_path.rstrip('/')}/", str(local_path)]
        return self._run_rsync(argv)
