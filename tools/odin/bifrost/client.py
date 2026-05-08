# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Thin subprocess wrappers around the ``osmo`` CLI.

One module, one public class :class:`OsmoClient`. Each method shells out
to a single ``osmo`` invocation and parses the output. Errors are typed:

- :class:`OsmoAuthError` — auth/credential failure; caller surfaces.
- :class:`OsmoTransientError` — retryable (HTTP 5xx, connection reset).
- :class:`OsmoCliError` — anything else (bad spec, parse failure).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "OsmoAuthError",
    "OsmoCliError",
    "OsmoClient",
    "OsmoTransientError",
    "TaskSnapshot",
    "WorkflowSnapshot",
]


class OsmoCliError(RuntimeError):
    """Generic ``osmo`` CLI failure (non-zero exit, parse failure, etc.)."""


class OsmoAuthError(OsmoCliError):
    """Auth failure (HTTP 401/403). Not retried."""


class OsmoTransientError(OsmoCliError):
    """Retryable failure (HTTP 5xx, connection issues)."""


@dataclass(frozen=True)
class TaskSnapshot:
    """One task's state in an OSMO workflow snapshot."""

    name: str
    status: str  # COMPLETED | FAILED | RUNNING | etc.
    exit_code: int | None


@dataclass(frozen=True)
class WorkflowSnapshot:
    """Snapshot of a workflow + per-task states from one ``osmo workflow status`` call."""

    workflow_id: str
    status: str
    tasks: list[TaskSnapshot]


_AUTH_PATTERN = re.compile(r"HTTP 40[13]|unauthori[sz]ed", re.IGNORECASE)
_TRANSIENT_PATTERN = re.compile(r"HTTP 5\d\d|connection (reset|refused|timed?\s+out|timeout)", re.IGNORECASE)
_WORKFLOW_ID_PATTERN = re.compile(r"^Workflow ID\s+-\s+(\S+)", re.MULTILINE)

_UNKNOWN_FLAG_PATTERNS = (
    re.compile(r"unknown flag", re.IGNORECASE),
    re.compile(r"unrecognized argument", re.IGNORECASE),
    re.compile(r"--output.*not recognized", re.IGNORECASE),
)


def _looks_like_unknown_flag(stderr: str) -> bool:
    return any(p.search(stderr) for p in _UNKNOWN_FLAG_PATTERNS)


def _classify(stderr: str) -> type[OsmoCliError]:
    if _AUTH_PATTERN.search(stderr):
        return OsmoAuthError
    if _TRANSIENT_PATTERN.search(stderr):
        return OsmoTransientError
    return OsmoCliError


class OsmoClient:
    """Subprocess-based wrapper around the ``osmo`` CLI.

    Args:
        profile: OSMO profile name. Passed via ``OSMO_PROFILE`` env var on
            every invocation.
        executable: ``osmo`` binary path. Defaults to ``"osmo"`` (relies
            on ``$PATH``).
    """

    def __init__(self, *, profile: str, executable: str = "osmo") -> None:
        self._profile = profile
        self._exe = executable

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["OSMO_PROFILE"] = self._profile
        return env

    def _run(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            cmd,
            env=self._env(),
            capture_output=True,
            text=True,
            check=False,
        )

    def submit(self, yaml_path: Path, *, rsync_pairs: Iterable[tuple[str, str]] = (), pool: str | None = None) -> str:
        """Submit a workflow YAML and return the workflow_id.

        Args:
            yaml_path: Path to the rendered workflow YAML.
            rsync_pairs: Pairs of ``(local_path, container_path)`` for OSMO's
                ``--rsync`` continuous-sync feature.

        Returns:
            The OSMO workflow ID parsed from stdout.

        Raises:
            OsmoAuthError, OsmoTransientError, OsmoCliError: per :func:`_classify`.
        """
        cmd: list[str] = [self._exe, "workflow", "submit", str(yaml_path)]
        if pool is not None:
            cmd.extend(["--pool", pool])
        for local, remote in rsync_pairs:
            cmd.extend(["--rsync", f"{local}:{remote}"])
        cp = self._run(cmd)
        if cp.returncode != 0:
            raise _classify(cp.stderr)(f"`osmo workflow submit` failed: {cp.stderr.strip()}")
        m = _WORKFLOW_ID_PATTERN.search(cp.stdout)
        if not m:
            raise OsmoCliError(f"could not parse Workflow ID from osmo stdout: {cp.stdout!r}")
        return m.group(1)

    def status(self, workflow_id: str) -> WorkflowSnapshot:
        """Fetch the workflow snapshot.

        Tries ``--output json`` first; if the flag is unrecognized, retries
        with the default table output and parses that.

        Args:
            workflow_id: The OSMO workflow ID returned by :meth:`submit`.

        Returns:
            A :class:`WorkflowSnapshot` populated from the OSMO response.

        Raises:
            OsmoAuthError, OsmoTransientError, OsmoCliError: per :func:`_classify`.
        """
        cmd_json = [self._exe, "workflow", "status", workflow_id, "--output", "json"]
        cp = self._run(cmd_json)
        if cp.returncode == 0:
            return self._parse_status_json(cp.stdout, workflow_id)
        if _looks_like_unknown_flag(cp.stderr):
            cp2 = self._run([self._exe, "workflow", "status", workflow_id])
            if cp2.returncode != 0:
                raise _classify(cp2.stderr)(f"`osmo workflow status` failed: {cp2.stderr.strip()}")
            return self._parse_status_table(cp2.stdout, workflow_id)
        raise _classify(cp.stderr)(f"`osmo workflow status` failed: {cp.stderr.strip()}")

    @staticmethod
    def _parse_status_json(stdout: str, workflow_id: str) -> WorkflowSnapshot:
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as e:
            raise OsmoCliError(f"could not parse JSON status: {e}") from e
        tasks = [
            TaskSnapshot(
                name=str(t["name"]),
                status=str(t["status"]),
                exit_code=(None if t.get("exit_code") in (None, "-") else int(t["exit_code"])),
            )
            for t in data.get("tasks") or []
        ]
        return WorkflowSnapshot(
            workflow_id=str(data.get("id", workflow_id)),
            status=str(data["status"]),
            tasks=tasks,
        )

    @staticmethod
    def _parse_status_table(stdout: str, workflow_id: str) -> WorkflowSnapshot:
        wf_status = "UNKNOWN"
        tasks: list[TaskSnapshot] = []
        in_tasks = False
        for raw in stdout.splitlines():
            line = raw.strip()
            if line.startswith("Status:"):
                wf_status = line.split(":", 1)[1].strip()
            elif line.startswith("NAME"):
                in_tasks = True
                continue
            elif in_tasks and line:
                parts = line.split()
                if len(parts) < 2:
                    continue
                name, status = parts[0], parts[1]
                exit_str = parts[2] if len(parts) >= 3 else "-"
                exit_code = None if exit_str in ("-", "") else int(exit_str)
                tasks.append(TaskSnapshot(name=name, status=status, exit_code=exit_code))
        return WorkflowSnapshot(workflow_id=workflow_id, status=wf_status, tasks=tasks)

    def dataset_download(self, name: str, dest_dir: Path) -> None:
        """Download an OSMO dataset to a local directory.

        Creates ``dest_dir`` (and parents) if missing.

        Args:
            name: OSMO dataset name (typically ``{prefix}-{dispatch_id}-{run_id}``).
            dest_dir: Local directory to download into.

        Raises:
            OsmoAuthError, OsmoTransientError, OsmoCliError: per :func:`_classify`.
        """
        dest_dir.mkdir(parents=True, exist_ok=True)
        cmd = [self._exe, "dataset", "download", name, str(dest_dir)]
        cp = self._run(cmd)
        if cp.returncode != 0:
            raise _classify(cp.stderr)(f"`osmo dataset download` failed: {cp.stderr.strip()}")

    def cancel(self, workflow_id: str) -> None:
        """Cancel an in-flight workflow.

        Args:
            workflow_id: The OSMO workflow ID.

        Raises:
            OsmoAuthError, OsmoTransientError, OsmoCliError: per :func:`_classify`.
        """
        cp = self._run([self._exe, "workflow", "cancel", workflow_id])
        if cp.returncode != 0:
            raise _classify(cp.stderr)(f"`osmo workflow cancel` failed: {cp.stderr.strip()}")

    def logs(self, workflow_id: str, task_name: str, *, follow: bool = False) -> Iterator[bytes]:
        """Yield log bytes from the named task.

        With ``follow=False``, runs the command to completion and yields
        the captured stdout as a single bytes chunk. With ``follow=True``,
        streams stdout line-by-line until the subprocess exits.

        Args:
            workflow_id: The OSMO workflow ID.
            task_name: The task name (must match an OSMO task name in the workflow).
            follow: If True, follow the live stream (like ``tail -f``).

        Yields:
            Bytes chunks of stdout.

        Raises:
            OsmoAuthError, OsmoTransientError, OsmoCliError: on non-zero exit (non-follow path only).
        """
        cmd = [self._exe, "workflow", "logs", workflow_id, task_name]
        if follow:
            cmd.append("--follow")
            return self._stream_logs(cmd)
        cp = self._run(cmd)
        if cp.returncode != 0:
            raise _classify(cp.stderr)(f"`osmo workflow logs` failed: {cp.stderr.strip()}")
        return iter([cp.stdout.encode()])

    def _stream_logs(self, cmd: list[str]) -> Iterator[bytes]:
        proc = subprocess.Popen(cmd, env=self._env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
        assert proc.stdout is not None
        try:
            yield from iter(proc.stdout.readline, b"")
        finally:
            proc.stdout.close()
            proc.wait()
