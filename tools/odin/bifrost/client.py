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
from collections.abc import Iterable
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

    def submit(self, yaml_path: Path, *, rsync_pairs: Iterable[tuple[str, str]] = ()) -> str:
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
