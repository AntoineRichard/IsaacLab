# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locked environment provenance for the Kamino DVI benchmark."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from subprocess import CompletedProcess
from typing import Self

from .models import BenchmarkMatrix, EnvironmentLabel


@dataclass(frozen=True)
class GitState:
    """A repository head and the revisions reachable from it."""

    head: str
    ancestors: frozenset[str]
    dirty: bool

    def contains(self, revision: str) -> bool:
        """Return whether a revision is the head or one of its ancestors."""
        return revision == self.head or revision in self.ancestors


@dataclass(frozen=True)
class EnvironmentProvenance:
    """Software provenance captured from one locked Python environment."""

    python: Path
    packages: dict[str, str]
    newton_path: Path
    newton_revision: str
    isaaclab: GitState

    def replace(self, **changes) -> Self:
        """Return a copy with selected fields replaced."""
        return replace(self, **changes)


def python_executable(repo_root: Path, label: EnvironmentLabel) -> Path:
    """Return the interpreter for a locked benchmark environment."""
    directory = ".venv-current" if label is EnvironmentLabel.CURRENT else ".venv-pr3570"
    return repo_root / directory / "bin" / "python"


def validate_environment(
    matrix: BenchmarkMatrix,
    label: EnvironmentLabel,
    provenance: EnvironmentProvenance,
) -> None:
    """Validate that an environment matches the immutable experiment revisions.

    Args:
        matrix: Validated benchmark matrix.
        label: Environment role being validated.
        provenance: Captured interpreter and repository provenance.

    Raises:
        ValueError: If Isaac Lab or Newton does not match the approved revisions.
    """
    expected_newton = (
        matrix.revisions.newton_current if label is EnvironmentLabel.CURRENT else matrix.revisions.newton_pr
    )
    if provenance.newton_revision != expected_newton:
        raise ValueError(
            f"Newton revision mismatch for {label}: expected {expected_newton}, got {provenance.newton_revision}"
        )
    if not provenance.isaaclab.contains(matrix.revisions.isaaclab):
        raise ValueError(f"IsaacLab base revision {matrix.revisions.isaaclab} is not in the branch ancestry")
    if not provenance.isaaclab.contains(matrix.revisions.schema):
        raise ValueError(f"schema prerequisite {matrix.revisions.schema} is not in the branch ancestry")


_PROBE_SCRIPT = """
import importlib.metadata
import json
from pathlib import Path

import newton

packages = {
    (distribution.metadata.get("Name") or "").lower(): distribution.version
    for distribution in importlib.metadata.distributions()
    if distribution.metadata.get("Name")
}
distribution = importlib.metadata.distribution("newton")
direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
revision = direct_url.get("vcs_info", {}).get("commit_id")
print(json.dumps({
    "packages": packages,
    "newton_path": str(Path(newton.__file__).resolve()),
    "newton_revision": revision,
}, sort_keys=True))
"""


def probe_environment(
    python: Path,
    repo_root: Path,
    *,
    runner: Callable[..., CompletedProcess[str]] = subprocess.run,
) -> EnvironmentProvenance:
    """Capture package, Newton, and Isaac Lab provenance.

    Args:
        python: Interpreter belonging to the environment to probe.
        repo_root: Isaac Lab worktree root.
        runner: Subprocess function, injectable for tests.

    Returns:
        Captured environment provenance.

    Raises:
        ValueError: If the Newton installation lacks an immutable VCS revision.
    """
    options = {"check": True, "capture_output": True, "text": True}
    probe = runner([str(python), "-c", _PROBE_SCRIPT], **options)
    data = json.loads(probe.stdout.splitlines()[-1])
    revision = data.get("newton_revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("Newton installation does not expose a 40-character VCS revision")

    git = ["git", "-C", str(repo_root)]
    head = runner(git + ["rev-parse", "HEAD"], **options).stdout.strip()
    reachable = runner(git + ["rev-list", "HEAD"], **options).stdout.splitlines()
    dirty = bool(runner(git + ["status", "--porcelain"], **options).stdout.strip())
    return EnvironmentProvenance(
        python=python,
        packages=dict(data["packages"]),
        newton_path=Path(data["newton_path"]),
        newton_revision=revision,
        isaaclab=GitState(head=head, ancestors=frozenset(reachable) - {head}, dirty=dirty),
    )
