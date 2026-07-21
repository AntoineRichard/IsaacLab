# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locked environment provenance for the Kamino DVI benchmark."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any, Self

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
class PackageLocation:
    """Resolved import and distribution location for one Python package."""

    module_path: str
    distribution_path: str
    direct_url: dict[str, Any]


@dataclass(frozen=True)
class EnvironmentProvenance:
    """Software provenance captured from one locked Python environment."""

    python: Path
    packages: dict[str, str]
    newton_path: Path
    newton_revision: str
    isaaclab: GitState
    isaaclab_newton: PackageLocation

    def replace(self, **changes) -> Self:
        """Return a copy with selected fields replaced."""
        return replace(self, **changes)


def validate_package_location(location: PackageLocation, repo_root: Path) -> None:
    """Require a package import to resolve to the launched Isaac Lab checkout."""
    package_root = (repo_root / "source" / "isaaclab_newton").resolve()
    expected_module = package_root / "isaaclab_newton" / "__init__.py"
    if Path(location.module_path).resolve() != expected_module:
        raise ValueError("isaaclab_newton import is outside the launched checkout")
    if not Path(location.distribution_path).is_absolute():
        raise ValueError("isaaclab_newton distribution path must be absolute")
    if not isinstance(location.direct_url, dict):
        raise ValueError("isaaclab_newton direct-url metadata must be a mapping")
    if location.direct_url.get("dir_info", {}).get("editable") is True:
        if location.direct_url.get("url") != package_root.as_uri():
            raise ValueError("isaaclab_newton editable URL does not match the launched checkout")


def python_executable(repo_root: Path, label: EnvironmentLabel) -> Path:
    """Return the interpreter for a locked benchmark environment."""
    directory = ".venv-current" if label is EnvironmentLabel.CURRENT else ".venv-pr3570"
    return repo_root / directory / "bin" / "python"


def validate_environment(
    matrix: BenchmarkMatrix,
    label: EnvironmentLabel,
    provenance: EnvironmentProvenance,
    repo_root: Path,
) -> None:
    """Validate that an environment matches the immutable experiment revisions.

    Args:
        matrix: Validated benchmark matrix.
        label: Environment role being validated.
        provenance: Captured interpreter and repository provenance.
        repo_root: Isaac Lab worktree whose source packages must be imported.

    Raises:
        ValueError: If Isaac Lab or Newton does not match the approved revisions.
    """
    if provenance.isaaclab.dirty:
        raise ValueError("IsaacLab worktree has dirty tracked files")
    validate_package_location(provenance.isaaclab_newton, repo_root)
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

import isaaclab_newton
import newton

packages = {
    (distribution.metadata.get("Name") or "").lower(): distribution.version
    for distribution in importlib.metadata.distributions()
    if distribution.metadata.get("Name")
}
distribution = importlib.metadata.distribution("newton")
direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
revision = direct_url.get("vcs_info", {}).get("commit_id")
isaaclab_distribution = importlib.metadata.distribution("isaaclab-newton")
isaaclab_direct_url = json.loads(isaaclab_distribution.read_text("direct_url.json") or "{}")
isaaclab_distribution_path = Path(isaaclab_distribution.locate_file("")).resolve()
print(json.dumps({
    "packages": packages,
    "newton_path": str(Path(newton.__file__).resolve()),
    "newton_revision": revision,
    "isaaclab_newton": {
        "module_path": str(Path(isaaclab_newton.__file__).resolve()),
        "distribution_path": str(isaaclab_distribution_path),
        "direct_url": isaaclab_direct_url,
    },
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
    probe_env = os.environ.copy()
    probe_env["PYTHONPATH"] = f"{repo_root / 'source' / 'isaaclab_newton'}:{repo_root / 'source' / 'isaaclab_tasks'}"
    options = {"check": True, "capture_output": True, "text": True, "env": probe_env}
    probe = runner([str(python), "-c", _PROBE_SCRIPT], **options)
    data = json.loads(probe.stdout.splitlines()[-1])
    revision = data.get("newton_revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("Newton installation does not expose a 40-character VCS revision")

    git = ["git", "-C", str(repo_root)]
    head = runner(git + ["rev-parse", "HEAD"], **options).stdout.strip()
    reachable = runner(git + ["rev-list", "HEAD"], **options).stdout.splitlines()
    dirty = bool(runner(git + ["status", "--porcelain", "--untracked-files=no"], **options).stdout.strip())
    return EnvironmentProvenance(
        python=python,
        packages=dict(data["packages"]),
        newton_path=Path(data["newton_path"]),
        newton_revision=revision,
        isaaclab=GitState(head=head, ancestors=frozenset(reachable) - {head}, dirty=dirty),
        isaaclab_newton=PackageLocation(**data["isaaclab_newton"]),
    )
