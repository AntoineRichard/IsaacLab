# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Decide where registry packages go: the user's uv project or the running environment."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib

from . import DEPENDENCY_GROUP, RegistryError, is_isaaclab_distribution, normalize_name, warn

CHECKOUT_PROJECT_NAME = "isaaclab-dev"
"""``[project].name`` of the Isaac Lab checkout's root ``pyproject.toml``."""

_INSPECT_INTERPRETER = """
import importlib.metadata, importlib.util, json, pathlib, platform, sys, sysconfig

# Mirrors checkout_root() and isaaclab_release_version(): a checkout reports its VERSION file.
release = None
spec = importlib.util.find_spec("isaaclab")
if spec is not None and spec.submodule_search_locations:
    package = pathlib.Path(list(spec.submodule_search_locations)[0]).resolve()
    root = package.parents[2] if len(package.parents) > 2 else None
    if root is not None and root / "source" / "isaaclab" / "isaaclab" == package and (root / "VERSION").is_file():
        release = (root / "VERSION").read_text(encoding="utf-8").strip()
if release is None:
    try:
        release = importlib.metadata.version("isaaclab").split("+", 1)[0]
    except importlib.metadata.PackageNotFoundError:
        pass
paths = sorted({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")})
print(json.dumps({"prefix": sys.prefix, "paths": paths, "release": release, "python": platform.python_version()}))
"""
"""Script run with another interpreter (``--python``) to describe its environment."""


@dataclass(frozen=True)
class Target:
    """Where an install, uninstall, or listing applies.

    Attributes:
        mode: ``"project"`` to edit the user's uv project with ``uv add``/``uv remove``, or ``"env"`` to
            change an environment with ``uv pip``.
        python: Interpreter of the target environment, passed to ``uv pip --python``.
        prefix: Prefix of the target environment; the env-mode state file lives there.
        site_paths: Directories to scan for installed distributions, or ``None`` for the running
            interpreter's ``sys.path``.
        project_root: Root of the user's project (project mode only).
        uv_cwd: Working directory for ``uv pip`` in env mode: the Isaac Lab checkout when the target is the
            checkout's own environment, so its indexes and overrides apply, or ``None`` for a fresh temporary
            directory.
        uvx: Whether the target is a throwaway ``uvx`` environment inside the uv cache.
        isaaclab_version: Isaac Lab release version of the target environment, for the compatibility gate.
        python_version: Python version of the target environment, for the compatibility gate.
    """

    mode: str
    python: str
    prefix: Path
    site_paths: tuple[str, ...] | None = None
    project_root: Path | None = None
    uv_cwd: Path | None = None
    uvx: bool = False
    isaaclab_version: str | None = None
    python_version: str | None = None

    def describe(self) -> str:
        """Return a one-line description for summaries."""
        if self.mode == "project":
            return f"project {self.project_root} (dependency group '{DEPENDENCY_GROUP}', uv add)"
        details = []
        if self.uv_cwd is not None:
            details.append(f"Isaac Lab checkout; uv runs from {self.uv_cwd}")
        if self.site_paths is not None:
            details.append(f"interpreter {self.python}")
        suffix = f" ({'; '.join(details)})" if details else ""
        return f"environment {self.prefix} (uv pip){suffix}"


def checkout_root() -> Path | None:
    """Return the Isaac Lab source checkout that provides the running ``isaaclab``, if any.

    This is the single "is this a checkout" predicate used for both the release version and the env-mode
    working directory. A wheel install resolves ``ISAACLAB_ROOT`` to the package directory, which has no
    ``source/isaaclab/isaaclab`` tree.
    """
    from isaaclab.paths import ISAACLAB_ROOT

    if (ISAACLAB_ROOT / "VERSION").is_file() and (ISAACLAB_ROOT / "source" / "isaaclab" / "isaaclab").is_dir():
        return ISAACLAB_ROOT
    return None


def isaaclab_release_version() -> str | None:
    """Return the Isaac Lab *release* version that registry compatibility ranges refer to.

    A checkout reports its ``VERSION`` file (the ``isaaclab`` distribution there has an unrelated version,
    such as 37.x). A wheel install reports the public part of the ``isaaclab`` distribution version.

    Returns:
        The release version, or ``None`` when Isaac Lab is not installed.
    """
    root = checkout_root()
    if root is not None:
        return (root / "VERSION").read_text(encoding="utf-8").strip()
    try:
        return importlib.metadata.version("isaaclab").split("+", 1)[0]
    except importlib.metadata.PackageNotFoundError:
        return None


def uv_binary() -> str:
    """Return the uv executable: ``$UV`` (set by ``uv run``) when it exists, else ``uv`` on ``PATH``.

    Raises:
        RegistryError: If uv cannot be found.
    """
    configured = os.environ.get("UV")
    if configured and (Path(configured).is_file() or shutil.which(configured)):
        return configured
    uv = shutil.which("uv")
    if not uv:
        stale = f" (UV={configured} does not exist)" if configured else ""
        raise RegistryError(f"uv was not found{stale}. Install it (https://docs.astral.sh/uv/) or set UV=/path/to/uv.")
    if configured:
        warn(f"UV={configured} does not exist; using {uv}.")
    return uv


def find_project(start: Path) -> tuple[Path, dict[str, Any]] | None:
    """Return the nearest directory at or above ``start`` whose ``pyproject.toml`` has a ``[project]`` table.

    Args:
        start: Directory to start from.

    Returns:
        The project root and its parsed ``pyproject.toml``, or ``None``.
    """
    for directory in (start, *start.parents):
        pyproject = directory / "pyproject.toml"
        if not pyproject.is_file():
            continue
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise RegistryError(f"Cannot read {pyproject}: {error}") from error
        if isinstance(data.get("project"), dict):
            return directory, data
    return None


def project_environment(root: Path) -> Path:
    """Return the environment uv uses for a project: ``$UV_PROJECT_ENVIRONMENT`` or ``<root>/.venv``."""
    configured = os.environ.get("UV_PROJECT_ENVIRONMENT")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else root / path
    return root / ".venv"


def detect_target(mode: str = "auto", project: str | None = None, python: str | None = None) -> Target:
    """Choose the install target.

    In ``auto`` mode, the nearest project above the working directory is used when its uv environment is the one
    running this command, and a command run with ``uv run --project ROOT`` from elsewhere uses ``ROOT``, the
    project that owns the running environment. The Isaac Lab checkout is never a project target: neither its
    tracked ``pyproject.toml`` files nor its environment (which ``uv add`` would sync against another project's
    lock) are touched. Everything else uses env mode. An explicit ``project`` (or ``mode="project"``) never
    falls back to env mode: when that project cannot be the target, the command fails instead of changing the
    running environment.

    Args:
        mode: ``"auto"``, ``"project"``, or ``"env"``.
        project: Directory to search for the project from, instead of the working directory.
        python: Explicit interpreter of another environment to manage (env mode only).

    Returns:
        The resolved target.

    Raises:
        RegistryError: If the requested mode cannot apply here.
    """
    if python is not None:
        if mode == "project":
            raise RegistryError("--python selects an environment, so it cannot be combined with --mode project.")
        return _foreign_target(python)

    start = Path(project).expanduser().resolve() if project else Path.cwd()
    if project and not start.is_dir():
        raise RegistryError(f"--project {project} is not a directory.")
    prefix = Path(sys.prefix)
    checkout = checkout_root()
    versions = {"isaaclab_version": isaaclab_release_version(), "python_version": platform.python_version()}
    found = find_project(start) if mode in ("auto", "project") else None
    if not project and mode in ("auto", "project") and (found is None or not _runs_in(found[0], prefix)):
        found = _environment_owner(prefix) or found
    explicit = mode == "project" or (mode == "auto" and project is not None)
    if found is not None:
        root, data = found
        runs_here = _runs_in(root, prefix)
        checkout_problem = _checkout_problem(root, data, checkout, prefix)
        if runs_here and checkout_problem is None:
            return Target(mode="project", python=sys.executable, prefix=prefix, project_root=root, **versions)
        uv_run = f"Run it as 'uv run --project {root} isaaclab registry …'."
        if explicit and checkout_problem is not None:
            part_of_lab = is_isaaclab_distribution(str(data["project"].get("name", ""))) or (
                checkout is not None and _is_within(root, checkout)
            )
            hint = "Without --project and --mode project, packages go into the running environment."
            raise RegistryError(f"{checkout_problem} {hint if part_of_lab else uv_run}")
        if explicit:
            raise RegistryError(
                f"This command runs in {prefix}, not in the environment of {root} ({project_environment(root)})."
                f" {uv_run}"
            )
    if explicit:
        raise RegistryError(f"No pyproject.toml with a [project] table was found at or above {start}.")

    # Run uv from the checkout only for the checkout's own environment, where its indexes and overrides apply.
    uv_cwd = checkout if checkout is not None and _runs_in(checkout, prefix) else None
    return Target(
        mode="env", python=sys.executable, prefix=prefix, uv_cwd=uv_cwd, uvx=_is_uvx_environment(prefix), **versions
    )


def _checkout_problem(root: Path, data: dict[str, Any], checkout: Path | None, prefix: Path) -> str | None:
    """Explain why a project must not be edited with ``uv add``, or return ``None`` when it may be.

    Args:
        root: Project root.
        data: Parsed ``pyproject.toml`` of the project.
        checkout: Isaac Lab checkout providing the running ``isaaclab``, if any.
        prefix: Running environment.
    """
    name = normalize_name(str(data["project"].get("name", "")))
    if name == CHECKOUT_PROJECT_NAME:
        return f"{root} is the Isaac Lab checkout: its tracked pyproject.toml and uv.lock are never edited."
    if is_isaaclab_distribution(name):
        return f"{root} is the Isaac Lab package {name}: its pyproject.toml is never edited."
    if checkout is not None and _is_within(root, checkout):
        return f"{root} is part of the Isaac Lab checkout {checkout}: its tracked files are never edited."
    if checkout is not None and _runs_in(checkout, prefix):
        # For example Docker images that set an absolute UV_PROJECT_ENVIRONMENT for every project.
        return (
            f"This environment ({prefix}) belongs to the Isaac Lab checkout {checkout}, which 'uv add' in {root}"
            " would sync against that project's lock."
        )
    return None


def _environment_owner(prefix: Path) -> tuple[Path, dict[str, Any]] | None:
    """Return the project whose uv environment is ``prefix``, such as the one ``uv run --project ROOT`` selects."""
    root = prefix.parent
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file() or not _runs_in(root, prefix):
        return None
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return (root, data) if isinstance(data.get("project"), dict) else None


def _foreign_target(python: str) -> Target:
    """Describe the environment of another interpreter: its prefix, site directories, Isaac Lab, and Python."""
    executable = Path(python).expanduser()
    if not executable.is_absolute():
        executable = Path(shutil.which(python) or executable).absolute()
    try:
        result = subprocess.run(
            [str(executable), "-I", "-c", _INSPECT_INTERPRETER], capture_output=True, text=True, check=True
        )
        info = json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        detail = error.stderr.strip() if isinstance(error, subprocess.CalledProcessError) else error
        raise RegistryError(f"Cannot inspect the interpreter {executable}: {detail}") from error
    return Target(
        mode="env",
        python=str(executable),
        prefix=Path(info["prefix"]),
        site_paths=tuple(info["paths"]),
        isaaclab_version=info["release"],
        python_version=info["python"],
    )


def _is_uvx_environment(prefix: Path) -> bool:
    """Return whether ``prefix`` is an ephemeral ``uvx`` environment inside the uv cache."""
    cache = os.environ.get("UV_CACHE_DIR")
    if not cache:
        try:
            result = subprocess.run([uv_binary(), "cache", "dir"], capture_output=True, text=True, check=False)
        except (RegistryError, OSError):
            return False
        cache = result.stdout.strip() if result.returncode == 0 else ""
    if not cache:
        return False
    try:
        return prefix.resolve().is_relative_to(Path(cache).expanduser().resolve())
    except OSError:
        return False


def _runs_in(root: Path, prefix: Path) -> bool:
    """Return whether ``prefix`` is the uv environment of the project at ``root``."""
    return _same_path(project_environment(root), prefix)


def _same_path(left: Path, right: Path) -> bool:
    """Return whether two paths resolve to the same location."""
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


def _is_within(path: Path, directory: Path) -> bool:
    """Return whether ``path`` is ``directory`` or lies inside it, after resolving both."""
    try:
        return path.resolve().is_relative_to(directory.resolve())
    except OSError:
        return False
