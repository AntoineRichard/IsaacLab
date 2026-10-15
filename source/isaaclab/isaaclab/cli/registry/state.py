# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Track registry installs and report the installed task packages against the index."""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tomllib

from . import DEPENDENCY_GROUP, TASKS_ENTRY_POINT_GROUP, RegistryError, normalize_name
from .environment import Target
from .index import Package, RegistryIndex, Release, latest_compatible

STATE_FILE_NAME = "isaaclab-registry.json"
"""Env-mode state file, stored in the environment prefix so it survives ``uv sync``."""

STATE_SCHEMA = 1


@dataclass(frozen=True)
class InstalledDistribution:
    """An installed distribution, with its PEP 610 origin when it has one.

    Attributes:
        name: Normalized distribution name.
        version: Installed version.
        entry_points: ``isaaclab.tasks`` entry points, as ``name = value``.
        url: Origin URL from ``direct_url.json``, if any.
        commit: VCS commit from ``direct_url.json``, if any.
        local: Whether it is an editable or local-directory install.
        requirement: Requirement that reinstalls exactly this distribution: commit-pinned for VCS installs,
            the origin URL for other direct installs, and ``name==version`` otherwise.
    """

    name: str
    version: str
    entry_points: tuple[str, ...] = ()
    url: str | None = None
    commit: str | None = None
    local: bool = False
    requirement: str = ""


@dataclass(frozen=True)
class InstalledPackage:
    """One row of ``isaaclab registry installed``."""

    name: str
    version: str | None
    status: str
    commit: str | None = None
    recorded_in: str | None = None
    note: str = ""

    def to_json(self) -> dict[str, Any]:
        """Return the row as a JSON-serializable dictionary."""
        return asdict(self)


def state_path(prefix: Path) -> Path:
    """Return the env-mode state file of an environment."""
    return prefix / STATE_FILE_NAME


def load_state(prefix: Path) -> dict[str, dict[str, Any]]:
    """Return the packages recorded by env-mode installs, keyed by normalized name.

    Raises:
        RegistryError: If the state file exists but cannot be parsed.
    """
    path = state_path(prefix)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RegistryError(f"Cannot read the registry state file {path}: {error}") from error
    if not isinstance(data, dict) or data.get("schema") != STATE_SCHEMA:
        raise RegistryError(f"Unsupported registry state file {path}; expected schema {STATE_SCHEMA}.")
    return dict(data.get("packages", {}))


def save_state(prefix: Path, packages: dict[str, dict[str, Any]]) -> None:
    """Write the env-mode state file atomically, or delete it once no package is recorded."""
    path = state_path(prefix)
    if not packages:
        path.unlink(missing_ok=True)
        return
    document = {"schema": STATE_SCHEMA, "packages": dict(sorted(packages.items()))}
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def record_install(prefix: Path, name: str, entry: dict[str, Any]) -> None:
    """Add or replace one package in the state file, stamping ``installed_at``."""
    packages = load_state(prefix)
    packages[normalize_name(name)] = {
        **{key: value for key, value in entry.items() if value is not None},
        "installed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    save_state(prefix, packages)


def forget_install(prefix: Path, name: str) -> bool:
    """Remove one package from the state file; return whether it was recorded."""
    packages = load_state(prefix)
    if packages.pop(normalize_name(name), None) is None:
        return False
    save_state(prefix, packages)
    return True


def installed_distributions(site_paths: tuple[str, ...] | None = None) -> dict[str, importlib.metadata.Distribution]:
    """Return every installed distribution keyed by normalized name (first on the path wins).

    Args:
        site_paths: Directories to scan, or ``None`` for the running interpreter's ``sys.path``.
    """
    importlib.invalidate_caches()
    found = (
        importlib.metadata.distributions()
        if site_paths is None
        else importlib.metadata.distributions(path=list(site_paths))
    )
    distributions: dict[str, importlib.metadata.Distribution] = {}
    for distribution in found:
        name = distribution.metadata["Name"]
        if name:
            distributions.setdefault(normalize_name(name), distribution)
    return distributions


def describe_distribution(distribution: importlib.metadata.Distribution) -> InstalledDistribution:
    """Summarize a distribution: version, task entry points, and its PEP 610 ``direct_url.json`` origin."""
    try:
        origin = json.loads(distribution.read_text("direct_url.json") or "{}")
    except json.JSONDecodeError:
        origin = {}
    entry_points = tuple(
        f"{entry_point.name} = {entry_point.value}"
        for entry_point in distribution.entry_points
        if entry_point.group == TASKS_ENTRY_POINT_GROUP
    )
    name = normalize_name(distribution.metadata["Name"])
    vcs_info = origin.get("vcs_info") or {}
    url = origin.get("url")
    if not url:
        requirement = f"{name}=={distribution.version}"
    elif vcs_info.get("commit_id"):
        requirement = f"{name} @ {vcs_info.get('vcs', 'git')}+{url}@{vcs_info['commit_id']}"
    else:
        requirement = f"{name} @ {url}"
    if url and origin.get("subdirectory"):
        requirement += f"#subdirectory={origin['subdirectory']}"
    return InstalledDistribution(
        name=name,
        version=distribution.version,
        entry_points=entry_points,
        url=url,
        commit=vcs_info.get("commit_id"),
        local="dir_info" in origin,
        requirement=requirement,
    )


def task_distributions(site_paths: tuple[str, ...] | None = None) -> dict[str, InstalledDistribution]:
    """Return the installed distributions that declare an ``isaaclab.tasks`` entry point."""
    described = (describe_distribution(dist) for dist in installed_distributions(site_paths).values())
    return {dist.name: dist for dist in described if dist.entry_points}


def project_group(root: Path) -> list[str]:
    """Return the normalized names listed in a project's ``isaaclab-registry`` dependency group."""
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    names = []
    for requirement in data.get("dependency-groups", {}).get(DEPENDENCY_GROUP, []):
        match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement.strip()) if isinstance(requirement, str) else None
        if match:
            names.append(normalize_name(match.group(0)))
    return names


def installed_packages(
    target: Target, index: RegistryIndex | None, isaaclab_version: str | None, python_version: str | None = None
) -> list[InstalledPackage]:
    """Join installed task packages, the env state file, the project group, and the index.

    Each package gets one status:

    * ``registry``: matches a release in the index;
    * ``outdated (X available)``: matches a release, and a newer compatible one exists;
    * ``blocked``: the registry blocklists the package, or its installed (or recorded) commit or version;
    * ``drift``: the package is in the index, but its commit, version, or wheel is not;
    * ``local``: an editable or local-directory install, such as the user's own project;
    * ``direct``: installed by ``isaaclab registry install`` from a raw spec that is not in the index;
    * ``unknown``: installed by ``isaaclab registry install``, but the index is unavailable;
    * ``unmanaged``: installed outside the registry;
    * ``missing``: recorded by the registry but not installed.

    Args:
        target: Environment or project to report on.
        index: Registry index, or ``None`` when it is unavailable.
        isaaclab_version: Isaac Lab release version, for the ``outdated`` check.
        python_version: Python version of the target, for the ``outdated`` check.

    Returns:
        One row per package, sorted by name.
    """
    distributions = task_distributions(target.site_paths)
    recorded: dict[str, tuple[str, dict[str, Any]]] = {}
    if target.mode == "env":
        recorded.update({name: ("env", entry) for name, entry in load_state(target.prefix).items()})
    if target.project_root is not None:
        recorded.update({name: ("project", {}) for name in project_group(target.project_root)})

    rows = []
    for name in sorted({*distributions, *recorded}):
        where, entry = recorded.get(name, (None, {}))
        dist = distributions.get(name)
        if dist is None:
            block = index.block_for_install(name, entry.get("commit"), entry.get("version")) if index else None
            if block is not None:  # never point at 'restore' for a package the registry has since blocked
                status, hint = "blocked", f"{block.reason}; drop it with 'isaaclab registry uninstall {name}'"
            else:
                status, hint = "missing", "run 'uv sync'" if where == "project" else "run 'isaaclab registry restore'"
            rows.append(InstalledPackage(name, entry.get("version"), status, entry.get("commit"), where, hint))
            continue
        status, note = _status(dist, where, entry, index, isaaclab_version, python_version)
        rows.append(InstalledPackage(name, dist.version, status, dist.commit, where, note))
    return rows


def _status(
    dist: InstalledDistribution,
    recorded_in: str | None,
    entry: dict[str, Any],
    index: RegistryIndex | None,
    isaaclab_version: str | None,
    python_version: str | None,
) -> tuple[str, str]:
    """Return the status and note of one installed task distribution.

    Args:
        dist: The installed distribution.
        recorded_in: ``"env"`` or ``"project"`` when the registry installed it, else ``None``.
        entry: Its env-mode state entry (empty in project mode).
        index: Registry index, or ``None`` when it is unavailable.
        isaaclab_version: Isaac Lab release version, for the ``outdated`` check.
        python_version: Python version of the target, for the ``outdated`` check.
    """
    if dist.local:
        return "local", dist.url or ""
    if index is None:
        return ("unknown" if recorded_in else "unmanaged"), "index unavailable"
    block = index.block_for_install(dist.name, dist.commit, dist.version)
    if block is not None:
        return "blocked", f"{block.reason}; remove it with 'isaaclab registry uninstall {dist.name}'"
    package = index.get(dist.name)
    if package is None:
        return ("direct" if recorded_in else "unmanaged"), ""
    release = _matching_release(package, dist, entry)
    if release is None:
        if dist.commit:
            found = f"commit {dist.commit[:12]}"
        elif package.release(dist.version) is not None:
            found = f"the installed wheel of {dist.version}"
        else:
            found = f"version {dist.version}"
        return "drift", f"{found} is not a registered release"
    notes = []
    if release.yanked_reason is not None:
        notes.append(f"yanked: {release.yanked_reason}")
    latest = latest_compatible(index, package, isaaclab_version, python_version)
    if latest is not None and latest is not release and _is_newer(latest.version, release.version):
        return f"outdated ({latest.version} available)", "; ".join(notes)
    return "registry", "; ".join(notes)


def _matching_release(package: Package, dist: InstalledDistribution, entry: dict[str, Any]) -> Release | None:
    """Return the index release an installed distribution came from, if any.

    A PyPI release matches only when the wheel came from the registered URL (PEP 610 ``direct_url.json``) and,
    when the registry recorded its install, with the registered sha256.
    """
    if dist.commit is not None:
        return next((release for release in package.releases if release.commit == dist.commit), None)
    if package.source_type != "pypi":
        return None
    release = package.release(dist.version)
    wheel = (release.wheel if release else None) or {}
    if dist.url and wheel.get("url") and dist.url != wheel["url"]:
        return None
    if entry.get("sha256") and wheel.get("sha256") and entry["sha256"] != wheel["sha256"]:
        return None
    return release


def _is_newer(candidate: str, current: str) -> bool:
    """Return whether one PEP 440 version is newer than another."""
    from packaging.version import InvalidVersion, Version

    try:
        return Version(candidate) > Version(current)
    except InvalidVersion:
        return False
