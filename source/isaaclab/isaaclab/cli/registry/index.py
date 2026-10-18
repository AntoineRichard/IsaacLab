# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fetch and parse the compiled registry index, and select compatible releases."""

from __future__ import annotations

import http.client
import json
import os
import platform
import urllib.request
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from . import RegistryError, normalize_name

DEFAULT_INDEX_URL = "https://isaac-sim.github.io/IsaacLab-Registry/index.json"
"""Placeholder location of the published index."""

INDEX_URL_ENV = "ISAACLAB_REGISTRY_URL"
"""Environment variable that overrides the index location."""

INDEX_SCHEMA = 1
"""Version of the ``index.json`` format this client reads."""


@dataclass(frozen=True)
class Release:
    """One registered release of a package, as written by the registry bot."""

    version: str
    isaaclab: str = ""
    """Specifier on the Isaac Lab release version, for example ``~=3.0.0``."""
    commit: str | None = None
    git_tag: str | None = None
    wheel: dict[str, str] | None = None
    """``filename``, ``url`` and ``sha256`` of the wheel of a PyPI release."""
    summary: str = ""
    license: str = ""
    requires_python: str = ""
    tasks: list[str] = field(default_factory=list)
    rl_libraries: list[str] = field(default_factory=list)
    registered_at: str = ""
    yanked: bool | str = False
    """``False``, or the reason the release was yanked."""

    @property
    def yanked_reason(self) -> str | None:
        """Return why the release was yanked, or ``None`` for a live release."""
        if self.yanked is False:
            return None
        return self.yanked if isinstance(self.yanked, str) and self.yanked else "yanked"


@dataclass(frozen=True)
class Package:
    """A registered package and its releases, newest first."""

    name: str
    source: dict[str, str]
    """``type`` (``git`` or ``pypi``) and where the code lives."""
    releases: tuple[Release, ...] = ()
    description: str = ""
    category: str = "other"
    tags: list[str] = field(default_factory=list)
    maintainers: list[str] = field(default_factory=list)
    homepage: str = ""
    media: dict[str, str] = field(default_factory=dict)
    status: str = "maintained"
    status_description: str = ""

    @property
    def source_type(self) -> str:
        """Return ``"git"`` or ``"pypi"``."""
        return self.source["type"]

    @property
    def summary(self) -> str:
        """Return the package description, falling back to the newest release summary."""
        return self.description or (self.releases[0].summary if self.releases else "")

    def release(self, version: str) -> Release | None:
        """Return the release with the given PEP 440 version, if any."""
        return next((release for release in self.releases if _same_version(release.version, version)), None)


@dataclass(frozen=True)
class BlockEntry:
    """A blocklist entry: a whole package, or one of its commits or versions."""

    name: str
    reason: str = ""
    commit: str | None = None
    version: str | None = None


@dataclass(frozen=True)
class RegistryIndex:
    """The compiled ``index.json`` of a registry."""

    name: str
    packages: dict[str, Package]
    blocklist: tuple[BlockEntry, ...] = ()
    generated_at: str = ""

    def get(self, name: str) -> Package | None:
        """Return a package by (unnormalized) name."""
        return self.packages.get(normalize_name(name))

    def blocked(self, name: str, commit: str | None = None, version: str | None = None) -> BlockEntry | None:
        """Return the blocklist entry that covers a package, or the install of it at ``commit`` or ``version``.

        Without ``commit`` and ``version``, only entries that block the whole package match.
        """
        for entry in self.blocklist:
            if normalize_name(entry.name) != normalize_name(name):
                continue
            whole = entry.commit is None and entry.version is None
            same_version = version is not None and entry.version is not None and _same_version(version, entry.version)
            if whole or same_version or (commit is not None and entry.commit == commit):
                return entry
        return None


def resolve_index_location(value: str | None = None) -> str:
    """Return the index URL or local file to read: ``value``, else ``$ISAACLAB_REGISTRY_URL``, else the default.

    Args:
        value: The ``--index`` option: a URL, an ``index.json`` file, or a registry checkout (its ``dist/index.json``).

    Returns:
        A URL, or an absolute local path.
    """
    location = value or os.environ.get(INDEX_URL_ENV) or DEFAULT_INDEX_URL
    if "://" in location:
        return location
    path = Path(location).expanduser().resolve()
    return str(path / "dist" / "index.json" if path.is_dir() else path)


def load_index(location: str) -> RegistryIndex:
    """Read and parse a registry index.

    Args:
        location: Value returned by :func:`resolve_index_location`.

    Returns:
        The parsed index.

    Raises:
        RegistryError: If the index cannot be read or is malformed.
    """
    try:
        if "://" in location:
            request = urllib.request.Request(location, headers={"User-Agent": "isaaclab-registry-client/0 (prototype)"})
            with urllib.request.urlopen(request, timeout=20) as response:
                raw = response.read()
        else:
            raw = Path(location).read_bytes()
        data = json.loads(raw)
    except (OSError, ValueError, http.client.HTTPException) as error:
        message = f"Cannot read the registry index at {location}: {error}"
        if location == DEFAULT_INDEX_URL:
            message += f" (the default index is a placeholder: set {INDEX_URL_ENV} or pass --index)"
        raise RegistryError(message) from error
    return parse_index(data, location)


def parse_index(data: Any, location: str = "the registry index") -> RegistryIndex:
    """Build a :class:`RegistryIndex` from the decoded ``index.json``; one malformed entry refuses the whole index.

    Args:
        data: Decoded JSON document.
        location: Where the document came from, for messages.

    Returns:
        The parsed index, with each package's releases sorted newest first.

    Raises:
        RegistryError: If the schema version is unsupported or an entry is malformed.
    """
    from packaging.version import Version

    if not isinstance(data, dict) or data.get("schema") != INDEX_SCHEMA:
        raise RegistryError(f"{location} is not a registry index of schema {INDEX_SCHEMA}.")
    try:
        packages = {}
        for entry in data.get("packages", []):
            releases = sorted(
                (_build(Release, release) for release in entry.get("releases", [])),
                key=lambda release: Version(release.version),
                reverse=True,
            )
            package = _build(Package, {**entry, "releases": tuple(releases)})
            _check_types(package)
            packages[normalize_name(package.name)] = package
        blocklist = tuple(
            _build(BlockEntry, {**entry, "reason": entry.get("reason") or "blocked by the registry"})
            for entry in data.get("blocklist", [])
        )
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise RegistryError(f"{location} is malformed ({type(error).__name__}: {error}).") from error
    return RegistryIndex(
        data.get("name") or "registry", dict(sorted(packages.items())), blocklist, data.get("generated_at", "")
    )


def compatibility_problem(release: Release, isaaclab_version: str | None) -> str | None:
    """Explain why a release cannot run on this Isaac Lab and Python, or return ``None`` when it can.

    The release's ``isaaclab`` field is a specifier on the Isaac Lab *release* version (``VERSION`` in a
    checkout), not on the version of the ``isaaclab`` distribution.

    Args:
        release: Release to check.
        isaaclab_version: Isaac Lab release version of the target environment.

    Returns:
        A short reason, such as ``"needs isaaclab >=4,<5 (have 3.0.0)"``, or ``None``.
    """
    from packaging.specifiers import SpecifierSet

    if not release.isaaclab:
        return "declares no isaaclab range"
    if isaaclab_version is None:
        return "cannot determine the installed Isaac Lab release"
    python_version = platform.python_version()
    try:
        if not SpecifierSet(release.isaaclab).contains(isaaclab_version, prereleases=True):
            return f"needs isaaclab {release.isaaclab} (have {isaaclab_version})"
        python_range = SpecifierSet(release.requires_python)
        if not python_range.contains(python_version, prereleases=True):
            return f"needs Python {release.requires_python} (have {python_version})"
    except ValueError as error:  # InvalidSpecifier and InvalidVersion
        return f"has an invalid version range ({error})"
    return None


def latest_compatible(index: RegistryIndex, package: Package, isaaclab_version: str | None) -> Release | None:
    """Return the newest release that is live, not blocklisted, and compatible, if any."""
    releases = _installable(index, package)
    return next((release for release in releases if compatibility_problem(release, isaaclab_version) is None), None)


def select_release(
    index: RegistryIndex, name: str, isaaclab_version: str | None, version: str | None = None, force: bool = False
) -> tuple[Package, Release, list[str]]:
    """Choose the release to install, following the policy of the registry docs page, or refuse with a reason.

    Args:
        index: Parsed registry index.
        name: Package name.
        isaaclab_version: Isaac Lab release version of the target environment.
        version: Exact version requested with ``NAME==VERSION``, if any.
        force: Install an incompatible release anyway.

    Returns:
        The package, the selected release, and the warnings to print.

    Raises:
        RegistryError: If the package is unknown or no acceptable release exists.
    """
    if block := index.blocked(name):
        raise RegistryError(f"'{name}' is blocklisted by the registry: {block.reason}")
    package = index.get(name)
    if package is None:
        raise RegistryError(f"'{name}' is not in the {index.name} registry. Try 'isaaclab registry search {name}'.")

    warnings = []
    if version is not None:
        release = package.release(version)
        if release is None:
            available = ", ".join(release.version for release in package.releases) or "none"
            raise RegistryError(f"'{package.name}' has no release {version} (available: {available}).")
        if block := index.blocked(package.name, release.commit, release.version):
            raise RegistryError(f"{package.name} {release.version} is blocklisted by the registry: {block.reason}")
        if release.yanked_reason is not None:
            warnings.append(f"{package.name} {release.version} was yanked: {release.yanked_reason}")
    else:
        candidates = _installable(index, package)
        if not candidates:
            raise RegistryError(f"'{package.name}' has no installable release (all are yanked or blocklisted).")
        release = latest_compatible(index, package, isaaclab_version) or candidates[0]

    problem = compatibility_problem(release, isaaclab_version)
    if problem is not None:
        if not force:
            raise RegistryError(
                f"{package.name} {release.version} is not compatible: it {problem}. Pass --force to install it anyway."
            )
        warnings.append(f"--force: installing {package.name} {release.version}, which {problem}.")
    if package.status == "end-of-life":
        warnings.append(f"'{package.name}' is end-of-life. {package.status_description}".strip())
    return package, release, warnings


def _installable(index: RegistryIndex, package: Package) -> list[Release]:
    """Return the releases of a package that are neither yanked nor blocklisted, newest first."""
    if index.blocked(package.name):
        return []
    return [
        release
        for release in package.releases
        if release.yanked_reason is None and not index.blocked(package.name, release.commit, release.version)
    ]


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Build a dataclass from the entries of ``data`` that it declares as fields, ignoring the others."""
    return cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data})


def _check_types(package: Package) -> None:
    """Raise ``ValueError`` for the field types that :func:`_build` lets through but the commands rely on."""
    tables = [package.source, package.media, *(release.wheel or {} for release in package.releases)]
    lists = [package.tags, package.maintainers, *(f for r in package.releases for f in (r.tasks, r.rl_libraries))]
    if not all(isinstance(table, dict) and all(isinstance(v, str) for v in table.values()) for table in tables):
        raise ValueError(f"a source, media, or wheel entry of {package.name} is not a table of strings")
    if not all(isinstance(values, list) and all(isinstance(v, str) for v in values) for values in lists):
        raise ValueError(
            f"a tags, maintainers, tasks, or rl_libraries entry of {package.name} is not a list of strings"
        )
    if package.source_type not in ("git", "pypi"):
        raise ValueError(f"{package.name} has no source type 'git' or 'pypi'")


def _same_version(left: str, right: str) -> bool:
    """Return whether two version strings denote the same PEP 440 version."""
    from packaging.version import InvalidVersion, Version

    try:
        return Version(left) == Version(right)
    except InvalidVersion:
        return left == right
