# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fetch, cache, and parse the compiled registry index, and select compatible releases."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import platform
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import RegistryError, normalize_name, warn

DEFAULT_INDEX_URL = "https://isaac-sim.github.io/isaaclab-registry/index.json"
"""Placeholder location of the published index."""

INDEX_URL_ENV = "ISAACLAB_REGISTRY_URL"
INDEX_TTL_ENV = "ISAACLAB_REGISTRY_TTL"
OFFLINE_ENV = "ISAACLAB_OFFLINE"
DEFAULT_TTL_SECONDS = 3600
FETCH_TIMEOUT_SECONDS = 20
INDEX_SCHEMA = 1

_NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


@dataclass(frozen=True)
class Release:
    """One registered release of a package, as written by the registry bot."""

    version: str
    isaaclab: str = ""
    commit: str | None = None
    git_tag: str | None = None
    wheel: dict[str, str] | None = None
    summary: str = ""
    license: str = ""
    requires_python: str = ""
    entry_points: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()
    rl_libraries: tuple[str, ...] = ()
    registered_at: str = ""
    yanked: bool | str = False

    @property
    def yanked_reason(self) -> str | None:
        """Return the yank reason, ``"yanked"`` when none was given, or ``None`` for a live release."""
        if self.yanked is False or self.yanked is None or self.yanked == "":
            return None
        return self.yanked if isinstance(self.yanked, str) else "yanked"


@dataclass(frozen=True)
class Package:
    """A registered package and its releases, newest first."""

    name: str
    source: dict[str, str]
    releases: tuple[Release, ...]
    description: str = ""
    category: str = "other"
    tags: tuple[str, ...] = ()
    maintainers: tuple[str, ...] = ()
    homepage: str = ""
    media: dict[str, str] = field(default_factory=dict)
    status: str = "maintained"
    status_description: str = ""
    latest: str | None = None

    @property
    def source_type(self) -> str:
        """Return ``"git"`` or ``"pypi"``."""
        return self.source.get("type", "git")

    @property
    def summary(self) -> str:
        """Return the package description, falling back to the newest release summary."""
        if self.description:
            return self.description
        return self.releases[0].summary if self.releases else ""

    def release(self, version: str) -> Release | None:
        """Return the release with the given version, comparing PEP 440 versions when possible."""
        for release in self.releases:
            if _same_version(release.version, version):
                return release
        return None


@dataclass(frozen=True)
class BlockEntry:
    """A client-honored blocklist entry: a whole package, one commit, or one version."""

    name: str
    reason: str
    commit: str | None = None
    version: str | None = None

    def matches(self, name: str, release: Release | None = None) -> bool:
        """Return whether this entry blocks a package, or a specific release of it."""
        if release is None:
            return self.covers(name)
        return self.covers(name, commit=release.commit, version=release.version)

    def covers(self, name: str, *, commit: str | None = None, version: str | None = None) -> bool:
        """Return whether this entry blocks a package, or the install of it at ``commit`` or ``version``.

        Unlike :meth:`matches`, this needs no index release, so it also applies to recorded installs whose
        release has since left the index.
        """
        if normalize_name(self.name) != normalize_name(name):
            return False
        if self.commit is None and self.version is None:
            return True
        if self.commit is not None and commit == self.commit:
            return True
        return self.version is not None and version is not None and _same_version(version, self.version)


@dataclass(frozen=True)
class RegistryIndex:
    """The compiled ``index.json`` of a registry."""

    name: str
    location: str
    packages: dict[str, Package]
    blocklist: tuple[BlockEntry, ...] = ()
    generated_at: str = ""
    source_commit: str = ""

    def get(self, name: str) -> Package | None:
        """Return a package by (unnormalized) name."""
        return self.packages.get(normalize_name(name))

    def block_for(self, name: str, release: Release | None = None) -> BlockEntry | None:
        """Return the blocklist entry covering a package (``release=None``) or one of its releases."""
        return next((entry for entry in self.blocklist if entry.matches(name, release)), None)

    def block_for_install(self, name: str, commit: str | None, version: str | None) -> BlockEntry | None:
        """Return the blocklist entry covering an installed (or recorded) package at ``commit`` or ``version``."""
        return next((entry for entry in self.blocklist if entry.covers(name, commit=commit, version=version)), None)


@dataclass(frozen=True)
class Selection:
    """A release chosen for installation and the warnings the user must see."""

    package: Package
    release: Release
    warnings: tuple[str, ...] = ()


def resolve_index_location(value: str | None = None) -> str:
    """Return the index URL or local file to read.

    Precedence is ``value`` (the ``--index`` option), then ``ISAACLAB_REGISTRY_URL``, then the default URL.
    A directory is treated as a registry checkout and resolves to its ``dist/index.json`` (or to an
    ``index.json`` directly inside it, as served by the demo).

    Args:
        value: Explicit URL or path, if any.

    Returns:
        An ``http(s)://`` or ``file://`` URL, or an absolute local path.
    """
    location = value or os.environ.get(INDEX_URL_ENV) or DEFAULT_INDEX_URL
    if "://" in location:
        return location
    path = Path(location).expanduser().resolve()
    if path.is_dir():
        for candidate in (path / "dist" / "index.json", path / "index.json"):
            if candidate.is_file():
                return str(candidate)
        return str(path / "dist" / "index.json")
    return str(path)


def cache_path_for(url: str) -> Path:
    """Return the cache file for a remote index URL."""
    cache_home = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return cache_home / "isaaclab" / "registry" / f"{digest}.json"


def load_index(location: str, *, refresh: bool = False, offline: bool = False) -> RegistryIndex:
    """Read and parse a registry index, using the on-disk cache for remote URLs.

    Local paths and ``file://`` URLs are read directly and never cached. Remote indexes are cached for
    ``ISAACLAB_REGISTRY_TTL`` seconds (default one hour). When the network fails, a stale cache is used
    with a warning.

    Args:
        location: Value returned by :func:`resolve_index_location`.
        refresh: Fetch a remote index even when the cache is fresh.
        offline: Use the cached copy of a remote index regardless of its age. ``ISAACLAB_OFFLINE=1`` has
            the same effect.

    Returns:
        The parsed index.

    Raises:
        RegistryError: If the index cannot be read or does not have the expected schema.
    """
    scheme = urllib.parse.urlparse(location).scheme
    if scheme in ("http", "https"):
        return _load_remote(location, refresh=refresh, offline=offline or os.environ.get(OFFLINE_ENV) == "1")
    path = Path(urllib.request.url2pathname(urllib.parse.urlparse(location).path)) if scheme == "file" else None
    path = path or Path(location)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise RegistryError(f"Cannot read the registry index at {path}: {error.strerror or error}") from error
    return _decode_index(raw, location)


def parse_index(data: Any, location: str = "") -> RegistryIndex:
    """Build a :class:`RegistryIndex` from the decoded ``index.json``.

    Every string is stripped of control characters, which could otherwise rewrite the terminal output a user
    reviews before installing. A package or release with a malformed field is left out with a warning. A
    malformed blocklist makes the whole index unusable, since installing without it would not be safe.

    Args:
        data: Decoded JSON document.
        location: Where the document came from, for messages.

    Returns:
        The parsed index. Releases are sorted newest first regardless of their order in the file.

    Raises:
        RegistryError: If the schema version is unsupported, or the package list or the blocklist is malformed.
    """
    if not isinstance(data, dict) or data.get("schema") != INDEX_SCHEMA:
        schema = data.get("schema") if isinstance(data, dict) else None
        raise RegistryError(
            f"Unsupported registry index schema {schema!r} at {location or 'the given location'};"
            f" this client reads schema {INDEX_SCHEMA}."
        )
    entries, blocks = data.get("packages", []), data.get("blocklist", [])
    if not isinstance(entries, list) or not isinstance(blocks, list):
        raise RegistryError(f"Malformed registry index at {location}: 'packages' and 'blocklist' must be lists.")
    packages: dict[str, Package] = {}
    for number, entry in enumerate(entries, start=1):
        try:
            package = _parse_package(entry)
        except (KeyError, TypeError, ValueError) as error:
            label = _clean(entry.get("name")) if isinstance(entry, dict) and isinstance(entry.get("name"), str) else ""
            where = f" at {location}" if location else ""
            warn(f"Ignoring package {label or f'#{number}'} of the registry index{where}: {error}")
            continue
        packages[package.name] = package
    try:
        blocklist = tuple(
            BlockEntry(
                name=_text(entry["name"], "name") or _invalid("an empty name"),
                reason=_text(entry.get("reason"), "reason") or "blocked by the registry",
                commit=_text(entry.get("commit"), "commit") or None,
                version=_text(entry.get("version"), "version") or None,
            )
            for entry in blocks
        )
        name = _text(data.get("name"), "name") or "registry"
        generated_at, source_commit = (
            _text(data.get("generated_at"), "generated_at"),
            _text(data.get("source_commit"), "source_commit"),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RegistryError(f"Malformed registry index at {location}: invalid field {error}") from error
    return RegistryIndex(
        name=name,
        location=location,
        packages=dict(sorted(packages.items())),
        blocklist=blocklist,
        generated_at=generated_at,
        source_commit=source_commit,
    )


def compatibility_problem(
    release: Release, isaaclab_version: str | None, python_version: str | None = None
) -> str | None:
    """Explain why a release cannot run on an Isaac Lab and Python, or return ``None`` when it can.

    The release's ``isaaclab`` field is a specifier on the Isaac Lab *release* version (``VERSION`` in a
    checkout), not on the version of the ``isaaclab`` distribution.

    Args:
        release: Release to check.
        isaaclab_version: Isaac Lab release version of the target environment.
        python_version: Python version of the target environment. Defaults to the running interpreter's.

    Returns:
        A short reason, such as ``"needs isaaclab >=4,<5 (have 3.0.0)"``, or ``None``.
    """
    from packaging.specifiers import InvalidSpecifier, SpecifierSet
    from packaging.version import InvalidVersion, Version

    if not release.isaaclab:
        return "declares no isaaclab range"
    if isaaclab_version is None:
        return "cannot determine the installed Isaac Lab release"
    try:
        if not SpecifierSet(release.isaaclab).contains(Version(isaaclab_version), prereleases=True):
            return f"needs isaaclab {release.isaaclab} (have {isaaclab_version})"
    except (InvalidSpecifier, InvalidVersion):
        return f"has an invalid isaaclab range {release.isaaclab!r}"
    python_version = python_version or platform.python_version()
    if release.requires_python:
        try:
            if not SpecifierSet(release.requires_python).contains(python_version, prereleases=True):
                return f"needs Python {release.requires_python} (have {python_version})"
        except InvalidSpecifier:
            return f"has an invalid requires-python {release.requires_python!r}"
    return None


def latest_compatible(
    index: RegistryIndex, package: Package, isaaclab_version: str | None, python_version: str | None = None
) -> Release | None:
    """Return the newest release that is live, not blocked, and compatible, if any."""
    if index.block_for(package.name) is not None:
        return None
    for release in package.releases:
        if (
            release.yanked_reason is None
            and index.block_for(package.name, release) is None
            and compatibility_problem(release, isaaclab_version, python_version) is None
        ):
            return release
    return None


def select_release(
    index: RegistryIndex,
    name: str,
    isaaclab_version: str | None,
    version: str | None = None,
    force: bool = False,
    python_version: str | None = None,
) -> Selection:
    """Choose the release to install, or refuse with a reason.

    Without ``version``, the newest live, unblocked, compatible release is chosen. An explicit ``version``
    may be yanked (with a warning). Blocklisted packages and releases are always refused. Incompatible
    releases are refused unless ``force`` is set. End-of-life packages install with a warning.

    Args:
        index: Parsed registry index.
        name: Package name.
        isaaclab_version: Isaac Lab release version of the target environment.
        version: Exact version requested with ``NAME==VERSION``, if any.
        force: Install an incompatible release anyway.
        python_version: Python version of the target environment. Defaults to the running interpreter's.

    Returns:
        The selected release and the warnings to print.

    Raises:
        RegistryError: If the package is unknown or no acceptable release exists.
    """
    block = index.block_for(name)
    if block is not None:
        raise RegistryError(f"'{name}' is blocklisted by the registry: {block.reason}")
    package = index.get(name)
    if package is None:
        raise RegistryError(f"'{name}' is not in the {index.name} registry. Try 'isaaclab registry search {name}'.")

    warnings: list[str] = []
    if version is not None:
        release = package.release(version)
        if release is None:
            available = ", ".join(r.version for r in package.releases) or "none"
            raise RegistryError(f"'{package.name}' has no release {version} (available: {available}).")
        block = index.block_for(package.name, release)
        if block is not None:
            raise RegistryError(f"{package.name} {release.version} is blocklisted by the registry: {block.reason}")
        if release.yanked_reason is not None:
            warnings.append(f"{package.name} {release.version} was yanked: {release.yanked_reason}")
    else:
        candidates = [
            release
            for release in package.releases
            if release.yanked_reason is None and index.block_for(package.name, release) is None
        ]
        if not candidates:
            raise RegistryError(f"'{package.name}' has no installable release (all are yanked or blocklisted).")
        release = next(
            (r for r in candidates if compatibility_problem(r, isaaclab_version, python_version) is None),
            candidates[0],
        )

    problem = compatibility_problem(release, isaaclab_version, python_version)
    if problem is not None:
        if not force:
            raise RegistryError(
                f"{package.name} {release.version} is not compatible: it {problem}. Pass --force to install it anyway."
            )
        warnings.append(f"--force: installing {package.name} {release.version}, which {problem}.")
    if package.status == "end-of-life":
        replacement = f" {package.status_description}" if package.status_description else ""
        warnings.append(f"'{package.name}' is end-of-life.{replacement}")
    return Selection(package=package, release=release, warnings=tuple(warnings))


def _decode_index(raw: bytes, location: str) -> RegistryIndex:
    """Decode and parse the bytes of an ``index.json``."""
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RegistryError(f"The registry index at {location} is not valid JSON: {error}") from error
    return parse_index(data, location)


def _load_remote(url: str, *, refresh: bool, offline: bool) -> RegistryIndex:
    """Return a remote index, honoring the cache, TTL, refresh, and offline settings.

    The cache only ever holds an index that parsed: a bad response (a captive portal page, a truncated file)
    is reported and the last good copy is used instead, and a corrupt cache file is fetched again.
    """
    cache = cache_path_for(url)
    cached, age = None, 0.0
    if cache.is_file():
        age = time.time() - cache.stat().st_mtime
        try:
            cached = _decode_index(cache.read_bytes(), url)
        except (OSError, RegistryError) as error:
            warn(f"Ignoring the cached registry index {cache}: {error}")
    if offline:
        if cached is None:
            raise RegistryError(
                f"Offline mode is on and there is no usable cached copy of {url} ({cache}). Run once with network"
                " access (or with --refresh)."
            )
        return cached
    try:
        ttl = float(os.environ.get(INDEX_TTL_ENV, DEFAULT_TTL_SECONDS))
    except ValueError:
        ttl = DEFAULT_TTL_SECONDS
    if not refresh and cached is not None and age < ttl:
        return cached
    stale = f"using the cached copy from {int(age / 60)} min ago ({cache})"
    try:
        raw = _fetch(url)
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as error:
        # ValueError covers malformed URLs (http.client.InvalidURL); HTTPException covers truncated responses.
        if cached is not None:
            warn(f"Could not fetch {url} ({error}); {stale}.")
            return cached
        raise RegistryError(f"Could not fetch the registry index from {url}: {error}") from error
    try:
        index = _decode_index(raw, url)
    except RegistryError as error:
        if cached is not None:
            warn(f"{error} Ignoring this response and {stale}.")
            return cached
        raise
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, cache)
    return index


def _fetch(url: str) -> bytes:
    """Download a URL with a timeout and an identifying User-Agent."""
    request = urllib.request.Request(url, headers={"User-Agent": "isaaclab-registry-client/0 (prototype)"})
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
        return response.read()


def _clean(text: str) -> str:
    """Replace control and other non-printable characters, which a terminal would interpret, with ``?``."""
    return "".join(character if character.isprintable() else "?" for character in text)


def _invalid(what: str) -> Any:
    """Raise the error a malformed index field reports."""
    raise ValueError(what)


def _text(value: Any, field_name: str) -> str:
    """Return a string field of the index (``""`` when absent), without control characters."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, not {type(value).__name__}")
    return _clean(value)


def _texts(value: Any, field_name: str) -> tuple[str, ...]:
    """Return a list-of-strings field of the index."""
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list, not {type(value).__name__}")
    return tuple(_text(item, field_name) for item in value)


def _table(value: Any, field_name: str) -> dict[str, str]:
    """Return a string-to-string table of the index."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be an object, not {type(value).__name__}")
    return {_text(key, field_name): _text(item, f"{field_name}.{key}") for key, item in value.items()}


def _parse_release(entry: Any) -> Release:
    """Build a :class:`Release` from one ``releases`` entry, validating every field it uses."""
    from packaging.version import Version

    if not isinstance(entry, dict):
        raise TypeError("a release must be an object")
    version = _text(entry.get("version"), "version")
    Version(version)  # InvalidVersion is a ValueError
    yanked = entry.get("yanked", False)
    if not isinstance(yanked, bool):
        yanked = _text(yanked, "yanked")
    return Release(
        version=version,
        isaaclab=_text(entry.get("isaaclab"), "isaaclab"),
        commit=_text(entry.get("commit"), "commit") or None,
        git_tag=_text(entry.get("git_tag"), "git_tag") or None,
        wheel=_table(entry["wheel"], "wheel") if entry.get("wheel") is not None else None,
        summary=_text(entry.get("summary"), "summary"),
        license=_text(entry.get("license"), "license"),
        requires_python=_text(entry.get("requires_python"), "requires_python"),
        entry_points=_texts(entry.get("entry_points"), "entry_points"),
        tasks=_texts(entry.get("tasks"), "tasks"),
        rl_libraries=_texts(entry.get("rl_libraries"), "rl_libraries"),
        registered_at=_text(entry.get("registered_at"), "registered_at"),
        yanked=yanked,
    )


def _parse_package(entry: Any) -> Package:
    """Build a :class:`Package` from one ``packages`` entry of the index.

    Raises:
        KeyError, TypeError, ValueError: If the name, the source, or another field is malformed. A malformed
            release is left out with a warning instead.
    """
    if not isinstance(entry, dict):
        raise TypeError("a package must be an object")
    name = _text(entry["name"], "name")
    if not _NAME.match(name):
        raise ValueError(f"invalid name {name!r} (expected a PEP 503 normalized name)")
    source = _table(entry["source"], "source")
    if source.get("type") not in ("git", "pypi"):
        raise ValueError(f"unsupported source type {source.get('type')!r}")
    releases = []
    for raw in entry.get("releases") or []:
        try:
            releases.append(_parse_release(raw))
        except (KeyError, TypeError, ValueError) as error:
            warn(f"Ignoring a release of {name} in the registry index: {error}")
    return Package(
        name=name,
        source=source,
        releases=tuple(sorted(releases, key=lambda release: _version_key(release.version), reverse=True)),
        description=_text(entry.get("description"), "description"),
        category=_text(entry.get("category"), "category") or "other",
        tags=_texts(entry.get("tags"), "tags"),
        maintainers=_texts(entry.get("maintainers"), "maintainers"),
        homepage=_text(entry.get("homepage"), "homepage"),
        media=_table(entry.get("media"), "media"),
        status=_text(entry.get("status"), "status") or "maintained",
        status_description=_text(entry.get("status_description"), "status_description"),
        latest=_text(entry.get("latest"), "latest") or None,
    )


def _version_key(version: str) -> tuple[int, Any]:
    """Sort key that orders PEP 440 versions and puts unparsable ones first (oldest)."""
    from packaging.version import InvalidVersion, Version

    try:
        return (1, Version(version))
    except InvalidVersion:
        return (0, version)


def _same_version(left: str, right: str) -> bool:
    """Return whether two version strings denote the same PEP 440 version."""
    from packaging.version import InvalidVersion, Version

    try:
        return Version(left) == Version(right)
    except InvalidVersion:
        return left == right
