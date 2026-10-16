# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Draft a registration request for the registry bot from an Isaac Lab project."""

from __future__ import annotations

import getpass
import json
import os
import re
import subprocess
import sys
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib

from ..utils import print_info
from . import TASKS_ENTRY_POINT_GROUP, RegistryError, normalize_name, warn
from .environment import uv_binary
from .installer import PYPI_SIMPLE_URLS, format_command

REQUEST_SCHEMA = 1
LICENSE_FILE_PREFIXES = ("LICENSE", "LICENCE", "COPYING")
OPTIONAL_PACKAGE_FIELDS = ("status", "status_description", "tracking")
"""``[tool.isaaclab.registry]`` keys forwarded to the request only when the author sets them."""


@dataclass(frozen=True)
class PreflightCheck:
    """Result of one author-side check. The bot repeats these checks authoritatively."""

    name: str
    ok: bool
    message: str


@dataclass
class Draft:
    """A registration request and the pre-flight checks it passed or failed."""

    request: dict[str, Any]
    checks: list[PreflightCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return whether every pre-flight check passed."""
        return all(check.ok for check in self.checks)


def draft_request(
    path: Path,
    *,
    ref: str | None = None,
    repository: str | None = None,
    requested_by: str | None = None,
    pypi: bool = False,
    package_index: str | None = None,
) -> Draft:
    """Read a project and its git state, run the pre-flight checks, and build the registration request.

    Args:
        path: Project directory containing ``pyproject.toml``.
        ref: Git tag to register. Defaults to ``v<version>``.
        repository: Repository URL. Defaults to the ``origin`` remote, then ``[project.urls].Repository``,
            then the local repository itself as a ``file://`` URL (prototype convenience).
        requested_by: GitHub handle of the requester. Defaults to the first maintainer, then the OS user.
        pypi: Register the wheel published on a package index instead of a git commit.
        package_index: PEP 503 index URL for ``pypi``. Defaults to PyPI.

    Returns:
        The request (schema of the registry's ``request.schema.json``) and the check results.

    Raises:
        RegistryError: If the project cannot be read or lacks a static name and version.
    """
    path = path.expanduser().resolve()
    pyproject = path / "pyproject.toml"
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RegistryError(f"Cannot read {pyproject}: {error}") from error
    project = data.get("project")
    if not isinstance(project, dict) or not project.get("name"):
        raise RegistryError(f"{pyproject} has no [project] name.")
    if not project.get("version"):
        raise RegistryError(f"{pyproject} must declare a static [project].version to be registered.")
    name, version = normalize_name(project["name"]), str(project["version"])
    metadata = data.get("tool", {}).get("isaaclab", {}).get("registry", {})

    entry_points = project.get("entry-points", {}).get(TASKS_ENTRY_POINT_GROUP, {})
    checks = [
        PreflightCheck(
            "entry-point",
            bool(entry_points),
            ", ".join(f"{key} = {value}" for key, value in entry_points.items())
            or f'no [project.entry-points."{TASKS_ENTRY_POINT_GROUP}"] table',
        ),
        *_requirement_checks(project),
        _license_check(path, project),
    ]
    if pypi:
        index = (package_index or PYPI_SIMPLE_URLS[0]).rstrip("/")
        source: dict[str, Any] = {"type": "pypi", "index": index, "project": name, "version": version}
    else:
        source, git_checks = _git_source(path, version, ref, repository, project.get("urls", {}).get("Repository"))
        checks += git_checks

    # Author-owned fields; an empty value tells the bot to keep what the registry already has.
    maintainers = [str(handle) for handle in metadata.get("maintainers", [])]
    package = {
        "category": str(metadata.get("category", "")),
        "tags": [str(tag) for tag in metadata.get("tags", [])],
        "maintainers": maintainers,
        "homepage": str(metadata.get("homepage") or project.get("urls", {}).get("Homepage", "")),
        "media": {str(key): str(value) for key, value in metadata.get("media", {}).items()},
        "description": str(metadata.get("description", "")),
    }
    package.update({key: str(metadata[key]) for key in OPTIONAL_PACKAGE_FIELDS if key in metadata})
    request = {
        "schema": REQUEST_SCHEMA,
        "name": name,
        "requested_by": requested_by or (maintainers[0] if maintainers else getpass.getuser()),
        "source": source,
        "package": package,
    }
    return Draft(request=request, checks=checks)


def render_request(request: dict[str, Any]) -> str:
    """Serialize a registration request as TOML.

    Args:
        request: Request built by :func:`draft_request`: top-level scalars plus the ``source`` and
            ``package`` tables, whose values are strings, lists of strings, or flat string tables.

    Returns:
        TOML text that ``tomllib`` parses back to ``request``.
    """
    lines = ["# Isaac Lab registry registration request, written by 'isaaclab registry publish'."]
    tables = {key: value for key, value in request.items() if isinstance(value, dict) and key in ("source", "package")}
    lines += [f"{_toml_key(key)} = {_toml_value(value)}" for key, value in request.items() if key not in tables]
    for table, values in tables.items():
        lines += ["", f"[{table}]"]
        lines += [f"{_toml_key(key)} = {_toml_value(value)}" for key, value in values.items()]
    return "\n".join(lines) + "\n"


def submit_request(registry: Path, request_file: Path, bot_args: list[str] | None = None) -> int:
    """Hand a request to the registry bot, simulating the author's registration comment.

    Runs ``uv run <registry>/scripts/registry_bot.py register --registry <registry> --request <file>``.

    Args:
        registry: Registry repository checkout.
        request_file: Request written by :func:`render_request`.
        bot_args: Extra ``register`` options, such as ``--smoke``.

    Returns:
        The bot's exit code.

    Raises:
        RegistryError: If ``registry`` is not a registry repository.
    """
    registry = registry.expanduser().resolve()
    script = registry / "scripts" / "registry_bot.py"
    if not script.is_file():
        raise RegistryError(f"{script} does not exist; --registry must point at a registry repository checkout.")
    argv = [uv_binary(), "run", str(script), "register", "--registry", str(registry), "--request", str(request_file)]
    argv += bot_args or []
    print_info(f"Running: {format_command(argv)}")
    sys.stdout.flush()
    # The bot is a self-contained script; do not let the caller's activated environment leak into it.
    environment = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
    return subprocess.run(argv, cwd=registry, env=environment, check=False).returncode


def _requirement_checks(project: dict[str, Any]) -> list[PreflightCheck]:
    """Check the Isaac Lab requirement: plain ``isaaclab`` with an upper bound, and no ``isaaclab-dev``."""
    from packaging.requirements import InvalidRequirement, Requirement

    def parse(values: list[str]) -> list[Requirement]:
        parsed = []
        for value in values:
            try:
                parsed.append(Requirement(value))
            except InvalidRequirement:
                continue
        return parsed

    dependencies = parse(project.get("dependencies", []))
    optional = parse([value for values in project.get("optional-dependencies", {}).values() for value in values])
    isaaclab = [req for req in dependencies if normalize_name(req.name) == "isaaclab"]
    unbounded = [req for req in isaaclab if not _has_upper_bound(req.specifier)]
    if not isaaclab:
        isaaclab_check = PreflightCheck("isaaclab-requirement", False, "[project].dependencies has no 'isaaclab'")
    elif unbounded:
        isaaclab_check = PreflightCheck(
            "isaaclab-requirement", False, f"'{unbounded[0]}' needs an upper bound, for example 'isaaclab~=3.0.0'"
        )
    else:
        isaaclab_check = PreflightCheck("isaaclab-requirement", True, ", ".join(str(req) for req in isaaclab))
    development = [str(req) for req in dependencies + optional if normalize_name(req.name) == "isaaclab-dev"]
    return [
        isaaclab_check,
        PreflightCheck(
            "no-isaaclab-dev",
            not development,
            f"published requirements must not use the checkout-only 'isaaclab-dev': {', '.join(development)}"
            if development
            else "no isaaclab-dev requirement",
        ),
    ]


def _has_upper_bound(specifier: Any) -> bool:
    """Return whether a ``packaging`` specifier set bounds versions from above."""
    return any(spec.operator in ("<", "<=", "~=", "==", "===") for spec in specifier)


def _license_check(path: Path, project: dict[str, Any]) -> PreflightCheck:
    """Check for a license expression or a LICENSE file in the project or its repository root."""
    license_field = project.get("license")
    if isinstance(license_field, str) and license_field:
        return PreflightCheck("license", True, f"License-Expression {license_field}")
    directories = [path]
    toplevel = _git(path, "rev-parse", "--show-toplevel")
    if toplevel:
        directories.append(Path(toplevel))
    for directory in directories:
        for candidate in sorted(directory.iterdir()):
            if candidate.is_file() and candidate.name.upper().startswith(LICENSE_FILE_PREFIXES):
                return PreflightCheck("license", True, str(candidate.relative_to(directories[-1])))
    if isinstance(license_field, dict) and license_field.get("text"):
        return PreflightCheck("license", True, f"license text {license_field['text'][:40]!r}")
    return PreflightCheck("license", False, "no LICENSE file and no [project].license expression")


def _git_source(
    path: Path, version: str, ref: str | None, repository: str | None, declared: str | None = None
) -> tuple[dict[str, Any], list[PreflightCheck]]:
    """Return the git ``[source]`` table of the request and the git pre-flight checks.

    The repository is ``repository`` (``--repository``), else the ``origin`` remote, else ``declared`` (the
    project's ``[project.urls].Repository``), else the local repository as a ``file://`` URL.
    """
    ref = ref or f"v{version}"
    toplevel = _git(path, "rev-parse", "--show-toplevel")
    if toplevel is None:
        source = {
            "type": "git",
            "repository": repository or declared or "",
            "subdirectory": "",
            "ref": ref,
            "commit": "",
        }
        return source, [PreflightCheck("git", False, f"{path} is not inside a git repository")]
    root = Path(toplevel).resolve()
    checks = []
    origin = _git(path, "remote", "get-url", "origin")
    if not repository and origin and declared and _normalize_repository(origin) != _normalize_repository(declared):
        warn(f"Registering the 'origin' remote {origin}; [project.urls].Repository says {declared}.")
    repository = repository or origin or declared or None
    if not repository:
        repository = root.as_uri()
        warn(f"No 'origin' remote: registering the local repository {repository} (prototype only).")
    repository = _normalize_repository(repository)
    scheme = urllib.parse.urlparse(repository).scheme
    checks.append(
        PreflightCheck(
            "repository",
            scheme in ("https", "file"),
            repository if scheme in ("https", "file") else f"{repository}: the registry needs an https:// URL",
        )
    )
    commit = _git(path, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    head = _git(path, "rev-parse", "HEAD")
    checks.append(
        PreflightCheck(
            "tag",
            commit is not None,
            f"{ref} -> {commit}" if commit else f"no tag {ref}; create it with 'git tag {ref}' (or pass --ref)",
        )
    )
    if commit is not None:
        checks.append(
            PreflightCheck(
                "tag-at-head",
                commit == head,
                "the tag points at HEAD"
                if commit == head
                else f"{ref} is {commit[:12]} but HEAD is {(head or '?')[:12]}",
            )
        )
        remote = _remote_commit(repository, ref)
        checks.append(
            PreflightCheck(
                "tag-published",
                remote == commit,
                f"{repository} has {ref}"
                if remote == commit
                else f"{repository} does not have {ref} at {commit[:12]}; push it with 'git push origin {ref}'",
            )
        )
    dirty = _git(path, "status", "--porcelain", "--untracked-files=no")
    checks.append(
        PreflightCheck("clean-tree", not dirty, "no uncommitted changes" if not dirty else "uncommitted changes")
    )
    subdirectory = path.relative_to(root).as_posix()
    source = {
        "type": "git",
        "repository": repository,
        "subdirectory": "" if subdirectory == "." else subdirectory,
        "ref": ref,
        "commit": commit or "",
    }
    return source, checks


def _normalize_repository(repository: str) -> str:
    """Turn a local path into a ``file://`` URL and an scp-style remote into an https URL."""
    if "://" in repository:
        return repository
    scp = re.fullmatch(r"[\w.-]+@([\w.-]+):(.+)", repository)
    if scp:
        return f"https://{scp.group(1)}/{scp.group(2)}"
    return Path(repository).expanduser().resolve().as_uri()


def _remote_commit(repository: str, ref: str) -> str | None:
    """Return the commit a tag resolves to on the remote, peeling annotated tags."""
    try:
        result = subprocess.run(
            ["git", "ls-remote", repository, f"refs/tags/{ref}", f"refs/tags/{ref}^{{}}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    refs = {}
    for line in result.stdout.splitlines():
        sha, _, name = line.partition("\t")
        refs[name] = sha
    return refs.get(f"refs/tags/{ref}^{{}}") or refs.get(f"refs/tags/{ref}")


def _git(path: Path, *args: str) -> str | None:
    """Run a git command in ``path`` and return its stripped output, or ``None`` on failure."""
    try:
        result = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=False)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _toml_key(key: str) -> str:
    """Return a bare TOML key when possible, else a quoted one."""
    return key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else _toml_value(key)


def _toml_value(value: Any) -> str:
    """Serialize a string, boolean, integer, list, or flat table as an inline TOML value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        # JSON string escapes are valid TOML basic-string escapes; TOML also forbids a raw DEL.
        return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        return "{ " + ", ".join(f"{_toml_key(key)} = {_toml_value(item)}" for key, item in value.items()) + " }"
    raise TypeError(f"Cannot serialize {type(value).__name__} as TOML")
