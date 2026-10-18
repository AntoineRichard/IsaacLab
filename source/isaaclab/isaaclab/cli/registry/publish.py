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
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import tomllib

from ..utils import print_info
from . import TASKS_ENTRY_POINT_GROUP, RegistryError, normalize_name
from .install import uv_binary

REQUEST_SCHEMA = 1
"""Version of the request format that the registry bot reads."""

DEFAULT_PACKAGE_INDEX = "https://pypi.org/simple"
"""PEP 503 index of ``--pypi`` requests."""

OPTIONAL_PACKAGE_FIELDS = ("status", "status_description", "tracking")
"""``[tool.isaaclab.registry]`` keys forwarded to the request only when the author sets them."""


def draft_request(
    path: Path,
    *,
    ref: str | None = None,
    repository: str | None = None,
    requested_by: str | None = None,
    pypi: bool = False,
    package_index: str | None = None,
) -> dict[str, Any]:
    """Build the registration request of a project.

    Three problems that the registry bot would reject stop the request here: no ``isaaclab.tasks`` entry point,
    an Isaac Lab requirement without an upper bound (or on an ``isaaclab-*`` package, such as the checkout-only
    ``isaaclab-dev``), and a git tag that is missing or not pushed. The bot runs these and every other check again,
    authoritatively.

    Args:
        path: Project directory containing ``pyproject.toml``.
        ref: Git tag to register. Defaults to ``v<version>``.
        repository: Repository URL. Defaults to the ``origin`` remote.
        requested_by: GitHub handle of the requester. Defaults to the first maintainer, then the OS user.
        pypi: Register the wheel published on a package index instead of a git commit.
        package_index: PEP 503 index URL for ``pypi``. Defaults to PyPI.

    Returns:
        The request, in the format of the registry's ``request.schema.json``.

    Raises:
        RegistryError: If the project cannot be registered as it is.
    """
    path = path.expanduser().resolve()
    pyproject = path / "pyproject.toml"
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RegistryError(f"Cannot read {pyproject}: {error}") from error
    project = data.get("project", {})
    if not project.get("name") or not project.get("version"):
        raise RegistryError(f"{pyproject} must declare a static [project] name and version to be registered.")
    name, version = normalize_name(project["name"]), str(project["version"])
    if not project.get("entry-points", {}).get(TASKS_ENTRY_POINT_GROUP):
        raise RegistryError(f'{pyproject} has no [project.entry-points."{TASKS_ENTRY_POINT_GROUP}"] table.')
    _check_isaaclab_requirement(project, pyproject)
    if pypi:
        index = (package_index or DEFAULT_PACKAGE_INDEX).rstrip("/")
        source = {"type": "pypi", "index": index, "project": name, "version": version}
    else:
        source = _git_source(path, ref or f"v{version}", repository)

    metadata = data.get("tool", {}).get("isaaclab", {}).get("registry", {})
    maintainers = [str(handle) for handle in metadata.get("maintainers", [])]
    # Author-owned fields; an empty value tells the bot to keep what the registry already has.
    package = {
        "category": str(metadata.get("category", "")),
        "tags": [str(tag) for tag in metadata.get("tags", [])],
        "maintainers": maintainers,
        "homepage": str(metadata.get("homepage") or project.get("urls", {}).get("Homepage", "")),
        "media": {str(key): str(value) for key, value in metadata.get("media", {}).items()},
        "description": str(metadata.get("description", "")),
        **{key: str(metadata[key]) for key in OPTIONAL_PACKAGE_FIELDS if key in metadata},
    }
    return {
        "schema": REQUEST_SCHEMA,
        "name": name,
        "requested_by": requested_by or (maintainers[0] if maintainers else getpass.getuser()),
        "source": source,
        "package": package,
    }


def render_request(request: dict[str, Any]) -> str:
    """Serialize a registration request as TOML.

    Args:
        request: Request built by :func:`draft_request`: top-level scalars plus the ``source`` and
            ``package`` tables, whose values are strings, lists of strings, or flat string tables.

    Returns:
        TOML text that ``tomllib`` parses back to ``request``.
    """
    lines = ["# Isaac Lab registry registration request, written by 'isaaclab registry publish'."]
    tables = {key: value for key, value in request.items() if key in ("source", "package")}
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
    print_info(f"Running: {shlex.join(['uv', *argv[1:]])}")
    sys.stdout.flush()
    # The bot is a self-contained script; do not let the caller's activated environment leak into it.
    environment = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
    return subprocess.run(argv, cwd=registry, env=environment, check=False).returncode


def _check_isaaclab_requirement(project: dict[str, Any], pyproject: Path) -> None:
    """Require a plain ``isaaclab`` dependency with an upper bound, so the package claims no future release.

    Raises:
        RegistryError: If the requirement is missing or has no upper bound, or an ``isaaclab-*`` package is required.
    """
    from packaging.requirements import InvalidRequirement, Requirement

    dependencies = project.get("dependencies", [])
    optional = [value for values in project.get("optional-dependencies", {}).values() for value in values]
    try:
        requirements = [Requirement(value) for value in [*dependencies, *optional]]
    except InvalidRequirement as error:
        raise RegistryError(f"{pyproject} has an invalid requirement: {error}") from error
    # Isaac Lab is published as one 'isaaclab' distribution; 'isaaclab-dev' only exists in a checkout.
    if name := next((req.name for req in requirements if normalize_name(req.name).startswith("isaaclab-")), None):
        raise RegistryError(f"{pyproject} requires '{name}': depend on the 'isaaclab' release instead.")
    isaaclab = [req for req in requirements[: len(dependencies)] if normalize_name(req.name) == "isaaclab"]
    upper = ("<", "<=", "~=", "==", "===")
    if not isaaclab or not all(any(spec.operator in upper for spec in req.specifier) for req in isaaclab):
        found = ", ".join(f"'{req}'" for req in isaaclab) or "none"
        raise RegistryError(
            f"{pyproject}: the 'isaaclab' dependency needs an upper bound, for example 'isaaclab~=3.0.0' (found:"
            f" {found})."
        )


def _git_source(path: Path, ref: str, repository: str | None) -> dict[str, str]:
    """Return the git ``[source]`` table of the request, pinned to the commit of a pushed tag.

    Raises:
        RegistryError: If ``path`` is not in a git repository, the repository URL is unknown, or the tag is
            missing locally or on the remote.
    """
    toplevel = _git(path, "rev-parse", "--show-toplevel")
    if toplevel is None:
        raise RegistryError(f"{path} is not inside a git repository.")
    repository = repository or _git(path, "remote", "get-url", "origin")
    if not repository:
        raise RegistryError("The repository has no 'origin' remote; pass its public URL with --repository.")
    repository = _normalize_repository(repository, Path(toplevel))
    commit = _git(path, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    if commit is None:
        raise RegistryError(f"There is no tag {ref}: create it with 'git tag {ref}' and push it, or pass --ref.")
    # The bot builds the commit from a clone of the repository, so the tag must be published there.
    listing = _git(path, "ls-remote", repository, f"refs/tags/{ref}", f"refs/tags/{ref}^{{}}")
    if listing is None:
        raise RegistryError(f"Cannot reach {repository} with 'git ls-remote'; check the URL and your access to it.")
    remote = {name: sha for sha, _, name in (line.partition("\t") for line in listing.splitlines())}
    if remote.get(f"refs/tags/{ref}^{{}}", remote.get(f"refs/tags/{ref}")) != commit:
        raise RegistryError(f"{repository} does not have {ref} at {commit[:12]}: push it with 'git push origin {ref}'.")
    subdirectory = path.relative_to(Path(toplevel).resolve()).as_posix()
    return {
        "type": "git",
        "repository": repository,
        "subdirectory": "" if subdirectory == "." else subdirectory,
        "ref": ref,
        "commit": commit,
    }


def _normalize_repository(repository: str, toplevel: Path) -> str:
    """Return a remote as a URL: an scp-style remote as https, and a local path as ``file://``.

    Like git, a relative path is read from the top of the working tree, ``toplevel``.
    """
    if "://" in repository:
        return repository
    scp = re.fullmatch(r"[\w.-]+@([\w.-]+):(.+)", repository)
    if scp:
        return f"https://{scp.group(1)}/{scp.group(2)}"
    return (toplevel / Path(repository).expanduser()).resolve().as_uri()


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
    """Serialize a string, integer, list, or flat table as an inline TOML value."""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        # JSON string escapes are valid TOML basic-string escapes; TOML also forbids a raw DEL.
        return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        items = ", ".join(f"{_toml_key(key)} = {_toml_value(item)}" for key, item in value.items())
        return f"{{ {items} }}" if items else "{}"
    raise TypeError(f"Cannot serialize {type(value).__name__} as TOML")
