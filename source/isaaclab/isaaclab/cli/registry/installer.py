# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build and run the uv commands that install and remove registry packages."""

from __future__ import annotations

import importlib.machinery
import importlib.metadata
import importlib.util
import json
import re
import shlex
import subprocess
import sys
import tempfile
import urllib.parse
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import tomllib

from ..utils import print_info
from . import DEPENDENCY_GROUP, RegistryError, is_isaaclab_distribution, normalize_name, warn
from .environment import Target, uv_binary
from .index import Package, Release
from .state import (
    InstalledDistribution,
    describe_distribution,
    installed_distributions,
    load_state,
    record_install,
)

PYPI_SIMPLE_URLS = ("https://pypi.org/simple", "https://pypi.python.org/simple")
"""The default PEP 503 index (PyPI)."""

NO_SOURCES_PACKAGES = ("isaaclab", "isaaclab-dev")
"""Packages whose ``[tool.uv.sources]`` in a registry package are ignored: Isaac Lab is a peer."""

RL_LIBRARIES = {
    "rsl_rl": ("rsl_rl", "rsl-rl"),
    "skrl": ("skrl", "skrl"),
    "rl_games": ("rl_games", "rl-games"),
    "sb3": ("stable_baselines3", "sb3"),
    "rlinf": ("rlinf", "rlinf"),
    "torchrl": ("torchrl", "torchrl"),
}
"""Registry RL library name -> (import name, Isaac Lab extra)."""

_NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SUBDIRECTORY = re.compile(r"^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$")


@dataclass(frozen=True)
class InstallPlan:
    """A requirement to install and the facts recorded about it.

    Attributes:
        name: Normalized distribution name.
        requirement: PEP 508 requirement passed to uv.
        version: Version the registry recorded, if any.
        commit: Git commit the requirement pins, if any.
        sha256: sha256 the downloaded wheel must have; uv refuses any other file.
    """

    name: str
    requirement: str
    version: str | None = None
    commit: str | None = None
    sha256: str | None = None


def plan_for_release(package: Package, release: Release) -> InstallPlan:
    """Build the uv requirement for an index release from validated fields.

    Git releases become ``name @ git+<repository>@<commit>[#subdirectory=<dir>]``. PyPI releases become
    ``name @ <wheel url>`` with the wheel's registered sha256: the exact file the registry checked is installed,
    and the author's index is never added to the resolution, so it cannot serve any other package.

    Raises:
        RegistryError: If a field the requirement is built from is malformed.
    """
    name = normalize_name(package.name)
    if not _NAME.match(name):
        raise RegistryError(f"Refusing malformed package name {package.name!r} from the index.")
    source = package.source
    if package.source_type == "git":
        repository = source.get("repository", "")
        parsed = urllib.parse.urlparse(repository)
        if (
            parsed.scheme not in ("https", "http", "file", "ssh")
            or any(character in repository for character in " \t\n#?")
            or "@" in parsed.path
        ):
            raise RegistryError(f"Refusing malformed repository URL {repository!r} for {name}.")
        if not release.commit or not _COMMIT.match(release.commit):
            raise RegistryError(f"{name} {release.version} has no valid 40-hex commit in the index.")
        requirement = f"{name} @ git+{repository}@{release.commit}"
        subdirectory = source.get("subdirectory") or ""
        if subdirectory:
            if not _SUBDIRECTORY.match(subdirectory) or ".." in subdirectory.split("/"):
                raise RegistryError(f"Refusing malformed subdirectory {subdirectory!r} for {name}.")
            requirement += f"#subdirectory={subdirectory}"
        return InstallPlan(name, requirement, version=release.version, commit=release.commit)
    if package.source_type == "pypi":
        from packaging.utils import InvalidWheelFilename, parse_wheel_filename
        from packaging.version import InvalidVersion, Version

        wheel = release.wheel or {}
        url, filename, sha256 = wheel.get("url", ""), wheel.get("filename", ""), wheel.get("sha256", "")
        if not _SHA256.match(sha256):
            raise RegistryError(
                f"{name} {release.version} has no valid wheel sha256 in the index; refusing to install it."
            )
        parsed = urllib.parse.urlparse(url)
        if (
            parsed.scheme not in ("https", "http", "file")
            or any(character.isspace() or character == "#" for character in url)
            or PurePosixPath(urllib.parse.unquote(parsed.path)).name != filename
        ):
            raise RegistryError(f"Refusing malformed wheel URL {url!r} for {name}.")
        try:
            wheel_name, wheel_version, _, _ = parse_wheel_filename(filename)
            matches = normalize_name(wheel_name) == name and wheel_version == Version(release.version)
        except (InvalidWheelFilename, InvalidVersion):
            matches = False
        if not matches:
            raise RegistryError(f"The index's wheel {filename!r} is not {name} {release.version}.")
        return InstallPlan(name, f"{name} @ {url}", version=str(wheel_version), sha256=sha256)
    raise RegistryError(f"{name} has an unsupported source type {package.source_type!r}.")


def is_raw_spec(spec: str) -> bool:
    """Return whether an install argument is a direct spec (URL or local path) rather than an index name.

    Index names never contain a path separator, while URLs (``NAME @ git+https://…``) and paths do.
    """
    return "/" in spec or "\\" in spec or spec.startswith((".", "~")) or spec.endswith(".whl")


def plan_for_spec(spec: str) -> InstallPlan:
    """Build a plan for a Pi-style direct spec, installed as-is without the index.

    Accepted forms are ``NAME @ <url>`` (for example ``NAME @ git+https://host/repo@<ref>``), a local project
    directory, and a local wheel file.

    Raises:
        RegistryError: If the spec does not name its package or the path does not exist.
    """
    name, separator, url = spec.partition("@")
    if separator and "://" in url and re.fullmatch(r"\s*[A-Za-z0-9][A-Za-z0-9._-]*\s*", name):
        return InstallPlan(normalize_name(name.strip()), f"{name.strip()} @ {url.strip()}")
    if "://" in spec or spec.startswith("git+"):
        raise RegistryError(
            f"Write URL specs as 'NAME @ {spec}' so the installed package can be tracked and uninstalled."
        )
    path = Path(spec).expanduser().resolve()
    if path.is_dir() and (path / "pyproject.toml").is_file():
        data = tomllib.loads((path / "pyproject.toml").read_text(encoding="utf-8"))
        name = data.get("project", {}).get("name")
        if not name:
            raise RegistryError(f"{path / 'pyproject.toml'} has no [project] name.")
    elif path.is_file() and path.suffix == ".whl":
        name = path.name.split("-", 1)[0]
    else:
        raise RegistryError(f"{spec} is neither a registry package name, a 'NAME @ URL' spec, nor a local project.")
    return InstallPlan(normalize_name(name), f"{normalize_name(name)} @ {path.as_uri()}")


def run_uv(argv: list[str], *, cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess[str]:
    """Run uv.

    Args:
        argv: Full command line, starting with the uv executable.
        cwd: Working directory.
        capture: Return stdout and stderr without printing them. Otherwise stdout goes to the terminal and
            stderr is echoed while it is also kept for diagnosis.

    Returns:
        The completed process; the caller checks the return code.
    """
    sys.stdout.flush()
    if capture:
        return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)
    with subprocess.Popen(argv, cwd=cwd, stderr=subprocess.PIPE, text=True) as process:
        lines = []
        try:
            for line in process.stderr:
                sys.stderr.write(line)
                lines.append(line)
            process.wait()
            if lines and not lines[-1].endswith("\n"):
                sys.stderr.write("\n")
        except KeyboardInterrupt:
            process.terminate()
            raise
    return subprocess.CompletedProcess(argv, process.returncode, stdout=None, stderr="".join(lines))


def format_command(argv: list[str]) -> str:
    """Return a shell-quoted command line, with the uv executable shown as ``uv``."""
    return shlex.join(["uv", *argv[1:]])


def peer_overrides(distributions: dict[str, importlib.metadata.Distribution]) -> list[str]:
    """Return ``name[<active extras>]==<version>`` for every installed ``isaaclab`` / ``isaaclab-*`` package.

    Overriding the peers keeps the installed Isaac Lab in place whatever range a registry package declares.
    The extras matter: an override without them would drop the extra's dependencies.
    """
    installed = {name: distribution.version for name, distribution in distributions.items()}
    lines = []
    for name, distribution in sorted(distributions.items()):
        if is_isaaclab_distribution(name):
            extras = active_extras(distribution, installed)
            lines.append(f"{name}{'[' + ','.join(extras) + ']' if extras else ''}=={distribution.version}")
    return lines


def active_extras(distribution: importlib.metadata.Distribution, installed: dict[str, str]) -> list[str]:
    """Return the extras of a distribution whose requirements (on this platform) are all installed.

    Args:
        distribution: Distribution to inspect.
        installed: Installed versions keyed by normalized name.
    """
    from packaging.requirements import InvalidRequirement, Requirement

    requirements = []
    for raw in distribution.requires or ():
        try:
            requirements.append(Requirement(raw))
        except InvalidRequirement:
            continue
    active = []
    for extra in distribution.metadata.get_all("Provides-Extra") or ():
        needed = [
            requirement
            for requirement in requirements
            if requirement.marker is not None
            and requirement.marker.evaluate({"extra": extra})
            and not requirement.marker.evaluate({"extra": ""})
        ]
        if needed and all(_is_satisfied(requirement, installed) for requirement in needed):
            active.append(extra)
    return active


def install_into_environment(
    plans: list[InstallPlan], target: Target, *, dry_run: bool = False, keep_conflicts: bool = False
) -> None:
    """Install packages with ``uv pip install`` and record them in the environment's state file.

    Isaac Lab packages are pinned through overrides and every other installed distribution through
    constraints, so the install can only add distributions and replace the requested ones. Hash-pinned plans
    (PyPI releases) go through a requirements file with their sha256, which uv verifies. In an Isaac Lab
    checkout, uv also applies the checkout's own ``override-dependencies``, which can rewrite a registry
    package's requirements (``numpy<2`` becomes ``numpy>=2``). ``uv pip check`` before and after catches
    the conflicts this lets through, and the install is then rolled back.

    Args:
        plans: Packages to install.
        target: Environment to install into.
        dry_run: Show what uv would do without changing anything.
        keep_conflicts: Keep an install that introduces dependency conflicts, with a warning.

    Raises:
        RegistryError: If uv fails, or if the install introduced conflicts and was rolled back.
    """
    uv = uv_binary()
    load_state(target.prefix)  # An unreadable state file must stop the command before uv changes anything.
    distributions = installed_distributions(target.site_paths)
    previous = {name: describe_distribution(distribution) for name, distribution in distributions.items()}
    names = {plan.name for plan in plans}
    overrides = peer_overrides(distributions)
    constraints = [
        f"{name}=={distribution.version}"
        for name, distribution in sorted(distributions.items())
        if name not in names and not is_isaaclab_distribution(name)
    ]
    with tempfile.TemporaryDirectory(prefix="isaaclab-registry-") as temporary:
        directory = Path(temporary)
        argv = [uv, "pip", "install", "--python", target.python]
        if overrides:
            (directory / "overrides.txt").write_text("\n".join(overrides) + "\n", encoding="utf-8")
            argv += ["--overrides", str(directory / "overrides.txt")]
        if constraints:
            (directory / "constraints.txt").write_text("\n".join(constraints) + "\n", encoding="utf-8")
            argv += ["-c", str(directory / "constraints.txt")]
        for package in NO_SOURCES_PACKAGES:
            argv += ["--no-sources-package", package]
        if dry_run:
            argv.append("--dry-run")
        # Hashes can only be given in a requirements file; uv verifies every hash it is given. The refresh makes uv
        # check the file the URL serves now, rather than a copy it cached earlier (even one that failed the check).
        hashed = [plan for plan in plans if plan.sha256]
        for plan in hashed:
            argv += ["--refresh-package", plan.name]
        if hashed:
            lines = [f"{plan.requirement} --hash=sha256:{plan.sha256}" for plan in hashed]
            (directory / "requirements.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            argv += ["-r", str(directory / "requirements.txt")]
        argv += [plan.requirement for plan in plans if not plan.sha256]
        cwd = target.uv_cwd or directory

        print_info(f"Peer overrides: {', '.join(overrides) or '(none)'}; {len(constraints)} constraints.")
        for plan in hashed:
            print_info(f"uv verifies {plan.name} against sha256:{plan.sha256}.")
        before = set() if dry_run else _pip_check(uv, target, cwd)
        print_info(f"Running: {format_command(argv)}")
        result = run_uv(argv, cwd=cwd)
        if result.returncode != 0:
            replaced = (
                " A downloaded wheel is not the one the registry recorded: its author's index may have replaced it."
                if "Hash mismatch" in (result.stderr or "")
                else ""
            )
            raise RegistryError(
                f"uv pip install failed with exit code {result.returncode}; nothing was recorded.{replaced}"
            )
        if dry_run:
            return
        conflicts = _new_conflicts(before, _pip_check(uv, target, cwd))
        if conflicts and not keep_conflicts:
            rolled_back = _roll_back(uv, target, cwd, previous)
            raise RegistryError(
                "The install introduced dependency conflicts:\n    "
                + "\n    ".join(conflicts)
                + f"\n{rolled_back} Nothing was recorded. (In an Isaac Lab checkout, its override-dependencies let uv"
                " install packages whose own requirements do not hold; 'isaaclab registry install --force' keeps"
                " such an install anyway.)"
            )
        for line in conflicts:
            warn(f"New dependency conflict, kept because of --force: {line}")

    installed = {name: describe_distribution(dist) for name, dist in installed_distributions(target.site_paths).items()}
    for plan in plans:
        described = installed.get(plan.name)
        # Record the commit-pinned requirement, so that 'restore' reinstalls this exact commit even when the
        # request named a branch or a tag.
        requirement = described.requirement if described else plan.requirement
        record_install(
            target.prefix,
            plan.name,
            {
                "requirement": requirement,
                "requested": plan.requirement if plan.requirement != requirement else None,
                "version": described.version if described else plan.version,
                "commit": (described.commit if described else None) or plan.commit,
                "sha256": plan.sha256,
            },
        )


def install_into_project(plans: list[InstallPlan], target: Target, *, dry_run: bool = False) -> None:
    """Add packages to the project's ``isaaclab-registry`` dependency group, then install them.

    Each package is added with ``uv add --no-sync`` (one call per package), which only edits ``pyproject.toml``
    and ``uv.lock``. Wheel hashes locked for hash-pinned packages are compared with the registry's before
    ``uv sync --frozen --inexact`` installs anything; the sync keeps packages that are not in the lock. On any
    failure, ``pyproject.toml`` and ``uv.lock`` get their previous content back and nothing is installed.

    Raises:
        RegistryError: If uv fails or a locked wheel is not the one the registry recorded.
    """
    uv = uv_binary()
    root = target.project_root
    files = (root / "pyproject.toml", root / "uv.lock")
    adds = [
        [uv, "add", "--no-sync", "--project", str(root), "--group", DEPENDENCY_GROUP]
        + (["--refresh-package", plan.name] if plan.sha256 else [])
        + [plan.requirement]
        for plan in plans
    ]
    sync = [uv, "sync", "--frozen", "--inexact", "--project", str(root), "--group", DEPENDENCY_GROUP]
    if dry_run:
        for argv in (*adds, sync):
            print_info(f"Would run: {format_command(argv)}")
        return
    backup = {path: path.read_bytes() for path in files if path.is_file()}
    try:
        for plan, argv in zip(plans, adds):
            print_info(f"Running: {format_command(argv)}")
            result = run_uv(argv, cwd=root)
            if result.returncode != 0:
                hint = _peer_override_hint(root, result.stderr or "")
                raise RegistryError(
                    f"uv add {plan.name} failed with exit code {result.returncode}." + (f"\n{hint}" if hint else "")
                )
            if plan.sha256:
                _check_locked_hash(files[1], plan)
        print_info(f"Running: {format_command(sync)}")
        result = run_uv(sync, cwd=root)
        if result.returncode != 0:
            raise RegistryError(f"uv sync failed with exit code {result.returncode}.")
    except (RegistryError, KeyboardInterrupt) as error:
        for path in files:
            if path in backup:
                path.write_bytes(backup[path])
            else:
                path.unlink(missing_ok=True)
        if isinstance(error, KeyboardInterrupt):
            raise
        raise RegistryError(f"{error} pyproject.toml and uv.lock are back to their previous content.") from error
    warn_default_groups(root)


def uninstall_from_environment(name: str, target: Target, *, dry_run: bool = False) -> None:
    """Remove a package with ``uv pip uninstall``. Its now-unused dependencies stay installed.

    Raises:
        RegistryError: If uv fails.
    """
    uv = uv_binary()
    argv = [uv, "pip", "uninstall", "--python", target.python, name]
    if dry_run:
        argv.append("--dry-run")
    with tempfile.TemporaryDirectory(prefix="isaaclab-registry-") as temporary:
        print_info(f"Running: {format_command(argv)}")
        result = run_uv(argv, cwd=target.uv_cwd or Path(temporary))
    if result.returncode != 0:
        raise RegistryError(f"uv pip uninstall failed with exit code {result.returncode}.")


def uninstall_from_project(name: str, target: Target, *, dry_run: bool = False) -> None:
    """Remove a package from the project's ``isaaclab-registry`` group, then from its environment.

    ``uv remove --no-sync`` edits ``pyproject.toml`` and ``uv.lock`` only, and ``uv pip uninstall`` removes just
    this package: a syncing ``uv remove`` would also remove everything else the lock does not list (packages
    installed with ``uv pip install`` or with other groups and extras). Dependencies that only this package
    needed stay installed until the next exact ``uv sync``.

    Raises:
        RegistryError: If uv fails.
    """
    uv = uv_binary()
    root = target.project_root
    remove = [uv, "remove", "--no-sync", "--project", str(root), "--group", DEPENDENCY_GROUP, name]
    uninstall = [uv, "pip", "uninstall", "--python", target.python, name]
    if dry_run:
        for argv in (remove, uninstall):
            print_info(f"Would run: {format_command(argv)}")
        return
    for argv in (remove, uninstall):
        print_info(f"Running: {format_command(argv)}")
        result = run_uv(argv, cwd=root)
        if result.returncode != 0:
            raise RegistryError(f"{format_command(argv[:3])} failed with exit code {result.returncode}.")


def warn_default_groups(root: Path) -> None:
    """Warn when a plain ``uv sync`` would drop the ``isaaclab-registry`` group from a project."""
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    groups = data.get("tool", {}).get("uv", {}).get("default-groups", ["dev"])
    if groups == "all" or DEPENDENCY_GROUP in groups:
        return
    warn(
        f"'{DEPENDENCY_GROUP}' is not a default group of {root / 'pyproject.toml'}, so a plain 'uv sync' removes"
        " registry packages ('uv run' keeps them). Add it under [tool.uv]:\n"
        f"    default-groups = {json.dumps([*groups, DEPENDENCY_GROUP])}"
    )


def missing_rl_libraries(libraries: tuple[str, ...], target: Target) -> list[tuple[str, str]]:
    """Return ``(library, isaaclab extra)`` for each RL library a release uses that is not importable."""
    importlib.invalidate_caches()
    missing = []
    for library in libraries:
        module, extra = RL_LIBRARIES.get(library.replace("-", "_"), (library.replace("-", "_"), library))
        if target.site_paths is None:
            found = importlib.util.find_spec(module) is not None
        else:
            found = importlib.machinery.PathFinder.find_spec(module, list(target.site_paths)) is not None
        if not found:
            missing.append((library, extra))
    return missing


def _is_satisfied(requirement: Any, installed: dict[str, str]) -> bool:
    """Return whether an installed version satisfies a ``packaging`` requirement."""
    version = installed.get(normalize_name(requirement.name))
    if version is None:
        return False
    return not requirement.specifier or requirement.specifier.contains(version, prereleases=True)


def _pip_check(uv: str, target: Target, cwd: Path) -> set[str]:
    """Return the incompatibility lines ``uv pip check`` reports for the target environment."""
    result = run_uv([uv, "pip", "check", "--python", target.python], cwd=cwd, capture=True)
    lines = f"{result.stdout}\n{result.stderr}".splitlines()
    return {line.strip() for line in lines if line.strip().startswith("The package")}


def _new_conflicts(before: set[str], after: set[str]) -> list[str]:
    """Return the ``uv pip check`` lines an install introduced, other than the expected Isaac Lab peer ones."""
    new = sorted(after - before)
    peers = [line for line in new if "requires `isaaclab" in line]
    if peers:
        print_info(
            "As intended, the installed Isaac Lab satisfies these through peer overrides"
            " (uv pip check still reports the declared ranges):\n    " + "\n    ".join(peers)
        )
    return [line for line in new if line not in peers]


def _roll_back(uv: str, target: Target, cwd: Path, previous: dict[str, InstalledDistribution]) -> str:
    """Undo an install: remove the distributions it added and reinstall the ones it replaced.

    The constraints keep every other distribution in place, so comparing the environment before and after
    finds exactly what the install changed. Isaac Lab itself is never touched.

    Returns:
        A sentence describing the rollback.
    """
    current = {name: describe_distribution(dist) for name, dist in installed_distributions(target.site_paths).items()}
    changed = sorted(
        name for name, dist in current.items() if previous.get(name) != dist and not is_isaaclab_distribution(name)
    )
    replaced = [previous[name].requirement for name in changed if name in previous]
    if not changed:
        return "The environment is unchanged."
    print_info(f"Rolling back: {', '.join(changed)}.")
    result = run_uv([uv, "pip", "uninstall", "--python", target.python, *changed], cwd=cwd)
    if result.returncode != 0:
        return f"Rolling back failed: 'uv pip uninstall {' '.join(changed)}' exited with {result.returncode}."
    if not replaced:
        return f"Rolled back: removed {', '.join(changed)}."
    argv = [uv, "pip", "install", "--python", target.python, "--no-deps"]
    result = run_uv([*argv, *replaced], cwd=cwd)
    if result.returncode != 0:
        return f"Rolled back, but could not reinstall {', '.join(replaced)} (exit code {result.returncode})."
    return f"Rolled back: removed {', '.join(changed)} and reinstalled {', '.join(replaced)}."


def _peer_override_hint(root: Path, stderr: str) -> str | None:
    """Explain how to fix a failed ``uv add`` in a project that takes Isaac Lab from a local checkout."""
    if "isaaclab" not in stderr:
        return None
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    tool_uv = data.get("tool", {}).get("uv", {})
    sources = tool_uv.get("sources", {})
    if not any(normalize_name(name) == "isaaclab" and "path" in source for name, source in sources.items()):
        return None
    overridden = {
        normalize_name(match.group(0))
        for line in tool_uv.get("override-dependencies", [])
        if (match := re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", line.strip()))
    }
    if "isaaclab" in overridden:
        return None
    return (
        "This project takes Isaac Lab from a local checkout, whose 'isaaclab' version does not match the release"
        " range the package declares. Let the checkout satisfy it with a peer override in pyproject.toml:\n"
        '    [tool.uv]\n    override-dependencies = ["isaaclab"]\n'
        "Merge it with existing overrides; projects generated by the current template include it."
    )


def _check_locked_hash(lock: Path, plan: InstallPlan) -> None:
    """Verify that ``uv.lock`` pins ``plan``'s wheel to the sha256 the registry recorded.

    ``uv add`` locks whatever file the URL serves at that moment, and ``uv sync`` then installs exactly what the
    lock says, so a wheel replaced after registration is caught here, before anything is installed.

    Raises:
        RegistryError: If the lock has no entry for the package or records another hash.
    """
    expected = f"sha256:{plan.sha256}"
    try:
        packages = tomllib.loads(lock.read_text(encoding="utf-8")).get("package", [])
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RegistryError(f"Cannot read {lock} to verify {plan.name}: {error}") from error
    hashes = [
        artifact.get("hash", "")
        for package in packages
        if normalize_name(str(package.get("name", ""))) == plan.name
        for artifact in [*package.get("wheels", []), *([package["sdist"]] if "sdist" in package else [])]
    ]
    if not hashes or any(digest != expected for digest in hashes):
        raise RegistryError(
            f"uv locked {plan.name} with {', '.join(hashes) or 'no hash'}, but the registry recorded {expected}:"
            f" the file at {plan.requirement.partition(' @ ')[2]} is not the wheel that was registered."
        )
