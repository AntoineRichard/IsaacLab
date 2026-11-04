# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Decide where registry packages go, install and remove them with uv, and report what is installed."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib

from ..utils import print_info
from . import DEPENDENCY_GROUP, TASKS_ENTRY_POINT_GROUP, RegistryError, is_isaaclab_distribution, normalize_name, warn
from .index import Package, RegistryIndex, Release, latest_compatible

RL_IMPORT_NAMES = {"sb3": "stable_baselines3"}
"""Registry RL library names whose import name differs; each library's Isaac Lab extra is its name with hyphens."""

_NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SUBDIRECTORY = re.compile(r"^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$")


@dataclass(frozen=True)
class Target:
    """Where an install, uninstall, or listing applies.

    Attributes:
        mode: ``"project"`` to edit the user's uv project with ``uv add``/``uv remove``, or ``"env"`` to
            change the running environment with ``uv pip``.
        python: Interpreter of the target environment, passed to ``uv pip --python``.
        prefix: Prefix of the target environment.
        project_root: Root of the user's project (project mode only).
        uv_cwd: Working directory for ``uv pip`` in env mode: the Isaac Lab checkout when the target is the
            checkout's own environment, so that its indexes and overrides apply, else ``None``.
        isaaclab_version: Isaac Lab release version of the target environment, for the compatibility gate.
        site_paths: Directories that hold the installed distributions, or ``None`` for ``sys.path``.
    """

    mode: str
    python: str
    prefix: Path
    project_root: Path | None = None
    uv_cwd: Path | None = None
    isaaclab_version: str | None = None
    site_paths: tuple[str, ...] | None = None

    def describe(self) -> str:
        """Return a one-line description for summaries."""
        if self.mode == "project":
            return f"project {self.project_root} (dependency group '{DEPENDENCY_GROUP}', uv add)"
        checkout = f" (Isaac Lab checkout; uv runs from {self.uv_cwd})" if self.uv_cwd else ""
        return f"environment {self.prefix} (uv pip){checkout}"


@dataclass(frozen=True)
class InstallPlan:
    """A requirement to install.

    Attributes:
        name: Normalized distribution name.
        requirement: PEP 508 requirement passed to uv.
        sha256: sha256 the wheel of a PyPI release must have; uv refuses any other file.
    """

    name: str
    requirement: str
    sha256: str | None = None


@dataclass(frozen=True)
class InstalledPackage:
    """One row of ``isaaclab registry installed``."""

    name: str
    version: str
    status: str
    commit: str | None = None
    note: str = ""


# Target.


def checkout_root() -> Path | None:
    """Return the Isaac Lab source checkout that provides the running ``isaaclab``, if any.

    A wheel install resolves ``ISAACLAB_ROOT`` to the package directory, which has no ``source/isaaclab/isaaclab``
    tree.
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


def detect_target() -> Target:
    """Choose the install target from where this command runs.

    The nearest project above the working directory, else the project that owns the running environment (as
    ``uv run --project ROOT`` selects from elsewhere), is the target when its uv environment is the one running
    this command. Everything else is env mode. The Isaac Lab checkout is never a project target: neither its
    tracked ``pyproject.toml`` files nor its environment (which ``uv add`` in another project would sync against
    that project's lock) are edited.

    Returns:
        The resolved target.
    """
    prefix = Path(sys.prefix)
    checkout = checkout_root()
    checkout_environment = checkout is not None and _runs_in(checkout, prefix)
    common = {"python": sys.executable, "prefix": prefix, "isaaclab_version": isaaclab_release_version()}
    cwd = Path.cwd()
    nearest = next((directory for directory in (cwd, *cwd.parents) if _project_name(directory) is not None), None)
    for root in (nearest, prefix.parent):
        name = _project_name(root) if root is not None else None
        if name is None or not _runs_in(root, prefix):
            continue
        part_of_lab = is_isaaclab_distribution(name) or (
            checkout is not None and root.resolve().is_relative_to(checkout.resolve())
        )
        if not part_of_lab and not checkout_environment:
            return Target(mode="project", project_root=root, **common)
        break
    return Target(mode="env", uv_cwd=checkout if checkout_environment else None, **common)


def uv_binary() -> str:
    """Return the uv executable: ``$UV`` (set by ``uv run``), else ``uv`` on ``PATH``.

    Raises:
        RegistryError: If uv cannot be found.
    """
    uv = shutil.which(os.environ.get("UV") or "uv")
    if not uv:
        raise RegistryError("uv was not found. Install it (https://docs.astral.sh/uv/) or set UV=/path/to/uv.")
    return uv


def _project_name(directory: Path) -> str | None:
    """Return the ``[project]`` name of ``directory/pyproject.toml``, or ``None`` when it declares no project."""
    pyproject = directory / "pyproject.toml"
    if not pyproject.is_file():
        return None
    try:
        project = tomllib.loads(pyproject.read_text(encoding="utf-8")).get("project")
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RegistryError(f"Cannot read {pyproject}: {error}") from error
    return str(project.get("name", "")) if isinstance(project, dict) else None


def _runs_in(root: Path, prefix: Path) -> bool:
    """Return whether ``prefix`` is the environment uv uses for the project at ``root``."""
    environment = root / Path(os.environ.get("UV_PROJECT_ENVIRONMENT") or ".venv").expanduser()
    return environment.resolve() == prefix.resolve()


# Install and uninstall.


def plan_for_release(package: Package, release: Release) -> InstallPlan:
    """Build the uv requirement for an index release from validated fields.

    Git releases become ``name @ git+<repository>@<commit>[#subdirectory=<dir>]``. PyPI releases become
    ``name @ <wheel url>`` with the wheel's registered sha256: the exact file the registry checked is installed,
    and the author's index never takes part in the resolution, so it cannot serve any other package.

    Raises:
        RegistryError: If a field the requirement is built from is malformed.
    """
    name = normalize_name(package.name)
    if not _NAME.match(name):
        raise RegistryError(f"Refusing malformed package name {package.name!r} from the index.")
    if package.source_type == "git":
        repository, subdirectory = package.source.get("repository", ""), package.source.get("subdirectory") or ""
        # A query or an '@' in the path would move the pinned commit out of the requirement.
        parsed = urllib.parse.urlparse(repository)
        if not _plain_url(repository, ("https", "http", "file", "ssh")) or "?" in repository or "@" in parsed.path:
            raise RegistryError(f"Refusing malformed repository URL {repository!r} for {name}.")
        if not _COMMIT.match(release.commit or ""):
            raise RegistryError(f"{name} {release.version} has no valid 40-hex commit in the index.")
        if subdirectory and (not _SUBDIRECTORY.match(subdirectory) or ".." in subdirectory.split("/")):
            raise RegistryError(f"Refusing malformed subdirectory {subdirectory!r} for {name}.")
        requirement = f"{name} @ git+{repository}@{release.commit}"
        return InstallPlan(name, requirement + (f"#subdirectory={subdirectory}" if subdirectory else ""))
    if package.source_type == "pypi":
        url, sha256 = (release.wheel or {}).get("url", ""), (release.wheel or {}).get("sha256", "")
        if not _SHA256.match(sha256):
            raise RegistryError(f"{name} {release.version} has no valid wheel sha256 in the index.")
        if not _plain_url(url, ("https", "http", "file")):
            raise RegistryError(f"Refusing malformed wheel URL {url!r} for {name}.")
        return InstallPlan(name, f"{name} @ {url}", sha256)
    raise RegistryError(f"{name} has an unsupported source type {package.source_type!r}.")


def install_into_environment(plans: list[InstallPlan], target: Target, *, dry_run: bool = False) -> None:
    """Install packages into the target environment with ``uv pip install``.

    Installed Isaac Lab packages are pinned through overrides, which win over any range a registry package
    declares, and every other installed distribution through constraints. Resolution runs against a temporary
    copy of the installed metadata first, refusing replacements of unrequested packages even at the same version.
    The selected packages and resolved additions are then installed with ``--no-deps`` so the real environment's
    other packages cannot change. Hash-pinned wheels are reinstalled to verify their registered sha256.

    Args:
        plans: Packages to install.
        target: Environment to install into.
        dry_run: Show what uv would do without changing anything.

    Raises:
        RegistryError: If uv fails, for example on a hash mismatch.
    """
    names = {plan.name for plan in plans}
    distributions = installed_distributions(target.site_paths)
    pins = {name: f"{name}=={distribution.version}" for name, distribution in sorted(distributions.items())}
    overrides = [pin for name, pin in pins.items() if is_isaaclab_distribution(name)]
    constraints = [pin for name, pin in pins.items() if name not in names and not is_isaaclab_distribution(name)]
    hashed = [f"{plan.requirement} --hash=sha256:{plan.sha256}" for plan in plans if plan.sha256]
    # The isaaclab override already keeps a registry package's own [tool.uv.sources] entry for isaaclab (a path
    # into its author's checkout, say) out of the resolution; ignoring that source as well costs nothing.
    argv = [uv_binary(), "pip", "install", "--python", target.python, "--no-sources-package", "isaaclab"]
    print_info(f"Isaac Lab stays at {', '.join(overrides) or '-'}; {len(constraints)} other packages are constrained.")
    with tempfile.TemporaryDirectory(prefix="isaaclab-registry-") as temporary:
        for option, lines, filename in (
            ("--overrides", overrides, "overrides.txt"),
            ("-c", constraints, "constraints.txt"),
            ("-r", hashed, "requirements.txt"),
        ):
            if lines:
                (Path(temporary) / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")
                argv += [option, str(Path(temporary) / filename)]
        # Refreshing metadata alone lets uv keep an installed wheel without checking its hash.
        argv += [option for plan in plans if plan.sha256 for option in ("--reinstall-package", plan.name)]
        argv += [plan.requirement for plan in plans if not plan.sha256]
        additions = _resolve_environment_install(argv, target, distributions, names, Path(temporary))
        argv += ["--no-deps"]
        if additions.stat().st_size:
            argv += ["-r", str(additions)]
        _run_uv([*argv, "--dry-run"] if dry_run else argv, cwd=target.uv_cwd or Path(temporary))


def _resolve_environment_install(
    argv: list[str],
    target: Target,
    distributions: dict[str, importlib.metadata.Distribution],
    names: set[str],
    temporary: Path,
) -> Path:
    """Resolve with uv in scratch metadata, reject protected replacements, and freeze only new dependencies."""
    stage = temporary / "site"
    stage.mkdir()
    protected = {}
    for name, distribution in distributions.items():
        metadata = stage / f"{name.replace('-', '_')}-{distribution.version}.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(
            distribution.read_text("METADATA") or str(distribution.metadata), encoding="utf-8"
        )
        for filename in ("WHEEL", "direct_url.json", "entry_points.txt"):
            if (content := distribution.read_text(filename)) is not None:
                (metadata / filename).write_text(content, encoding="utf-8")
        marker = metadata / "_registry_protected"
        marker.touch()
        if name not in names:
            protected[name] = marker
        # Never copy installed RECORD paths: they can lead outside this scratch directory. A uv replacement
        # removes this marker through the scratch RECORD, even when the package's version and source match.
        (metadata / "RECORD").write_text(
            "".join(f"{metadata.name}/{path.name},,\n" for path in metadata.iterdir()), encoding="utf-8"
        )
    cwd = target.uv_cwd or temporary
    _run_uv([*argv, "--target", str(stage)], cwd=cwd)
    replaced = sorted(name for name, marker in protected.items() if not marker.exists())
    if replaced:
        raise RegistryError(f"The install would replace protected packages: {', '.join(replaced)}.")
    excluded = sorted(distributions.keys() | names)
    freeze = [argv[0], "pip", "freeze", "--python", target.python, "--path", str(stage)]
    freeze += [option for name in excluded for option in ("--exclude", name)]
    additions = temporary / "additions.txt"
    additions.write_text(_run_uv(freeze, cwd=cwd, capture_stdout=True), encoding="utf-8")
    return additions


def install_into_project(plans: list[InstallPlan], target: Target, *, dry_run: bool = False) -> None:
    """Add packages to the project's ``isaaclab-registry`` dependency group, then install them.

    ``uv add --no-sync`` only edits ``pyproject.toml`` and ``uv.lock``, and leaves both as they were when it
    fails. The new lock is checked before ``uv sync --frozen --inexact`` installs anything (keeping packages that
    are not in the lock): it must leave every Isaac Lab package as it was locked, and pin each hash-pinned plan to
    the registered sha256, since uv locks whatever file a wheel URL serves. If a check or the sync fails, both
    files get their previous content back.

    Raises:
        RegistryError: If uv fails, the install would change Isaac Lab, or a locked wheel is not the registered one.
    """
    uv, root = uv_binary(), target.project_root
    add = [uv, "add", "--no-sync", "--project", str(root), "--group", DEPENDENCY_GROUP, *_refresh_options(plans)]
    add += [plan.requirement for plan in plans]
    sync = [uv, "sync", "--frozen", "--inexact", "--project", str(root), "--group", DEPENDENCY_GROUP]
    if dry_run:
        for argv in (add, sync):
            print_info(f"Would run: {_command_line(argv)}")
        return
    files = (root / "pyproject.toml", root / "uv.lock")
    backup = {path: path.read_bytes() for path in files if path.is_file()}
    _run_uv(add, cwd=root)
    try:
        before = _locked_isaaclab(backup.get(files[1], b"").decode())
        after = _locked_isaaclab(files[1].read_text(encoding="utf-8"))
        changes = [
            f"{name} {before.get(name, ('(none)',))[0]} -> {entry[0]}"
            for name, entry in after.items()
            if before.get(name) != entry
        ]
        if changes:
            raise RegistryError(
                f"The install would change Isaac Lab ({', '.join(changes)}), which the registry never does."
                " Upgrade Isaac Lab yourself first if you need this release."
            )
        for plan in plans:
            if plan.sha256:
                _check_locked_hash(files[1], plan)
        _run_uv(sync, cwd=root)
    except BaseException:
        for path in files:
            if path in backup:
                path.write_bytes(backup[path])
            else:
                path.unlink(missing_ok=True)
        print_info("pyproject.toml and uv.lock are back to their previous content.")
        raise
    groups = tomllib.loads(files[0].read_text(encoding="utf-8")).get("tool", {}).get("uv", {}).get("default-groups")
    if groups != "all" and DEPENDENCY_GROUP not in (groups or ["dev"]):
        warn(f"A plain 'uv sync' removes registry packages: add '{DEPENDENCY_GROUP}' to default-groups in {files[0]}.")


def uninstall_from_environment(names: list[str], target: Target, *, dry_run: bool = False) -> None:
    """Remove packages with ``uv pip uninstall``. Dependencies that only they needed stay installed.

    Raises:
        RegistryError: If uv fails.
    """
    argv = [uv_binary(), "pip", "uninstall", "--python", target.python, *names]
    _run_uv([*argv, "--dry-run"] if dry_run else argv, cwd=target.uv_cwd)


def uninstall_from_project(names: list[str], target: Target, *, dry_run: bool = False) -> None:
    """Remove packages from the project's ``isaaclab-registry`` group, then from its environment.

    ``uv remove --no-sync`` edits ``pyproject.toml`` and ``uv.lock`` only. Packages still present in the new lock
    remain installed because another dependency, group, or extra needs them. ``uv pip uninstall`` removes only
    the others, keeping packages installed outside the project's lock.

    Raises:
        RegistryError: If uv fails.
    """
    uv, root = uv_binary(), target.project_root
    remove = [uv, "remove", "--no-sync", "--project", str(root), "--group", DEPENDENCY_GROUP, *names]
    if dry_run:
        print_info(f"Would run: {_command_line(remove)}")
        print_info(f"Would uninstall {', '.join(names)} unless the remaining project lock still requires them.")
        return
    _run_uv(remove, cwd=root)
    locked = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8")).get("package", [])
    remaining = {normalize_name(package["name"]) for package in locked}
    removed = [name for name in names if name not in remaining]
    retained = [name for name in names if name in remaining]
    if retained:
        print_info(f"Kept {', '.join(retained)} installed: the project still requires them.")
    if removed:
        _run_uv([uv, "pip", "uninstall", "--python", target.python, *removed], cwd=root)


def _plain_url(url: str, schemes: tuple[str, ...]) -> bool:
    """Return whether a URL has one of ``schemes`` and no whitespace or fragment that would change the requirement."""
    return urllib.parse.urlparse(url).scheme in schemes and not any(c.isspace() or c == "#" for c in url)


def _refresh_options(plans: list[InstallPlan]) -> list[str]:
    """Return ``--refresh-package`` for each hash-pinned plan, so that uv checks the file the URL serves now.

    Without it, uv reuses a copy it cached earlier, even one that failed the hash check. uv rejects the option
    in offline mode, where there is nothing to fetch again anyway.
    """
    if os.environ.get("UV_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on", "t", "y"):
        return []
    return [option for plan in plans if plan.sha256 for option in ("--refresh-package", plan.name)]


def _check_locked_hash(lock: Path, plan: InstallPlan) -> None:
    """Verify that ``uv.lock`` pins the wheel of ``plan`` to the sha256 the registry recorded.

    Raises:
        RegistryError: If the lock records another file for the package.
    """
    packages = tomllib.loads(lock.read_text(encoding="utf-8")).get("package", [])
    locked = {
        str(wheel.get("hash"))
        for package in packages
        if package.get("name") == plan.name
        for wheel in package.get("wheels", [])
    }
    if locked != {f"sha256:{plan.sha256}"}:
        raise RegistryError(
            f"uv locked {plan.name} as {', '.join(sorted(locked)) or 'no wheel'}, but the registry recorded"
            f" sha256:{plan.sha256}: the file at {plan.requirement.partition(' @ ')[2]} is not the wheel that was"
            " registered."
        )


def _locked_isaaclab(lock: str) -> dict[str, tuple[Any, Any]]:
    """Return ``name -> (version, source)`` for the Isaac Lab packages that a ``uv.lock`` text pins."""
    packages = tomllib.loads(lock).get("package", [])
    return {p["name"]: (p.get("version"), p.get("source")) for p in packages if is_isaaclab_distribution(p["name"])}


def _command_line(argv: list[str]) -> str:
    """Return a shell-quoted command line, with the uv executable shown as ``uv``."""
    return shlex.join(["uv", *argv[1:]])


def _run_uv(argv: list[str], cwd: Path | None = None, *, capture_stdout: bool = False) -> str:
    """Run uv with stderr on the terminal and optionally capture stdout for machine-readable output.

    Raises:
        RegistryError: If uv fails.
    """
    print_info(f"Running: {_command_line(argv)}")
    sys.stdout.flush()
    result = subprocess.run(argv, cwd=cwd, stdout=subprocess.PIPE if capture_stdout else None, text=True, check=False)
    if result.returncode != 0:
        raise RegistryError(f"'uv {' '.join(argv[1:3])}' failed with exit code {result.returncode}.")
    return result.stdout or ""


# Installed packages.


def installed_distributions(site_paths: tuple[str, ...] | None = None) -> dict[str, importlib.metadata.Distribution]:
    """Return the installed distributions keyed by normalized name (the first one on the path wins).

    Args:
        site_paths: Directories to scan, or ``None`` for ``sys.path``.
    """
    importlib.invalidate_caches()
    distributions = {}
    for distribution in importlib.metadata.distributions(path=list(site_paths or sys.path)):
        if name := distribution.metadata["Name"]:
            distributions.setdefault(normalize_name(name), distribution)
    return distributions


def direct_url(distribution: importlib.metadata.Distribution) -> dict[str, Any]:
    """Return the PEP 610 ``direct_url.json`` origin of a distribution, or ``{}`` when it came from an index."""
    try:
        return json.loads(distribution.read_text("direct_url.json") or "{}")
    except json.JSONDecodeError:
        return {}


def provides_tasks(distribution: importlib.metadata.Distribution) -> bool:
    """Return whether a distribution declares an ``isaaclab.tasks`` entry point."""
    return any(entry_point.group == TASKS_ENTRY_POINT_GROUP for entry_point in distribution.entry_points)


def installed_packages(target: Target, index: RegistryIndex | None) -> list[InstalledPackage]:
    """Classify the installed task packages against the index (the registry docs page lists the statuses).

    Args:
        target: Environment or project to report on.
        index: Registry index, or ``None`` when it is unavailable.

    Returns:
        One row per installed task package, sorted by name.
    """
    rows = []
    for name, distribution in sorted(installed_distributions(target.site_paths).items()):
        if not provides_tasks(distribution):
            continue
        origin = direct_url(distribution)
        commit = origin.get("vcs_info", {}).get("commit_id")
        status, note = _status(name, distribution.version, origin, commit, index, target.isaaclab_version)
        rows.append(InstalledPackage(name, distribution.version, status, commit, note))
    return rows


def missing_rl_libraries(libraries: list[str]) -> list[tuple[str, str]]:
    """Return ``(library, Isaac Lab extra)`` for each RL library of a release that cannot be imported here."""
    importlib.invalidate_caches()
    missing = []
    for library in libraries:
        module = library.replace("-", "_")
        if importlib.util.find_spec(RL_IMPORT_NAMES.get(module, module)) is None:
            missing.append((library, library.replace("_", "-")))
    return missing


def _status(
    name: str,
    version: str,
    origin: dict[str, Any],
    commit: str | None,
    index: RegistryIndex | None,
    isaaclab_version: str | None,
) -> tuple[str, str]:
    """Return the status and note of one installed task package."""
    if "dir_info" in origin:
        return "local", origin.get("url", "")
    if index is None:
        return "unmanaged", "index unavailable"
    if block := index.blocked(name, commit, version):
        return "blocked", f"{block.reason}; remove it with 'isaaclab registry uninstall {name}'"
    package = index.get(name)
    if package is None:
        return "unmanaged", ""
    # A git release matches on its commit, a PyPI release on its wheel URL; uv checked the hash at install time.
    pin = commit or origin.get("url")
    release = next((r for r in package.releases if pin and pin in (r.commit, (r.wheel or {}).get("url"))), None)
    if release is None:
        return "drift", f"{f'commit {commit[:12]}' if commit else version} is not a registered release"
    note = f"yanked: {release.yanked_reason}" if release.yanked_reason else ""
    latest = latest_compatible(index, package, isaaclab_version)
    if latest is not None and package.releases.index(latest) < package.releases.index(release):
        return f"outdated ({latest.version} available)", note
    return "registry", note
