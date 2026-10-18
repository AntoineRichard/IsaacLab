# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Browse, install, and publish packages from the Isaac Lab registry (prototype)."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from ..registry import DEPENDENCY_GROUP, SECURITY_WARNING, RegistryError, is_isaaclab_distribution, normalize_name, warn
from ..registry.index import (
    Package,
    RegistryIndex,
    Release,
    compatibility_problem,
    latest_compatible,
    load_index,
    resolve_index_location,
    select_release,
)
from ..registry.install import (
    InstallPlan,
    detect_target,
    direct_url,
    install_into_environment,
    install_into_project,
    installed_distributions,
    installed_packages,
    isaaclab_release_version,
    missing_rl_libraries,
    plan_for_release,
    provides_tasks,
    uninstall_from_environment,
    uninstall_from_project,
)
from ..registry.publish import draft_request, render_request, submit_request
from ..utils import print_error, print_info

_EPILOG = f"""\
The index is read from --index, else $ISAACLAB_REGISTRY_URL, else the default index (a placeholder in this
prototype).

Inside your own uv project, packages are added to its '{DEPENDENCY_GROUP}' dependency group with 'uv add'.
Anywhere else (an Isaac Lab checkout, a wheel virtual environment), they are installed into the running
environment with 'uv pip install'.

{SECURITY_WARNING}

Documentation: https://isaac-sim.github.io/IsaacLab/develop/source/experimental-features/registry.html
"""


def command_registry(args: list[str] | None = None) -> int:
    """Run an ``isaaclab registry`` subcommand.

    Args:
        args: Arguments after ``registry``. Uses ``sys.argv`` when omitted.

    Returns:
        The process exit status.
    """
    # Warnings go to stderr; keep them in order with stdout when both are piped into one log.
    with contextlib.suppress(AttributeError, ValueError):
        sys.stdout.reconfigure(line_buffering=True)
    parser = _build_parser()
    parsed = parser.parse_args(args)
    if parsed.command is None:
        parser.print_help()
        return 0
    try:
        return parsed.handler(parsed)
    except (RegistryError, OSError) as error:  # OSError: a bad $UV or --output path, for example
        print_error(str(error), stream=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


def _build_parser() -> argparse.ArgumentParser:
    """Build the ``registry`` parser and its subcommands."""
    parser = argparse.ArgumentParser(
        prog="isaaclab registry",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    parser.add_argument(
        "--index",
        metavar="URL|PATH",
        help="Registry index: an index.json URL or file, or a registry checkout (uses its dist/index.json).",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND", title="commands")

    def add(name: str, handler: Callable[[argparse.Namespace], int], summary: str) -> argparse.ArgumentParser:
        subparser = subparsers.add_parser(name, help=summary, description=summary)
        subparser.set_defaults(handler=handler)
        return subparser

    subparser = add("list", _command_list, "List packages compatible with this Isaac Lab.")
    subparser.add_argument("--all", action="store_true", help="Include end-of-life and incompatible packages.")
    subparser.add_argument("--category", help="Only show this category (for example 'locomotion').")
    subparser = add("search", _command_search, "Search names, descriptions, tags, tasks, and categories.")
    subparser.add_argument("query", help="Case-insensitive text to look for.")
    subparser = add("info", _command_info, "Show a package, its releases, and their compatibility.")
    subparser.add_argument("name", help="Package name.")
    install = add("install", _command_install, "Install registry packages.")
    install.add_argument("packages", nargs="+", metavar="PACKAGE", help="NAME, or NAME==VERSION for one release.")
    install.add_argument("--force", action="store_true", help="Install a release that is not compatible.")
    uninstall = add("uninstall", _command_uninstall, "Uninstall registry packages.")
    uninstall.add_argument("packages", nargs="+", metavar="NAME", help="Packages to uninstall.")
    for subparser in (install, uninstall):
        subparser.add_argument("-y", "--yes", action="store_true", help="Do not ask for confirmation.")
        subparser.add_argument("--dry-run", action="store_true", help="Show what uv would do without doing it.")
    add("installed", _command_installed, "List installed task packages and how they relate to the registry.")

    subparser = add(
        "publish", _command_publish, "Check a tagged project and write (or submit) its registration request."
    )
    subparser.add_argument("path", nargs="?", default=".", help="Project directory (default: the cwd).")
    subparser.add_argument("--ref", help="Git tag to register (default: v<version>).")
    subparser.add_argument("--as", dest="requested_by", metavar="USER", help="GitHub handle of the requester.")
    subparser.add_argument("--repository", metavar="URL", help="Repository URL (default: the 'origin' remote).")
    subparser.add_argument("--output", metavar="FILE", help="Write the request to FILE ('-' for stdout).")
    subparser.add_argument("--pypi", action="store_true", help="Register the wheel published on a package index.")
    subparser.add_argument("--index", dest="package_index", metavar="URL", help="PEP 503 index for --pypi.")
    subparser.add_argument("--registry", metavar="DIR", help="Submit the request to the bot of this registry checkout.")
    subparser.add_argument("--smoke", action="store_true", help="With --registry: smoke-test with this Isaac Lab.")
    return parser


# Browsing.


def _command_list(args: argparse.Namespace) -> int:
    """Implement ``registry list``."""
    index, isaaclab_version = _load_index(args), isaaclab_release_version()
    packages = [package for package in index.packages.values() if args.category in (None, package.category)]
    shown = [
        package
        for package in packages
        if args.all or (package.status != "end-of-life" and latest_compatible(index, package, isaaclab_version))
    ]
    _print_packages(index, shown, isaaclab_version, f"{index.name}: packages for Isaac Lab {isaaclab_version}")
    if len(shown) < len(packages):
        hidden = len(packages) - len(shown)
        print_info(f"{hidden} incompatible or end-of-life package(s) hidden; use --all to show them.")
    return 0


def _command_search(args: argparse.Namespace) -> int:
    """Implement ``registry search``."""
    index, query = _load_index(args), args.query.lower()
    found = [package for package in index.packages.values() if any(query in text.lower() for text in _texts(package))]
    if not found:
        print_info(f"No package matches '{args.query}'.")
        return 0
    _print_packages(index, found, isaaclab_release_version(), f"{index.name}: packages matching '{args.query}'")
    return 0


def _command_info(args: argparse.Namespace) -> int:
    """Implement ``registry info``."""
    from prettytable import PrettyTable

    index = _load_index(args)
    package = index.get(args.name)
    if package is None:
        raise RegistryError(f"'{args.name}' is not in the {index.name} registry.")
    isaaclab_version = isaaclab_release_version()
    selected = latest_compatible(index, package, isaaclab_version)
    installed = installed_distributions().get(package.name)
    commit = direct_url(installed).get("vcs_info", {}).get("commit_id") if installed else None
    fields = [
        ("status", package.status + (f": {package.status_description}" if package.status_description else "")),
        ("category", package.category),
        ("tags", ", ".join(package.tags)),
        ("maintainers", ", ".join(package.maintainers)),
        ("homepage", package.homepage),
        ("media", ", ".join(f"{kind}: {url}" for kind, url in package.media.items())),
        ("source", _source(package)),
        ("registry", f"{index.name} (index generated {index.generated_at or 'unknown'})"),
        ("installed", (installed.version + (f" (commit {commit})" if commit else "")) if installed else ""),
        (
            "install",
            f"isaaclab registry install {package.name}  ->  {selected.version}"
            if selected
            else f"no release is compatible with Isaac Lab {isaaclab_version}",
        ),
    ]
    print(f"{package.name}: {package.summary}")
    for label, value in fields:
        if value:
            print(f"  {label:<12} {value}")
    if block := index.blocked(package.name):
        warn(f"Blocklisted by the registry: {block.reason}")
    table = PrettyTable(["Version", "Source", "Isaac Lab", "Python", "Tasks", "RL libraries", "Registered", "Verdict"])
    table.align = "l"
    for release in package.releases:
        table.add_row(
            [
                release.version + (" *" if release is selected else ""),
                _release_source(release),
                release.isaaclab or "-",
                release.requires_python or "-",
                "\n".join(release.tasks) or "-",
                ", ".join(release.rl_libraries) or "-",
                release.registered_at[:10] or "-",
                _verdict(index, package, release, isaaclab_version),
            ]
        )
    print(table)
    if selected is not None:
        print("* release selected by 'isaaclab registry install'")
    return 0


# Installing.


def _command_install(args: argparse.Namespace) -> int:
    """Implement ``registry install``."""
    target = detect_target()
    distributions = installed_distributions(target.site_paths)
    specs = (spec.partition("==") for spec in args.packages)
    requests = {normalize_name(name.strip()): version.strip() or None for name, _, version in specs}
    for name in requests:
        _check_managed(name, distributions.get(name))
    index = _load_index(args)
    selected: list[tuple[Package, Release, InstallPlan]] = []
    for name, version in requests.items():
        package, release, warnings = select_release(index, name, target.isaaclab_version, version, args.force)
        for warning in warnings:
            warn(warning)
        selected.append((package, release, plan_for_release(package, release)))

    for package, release, plan in selected:
        _print_install_summary(package, release, plan, target.isaaclab_version)
    names = ", ".join(plan.name for *_, plan in selected)
    print(f"Target: {target.describe()}")
    warn(SECURITY_WARNING)
    if not args.dry_run and not _confirm(f"Install {names}?", args.yes):
        print_info("Nothing was installed.")
        return 1
    install = install_into_project if target.mode == "project" else install_into_environment
    install([plan for *_, plan in selected], target, dry_run=args.dry_run)
    if args.dry_run:
        return 0

    print_info(f"Installed {names} into the {target.mode} target.")
    for package, release, _ in selected:
        for library, extra in missing_rl_libraries(release.rl_libraries):
            warn(f"{package.name} uses {library}, which is not installed here: add Isaac Lab's '{extra}' extra.")
    if target.uv_cwd is not None:
        print_info("A plain 'uv sync' in the checkout removes packages installed this way; install them again then.")
    return 0


def _command_uninstall(args: argparse.Namespace) -> int:
    """Implement ``registry uninstall``."""
    target = detect_target()
    distributions = installed_distributions(target.site_paths)
    names = list(dict.fromkeys(normalize_name(name) for name in args.packages))
    for name in names:
        _check_managed(name, distributions.get(name))
        if target.mode == "env" and name not in distributions:
            raise RegistryError(f"{name} is not installed in {target.prefix}.")

    print(f"Target: {target.describe()}")
    if not args.dry_run and not _confirm(f"Uninstall {', '.join(names)}?", args.yes):
        print_info("Nothing was uninstalled.")
        return 1
    uninstall = uninstall_from_project if target.mode == "project" else uninstall_from_environment
    uninstall(names, target, dry_run=args.dry_run)
    if not args.dry_run:
        print_info(f"Uninstalled {', '.join(names)}. Dependencies that only they needed stay installed.")
    return 0


def _command_installed(args: argparse.Namespace) -> int:
    """Implement ``registry installed``."""
    from prettytable import PrettyTable

    target = detect_target()
    try:
        index = _load_index(args)
    except RegistryError as error:
        warn(f"{error} Statuses that need the index are unavailable.")
        index = None
    rows = installed_packages(target, index)
    print(f"Target: {target.describe()}")
    if not rows:
        print_info("No Isaac Lab task packages are installed.")
        return 0
    table = PrettyTable(["Name", "Version", "Status", "Commit", "Note"])
    table.align = "l"
    for row in rows:
        table.add_row([row.name, row.version, row.status, (row.commit or "-")[:12], row.note])
    print(table)
    return 0


def _check_managed(name: str, distribution: importlib.metadata.Distribution | None) -> None:
    """Refuse to touch Isaac Lab, a local install (such as the user's own project), or a library without tasks.

    Raises:
        RegistryError: If ``name`` is not a package that the registry may install, replace, or remove.
    """
    if is_isaaclab_distribution(name):
        raise RegistryError(f"{name} is part of Isaac Lab, which the registry never installs or removes.")
    if distribution is None:
        return
    if "dir_info" in (origin := direct_url(distribution)):
        raise RegistryError(
            f"{name} is a local install ({origin.get('url')}); the registry never replaces or removes it."
        )
    if not provides_tasks(distribution):
        raise RegistryError(
            f"{name} {distribution.version} provides no Isaac Lab tasks; the registry never replaces or removes it."
        )


# Publishing.


def _command_publish(args: argparse.Namespace) -> int:
    """Implement ``registry publish``."""
    request = draft_request(
        Path(args.path),
        ref=args.ref,
        repository=args.repository,
        requested_by=args.requested_by,
        pypi=args.pypi,
        package_index=args.package_index,
    )
    text = render_request(request)
    output = Path(args.output).resolve() if args.output not in (None, "-") else None
    if output is not None:
        output.write_text(text, encoding="utf-8")
        print_info(f"Wrote the registration request to {output}.")
    elif args.output == "-" or not args.registry:
        sys.stdout.write(text)
    if not args.registry:
        return 0
    bot_args = ["--smoke", "--isaaclab-python", sys.executable] if args.smoke else []
    with tempfile.TemporaryDirectory(prefix="isaaclab-registry-") as temporary:
        request_file = output or Path(temporary) / f"{request['name']}-request.toml"
        request_file.write_text(text, encoding="utf-8")
        return submit_request(Path(args.registry), request_file, bot_args)


# Helpers.


def _load_index(args: argparse.Namespace) -> RegistryIndex:
    """Load the index selected by ``--index``."""
    return load_index(resolve_index_location(args.index))


def _confirm(question: str, assume_yes: bool) -> bool:
    """Ask a yes/no question unless ``--yes`` was given. End of input (Ctrl-D) means no.

    Raises:
        RegistryError: If confirmation is needed but stdin is not interactive.
    """
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        raise RegistryError("Confirmation is required; re-run with --yes.")
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        print()
        return False


def _texts(package: Package) -> list[str]:
    """Return the fields that ``search`` matches against."""
    texts = [package.name, package.description, package.category, *package.tags]
    for release in package.releases:
        texts += [release.summary, *release.tasks]
    return texts


def _print_packages(index: RegistryIndex, packages: list[Package], isaaclab_version: str | None, title: str) -> None:
    """Print ``list``/``search`` results as a table."""
    from prettytable import PrettyTable

    installed = {name: dist.version for name, dist in installed_distributions().items() if provides_tasks(dist)}
    table = PrettyTable(["Name", "Version", "Category", "Tasks", "Status", "Compatibility", "Installed", "Description"])
    table.title = title
    table.align = "l"
    table.max_width["Description"] = 48
    for package in packages:
        release = latest_compatible(index, package, isaaclab_version)
        newest_live = next((candidate for candidate in package.releases if candidate.yanked_reason is None), None)
        if release is not None:
            compatibility = "ok"
        elif newest_live is None:
            compatibility = "no live release"
        else:
            compatibility = _verdict(index, package, newest_live, isaaclab_version)
        shown = release or newest_live
        table.add_row(
            [
                package.name,
                release.version if release else "-",
                package.category,
                len(shown.tasks) if shown else 0,
                package.status,
                compatibility,
                installed.get(package.name, ""),
                package.summary,
            ]
        )
    print(table)


def _verdict(index: RegistryIndex, package: Package, release: Release, isaaclab_version: str | None) -> str:
    """Return whether a release can be installed here, and why not."""
    if block := index.blocked(package.name, release.commit, release.version):
        return f"blocklisted: {block.reason}"
    if release.yanked_reason is not None:
        return f"yanked: {release.yanked_reason}"
    return compatibility_problem(release, isaaclab_version) or "ok"


def _source(package: Package) -> str:
    """Return where the code of a package comes from."""
    if package.source_type == "git":
        subdirectory = package.source.get("subdirectory")
        return f"git {package.source.get('repository')}" + (f" (subdirectory {subdirectory})" if subdirectory else "")
    return f"pypi {package.source.get('project', package.name)} on {package.source.get('index', 'PyPI')}"


def _release_source(release: Release) -> str:
    """Return the commit (and tag) or the wheel (and sha256 prefix) that pins a release."""
    if release.commit:
        return release.commit[:12] + (f" ({release.git_tag})" if release.git_tag else "")
    if release.wheel:
        return f"{release.wheel.get('filename', 'wheel')} sha256:{release.wheel.get('sha256', '?')[:12]}"
    return "-"


def _print_install_summary(package: Package, release: Release, plan: InstallPlan, isaaclab_version: str | None) -> None:
    """Print what is about to be installed."""
    print(f"{package.name} {release.version}: {package.summary}")
    print(f"  source       {_source(package)} @ {_release_source(release)}")
    if plan.sha256:
        print(f"  sha256       {plan.sha256} (uv refuses any other file)")
    print(f"  license      {release.license or 'unknown'}")
    print(f"  isaaclab     {release.isaaclab or '-'} (installed release: {isaaclab_version})")
    print(f"  tasks        {', '.join(release.tasks) or '-'}")
    print(f"  requirement  {plan.requirement}")
