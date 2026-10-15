# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Browse, install, and publish packages from the Isaac Lab community registry (prototype)."""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import tomllib

from ..registry import (
    DEPENDENCY_GROUP,
    SECURITY_WARNING,
    RegistryError,
    is_isaaclab_distribution,
    normalize_name,
    warn,
)
from ..registry.environment import Target, detect_target, isaaclab_release_version
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
from ..registry.installer import (
    InstallPlan,
    install_into_environment,
    install_into_project,
    is_raw_spec,
    missing_rl_libraries,
    plan_for_release,
    plan_for_spec,
    uninstall_from_environment,
    uninstall_from_project,
)
from ..registry.publish import draft_request, render_request, submit_request
from ..registry.state import (
    describe_distribution,
    forget_install,
    installed_distributions,
    installed_packages,
    load_state,
    project_group,
    state_path,
    task_distributions,
)
from ..utils import print_error, print_info

_EPILOG = f"""\
environment:
  ISAACLAB_REGISTRY_URL  index location (default: the public registry index)
  ISAACLAB_REGISTRY_TTL  seconds a fetched index stays cached (default: 3600)
  ISAACLAB_OFFLINE=1     use the cached index only

Install targets: inside your own uv project, packages are added to its '{DEPENDENCY_GROUP}' dependency group
with 'uv add'. Anywhere else (an Isaac Lab checkout, a wheel virtual environment), they are installed into the
running environment with 'uv pip install' and recorded for 'isaaclab registry restore'.

{SECURITY_WARNING}
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
    except RegistryError as error:
        print_error(str(error), stream=sys.stderr)
        return 1
    except ModuleNotFoundError as error:
        if error.name not in ("packaging", "prettytable"):
            raise
        print_error(
            f"isaaclab registry needs the '{error.name}' module: uv pip install {error.name}", stream=sys.stderr
        )
        return 1
    except KeyboardInterrupt:
        print_error("Interrupted.", stream=sys.stderr)
        return 130


def _build_parser() -> argparse.ArgumentParser:
    """Build the ``registry`` parser and its subcommands."""
    # Index options are accepted before or after the subcommand; SUPPRESS keeps one from clobbering the other.
    index_options = argparse.ArgumentParser(add_help=False)
    group = index_options.add_argument_group("index options")
    group.add_argument(
        "--index",
        metavar="URL|PATH",
        default=argparse.SUPPRESS,
        help="Registry index: an index.json URL or file, or a registry checkout (uses its dist/index.json).",
    )
    group.add_argument(
        "--refresh", action="store_true", default=argparse.SUPPRESS, help="Fetch the index even if the cache is fresh."
    )
    group.add_argument(
        "--offline", action="store_true", default=argparse.SUPPRESS, help="Use the cached index without network."
    )

    target_options = argparse.ArgumentParser(add_help=False)
    group = target_options.add_argument_group("target options")
    group.add_argument(
        "--mode",
        choices=("auto", "project", "env"),
        default="auto",
        help="'project' edits your uv project with uv add/remove; 'env' changes the running environment with uv pip."
        " 'auto' (default) picks project mode when this command runs in its project's environment.",
    )
    group.add_argument("--project", metavar="PATH", help="Look for the uv project from PATH instead of the cwd.")
    group.add_argument(
        "--python",
        metavar="EXE",
        help="Manage the environment of this interpreter (for example Isaac Sim's python.sh environment) instead of"
        " the running one (env mode).",
    )

    parser = argparse.ArgumentParser(
        prog=f"{_program_name()} registry",
        description=__doc__,
        parents=[index_options],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND", title="commands")

    def add(name: str, handler: Any, summary: str, *parents: argparse.ArgumentParser) -> argparse.ArgumentParser:
        subparser = subparsers.add_parser(name, help=summary, description=summary, parents=list(parents))
        subparser.set_defaults(handler=handler)
        return subparser

    subparser = add("list", _command_list, "List packages compatible with this Isaac Lab.", index_options)
    subparser.add_argument("--all", action="store_true", help="Include end-of-life and incompatible packages.")
    subparser.add_argument("--category", help="Only show this category (for example 'locomotion').")
    subparser.add_argument("--json", action="store_true", help="Print JSON.")

    subparser = add(
        "search", _command_search, "Search names, descriptions, tags, tasks, and categories.", index_options
    )
    subparser.add_argument("query", help="Case-insensitive text to look for.")
    subparser.add_argument("--json", action="store_true", help="Print JSON.")

    subparser = add("info", _command_info, "Show a package, its releases, and their compatibility.", index_options)
    subparser.add_argument("name", help="Package name.")
    subparser.add_argument("--json", action="store_true", help="Print JSON.")

    subparser = add(
        "install",
        _command_install,
        "Install registry packages: NAME, NAME==VERSION, or (without the index) 'NAME @ git+URL@REF' or a path.",
        index_options,
        target_options,
    )
    subparser.add_argument("packages", nargs="+", metavar="PACKAGE", help="Packages to install.")
    subparser.add_argument("-y", "--yes", action="store_true", help="Do not ask for confirmation.")
    subparser.add_argument("--dry-run", action="store_true", help="Show what would happen without changing anything.")
    subparser.add_argument(
        "--force",
        action="store_true",
        help="Install a release that is not compatible, replace a library the registry did not install, or keep"
        " an install that introduces dependency conflicts.",
    )

    subparser = add("uninstall", _command_uninstall, "Uninstall registry packages.", target_options)
    subparser.add_argument("packages", nargs="+", metavar="NAME", help="Packages to uninstall.")
    subparser.add_argument("-y", "--yes", action="store_true", help="Do not ask for confirmation.")
    subparser.add_argument("--dry-run", action="store_true", help="Show what would happen without changing anything.")

    subparser = add(
        "installed",
        _command_installed,
        "List installed task packages and how they relate to the registry.",
        index_options,
        target_options,
    )
    subparser.add_argument("--json", action="store_true", help="Print JSON.")

    subparser = add(
        "restore",
        _command_restore,
        "Reinstall env-mode packages that were removed, e.g. by 'uv sync' (never ones the registry blocklisted).",
        index_options,
        target_options,
    )
    subparser.add_argument("-y", "--yes", action="store_true", help="Do not ask for confirmation.")
    subparser.add_argument("--dry-run", action="store_true", help="Show what would happen without changing anything.")

    subparser = add("publish", _command_publish, "Draft (and optionally submit) a registration request.")
    subparser.add_argument("path", nargs="?", default=".", help="Project directory (default: the cwd).")
    subparser.add_argument("--ref", help="Git tag to register (default: v<version>).")
    subparser.add_argument("--as", dest="requested_by", metavar="USER", help="GitHub handle of the requester.")
    subparser.add_argument("--repository", metavar="URL", help="Repository URL (default: the 'origin' remote).")
    subparser.add_argument("--output", metavar="FILE", help="Write the request to FILE ('-' for stdout).")
    subparser.add_argument(
        "--registry", metavar="DIR", help="Registry repository: submit the request to its bot (prototype)."
    )
    subparser.add_argument(
        "--pypi", action="store_true", help="Register the wheel published on a package index instead of git."
    )
    subparser.add_argument(
        "--index", dest="package_index", metavar="URL", help="PEP 503 index for --pypi (default: PyPI)."
    )
    subparser.add_argument(
        "--force", action="store_true", help="Write or submit the request even if pre-flight checks fail."
    )
    subparser.add_argument(
        "--smoke", action="store_true", help="With --registry: ask the bot to smoke-test with this Isaac Lab."
    )
    subparser.add_argument("--now", metavar="WHEN", help="With --registry: the bot's simulated time (e.g. +3d).")
    return parser


# Browsing.


def _command_list(args: argparse.Namespace) -> int:
    """Implement ``registry list``."""
    index = _load_index(args)
    isaaclab_version = isaaclab_release_version()
    installed = task_distributions()
    rows = []
    hidden = 0
    for package in index.packages.values():
        if args.category and package.category != args.category:
            continue
        row = _package_row(index, package, isaaclab_version, installed)
        if not args.all and (not row["compatible"] or package.status == "end-of-life"):
            hidden += 1
            continue
        rows.append(row)
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    _print_package_table(rows, f"{index.name}: packages for Isaac Lab {isaaclab_version}")
    if hidden:
        print_info(f"{hidden} incompatible or end-of-life package(s) hidden; use --all to show them.")
    return 0


def _command_search(args: argparse.Namespace) -> int:
    """Implement ``registry search``."""
    index = _load_index(args)
    isaaclab_version = isaaclab_release_version()
    installed = task_distributions()
    query = args.query.lower()
    rows = [
        _package_row(index, package, isaaclab_version, installed)
        for package in index.packages.values()
        if any(query in text.lower() for text in _searchable_text(package))
    ]
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        print_info(f"No package matches '{args.query}'.")
        return 0
    _print_package_table(rows, f"{index.name}: packages matching '{args.query}'")
    return 0


def _command_info(args: argparse.Namespace) -> int:
    """Implement ``registry info``."""
    index = _load_index(args)
    package = index.get(args.name)
    if package is None:
        raise RegistryError(f"'{args.name}' is not in the {index.name} registry.")
    isaaclab_version = isaaclab_release_version()
    selected = latest_compatible(index, package, isaaclab_version)
    installed = task_distributions().get(normalize_name(package.name))
    releases = [
        {**_release_json(release), "verdict": _verdict(index, package, release, isaaclab_version)}
        for release in package.releases
    ]
    if args.json:
        document = {
            **_package_json(package),
            "isaaclab_version": isaaclab_version,
            "selected": selected.version if selected else None,
            "installed": {"version": installed.version, "commit": installed.commit} if installed else None,
            "blocklisted": (block.reason if (block := index.block_for(package.name)) else None),
            "releases": releases,
        }
        print(json.dumps(document, indent=2))
        return 0

    source = package.source
    if package.source_type == "git":
        where = source.get("repository", "")
        where += f" (subdirectory {source['subdirectory']})" if source.get("subdirectory") else ""
    else:
        where = f"{source.get('project', package.name)} on {source.get('index', 'PyPI')}"
    status = package.status + (f": {package.status_description}" if package.status_description else "")
    fields = [
        ("status", status),
        ("category", package.category),
        ("tags", ", ".join(package.tags)),
        ("maintainers", ", ".join(package.maintainers)),
        ("homepage", package.homepage),
        ("media", ", ".join(f"{kind}: {url}" for kind, url in package.media.items())),
        ("source", f"{package.source_type} {where}"),
        ("registry", f"{index.name} (index generated {index.generated_at or 'unknown'})"),
        (
            "installed",
            f"{installed.version}" + (f" (commit {installed.commit})" if installed.commit else "") if installed else "",
        ),
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
    if block := index.block_for(package.name):
        warn(f"Blocklisted by the registry: {block.reason}")
    from prettytable import PrettyTable

    table = PrettyTable(["Version", "Source", "Isaac Lab", "Python", "Tasks", "RL libraries", "Registered", "Verdict"])
    table.align = "l"
    for release, document in zip(package.releases, releases):
        table.add_row(
            [
                release.version + (" *" if release is selected else ""),
                _release_source(release),
                release.isaaclab or "-",
                release.requires_python or "-",
                "\n".join(release.tasks) or "-",
                ", ".join(release.rl_libraries) or "-",
                release.registered_at[:10] or "-",
                document["verdict"],
            ]
        )
    print(table)
    if selected is not None:
        print("* release selected by 'isaaclab registry install'")
    return 0


# Installing.


def _command_install(args: argparse.Namespace) -> int:
    """Implement ``registry install``."""
    target = detect_target(args.mode, args.project, args.python)
    isaaclab_version = target.isaaclab_version
    index: RegistryIndex | None = None
    plans: dict[str, InstallPlan] = {}
    releases: dict[str, tuple[Package, Release]] = {}
    for spec in args.packages:
        if is_raw_spec(spec):
            plan = plan_for_spec(spec)
            warn(f"{spec} is installed as-is: no index, compatibility gate, or blocklist applies.")
        else:
            name, _, version = spec.partition("==")
            if re.search(r"[<>!=~@\s]", name.strip()):
                raise RegistryError(f"Use NAME or NAME==VERSION, not {spec!r}.")
            index = index or _load_index(args)
            selection = select_release(
                index, name.strip(), isaaclab_version, version.strip() or None, args.force, target.python_version
            )
            for warning in selection.warnings:
                warn(warning)
            plan = plan_for_release(selection.package, selection.release)
            releases[plan.name] = (selection.package, selection.release)
        if is_isaaclab_distribution(plan.name):
            raise RegistryError(f"Refusing to install {plan.name}: Isaac Lab itself is not managed by the registry.")
        plans[plan.name] = plan

    if target.uvx:
        with_options = " ".join(f'--with "{plan.requirement}"' for plan in plans.values())
        raise RegistryError(
            "This is a temporary uvx environment, so an install would be lost. Run Isaac Lab with the package instead:"
            f"\n    uvx {with_options} isaaclab …"
        )
    _check_replacements(plans, target, args.force)

    for plan in plans.values():
        _print_install_summary(plan, releases.get(plan.name), isaaclab_version)
    print(f"Target: {target.describe()}")
    warn(SECURITY_WARNING)
    if not args.dry_run and not _confirm(f"Install {', '.join(plans)}?", args.yes):
        print_info("Nothing was installed.")
        return 1

    if target.mode == "project":
        install_into_project(list(plans.values()), target, dry_run=args.dry_run)
    else:
        install_into_environment(list(plans.values()), target, dry_run=args.dry_run, keep_conflicts=args.force)
    if args.dry_run:
        return 0

    print_info(f"Installed {', '.join(plans)} into the {target.mode} target.")
    for name, (_, release) in releases.items():
        for library, extra in missing_rl_libraries(release.rl_libraries, target):
            warn(f"{name} uses {library}, which is not installed. {_rl_library_hint(target, extra)}")
    if target.mode == "env" and target.uv_cwd is not None:
        print_info(
            "'uv run' keeps packages installed this way, but a plain 'uv sync' in the checkout removes them;"
            " run 'isaaclab registry restore' afterwards."
        )
    return 0


def _check_replacements(plans: dict[str, InstallPlan], target: Target, force: bool) -> None:
    """Refuse to replace an installed distribution that the registry does not manage, unless forced.

    Mirrors the ``uninstall`` guard: a registry entry named like a library that is already installed (for
    example ``gymnasium``), or like the user's own local project, must not silently replace it.

    Raises:
        RegistryError: If a plan would replace such a distribution and ``force`` is not set. Also raised
            when the env-mode state file cannot be read, before anything changes.
    """
    managed = set(load_state(target.prefix)) if target.mode == "env" else set()
    if target.project_root is not None:
        managed.update(project_group(target.project_root))
    distributions = installed_distributions(target.site_paths)
    for name in plans:
        distribution = distributions.get(name)
        if distribution is None or name in managed:
            continue
        described = describe_distribution(distribution)
        if described.local:
            problem = f"it would replace the local install {described.url or ''} of {name} {described.version}"
        elif not described.entry_points:
            problem = f"it would replace {name} {described.version}, which provides no Isaac Lab tasks"
        else:
            continue
        if not force:
            raise RegistryError(
                f"Refusing to install {name}: {problem} and was not installed by the registry. Pass --force to"
                " replace it."
            )
        warn(f"--force: {problem}.")


def _command_uninstall(args: argparse.Namespace) -> int:
    """Implement ``registry uninstall``."""
    target = detect_target(args.mode, args.project, args.python)
    if target.uvx:
        raise RegistryError("This is a temporary uvx environment; drop the --with option instead.")
    distributions = installed_distributions(target.site_paths)
    recorded = load_state(target.prefix) if target.mode == "env" else {}
    group = project_group(target.project_root) if target.project_root is not None else []
    names = list(dict.fromkeys(normalize_name(name) for name in args.packages))
    for name in names:
        if is_isaaclab_distribution(name):
            raise RegistryError(f"Refusing to uninstall {name}: Isaac Lab packages are not managed by the registry.")
        if target.mode == "project":
            if name not in group:
                raise RegistryError(f"{name} is not in the '{DEPENDENCY_GROUP}' group of {target.project_root}.")
            continue
        distribution = distributions.get(name)
        if distribution is None:
            if name not in recorded:
                raise RegistryError(f"{name} is not installed in {target.prefix}.")
            continue
        described = describe_distribution(distribution)
        if name not in recorded and not described.entry_points:
            raise RegistryError(
                f"Refusing to uninstall {name}: the registry did not install it and it provides no tasks."
            )
        if name not in recorded and described.local:
            raise RegistryError(f"Refusing to uninstall {name}: it is a local install ({described.url}).")

    print(f"Target: {target.describe()}")
    if not args.dry_run and not _confirm(f"Uninstall {', '.join(names)}?", args.yes):
        print_info("Nothing was uninstalled.")
        return 1
    for name in names:
        if target.mode == "project":
            uninstall_from_project(name, target, dry_run=args.dry_run)
            continue
        if name in distributions:
            uninstall_from_environment(name, target, dry_run=args.dry_run)
        else:
            print_info(f"{name} was already removed from the environment.")
        if not args.dry_run and forget_install(target.prefix, name):
            print_info(f"Dropped {name} from {state_path(target.prefix)}.")
    if not args.dry_run:
        later = "the next exact 'uv sync' removes them" if target.mode == "project" else "'uv pip uninstall' keeps them"
        print_info(f"Dependencies that only these packages needed stay installed; {later}.")
    return 0


def _command_installed(args: argparse.Namespace) -> int:
    """Implement ``registry installed``."""
    target = detect_target(args.mode, args.project, args.python)
    try:
        index = _load_index(args)
    except RegistryError as error:
        warn(f"{error} Statuses that need the index are unavailable.")
        index = None
    rows = installed_packages(target, index, target.isaaclab_version, target.python_version)
    if args.json:
        document = {
            "mode": target.mode,
            "prefix": str(target.prefix),
            "project": str(target.project_root) if target.project_root else None,
            "packages": [row.to_json() for row in rows],
        }
        print(json.dumps(document, indent=2))
        return 0
    print(f"Target: {target.describe()}")
    if not rows:
        print_info("No Isaac Lab task packages are installed.")
        return 0
    from prettytable import PrettyTable

    table = PrettyTable(["Name", "Version", "Status", "Commit", "Recorded in", "Note"])
    table.align = "l"
    for row in rows:
        table.add_row(
            [
                row.name,
                row.version or "-",
                row.status,
                (row.commit or "-")[:12],
                row.recorded_in or "-",
                row.note,
            ]
        )
    print(table)
    return 0


def _command_restore(args: argparse.Namespace) -> int:
    """Implement ``registry restore``."""
    target = detect_target(args.mode, args.project, args.python)
    if target.mode == "project":
        print_info(f"Project mode: 'uv sync' restores the '{DEPENDENCY_GROUP}' group from uv.lock.")
        return 0
    if target.uvx:
        raise RegistryError("This is a temporary uvx environment; there is nothing to restore.")
    distributions = installed_distributions(target.site_paths)
    missing = {name: entry for name, entry in load_state(target.prefix).items() if name not in distributions}
    if not missing:
        print_info("Nothing to restore: every recorded package is installed.")
        return 0
    try:
        index = _load_index(args)
    except RegistryError as error:
        warn(f"{error} The blocklist cannot be checked; restoring the recorded commits and wheels as they are.")
        index = None
    refused = []
    for name, entry in missing.items():
        block = index.block_for_install(name, entry.get("commit"), entry.get("version")) if index else None
        if block is not None:
            # The registry's kill switch: a package blocklisted after it was installed must not come back.
            print_error(
                f"Not restoring {name}: the registry blocklisted it ({block.reason}). Drop it with"
                f" 'isaaclab registry uninstall {name}'.",
                stream=sys.stderr,
            )
            refused.append(name)
        elif index is not None and (yanked := _recorded_release_yank(index, name, entry)):
            warn(f"Restoring {name} {entry.get('version') or ''}, which was yanked since: {yanked}")
    plans = [
        InstallPlan(
            name,
            entry["requirement"],
            version=entry.get("version"),
            commit=entry.get("commit"),
            sha256=entry.get("sha256"),
        )
        for name, entry in missing.items()
        if name not in refused
    ]
    if not plans:
        return 1
    for plan in plans:
        print(f"  {plan.name} {plan.version or ''}  {plan.requirement}")
    print(f"Target: {target.describe()}")
    warn(SECURITY_WARNING)
    names = [plan.name for plan in plans]
    if not args.dry_run and not _confirm(f"Reinstall {', '.join(names)}?", args.yes):
        print_info("Nothing was restored.")
        return 1
    install_into_environment(plans, target, dry_run=args.dry_run)
    if not args.dry_run:
        print_info(f"Restored {', '.join(names)}.")
    return 1 if refused else 0


# Publishing.


def _command_publish(args: argparse.Namespace) -> int:
    """Implement ``registry publish``."""
    draft = draft_request(
        Path(args.path),
        ref=args.ref,
        repository=args.repository,
        requested_by=args.requested_by,
        pypi=args.pypi,
        package_index=args.package_index,
    )
    request = draft.request
    print(f"Pre-flight checks for {request['name']}:", file=sys.stderr)
    for check in draft.checks:
        print(f"  {'pass' if check.ok else 'FAIL'}  {check.name:<22} {check.message}", file=sys.stderr)
    if not draft.ok:
        if not args.force:
            print_error(
                "Pre-flight checks failed. Fix them, or pass --force to let the registry bot report them.",
                stream=sys.stderr,
            )
            return 1
        warn("--force: continuing despite failed pre-flight checks.")

    text = render_request(request)
    output = Path(args.output).resolve() if args.output and args.output != "-" else None
    if output is not None:
        output.write_text(text, encoding="utf-8")
        print_info(f"Wrote the registration request to {output}.")
    elif args.output == "-" or not args.registry:
        sys.stdout.write(text)
    if not args.registry:
        return 0
    bot_args = ["--smoke", "--isaaclab-python", sys.executable] if args.smoke else []
    bot_args += ["--now", args.now] if args.now else []
    if output is not None:
        return submit_request(Path(args.registry), output, bot_args)
    with tempfile.TemporaryDirectory(prefix="isaaclab-registry-") as temporary:
        request_file = Path(temporary) / f"{request['name']}-request.toml"
        request_file.write_text(text, encoding="utf-8")
        return submit_request(Path(args.registry), request_file, bot_args)


# Helpers.


def _program_name() -> str:
    """Return the executable name to show in usage lines."""
    name = Path(sys.argv[0]).name
    return "isaaclab" if name in ("", "-c", "__main__.py") else name


def _load_index(args: argparse.Namespace) -> RegistryIndex:
    """Load the index selected by the global index options."""
    location = resolve_index_location(getattr(args, "index", None))
    return load_index(location, refresh=getattr(args, "refresh", False), offline=getattr(args, "offline", False))


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


def _recorded_release_yank(index: RegistryIndex, name: str, entry: dict[str, Any]) -> str | None:
    """Return the yank reason of the index release a state entry recorded, if it was yanked."""
    package = index.get(name)
    if package is None:
        return None
    release = next((r for r in package.releases if entry.get("commit") and r.commit == entry["commit"]), None)
    release = release or (package.release(entry["version"]) if entry.get("version") else None)
    return release.yanked_reason if release is not None else None


def _searchable_text(package: Package) -> list[str]:
    """Return the fields ``search`` matches against."""
    texts = [package.name, package.description, package.category, *package.tags]
    for release in package.releases:
        texts += [release.summary, *release.tasks]
    return texts


def _package_row(
    index: RegistryIndex, package: Package, isaaclab_version: str | None, installed: dict[str, Any]
) -> dict[str, Any]:
    """Return one ``list``/``search`` row."""
    release = latest_compatible(index, package, isaaclab_version)
    newest_live = next((candidate for candidate in package.releases if candidate.yanked_reason is None), None)
    if (block := index.block_for(package.name)) is not None:
        compat = f"blocklisted: {block.reason}"
    elif release is not None:
        compat = "ok"
    elif newest_live is None:
        compat = "no live release"
    else:
        compat = compatibility_problem(newest_live, isaaclab_version) or "blocklisted"
    shown = release or newest_live
    distribution = installed.get(normalize_name(package.name))
    return {
        "name": package.name,
        "version": release.version if release else None,
        "latest": package.latest,
        "status": package.status,
        "category": package.category,
        "tasks": len(shown.tasks) if shown else 0,
        "compatible": release is not None,
        "compat": compat,
        "installed": distribution.version if distribution else None,
        "description": package.summary,
    }


def _print_package_table(rows: list[dict[str, Any]], title: str) -> None:
    """Print ``list``/``search`` rows as a table."""
    from prettytable import PrettyTable

    table = PrettyTable(["Name", "Version", "Category", "Tasks", "Status", "Compatibility", "Installed", "Description"])
    table.title = title
    table.align = "l"
    table.max_width["Description"] = 48
    for row in rows:
        table.add_row(
            [
                row["name"],
                row["version"] or "-",
                row["category"],
                row["tasks"],
                row["status"],
                row["compat"],
                row["installed"] or "",
                row["description"],
            ]
        )
    print(table)


def _package_json(package: Package) -> dict[str, Any]:
    """Return the identity fields of a package as JSON."""
    return {
        "name": package.name,
        "description": package.summary,
        "category": package.category,
        "tags": list(package.tags),
        "maintainers": list(package.maintainers),
        "homepage": package.homepage,
        "media": package.media,
        "status": package.status,
        "status_description": package.status_description,
        "source": package.source,
        "latest": package.latest,
    }


def _release_json(release: Release) -> dict[str, Any]:
    """Return a release as JSON."""
    return {
        "version": release.version,
        "commit": release.commit,
        "git_tag": release.git_tag,
        "wheel": release.wheel,
        "summary": release.summary,
        "license": release.license,
        "isaaclab": release.isaaclab,
        "requires_python": release.requires_python,
        "entry_points": list(release.entry_points),
        "tasks": list(release.tasks),
        "rl_libraries": list(release.rl_libraries),
        "registered_at": release.registered_at,
        "yanked": release.yanked,
    }


def _verdict(index: RegistryIndex, package: Package, release: Release, isaaclab_version: str | None) -> str:
    """Return whether a release can be installed here, and why not."""
    if (block := index.block_for(package.name, release)) is not None:
        return f"blocklisted: {block.reason}"
    if release.yanked_reason is not None:
        return f"yanked: {release.yanked_reason}"
    return compatibility_problem(release, isaaclab_version) or "ok"


def _release_source(release: Release) -> str:
    """Return the pinned source of a release for tables."""
    if release.commit:
        return release.commit[:12] + (f" ({release.git_tag})" if release.git_tag else "")
    if release.wheel:
        return f"{release.wheel.get('filename', 'wheel')} sha256:{release.wheel.get('sha256', '?')[:12]}"
    return "-"


def _print_install_summary(
    plan: InstallPlan, selected: tuple[Package, Release] | None, isaaclab_version: str | None
) -> None:
    """Print what is about to be installed."""
    if selected is None:
        print(f"{plan.name} (direct spec)\n  requirement  {plan.requirement}")
        return
    package, release = selected
    if package.source_type == "git":
        source = f"git {package.source.get('repository')} @ {release.commit}"
        source += f" (tag {release.git_tag})" if release.git_tag else ""
        source += f", subdirectory {package.source['subdirectory']}" if package.source.get("subdirectory") else ""
    else:
        wheel = release.wheel or {}
        source = f"pypi {package.source.get('index', 'https://pypi.org/simple')}: {wheel.get('filename', plan.name)}"
        source += f"\n  sha256       {wheel.get('sha256', 'unknown')} (uv refuses any other file)"
    print(f"{package.name} {release.version}: {package.summary}")
    print(f"  source       {source}")
    print(f"  license      {release.license or 'unknown'}")
    print(f"  isaaclab     {release.isaaclab or '-'} (installed release: {isaaclab_version})")
    print(f"  tasks        {', '.join(release.tasks) or '-'}")
    print(f"  requirement  {plan.requirement}")


def _rl_library_hint(target: Target, extra: str) -> str:
    """Explain how to make an RL library available in the target."""
    if target.mode == "project":
        data = tomllib.loads((target.project_root / "pyproject.toml").read_text(encoding="utf-8"))
        sources = data.get("tool", {}).get("uv", {}).get("sources", {})
        if any(normalize_name(name) == "isaaclab" and "path" in source for name, source in sources.items()):
            # Generated from a checkout: the peer override drops extras on 'isaaclab', which the checkout's
            # 'isaaclab' does not provide anyway; they live on 'isaaclab-dev'.
            add = f"add 'isaaclab-dev[{extra}]' to the 'isaaclab-dev' dependency group of pyproject.toml"
            if extra in data.get("dependency-groups", {}):
                return f"This project takes Isaac Lab from a checkout: run 'uv run --group {extra} …', or {add}."
            return f"This project takes Isaac Lab from a checkout: {add}."
        return f"Add the '{extra}' extra to your project's isaaclab dependency."
    if target.uv_cwd is not None:
        return (
            f"Run Isaac Lab with it, e.g. 'uv run --extra {extra} isaaclab train …'"
            f" ('uv sync --extra {extra}' would also remove registry packages)."
        )
    return f"Install it with: uv pip install 'isaaclab[{extra}]'"
