# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ``isaaclab registry`` client."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import pytest
import tomllib

from isaaclab.cli.commands.registry import command_registry
from isaaclab.cli.registry import RegistryError, environment, installer
from isaaclab.cli.registry import index as index_module
from isaaclab.cli.registry.environment import Target, detect_target
from isaaclab.cli.registry.index import load_index, parse_index, resolve_index_location, select_release
from isaaclab.cli.registry.installer import (
    InstallPlan,
    install_into_environment,
    install_into_project,
    plan_for_release,
    uninstall_from_project,
)
from isaaclab.cli.registry.state import installed_packages, load_state, record_install

pytestmark = pytest.mark.unit

_GIT_SOURCE = {"type": "git", "repository": "https://github.com/alice/walker", "subdirectory": ""}
_PYPI_SOURCE = {"type": "pypi", "index": "http://127.0.0.1:8765/simple", "project": "walker"}
_WHEEL_URL = "http://127.0.0.1:8765/files/walker-1.0-py3-none-any.whl"


def _commit(character: str) -> str:
    """Return a fake 40-hex commit."""
    return character * 40


def _release(version: str, commit: str | None = None, isaaclab: str = ">=3.0.0,<3.1", **fields) -> dict:
    """Return an ``index.json`` release entry."""
    return {"version": version, "commit": commit, "isaaclab": isaaclab, "tasks": ["Walker-v0"], **fields}


def _write_distribution(
    site: Path,
    name: str,
    version: str,
    *,
    tasks: bool = True,
    direct_url: dict | None = None,
    extras: tuple[str, ...] = (),
    requires: tuple[str, ...] = (),
) -> None:
    """Write a minimal installed ``.dist-info`` directory."""
    module = name.replace("-", "_")
    dist_info = site / f"{module}-{version}.dist-info"
    dist_info.mkdir(parents=True)
    lines = ["Metadata-Version: 2.1", f"Name: {name}", f"Version: {version}"]
    lines += [f"Provides-Extra: {extra}" for extra in extras] + [f"Requires-Dist: {req}" for req in requires]
    (dist_info / "METADATA").write_text("\n".join(lines) + "\n")
    if tasks:
        (dist_info / "entry_points.txt").write_text(f"[isaaclab.tasks]\n{module} = {module}.tasks\n")
    if direct_url is not None:
        (dist_info / "direct_url.json").write_text(json.dumps(direct_url))


@pytest.fixture
def registry_index():
    """An index whose releases are deliberately out of order."""
    return parse_index(
        {
            "schema": 1,
            "name": "test-registry",
            "blocklist": [
                {"name": "walker", "commit": _commit("c"), "version": None, "reason": "leaks credentials"},
                {"name": "evil", "commit": None, "version": None, "reason": "malware"},
                {"name": "gone-bad", "commit": None, "version": "2.0", "reason": "malware"},
            ],
            "packages": [
                {
                    "name": "walker",
                    "source": _GIT_SOURCE,
                    "releases": [
                        _release("0.1.0", _commit("a")),
                        _release("0.4.0", _commit("d"), isaaclab=">=4,<5"),
                        _release("0.2.0", _commit("b")),
                        _release("0.2.1", _commit("c")),
                        _release("0.3.0", _commit("e"), yanked="broken build"),
                    ],
                },
                {
                    "name": "reacher",
                    "source": {"type": "pypi", "index": "https://pypi.org/simple", "project": "reacher"},
                    "releases": [
                        _release("1.0.0", wheel={"filename": "reacher-1.0.0-py3-none-any.whl", "sha256": "f" * 64})
                    ],
                },
                {
                    "name": "retired",
                    "source": _GIT_SOURCE,
                    "status": "end-of-life",
                    "status_description": "Use walker.",
                    "releases": [_release("1.0.0", _commit("f"))],
                },
            ],
        }
    )


@pytest.mark.parametrize(
    ("name", "version", "force", "expected_version", "message"),
    [
        ("Walker", None, False, "0.2.0", None),
        ("walker", "0.3.0", False, "0.3.0", "yanked: broken build"),
        ("walker", "0.4.0", False, None, "needs isaaclab >=4,<5 (have 3.0.0)"),
        ("walker", "0.4.0", True, "0.4.0", "--force"),
        ("walker", "0.2.1", False, None, "blocklisted by the registry: leaks credentials"),
        ("evil", None, False, None, "blocklisted by the registry: malware"),
        ("retired", None, False, "1.0.0", "end-of-life. Use walker."),
    ],
    ids=["newest-compatible", "pinned-yanked", "incompatible", "forced", "blocked-commit", "blocked-package", "eol"],
)
def test_release_selection_policy(registry_index, name, version, force, expected_version, message):
    """Install picks the newest live, unblocked, compatible release and refuses or warns on everything else."""
    if expected_version is None:
        with pytest.raises(RegistryError, match=re.escape(message)):
            select_release(registry_index, name, "3.0.0", version, force)
        return
    selection = select_release(registry_index, name, "3.0.0", version, force)
    assert selection.release.version == expected_version
    assert any(message in warning for warning in selection.warnings) if message else not selection.warnings


def test_index_locations_and_remote_cache(tmp_path, monkeypatch, capsys):
    """Remote indexes are cached with a TTL and survive network failures and bad responses; local ones are not."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv("ISAACLAB_OFFLINE", raising=False)
    monkeypatch.delenv("ISAACLAB_REGISTRY_TTL", raising=False)
    fetched = []

    def fetch(url):
        fetched.append(url)
        if len(fetched) == 3:
            raise urllib.error.URLError("network down")
        if len(fetched) == 4:
            return b"<html>Sign in to Wi-Fi</html>"  # a captive portal answers 200
        return json.dumps({"schema": 1, "name": f"fetch-{len(fetched)}", "packages": []}).encode()

    monkeypatch.setattr(index_module, "_fetch", fetch)
    url = "https://registry.example.org/index.json"
    assert load_index(url).name == "fetch-1"
    assert load_index(url).name == "fetch-1"
    assert load_index(url, refresh=True).name == "fetch-2"
    monkeypatch.setenv("ISAACLAB_REGISTRY_TTL", "0")
    assert load_index(url).name == "fetch-2"
    assert "using the cached copy" in capsys.readouterr().err
    assert load_index(url, refresh=True).name == "fetch-2"  # the bad response never replaces the good copy
    assert "not valid JSON" in capsys.readouterr().err
    assert load_index(url, offline=True).name == "fetch-2"
    cache = index_module.cache_path_for(url)
    cache.write_text("{truncated")
    with pytest.raises(RegistryError, match=f"no usable cached copy .*{re.escape(str(cache))}"):
        load_index(url, offline=True)
    assert load_index(url).name == "fetch-5"  # a corrupt cache is fetched again
    assert len(fetched) == 5

    checkout = tmp_path / "isaaclab-registry"
    (checkout / "dist").mkdir(parents=True)
    (checkout / "dist" / "index.json").write_text(json.dumps({"schema": 1, "name": "local", "packages": []}))
    assert load_index(resolve_index_location(str(checkout))).name == "local"
    assert len(list((tmp_path / "cache" / "isaaclab" / "registry").iterdir())) == 1
    with pytest.raises(RegistryError, match="schema"):
        parse_index({"schema": 2, "packages": []})


def test_index_entries_are_validated_and_sanitized(capsys):
    """Malformed entries are left out with a warning, and control characters never reach the terminal."""
    good = {"name": "walker", "source": _GIT_SOURCE, "releases": [_release("0.1.0", _commit("a"))]}
    index = parse_index(
        {
            "schema": 1,
            "packages": [
                {**good, "description": "Nice\r\x1b[2K\x1b[1Asource  git https://trusted.example.org"},
                {**good, "name": "Dummy_Walk[evil]"},
                {**good, "name": "stringly", "source": "git"},
                {**good, "name": "badversion", "releases": [_release("not a version", _commit("b"))]},
            ],
        }
    )

    assert sorted(index.packages) == ["badversion", "walker"]
    assert index.get("walker").description == "Nice??[2K?[1Asource  git https://trusted.example.org"
    assert index.get("badversion").releases == ()
    warnings = capsys.readouterr().err
    assert "Dummy_Walk[evil]" in warnings and "stringly" in warnings and "not a version" in warnings
    with pytest.raises(RegistryError, match="blocklist"):
        parse_index({"schema": 1, "packages": [], "blocklist": {"name": "evil"}})


@pytest.mark.parametrize(
    ("project_name", "prefix", "mode", "expected", "setting"),
    [
        ("my-lab", "project/.venv", "auto", "project", None),
        ("my-lab", "project/envs/dev", "auto", "project", "envs/dev"),
        ("my-lab", "other-venv", "auto", "env", None),
        ("my-lab", "project/.venv", "auto", "project", "uv-run-from-elsewhere"),
        ("isaaclab-dev", "project/.venv", "auto", "env", None),
        ("isaaclab-dev", "project/.venv", "project", "error", None),
        ("isaaclab-tasks", "project/.venv", "auto", "env", None),
        ("my-lab", "project/.venv", "auto", "env", "inside-checkout"),
        ("my-lab", "shared-venv", "auto", "env", "checkout-environment"),
        ("my-lab", "uv-cache/archive-v0/abc", "auto", "uvx", None),
        ("my-lab", "project/.venv", "auto", "project", "explicit-project"),
        ("my-lab", "other-venv", "auto", "error", "explicit-project"),
        ("my-lab", "shared-venv", "auto", "error", "explicit-project-checkout-environment"),
    ],
    ids=[
        "project-env",
        "uv-project-environment",
        "other-env",
        "uv-run-project-from-elsewhere",
        "checkout",
        "checkout-forced-project",
        "checkout-sub-package",
        "inside-checkout",
        "shared-checkout-environment",
        "uvx",
        "explicit-project",
        "explicit-project-other-env",
        "explicit-project-checkout-environment",
    ],
)
def test_target_detection(tmp_path, monkeypatch, project_name, prefix, mode, expected, setting):
    """Project mode applies only to the user's own project whose environment runs the command, never the checkout.

    An explicit ``--project`` never falls back to installing into the running environment.
    """
    root = tmp_path / "project"
    (root / "src" / "nested").mkdir(parents=True)
    (root / "pyproject.toml").write_text(f'[project]\nname = "{project_name}"\nversion = "0.1.0"\n')
    checkout = tmp_path if setting == "inside-checkout" else tmp_path / "IsaacLab"
    checkout.mkdir(exist_ok=True)
    (checkout / "VERSION").write_text("3.0.0\n")
    monkeypatch.setattr(environment, "checkout_root", lambda: checkout)
    (tmp_path / prefix).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / prefix))
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))
    monkeypatch.delenv("UV_PROJECT_ENVIRONMENT", raising=False)
    if setting == "envs/dev":
        monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "envs/dev")
    elif setting and setting.endswith("checkout-environment"):  # Docker images set one environment for all projects.
        monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(tmp_path / prefix))
    (tmp_path / "elsewhere").mkdir()
    elsewhere = setting == "uv-run-from-elsewhere" or (setting or "").startswith("explicit-project")
    monkeypatch.chdir(tmp_path / "elsewhere" if elsewhere else root / "src" / "nested")
    project = str(root) if (setting or "").startswith("explicit-project") else None

    if expected == "error":
        with pytest.raises(RegistryError, match="checkout|uv run --project"):
            detect_target(mode, project)
        return
    target = detect_target(mode, project)
    assert target.mode == ("project" if expected == "project" else "env")
    assert target.project_root == (root if expected == "project" else None)
    assert target.uvx == (expected == "uvx")
    assert target.isaaclab_version == "3.0.0"


def test_another_interpreter_reports_the_same_isaaclab_release():
    """``--python`` inspects an interpreter with a copy of the release detection, which must agree with it."""
    target = environment._foreign_target(sys.executable)

    assert target.isaaclab_version == environment.isaaclab_release_version()
    assert Path(target.prefix) == Path(sys.prefix)


@pytest.mark.parametrize(
    ("source", "release", "expected"),
    [
        (
            {**_GIT_SOURCE, "subdirectory": "packages/walker"},
            _release("0.1.0", _commit("a")),
            (f"walker @ git+https://github.com/alice/walker@{_commit('a')}#subdirectory=packages/walker", None),
        ),
        (
            _PYPI_SOURCE,
            _release("1.0", wheel={"filename": "walker-1.0-py3-none-any.whl", "sha256": "f" * 64, "url": _WHEEL_URL}),
            (f"walker @ {_WHEEL_URL}", "f" * 64),
        ),
        (
            _PYPI_SOURCE,
            _release("1.0", wheel={"filename": "walker-1.0-py3-none-any.whl", "sha256": "", "url": _WHEEL_URL}),
            "no valid wheel sha256",
        ),
        (
            _PYPI_SOURCE,
            _release("1.0", wheel={"filename": "other-1.0-py3-none-any.whl", "sha256": "f" * 64, "url": _WHEEL_URL}),
            "malformed wheel URL",
        ),
        (_GIT_SOURCE, _release("0.1.0", "v0.1.0"), "40-hex commit"),
        ({**_GIT_SOURCE, "subdirectory": "../escape"}, _release("0.1.0", _commit("a")), "subdirectory"),
        ({**_GIT_SOURCE, "repository": "https://x.org/a b"}, _release("0.1.0", _commit("a")), "repository"),
    ],
    ids=["git-subdirectory", "pypi-wheel", "pypi-no-hash", "pypi-other-wheel", "tag-not-commit", "escape", "bad-url"],
)
def test_requirement_is_built_from_validated_index_fields(source, release, expected):
    """Requirements pin git releases by commit and PyPI releases by wheel URL and sha256, and refuse bad fields."""
    package = parse_index({"schema": 1, "packages": [{"name": "walker", "source": source, "releases": [release]}]})
    package = package.get("walker")
    if isinstance(expected, str):
        with pytest.raises(RegistryError, match=expected):
            plan_for_release(package, package.releases[0])
        return
    plan = plan_for_release(package, package.releases[0])
    assert (plan.requirement, plan.sha256) == expected


def _recording_runner(calls: list, on_call: Callable[[list[str]], tuple[int, str]] | None = None) -> Callable:
    """Return a ``run_uv`` stand-in that records each call and the override/constraint files it saw.

    Args:
        calls: List that receives ``(argv, cwd, files)`` for each call.
        on_call: Simulates uv: returns the exit code and output of a call.
    """

    def run_uv(argv, *, cwd=None, capture=False):
        flags = ("--overrides", "-c", "-r")
        files = {flag: Path(argv[argv.index(flag) + 1]).read_text() for flag in flags if flag in argv}
        calls.append((argv, cwd, files))
        returncode, output = on_call(argv) if on_call else (0, "")
        return subprocess.CompletedProcess(argv, returncode, stdout=output, stderr=output)

    return run_uv


def _env_target(tmp_path: Path) -> tuple[Path, Target]:
    """Return a site directory with Isaac Lab and one library installed, and an env-mode target over it."""
    site = tmp_path / "site"
    _write_distribution(
        site,
        "isaaclab",
        "3.1.0",
        tasks=False,
        extras=("rsl-rl", "skrl"),
        requires=('iniconfig ; extra == "rsl-rl"', 'skrl>=1 ; extra == "skrl"'),
    )
    _write_distribution(site, "isaaclab-rl", "1.0.0", tasks=False)
    _write_distribution(site, "iniconfig", "2.0.0", tasks=False)
    prefix = tmp_path / "venv"
    prefix.mkdir()
    return site, Target(mode="env", python="/venv/bin/python", prefix=prefix, site_paths=(str(site),))


def _git_origin(commit: str) -> dict:
    """Return the PEP 610 ``direct_url.json`` of a git install of the test repository."""
    return {"url": "https://github.com/alice/walker", "vcs_info": {"vcs": "git", "commit_id": commit}}


def test_environment_install_pins_peers_and_constrains_everything_else(tmp_path, monkeypatch):
    """Env-mode installs override Isaac Lab with its active extras, constrain the rest, and record pinned commits."""
    site, target = _env_target(tmp_path)

    reacher_url = "http://127.0.0.1:8765/files/reacher-1.0.0-py3-none-any.whl"

    def uv(argv):
        if argv[1:3] == ["pip", "install"]:
            _write_distribution(site, "walker", "0.2.0", direct_url=_git_origin(_commit("b")))
            _write_distribution(site, "reacher", "1.0.0", direct_url={"url": reacher_url, "archive_info": {}})
        return 0, ""

    calls = []
    monkeypatch.setattr(installer, "run_uv", _recording_runner(calls, uv))
    monkeypatch.setenv("UV", "uv")
    git_plan = InstallPlan("walker", "walker @ git+https://github.com/alice/walker@main")
    pypi_plan = InstallPlan("reacher", f"reacher @ {reacher_url}", "1.0.0", sha256="f" * 64)

    install_into_environment([git_plan, pypi_plan], target)

    argv, cwd, files = next(call for call in calls if call[0][1:3] == ["pip", "install"])
    # No --index: the author's index never takes part in the resolution, and uv verifies the wheel's sha256.
    assert argv == [
        "uv", "pip", "install", "--python", "/venv/bin/python",
        "--overrides", mock.ANY, "-c", mock.ANY,
        "--no-sources-package", "isaaclab", "--no-sources-package", "isaaclab-dev",
        "--refresh-package", "reacher", "-r", mock.ANY, git_plan.requirement,
    ]  # fmt: skip
    assert files == {
        "--overrides": "isaaclab[rsl-rl]==3.1.0\nisaaclab-rl==1.0.0\n",
        "-c": "iniconfig==2.0.0\n",
        "-r": f"reacher @ {reacher_url} --hash=sha256:{'f' * 64}\n",
    }
    assert cwd.name.startswith("isaaclab-registry-")
    assert [call[0] for call in calls].count(["uv", "pip", "check", "--python", "/venv/bin/python"]) == 2
    state = load_state(target.prefix)
    # 'restore' reinstalls the commit that was installed, not the current head of the requested branch.
    assert state["walker"]["requirement"] == f"walker @ git+https://github.com/alice/walker@{_commit('b')}"
    assert state["walker"]["requested"] == git_plan.requirement
    assert state["reacher"] | {"installed_at": None} == {
        "requirement": f"reacher @ {reacher_url}",
        "version": "1.0.0",
        "sha256": "f" * 64,
        "installed_at": None,
    }


def test_environment_install_rolls_back_new_conflicts(tmp_path, monkeypatch):
    """An install whose requirements end up violated (checkout overrides) is undone, and nothing is recorded."""
    site, target = _env_target(tmp_path)
    _write_distribution(site, "walker", "0.1.0", direct_url=_git_origin(_commit("a")))
    installed = []

    def uv(argv):
        if argv[1:3] == ["pip", "install"] and "--no-deps" not in argv:
            shutil.rmtree(site / "walker-0.1.0.dist-info")
            _write_distribution(site, "walker", "0.2.0", direct_url=_git_origin(_commit("b")))
            _write_distribution(site, "old-numpy-user", "1.0.0", tasks=False)
            installed.append(True)
        if argv[1:3] == ["pip", "check"] and installed:
            return 1, (
                "The package `walker` requires `isaaclab~=3.0.0`, but `3.1.0` is installed\n"
                "The package `walker` requires `numpy<2`, but `2.5.1` is installed\n"
            )
        return 0, ""

    calls = []
    monkeypatch.setattr(installer, "run_uv", _recording_runner(calls, uv))
    monkeypatch.setenv("UV", "uv")
    plan = InstallPlan("walker", f"walker @ git+https://github.com/alice/walker@{_commit('b')}", "0.2.0")

    with pytest.raises(RegistryError, match="numpy<2") as error:
        install_into_environment([plan], target)

    assert "isaaclab~=3.0.0" not in str(error.value)
    python = ["--python", "/venv/bin/python"]
    assert calls[-2][0] == ["uv", "pip", "uninstall", *python, "old-numpy-user", "walker"]
    assert calls[-1][0] == [
        "uv", "pip", "install", *python, "--no-deps", f"walker @ git+https://github.com/alice/walker@{_commit('a')}"
    ]  # fmt: skip
    assert load_state(target.prefix) == {}


def test_project_install_verifies_the_lock_before_installing(tmp_path, monkeypatch, capsys):
    """Project mode locks first, checks the registered wheel hash, then syncs; any failure restores both files."""
    root = tmp_path / "project"
    root.mkdir()
    pyproject, lock = root / "pyproject.toml", root / "uv.lock"
    original = '[project]\nname = "my-lab"\nversion = "0.1.0"\n\n[dependency-groups]\nisaaclab-registry = []\n'
    pyproject.write_text(original)
    lock.write_text("version = 1\n")
    target = Target(mode="project", python="/project/.venv/bin/python", prefix=root / ".venv", project_root=root)
    reacher_url = "http://127.0.0.1:8765/files/reacher-1.0.0-py3-none-any.whl"
    served = {"hash": "sha256:" + "f" * 64}

    def uv(argv):
        """Lock what the URL serves, the way uv does; let the git package fail to resolve."""
        if argv[1] == "add" and argv[-1].startswith("walker"):
            return 1, "No solution found"
        if argv[1] == "add":
            pyproject.write_text(original + f'\n[tool.uv.sources]\nreacher = {{ url = "{reacher_url}" }}\n')
            wheel = f'{{ url = "{reacher_url}", hash = "{served["hash"]}" }}'
            lock.write_text(f'version = 1\n\n[[package]]\nname = "reacher"\nversion = "1.0.0"\nwheels = [{wheel}]\n')
        return 0, ""

    calls = []
    monkeypatch.setattr(installer, "run_uv", _recording_runner(calls, uv))
    monkeypatch.setenv("UV", "uv")
    reacher = InstallPlan("reacher", f"reacher @ {reacher_url}", "1.0.0", sha256="f" * 64)
    walker = InstallPlan("walker", f"walker @ git+https://github.com/alice/walker@{_commit('a')}")
    add = ["uv", "add", "--no-sync", "--project", str(root), "--group", "isaaclab-registry"]
    sync = ["uv", "sync", "--frozen", "--inexact", "--project", str(root), "--group", "isaaclab-registry"]

    with pytest.raises(RegistryError, match="back to their previous content"):
        install_into_project([reacher, walker], target)
    refresh = ["--refresh-package", "reacher"]  # check the file served now, not a copy uv cached
    assert [call[0] for call in calls] == [add + refresh + [reacher.requirement], add + [walker.requirement]]
    assert (pyproject.read_text(), lock.read_text()) == (original, "version = 1\n")

    served["hash"] = "sha256:" + "0" * 64  # the wheel was replaced on the author's index after registration
    with pytest.raises(RegistryError, match="not the wheel that was registered"):
        install_into_project([reacher], target)
    assert sync not in [call[0] for call in calls]
    assert (pyproject.read_text(), lock.read_text()) == (original, "version = 1\n")

    served["hash"] = "sha256:" + "f" * 64
    calls.clear()
    install_into_project([reacher], target)
    assert [call[0] for call in calls] == [add + refresh + [reacher.requirement], sync]
    assert 'default-groups = ["dev", "isaaclab-registry"]' in capsys.readouterr().err

    # Uninstalling never re-syncs the environment exactly: only the package itself leaves it.
    calls.clear()
    uninstall_from_project("reacher", target)
    assert [call[0] for call in calls] == [
        ["uv", "remove", "--no-sync", "--project", str(root), "--group", "isaaclab-registry", "reacher"],
        ["uv", "pip", "uninstall", "--python", "/project/.venv/bin/python", "reacher"],
    ]


def test_installed_status_joins_distributions_state_and_index(tmp_path, registry_index):
    """Each installed task package is classified against the index and the registry's own records."""
    site = tmp_path / "site"
    prefix = tmp_path / "venv"
    prefix.mkdir()
    git = {"url": "https://github.com/alice/walker", "vcs_info": {"vcs": "git", "commit_id": _commit("a")}}
    _write_distribution(site, "walker", "0.1.0", direct_url=git)
    _write_distribution(site, "reacher", "1.0.0")
    _write_distribution(site, "retired", "1.0.0", direct_url={**git, "vcs_info": {"commit_id": _commit("9")}})
    _write_distribution(site, "my-lab", "0.1.0", direct_url={"url": "file:///home/me/my_lab", "dir_info": {}})
    _write_distribution(site, "other-tasks", "1.0.0")
    _write_distribution(site, "direct-pkg", "0.1.0", direct_url=git)
    _write_distribution(site, "plain-library", "1.0.0", tasks=False)
    _write_distribution(site, "evil", "1.0.0")
    record_install(prefix, "direct-pkg", {"requirement": "direct-pkg @ git+https://x.org/d@main"})
    record_install(prefix, "gone", {"requirement": "gone==1.0", "version": "1.0"})
    record_install(prefix, "gone-bad", {"requirement": "gone-bad==2.0", "version": "2.0"})
    target = Target(mode="env", python="python", prefix=prefix, site_paths=(str(site),))

    rows = {row.name: row for row in installed_packages(target, registry_index, "3.0.0")}

    assert {name: row.status for name, row in rows.items()} == {
        "walker": "outdated (0.2.0 available)",
        "reacher": "registry",
        "retired": "drift",
        "my-lab": "local",
        "other-tasks": "unmanaged",
        "direct-pkg": "direct",
        "evil": "blocked",
        "gone": "missing",
        "gone-bad": "blocked",
    }
    assert rows["gone"].note == "run 'isaaclab registry restore'"
    assert "malware" in rows["gone-bad"].note and "registry uninstall gone-bad" in rows["gone-bad"].note
    assert rows["walker"].commit == _commit("a")
    offline = {row.name: row.status for row in installed_packages(target, None, "3.0.0")}
    assert (offline["direct-pkg"], offline["walker"], offline["my-lab"]) == ("unknown", "unmanaged", "local")


def _git(path: Path, *args: str) -> str:
    """Run git with a fixed identity and return its output."""
    identity = ["-c", "user.name=Test", "-c", "user.email=test@example.org", "-c", "commit.gpgsign=false"]
    command = ["git", *identity, "-c", "tag.gpgsign=false", "-C", str(path), *args]
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()


def test_publish_writes_a_request_after_preflight_checks(tmp_path, capsys):
    """Publishing refuses an untagged release, then writes a request the registry bot can parse."""
    project = tmp_path / "dummy_walk"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        "[project]\n"
        'name = "Dummy_Walk"\nversion = "0.1.0"\n'
        'dependencies = ["isaaclab[rsl-rl]~=3.0.0"]\n'
        '[project.entry-points."isaaclab.tasks"]\ndummy_walk = "dummy_walk.tasks"\n'
        "[tool.isaaclab.registry]\n"
        'category = "locomotion"\ntags = ["anymal"]\nmaintainers = ["alice"]\n'
        'media = { image = "https://example.org/walk.gif" }\ndescription = "Walks \\"fast\\"."\n'
    )
    (project / "LICENSE").write_text("BSD-3-Clause\n")
    _git(project, "init", "-q")
    _git(project, "add", "-A")
    _git(project, "commit", "-qm", "Initial commit")
    output = tmp_path / "request.toml"

    assert command_registry(["publish", str(project), "--output", str(output)]) == 1
    assert "FAIL  tag" in capsys.readouterr().err
    assert not output.exists()

    _git(project, "tag", "v0.1.0")
    assert command_registry(["publish", str(project), "--output", str(output)]) == 0
    assert tomllib.loads(output.read_text()) == {
        "schema": 1,
        "name": "dummy-walk",
        "requested_by": "alice",
        "source": {
            "type": "git",
            "repository": project.resolve().as_uri(),
            "subdirectory": "",
            "ref": "v0.1.0",
            "commit": _git(project, "rev-parse", "HEAD"),
        },
        "package": {
            "category": "locomotion",
            "tags": ["anymal"],
            "maintainers": ["alice"],
            "homepage": "",
            "media": {"image": "https://example.org/walk.gif"},
            "description": 'Walks "fast".',
        },
    }


@pytest.mark.skipif(
    not (os.environ.get("UV") or shutil.which("uv")) or not shutil.which("git"), reason="needs uv and git"
)
def test_registry_round_trip_with_uv(tmp_path, capsys):
    """A git release installs into another venv, reports as a registry install, and uninstalls cleanly.

    The venv has its own Isaac Lab release (4.0.0), which the compatibility gate must use instead of the
    running one. The guards around it run against real uv too: neither Isaac Lab nor a library the registry did
    not install is replaced or removed, and 'restore' never brings back a package blocklisted since.
    """
    uv = os.environ.get("UV") or shutil.which("uv")
    repository = tmp_path / "rt_tasks"
    (repository / "rt_tasks").mkdir(parents=True)
    (repository / "rt_tasks" / "__init__.py").write_text("")
    (repository / "pyproject.toml").write_text(
        '[build-system]\nrequires = []\nbuild-backend = "backend"\nbackend-path = ["."]\n'
        '[project]\nname = "rt-tasks"\nversion = "0.1.0"\n'
    )
    (repository / "backend.py").write_text(_IN_TREE_BACKEND)
    _git(repository, "init", "-q")
    _git(repository, "add", "-A")
    _git(repository, "commit", "-qm", "Initial commit")
    commit = _git(repository, "rev-parse", "HEAD")
    packages = [
        {
            "name": name,
            "source": {"type": "git", "repository": repository.as_uri(), "subdirectory": ""},
            "releases": [_release("0.1.0", commit, isaaclab=">=4,<5")],
        }
        for name in ("rt-tasks", "plain-library")
    ]
    block = {"name": "rt-tasks", "commit": commit, "version": None, "reason": "malware"}
    indexes = {
        "good": {"schema": 1, "name": "round-trip", "packages": packages},
        "blocked": {"schema": 1, "name": "round-trip", "packages": packages, "blocklist": [block]},
        "empty": {"schema": 1, "name": "round-trip", "packages": []},
    }
    for name, document in indexes.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(document))
    subprocess.run([uv, "venv", "-q", str(tmp_path / "venv"), "--python", sys.executable], check=True)
    python = str(tmp_path / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    site = Path(
        subprocess.check_output(
            [python, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True
        ).strip()
    )
    _write_distribution(site, "isaaclab", "4.0.0", tasks=False)
    _write_distribution(site, "plain-library", "1.0.0", tasks=False)

    def registry(*args: str, index: str = "good") -> tuple[int, str, str]:
        code = command_registry(["--index", str(tmp_path / f"{index}.json"), *args, "--python", python])
        output = capsys.readouterr()
        return code, output.out, output.err

    def installed(index: str = "good") -> dict[str, tuple[str, str | None]]:
        code, out, _ = registry("installed", "--json", index=index)
        assert code == 0
        return {row["name"]: (row["status"], row["commit"]) for row in json.loads(out)["packages"]}

    # A registry entry named like an installed library that the registry did not install must not replace it.
    code, _, err = registry("install", "plain-library", "--yes")
    assert code == 1 and "Refusing to install plain-library" in err
    assert registry("install", "rt-tasks", "--yes")[0] == 0
    assert installed() == {"rt-tasks": ("registry", commit)}
    for protected in ("isaaclab", "plain-library"):
        code, _, err = registry("uninstall", protected, "--yes")
        assert code == 1 and f"Refusing to uninstall {protected}" in err
    kept = {"isaaclab-4.0.0.dist-info", "plain_library-1.0.0.dist-info"}
    assert kept <= {path.name for path in site.glob("*.dist-info")}

    # 'uv sync' removed it, and the registry blocklisted it since: restore refuses it.
    subprocess.run([uv, "pip", "uninstall", "-q", "--python", python, "rt-tasks"], check=True)
    code, _, err = registry("restore", "--yes", index="blocked")
    assert code == 1 and "Not restoring rt-tasks" in err
    assert installed("blocked") == {"rt-tasks": ("blocked", commit)}
    assert not list(site.glob("rt_tasks-*.dist-info"))
    assert registry("restore", "--yes")[0] == 0
    assert installed() == {"rt-tasks": ("registry", commit)}
    assert registry("uninstall", "rt-tasks", "--yes")[0] == 0

    # A direct spec installs without the index.
    assert registry("install", f"rt-tasks @ git+{repository.as_uri()}@{commit}", "--yes", index="empty")[0] == 0
    assert installed("empty") == {"rt-tasks": ("direct", commit)}
    assert registry("uninstall", "rt-tasks", "--yes")[0] == 0
    assert installed() == {}
    assert not (tmp_path / "venv" / "isaaclab-registry.json").exists()


# A PEP 517 backend with no requirements, so the round trip builds offline with any uv version.
_IN_TREE_BACKEND = """
import base64
import hashlib
import pathlib
import zipfile


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    dist_info = "rt_tasks-0.1.0.dist-info"
    files = {
        "rt_tasks/__init__.py": b"",
        f"{dist_info}/METADATA": b"Metadata-Version: 2.1\\nName: rt-tasks\\nVersion: 0.1.0\\n",
        f"{dist_info}/WHEEL": b"Wheel-Version: 1.0\\nGenerator: test\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n",
        f"{dist_info}/entry_points.txt": b"[isaaclab.tasks]\\nrt_tasks = rt_tasks\\n",
    }
    records = []
    filename = "rt_tasks-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(pathlib.Path(wheel_directory) / filename, "w") as wheel:
        for path, data in files.items():
            wheel.writestr(path, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            records.append(f"{path},sha256={digest},{len(data)}")
        wheel.writestr(f"{dist_info}/RECORD", "\\n".join([*records, f"{dist_info}/RECORD,,"]) + "\\n")
    return filename
"""
