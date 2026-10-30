# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ``isaaclab registry`` client."""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import tomllib

from isaaclab.cli.commands import registry as registry_command
from isaaclab.cli.commands.registry import command_registry
from isaaclab.cli.registry import RegistryError, install
from isaaclab.cli.registry.index import load_index, parse_index, select_release
from isaaclab.cli.registry.install import InstallPlan, Target, installed_packages
from isaaclab.cli.registry.publish import draft_request

pytestmark = pytest.mark.unit

needs_uv = pytest.mark.skipif(not shutil.which("uv") or not shutil.which("git"), reason="needs uv and git")

_GIT_SOURCE = {"type": "git", "repository": "https://github.com/alice/walker", "subdirectory": ""}
_PYPI_SOURCE = {"type": "pypi", "index": "https://pypi.org/simple", "project": "walker"}
_WHEEL_URL = "https://files.example.org/walker-1.0-py3-none-any.whl"


def _commit(character: str) -> str:
    """Return a fake 40-hex commit."""
    return character * 40


def _release(version: str, commit: str | None = None, isaaclab: str = ">=3.0.0,<3.1", **fields) -> dict:
    """Return an ``index.json`` release entry."""
    return {"version": version, "commit": commit, "isaaclab": isaaclab, **fields}


def _write_distribution(site: Path, name: str, version: str, *, tasks: bool = True, origin: dict | None = None):
    """Write a minimal installed ``.dist-info`` directory, with a PEP 610 ``direct_url.json`` when given."""
    module = name.replace("-", "_")
    dist_info = site / f"{module}-{version}.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    if tasks:
        (dist_info / "entry_points.txt").write_text(f"[isaaclab.tasks]\n{module} = {module}.tasks\n")
    if origin is not None:
        (dist_info / "direct_url.json").write_text(json.dumps(origin))


def _write_wheel(
    directory, name: str, requires: tuple[str, ...] = (), version: str = "0.1.0", *, tasks: bool = True
) -> Path:
    """Write the wheel of a task package without any build tool."""
    module = name.replace("-", "_")
    info = f"{module}-{version}.dist-info"
    metadata = [
        "Metadata-Version: 2.1",
        f"Name: {name}",
        f"Version: {version}",
        *(f"Requires-Dist: {r}" for r in requires),
    ]
    files = {
        f"{module}/__init__.py": b"",
        f"{info}/METADATA": "\n".join([*metadata, ""]).encode(),
        f"{info}/WHEEL": b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    if tasks:
        files[f"{info}/entry_points.txt"] = f"[isaaclab.tasks]\n{module} = {module}\n".encode()
    digests = {
        path: base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=") for path, data in files.items()
    }
    records = [f"{path},sha256={digests[path].decode()},{len(data)}" for path, data in files.items()]
    files[f"{info}/RECORD"] = "\n".join([*records, f"{info}/RECORD,,", ""]).encode()
    wheel = Path(directory) / f"{module}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for path, data in files.items():
            archive.writestr(path, data)
    return wheel


# In-tree PEP 517 backend of the test repository: it has no build requirements, so uv builds it offline.
_BACKEND = f"""\
import base64, hashlib, zipfile
from pathlib import Path

{inspect.getsource(_write_wheel)}

def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return _write_wheel(wheel_directory, "rt-tasks", ("isaaclab~=3.0.0",)).name
"""


def _git(path: Path, *args: str) -> str:
    """Run git with a fixed identity and return its output."""
    identity = ["user.name=Test", "user.email=test@example.org", "commit.gpgsign=false", "tag.gpgsign=false"]
    command = ["git", *(argument for option in identity for argument in ("-c", option)), "-C", str(path), *args]
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def registry_index():
    """An index whose releases are deliberately out of order."""
    walker = [
        _release("0.1.0", _commit("a")),
        _release("0.4.0", _commit("d"), isaaclab=">=4,<5"),
        _release("0.2.0", _commit("b")),
        _release("0.2.1", _commit("c")),
        _release("0.2.2", _commit("2"), requires_python=">=3.99"),
        _release("0.3.0", _commit("e"), yanked="broken build"),
    ]
    reacher = {"filename": "reacher-1.0.0-py3-none-any.whl", "url": _WHEEL_URL, "sha256": "f" * 64}
    return parse_index(
        {
            "schema": 1,
            "name": "test-registry",
            "blocklist": [
                {"name": "walker", "commit": _commit("c"), "reason": "leaks credentials"},
                {"name": "evil", "reason": "malware"},
            ],
            "packages": [
                {"name": "walker", "source": _GIT_SOURCE, "releases": walker},
                {"name": "runner", "source": _GIT_SOURCE, "releases": [_release("1.0.0", _commit("1"))]},
                {"name": "reacher", "source": _PYPI_SOURCE, "releases": [_release("1.0.0", wheel=reacher)]},
            ],
        }
    )


@pytest.mark.parametrize(
    ("name", "version", "force", "expected", "message"),
    [
        ("Walker", None, False, "0.2.0", None),
        ("walker", "0.3.0", False, "0.3.0", "yanked: broken build"),
        ("walker", "0.4.0", False, None, "needs isaaclab >=4,<5 (have 3.0.0)"),
        ("walker", "0.4.0", True, "0.4.0", "--force"),
        ("walker", "0.2.2", False, None, "needs Python >=3.99"),
        ("walker", "0.2.1", False, None, "blocklisted by the registry: leaks credentials"),
        ("evil", None, False, None, "blocklisted by the registry: malware"),
    ],
    ids=["newest-compatible", "pinned-yanked", "incompatible", "forced", "python", "blocked-commit", "blocked-package"],
)
def test_release_selection_policy(registry_index, name, version, force, expected, message):
    """Install picks the newest live, unblocked, compatible release and refuses or warns on everything else."""
    if expected is None:
        with pytest.raises(RegistryError, match=re.escape(message)):
            select_release(registry_index, name, "3.0.0", version, force)
        return
    _, release, warnings = select_release(registry_index, name, "3.0.0", version, force)
    assert release.version == expected
    assert any(message in warning for warning in warnings) if message else not warnings


@pytest.mark.parametrize(
    ("name", "venv", "cwd", "checkout", "shared_environment", "expected"),
    [
        ("my-lab", "project/.venv", "project/src", "IsaacLab", False, "project"),
        ("my-lab", "project/.venv", ".", "IsaacLab", False, "project"),
        ("my-lab", "other-venv", "project/src", "IsaacLab", False, "env"),
        ("isaaclab-dev", "project/.venv", "project/src", "IsaacLab", False, "env"),
        ("isaaclab-tasks", "project/.venv", "project/src", "IsaacLab", False, "env"),
        ("my-lab", "project/.venv", "project/src", ".", False, "env"),
        ("my-lab", "IsaacLab/.venv", "project/src", "IsaacLab", True, "env"),
    ],
    ids=["project-env", "uv-run-project", "other-env", "checkout", "checkout-package", "inside-checkout", "docker"],
)
def test_target_detection(tmp_path, monkeypatch, name, venv, cwd, checkout, shared_environment, expected):
    """Project mode applies only to the user's own project whose environment runs the command, never the checkout.

    Docker images set one absolute ``UV_PROJECT_ENVIRONMENT`` for every project: when that is the checkout's
    environment, packages go into it with uv running from the checkout.
    """
    (tmp_path / "project" / "src").mkdir(parents=True)
    (tmp_path / "project" / "pyproject.toml").write_text(f'[project]\nname = "{name}"\nversion = "0.1.0"\n')
    (tmp_path / checkout).mkdir(exist_ok=True)
    (tmp_path / checkout / "VERSION").write_text("3.0.0\n")
    (tmp_path / venv).mkdir(parents=True)
    monkeypatch.setattr(install, "checkout_root", lambda: tmp_path / checkout)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / venv))
    monkeypatch.chdir(tmp_path / cwd)
    monkeypatch.delenv("UV_PROJECT_ENVIRONMENT", raising=False)
    if shared_environment:
        monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(tmp_path / venv))

    target = install.detect_target()

    assert (target.mode, target.project_root) == (expected, tmp_path / "project" if expected == "project" else None)
    assert target.uv_cwd == (tmp_path / checkout if shared_environment else None)
    assert target.isaaclab_version == "3.0.0"


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
            _release("1.0", wheel={"url": _WHEEL_URL, "sha256": "f" * 64}),
            (f"walker @ {_WHEEL_URL}", "f" * 64),
        ),
        (_PYPI_SOURCE, _release("1.0", wheel={"url": _WHEEL_URL, "sha256": ""}), "no valid wheel sha256"),
        (_PYPI_SOURCE, _release("1.0", wheel={"url": "ftp://x.org/w.whl", "sha256": "f" * 64}), "malformed wheel URL"),
        (_GIT_SOURCE, _release("0.1.0", "v0.1.0"), "40-hex commit"),
        ({**_GIT_SOURCE, "subdirectory": "../escape"}, _release("0.1.0", _commit("a")), "subdirectory"),
        ({**_GIT_SOURCE, "repository": "https://x.org/a b"}, _release("0.1.0", _commit("a")), "repository"),
        ({**_GIT_SOURCE, "repository": "https://x.org/a@b"}, _release("0.1.0", _commit("a")), "repository"),
    ],
    ids=[
        "git-subdirectory",
        "pypi-wheel",
        "pypi-no-hash",
        "pypi-bad-scheme",
        "tag-not-commit",
        "escape",
        "bad-url",
        "at",
    ],
)
def test_requirement_is_built_from_validated_index_fields(source, release, expected):
    """Requirements pin git releases by commit and PyPI releases by wheel URL and sha256, and refuse bad fields."""
    package = parse_index({"schema": 1, "packages": [{"name": "walker", "source": source, "releases": [release]}]})
    package = package.get("walker")
    if isinstance(expected, str):
        with pytest.raises(RegistryError, match=expected):
            install.plan_for_release(package, package.releases[0])
        return
    plan = install.plan_for_release(package, package.releases[0])
    assert (plan.requirement, plan.sha256) == expected


def test_installed_status(tmp_path, registry_index):
    """Installed task packages are classified against the index; libraries without tasks are left out."""
    site = tmp_path / "site"
    git = {"url": "https://github.com/alice/walker", "vcs_info": {"vcs": "git", "commit_id": _commit("a")}}
    _write_distribution(site, "walker", "0.1.0", origin=git)
    _write_distribution(site, "runner", "1.0.0", origin={**git, "vcs_info": {"vcs": "git", "commit_id": _commit("9")}})
    _write_distribution(site, "reacher", "1.0.0", origin={"url": _WHEEL_URL, "archive_info": {}})
    _write_distribution(site, "evil", "1.0.0")
    _write_distribution(site, "my-lab", "0.1.0", origin={"url": "file:///home/me/my_lab", "dir_info": {}})
    _write_distribution(site, "other-tasks", "1.0.0")
    _write_distribution(site, "plain-library", "1.0.0", tasks=False)
    target = Target("env", "python", tmp_path, isaaclab_version="3.0.0", site_paths=(str(site),))

    def statuses(index):
        return {row.name: row.status for row in installed_packages(target, index)}

    assert statuses(registry_index) == {
        "walker": "outdated (0.2.0 available)",
        "runner": "drift",
        "reacher": "registry",
        "evil": "blocked",
        "my-lab": "local",
        "other-tasks": "unmanaged",
    }
    offline = statuses(None)  # without the index, only local installs keep their status
    assert offline.pop("my-lab") == "local" and set(offline.values()) == {"unmanaged"}


def test_isaac_lab_local_installs_and_libraries_are_never_managed(tmp_path, monkeypatch, capsys):
    """Install and uninstall refuse Isaac Lab, a local install such as the user's project, and a library."""
    site = tmp_path / "site"
    _write_distribution(site, "my-lab", "0.1.0", origin={"url": "file:///home/me/my_lab", "dir_info": {}})
    _write_distribution(site, "plain-library", "1.0.0", tasks=False)
    target = Target("env", sys.executable, tmp_path, site_paths=(str(site),))
    monkeypatch.setattr(registry_command, "detect_target", lambda: target)
    refusals = {
        "isaaclab": "part of Isaac Lab",
        "isaaclab-tasks": "part of Isaac Lab",
        "my-lab": "is a local install",
        "plain-library": "provides no Isaac Lab tasks",
    }
    for verb in ("install", "uninstall"):
        for name, reason in refusals.items():
            assert command_registry(["--index", str(tmp_path / "no-index.json"), verb, name, "--yes"]) == 1
            assert reason in capsys.readouterr().err


@pytest.mark.skipif(not shutil.which("git"), reason="needs git")
def test_publish_checks_the_project_and_writes_the_request(tmp_path):
    """Publishing refuses what the registry bot would reject, then writes a request the bot can parse."""
    project, remote = tmp_path / "dummy_walk", tmp_path / "dummy_walk.git"
    _git(tmp_path, "init", "-q", "--bare", str(remote))
    _git(tmp_path, "clone", "-q", remote.as_uri(), str(project))

    def write_pyproject(isaaclab: str, tasks: bool = True) -> None:
        entry_point = '[project.entry-points."isaaclab.tasks"]\ndummy_walk = "dummy_walk.tasks"\n' if tasks else ""
        (project / "pyproject.toml").write_text(
            f'[project]\nname = "Dummy_Walk"\nversion = "0.1.0"\ndependencies = ["{isaaclab}"]\n{entry_point}'
            '[tool.isaaclab.registry]\ncategory = "locomotion"\ntags = ["anymal"]\nmaintainers = ["alice"]\n'
            'media = { image = "https://example.org/walk.gif" }\ndescription = "Walks \\"fast\\"."\n'
        )

    for isaaclab, tasks, problem in [
        ("isaaclab~=3.0.0", False, "isaaclab.tasks"),
        ("isaaclab[rsl-rl]>=3.0", True, "needs an upper bound"),
        ("isaaclab-dev", True, "requires 'isaaclab-dev'"),
    ]:
        write_pyproject(isaaclab, tasks)
        with pytest.raises(RegistryError, match=problem):
            draft_request(project)
    write_pyproject("isaaclab[rsl-rl]~=3.0.0")
    _git(project, "add", "-A")
    _git(project, "commit", "-qm", "Add the project")
    with pytest.raises(RegistryError, match="no tag v0.1.0"):
        draft_request(project)
    _git(project, "tag", "v0.1.0")
    with pytest.raises(RegistryError, match="push it"):
        draft_request(project)
    _git(project, "push", "-q", "origin", "HEAD:refs/heads/main", "v0.1.0")

    output = tmp_path / "request.toml"
    assert command_registry(["publish", str(project), "--output", str(output)]) == 0
    assert tomllib.loads(output.read_text()) == {
        "schema": 1,
        "name": "dummy-walk",
        "requested_by": "alice",
        "source": {
            "type": "git",
            "repository": remote.as_uri(),
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


@needs_uv
def test_environment_install_keeps_isaac_lab_and_checks_the_wheel(tmp_path, monkeypatch, capfd):
    """A release that needs another Isaac Lab, and carries an ``isaaclab`` path source, installs offline next to
    the installed Isaac Lab without replacing it; a wheel that is not the registered one is refused."""
    monkeypatch.setenv("UV_NO_INDEX", "1")  # nothing may come from an index
    uv = install.uv_binary()
    repository = tmp_path / "rt_tasks"
    repository.mkdir()
    (repository / "backend.py").write_text(_BACKEND)
    (repository / "pyproject.toml").write_text(
        '[build-system]\nrequires = []\nbuild-backend = "backend"\nbackend-path = ["."]\n'
        '[project]\nname = "rt-tasks"\nversion = "0.1.0"\ndependencies = ["isaaclab~=3.0.0"]\n'
        '[tool.uv.sources]\nisaaclab = { path = "../missing-checkout/source/isaaclab" }\n'
    )
    for command in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "Initial commit"]):
        _git(repository, *command)
    subprocess.run([uv, "venv", "-q", str(tmp_path / "venv"), "--python", sys.executable], check=True)
    python = str(tmp_path / "venv" / "bin" / "python")
    purelib = "import sysconfig; print(sysconfig.get_path('purelib'))"
    site = Path(subprocess.run([python, "-c", purelib], capture_output=True, text=True, check=True).stdout.strip())
    _write_distribution(site, "isaaclab", "4.0.0", tasks=False)
    target = Target("env", python, tmp_path / "venv", isaaclab_version="4.0.0", site_paths=(str(site),))
    monkeypatch.setattr(registry_command, "detect_target", lambda: target)
    index = tmp_path / "index.json"

    def registry(source: dict, release: dict, *args: str) -> int:
        package = {"name": "rt-tasks", "source": source, "releases": [{**release, "isaaclab": ">=4,<5"}]}
        index.write_text(json.dumps({"schema": 1, "packages": [package]}))
        return command_registry(["--index", str(index), *args])

    helper = _write_wheel(tmp_path, "rt-helper", tasks=False)
    wheel = _write_wheel(tmp_path, "rt-tasks", ("isaaclab~=3.0.0", f"rt-helper @ {helper.as_uri()}"))
    pypi = {"type": "pypi", "index": "https://pypi.org/simple", "project": "rt-tasks"}
    sha256 = hashlib.sha256(wheel.read_bytes()).hexdigest()
    registered = {"filename": wheel.name, "url": wheel.as_uri(), "sha256": sha256}
    assert registry(pypi, _release("0.1.0", wheel={**registered, "sha256": "0" * 64}), "install", "rt-tasks", "-y") == 1
    assert "Hash mismatch" in capfd.readouterr().err
    monkeypatch.setenv("UV_OFFLINE", "1")  # uv rejects --refresh-package offline
    assert registry(pypi, _release("0.1.0", wheel=registered), "install", "rt-tasks", "-y") == 0
    assert "rt-helper" in install.installed_distributions(target.site_paths)
    monkeypatch.delenv("UV_OFFLINE")
    assert registry(pypi, _release("0.1.0", wheel={**registered, "sha256": "0" * 64}), "install", "rt-tasks", "-y") == 1
    assert "Hash mismatch" in capfd.readouterr().err
    assert command_registry(["uninstall", "rt-tasks", "--yes"]) == 0

    git = {"type": "git", "repository": repository.as_uri(), "subdirectory": ""}
    release = _release("0.1.0", _git(repository, "rev-parse", "HEAD"))
    assert registry(git, release, "install", "rt-tasks", "--dry-run") == 0
    assert not list(site.glob("rt_tasks-*.dist-info"))
    assert registry(git, release, "install", "rt-tasks", "--yes") == 0
    rows = installed_packages(target, load_index(str(index)))
    assert [(row.name, row.status) for row in rows] == [("rt-tasks", "registry")]
    assert {path.name for path in site.glob("*.dist-info")} == {
        "isaaclab-4.0.0.dist-info",
        "rt_tasks-0.1.0.dist-info",
        "rt_helper-0.1.0.dist-info",
    }


@needs_uv
@pytest.mark.parametrize("origin", ["index", "wheel"])
def test_environment_install_preserves_unrequested_dependencies(tmp_path, monkeypatch, origin):
    """Transitive source requirements cannot replace an installed dependency, including an index install."""
    monkeypatch.setenv("UV_NO_INDEX", "1")
    uv = install.uv_binary()
    original, replacement = tmp_path / "original", tmp_path / "replacement"
    original.mkdir()
    replacement.mkdir()
    original_wheel = _write_wheel(original, "protected-library", tasks=False)
    replacement_wheel = _write_wheel(replacement, "protected-library", tasks=False)
    dependency = _write_wheel(
        tmp_path, "new-dependency", (f"protected-library @ {replacement_wheel.as_uri()}",), tasks=False
    )
    task = _write_wheel(tmp_path, "rt-tasks", (f"new-dependency @ {dependency.as_uri()}",))
    prefix = tmp_path / "venv"
    subprocess.run([uv, "venv", "-q", str(prefix), "--python", sys.executable], check=True)
    python = str(prefix / "bin" / "python")
    requirement = ["--find-links", str(original), "protected-library"] if origin == "index" else [str(original_wheel)]
    subprocess.run([uv, "pip", "install", "-q", "--python", python, *requirement], check=True)
    site = subprocess.check_output(
        [python, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True
    ).strip()
    target = Target("env", python, prefix, site_paths=(site,))
    before = install.direct_url(install.installed_distributions(target.site_paths)["protected-library"])

    with pytest.raises(RegistryError, match="protected-library"):
        install.install_into_environment([InstallPlan("rt-tasks", f"rt-tasks @ {task.as_uri()}")], target)

    distributions = install.installed_distributions(target.site_paths)
    assert set(distributions) == {"protected-library"}
    assert install.direct_url(distributions["protected-library"]) == before


@needs_uv
def test_project_install_locks_only_the_registered_wheel(tmp_path, monkeypatch):
    """A wheel that is not the registered one, or a package that needs another Isaac Lab, never reaches the project
    environment. ``pyproject.toml`` and ``uv.lock`` come back byte-identical after a refused install and after an
    uninstall, and the sync keeps the packages that are not in the lock."""
    links = tmp_path / "links"
    links.mkdir()
    _write_wheel(links, "isaaclab", version="3.0.0")
    _write_wheel(links, "rt-shared")
    monkeypatch.setenv("UV_NO_INDEX", "1")
    monkeypatch.setenv("UV_FIND_LINKS", str(links))
    monkeypatch.setenv("UV_PYTHON", sys.executable)
    monkeypatch.delenv("UV_PROJECT_ENVIRONMENT", raising=False)
    uv, root = install.uv_binary(), tmp_path / "my_lab"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname = "my-lab"\nversion = "0.1.0"\nrequires-python = ">=3.10"\n'
        'dependencies = ["isaaclab~=3.0.0", "rt-shared"]\n[dependency-groups]\nisaaclab-registry = []\n'
    )
    subprocess.run([uv, "sync", "-q", "--project", str(root)], check=True)
    python = str(root / ".venv" / "bin" / "python")
    subprocess.run([uv, "pip", "install", "-q", "--python", python, _write_wheel(tmp_path, "rt-extra")], check=True)
    _write_wheel(links, "isaaclab", version="3.0.1")
    original = {path: path.read_bytes() for path in (root / "pyproject.toml", root / "uv.lock")}
    target = Target("project", python, root / ".venv", project_root=root)

    def installed(module: str) -> bool:
        return any(root.glob(f".venv/lib/*/site-packages/{module}-*.dist-info"))

    def plan(name: str, requires: tuple[str, ...] = (), sha256: str | None = None) -> InstallPlan:
        wheel = _write_wheel(tmp_path, name, requires)
        return InstallPlan(name, f"{name} @ {wheel.as_uri()}", sha256 or hashlib.sha256(wheel.read_bytes()).hexdigest())

    with pytest.raises(RegistryError, match="not the wheel that was registered"):
        install.install_into_project([plan("rt-tasks", sha256="0" * 64)], target)
    assert {path: path.read_bytes() for path in original} == original and not installed("rt_tasks")
    with pytest.raises(RegistryError, match=r"change Isaac Lab \(isaaclab 3\.0\.0 -> 3\.0\.1\)"):
        install.install_into_project([plan("rt-newer", ("isaaclab>=3.0.1",))], target)
    assert {path: path.read_bytes() for path in original} == original and not installed("rt_newer")
    install.install_into_project([plan("rt-tasks")], target)
    assert installed("rt_tasks") and installed("rt_extra")
    install.uninstall_from_project(["rt-tasks"], target)
    assert {path: path.read_bytes() for path in original} == original and not installed("rt_tasks")
    install.install_into_project([plan("rt-shared")], target)
    install.uninstall_from_project(["rt-shared"], target)
    assert installed("rt_shared"), "A dependency still required by the project must remain installed."
