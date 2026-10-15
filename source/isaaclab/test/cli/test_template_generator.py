# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the project template interactive prompts."""

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path
from unittest import mock

import pytest
import tomllib

_TEMPLATE_DIR = Path(__file__).parents[4] / "tools" / "template"
_SPEC = importlib.util.spec_from_file_location("isaaclab_template_cli", _TEMPLATE_DIR / "cli.py")
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.path.insert(0, str(_TEMPLATE_DIR))
try:
    _SPEC.loader.exec_module(_MODULE)
finally:
    sys.path.pop(0)

CLIHandler = _MODULE.CLIHandler
_GENERATOR = sys.modules["generator"]


def _external_specification(
    tmp_path: Path, include_ui_extension: bool = False, initial_content: str = "cartpole"
) -> dict:
    """Create a canonical external-project specification."""
    return {
        "external": True,
        "path": str(tmp_path),
        "name": "test_project",
        "authors": ["Test Author", "Example Organization"],
        "initial_content": initial_content,
        "isaaclab_version": "3.0.0",
        "isaaclab_optional_extras": ["all", "isaacsim", "ov", "ovphysx", "ovrtx", "rerun", "viser"],
        "task_name": "place_vial",
        "robot_name": "so101",
        "include_ui_extension": include_ui_extension,
        "workflows": [{"name": "manager-based", "type": "single-agent"}] if initial_content != "blank" else [],
        "rl_libraries": [{"name": "rsl_rl", "algorithms": ["ppo"]}] if initial_content != "blank" else [],
    }


def test_main_collects_canonical_external_project_choices():
    """The prompts must collect project layers and list flagship choices first."""
    initial_content = "stubbed"
    handler = mock.Mock(spec=CLIHandler)
    handler.input_select.side_effect = ["External", initial_content.title(), "No"]
    handler.input_path.return_value = "/tmp"
    handler.input_text.side_effect = ["test_project", "Test Author, Example Organization", "place_vial", "so101"]
    handler.input_checkbox.side_effect = lambda message, choices: [choices[0]]
    handler.get_choices.side_effect = CLIHandler.get_choices

    source_install = types.SimpleNamespace(__file__="/repo/source/isaaclab/isaaclab/__init__.py", __version__="3.0.0")
    with (
        mock.patch.object(_MODULE, "CLIHandler", return_value=handler),
        mock.patch.object(_MODULE.importlib, "import_module", return_value=source_install),
        mock.patch.object(_MODULE, "generate") as generate,
    ):
        _MODULE.main([])

    checkbox_calls = handler.input_checkbox.call_args_list
    assert checkbox_calls[0].kwargs["choices"][0] == "Manager-based | single-agent"
    assert checkbox_calls[1].kwargs["choices"][0] == "rsl_rl"
    assert checkbox_calls[2].kwargs["choices"][0] == "PPO"
    content_prompt = handler.input_select.call_args_list[1]
    assert content_prompt.kwargs["choices"] == ["Cartpole", "Stubbed", "Blank"]
    assert content_prompt.kwargs["default"] == "Cartpole"
    ui_prompt = handler.input_select.call_args_list[2]
    assert ui_prompt.kwargs["choices"] == ["No", "Yes"]
    assert ui_prompt.kwargs["default"] == "No"
    specification = generate.call_args.args[0]
    assert specification["task_name"] == "place_vial"
    assert specification["robot_name"] == "so101"
    assert specification["authors"] == ["Test Author", "Example Organization"]
    assert specification["initial_content"] == initial_content
    assert specification["include_ui_extension"] is False
    assert specification["isaaclab_version"] == "3.0.0"
    assert specification["isaaclab_source_path"] == _MODULE.ROOT_DIR


def test_main_skips_task_prompts_for_blank_project():
    """Blank projects must collect project metadata without asking for task or RL choices."""
    handler = mock.Mock(spec=CLIHandler)
    handler.input_select.side_effect = ["External", "Blank", "No"]
    handler.input_path.return_value = "/tmp"
    handler.input_text.side_effect = ["empty_project", "Test Author"]

    source_install = types.SimpleNamespace(__file__="/repo/source/isaaclab/isaaclab/__init__.py", __version__="3.0.0")
    with (
        mock.patch.object(_MODULE, "CLIHandler", return_value=handler),
        mock.patch.object(_MODULE.importlib, "import_module", return_value=source_install),
        mock.patch.object(_MODULE, "generate") as generate,
    ):
        _MODULE.main([])

    handler.input_checkbox.assert_not_called()
    specification = generate.call_args.args[0]
    assert specification["initial_content"] == "blank"
    assert specification["workflows"] == []
    assert specification["rl_libraries"] == []


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        (
            [],
            {
                "initial_content": "cartpole",
                "task_name": "balance",
                "robot_name": "cartpole",
                "workflows": [{"name": "manager-based", "type": "single-agent"}],
                "rl_libraries": [{"name": "rsl_rl", "algorithms": ["ppo"]}],
            },
        ),
        (["--initial_content", "blank"], {"workflows": [], "rl_libraries": []}),
        (
            ["--initial_content", "stubbed"],
            {
                "initial_content": "stubbed",
                "workflows": [{"name": "manager-based", "type": "single-agent"}],
                "rl_libraries": [{"name": "rsl_rl", "algorithms": ["ppo"]}],
            },
        ),
        (
            [
                "--workflow",
                "direct:multi-agent",
                "--rl_library",
                "skrl",
                "--rl_algorithm",
                "skrl:ippo",
                "--include_ui_extension",
            ],
            {
                "workflows": [{"name": "direct", "type": "multi-agent"}],
                "rl_libraries": [{"name": "skrl", "algorithms": ["ippo"]}],
                "include_ui_extension": True,
            },
        ),
        (
            ["--workflow", "direct:single-agent", "--workflow", "direct:multi-agent"],
            {
                "workflows": [
                    {"name": "direct", "type": "single-agent"},
                    {"name": "direct", "type": "multi-agent"},
                ],
                "rl_libraries": [{"name": "skrl", "algorithms": ["ppo", "ippo"]}],
            },
        ),
    ],
)
def test_non_interactive_generation(options, expected, tmp_path):
    """Non-interactive mode must support defaults, Blank projects, and explicit selections."""
    source_install = types.SimpleNamespace(__file__="/repo/source/isaaclab/isaaclab/__init__.py", __version__="3.0.0")
    with (
        mock.patch.object(_MODULE, "CLIHandler", side_effect=AssertionError("unexpected prompt")),
        mock.patch.object(_MODULE.importlib, "import_module", return_value=source_install),
        mock.patch.object(_MODULE, "generate") as generate,
    ):
        _MODULE.main(
            [
                "--non_interactive",
                "--project_path",
                str(tmp_path),
                "--name",
                "automated_project",
                "--author",
                "Test Author",
                *options,
            ]
        )

    specification = generate.call_args.args[0]
    assert {key: specification[key] for key in expected} == expected


def test_generation_arguments_require_non_interactive_opt_in(capsys):
    """Automation arguments must not silently change the default interactive flow."""
    with pytest.raises(SystemExit, match="2"):
        _MODULE.main(["--name", "unexpected_project"])

    assert "template arguments require --non_interactive" in capsys.readouterr().err

    source_install = types.SimpleNamespace(__file__="/repo/source/isaaclab/isaaclab/__init__.py", __version__="3.0.0")
    with (
        mock.patch.object(_MODULE.importlib, "import_module", return_value=source_install),
        pytest.raises(SystemExit, match="2"),
    ):
        _MODULE.main(
            [
                "--non_interactive",
                "--project_path",
                "/tmp",
                "--name",
                "blank_project",
                "--author",
                "Test Author",
                "--initial_content",
                "blank",
                "--workflow",
                "direct:single-agent",
            ]
        )

    assert "--workflow cannot be used with --initial_content blank" in capsys.readouterr().err


def test_generated_project_matches_canonical_uv_layout(tmp_path):
    """A generated project must use one uv project with a src package and tests, publishable to the registry."""
    specification = _external_specification(tmp_path)
    # Wheel builds carry a local label; the requirement must accept patch releases of the public version.
    specification["isaaclab_version"] = "3.0.0+build7.abc1234"

    with mock.patch.object(_GENERATOR, "_setup_git_repo"):
        _GENERATOR.generate(specification)

    project_dir = tmp_path / "test_project"
    with (project_dir / "pyproject.toml").open("rb") as file:
        project_config = tomllib.load(file)

    assert project_config["build-system"] == {
        "requires": ["uv_build>=0.12.6,<0.13"],
        "build-backend": "uv_build",
    }
    assert project_config["project"]["dependencies"] == ["isaaclab[rsl-rl]~=3.0.0"]
    assert project_config["project"]["authors"] == [
        {"name": "Test Author"},
        {"name": "Example Organization"},
    ]
    assert project_config["project"]["urls"] == {"Repository": ""}
    assert project_config["project"]["entry-points"]["isaaclab.tasks"] == {"test_project": "test_project.tasks"}
    assert project_config["project"]["optional-dependencies"] == {
        extra: [f"isaaclab[{extra}]~=3.0.0"]
        for extra in ("all", "isaacsim", "ov", "ovphysx", "ovrtx", "rerun", "viser")
    }
    assert project_config["dependency-groups"] == {
        "dev": ["codespell>=2.4", "pre-commit>=4.2", "pytest>=8.3", "ruff>=0.11"],
        "isaaclab-registry": [],
    }
    assert project_config["tool"]["isaaclab"]["registry"] == {
        "category": "other",
        "tags": [],
        "maintainers": [],
        "media": {},
    }
    uv_config = project_config["tool"]["uv"]
    assert uv_config["default-groups"] == ["dev", "isaaclab-registry"]
    assert uv_config["no-sources-package"] == ["isaaclab"]
    assert "sources" not in uv_config
    assert "override-dependencies" not in uv_config
    assert uv_config["build-backend"]["module-name"] == "test_project"
    assert project_config["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]
    assert project_config["tool"]["pytest"]["ini_options"]["markers"] == [
        "unit: test exercises isolated logic and does not launch the simulator",
        "integration: test drives the simulator/scene/environment end-to-end",
        "smoke: tests for core installation, task, and RL functionality",
        "kitless: test must pass inside the Kit-less container, which has no Isaac Sim runtime",
    ]

    module_dir = project_dir / "src" / "test_project"
    task_dir = module_dir / "tasks" / "place_vial" / "config" / "so101"
    assert (project_dir / "LICENSE").is_file()
    assert "Test Author, Example Organization" in (project_dir / "LICENSE").read_text()
    assert (project_dir / "tests" / "test_registration.py").is_file()
    assert "command_list_envs" in (project_dir / "scripts" / "list_envs.py").read_text()
    readme = (project_dir / "README.md").read_text()
    assert "uv run isaaclab list_envs --show_presets" in readme
    assert "## Share it on the Isaac Lab registry" in readme
    assert "uv run isaaclab registry publish" in readme
    assert (task_dir / "env_cfg.py").is_file()
    assert not (task_dir / "env.py").exists()
    assert (task_dir / "agents" / "rsl_rl_ppo_cfg.py").is_file()
    assert not (project_dir / "source").exists()
    assert not (project_dir / "config" / "extension.toml").exists()
    assert not (module_dir / "ui_extension_example.py").exists()
    assert "from .tasks import" not in (module_dir / "__init__.py").read_text()
    assets_module = (module_dir / "assets" / "__init__.py").read_text()
    assert 'TEST_PROJECT_ASSETS_DIR = Path(__file__).resolve().parent / "data"' in assets_module
    assert (module_dir / "assets" / "data").is_dir()
    assert "from test_project.assets import TEST_PROJECT_ASSETS_DIR" in (project_dir / "README.md").read_text()
    pyproject_text = (project_dir / "pyproject.toml").read_text()
    assert '\n[test_project.entry-points."isaaclab.tasks"]' not in pyproject_text
    assert '\n[project.entry-points."isaaclab.tasks"]\ntest_project = "test_project.tasks"' in pyproject_text
    assert "RigidBodyPropertiesCfg" not in (task_dir / "env_cfg.py").read_text()
    assert "ArticulationRootPropertiesCfg" not in (task_dir / "env_cfg.py").read_text()


def test_generated_project_uses_active_source_checkout(tmp_path):
    """Source-generated projects must use the checkout instead of requiring an unpublished wheel."""
    specification = _external_specification(tmp_path)
    specification["isaaclab_source_path"] = _MODULE.ROOT_DIR

    with mock.patch.object(_GENERATOR, "_setup_git_repo"):
        _GENERATOR.generate(specification)

    project_dir = tmp_path / "test_project"
    with (project_dir / "pyproject.toml").open("rb") as file:
        project_config = tomllib.load(file)

    expected_root = Path(_MODULE.ROOT_DIR)
    release = (expected_root / "VERSION").read_text().strip()
    with (expected_root / "pyproject.toml").open("rb") as file:
        source_config = tomllib.load(file)
    # The published metadata requires the checkout's release; the dev environment still gets the checkout.
    assert project_config["project"]["dependencies"] == [f"isaaclab[rsl-rl]~={release}"]
    # Only extras that the release wheel built from this checkout provides are published.
    unpublished = _GENERATOR._wheel_excluded_extras(_MODULE.ROOT_DIR)
    assert "rl-games" in unpublished
    assert project_config["project"]["optional-dependencies"] == {
        extra: [f"isaaclab[{extra}]~={release}"]
        for extra in source_config["project"]["optional-dependencies"]
        if extra not in unpublished
    }
    # The published extras resolve to the checkout's `isaaclab` without extras, so each checkout extra is also a
    # group; the checkout's `dev` extra would collide with the project's own `dev` group.
    assert project_config["dependency-groups"] == {
        "dev": ["codespell>=2.4", "pre-commit>=4.2", "pytest>=8.3", "ruff>=0.11"],
        "isaaclab-dev": ["isaaclab-dev[rsl-rl]"],
        "isaaclab-registry": [],
        **{
            extra: [f"isaaclab-dev[{extra}]"]
            for extra in source_config["project"]["optional-dependencies"]
            if extra != "dev"
        },
    }
    uv_config = project_config["tool"]["uv"]
    assert uv_config["default-groups"] == ["dev", "isaaclab-dev", "isaaclab-registry"]
    # Ignoring the `isaaclab` source would replace the checkout with a downloaded wheel.
    assert "no-sources-package" not in uv_config
    sources = uv_config["sources"]
    assert (project_dir / sources["isaaclab-dev"]["path"]).resolve() == expected_root.resolve()
    assert (project_dir / sources["isaaclab"]["path"]).resolve() == (expected_root / "source" / "isaaclab").resolve()
    assert sources["isaaclab-dev"]["editable"] is True
    assert sources["isaaclab"]["editable"] is True
    checkout_overrides = source_config["tool"]["uv"]["override-dependencies"]
    assert uv_config["override-dependencies"][: len(checkout_overrides)] == checkout_overrides
    peer_overrides = uv_config["override-dependencies"][len(checkout_overrides) :]
    assert "isaaclab" in peer_overrides
    assert "isaaclab-dev" not in peer_overrides
    assert uv_config["environments"] == source_config["tool"]["uv"]["environments"]
    assert sources["torch"] == source_config["tool"]["uv"]["sources"]["torch"]
    readme = (project_dir / "README.md").read_text()
    assert "### Develop against the Isaac Lab checkout" in readme
    # `--extra` selects the published extras, which install nothing from the checkout.
    assert "uv run --group isaacsim isaaclab random_agent" in readme
    assert "--extra isaacsim" not in readme

    # Path consumers must be able to build the dependency-only root, not just resolve its TOML.
    result = subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(tmp_path / "installed"),
            "--editable",
            str(expected_root),
        ],
        cwd=project_dir,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert list((tmp_path / "installed").glob("isaaclab_dev-*.dist-info"))


def _write_package(path: Path, name: str, version: str, dependencies: list[str], tables: str = "") -> None:
    """Write a minimal uv-buildable package with static metadata."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["uv_build>=0.11,<0.13"]\nbuild-backend = "uv_build"\n\n'
        f'[project]\nname = "{name}"\nversion = "{version}"\nrequires-python = ">=3.12"\n'
        f"dependencies = {json.dumps(dependencies)}\n{tables}"
    )


def test_source_project_resolves_registry_packages_against_checkout(tmp_path):
    """Registry packages requiring the release must resolve to the checkout, and removing one must restore the project.

    A checkout's ``isaaclab`` reports its package version rather than the release that registry packages require,
    so ``uv add`` only resolves through the generated peer overrides, which must keep the ``isaaclab-dev`` extras.
    The registry package is another checkout-generated project fetched from git, so it also carries ``isaaclab``
    path sources that point outside its repository.
    """
    checkout = tmp_path / "IsaacLab"
    _write_package(
        checkout,
        "isaaclab-dev",
        "0.1.0",
        ["isaaclab"],
        '\n[project.optional-dependencies]\nrsl-rl = ["rsl-rl-lib"]\n\n[tool.uv.sources]\n'
        'isaaclab = { path = "source/isaaclab", editable = true }\n'
        'rsl-rl-lib = { path = "source/rsl_rl_lib", editable = true }\n',
    )
    (checkout / "VERSION").write_text("3.0.0\n")
    _write_package(checkout / "source" / "isaaclab", "isaaclab", "37.0.0", [])
    _write_package(checkout / "source" / "rsl_rl_lib", "rsl-rl-lib", "5.0.0", [])
    for parent, name in ((tmp_path / "projects", "test_project"), (tmp_path / "authors" / "deep", "registry_package")):
        specification = _external_specification(parent)
        specification["name"] = name
        specification["isaaclab_source_path"] = str(checkout)
        with mock.patch.object(_GENERATOR, "_setup_git_repo"):
            _GENERATOR.generate(specification)
    registry_package = tmp_path / "authors" / "deep" / "registry_package"
    identity = ["-c", "user.name=Test Author", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false"]
    for command in (["init", "-q"], ["add", "-A"], [*identity, "commit", "-q", "-m", "Initial commit"]):
        subprocess.run(["git", *command], cwd=registry_package, capture_output=True, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=registry_package, capture_output=True, text=True, check=True
    ).stdout.strip()
    project_dir = tmp_path / "projects" / "test_project"
    original_pyproject = (project_dir / "pyproject.toml").read_bytes()

    def uv(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["uv", *args], cwd=project_dir, capture_output=True, text=True, timeout=300, check=False)

    requirement = f"registry-package @ git+{registry_package.as_uri()}@{commit}"
    result = uv("add", "--no-sync", "--group", "isaaclab-registry", requirement)
    assert result.returncode == 0, result.stderr
    with (project_dir / "uv.lock").open("rb") as file:
        packages = {package["name"]: package for package in tomllib.load(file)["package"]}
    assert packages["isaaclab"]["version"] == "37.0.0"
    assert packages["isaaclab"]["source"] == {"editable": "../../IsaacLab/source/isaaclab"}
    assert "rsl-rl-lib" in packages
    assert packages["registry-package"]["source"]["git"].endswith(commit)

    result = uv("remove", "--no-sync", "--group", "isaaclab-registry", "registry-package")
    assert result.returncode == 0, result.stderr
    assert (project_dir / "pyproject.toml").read_bytes() == original_pyproject


def test_generation_warns_when_initial_commit_fails(tmp_path, monkeypatch, caplog):
    """A missing git identity must be reported instead of silently leaving a repository without commits."""
    git_config = tmp_path / "gitconfig"
    git_config.write_text("[user]\n\tuseConfigOnly = true\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(git_config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for variable in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL", "EMAIL"):
        monkeypatch.delenv(variable, raising=False)

    _GENERATOR.generate(_external_specification(tmp_path, initial_content="blank"))

    assert "'git commit' failed" in caplog.text
    assert 'git commit -q -m "Initial commit"' in caplog.text


def test_installed_project_discovers_all_distribution_extras(tmp_path):
    """Wheel-generated projects must forward every extra declared in package metadata."""
    specification = _external_specification(tmp_path)
    specification.pop("isaaclab_optional_extras")
    distribution = mock.Mock()
    distribution.metadata.get_all.return_value = ["viser", "all", "rerun", "rerun"]

    with mock.patch.object(_GENERATOR.importlib.metadata, "distribution", return_value=distribution) as lookup:
        prepared = _GENERATOR._prepare_external_dependencies(specification, str(tmp_path / "test_project"))

    lookup.assert_called_once_with("isaaclab")
    assert prepared["isaaclab_optional_requirements"] == {
        "all": "isaaclab[all]~=3.0.0",
        "rerun": "isaaclab[rerun]~=3.0.0",
        "viser": "isaaclab[viser]~=3.0.0",
    }


def test_generated_blank_project_contains_no_example_task(tmp_path):
    """A blank project must include packaging infrastructure without cart-pole task files."""
    specification = _external_specification(tmp_path, initial_content="blank")

    with mock.patch.object(_GENERATOR, "_setup_git_repo"):
        _GENERATOR.generate(specification)

    project_dir = tmp_path / "test_project"
    tasks_dir = project_dir / "src" / "test_project" / "tasks"
    assert [path.name for path in tasks_dir.iterdir()] == ["__init__.py"]
    assert (project_dir / "src" / "test_project" / "assets" / "__init__.py").is_file()
    assert (project_dir / "src" / "test_project" / "assets" / "data" / ".gitkeep").is_file()
    assert "No tasks are registered yet" in (project_dir / "README.md").read_text()
    assert "cartpole" not in (project_dir / "README.md").read_text().lower()
    compile((project_dir / "tests" / "test_registration.py").read_text(), "test_registration.py", "exec")
    with (project_dir / "pyproject.toml").open("rb") as file:
        project_config = tomllib.load(file)
    assert project_config["project"]["dependencies"] == ["isaaclab~=3.0.0"]


def test_source_path_falls_back_to_absolute_path_across_windows_drives(monkeypatch):
    """A project on another Windows drive must still receive a valid source path."""
    monkeypatch.setattr(_GENERATOR.os.path, "relpath", mock.Mock(side_effect=ValueError))
    monkeypatch.setattr(_GENERATOR.os.path, "realpath", lambda path: path)

    assert _GENERATOR._project_source_path(r"D:\IsaacLab", r"C:\projects\robot") == "D:/IsaacLab"


def test_generated_project_can_opt_into_ui_extension(tmp_path):
    """Direct tasks and UI extension files must follow the canonical project layout."""
    specification = _external_specification(tmp_path, include_ui_extension=True)
    specification["authors"] = ['Test "Quoted" Author']
    specification["workflows"] = [{"name": "direct", "type": "single-agent"}]

    with mock.patch.object(_GENERATOR, "_setup_git_repo"):
        _GENERATOR.generate(specification)

    project_dir = tmp_path / "test_project"
    task_dir = project_dir / "src" / "test_project" / "tasks" / "place_vial_direct" / "config" / "so101"
    registration_test = (project_dir / "tests" / "test_registration.py").read_text()
    with (project_dir / "config" / "extension.toml").open("rb") as file:
        extension_config = tomllib.load(file)
    assert (task_dir / "env.py").is_file()
    assert (task_dir / "env_cfg.py").is_file()
    assert "test_project.tasks.place_vial_direct.config.so101.env:PlaceVialEnv" in registration_test
    assert extension_config["package"]["author"] == 'Test "Quoted" Author'
    assert extension_config["python"]["module"] == [{"name": "test_project.ui_extension_example"}]
    assert (project_dir / "src" / "test_project" / "ui_extension_example.py").is_file()


def test_internal_task_keeps_repository_layout(tmp_path):
    """Aligning external projects must not change internal task filenames or layout."""
    specification = {
        "external": False,
        "name": "test_task",
        "workflows": [{"name": "manager-based", "type": "single-agent"}],
        "rl_libraries": [{"name": "rsl_rl", "algorithms": ["ppo"]}],
    }

    generated = _GENERATOR._generate_tasks(specification, str(tmp_path))

    task = generated[0]["task"]
    task_dir = tmp_path / "test_task" / "config" / "cartpole"
    assert task["id"] == "Isaac-Test-Task"
    assert (task_dir / "test_task_env_cfg.py").is_file()
