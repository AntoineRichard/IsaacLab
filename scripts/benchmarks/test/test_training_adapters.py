# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for RL-library benchmark adapter behavior."""

import argparse
import ast
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.benchmarks import _compat, play, training
from scripts.benchmarks.rsl_rl import benchmark_rsl_rl_play as play_rsl_rl
from scripts.benchmarks.rsl_rl import benchmark_rsl_rl_train as train_rsl_rl

ROOT = Path(__file__).resolve().parents[3]
_TASK = "Isaac-Cartpole-v0"


def test_training_dispatches_rsl_rl_to_library_named_adapter():
    """The training dispatcher uses the RSL-RL-named benchmark adapter."""
    assert training.LIBRARY_ENTRYPOINTS["rsl_rl"].name == "benchmark_rsl_rl_train.py"


def test_play_dispatches_rsl_rl_to_library_named_adapter():
    """The play dispatcher uses the RSL-RL-named benchmark adapter."""
    assert play.LIBRARY_ENTRYPOINTS["rsl_rl"].name == "benchmark_rsl_rl_play.py"


def test_play_adapter_help_does_not_require_task():
    """Play adapter help exits successfully without a task selection."""
    with pytest.raises(SystemExit) as exc_info:
        play_rsl_rl._parse_args(["--help"])
    assert exc_info.value.code == 0


def test_rsl_rl_disables_code_state_capture():
    """RSL-RL cannot inspect Git state in the benchmark container."""
    logger = SimpleNamespace(git_status_repos=["rsl_rl.py"])
    runner = SimpleNamespace(logger=logger)

    train_rsl_rl._disable_code_state_capture(runner)

    assert logger.git_status_repos == []


def test_training_parser_accepts_measured_cli_and_physx_selector(monkeypatch: pytest.MonkeyPatch):
    """The 2.x adapter accepts the exact measured benchmark command surface."""

    class _FakeAppLauncher:
        @staticmethod
        def add_app_launcher_args(parser: argparse.ArgumentParser) -> None:
            parser.add_argument("--headless", action="store_true")
            parser.add_argument("--device")

    app_module = types.ModuleType("isaaclab.app")
    app_module.AppLauncher = _FakeAppLauncher
    monkeypatch.setitem(sys.modules, "isaaclab.app", app_module)

    args, remaining, _ = train_rsl_rl._parse_args(
        [
            "--task",
            _TASK,
            "--num_envs",
            "16",
            "--seed",
            "7",
            "--max_iterations",
            "2",
            "--benchmark_formatter",
            "schema,json",
            "presets=physx",
            "--headless",
        ]
    )

    assert args.task == _TASK
    assert args.num_envs == 16
    assert args.seed == 7
    assert args.max_iterations == 2
    assert args.benchmark_formatter == "schema,json"
    assert args.headless is True
    assert remaining == ["presets=physx"]
    assert sys.argv == [sys.argv[0]]


def test_training_overrides_propagate_seed_and_iteration_limit():
    """Legacy RSL-RL config updates drive the environment seed and iteration limit."""
    args = SimpleNamespace(num_envs=16, seed=7, max_iterations=2, distributed=False, device=None)
    env_cfg = SimpleNamespace(
        scene=SimpleNamespace(num_envs=1),
        sim=SimpleNamespace(device="cuda:0"),
        seed=0,
    )
    agent_cfg = SimpleNamespace(seed=0, max_iterations=100)

    class _FakeCliArgs:
        @staticmethod
        def update_rsl_rl_cfg(config, parsed_args):
            config.seed = parsed_args.seed
            return config

    updated = train_rsl_rl._apply_training_overrides(args, env_cfg, agent_cfg, _FakeCliArgs)

    assert updated is agent_cfg
    assert env_cfg.scene.num_envs == 16
    assert env_cfg.seed == agent_cfg.seed == 7
    assert agent_cfg.max_iterations == 2


@pytest.mark.parametrize("script_name", ["benchmark_rsl_rl_train.py", "benchmark_rsl_rl_play.py"])
def test_rsl_rl_task_config_lookup_happens_after_app_launch(script_name: str):
    """The 2.x adapters launch Kit before task registration and config lookup."""
    tree = ast.parse((ROOT / "scripts" / "benchmarks" / "rsl_rl" / script_name).read_text())
    calls = {
        node.func.id: node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"launch_app", "resolve_task_config"}
    }

    assert calls["launch_app"] < calls["resolve_task_config"]


@pytest.mark.parametrize("script_name", ["benchmark_rsl_rl_train.py", "benchmark_rsl_rl_play.py"])
def test_rsl_rl_post_launch_work_is_guarded_by_app_cleanup(script_name: str):
    """Both adapters close SimulationApp even when post-launch work raises."""
    tree = ast.parse((ROOT / "scripts" / "benchmarks" / "rsl_rl" / script_name).read_text())
    run_function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    simulation_app_index = next(
        index
        for index, node in enumerate(run_function.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "simulation_app" for target in node.targets)
    )

    cleanup_guard = run_function.body[simulation_app_index + 1]
    assert isinstance(cleanup_guard, ast.Try)
    assert any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "close"
        for node in ast.walk(ast.Module(body=cleanup_guard.finalbody, type_ignores=[]))
    )


def test_rsl_rl_latest_checkpoint_ignores_newer_mismatched_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The latest selector ignores newer runs from different checkout SHAs."""
    monkeypatch.setenv("ISAACLAB_BENCHMARK_LAB2_SHA", "2" * 40)
    monkeypatch.setenv("ISAACLAB_BENCHMARK_LAB3_SHA", "3" * 40)
    monkeypatch.setattr(_compat, "_git_sha", lambda _path: None)
    older = tmp_path / "2026-01-01_00-00-00"
    newer = tmp_path / "2026-01-02_00-00-00"
    wrong_agent = tmp_path / "2026-01-03_00-00-00"
    for run_dir, agent in ((older, "agent"), (newer, "agent"), (wrong_agent, "other")):
        _compat.write_run_manifest(str(run_dir), library="rsl_rl", task=_TASK, metadata={"agent": agent})
        (run_dir / "model_1.pt").write_bytes(b"checkpoint")
    older_manifest = json.loads((older / "run.json").read_text())
    newer_manifest = json.loads((newer / "run.json").read_text())
    wrong_manifest = json.loads((wrong_agent / "run.json").read_text())
    older_manifest["created_at"] = "2026-01-01T00:00:00+00:00"
    newer_manifest["created_at"] = "2026-01-02T00:00:00+00:00"
    wrong_manifest["created_at"] = "2026-01-03T00:00:00+00:00"
    newer_manifest["git_shas"] = {"lab2": "4" * 40, "lab3": "3" * 40}
    older.joinpath("run.json").write_text(json.dumps(older_manifest))
    newer.joinpath("run.json").write_text(json.dumps(newer_manifest))
    wrong_agent.joinpath("run.json").write_text(json.dumps(wrong_manifest))

    checkpoint = _compat.resolve_checkpoint_selector(
        str(tmp_path),
        "latest",
        library="rsl_rl",
        task=_TASK,
        checkpoint_pattern=r"model_.*\.pt",
        metadata={"agent": "agent"},
    )

    assert checkpoint == str((older / "model_1.pt").resolve())
    assert json.loads((older / "run.json").read_text())["git_shas"] == {
        "lab2": "2" * 40,
        "lab3": "3" * 40,
    }


def test_rsl_rl_play_passes_expected_provenance_to_checkpoint_selector():
    """Play resolves current checkout identity before selecting a manifested run."""
    tree = ast.parse((ROOT / "scripts" / "benchmarks" / "rsl_rl" / "benchmark_rsl_rl_play.py").read_text())
    assignments = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    selector_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "resolve_checkpoint_selector"
    )

    assert "expected_git_shas" in assignments
    assert any(keyword.arg == "expected_git_shas" for keyword in selector_call.keywords)
