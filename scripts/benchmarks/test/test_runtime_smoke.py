# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke and compatibility tests for the runtime benchmark entry point."""

import argparse
import ast
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from scripts.benchmarks import _compat

ROOT = Path(__file__).resolve().parents[3]

_TASK = "Isaac-Cartpole-v0"


def test_compat_import_statements_are_standard_library_only():
    """The private adapter has no import statement outside the standard library."""
    tree = ast.parse((ROOT / "scripts" / "benchmarks" / "_compat.py").read_text())
    import_roots = {node.names[0].name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)} | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert import_roots <= sys.stdlib_module_names | {"__future__"}


def test_launcher_arguments_are_registered_once(monkeypatch):
    """The compatibility helper makes launcher argument registration idempotent."""
    calls = []

    class _FakeAppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            calls.append(parser)

    app_module = types.ModuleType("isaaclab.app")
    app_module.AppLauncher = _FakeAppLauncher
    monkeypatch.setitem(sys.modules, "isaaclab.app", app_module)

    parser = argparse.ArgumentParser()
    _compat.add_launcher_args(parser)
    _compat.add_launcher_args(parser)

    assert calls == [parser]


@pytest.mark.parametrize("script_name", ["runtime.py", "startup.py"])
def test_task_config_lookup_happens_after_app_launch(script_name):
    """Every 2.x benchmark script launches Kit before task registration/config lookup."""
    tree = ast.parse((ROOT / "scripts" / "benchmarks" / script_name).read_text())
    calls = {
        node.func.id: node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"launch_app", "resolve_task_config"}
    }

    assert calls["launch_app"] < calls["resolve_task_config"]


@pytest.mark.parametrize("script_name", ["runtime.py", "startup.py"])
def test_post_launch_work_is_guarded_by_app_cleanup(script_name):
    """The first post-launch statement enters a try that always closes the app."""
    tree = ast.parse((ROOT / "scripts" / "benchmarks" / script_name).read_text())
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


def test_runtime_import_timer_starts_after_app_launch():
    """Runtime import timing excludes the separately reported AppLauncher interval."""
    tree = ast.parse((ROOT / "scripts" / "benchmarks" / "runtime.py").read_text())
    run_function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    assignment_lines = {
        target.id: node.lineno
        for node in ast.walk(run_function)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in {"app_t1", "imports_t0", "imports_t1"}
    }

    assert assignment_lines["app_t1"] < assignment_lines["imports_t0"] < assignment_lines["imports_t1"]


def test_newton_preset_is_rejected_before_launch():
    """A 3.0-only solver preset reports the supported 2.x selection."""
    with pytest.raises(ValueError, match=r"Isaac Lab 2.x.*presets=physx"):
        _compat.parse_benchmark_args(argparse.ArgumentParser(), ["presets=newton_mjwarp"])


def test_run_manifest_records_both_checkout_shas(tmp_path, monkeypatch):
    """Run manifests pin both sides of the cross-version comparison."""
    monkeypatch.setenv("ISAACLAB_BENCHMARK_LAB2_SHA", "2" * 40)
    monkeypatch.setenv("ISAACLAB_BENCHMARK_LAB3_SHA", "3" * 40)
    monkeypatch.setattr(_compat, "_git_sha", lambda _path: None)

    _compat.write_run_manifest(str(tmp_path), library="rsl_rl", task=_TASK)

    manifest = json.loads((tmp_path / "run.json").read_text())
    assert manifest["git_shas"] == {"lab2": "2" * 40, "lab3": "3" * 40}


def test_run_manifest_rejects_missing_checkout_shas(tmp_path, monkeypatch):
    """A manifest cannot silently claim reproducibility with null Git SHAs."""
    monkeypatch.delenv("ISAACLAB_BENCHMARK_LAB2_SHA", raising=False)
    monkeypatch.delenv("ISAACLAB_BENCHMARK_LAB3_SHA", raising=False)
    monkeypatch.setattr(_compat, "_git_sha", lambda _path: None)

    with pytest.raises(RuntimeError, match="ISAACLAB_BENCHMARK_LAB2_SHA.*ISAACLAB_BENCHMARK_LAB3_SHA"):
        _compat.write_run_manifest(str(tmp_path), library="rsl_rl", task=_TASK)


def test_run_manifest_rejects_environment_sha_that_disagrees_with_checkout(tmp_path, monkeypatch):
    """An environment override cannot misidentify an available checkout."""
    monkeypatch.setenv("ISAACLAB_BENCHMARK_LAB2_SHA", "4" * 40)
    monkeypatch.setenv("ISAACLAB_BENCHMARK_LAB3_SHA", "3" * 40)
    monkeypatch.setattr(
        _compat,
        "_git_sha",
        lambda path: "3" * 40 if path.name == "lab3-develop" else "2" * 40,
    )

    with pytest.raises(RuntimeError, match=r"ISAACLAB_BENCHMARK_LAB2_SHA.*does not match.*checkout"):
        _compat.write_run_manifest(str(tmp_path), library="rsl_rl", task=_TASK)


def test_run_manifest_rejects_abbreviated_sha_identity(tmp_path, monkeypatch):
    """A run manifest requires two full 40-character Git SHAs."""
    monkeypatch.setenv("ISAACLAB_BENCHMARK_LAB2_SHA", "2" * 10)
    monkeypatch.setenv("ISAACLAB_BENCHMARK_LAB3_SHA", "3" * 40)
    monkeypatch.setattr(_compat, "_git_sha", lambda _path: None)

    with pytest.raises(RuntimeError, match=r"full 40-character Git SHA"):
        _compat.write_run_manifest(str(tmp_path), library="rsl_rl", task=_TASK)


def test_runtime_writes_all_requested_formats(tmp_path):
    """The runtime entry point writes schema and JSON data in one run."""
    sh = ROOT / "isaaclab.sh"
    cmd = [
        str(sh),
        "-p",
        "scripts/benchmarks/runtime.py",
        "--task",
        _TASK,
        "--num_envs",
        "16",
        "--num_frames",
        "10",
        "--seed",
        "0",
        "--device",
        "cpu",
        "--output_path",
        str(tmp_path),
        "--benchmark_formatter",
        "schema,json",
        "presets=physx",
        "--headless",
    ]
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    if res.returncode != 0:
        pytest.fail(f"runtime.py rc={res.returncode}\nSTDOUT:\n{res.stdout[-2000:]}\nSTDERR:\n{res.stderr[-2000:]}")

    device_lines = [line for line in res.stdout.splitlines() if "Environment device" in line]
    assert device_lines and device_lines[-1].endswith(": cpu"), f"unexpected device output: {device_lines}"

    files = sorted(tmp_path.glob("*.json"))
    schema_files = [path for path in files if path.name.endswith("_schema.json")]
    json_files = [path for path in files if path.name.endswith("_json.json")]
    assert len(schema_files) == len(json_files) == 1

    schema_data = json.loads(schema_files[0].read_text())
    assert schema_data["run"]["config"]["physics_backend"] == "physx"
    assert json.loads(json_files[0].read_text())
