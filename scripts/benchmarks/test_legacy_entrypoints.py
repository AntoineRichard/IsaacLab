# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility tests for the deprecated benchmark entry points."""

import ast
import importlib
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LEGACY_ENTRYPOINTS = {
    "benchmark_non_rl": ("runtime", None),
    "benchmark_rlgames": ("training", "rl_games"),
    "benchmark_rsl_rl": ("training", "rsl_rl"),
}
BACKEND_FORMATTERS = {
    "LocalLogMetrics": "summary",
    "JSONFileMetrics": "json",
    "OsmoKPIFile": "osmo",
    "OmniPerfKPIFile": "omniperf",
}


def _import_legacy_entrypoint(module_name: str):
    """Import a legacy entry point after asserting it has no launch-time work."""
    path = ROOT / "scripts" / "benchmarks" / f"{module_name}.py"
    tree = ast.parse(path.read_text())
    top_level_calls = [node for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr))]
    assert not any("AppLauncher" in ast.unparse(node) for node in top_level_calls), (
        f"{module_name}.py must be an import-safe compatibility wrapper"
    )
    return importlib.import_module(f"scripts.benchmarks.{module_name}")


@pytest.mark.parametrize("module_name", LEGACY_ENTRYPOINTS)
@pytest.mark.parametrize("backend,formatter", BACKEND_FORMATTERS.items())
def test_legacy_backend_is_translated_and_supported_flags_are_forwarded(
    module_name: str,
    backend: str,
    formatter: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Every legacy backend maps to the typed formatter with supported workload flags."""
    module = _import_legacy_entrypoint(module_name)
    target_name, library = LEGACY_ENTRYPOINTS[module_name]
    calls = []
    target = types.ModuleType(f"scripts.benchmarks.{target_name}")

    if target_name == "runtime":
        target.run = lambda argv: calls.append(argv)
        expected = ["--task", "Isaac-Cartpole-v0", "--num_frames", "7"]
    else:
        target.main = lambda argv: calls.append(argv) or 0
        expected = ["--rl_library", library, "--task", "Isaac-Cartpole-v0", "--max_iterations", "7"]
        if library == "rsl_rl":
            expected += ["--num_envs", "4096", "--seed", "42"]

    monkeypatch.setitem(sys.modules, f"scripts.benchmarks.{target_name}", target)
    result = module.main(
        [
            "--task",
            "Isaac-Cartpole-v0",
            "--benchmark_backend",
            backend,
            "--num_frames" if target_name == "runtime" else "--max_iterations",
            "7",
        ]
    )

    assert result == 0
    assert calls == [[*expected, "--benchmark_formatter", formatter]]
    stderr = capsys.readouterr().err
    assert "deprecated" in stderr.lower()
    assert f"scripts/benchmarks/{target_name}.py" in stderr


@pytest.mark.parametrize("module_name", LEGACY_ENTRYPOINTS)
def test_legacy_default_backend_maps_to_omniperf(module_name: str):
    """Omitting the historical backend retains its OmniPerf default."""
    module = _import_legacy_entrypoint(module_name)

    assert module._translate_legacy_args(["--headless"])[-2:] == ["--benchmark_formatter", "omniperf"]


@pytest.mark.parametrize(
    "module_name,defaults",
    [
        ("benchmark_rlgames", {"--max_iterations": "10"}),
        ("benchmark_rsl_rl", {"--num_envs": "4096", "--seed": "42", "--max_iterations": "10"}),
    ],
)
def test_legacy_training_defaults_are_forwarded(module_name: str, defaults: dict[str, str]):
    """Training wrappers retain the historical workload defaults."""
    module = _import_legacy_entrypoint(module_name)

    forwarded = module._translate_legacy_args([])

    for option, value in defaults.items():
        assert forwarded[forwarded.index(option) + 1] == value


@pytest.mark.parametrize("module_name", ["benchmark_rlgames", "benchmark_rsl_rl"])
def test_explicit_legacy_training_values_override_defaults(module_name: str):
    """Explicit historical workload values are forwarded once and unchanged."""
    module = _import_legacy_entrypoint(module_name)
    argv = ["--num_envs=8", "--seed", "7", "--max_iterations", "3"]

    forwarded = module._translate_legacy_args(argv)

    assert forwarded.count("--num_envs=8") == 1
    assert forwarded.count("--seed") == 1
    assert forwarded.count("--max_iterations") == 1


@pytest.mark.parametrize("module_name", LEGACY_ENTRYPOINTS)
def test_legacy_invalid_backend_keeps_argparse_exit_code_and_error(
    module_name: str, capsys: pytest.CaptureFixture[str]
):
    """An invalid legacy backend still exits with argparse status 2 and names the bad value."""
    module = _import_legacy_entrypoint(module_name)

    with pytest.raises(SystemExit) as exc_info:
        module.main(["--benchmark_backend", "NotABackend"])

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "invalid choice" in stderr
    assert "NotABackend" in stderr


@pytest.mark.parametrize("module_name", ["benchmark_rlgames", "benchmark_rsl_rl"])
def test_legacy_training_propagates_delegated_exit_status(module_name: str, monkeypatch: pytest.MonkeyPatch):
    """Training wrappers return the unified dispatcher's status unchanged."""
    module = _import_legacy_entrypoint(module_name)
    target = types.ModuleType("scripts.benchmarks.training")
    target.main = lambda _argv: 23
    monkeypatch.setitem(sys.modules, "scripts.benchmarks.training", target)

    assert module.main([]) == 23


def test_legacy_non_rl_help_preserves_success_exit(capsys: pytest.CaptureFixture[str]):
    """The historical non-RL help path exits successfully without requiring a task."""
    module = _import_legacy_entrypoint("benchmark_non_rl")

    with pytest.raises(SystemExit) as exc_info:
        module.main(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr()
    assert "Benchmark environment runtime" in output.out
    assert "deprecated" in output.err.lower()


@pytest.mark.parametrize("module_name", LEGACY_ENTRYPOINTS)
def test_legacy_script_runs_from_outside_repository(module_name: str, tmp_path: Path):
    """A legacy script resolves the unified entry point without repository PYTHONPATH state."""
    script = ROOT / "scripts" / "benchmarks" / f"{module_name}.py"
    env = os.environ.copy()
    python_path = env.get("PYTHONPATH", "").split(os.pathsep)
    env["PYTHONPATH"] = os.pathsep.join(
        entry for entry in python_path if entry and Path(entry).resolve() != ROOT.resolve()
    )

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "deprecated" in result.stderr.lower()
