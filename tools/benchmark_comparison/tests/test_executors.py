# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for benchmark command generation and executor preflight."""

from __future__ import annotations

import hashlib
import signal
import sys
from pathlib import Path

import pytest

from tools.benchmark_comparison.executors import (
    CommandResult,
    ExecutorConfig,
    Invocation,
    Lab2DockerExecutor,
    Lab3UvExecutor,
    PreflightError,
    ProcessLauncher,
    ProcessResult,
    run_preflight,
)
from tools.benchmark_comparison.matrix import expand_final_matrix, load_matrix
from tools.benchmark_comparison.models import Version

LAB2_SHA = "28a9560c59df2306690ea717d6cf36f1e63c66e3"
LAB3_SHA = "cb508381fb4874ce7afffeb9197bd91c20db7dad"


def _config(tmp_path: Path) -> ExecutorConfig:
    lab2 = tmp_path / "lab2-main"
    lab3 = tmp_path / "lab3-develop"
    artifacts = tmp_path / "artifacts"
    lab2.mkdir()
    lab3.mkdir()
    artifacts.mkdir()
    (lab3 / "uv.lock").write_text("locked\n", encoding="utf-8")
    return ExecutorConfig(
        lab2_root=lab2,
        lab3_root=lab3,
        artifact_root=artifacts,
        lab2_sha=LAB2_SHA,
        lab3_sha=LAB3_SHA,
        lab2_image="isaac-lab-base-benchmark",
        lab2_image_id="sha256:" + "a" * 64,
    )


def _attempt(version: Version, mode: str = "runtime-100"):
    return next(
        attempt
        for attempt in expand_final_matrix(load_matrix()).attempts
        if attempt.version is version and attempt.mode.id == mode
    )


def test_lab2_runtime_command_is_an_argument_vector_with_container_output(tmp_path: Path):
    config = _config(tmp_path)
    executor = Lab2DockerExecutor(config)
    attempt = _attempt(Version.LAB2)

    invocation = executor.invocation(attempt)

    assert invocation.argv[:10] == (
        "docker",
        "compose",
        "--env-file",
        str(config.lab2_root / "docker/.env.base"),
        "-f",
        str(config.lab2_root / "docker/docker-compose.yaml"),
        "-f",
        str(config.lab2_root / "tools/benchmark_comparison/docker-compose.benchmark.yaml"),
        "run",
        "--rm",
    )
    assert "--name" in invocation.argv
    assert "isaac-lab-benchmark" in invocation.argv
    assert "/workspace/isaaclab/scripts/benchmarks/runtime.py" in invocation.argv
    assert invocation.argv[-12:] == (
        "--task",
        attempt.concrete_task,
        "--num_envs",
        "4096",
        "--seed",
        "42",
        "--num_frames",
        "100",
        "--benchmark_formatter",
        "schema,json",
        "presets=physx",
        "--headless",
    )
    output_index = invocation.argv.index("--output_path")
    assert invocation.argv[output_index + 1].startswith("/benchmark_artifacts/")
    assert invocation.shell is False


def test_lab2_training_command_uses_rsl_rl_and_exact_sha_environment(tmp_path: Path):
    config = _config(tmp_path)
    invocation = Lab2DockerExecutor(config).invocation(_attempt(Version.LAB2, "training-100"))

    assert "/workspace/isaaclab/scripts/benchmarks/training.py" in invocation.argv
    assert invocation.argv[invocation.argv.index("--rl_library") : invocation.argv.index("--rl_library") + 2] == (
        "--rl_library",
        "rsl_rl",
    )
    assert invocation.argv[
        invocation.argv.index("--max_iterations") : invocation.argv.index("--max_iterations") + 2
    ] == ("--max_iterations", "100")
    assert invocation.environment["ISAACLAB_BENCHMARK_LAB2_SHA"] == LAB2_SHA
    assert invocation.environment["ISAACLAB_BENCHMARK_LAB3_SHA"] == LAB3_SHA
    assert invocation.environment["OMNI_KIT_ACCEPT_EULA"] == "yes"


def test_lab3_runtime_and_training_commands_use_locked_uv_project(tmp_path: Path):
    config = _config(tmp_path)
    executor = Lab3UvExecutor(config)

    runtime = executor.invocation(_attempt(Version.LAB3))
    training = executor.invocation(_attempt(Version.LAB3, "training-100"))

    prefix = (
        "uv",
        "run",
        "--project",
        str(config.lab3_root),
        "--extra",
        "isaacsim",
        "--extra",
        "rsl-rl",
        "--locked",
        "python",
    )
    assert runtime.argv[: len(prefix)] == prefix
    assert str(config.lab3_root / "scripts/benchmarks/runtime.py") in runtime.argv
    assert str(config.lab3_root / "scripts/benchmarks/training.py") in training.argv
    assert "presets=physx" in runtime.argv
    assert "presets=physx" in training.argv
    assert runtime.environment["ISAACLAB_BENCHMARK_LAB2_SHA"] == LAB2_SHA
    assert runtime.environment["ISAACLAB_BENCHMARK_LAB3_SHA"] == LAB3_SHA


def test_version_probe_commands_are_argv_vectors(tmp_path: Path):
    config = _config(tmp_path)

    lab2 = Lab2DockerExecutor(config).version_invocation()
    lab3 = Lab3UvExecutor(config).version_invocation()

    assert lab2.shell is lab3.shell is False
    assert lab2.argv[-2] == "-c"
    assert lab3.argv[:9] == (
        "uv",
        "run",
        "--project",
        str(config.lab3_root),
        "--extra",
        "isaacsim",
        "--extra",
        "rsl-rl",
        "--locked",
    )
    assert "MetricsFormatter.get_instance('schema')" in lab2.argv[-1]
    assert "MetricsFormatter.get_instance('json')" in lab3.argv[-1]


def test_version_probes_require_every_configured_task_registration(tmp_path: Path):
    config = _config(tmp_path)
    matrix = load_matrix()

    lab2_probe = Lab2DockerExecutor(config).version_invocation().argv[-1]
    lab3_probe = Lab3UvExecutor(config).version_invocation().argv[-1]

    assert all(task.lab2_id in lab2_probe for task in matrix.tasks)
    assert all(task.lab3_id in lab3_probe for task in matrix.tasks)


def test_version_probes_use_version_specific_app_startup_and_sentinel(tmp_path: Path):
    config = _config(tmp_path)

    lab2_probe = Lab2DockerExecutor(config).version_invocation().argv[-1]
    lab3_probe = Lab3UvExecutor(config).version_invocation().argv[-1]

    app_launcher = "AppLauncher(headless=True)"
    assert lab2_probe.index(app_launcher) < lab2_probe.index("MetricsFormatter")
    assert lab2_probe.index(app_launcher) < lab2_probe.index("isaaclab_tasks")
    assert "__ISAACLAB_BENCHMARK_PREFLIGHT_OK__" in lab2_probe
    assert "flush=True" in lab2_probe
    assert "AppLauncher" not in lab3_probe
    assert "__ISAACLAB_BENCHMARK_PREFLIGHT_OK__" not in lab3_probe


def test_child_timeout_terminates_process_group_and_cleans_only_owned_container(tmp_path: Path):
    class Commands:
        def __init__(self):
            self.argvs = []

        def run(self, argv, **_kwargs):
            self.argvs.append(tuple(argv))
            return CommandResult(tuple(argv), 0, "", "")

    commands = Commands()
    owned_group_ids = []

    class OwnedGroups:
        def add(self, process_group_id):
            owned_group_ids.append(process_group_id)

    launcher = ProcessLauncher(commands, terminate_grace_s=0.05, owned_process_groups=OwnedGroups())
    invocation = Invocation(
        argv=(sys.executable, "-c", "import time; time.sleep(60)"),
        environment={},
        cwd=tmp_path,
        container_name="owned-benchmark-container",
    )

    result = launcher.run(invocation, timeout_s=0.05)

    assert result.timed_out is True
    assert len(owned_group_ids) == 1
    assert commands.argvs == [("docker", "rm", "--force", "owned-benchmark-container")]


def test_process_launcher_installs_and_restores_sigterm_handler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls = []

    def fake_signal(signal_number, handler):
        calls.append((signal_number, handler))
        return "previous-handler"

    monkeypatch.setattr("tools.benchmark_comparison.executors.signal.signal", fake_signal)
    launcher = ProcessLauncher()
    invocation = Invocation(
        argv=(sys.executable, "-c", "print('complete')"),
        environment={},
        cwd=tmp_path,
    )

    result = launcher.run(invocation, timeout_s=1)

    assert result.returncode == 0
    assert calls[0][0] == signal.SIGTERM
    assert callable(calls[0][1])
    assert calls[-1] == (signal.SIGTERM, "previous-handler")


def test_compose_uses_image_native_workspace_and_only_artifact_mount():
    compose = Path("tools/benchmark_comparison/docker-compose.benchmark.yaml").read_text(encoding="utf-8")

    assert "ISAACLAB_BENCHMARK_LAB2_ROOT" not in compose
    assert "source: ${ISAACLAB_BENCHMARK_ARTIFACT_ROOT}" in compose
    assert compose.count("type: bind") == 1
    assert "image: ${ISAACLAB_BENCHMARK_IMAGE_ID}" in compose


def test_lab2_invocation_uses_exact_image_id_without_host_source_mount(tmp_path: Path):
    config = _config(tmp_path)

    invocation = Lab2DockerExecutor(config).invocation(_attempt(Version.LAB2))

    assert "ISAACLAB_BENCHMARK_LAB2_ROOT" not in invocation.environment
    assert invocation.environment["ISAACLAB_BENCHMARK_IMAGE_ID"] == config.lab2_image_id
    assert "/workspace/isaaclab/isaaclab.sh" in invocation.argv


class _PreflightCommands:
    def __init__(
        self,
        config: ExecutorConfig,
        *,
        fail_contains: str | None = None,
        lab2_probe_stdout: str = "kit startup log\n__ISAACLAB_BENCHMARK_PREFLIGHT_OK__\n",
        lab3_probe_stdout: str = "ok\n",
    ):
        self.config = config
        self.fail_contains = fail_contains
        self.lab2_probe_stdout = lab2_probe_stdout
        self.lab3_probe_stdout = lab3_probe_stdout
        self.argvs: list[tuple[str, ...]] = []

    def run(self, argv, *, cwd=None, environment=None, timeout=None):
        argv = tuple(argv)
        self.argvs.append(argv)
        rendered = " ".join(argv)
        if self.fail_contains and self.fail_contains in rendered:
            return CommandResult(argv, 1, "", "forced failure")
        if argv[:2] == ("git", "-C") and argv[-2:] == ("rev-parse", "HEAD"):
            sha = self.config.lab2_sha if Path(argv[2]) == self.config.lab2_root else self.config.lab3_sha
            return CommandResult(argv, 0, sha + "\n", "")
        if argv[:2] == ("git", "-C") and argv[-2:] == ("status", "--porcelain"):
            return CommandResult(argv, 0, "", "")
        if argv[:3] == ("docker", "image", "inspect"):
            return CommandResult(argv, 0, self.config.lab2_image_id + "\n", "")
        if argv[:2] == ("uv", "lock"):
            return CommandResult(argv, 0, "", "")
        if argv and argv[0] == "nvidia-smi":
            return CommandResult(argv, 0, "100, 0\n", "")
        if "MetricsFormatter.get_instance" in rendered:
            stdout = self.lab2_probe_stdout if argv[0] == "docker" else self.lab3_probe_stdout
            return CommandResult(argv, 0, stdout, "")
        raise AssertionError(f"unexpected preflight command: {rendered}")


def test_preflight_validates_all_required_system_identities(tmp_path: Path):
    config = _config(tmp_path)
    commands = _PreflightCommands(config)

    result = run_preflight(config, commands, min_free_bytes=1)

    rendered = [" ".join(argv) for argv in commands.argvs]
    assert result.idle_memory_baseline_mib == 100
    assert result.uv_lock_sha256 == hashlib.sha256(b"locked\n").hexdigest()
    assert sum("git -C" in command and "rev-parse HEAD" in command for command in rendered) == 2
    assert sum("git -C" in command and "status --porcelain" in command for command in rendered) == 2
    assert any("docker image inspect" in command for command in rendered)
    assert any("uv lock --check" in command for command in rendered)
    assert any(command.startswith("nvidia-smi") for command in rendered)
    assert len([command for command in rendered if "MetricsFormatter.get_instance" in command]) == 2


def test_preflight_rejects_lab2_noise_without_unique_sentinel(tmp_path: Path):
    config = _config(tmp_path)
    commands = _PreflightCommands(config, lab2_probe_stdout="kit startup log\nok\n")

    with pytest.raises(PreflightError, match="__ISAACLAB_BENCHMARK_PREFLIGHT_OK__"):
        run_preflight(config, commands, min_free_bytes=1)


def test_preflight_keeps_lab3_probe_stdout_strict(tmp_path: Path):
    config = _config(tmp_path)
    commands = _PreflightCommands(config, lab3_probe_stdout="kit startup log\nok\n")

    with pytest.raises(
        PreflightError,
        match=r"lab3 task registration and formatters failed: expected 'ok', got 'kit startup log\\nok'",
    ):
        run_preflight(config, commands, min_free_bytes=1)


def test_preflight_provenance_is_written_by_actual_executor_payloads(tmp_path: Path):
    """Validated lock/image/SHA identities flow into both version artifact payloads."""
    config = _config(tmp_path)
    preflight = run_preflight(config, _PreflightCommands(config), min_free_bytes=1)

    class Launcher:
        def run(self, _invocation, _timeout_s):
            return ProcessResult(returncode=0, stdout="", stderr="")

    lab2 = Lab2DockerExecutor(config, launcher=Launcher(), provenance=preflight.provenance).execute(
        _attempt(Version.LAB2)
    )
    lab3 = Lab3UvExecutor(config, launcher=Launcher(), provenance=preflight.provenance).execute(_attempt(Version.LAB3))

    expected_common = {
        "lab2_sha": LAB2_SHA,
        "lab3_sha": LAB3_SHA,
        "lab2_image_id": config.lab2_image_id,
        "uv_lock_sha256": hashlib.sha256(b"locked\n").hexdigest(),
    }
    for execution in (lab2, lab3):
        assert {key: execution.environment[key] for key in expected_common} == expected_common
    assert lab2.environment["environment_identity"] == config.lab2_image_id
    assert lab3.environment["environment_identity"] == f"uv-lock:{expected_common['uv_lock_sha256']}"


@pytest.mark.parametrize(
    "failure",
    ["rev-parse HEAD", "status --porcelain", "docker image inspect", "uv lock --check", "nvidia-smi"],
)
def test_preflight_stops_on_required_check_failure(tmp_path: Path, failure: str):
    config = _config(tmp_path)

    with pytest.raises(PreflightError, match="preflight"):
        run_preflight(config, _PreflightCommands(config, fail_contains=failure), min_free_bytes=1)
