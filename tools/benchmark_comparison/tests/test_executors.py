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
    _merged_environment,
    run_preflight,
)
from tools.benchmark_comparison.matrix import expand_final_matrix, load_matrix
from tools.benchmark_comparison.models import RunSet, Version

LAB2_SHA = "28a9560c59df2306690ea717d6cf36f1e63c66e3"
LAB3_SHA = "cb508381fb4874ce7afffeb9197bd91c20db7dad"
GPU_UUID = "GPU-01234567-89ab-cdef-0123-456789abcdef"


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


def _attempt(version: Version, mode: str = "runtime-100", task: str = "cartpole"):
    return next(
        attempt
        for attempt in expand_final_matrix(load_matrix()).attempts
        if attempt.version is version and attempt.mode.id == mode and attempt.logical_task == task
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
    assert invocation.argv[-14:] == (
        "--task",
        attempt.concrete_task,
        "--num_envs",
        "4096",
        "--seed",
        "42",
        "--device",
        "cuda:0",
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


@pytest.mark.parametrize(
    ("version", "executor_type"),
    [(Version.LAB2, Lab2DockerExecutor), (Version.LAB3, Lab3UvExecutor)],
)
def test_measured_commands_pin_physical_gpu_zero(
    tmp_path: Path, version: Version, executor_type: type[Lab2DockerExecutor] | type[Lab3UvExecutor]
) -> None:
    invocation = executor_type(_config(tmp_path), selected_gpu_uuid=GPU_UUID).invocation(_attempt(version))

    assert invocation.argv[invocation.argv.index("--device") : invocation.argv.index("--device") + 2] == (
        "--device",
        "cuda:0",
    )
    assert invocation.environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert invocation.environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert invocation.environment["NVIDIA_VISIBLE_DEVICES"] == "0"
    assert invocation.environment["ISAACLAB_BENCHMARK_GPU_INDEX"] == "0"
    assert invocation.environment["ISAACLAB_BENCHMARK_GPU_UUID"] == GPU_UUID


def test_subprocess_environment_drops_inherited_distributed_and_gpu_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inherited = {
        "CUDA_VISIBLE_DEVICES": "3,2",
        "NVIDIA_VISIBLE_DEVICES": "all",
        "RANK": "7",
        "LOCAL_RANK": "1",
        "WORLD_SIZE": "8",
        "MASTER_ADDR": "example.invalid",
        "MASTER_PORT": "1234",
        "TORCHELASTIC_RUN_ID": "inherited",
        "OMPI_COMM_WORLD_RANK": "4",
        "SLURM_PROCID": "5",
        "JAX_RANK": "6",
        "JAX_LOCAL_RANK": "2",
    }
    for name, value in inherited.items():
        monkeypatch.setenv(name, value)

    merged = _merged_environment(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "0",
            "NVIDIA_VISIBLE_DEVICES": "0",
        }
    )

    assert merged["CUDA_VISIBLE_DEVICES"] == "0"
    assert merged["NVIDIA_VISIBLE_DEVICES"] == "0"
    explicitly_replaced = {"CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"}
    assert all(name not in merged for name in inherited if name not in explicitly_replaced)


def test_lab3_omits_redundant_physx_alias_only_for_default_physx_tasks(tmp_path: Path):
    config = _config(tmp_path)
    executor = Lab3UvExecutor(config)

    g1 = executor.invocation(_attempt(Version.LAB3, task="g1_flat"))
    allegro = executor.invocation(_attempt(Version.LAB3, task="allegro_cube"))
    cartpole = executor.invocation(_attempt(Version.LAB3, task="cartpole"))
    lab2_g1 = Lab2DockerExecutor(config).invocation(_attempt(Version.LAB2, task="g1_flat"))

    assert "presets=physx" not in g1.argv
    assert "presets=physx" not in allegro.argv
    assert "presets=physx" in cartpole.argv
    assert "presets=physx" in lab2_g1.argv


def test_rgb_cartpole_runtime_commands_enable_cameras_and_select_kit_rgb(tmp_path: Path) -> None:
    config = _config(tmp_path)
    lab2 = Lab2DockerExecutor(config).invocation(_attempt(Version.LAB2, task="cartpole_rgb_kit"))
    lab3 = Lab3UvExecutor(config).invocation(_attempt(Version.LAB3, task="cartpole_rgb_kit"))

    assert "--enable_cameras" in lab2.argv
    assert "--enable_cameras" in lab3.argv
    assert "presets=physx" in lab2.argv
    assert "presets=physx,rgb" in lab3.argv
    assert "presets=physx" not in lab3.argv
    assert all("newton_renderer" not in argument for argument in lab3.argv)


def test_non_camera_tasks_keep_camera_flags_and_extra_presets_disabled(tmp_path: Path) -> None:
    config = _config(tmp_path)

    for version, executor in (
        (Version.LAB2, Lab2DockerExecutor(config)),
        (Version.LAB3, Lab3UvExecutor(config)),
    ):
        invocation = executor.invocation(_attempt(version, task="cartpole"))
        assert "--enable_cameras" not in invocation.argv
        assert all(argument != "presets=physx,rgb" for argument in invocation.argv)


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


def test_lab2_version_probe_forwards_gpu_zero_selection_into_container(tmp_path: Path) -> None:
    invocation = Lab2DockerExecutor(_config(tmp_path), selected_gpu_uuid=GPU_UUID).version_invocation()
    forwarded = {invocation.argv[index + 1] for index, argument in enumerate(invocation.argv[:-1]) if argument == "-e"}

    assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in forwarded
    assert "CUDA_VISIBLE_DEVICES=0" in forwarded
    assert "NVIDIA_VISIBLE_DEVICES=0" in forwarded
    assert "ISAACLAB_BENCHMARK_GPU_UUID=" + GPU_UUID in forwarded


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

    app_launcher = "AppLauncher(headless=True, enable_cameras=True)"
    assert lab2_probe.index(app_launcher) < lab2_probe.index("MetricsFormatter")
    assert lab2_probe.index(app_launcher) < lab2_probe.index("isaaclab_tasks")
    assert "__ISAACLAB_BENCHMARK_PREFLIGHT_OK__" in lab2_probe
    assert "flush=True" in lab2_probe
    assert "AppLauncher" not in lab3_probe
    assert lab2_probe.index("flush=True") < lab2_probe.index("simulation_app.close()")
    assert "__ISAACLAB_BENCHMARK_PREFLIGHT_OK__" not in lab3_probe


def test_version_probes_require_rsl_only_for_training_tasks_and_validate_rgb_camera(tmp_path: Path) -> None:
    config = _config(tmp_path)

    lab2_probe = Lab2DockerExecutor(config).version_invocation().argv[-1]
    lab3_probe = Lab3UvExecutor(config).version_invocation().argv[-1]

    assert "AppLauncher(headless=True, enable_cameras=True)" in lab2_probe
    assert "if supports_training" in lab2_probe
    assert "if supports_training" in lab3_probe
    assert "rsl_rl_cfg_entry_point" in lab2_probe
    assert "rsl_rl_cfg_entry_point" in lab3_probe
    assert "Isaac-Cartpole-RGB-v0" in lab2_probe
    assert "Isaac-Cartpole-Camera" in lab3_probe
    assert "('Isaac-Cartpole-Camera', False, True, ('rgb',))" in lab3_probe
    assert "presets={','.join(presets)}" in lab3_probe
    assert "env_cfg.scene.tiled_camera.data_types == ['rgb']" in lab2_probe
    assert "env_cfg.scene.tiled_camera.data_types == ['rgb']" in lab3_probe
    assert "IsaacRtxRendererCfg" in lab3_probe


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
    assert 'device_ids: ["0"]' in compose
    assert "count: all" not in compose


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
            return CommandResult(argv, 0, f"Fixture GPU, 590.48.01, 100, 0, {GPU_UUID}\n", "")
        if "ISAACLAB_BENCHMARK_HOST_METADATA" in rendered:
            return CommandResult(
                argv,
                0,
                (
                    '__ISAACLAB_BENCHMARK_METADATA__{"hostname":"fixture-host","os":"Fixture OS",'
                    '"cpu_model":"Fixture CPU","logical_cpu_count":32}\n'
                ),
                "",
            )
        if "ISAACLAB_BENCHMARK_SOFTWARE_METADATA" in rendered:
            version = "lab2" if argv[0] == "docker" else "lab3"
            value = (
                ('{"isaac_lab":"2.3.2","isaac_sim":"5.1","python":"3.11","pytorch":"2.7","rsl_rl":"5.0","cuda":"12.8"}')
                if version == "lab2"
                else (
                    '{"isaac_lab":"3.0.0","isaac_sim":"6.0","python":"3.12",'
                    '"pytorch":"2.8","rsl_rl":"5.4","cuda":"12.8"}'
                )
            )
            return CommandResult(argv, 0, "__ISAACLAB_BENCHMARK_METADATA__" + value + "\n", "")
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
    expansion = expand_final_matrix(load_matrix())
    manifest = result.manifest(RunSet.FINAL, "measured", expansion)
    assert manifest.schema_version == "2.0"
    assert manifest.host.gpu_model == "Fixture GPU"
    assert manifest.host.gpu_driver == "590.48.01"
    assert manifest.host.gpu_index == 0
    assert manifest.host.gpu_uuid == GPU_UUID
    assert manifest.expansion == expansion
    assert manifest.host.cpu_model == "Fixture CPU"
    assert manifest.lab2.python == "3.11"
    assert manifest.lab3.rsl_rl == "5.4"
    assert sum("git -C" in command and "rev-parse HEAD" in command for command in rendered) == 2
    assert sum("git -C" in command and "status --porcelain" in command for command in rendered) == 2
    assert any("docker image inspect" in command for command in rendered)
    assert any("uv lock --check" in command for command in rendered)
    assert any(command.startswith("nvidia-smi") for command in rendered)
    nvidia_command = next(argv for argv in commands.argvs if argv and argv[0] == "nvidia-smi")
    assert "--id=0" in nvidia_command
    assert any("uuid" in argument for argument in nvidia_command)
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

    lab2 = Lab2DockerExecutor(
        config,
        launcher=Launcher(),
        provenance=preflight.provenance,
        selected_gpu_uuid=preflight.host.gpu_uuid,
    ).execute(_attempt(Version.LAB2))
    lab3 = Lab3UvExecutor(
        config,
        launcher=Launcher(),
        provenance=preflight.provenance,
        selected_gpu_uuid=preflight.host.gpu_uuid,
    ).execute(_attempt(Version.LAB3))

    expected_common = {
        "lab2_sha": LAB2_SHA,
        "lab3_sha": LAB3_SHA,
        "lab2_image_id": config.lab2_image_id,
        "uv_lock_sha256": hashlib.sha256(b"locked\n").hexdigest(),
    }
    for execution in (lab2, lab3):
        assert {key: execution.environment[key] for key in expected_common} == expected_common
        assert execution.environment["selected_gpu"] == {"physical_index": 0, "uuid": GPU_UUID}
        assert execution.environment["values"]["CUDA_VISIBLE_DEVICES"] == "0"
        assert execution.environment["values"]["NVIDIA_VISIBLE_DEVICES"] == "0"
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
