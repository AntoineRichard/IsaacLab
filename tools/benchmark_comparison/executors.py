# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Version-specific benchmark commands, execution, and preflight checks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .models import BenchmarkAttempt, BoundUnit, Version
from .validate import attempt_identity


@dataclass(frozen=True)
class ExecutorConfig:
    """Pinned paths and identities used by both executors."""

    lab2_root: Path
    lab3_root: Path
    artifact_root: Path
    lab2_sha: str
    lab3_sha: str
    lab2_image: str
    lab2_image_id: str


@dataclass(frozen=True)
class Invocation:
    """One subprocess invocation that never uses shell evaluation."""

    argv: tuple[str, ...]
    environment: dict[str, str]
    cwd: Path
    shell: bool = False
    container_name: str | None = None


@dataclass(frozen=True)
class CommandResult:
    """Captured result of a short, non-simulator command."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class ProcessResult:
    """Captured outcome of a simulator child process."""

    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    interrupted: bool = False


@dataclass(frozen=True)
class PreflightResult:
    """Validated execution identities and idle memory baseline."""

    idle_memory_baseline_mib: int
    uv_lock_sha256: str
    free_disk_bytes: int


class PreflightError(RuntimeError):
    """Raised when a required executor preflight check fails."""


class CommandRunner(Protocol):
    """Interface used to make executor preflight deterministic in tests."""

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        environment: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        """Run one argument vector and capture its result."""


class ProcessGroupRegistry(Protocol):
    """Registry of process groups created by benchmark attempts."""

    def add(self, process_group_id: int) -> None:
        """Record an owned process group for later idle checks."""


class SystemCommandRunner:
    """Run short subprocesses without shell evaluation."""

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        environment: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        """Run one argument vector and capture its result."""
        completed = subprocess.run(
            list(argv),
            cwd=cwd,
            env=_merged_environment(environment),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            shell=False,
        )
        return CommandResult(tuple(argv), completed.returncode, completed.stdout, completed.stderr)


class ProcessLauncher:
    """Launch one process group and clean only its uniquely named container."""

    def __init__(
        self,
        commands: CommandRunner | None = None,
        terminate_grace_s: float = 30.0,
        owned_process_groups: ProcessGroupRegistry | None = None,
    ):
        self._commands = commands or SystemCommandRunner()
        self._terminate_grace_s = terminate_grace_s
        self._owned_process_groups = owned_process_groups

    def run(self, invocation: Invocation, timeout_s: float) -> ProcessResult:
        """Execute one invocation with timeout and interruption classification."""
        process = subprocess.Popen(
            list(invocation.argv),
            cwd=invocation.cwd,
            env=_merged_environment(invocation.environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            shell=False,
        )
        if self._owned_process_groups is not None:
            self._owned_process_groups.add(process.pid)
        previous_sigterm = signal.signal(signal.SIGTERM, _raise_interruption)
        try:
            try:
                stdout, stderr = process.communicate(timeout=timeout_s)
                return ProcessResult(process.returncode, stdout, stderr)
            except subprocess.TimeoutExpired:
                stdout, stderr = self._terminate(process)
                self._cleanup_container(invocation.container_name)
                return ProcessResult(process.returncode, stdout, stderr, timed_out=True)
            except KeyboardInterrupt:
                stdout, stderr = self._terminate(process)
                self._cleanup_container(invocation.container_name)
                return ProcessResult(process.returncode, stdout, stderr, interrupted=True)
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)

    def _terminate(self, process: subprocess.Popen[str]) -> tuple[str, str]:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            return process.communicate(timeout=self._terminate_grace_s)
        except subprocess.TimeoutExpired:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            return process.communicate()

    def _cleanup_container(self, container_name: str | None) -> None:
        if container_name is not None:
            self._commands.run(("docker", "rm", "--force", container_name), timeout=30)


class _Executor:
    """Shared implementation for native and container executors."""

    def __init__(
        self,
        config: ExecutorConfig,
        *,
        launcher: ProcessLauncher | None = None,
        timeout_s: float = 7200.0,
    ):
        self.config = config
        self._launcher = launcher or ProcessLauncher()
        self._timeout_s = timeout_s

    def invocation(self, attempt: BenchmarkAttempt, output_suffix: str | None = None) -> Invocation:
        """Build the measured command for ``attempt``."""
        raise NotImplementedError

    def version_invocation(self) -> Invocation:
        """Build the task-registration and formatter probe."""
        raise NotImplementedError

    def execute(self, attempt: BenchmarkAttempt):
        """Run one attempt and return runner artifact inputs."""
        from .runner import AttemptExecution

        output_suffix = f"{attempt.identity}-{uuid.uuid4().hex}"
        output_path = self.config.artifact_root / ".outputs" / output_suffix
        output_path.mkdir(parents=True, exist_ok=False)
        invocation = self.invocation(attempt, output_suffix)
        started = datetime.now(timezone.utc)
        start_monotonic = time.monotonic()
        result = self._launcher.run(invocation, self._timeout_s)
        ended = datetime.now(timezone.utc)
        output = result.stdout + "\n" + result.stderr
        identity = attempt_identity(attempt)
        return AttemptExecution(
            command={
                "identity": identity,
                "argv": list(invocation.argv),
                "cwd": str(invocation.cwd),
                "shell": False,
            },
            environment={
                "identity": identity,
                "values": dict(sorted(invocation.environment.items())),
                "lab2_image": self.config.lab2_image if attempt.version is Version.LAB2 else None,
                "lab2_image_id": self.config.lab2_image_id if attempt.version is Version.LAB2 else None,
                "lab2_sha": self.config.lab2_sha,
                "lab3_sha": self.config.lab3_sha,
            },
            stdout=result.stdout,
            stderr=result.stderr,
            exit_status={
                "exit_code": result.returncode,
                "failure_stage": None,
                "timed_out": result.timed_out,
                "interrupted": result.interrupted,
                "out_of_memory": _looks_like_out_of_memory(output),
                "started_at": started.isoformat(),
                "ended_at": ended.isoformat(),
                "wall_time_s": time.monotonic() - start_monotonic,
            },
            schema=_load_single_json(output_path, "*_schema.json"),
            measurements=_load_single_json(output_path, "*_json.json"),
        )

    def _environment(self) -> dict[str, str]:
        return {
            "ISAACLAB_BENCHMARK_LAB2_SHA": self.config.lab2_sha,
            "ISAACLAB_BENCHMARK_LAB3_SHA": self.config.lab3_sha,
            "OMNI_KIT_ACCEPT_EULA": "yes",
        }

    @staticmethod
    def _benchmark_arguments(attempt: BenchmarkAttempt, output_path: str) -> tuple[str, ...]:
        common = (
            "--output_path",
            output_path,
            "--task",
            attempt.concrete_task,
            "--num_envs",
            str(attempt.num_envs),
            "--seed",
            str(attempt.seed),
        )
        if attempt.bound.unit is BoundUnit.STEPS:
            bounded = ("--num_frames", str(attempt.bound.value))
        else:
            bounded = ("--rl_library", attempt.framework, "--max_iterations", str(attempt.bound.value))
        return (*common, *bounded, "--benchmark_formatter", "schema,json", "presets=physx", "--headless")


class Lab2DockerExecutor(_Executor):
    """Run the pinned Lab 2 worktree through its local benchmark image."""

    def invocation(self, attempt: BenchmarkAttempt, output_suffix: str | None = None) -> Invocation:
        """Build a Docker Compose argument vector for one Lab 2 attempt."""
        if attempt.version is not Version.LAB2:
            raise ValueError("Lab2DockerExecutor requires a lab2 attempt")
        suffix = output_suffix or attempt.identity
        container_name = f"isaaclab-benchmark-{_safe_token(suffix)[:40]}"
        environment = self._environment()
        argv = [
            "docker",
            "compose",
            "--env-file",
            str(self.config.lab2_root / "docker/.env.base"),
            "-f",
            str(self.config.lab2_root / "docker/docker-compose.yaml"),
            "-f",
            str(self.config.lab2_root / "tools/benchmark_comparison/docker-compose.benchmark.yaml"),
            "run",
            "--rm",
            "--no-deps",
            "-T",
            "--name",
            container_name,
        ]
        for name, value in sorted(environment.items()):
            argv.extend(("-e", f"{name}={value}"))
        argv.extend(
            (
                "isaac-lab-benchmark",
                "/workspace/isaaclab/isaaclab.sh",
                "-p",
                f"/workspace/isaaclab/scripts/benchmarks/{_script_name(attempt)}",
                *self._benchmark_arguments(attempt, f"/benchmark_artifacts/.outputs/{suffix}"),
            )
        )
        compose_environment = {
            **environment,
            "ISAACLAB_BENCHMARK_ARTIFACT_ROOT": str(self.config.artifact_root),
            "ISAACLAB_BENCHMARK_IMAGE_ID": self.config.lab2_image_id,
        }
        return Invocation(tuple(argv), compose_environment, self.config.lab2_root, container_name=container_name)

    def version_invocation(self) -> Invocation:
        """Build the Lab 2 registration and formatter probe."""
        environment = self._environment()
        argv = (
            "docker",
            "compose",
            "--env-file",
            str(self.config.lab2_root / "docker/.env.base"),
            "-f",
            str(self.config.lab2_root / "docker/docker-compose.yaml"),
            "-f",
            str(self.config.lab2_root / "tools/benchmark_comparison/docker-compose.benchmark.yaml"),
            "run",
            "--rm",
            "--no-deps",
            "-T",
            "isaac-lab-benchmark",
            "/workspace/isaaclab/isaaclab.sh",
            "-p",
            "-c",
            _registration_probe(Version.LAB2),
        )
        return Invocation(
            argv,
            {
                **environment,
                "ISAACLAB_BENCHMARK_ARTIFACT_ROOT": str(self.config.artifact_root),
                "ISAACLAB_BENCHMARK_IMAGE_ID": self.config.lab2_image_id,
            },
            self.config.lab2_root,
        )


class Lab3UvExecutor(_Executor):
    """Run the pinned Lab 3 worktree through its locked uv environment."""

    def _prefix(self) -> tuple[str, ...]:
        return (
            "uv",
            "run",
            "--project",
            str(self.config.lab3_root),
            "--extra",
            "isaacsim",
            "--extra",
            "rsl-rl",
            "--locked",
        )

    def invocation(self, attempt: BenchmarkAttempt, output_suffix: str | None = None) -> Invocation:
        """Build a locked uv argument vector for one Lab 3 attempt."""
        if attempt.version is not Version.LAB3:
            raise ValueError("Lab3UvExecutor requires a lab3 attempt")
        suffix = output_suffix or attempt.identity
        argv = (
            *self._prefix(),
            "python",
            str(self.config.lab3_root / "scripts/benchmarks" / _script_name(attempt)),
            *self._benchmark_arguments(attempt, str(self.config.artifact_root / ".outputs" / suffix)),
        )
        return Invocation(argv, self._environment(), self.config.lab3_root)

    def version_invocation(self) -> Invocation:
        """Build the Lab 3 registration and formatter probe."""
        return Invocation(
            (*self._prefix(), "python", "-c", _registration_probe(Version.LAB3)),
            self._environment(),
            self.config.lab3_root,
        )


def run_preflight(
    config: ExecutorConfig,
    commands: CommandRunner | None = None,
    *,
    min_free_bytes: int = 10 * 1024**3,
) -> PreflightResult:
    """Validate all immutable inputs before any measured attempt."""
    command_runner = commands or SystemCommandRunner()
    for root, expected_sha, name in (
        (config.lab2_root, config.lab2_sha, "lab2"),
        (config.lab3_root, config.lab3_sha, "lab3"),
    ):
        _require_command(
            command_runner,
            ("git", "-C", str(root), "rev-parse", "HEAD"),
            expected_stdout=expected_sha,
            description=f"{name} exact Git SHA",
        )
        _require_command(
            command_runner,
            ("git", "-C", str(root), "status", "--porcelain"),
            expected_stdout="",
            description=f"{name} clean worktree",
        )
    _require_command(
        command_runner,
        ("docker", "image", "inspect", "--format", "{{.Id}}", config.lab2_image),
        expected_stdout=config.lab2_image_id,
        description="Docker image identity",
    )
    _require_command(
        command_runner,
        ("uv", "lock", "--check", "--project", str(config.lab3_root)),
        description="uv lock state",
    )
    nvidia = _require_command(
        command_runner,
        ("nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"),
        description="NVIDIA SMI access",
    )
    idle_memory = _parse_idle_memory(nvidia.stdout)
    try:
        config.artifact_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=config.artifact_root):
            pass
        free_disk = shutil.disk_usage(config.artifact_root).free
    except OSError as error:
        raise PreflightError(f"preflight artifact root is not writable: {error}") from error
    if free_disk < min_free_bytes:
        raise PreflightError(f"preflight free disk is {free_disk} bytes; require {min_free_bytes}")
    for name, invocation in (
        ("lab2", Lab2DockerExecutor(config).version_invocation()),
        ("lab3", Lab3UvExecutor(config).version_invocation()),
    ):
        _require_command(
            command_runner,
            invocation.argv,
            cwd=invocation.cwd,
            environment=invocation.environment,
            expected_stdout="ok",
            description=f"{name} task registration and formatters",
        )
    lock_bytes = (config.lab3_root / "uv.lock").read_bytes()
    return PreflightResult(
        idle_memory_baseline_mib=idle_memory,
        uv_lock_sha256=hashlib.sha256(lock_bytes).hexdigest(),
        free_disk_bytes=free_disk,
    )


def _require_command(
    commands: CommandRunner,
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
    expected_stdout: str | None = None,
    description: str,
) -> CommandResult:
    try:
        result = commands.run(argv, cwd=cwd, environment=environment, timeout=120)
    except (OSError, subprocess.SubprocessError) as error:
        raise PreflightError(f"preflight {description} failed: {error}") from error
    if result.returncode != 0:
        raise PreflightError(f"preflight {description} failed: {result.stderr.strip()}")
    if expected_stdout is not None and result.stdout.strip() != expected_stdout:
        raise PreflightError(
            f"preflight {description} failed: expected {expected_stdout!r}, got {result.stdout.strip()!r}"
        )
    return result


def _parse_idle_memory(stdout: str) -> int:
    try:
        return max(int(line.split(",", maxsplit=1)[0].strip()) for line in stdout.splitlines() if line.strip())
    except (ValueError, IndexError) as error:
        raise PreflightError(f"preflight NVIDIA SMI output is malformed: {stdout!r}") from error


def _script_name(attempt: BenchmarkAttempt) -> str:
    return "runtime.py" if attempt.bound.unit is BoundUnit.STEPS else "training.py"


def _registration_probe(version: Version) -> str:
    from .matrix import load_matrix

    task_ids = tuple(task.concrete_id(version) for task in load_matrix().tasks)
    return (
        "from isaaclab.test.benchmark.formatters import MetricsFormatter;"
        "MetricsFormatter.get_instance('schema');"
        "MetricsFormatter.get_instance('json');"
        "import gymnasium as gym, isaaclab_tasks;"
        f"assert all(task_id in gym.registry for task_id in {task_ids!r});"
        "print('ok')"
    )


def _safe_token(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value).strip("-")


def _load_single_json(directory: Path, pattern: str) -> object:
    matches = tuple(directory.glob(pattern))
    if len(matches) != 1:
        return None
    try:
        return json.loads(matches[0].read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _looks_like_out_of_memory(output: str) -> bool:
    lowered = output.lower()
    return "out of memory" in lowered or "cuda_error_out_of_memory" in lowered


def _raise_interruption(_signal_number: int, _frame: object) -> None:
    raise KeyboardInterrupt


def _merged_environment(environment: Mapping[str, str] | None) -> dict[str, str]:
    merged = os.environ.copy()
    if environment is not None:
        merged.update(environment)
    return merged
