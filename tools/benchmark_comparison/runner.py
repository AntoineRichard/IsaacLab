# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Host-idle enforcement and serialized, resumable benchmark execution."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from .artifacts import finalize_attempt, verify_success
from .models import BenchmarkAttempt, MatrixExpansion
from .validate import attempt_identity


@dataclass(frozen=True)
class AttemptExecution:
    """Captured source documents needed to finalize one attempt."""

    command: object
    environment: object
    stdout: str
    stderr: str
    exit_status: object
    schema: object
    measurements: object


@dataclass(frozen=True)
class IdleSample:
    """One raw host/GPU observation."""

    index: int
    gpu_utilization_pct: int
    gpu_memory_mib: int
    load_1m: float


@dataclass(frozen=True)
class IdleInventory:
    """Processes that can invalidate an idle evaluation."""

    nvidia_compute_processes: tuple[int, ...]
    gpu_container_ids: tuple[str, ...]
    prior_child_pids: tuple[int, ...]


@dataclass(frozen=True)
class IdleThresholds:
    """Configurable idle acceptance thresholds."""

    gpu_utilization_max_pct: int = 5
    gpu_memory_headroom_mib: int = 1024
    host_load_cpu_fraction: float = 0.25
    sample_count: int = 60
    sample_interval_s: float = 1.0
    retry_interval_s: float = 300.0
    timeout_s: float = 3600.0


class IdleGateTimeout(RuntimeError):
    """Raised when the host does not become idle before timeout."""


class IdleMonitor(Protocol):
    """Source of process inventory and raw idle samples."""

    def inventory(self) -> IdleInventory:
        """Capture processes before one idle evaluation."""

    def sample(self) -> IdleSample:
        """Capture one GPU and host-load sample."""


class Clock(Protocol):
    """Injectable monotonic clock and sleeper."""

    def monotonic(self) -> float:
        """Return monotonic seconds."""

    def sleep(self, seconds: float) -> None:
        """Wait for ``seconds``."""


class SystemClock:
    """System monotonic clock implementation."""

    monotonic = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)


class SystemIdleMonitor:
    """Capture host-idle evidence using system CLIs and ``/proc``."""

    def inventory(self) -> IdleInventory:
        """Capture compute processes, GPU containers, and prior children."""
        compute = _command_lines(("nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"))
        return IdleInventory(
            nvidia_compute_processes=tuple(int(line) for line in compute if line.isdigit()),
            gpu_container_ids=_gpu_container_ids(),
            prior_child_pids=_direct_child_pids(),
        )

    def sample(self) -> IdleSample:
        """Capture maximum GPU utilization/memory and one-minute host load."""
        lines = _command_lines(
            (
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            )
        )
        values = [tuple(int(part.strip()) for part in line.split(",", maxsplit=1)) for line in lines]
        if not values:
            raise RuntimeError("nvidia-smi returned no GPU samples")
        return IdleSample(
            index=0,
            gpu_utilization_pct=max(value[0] for value in values),
            gpu_memory_mib=max(value[1] for value in values),
            load_1m=os.getloadavg()[0],
        )


class HostIdleGate:
    """Wait for a 60-sample idle window and persist every evaluation."""

    def __init__(
        self,
        *,
        monitor: IdleMonitor,
        clock: Clock,
        evidence_root: Path,
        idle_memory_baseline_mib: int,
        logical_cpu_count: int,
        thresholds: IdleThresholds | None = None,
    ):
        self.monitor = monitor
        self.clock = clock
        self.evidence_root = evidence_root
        self.idle_memory_baseline_mib = idle_memory_baseline_mib
        self.logical_cpu_count = logical_cpu_count
        self.thresholds = thresholds or IdleThresholds()

    def wait(self, attempt_identity_value: str) -> Path:
        """Wait until all idle criteria pass or timeout expires."""
        started = self.clock.monotonic()
        evaluation = 0
        while True:
            evaluation += 1
            inventory = self.monitor.inventory()
            samples: list[IdleSample] = []
            for index in range(self.thresholds.sample_count):
                raw = self.monitor.sample()
                samples.append(
                    IdleSample(
                        index=index,
                        gpu_utilization_pct=raw.gpu_utilization_pct,
                        gpu_memory_mib=raw.gpu_memory_mib,
                        load_1m=raw.load_1m,
                    )
                )
                if index + 1 < self.thresholds.sample_count:
                    self.clock.sleep(self.thresholds.sample_interval_s)
            reasons = self._rejection_reasons(inventory, samples)
            evidence = self._persist_evidence(attempt_identity_value, evaluation, inventory, samples, reasons)
            if not reasons:
                return evidence
            elapsed = self.clock.monotonic() - started
            if elapsed + self.thresholds.retry_interval_s > self.thresholds.timeout_s:
                raise IdleGateTimeout(f"host idle gate timed out after {elapsed:.1f}s; latest evidence: {evidence}")
            self.clock.sleep(self.thresholds.retry_interval_s)

    def _rejection_reasons(self, inventory: IdleInventory, samples: Sequence[IdleSample]) -> list[str]:
        reasons: list[str] = []
        if inventory.nvidia_compute_processes:
            reasons.append("nvidia_compute_process")
        if inventory.gpu_container_ids:
            reasons.append("gpu_container")
        if inventory.prior_child_pids:
            reasons.append("prior_child")
        if any(sample.gpu_utilization_pct > self.thresholds.gpu_utilization_max_pct for sample in samples):
            reasons.append("gpu_utilization")
        memory_limit = self.idle_memory_baseline_mib + self.thresholds.gpu_memory_headroom_mib
        if any(sample.gpu_memory_mib > memory_limit for sample in samples):
            reasons.append("gpu_memory")
        load_limit = self.logical_cpu_count * self.thresholds.host_load_cpu_fraction
        if any(sample.load_1m > load_limit for sample in samples):
            reasons.append("host_load")
        return reasons

    def _persist_evidence(
        self,
        attempt_identity_value: str,
        evaluation: int,
        inventory: IdleInventory,
        samples: Sequence[IdleSample],
        reasons: Sequence[str],
    ) -> Path:
        self.evidence_root.mkdir(parents=True, exist_ok=True)
        path = self.evidence_root / f"{attempt_identity_value}-idle-{evaluation:04d}.json"
        load_limit = self.logical_cpu_count * self.thresholds.host_load_cpu_fraction
        _write_json_atomic(
            path,
            {
                "attempt_identity": attempt_identity_value,
                "evaluation": evaluation,
                "decision": "rejected" if reasons else "accepted",
                "reasons": list(reasons),
                "inventory": asdict(inventory),
                "samples": [asdict(sample) for sample in samples],
                "idle_memory_baseline_mib": self.idle_memory_baseline_mib,
                "thresholds": {
                    **asdict(self.thresholds),
                    "load_1m_max": load_limit,
                    "gpu_memory_max_mib": (self.idle_memory_baseline_mib + self.thresholds.gpu_memory_headroom_mib),
                },
            },
        )
        return path


class RunStatus(str, Enum):
    """Terminal status of one run-set invocation."""

    COMPLETED = "completed"
    COMPLETED_WITH_FAILURES = "completed_with_failures"
    INTERRUPTED = "interrupted"
    PREFLIGHT_FAILED = "preflight_failed"


@dataclass(frozen=True)
class RunResult:
    """Summary and durable state location for one invocation."""

    status: RunStatus
    succeeded: int
    failed: int
    skipped: int
    state_path: Path


class AttemptExecutor(Protocol):
    """Version-specific attempt execution interface."""

    def execute(self, attempt: BenchmarkAttempt) -> AttemptExecution:
        """Execute exactly one matrix attempt."""


class AttemptIdleGate(Protocol):
    """Idle gate used immediately before every measured attempt."""

    def wait(self, identity: str) -> Path:
        """Wait for idle and return its evidence path."""


class BenchmarkRunner:
    """Execute one attempt at a time with immutable resume semantics."""

    def __init__(
        self,
        *,
        artifact_root: Path,
        executors: Mapping[str, AttemptExecutor],
        idle_gate: AttemptIdleGate,
    ):
        self.artifact_root = artifact_root
        self.executors = executors
        self.idle_gate = idle_gate
        self.after_persist = None

    def run(self, expansion: MatrixExpansion, *, retry_failures: bool = False) -> RunResult:
        """Run a deterministic expansion, resuming trustworthy results."""
        state_path = self.artifact_root / expansion.run_set.value / "runner-state.json"
        state = _load_state(state_path, expansion)
        succeeded = failed = skipped = 0
        for attempt in expansion.attempts:
            attempt_root = self.artifact_root / attempt.run_directory
            success = attempt_root / "success"
            if success.exists():
                if _valid_success(success, attempt):
                    skipped += 1
                    _append_history(state, attempt, "skipped_success")
                    self._persist(state_path, state)
                    continue
                _quarantine_corrupt_success(success)
            prior_failures = tuple(attempt_root.glob("attempt-[0-9][0-9][0-9][0-9]-*"))
            if prior_failures and not retry_failures:
                failed += 1
                skipped += 1
                _append_history(state, attempt, "skipped_failure")
                self._persist(state_path, state)
                continue
            try:
                idle_evidence = self.idle_gate.wait(attempt.identity)
            except IdleGateTimeout as error:
                state["status"] = RunStatus.PREFLIGHT_FAILED.value
                state["preflight_failure"] = str(error)
                self._persist(state_path, state)
                return RunResult(RunStatus.PREFLIGHT_FAILED, succeeded, failed, skipped, state_path)
            executor = self.executors[attempt.version.value]
            try:
                execution = executor.execute(attempt)
            except KeyboardInterrupt:
                execution = _interrupted_execution(attempt)
                final_path = _finalize(self.artifact_root, attempt, execution)
                failed += 1
                _append_history(
                    state,
                    attempt,
                    "interrupted",
                    artifact=str(final_path),
                    idle_evidence=str(idle_evidence),
                )
                state["status"] = RunStatus.INTERRUPTED.value
                self._persist(state_path, state)
                return RunResult(RunStatus.INTERRUPTED, succeeded, failed, skipped, state_path)
            except (OSError, subprocess.SubprocessError) as error:
                execution = _launch_failure_execution(attempt, error)
            final_path = _finalize(self.artifact_root, attempt, execution)
            status = _artifact_status(final_path)
            if status == "success":
                succeeded += 1
            else:
                failed += 1
            _append_history(
                state,
                attempt,
                status,
                artifact=str(final_path),
                idle_evidence=str(idle_evidence),
            )
            self._persist(state_path, state)
            if status == "interrupted":
                state["status"] = RunStatus.INTERRUPTED.value
                self._persist(state_path, state)
                return RunResult(RunStatus.INTERRUPTED, succeeded, failed, skipped, state_path)
        final_status = RunStatus.COMPLETED_WITH_FAILURES if failed else RunStatus.COMPLETED
        state["status"] = final_status.value
        self._persist(state_path, state)
        return RunResult(final_status, succeeded, failed, skipped, state_path)

    def _persist(self, state_path: Path, state: dict[str, object]) -> None:
        _write_json_atomic(state_path, state)
        if self.after_persist is not None:
            self.after_persist(state_path)


def _valid_success(path: Path, attempt: BenchmarkAttempt) -> bool:
    return verify_success(path, attempt)


def _quarantine_corrupt_success(success: Path) -> Path:
    attempt_root = success.parent
    index = 1
    while (attempt_root / f"corrupt-success-{index:04d}").exists():
        index += 1
    destination = attempt_root / f"corrupt-success-{index:04d}"
    os.rename(success, destination)
    return destination


def _interrupted_execution(attempt: BenchmarkAttempt) -> AttemptExecution:
    identity = attempt_identity(attempt)
    return AttemptExecution(
        command={"identity": identity, "argv": []},
        environment={"identity": identity, "values": {}},
        stdout="",
        stderr="benchmark interrupted",
        exit_status={
            "exit_code": None,
            "failure_stage": None,
            "timed_out": False,
            "interrupted": True,
            "out_of_memory": False,
        },
        schema=None,
        measurements=None,
    )


def _launch_failure_execution(attempt: BenchmarkAttempt, error: BaseException) -> AttemptExecution:
    identity = attempt_identity(attempt)
    return AttemptExecution(
        command={"identity": identity, "argv": []},
        environment={"identity": identity, "values": {}},
        stdout="",
        stderr=str(error),
        exit_status={
            "exit_code": None,
            "failure_stage": "launch",
            "timed_out": False,
            "interrupted": False,
            "out_of_memory": False,
        },
        schema=None,
        measurements=None,
    )


def _finalize(root: Path, attempt: BenchmarkAttempt, execution: AttemptExecution) -> Path:
    return finalize_attempt(
        root,
        attempt,
        command=execution.command,
        environment=execution.environment,
        stdout=execution.stdout,
        stderr=execution.stderr,
        exit_status=execution.exit_status,
        schema=execution.schema,
        measurements=execution.measurements,
    )


def _artifact_status(path: Path) -> str:
    if path.name == "success":
        return "success"
    return path.name.split("-", maxsplit=2)[-1]


def _load_state(path: Path, expansion: MatrixExpansion) -> dict[str, object]:
    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(state, dict) and isinstance(state.get("history"), list):
                return state
        except (OSError, UnicodeError, json.JSONDecodeError):
            corrupt = path.with_name(f"{path.stem}-corrupt-{time.time_ns()}.json")
            shutil.copy2(path, corrupt)
    return {"run_set": expansion.run_set.value, "status": "running", "history": []}


def _append_history(
    state: dict[str, object],
    attempt: BenchmarkAttempt,
    status: str,
    *,
    artifact: str | None = None,
    idle_evidence: str | None = None,
) -> None:
    history = state.setdefault("history", [])
    if not isinstance(history, list):
        raise RuntimeError("runner state history is not a list")
    history.append(
        {
            "attempt_identity": attempt.identity,
            "attempt_order": attempt.attempt_order,
            "version": attempt.version.value,
            "status": status,
            "artifact": artifact,
            "idle_evidence": idle_evidence,
        }
    )


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _command_lines(argv: Sequence[str]) -> tuple[str, ...]:
    result = subprocess.run(list(argv), text=True, capture_output=True, check=False, shell=False)
    if result.returncode != 0:
        raise RuntimeError(f"{argv[0]} failed: {result.stderr.strip()}")
    return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())


def _gpu_container_ids() -> tuple[str, ...]:
    ids = _command_lines(("docker", "ps", "--quiet"))
    gpu_ids: list[str] = []
    for container_id in ids:
        result = subprocess.run(
            ("docker", "inspect", "--format", "{{json .HostConfig.DeviceRequests}}", container_id),
            text=True,
            capture_output=True,
            check=False,
            shell=False,
        )
        if result.returncode == 0 and result.stdout.strip() not in ("", "null", "[]"):
            gpu_ids.append(container_id)
    return tuple(gpu_ids)


def _direct_child_pids() -> tuple[int, ...]:
    children: list[int] = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / "stat").read_text(encoding="ascii").split()
            if int(fields[3]) == os.getpid():
                children.append(int(path.name))
        except (OSError, UnicodeError, ValueError, IndexError):
            continue
    return tuple(sorted(children))
