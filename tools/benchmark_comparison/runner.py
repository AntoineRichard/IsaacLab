# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Host-idle enforcement and serialized, resumable benchmark execution."""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from .artifacts import finalize_attempt, verify_success
from .models import BenchmarkAttempt, ExecutionProvenance, MatrixExpansion
from .validate import attempt_identity

_FAILED_ATTEMPT_PATTERN = re.compile(r"attempt-[0-9]+-[a-z_]+$")


class ControllerLockError(RuntimeError):
    """Raised when another benchmark controller owns the artifact root."""


class ControllerLock:
    """Nonblocking process lock shared by every run set under one artifact root."""

    def __init__(self, artifact_root: Path):
        self.artifact_root = artifact_root if _is_descriptor_path(artifact_root) else artifact_root.resolve()
        self.path = self.artifact_root / ".benchmark-controller.lock"
        self._file = None

    def __enter__(self) -> ControllerLock:
        """Acquire the artifact-root lock or fail without waiting."""
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        lock_file = self.path.open("a+b")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            lock_file.close()
            raise ControllerLockError(
                f"another benchmark controller is already active for artifact root: {self.artifact_root}"
            ) from error
        self._file = lock_file
        return self

    def __exit__(self, _exception_type: object, _exception: object, _traceback: object) -> None:
        """Release the controller lock after normal or exceptional completion."""
        if self._file is not None:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            self._file.close()
            self._file = None


def _is_descriptor_path(path: Path) -> bool:
    """Return whether ``path`` is a direct Linux procfs descriptor view."""
    return path.parent == Path("/proc/self/fd") and path.name.isdecimal()


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


class OwnedProcessGroups:
    """Track every process group launched by this controller."""

    def __init__(self):
        self._group_ids: set[int] = set()
        self._lock = threading.Lock()

    def add(self, process_group_id: int) -> None:
        """Record a benchmark-owned process group."""
        with self._lock:
            self._group_ids.add(process_group_id)

    def alive_pids(self) -> tuple[int, ...]:
        """Return all live PIDs in owned groups and forget empty groups."""
        with self._lock:
            grouped = _process_group_pids(self._group_ids)
            self._group_ids.intersection_update(grouped)
        return tuple(sorted(pid for pids in grouped.values() for pid in pids))


class SystemIdleMonitor:
    """Capture host-idle evidence using system CLIs and ``/proc``."""

    def __init__(self, owned_process_groups: OwnedProcessGroups | None = None):
        self._owned_process_groups = owned_process_groups or OwnedProcessGroups()

    def inventory(self) -> IdleInventory:
        """Capture compute processes, GPU containers, and prior children."""
        return IdleInventory(
            nvidia_compute_processes=self._nvidia_compute_processes(),
            gpu_container_ids=self._gpu_container_ids(),
            prior_child_pids=self._owned_process_groups.alive_pids(),
        )

    def _nvidia_compute_processes(self) -> tuple[int, ...]:
        compute = _command_lines(("nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"))
        return tuple(int(line) for line in compute if line.isdigit())

    def _gpu_container_ids(self) -> tuple[str, ...]:
        return _gpu_container_ids()

    def sample(self) -> IdleSample:
        """Capture maximum GPU utilization/memory and one-minute host load."""
        lines = _command_lines(
            (
                "nvidia-smi",
                "--id=0",
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
        evaluation = _next_idle_evaluation_id(self.evidence_root, attempt_identity_value)
        while True:
            inventory_before = self.monitor.inventory()
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
            inventory_after = self.monitor.inventory()
            reasons = self._rejection_reasons((inventory_before, inventory_after), samples)
            evidence = self._persist_evidence(
                attempt_identity_value,
                evaluation,
                inventory_before,
                inventory_after,
                samples,
                reasons,
            )
            if not reasons:
                return evidence
            elapsed = self.clock.monotonic() - started
            if elapsed + self.thresholds.retry_interval_s > self.thresholds.timeout_s:
                raise IdleGateTimeout(f"host idle gate timed out after {elapsed:.1f}s; latest evidence: {evidence}")
            self.clock.sleep(self.thresholds.retry_interval_s)
            evaluation += 1

    def _rejection_reasons(self, inventories: Sequence[IdleInventory], samples: Sequence[IdleSample]) -> list[str]:
        reasons: list[str] = []
        if any(inventory.nvidia_compute_processes for inventory in inventories):
            reasons.append("nvidia_compute_process")
        if any(inventory.gpu_container_ids for inventory in inventories):
            reasons.append("gpu_container")
        if any(inventory.prior_child_pids for inventory in inventories):
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
        inventory_before: IdleInventory,
        inventory_after: IdleInventory,
        samples: Sequence[IdleSample],
        reasons: Sequence[str],
    ) -> Path:
        self.evidence_root.mkdir(parents=True, exist_ok=True)
        path = self.evidence_root / f"{attempt_identity_value}-idle-{evaluation:04d}.json"
        load_limit = self.logical_cpu_count * self.thresholds.host_load_cpu_fraction
        _write_json_new(
            path,
            {
                "attempt_identity": attempt_identity_value,
                "evaluation": evaluation,
                "decision": "rejected" if reasons else "accepted",
                "reasons": list(reasons),
                "inventory_before_samples": asdict(inventory_before),
                "inventory_after_samples": asdict(inventory_after),
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
        expected_provenance: ExecutionProvenance,
        expected_gpu_uuid: str,
    ):
        self.artifact_root = artifact_root
        self.executors = executors
        self.idle_gate = idle_gate
        self.expected_provenance = expected_provenance
        self.expected_gpu_uuid = expected_gpu_uuid
        self.after_persist = None

    def run(self, expansion: MatrixExpansion, *, retry_failures: bool = False) -> RunResult:
        """Run a deterministic expansion, resuming trustworthy results."""
        state_path = self.artifact_root / expansion.run_set.value / "runner-state.json"
        state = _load_state(state_path, expansion)
        succeeded = failed = skipped = 0
        for attempt in expansion.attempts:
            attempt_root = self.artifact_root / attempt.run_directory
            success = attempt_root / "success"
            quarantined_corrupt_success = False
            if success.exists():
                if self._valid_success(success, attempt):
                    skipped += 1
                    _append_history(state, attempt, "skipped_success")
                    self._persist(state_path, state)
                    continue
                _quarantine_corrupt_success(success)
                quarantined_corrupt_success = True
            prior_failures = _historical_failures(attempt_root)
            if prior_failures and not retry_failures and not quarantined_corrupt_success:
                failed += 1
                skipped += 1
                _append_history(state, attempt, "skipped_failure")
                self._persist(state_path, state)
                continue
            try:
                idle_evidence = self.idle_gate.wait(attempt.identity)
            except KeyboardInterrupt:
                _append_history(state, attempt, "interrupted_idle_gate")
                state["status"] = RunStatus.INTERRUPTED.value
                self._persist(state_path, state)
                return RunResult(RunStatus.INTERRUPTED, succeeded, failed, skipped, state_path)
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

    def _valid_success(self, path: Path, attempt: BenchmarkAttempt) -> bool:
        return verify_success(
            path,
            attempt,
            expected_provenance=self.expected_provenance,
            expected_gpu_uuid=self.expected_gpu_uuid,
        )

    def _persist(self, state_path: Path, state: dict[str, object]) -> None:
        _write_json_atomic(state_path, state)
        if self.after_persist is not None:
            self.after_persist(state_path)


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
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as file:
        temporary = Path(file.name)
        file.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        file.flush()
        os.fsync(file.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_new(path: Path, value: object) -> None:
    """Atomically publish a new JSON document without replacing evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        file.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _next_idle_evaluation_id(root: Path, attempt_identity_value: str) -> int:
    pattern = re.compile(rf"{re.escape(attempt_identity_value)}-idle-([0-9]+)\.json$")
    numbers: list[int] = []
    if root.exists():
        for path in root.iterdir():
            match = pattern.fullmatch(path.name)
            if match is not None:
                numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def _historical_failures(attempt_root: Path) -> tuple[Path, ...]:
    if not attempt_root.exists():
        return ()
    return tuple(
        path for path in attempt_root.iterdir() if path.is_dir() and _FAILED_ATTEMPT_PATTERN.fullmatch(path.name)
    )


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


def _process_group_pids(group_ids: set[int]) -> dict[int, tuple[int, ...]]:
    grouped: dict[int, list[int]] = {group_id: [] for group_id in group_ids}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = (path / "stat").read_text(encoding="ascii")
            fields = stat[stat.rfind(")") + 2 :].split()
            process_group_id = int(fields[2])
            if process_group_id in grouped:
                grouped[process_group_id].append(int(path.name))
        except (OSError, UnicodeError, ValueError, IndexError):
            continue
    return {group_id: tuple(sorted(pids)) for group_id, pids in grouped.items() if pids}
