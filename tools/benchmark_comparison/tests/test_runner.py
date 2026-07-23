# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the serialized, resumable comparison runner."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tools.benchmark_comparison.matrix import expand_canary_matrix, expand_final_matrix, load_matrix
from tools.benchmark_comparison.runner import (
    AttemptExecution,
    BenchmarkRunner,
    IdleGateTimeout,
    RunStatus,
)
from tools.benchmark_comparison.validate import attempt_identity

FIXTURES = Path(__file__).parent / "fixtures"


def _schema(attempt):
    name = "schema_training.json" if attempt.mode.id.startswith("training") else "schema_runtime.json"
    schema = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    schema["run"]["task"] = attempt.concrete_task
    schema["run"]["seed"] = attempt.seed
    schema["run"]["num_envs"] = attempt.num_envs
    schema["versions"]["isaaclab_release"] = "2.3.2" if attempt.version.value == "lab2" else "3.0.0"
    if attempt.bound.unit.value == "steps":
        schema["runtime"]["iterations_completed"] = attempt.bound.value
    else:
        schema["runtime"]["iterations_completed"] = attempt.bound.value
        schema["run"]["max_iterations"] = attempt.bound.value
    return schema


def _measurements():
    return json.loads((FIXTURES / "generic_runtime.json").read_text(encoding="utf-8"))


def _execution(attempt, *, exit_code=0, timed_out=False, interrupted=False, oom=False):
    identity = attempt_identity(attempt)
    return AttemptExecution(
        command={"identity": identity, "argv": ["fake"]},
        environment={"identity": identity, "values": {}},
        stdout="",
        stderr="CUDA out of memory" if oom else "",
        exit_status={
            "exit_code": exit_code,
            "failure_stage": None,
            "timed_out": timed_out,
            "interrupted": interrupted,
            "out_of_memory": oom,
        },
        schema=None if exit_code or timed_out or interrupted or oom else _schema(attempt),
        measurements=None if exit_code or timed_out or interrupted or oom else _measurements(),
    )


class _Executor:
    def __init__(self, outcomes=None):
        self.outcomes = list(outcomes or [])
        self.attempts = []

    def execute(self, attempt):
        self.attempts.append(attempt)
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome(attempt)
        return _execution(attempt)


class _Gate:
    def __init__(self, error: Exception | None = None):
        self.identities = []
        self.error = error

    def wait(self, identity: str):
        self.identities.append(identity)
        if self.error:
            raise self.error
        return Path("idle.json")


def _runner(tmp_path: Path, executors, gate=None) -> BenchmarkRunner:
    return BenchmarkRunner(
        artifact_root=tmp_path,
        executors=executors,
        idle_gate=gate or _Gate(),
    )


def test_runner_executes_all_canary_attempts_in_task_six_order(tmp_path: Path):
    expansion = expand_canary_matrix(load_matrix())
    lab2 = _Executor()
    lab3 = _Executor()

    result = _runner(tmp_path, {"lab2": lab2, "lab3": lab3}).run(expansion)

    observed = [entry["attempt_identity"] for entry in json.loads(result.state_path.read_text())["history"]]
    assert observed == [attempt.identity for attempt in expansion.attempts]
    assert result.status is RunStatus.COMPLETED
    assert len(lab2.attempts) == len(lab3.attempts) == 18
    assert all(
        expansion.attempts[index].version_order < expansion.attempts[index + 1].version_order
        for index in range(0, len(expansion.attempts), 2)
    )


def test_full_matrix_runner_preserves_108_counterbalanced_attempts(tmp_path: Path):
    expansion = expand_final_matrix(load_matrix())
    lab2 = _Executor()
    lab3 = _Executor()

    _runner(tmp_path, {"lab2": lab2, "lab3": lab3}).run(expansion)

    assert len(lab2.attempts) + len(lab3.attempts) == 108
    assert [attempt.version.value for attempt in expansion.attempts[:2]] == ["lab2", "lab3"]
    seed_43 = next(index for index, attempt in enumerate(expansion.attempts) if attempt.seed == 43)
    assert [attempt.version.value for attempt in expansion.attempts[seed_43 : seed_43 + 2]] == ["lab3", "lab2"]


def test_resume_skips_only_checksum_and_semantically_valid_success(tmp_path: Path):
    expansion = replace(expand_canary_matrix(load_matrix()), attempts=expand_canary_matrix(load_matrix()).attempts[:1])
    first = _Executor()
    _runner(tmp_path, {"lab2": first, "lab3": _Executor()}).run(expansion)
    resumed = _Executor()

    result = _runner(tmp_path, {"lab2": resumed, "lab3": _Executor()}).run(expansion)

    assert resumed.attempts == []
    assert result.skipped == 1


def test_corrupt_success_is_quarantined_and_rerun_without_overwrite(tmp_path: Path):
    attempt = expand_canary_matrix(load_matrix()).attempts[0]
    expansion = replace(expand_canary_matrix(load_matrix()), attempts=(attempt,))
    _runner(tmp_path, {"lab2": _Executor(), "lab3": _Executor()}).run(expansion)
    success = tmp_path / attempt.run_directory / "success"
    (success / "stdout.log").write_text("corrupt", encoding="utf-8")
    rerun = _Executor()

    _runner(tmp_path, {"lab2": rerun, "lab3": _Executor()}).run(expansion)

    attempt_root = tmp_path / attempt.run_directory
    assert len(rerun.attempts) == 1
    assert (attempt_root / "success").is_dir()
    assert (attempt_root / "corrupt-success-0001").is_dir()


def test_rechecksummed_forged_validation_is_quarantined_and_rerun(tmp_path: Path):
    attempt = expand_canary_matrix(load_matrix()).attempts[0]
    expansion = replace(expand_canary_matrix(load_matrix()), attempts=(attempt,))
    _runner(tmp_path, {"lab2": _Executor(), "lab3": _Executor()}).run(expansion)
    success = tmp_path / attempt.run_directory / "success"
    validation = json.loads((success / "validation.json").read_text(encoding="utf-8"))
    validation["metrics"]["collection_fps"] = 999999.0
    (success / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = (success / "checksums.sha256").read_text(encoding="ascii").splitlines()
    manifest = [
        f"{hashlib.sha256((success / 'validation.json').read_bytes()).hexdigest()}  validation.json"
        if line.endswith("  validation.json")
        else line
        for line in manifest
    ]
    (success / "checksums.sha256").write_text("\n".join(manifest) + "\n", encoding="ascii")
    rerun = _Executor()

    _runner(tmp_path, {"lab2": rerun, "lab3": _Executor()}).run(expansion)

    attempt_root = tmp_path / attempt.run_directory
    assert len(rerun.attempts) == 1
    assert (attempt_root / "corrupt-success-0001").is_dir()
    assert (attempt_root / "success").is_dir()


def test_failed_attempt_is_retried_only_when_explicitly_requested(tmp_path: Path):
    attempt = expand_canary_matrix(load_matrix()).attempts[0]
    expansion = replace(expand_canary_matrix(load_matrix()), attempts=(attempt,))
    failed = _Executor([lambda current: _execution(current, exit_code=7)])
    _runner(tmp_path, {"lab2": failed, "lab3": _Executor()}).run(expansion)
    no_retry = _Executor()

    first_resume = _runner(tmp_path, {"lab2": no_retry, "lab3": _Executor()}).run(expansion)
    retry = _Executor()
    second_resume = _runner(tmp_path, {"lab2": retry, "lab3": _Executor()}).run(expansion, retry_failures=True)

    assert no_retry.attempts == []
    assert first_resume.skipped == 1
    assert first_resume.status is RunStatus.COMPLETED_WITH_FAILURES
    assert len(retry.attempts) == 1
    assert second_resume.succeeded == 1


@pytest.mark.parametrize(
    ("execution", "suffix", "expected_status"),
    [
        (lambda attempt: _execution(attempt, timed_out=True), "timeout", RunStatus.COMPLETED_WITH_FAILURES),
        (lambda attempt: _execution(attempt, oom=True), "out_of_memory", RunStatus.COMPLETED_WITH_FAILURES),
        (lambda attempt: _execution(attempt, interrupted=True), "interrupted", RunStatus.INTERRUPTED),
    ],
)
def test_runner_classifies_recoverable_execution_failures(
    tmp_path: Path, execution, suffix: str, expected_status: RunStatus
):
    attempt = expand_canary_matrix(load_matrix()).attempts[0]
    expansion = replace(expand_canary_matrix(load_matrix()), attempts=(attempt,))

    result = _runner(tmp_path, {"lab2": _Executor([execution]), "lab3": _Executor()}).run(expansion)

    assert result.failed == 1
    assert next((tmp_path / attempt.run_directory).glob(f"attempt-0001-{suffix}")).is_dir()
    assert result.status is expected_status


def test_keyboard_interrupt_finalizes_failure_and_state_before_stopping(tmp_path: Path):
    attempt = expand_canary_matrix(load_matrix()).attempts[0]
    expansion = replace(expand_canary_matrix(load_matrix()), attempts=(attempt,))
    runner = _runner(tmp_path, {"lab2": _Executor([KeyboardInterrupt()]), "lab3": _Executor()})

    result = runner.run(expansion)

    assert result.status is RunStatus.INTERRUPTED
    assert next((tmp_path / attempt.run_directory).glob("attempt-0001-interrupted")).is_dir()
    state = json.loads(result.state_path.read_text(encoding="utf-8"))
    assert state["status"] == "interrupted"
    assert state["history"][-1]["status"] == "interrupted"


def test_idle_timeout_stops_run_set_as_preflight_failure(tmp_path: Path):
    expansion = replace(expand_canary_matrix(load_matrix()), attempts=expand_canary_matrix(load_matrix()).attempts[:2])
    executor = _Executor()

    result = _runner(
        tmp_path,
        {"lab2": executor, "lab3": _Executor()},
        gate=_Gate(IdleGateTimeout("busy host")),
    ).run(expansion)

    assert result.status is RunStatus.PREFLIGHT_FAILED
    assert executor.attempts == []
    assert json.loads(result.state_path.read_text(encoding="utf-8"))["status"] == "preflight_failed"


def test_launch_exception_is_finalized_as_recoverable_failure(tmp_path: Path):
    attempt = expand_canary_matrix(load_matrix()).attempts[0]
    expansion = replace(expand_canary_matrix(load_matrix()), attempts=(attempt,))

    result = _runner(
        tmp_path,
        {"lab2": _Executor([OSError("cannot start child")]), "lab3": _Executor()},
    ).run(expansion)

    assert result.status is RunStatus.COMPLETED_WITH_FAILURES
    failure = next((tmp_path / attempt.run_directory).glob("attempt-0001-launch"))
    assert failure.is_dir()
    exit_status = json.loads((failure / "exit.json").read_text(encoding="utf-8"))
    assert exit_status["failure_stage"] == "launch"
    assert "cannot start child" in (failure / "stderr.log").read_text(encoding="utf-8")


def test_state_is_atomically_durable_after_each_attempt(tmp_path: Path):
    expansion = replace(expand_canary_matrix(load_matrix()), attempts=expand_canary_matrix(load_matrix()).attempts[:3])
    observed = []

    def after_persist(path: Path):
        state = json.loads(path.read_text(encoding="utf-8"))
        observed.append(len(state["history"]))
        assert not path.with_suffix(".json.tmp").exists()

    runner = _runner(tmp_path, {"lab2": _Executor(), "lab3": _Executor()})
    runner.after_persist = after_persist

    runner.run(expansion)

    assert observed[:3] == [1, 2, 3]
