# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the single-GPU benchmark subprocess runner."""

import subprocess
from pathlib import Path

from benchmarks.kamino_dvi.matrix import DEFAULT_MATRIX_PATH, load_matrix
from benchmarks.kamino_dvi.models import TaskName, Variant
from benchmarks.kamino_dvi.run import (
    ProcessOutcome,
    build_parser,
    execute_command,
    execute_sequentially,
    select_identities,
)


class FakeProcess:
    """Small controllable stand-in for :class:`subprocess.Popen`."""

    def __init__(self, returncode=0, timeout=False):
        self.pid = 123
        self.returncode = returncode
        self.timeout = timeout
        self.calls = 0

    def communicate(self, timeout=None):
        self.calls += 1
        if self.timeout and self.calls == 1:
            raise subprocess.TimeoutExpired(["train"], timeout)
        return


def test_execute_command_streams_logs_and_starts_new_process_group(tmp_path: Path):
    """Training output must stream to files and execute in its own process group."""
    captured = {}

    def factory(command, **kwargs):
        captured.update(kwargs)
        kwargs["stdout"].write("stdout line\n")
        kwargs["stderr"].write("stderr line\n")
        return FakeProcess(returncode=0)

    outcome = execute_command(
        ["python", "train.py"],
        tmp_path / "stdout.log",
        tmp_path / "stderr.log",
        timeout_s=30,
        popen_factory=factory,
    )

    assert outcome == ProcessOutcome(returncode=0, timed_out=False)
    assert captured["start_new_session"] is True
    assert captured["text"] is True
    assert (tmp_path / "stdout.log").read_text(encoding="utf-8") == "stdout line\n"
    assert (tmp_path / "stderr.log").read_text(encoding="utf-8") == "stderr line\n"


def test_execute_command_terminates_timed_out_process_group(tmp_path: Path):
    """Timeout must terminate the entire isolated training process group."""
    killed: list[int] = []

    outcome = execute_command(
        ["python", "train.py"],
        tmp_path / "stdout.log",
        tmp_path / "stderr.log",
        timeout_s=30,
        popen_factory=lambda command, **kwargs: FakeProcess(returncode=-15, timeout=True),
        kill_process_group=killed.append,
    )

    assert outcome == ProcessOutcome(returncode=-15, timed_out=True)
    assert killed == [123]


def test_execute_sequentially_continues_after_failure():
    """One failed seed must not prevent later matrix cells from running."""
    calls: list[list[str]] = []

    def executor(command):
        calls.append(command)
        return ProcessOutcome(returncode=1 if len(calls) == 1 else 0, timed_out=False)

    outcomes = execute_sequentially((["run-1"], ["run-2"], ["run-3"]), executor)

    assert [outcome.returncode for outcome in outcomes] == [1, 0, 0]
    assert calls == [["run-1"], ["run-2"], ["run-3"]]


def test_cli_filters_select_exact_preflight_identity():
    """Phase, task, and variant filters must produce one deterministic dry run."""
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    args = build_parser().parse_args(
        [
            "--dry-run",
            "--resume",
            "--preflight-only",
            "--task",
            TaskName.ANT.value,
            "--variant",
            Variant.KAMINO_PR_DVI.value,
        ]
    )

    identities = select_identities(matrix, args)

    assert args.dry_run is True
    assert args.resume is True
    assert len(identities) == 1
    assert identities[0].task is TaskName.ANT
    assert identities[0].variant is Variant.KAMINO_PR_DVI
    assert identities[0].phase.value == "preflight"


def test_cli_full_seed_filter_keeps_all_applicable_task_variants():
    """A full-run seed filter must retain every applicable variant for its task."""
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    args = build_parser().parse_args(["--full-only", "--task", TaskName.ANT.value, "--seed", "46"])

    identities = select_identities(matrix, args)

    assert len(identities) == 5
    assert {identity.variant for identity in identities} == set(matrix.task(TaskName.ANT).variants)
    assert all(identity.seed == 46 for identity in identities)
