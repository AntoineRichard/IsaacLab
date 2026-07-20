# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the single-GPU benchmark subprocess runner."""

import json
import subprocess
from pathlib import Path

from benchmarks.kamino_dvi.manifests import read_manifest
from benchmarks.kamino_dvi.matrix import DEFAULT_MATRIX_PATH, load_matrix
from benchmarks.kamino_dvi.models import TaskName, TerminalState, Variant
from benchmarks.kamino_dvi.run import (
    ProcessOutcome,
    build_parser,
    execute_command,
    execute_identity,
    execute_sequentially,
    inspect_bundle,
    run_directory,
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


def test_run_directory_is_stable_and_keeps_outputs_together(tmp_path: Path):
    """Each identity must map to one deterministic artifact directory."""
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    identity = select_identities(
        matrix,
        build_parser().parse_args(
            ["--full-only", "--task", TaskName.CARTPOLE.value, "--variant", Variant.MJWARP.value, "--seed", "42"]
        ),
    )[0]

    path = run_directory(tmp_path, identity)

    assert path == tmp_path / "full__Isaac-Cartpole-Direct__mjwarp__seed42__env4096__iter300"


def test_inspect_bundle_requires_completed_expected_iterations(tmp_path: Path):
    """A schema bundle is successful only when status and iteration count match."""
    bundle_path = tmp_path / "benchmark_training_task.json"
    bundle_path.write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "run": {"status": "completed"},
                "runtime": {"iterations_completed": 300},
            }
        ),
        encoding="utf-8",
    )

    status = inspect_bundle(tmp_path, expected_iterations=300)

    assert status.path == bundle_path
    assert status.schema_version == "1.1"
    assert status.completed_iterations == 300
    assert status.complete is True


def test_execute_identity_writes_terminal_manifest_and_resumes(tmp_path: Path):
    """A successful exact run must persist completion and skip on resume."""
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    identity = select_identities(
        matrix,
        build_parser().parse_args(
            ["--full-only", "--task", TaskName.CARTPOLE.value, "--variant", Variant.MJWARP.value, "--seed", "42"]
        ),
    )[0]
    calls = 0

    def executor(command, stdout_path, stderr_path, *, timeout_s):
        nonlocal calls
        calls += 1
        output_path = Path(command[command.index("--output_path") + 1])
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "benchmark_training_task.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.1",
                    "run": {"status": "completed"},
                    "runtime": {"iterations_completed": 300},
                }
            ),
            encoding="utf-8",
        )
        stdout_path.write_text("ok\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return ProcessOutcome(returncode=0, timed_out=False)

    first = execute_identity(matrix, identity, Path("/repo"), tmp_path, resume=False, executor=executor)
    second = execute_identity(matrix, identity, Path("/repo"), tmp_path, resume=True, executor=executor)

    manifest = read_manifest(run_directory(tmp_path, identity) / "manifest.json")
    assert first is TerminalState.COMPLETED
    assert second is TerminalState.COMPLETED
    assert manifest.state is TerminalState.COMPLETED
    assert calls == 1
