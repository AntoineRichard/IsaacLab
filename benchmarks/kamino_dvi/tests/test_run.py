# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the single-GPU benchmark subprocess runner."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.kamino_dvi.commands import build_training_command
from benchmarks.kamino_dvi.manifests import (
    command_hash,
    read_manifest,
    sha256_file,
    stable_run_id,
    write_manifest,
)
from benchmarks.kamino_dvi.matrix import DEFAULT_MATRIX_PATH, load_matrix
from benchmarks.kamino_dvi.models import (
    FailureCategory,
    Phase,
    RetryLineage,
    RunIdentity,
    RunManifest,
    TaskName,
    TerminalState,
    Variant,
)
from benchmarks.kamino_dvi.parsing import MissingBenchmarkFieldError
from benchmarks.kamino_dvi.run import (
    ProcessOutcome,
    build_parser,
    execute_command,
    execute_identity,
    execute_sequentially,
    inspect_bundle,
    main,
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


def test_execute_command_streams_logs_and_starts_new_process_group(tmp_path: Path, monkeypatch):
    """Training output must stream to files and execute in its own process group."""
    captured = {}
    monkeypatch.setenv("PYTHONPATH", "/contaminated/kit/python")

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
    assert "PYTHONPATH" not in captured["env"]
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
    args = build_parser().parse_args(["--full-only", "--task", TaskName.ANT.value, "--seed", "44"])

    identities = select_identities(matrix, args)

    assert len(identities) == 5
    assert {identity.variant for identity in identities} == set(matrix.task(TaskName.ANT).variants)
    assert all(identity.seed == 44 for identity in identities)


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


def test_execute_identity_writes_terminal_manifest_and_resumes(tmp_path: Path, monkeypatch):
    """A successful exact run must persist completion and skip on resume."""
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    identity = select_identities(
        matrix,
        build_parser().parse_args(
            ["--full-only", "--task", TaskName.CARTPOLE.value, "--variant", Variant.MJWARP.value, "--seed", "42"]
        ),
    )[0]
    calls = 0
    event_path = tmp_path / "events.out.tfevents.test"
    event_path.write_bytes(b"event")
    monkeypatch.setattr("benchmarks.kamino_dvi.run.locate_rsl_rl_events", lambda *args: event_path, raising=False)

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

    first = execute_identity(
        matrix, identity, Path("/repo"), tmp_path, resume=False, executor=executor, isaaclab_head="f" * 40
    )
    second = execute_identity(
        matrix, identity, Path("/repo"), tmp_path, resume=True, executor=executor, isaaclab_head="f" * 40
    )
    third = execute_identity(
        matrix,
        identity,
        Path("/repo"),
        tmp_path,
        resume=True,
        retry=RetryLineage(attempt=1, parent_run_id="capacity-parent"),
        executor=executor,
        isaaclab_head="f" * 40,
    )

    manifest = read_manifest(run_directory(tmp_path, identity) / "manifest.json")
    assert first is TerminalState.COMPLETED
    assert second is TerminalState.COMPLETED
    assert third is TerminalState.COMPLETED
    assert manifest.state is TerminalState.COMPLETED
    assert manifest.isaaclab_head == "f" * 40
    assert manifest.tensorboard_event_path == str(event_path.resolve())
    assert manifest.tensorboard_event_hash == sha256_file(event_path)
    assert manifest.retry == RetryLineage(attempt=1, parent_run_id="capacity-parent")
    assert calls == 2


def test_capacity_preflight_retries_task_at_one_lower_common_count(tmp_path: Path, monkeypatch):
    """A capacity preflight must replace pending task runs with one lower-count schedule."""
    calls = []
    failed = False

    def fake_execute(matrix, identity, repo_root, artifact_root, *, isaaclab_head, resume, retry=RetryLineage()):
        nonlocal failed
        calls.append((identity, resume))
        if identity.phase is Phase.PREFLIGHT and identity.num_envs == 4096 and not failed:
            failed = True
            return TerminalState.FAILED
        return TerminalState.COMPLETED

    monkeypatch.setattr(
        "benchmarks.kamino_dvi.run.probe_environment",
        lambda *args: SimpleNamespace(isaaclab=SimpleNamespace(head="f" * 40)),
    )
    monkeypatch.setattr("benchmarks.kamino_dvi.run.validate_environment", lambda *args: None)
    monkeypatch.setattr("benchmarks.kamino_dvi.run.execute_identity", fake_execute)
    monkeypatch.setattr(
        "benchmarks.kamino_dvi.run.read_manifest",
        lambda path: SimpleNamespace(
            failure_category=FailureCategory.CAPACITY, retry=RetryLineage(), run_id=path.parent.name
        ),
    )

    result = main(["--task", TaskName.CARTPOLE.value, "--resume", "--artifact-root", str(tmp_path)])

    identities = [identity for identity, _ in calls]
    retried_preflights = [
        identity for identity in identities if identity.phase is Phase.PREFLIGHT and identity.num_envs == 2048
    ]
    retried_full = [identity for identity in identities if identity.phase is Phase.FULL and identity.num_envs == 2048]
    assert result == 0
    assert len(retried_preflights) == 5
    assert {identity.variant for identity in retried_preflights} == set(
        load_matrix(DEFAULT_MATRIX_PATH).task(TaskName.CARTPOLE).variants
    )
    assert len(retried_full) == 15
    assert not any(identity.phase is Phase.FULL and identity.num_envs == 4096 for identity in identities)
    assert all(resume for _, resume in calls)


def test_capacity_full_invalidates_task_results_and_repreflights_before_retry(tmp_path: Path, monkeypatch):
    """A capacity full run must re-preflight all variants and retry all requested full cells lower."""
    calls = []
    failed = False

    def fake_execute(matrix, identity, repo_root, artifact_root, *, isaaclab_head, resume, retry=RetryLineage()):
        nonlocal failed
        calls.append(identity)
        if identity.phase is Phase.FULL and identity.num_envs == 4096 and not failed:
            failed = True
            return TerminalState.FAILED
        return TerminalState.COMPLETED

    monkeypatch.setattr(
        "benchmarks.kamino_dvi.run.probe_environment",
        lambda *args: SimpleNamespace(isaaclab=SimpleNamespace(head="f" * 40)),
    )
    monkeypatch.setattr("benchmarks.kamino_dvi.run.validate_environment", lambda *args: None)
    monkeypatch.setattr("benchmarks.kamino_dvi.run.execute_identity", fake_execute)
    monkeypatch.setattr(
        "benchmarks.kamino_dvi.run.read_manifest",
        lambda path: SimpleNamespace(
            failure_category=FailureCategory.CAPACITY, retry=RetryLineage(), run_id=path.parent.name
        ),
    )

    result = main(
        [
            "--full-only",
            "--task",
            TaskName.ANT.value,
            "--seed",
            "42",
            "--artifact-root",
            str(tmp_path),
        ]
    )

    retried_preflights = [
        identity for identity in calls if identity.phase is Phase.PREFLIGHT and identity.num_envs == 2048
    ]
    retried_full = [identity for identity in calls if identity.phase is Phase.FULL and identity.num_envs == 2048]
    assert result == 0
    assert len(retried_preflights) == 5
    assert len(retried_full) == 5
    assert {identity.variant for identity in retried_full} == set(
        load_matrix(DEFAULT_MATRIX_PATH).task(TaskName.ANT).variants
    )
    assert all(identity.seed == 42 for identity in retried_full)


def test_late_full_capacity_invalidates_prior_completed_count_without_deleting_evidence(tmp_path: Path, monkeypatch):
    """A later capacity failure must retain but disqualify earlier full results at the invalid count."""
    real_execute_identity = execute_identity
    full_attempts_at_4096 = 0

    def process_executor(command, stdout_path, stderr_path, *, timeout_s):
        nonlocal full_attempts_at_4096
        num_envs = int(command[command.index("--num_envs") + 1])
        max_iterations = int(command[command.index("--max_iterations") + 1])
        if num_envs == 4096 and max_iterations == 300:
            full_attempts_at_4096 += 1
            if full_attempts_at_4096 == 2:
                stdout_path.write_text("", encoding="utf-8")
                stderr_path.write_text("CUDA out of memory\n", encoding="utf-8")
                return ProcessOutcome(returncode=1, timed_out=False)
        output_path = Path(command[command.index("--output_path") + 1])
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "benchmark_training_task.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.1",
                    "run": {"status": "completed"},
                    "runtime": {"iterations_completed": max_iterations},
                }
            ),
            encoding="utf-8",
        )
        stdout_path.write_text("ok\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return ProcessOutcome(returncode=0, timed_out=False)

    def execute_with_process(
        matrix, identity, repo_root, artifact_root, *, isaaclab_head, resume, retry=RetryLineage()
    ):
        return real_execute_identity(
            matrix,
            identity,
            repo_root,
            artifact_root,
            isaaclab_head=isaaclab_head,
            resume=resume,
            retry=retry,
            executor=process_executor,
        )

    monkeypatch.setattr(
        "benchmarks.kamino_dvi.run.probe_environment",
        lambda *args: SimpleNamespace(isaaclab=SimpleNamespace(head="f" * 40)),
    )
    monkeypatch.setattr("benchmarks.kamino_dvi.run.validate_environment", lambda *args: None)
    monkeypatch.setattr("benchmarks.kamino_dvi.run.execute_identity", execute_with_process)
    event_path = tmp_path / "events.out.tfevents.test"
    event_path.write_bytes(b"event")
    monkeypatch.setattr("benchmarks.kamino_dvi.run.locate_rsl_rl_events", lambda *args: event_path)

    result = main(
        [
            "--full-only",
            "--task",
            TaskName.ANT.value,
            "--seed",
            "42",
            "--artifact-root",
            str(tmp_path),
        ]
    )

    manifests_at_4096 = [
        json.loads(path.read_text(encoding="utf-8")) for path in tmp_path.glob("full__*__env4096__*/manifest.json")
    ]
    invalidated = [manifest for manifest in manifests_at_4096 if manifest["state"] == "invalidated"]
    assert result == 0
    assert len(invalidated) == 1
    assert Path(invalidated[0]["artifact_root"], "benchmark_training_task.json").is_file()
    lower_manifests = [read_manifest(path) for path in tmp_path.glob("full__*__env2048__*/manifest.json")]
    failed_manifest = next(manifest for manifest in manifests_at_4096 if manifest["failure_category"] == "capacity")
    assert len(lower_manifests) == 5
    assert all(manifest.retry.attempt == 1 for manifest in lower_manifests)
    assert all(manifest.retry.parent_run_id == failed_manifest["run_id"] for manifest in lower_manifests)


def test_non_capacity_fallback_preflight_failure_cancels_task_full_runs(tmp_path: Path, monkeypatch):
    """A failed retry preflight must prevent every full run for that task."""
    calls = []
    initial_capacity_failed = False
    fallback_preflight_failed = False

    def fake_execute(matrix, identity, repo_root, artifact_root, *, isaaclab_head, resume, retry=RetryLineage()):
        nonlocal initial_capacity_failed, fallback_preflight_failed
        calls.append(identity)
        if identity.phase is Phase.PREFLIGHT and identity.num_envs == 4096 and not initial_capacity_failed:
            initial_capacity_failed = True
            return TerminalState.FAILED
        if identity.phase is Phase.PREFLIGHT and identity.num_envs == 2048 and not fallback_preflight_failed:
            fallback_preflight_failed = True
            return TerminalState.FAILED
        return TerminalState.COMPLETED

    monkeypatch.setattr(
        "benchmarks.kamino_dvi.run.probe_environment",
        lambda *args: SimpleNamespace(isaaclab=SimpleNamespace(head="f" * 40)),
    )
    monkeypatch.setattr("benchmarks.kamino_dvi.run.validate_environment", lambda *args: None)
    monkeypatch.setattr("benchmarks.kamino_dvi.run.execute_identity", fake_execute)
    monkeypatch.setattr(
        "benchmarks.kamino_dvi.run.read_manifest",
        lambda path: SimpleNamespace(
            failure_category=(FailureCategory.CAPACITY if "__env4096__" in str(path) else FailureCategory.NUMERICAL),
            retry=RetryLineage(),
            run_id=path.parent.name,
        ),
    )

    result = main(["--task", TaskName.CARTPOLE.value, "--artifact-root", str(tmp_path)])

    assert result == 1
    assert not any(identity.phase is Phase.FULL for identity in calls)


def _write_test_manifest(
    tmp_path: Path,
    matrix,
    identity: RunIdentity,
    state: TerminalState,
    category=None,
    retry=None,
    *,
    revisions=None,
    head=None,
):
    repo_root = Path(__file__).resolve().parents[3]
    command = tuple(build_training_command(matrix, identity, repo_root, run_directory(tmp_path, identity)))
    manifest = RunManifest(
        run_id=stable_run_id(identity),
        identity=identity,
        command=command,
        command_hash=command_hash(command),
        revisions=revisions or matrix.revisions,
        schema_version="1.1",
        artifact_root=str(run_directory(tmp_path, identity)),
        isaaclab_head=head or "f" * 40,
        state=state,
        failure_category=category,
        retry=retry or RetryLineage(),
    )
    write_manifest(run_directory(tmp_path, identity) / "manifest.json", manifest)
    return manifest


def test_resume_rebuilds_interrupted_fallback_before_full_and_preserves_invalidation(tmp_path: Path, monkeypatch):
    """Resume must derive the selected lower count and never overwrite invalidated old-count evidence."""
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    failed_identity = RunIdentity(TaskName.ANT, Variant.PHYSX, 42, Phase.FULL, 4096, 300)
    _write_test_manifest(
        tmp_path,
        matrix,
        failed_identity,
        TerminalState.FAILED,
        FailureCategory.CAPACITY,
    )
    old_identity = RunIdentity(TaskName.ANT, Variant.MJWARP, 42, Phase.FULL, 4096, 300)
    _write_test_manifest(tmp_path, matrix, old_identity, TerminalState.INVALIDATED)
    corrupt_dir = tmp_path / "unrelated-corrupt"
    corrupt_dir.mkdir()
    (corrupt_dir / "manifest.json").write_text("{not-json", encoding="utf-8")
    calls = []

    def fake_execute(matrix, identity, repo_root, artifact_root, *, isaaclab_head, resume, retry=RetryLineage()):
        calls.append((identity, retry))
        return TerminalState.COMPLETED

    monkeypatch.setattr(
        "benchmarks.kamino_dvi.run.probe_environment",
        lambda *args: SimpleNamespace(isaaclab=SimpleNamespace(head="f" * 40)),
    )
    monkeypatch.setattr("benchmarks.kamino_dvi.run.validate_environment", lambda *args: None)
    monkeypatch.setattr("benchmarks.kamino_dvi.run.execute_identity", fake_execute)

    result = main(
        [
            "--resume",
            "--full-only",
            "--task",
            TaskName.ANT.value,
            "--seed",
            "42",
            "--artifact-root",
            str(tmp_path),
        ]
    )

    assert result == 0
    assert len(calls) == 10
    assert all(identity.num_envs == 2048 for identity, _ in calls)
    assert all(identity.phase is Phase.PREFLIGHT for identity, _ in calls[:5])
    assert all(identity.phase is Phase.FULL for identity, _ in calls[5:])
    assert all(retry.attempt == 1 and retry.parent_run_id == stable_run_id(failed_identity) for _, retry in calls)
    assert read_manifest(run_directory(tmp_path, old_identity) / "manifest.json").state is TerminalState.INVALIDATED


def test_initial_non_capacity_preflight_failure_cancels_task_queue(tmp_path: Path, monkeypatch):
    """Any initial preflight failure must prevent all remaining task runs, including full runs."""
    calls = []
    failed = False

    def fake_execute(matrix, identity, repo_root, artifact_root, *, isaaclab_head, resume, retry=RetryLineage()):
        nonlocal failed
        calls.append(identity)
        if identity.phase is Phase.PREFLIGHT and not failed:
            failed = True
            return TerminalState.FAILED
        return TerminalState.COMPLETED

    monkeypatch.setattr(
        "benchmarks.kamino_dvi.run.probe_environment",
        lambda *args: SimpleNamespace(isaaclab=SimpleNamespace(head="f" * 40)),
    )
    monkeypatch.setattr("benchmarks.kamino_dvi.run.validate_environment", lambda *args: None)
    monkeypatch.setattr("benchmarks.kamino_dvi.run.execute_identity", fake_execute)
    monkeypatch.setattr(
        "benchmarks.kamino_dvi.run.read_manifest",
        lambda path: SimpleNamespace(failure_category=FailureCategory.NUMERICAL),
    )

    result = main(["--task", TaskName.CARTPOLE.value, "--artifact-root", str(tmp_path)])

    assert result == 1
    assert len(calls) == 1
    assert calls[0].phase is Phase.PREFLIGHT


def test_execute_identity_rejects_missing_tensorboard_event(tmp_path: Path, monkeypatch):
    """A schema bundle without its required TensorBoard trace is an artifact failure."""
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    identity = RunIdentity(TaskName.CARTPOLE, Variant.MJWARP, 42, Phase.FULL, 4096, 300)

    def executor(command, stdout_path, stderr_path, *, timeout_s):
        output_path = Path(command[command.index("--output_path") + 1])
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

    monkeypatch.setattr(
        "benchmarks.kamino_dvi.run.locate_rsl_rl_events",
        lambda *args: (_ for _ in ()).throw(MissingBenchmarkFieldError("events")),
        raising=False,
    )

    state = execute_identity(
        matrix,
        identity,
        Path("/repo"),
        tmp_path,
        isaaclab_head="f" * 40,
        resume=False,
        executor=executor,
    )

    manifest = read_manifest(run_directory(tmp_path, identity) / "manifest.json")
    assert state is TerminalState.FAILED
    assert manifest.failure_category is FailureCategory.ARTIFACT


def test_resume_ignores_stale_deeper_capacity_manifest(tmp_path: Path, monkeypatch):
    """A stale revision or HEAD must not move the current campaign farther down the ladder."""
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    valid = RunIdentity(TaskName.ANT, Variant.PHYSX, 42, Phase.FULL, 4096, 300)
    _write_test_manifest(tmp_path, matrix, valid, TerminalState.FAILED, FailureCategory.CAPACITY)
    stale = RunIdentity(TaskName.ANT, Variant.MJWARP, 42, Phase.FULL, 2048, 300)
    _write_test_manifest(
        tmp_path,
        matrix,
        stale,
        TerminalState.FAILED,
        FailureCategory.CAPACITY,
        RetryLineage(attempt=1, parent_run_id=stable_run_id(valid)),
        head="0" * 40,
    )
    stale_data = json.loads((run_directory(tmp_path, stale) / "manifest.json").read_text(encoding="utf-8"))
    stale_data["revisions"]["isaaclab"] = "0" * 40
    (run_directory(tmp_path, stale) / "manifest.json").write_text(json.dumps(stale_data), encoding="utf-8")
    calls = []

    def fake_execute(matrix, identity, repo_root, artifact_root, *, isaaclab_head, resume, retry=RetryLineage()):
        calls.append(identity)
        return TerminalState.COMPLETED

    monkeypatch.setattr(
        "benchmarks.kamino_dvi.run.probe_environment",
        lambda *args: SimpleNamespace(isaaclab=SimpleNamespace(head="f" * 40)),
    )
    monkeypatch.setattr("benchmarks.kamino_dvi.run.validate_environment", lambda *args: None)
    monkeypatch.setattr("benchmarks.kamino_dvi.run.execute_identity", fake_execute)

    result = main(
        ["--resume", "--full-only", "--task", TaskName.ANT.value, "--seed", "42", "--artifact-root", str(tmp_path)]
    )

    assert result == 0
    assert calls
    assert all(identity.num_envs == 2048 for identity in calls)


def test_resume_fails_clearly_for_persisted_exhausted_capacity_ladder(tmp_path: Path, monkeypatch):
    """A current-campaign capacity failure at 128 must remain terminal across resume."""
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    exhausted = RunIdentity(TaskName.ANT, Variant.PHYSX, 42, Phase.PREFLIGHT, 128, 5)
    _write_test_manifest(
        tmp_path,
        matrix,
        exhausted,
        TerminalState.FAILED,
        FailureCategory.CAPACITY,
        RetryLineage(attempt=5, parent_run_id="previous"),
    )
    monkeypatch.setattr(
        "benchmarks.kamino_dvi.run.probe_environment",
        lambda *args: SimpleNamespace(isaaclab=SimpleNamespace(head="f" * 40)),
    )
    monkeypatch.setattr("benchmarks.kamino_dvi.run.validate_environment", lambda *args: None)
    monkeypatch.setattr("benchmarks.kamino_dvi.run.execute_identity", lambda *args, **kwargs: TerminalState.COMPLETED)

    with pytest.raises(RuntimeError, match="capacity ladder exhausted.*128"):
        main(["--resume", "--task", TaskName.ANT.value, "--artifact-root", str(tmp_path)])
