# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the resumable ANYmal-D DVI tuning runner."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.kamino_dvi import tune as tune_module
from benchmarks.kamino_dvi.environment import PackageLocation
from benchmarks.kamino_dvi.manifests import command_hash, write_json_atomic
from benchmarks.kamino_dvi.matrix import DEFAULT_MATRIX_PATH, load_matrix
from benchmarks.kamino_dvi.models import FailureCategory, RetryLineage, TaskName, TerminalState
from benchmarks.kamino_dvi.parsing import MissingBenchmarkFieldError, TrainingTrace
from benchmarks.kamino_dvi.run import ProcessOutcome
from benchmarks.kamino_dvi.tune import (
    SCHEMA_VERSION,
    ResolvedTuningCandidate,
    TuningIdentity,
    TuningManifest,
    build_canonical_command,
    build_parser,
    build_tuning_command,
    execute_tuning_identity,
    main,
    preflight_completed,
    read_tuning_manifest,
    select_tuning_identities,
    tuning_resume_matches,
    validate_tuning_command,
    write_tuning_manifest,
)
from benchmarks.kamino_dvi.tuning import (
    DEFAULT_TUNING_MATRIX_PATH,
    config_hash,
    load_tuning_matrix,
    resolve_config,
)


def completed_tuning_manifest(tmp_path: Path) -> TuningManifest:
    """Build one completed tuning manifest with full provenance."""
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    locked = load_matrix(DEFAULT_MATRIX_PATH)
    candidate = tuning.candidate("cr_iterations_3")
    identity = TuningIdentity("wave1", candidate.name, 42, 4096, 40, 0)
    command = tuple(build_tuning_command(locked, tuning, candidate, identity, tmp_path, tmp_path / "run"))
    resolved = resolve_config(tuning, candidate)
    return TuningManifest(
        run_id=identity.run_id,
        identity=identity,
        config_hash=config_hash(resolved),
        resolved_config=resolved,
        command=command,
        command_hash=command_hash(command),
        revisions=locked.revisions,
        schema_version=SCHEMA_VERSION,
        isaaclab_head="f" * 40,
        isaaclab_newton=PackageLocation(
            module_path=str(tmp_path / "source/isaaclab_newton/isaaclab_newton/__init__.py"),
            distribution_path=str(tmp_path / "source/isaaclab_newton"),
            direct_url={"url": (tmp_path / "source/isaaclab_newton").as_uri(), "dir_info": {"editable": True}},
        ),
        artifact_root=str(tmp_path / "run"),
        tensorboard_event_path=str(tmp_path / "events.out.tfevents.test"),
        tensorboard_event_hash="e" * 64,
        artifact_hashes={"stdout.log": "a" * 64, "stderr.log": "b" * 64, "bundle.json": "c" * 64},
        state=TerminalState.COMPLETED,
        failure_category=None,
        retry=RetryLineage(),
    )


def test_manifest_round_trip_persists_package_location_and_schema(tmp_path):
    """Manifest schema 1.2 retains exact imported-package provenance."""
    path = tmp_path / "manifest.json"
    manifest = completed_tuning_manifest(tmp_path)

    write_tuning_manifest(path, manifest)

    assert SCHEMA_VERSION == "1.2"
    assert read_tuning_manifest(path) == manifest


def test_resume_rejects_changed_package_location(tmp_path):
    """Completed evidence cannot resume under a different imported checkout."""
    manifest = completed_tuning_manifest(tmp_path)
    stale = replace(manifest.isaaclab_newton, module_path="/stale/isaaclab_newton/__init__.py")

    assert not tuning_resume_matches(
        manifest, manifest.identity, manifest.command, manifest.config_hash, "f" * 40, stale
    )


def write_winner_decision(tmp_path: Path, candidate) -> dict:
    """Write the canonical winner decision consumed by the runner."""
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    resolved = resolve_config(tuning, candidate)
    winner = {
        "schema_version": "1.1",
        "action": "select-winner",
        "candidate": candidate.name,
        "resolved_config": resolved,
        "config_hash": config_hash(resolved),
    }
    write_json_atomic(tmp_path / "winner.json", winner)
    return winner


def test_tuning_command_appends_only_declared_candidate_overrides(tmp_path):
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    locked = load_matrix(DEFAULT_MATRIX_PATH)
    candidate = tuning.candidate("cr_iterations_3")
    identity = TuningIdentity("wave1", candidate.name, 42, 4096, 40, 0)
    command = build_tuning_command(locked, tuning, candidate, identity, tmp_path, tmp_path / "run")
    assert command[-1] == "env.sim.physics.solver_cfg.dynamics_linear_solver_max_iterations=3"
    assert command.count("presets=newton_kamino_dvi") == 1
    assert not any("dynamics_solver=dvi" in value for value in command)


def test_resume_requires_exact_head_command_config_and_event_hash(tmp_path):
    manifest = completed_tuning_manifest(tmp_path)
    assert tuning_resume_matches(manifest, manifest.identity, manifest.command, manifest.config_hash, "f" * 40)
    assert not tuning_resume_matches(manifest, manifest.identity, manifest.command, "0" * 64, "f" * 40)
    assert not tuning_resume_matches(manifest, manifest.identity, manifest.command, manifest.config_hash, "0" * 40)


def test_resume_rejects_changed_command_or_missing_event_integrity(tmp_path):
    manifest = completed_tuning_manifest(tmp_path)
    assert not tuning_resume_matches(
        manifest, manifest.identity, manifest.command + ("--extra",), manifest.config_hash, "f" * 40
    )
    assert not tuning_resume_matches(
        TuningManifest(**{**manifest.__dict__, "tensorboard_event_hash": None}),
        manifest.identity,
        manifest.command,
        manifest.config_hash,
        "f" * 40,
    )


def test_canonical_command_uses_winner_identity_without_hydra_overrides(tmp_path):
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    locked = load_matrix(DEFAULT_MATRIX_PATH)
    winner = write_winner_decision(tmp_path, tuning.candidate("cr_iterations_3"))
    identity, candidate, command = build_canonical_command(locked, tuning, winner, tmp_path)
    assert identity.candidate == "canonical_winner"
    assert candidate.config_hash == winner["config_hash"]
    assert not any(value.startswith("env.sim.physics.solver_cfg.") for value in command)


@pytest.mark.parametrize(
    ("filename", "count", "stage"),
    (("wave2.json", 6, "wave2"), ("stage2.json", 8, "halve"), ("finalists.json", 3, "final")),
)
def test_adaptive_decisions_reject_reserved_canonical_candidate(tmp_path, filename, count, stage):
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    candidates = list(tuning.wave1[:count])
    write_candidates_decision(tmp_path / filename, candidates)
    data = json.loads((tmp_path / filename).read_text(encoding="utf-8"))
    data["resolved_candidates"][0]["name"] = "canonical_winner"
    write_json_atomic(tmp_path / filename, data)
    args = build_parser().parse_args(["--stage", stage, "--decision-root", str(tmp_path), "--measured-only"])

    with pytest.raises(ValueError, match="reserved"):
        select_tuning_identities(tuning, args)


def test_noncanonical_execution_rejects_reserved_candidate_instead_of_suppressing_overrides(tmp_path):
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    locked = load_matrix(DEFAULT_MATRIX_PATH)
    declared = tuning.candidate("cr_iterations_3")
    resolved = resolve_config(tuning, declared)
    spoofed = ResolvedTuningCandidate(
        "canonical_winner",
        declared.overrides,
        resolved,
        config_hash(resolved),
    )
    identity = TuningIdentity("wave2", "canonical_winner", 42, 4096, 40)

    with pytest.raises(ValueError, match="reserved"):
        execute_tuning_identity(
            locked,
            tuning,
            spoofed,
            identity,
            tmp_path,
            tmp_path / "artifacts",
            isaaclab_head="f" * 40,
            resume=False,
            executor=successful_executor,
        )


def test_canonical_execution_rejects_noncanonical_candidate(tmp_path):
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    locked = load_matrix(DEFAULT_MATRIX_PATH)
    candidate = tuning.candidate("cr_iterations_3")
    identity = TuningIdentity("canonical", candidate.name, 42, 4096, 300)

    with pytest.raises(ValueError, match="canonical stage"):
        execute_tuning_identity(
            locked,
            tuning,
            candidate,
            identity,
            tmp_path,
            tmp_path / "artifacts",
            isaaclab_head="f" * 40,
            resume=False,
            executor=successful_executor,
        )


def test_tuning_command_validator_rejects_any_grammar_change(tmp_path):
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    locked = load_matrix(DEFAULT_MATRIX_PATH)
    candidate = tuning.candidate("cr_iterations_3")
    identity = TuningIdentity("wave1", candidate.name, 42, 4096, 40)
    output_path = tmp_path / "run"
    command = build_tuning_command(locked, tuning, candidate, identity, tmp_path, output_path)

    validate_tuning_command(locked, tuning, candidate, identity, tmp_path, output_path, command)
    for changed in (
        command + ["--extra"],
        command[:-1],
        command[:-2] + command[-1:] + command[-2:-1],
        command[:-1] + ["env.sim.physics.solver_cfg.dynamics_linear_solver_max_iterations=5"],
    ):
        try:
            validate_tuning_command(locked, tuning, candidate, identity, tmp_path, output_path, changed)
        except ValueError as error:
            assert "exactly match" in str(error)
        else:
            raise AssertionError("changed tuning command was accepted")


def write_candidates_decision(path: Path, candidates) -> None:
    """Write resolved candidate records for one adaptive stage."""
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    records = []
    for candidate in candidates:
        resolved = resolve_config(tuning, candidate)
        records.append(
            {
                "name": candidate.name,
                "overrides": candidate.overrides,
                "resolved_config": resolved,
                "config_hash": config_hash(resolved),
            }
        )
    actions = {
        "wave2.json": "resolve-wave2",
        "stage2.json": "promote-stage2",
        "finalists.json": "promote-finalists",
    }
    write_json_atomic(
        path,
        {"schema_version": "1.1", "action": actions[path.name], "resolved_candidates": records},
    )


def prepare_decisions(tmp_path: Path) -> None:
    """Write exact-cardinality adaptive decisions for schedule tests."""
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    write_candidates_decision(tmp_path / "wave2.json", tuning.wave1[:6])
    write_candidates_decision(tmp_path / "stage2.json", tuning.wave1[:8])
    write_candidates_decision(tmp_path / "finalists.json", tuning.wave1[:3])
    write_winner_decision(tmp_path, tuning.wave1[0])


def test_measured_stage_schedules_have_exact_cardinalities(tmp_path):
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    prepare_decisions(tmp_path)
    expected = {"baseline": 3, "wave1": 18, "wave2": 6, "halve": 16, "final": 9, "canonical": 3}
    for stage, count in expected.items():
        args = build_parser().parse_args(["--stage", stage, "--decision-root", str(tmp_path), "--measured-only"])
        identities = select_tuning_identities(tuning, args)
        assert len(identities) == count
        assert all(identity.stage == stage for identity in identities)
        assert all(identity.num_envs == 4096 for identity in identities)


def test_preflight_schedules_run_once_per_candidate(tmp_path):
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    prepare_decisions(tmp_path)
    expected = {"baseline": 1, "wave1": 18, "wave2": 6, "halve": 8, "final": 3, "canonical": 1}
    for stage, count in expected.items():
        args = build_parser().parse_args(["--stage", stage, "--decision-root", str(tmp_path), "--preflight-only"])
        identities = select_tuning_identities(tuning, args)
        assert len(identities) == count
        expected_stage = "canonical" if stage == "canonical" else "preflight"
        assert all(identity.stage == expected_stage for identity in identities)
        assert all(identity.seed == 42 and identity.max_iterations == 5 for identity in identities)


def test_candidate_filter_keeps_one_candidate_seed_schedule(tmp_path):
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    args = build_parser().parse_args(["--stage", "wave1", "--candidate", "cr_iterations_3", "--measured-only"])

    assert select_tuning_identities(tuning, args) == (TuningIdentity("wave1", "cr_iterations_3", 42, 4096, 40),)


def successful_executor(command, stdout_path, stderr_path, *, timeout_s):
    """Write a complete synthetic schema bundle without launching training."""
    del timeout_s
    output_path = Path(command[command.index("--output_path") + 1])
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "benchmark_training_task.json").write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "run": {"status": "completed"},
                "runtime": {"iterations_completed": int(command[command.index("--max_iterations") + 1])},
            }
        ),
        encoding="utf-8",
    )
    stdout_path.write_text("complete\n", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    return ProcessOutcome(returncode=0, timed_out=False)


def matching_trace(identity: TuningIdentity) -> TrainingTrace:
    """Build a trace whose identity exactly matches one tuning run."""
    return TrainingTrace(
        task=TaskName.ANYMAL_D.value,
        seed=identity.seed,
        num_envs=identity.num_envs,
        iterations=identity.max_iterations,
        iteration_time_s=(),
        collection_fps=(),
        total_fps=(),
        reward=(),
        ep_length=(),
        success_rate=(),
        success_schema_mismatch=False,
        resources={},
    )


def test_execute_tuning_identity_requires_trace_and_hashes_all_evidence(tmp_path, monkeypatch):
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    locked = load_matrix(DEFAULT_MATRIX_PATH)
    candidate = tuning.candidate("cr_iterations_3")
    identity = TuningIdentity("wave1", candidate.name, 42, 4096, 40)
    event_path = tmp_path / "logs" / "events.out.tfevents.test"
    event_path.parent.mkdir()
    event_path.write_bytes(b"event-data")
    monkeypatch.setattr("benchmarks.kamino_dvi.tune.locate_rsl_rl_events", lambda bundle, logs: event_path)
    monkeypatch.setattr(
        "benchmarks.kamino_dvi.tune.parse_training_trace", lambda bundle, event: matching_trace(identity)
    )

    state = execute_tuning_identity(
        locked,
        tuning,
        candidate,
        identity,
        tmp_path,
        tmp_path / "artifacts",
        isaaclab_head="f" * 40,
        resume=False,
        executor=successful_executor,
    )

    manifest = read_tuning_manifest(tmp_path / "artifacts" / identity.run_id / "manifest.json")
    assert state is TerminalState.COMPLETED
    assert manifest.state is TerminalState.COMPLETED
    assert manifest.tensorboard_event_path == str(event_path.resolve())
    assert manifest.tensorboard_event_hash is not None
    assert set(manifest.artifact_hashes) == {
        "stdout.log",
        "stderr.log",
        "benchmark_training_task.json",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (("task", TaskName.ANT.value), ("seed", 43), ("num_envs", 2048), ("iterations", 39)),
)
def test_execute_rejects_trace_identity_mismatch(tmp_path, monkeypatch, field, value):
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    locked = load_matrix(DEFAULT_MATRIX_PATH)
    candidate = tuning.candidate("cr_iterations_3")
    identity = TuningIdentity("wave1", candidate.name, 42, 4096, 40)
    event_path = tmp_path / "events.out.tfevents.test"
    event_path.write_bytes(b"event-data")
    trace = replace(matching_trace(identity), **{field: value})
    monkeypatch.setattr("benchmarks.kamino_dvi.tune.locate_rsl_rl_events", lambda bundle, logs: event_path)
    monkeypatch.setattr("benchmarks.kamino_dvi.tune.parse_training_trace", lambda bundle, event: trace)

    state = execute_tuning_identity(
        locked,
        tuning,
        candidate,
        identity,
        tmp_path,
        tmp_path / "artifacts",
        isaaclab_head="f" * 40,
        resume=False,
        executor=successful_executor,
    )

    manifest = read_tuning_manifest(tmp_path / "artifacts" / identity.run_id / "manifest.json")
    assert state is TerminalState.FAILED
    assert manifest.failure_category is FailureCategory.ARTIFACT


def test_execute_tuning_identity_rejects_missing_metric_as_artifact(tmp_path, monkeypatch):
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    locked = load_matrix(DEFAULT_MATRIX_PATH)
    candidate = tuning.candidate("cr_iterations_3")
    identity = TuningIdentity("wave1", candidate.name, 42, 4096, 40)
    event_path = tmp_path / "events.out.tfevents.test"
    event_path.write_bytes(b"event-data")
    monkeypatch.setattr("benchmarks.kamino_dvi.tune.locate_rsl_rl_events", lambda bundle, logs: event_path)

    def reject_trace(bundle, event):
        raise MissingBenchmarkFieldError("TensorBoard:Perf/total_fps")

    monkeypatch.setattr("benchmarks.kamino_dvi.tune.parse_training_trace", reject_trace)
    state = execute_tuning_identity(
        locked,
        tuning,
        candidate,
        identity,
        tmp_path,
        tmp_path / "artifacts",
        isaaclab_head="f" * 40,
        resume=False,
        executor=successful_executor,
    )

    manifest = read_tuning_manifest(tmp_path / "artifacts" / identity.run_id / "manifest.json")
    assert state is TerminalState.FAILED
    assert manifest.failure_category is FailureCategory.ARTIFACT


def test_failed_attempt_is_preserved_and_retry_links_to_parent(tmp_path):
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    locked = load_matrix(DEFAULT_MATRIX_PATH)
    candidate = tuning.candidate("cr_iterations_3")
    identity = TuningIdentity("wave1", candidate.name, 42, 4096, 40)

    def crash(command, stdout_path, stderr_path, *, timeout_s):
        del command, timeout_s
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("RuntimeError: crash\n", encoding="utf-8")
        return ProcessOutcome(returncode=1, timed_out=False)

    for resume in (False, True):
        assert (
            execute_tuning_identity(
                locked,
                tuning,
                candidate,
                identity,
                tmp_path,
                tmp_path / "artifacts",
                isaaclab_head="f" * 40,
                resume=resume,
                executor=crash,
            )
            is TerminalState.FAILED
        )

    first = read_tuning_manifest(tmp_path / "artifacts" / identity.run_id / "manifest.json")
    retry_identity = TuningIdentity("wave1", candidate.name, 42, 4096, 40, 1)
    second = read_tuning_manifest(tmp_path / "artifacts" / retry_identity.run_id / "manifest.json")
    assert first.failure_category is FailureCategory.CRASH
    assert second.retry == RetryLineage(attempt=1, parent_run_id=first.run_id)
    assert first.run_id != second.run_id


def test_resume_skips_exact_evidence_but_tampered_event_creates_retry(tmp_path, monkeypatch):
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    locked = load_matrix(DEFAULT_MATRIX_PATH)
    candidate = tuning.candidate("cr_iterations_3")
    identity = TuningIdentity("wave1", candidate.name, 42, 4096, 40)
    event_path = tmp_path / "events.out.tfevents.test"
    event_path.write_bytes(b"original")
    monkeypatch.setattr("benchmarks.kamino_dvi.tune.locate_rsl_rl_events", lambda bundle, logs: event_path)
    monkeypatch.setattr(
        "benchmarks.kamino_dvi.tune.parse_training_trace", lambda bundle, event: matching_trace(identity)
    )
    calls = 0

    def executor(*args, **kwargs):
        nonlocal calls
        calls += 1
        return successful_executor(*args, **kwargs)

    for resume in (False, True):
        assert (
            execute_tuning_identity(
                locked,
                tuning,
                candidate,
                identity,
                tmp_path,
                tmp_path / "artifacts",
                isaaclab_head="f" * 40,
                resume=resume,
                executor=executor,
            )
            is TerminalState.COMPLETED
        )
    assert calls == 1

    event_path.write_bytes(b"tampered")
    assert (
        execute_tuning_identity(
            locked,
            tuning,
            candidate,
            identity,
            tmp_path,
            tmp_path / "artifacts",
            isaaclab_head="f" * 40,
            resume=True,
            executor=executor,
        )
        is TerminalState.COMPLETED
    )
    assert calls == 2
    retry = TuningIdentity("wave1", candidate.name, 42, 4096, 40, 1)
    assert (tmp_path / "artifacts" / retry.run_id / "manifest.json").is_file()


def test_preflight_gate_requires_exact_config_and_source_head(tmp_path, monkeypatch):
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    locked = load_matrix(DEFAULT_MATRIX_PATH)
    candidate = tuning.candidate("cr_iterations_3")
    identity = TuningIdentity("preflight", candidate.name, 42, 4096, 5)
    event_path = tmp_path / "events.out.tfevents.test"
    event_path.write_bytes(b"event")
    monkeypatch.setattr("benchmarks.kamino_dvi.tune.locate_rsl_rl_events", lambda bundle, logs: event_path)
    monkeypatch.setattr(
        "benchmarks.kamino_dvi.tune.parse_training_trace", lambda bundle, event: matching_trace(identity)
    )
    assert not preflight_completed(locked, tuning, candidate, tmp_path, tmp_path / "artifacts", "f" * 40)
    execute_tuning_identity(
        locked,
        tuning,
        candidate,
        identity,
        tmp_path,
        tmp_path / "artifacts",
        isaaclab_head="f" * 40,
        resume=False,
        executor=successful_executor,
    )
    assert preflight_completed(locked, tuning, candidate, tmp_path, tmp_path / "artifacts", "f" * 40)
    assert not preflight_completed(locked, tuning, candidate, tmp_path, tmp_path / "artifacts", "0" * 40)


def test_measured_only_cli_refuses_absent_preflight(tmp_path, monkeypatch):
    package = PackageLocation(
        module_path=str(tmp_path / "source/isaaclab_newton/isaaclab_newton/__init__.py"),
        distribution_path=str(tmp_path / "source/isaaclab_newton"),
        direct_url={},
    )
    monkeypatch.setattr(
        "benchmarks.kamino_dvi.tune._validated_environment",
        lambda matrix, root: SimpleNamespace(isaaclab=SimpleNamespace(head="f" * 40), isaaclab_newton=package),
    )
    with pytest.raises(RuntimeError, match="completed exact preflight"):
        main(
            [
                "--stage",
                "wave1",
                "--candidate",
                "cr_iterations_3",
                "--measured-only",
                "--artifact-root",
                str(tmp_path / "artifacts"),
            ]
        )


def test_unreadable_attempt_directory_is_never_overwritten(tmp_path):
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    locked = load_matrix(DEFAULT_MATRIX_PATH)
    candidate = tuning.candidate("cr_iterations_3")
    identity = TuningIdentity("wave1", candidate.name, 42, 4096, 40)
    artifact_root = tmp_path / "artifacts"
    raw_path = artifact_root / identity.run_id
    raw_path.mkdir(parents=True)
    (raw_path / "partial.log").write_text("raw evidence", encoding="utf-8")

    def crash(command, stdout_path, stderr_path, *, timeout_s):
        del command, timeout_s
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("RuntimeError: crash\n", encoding="utf-8")
        return ProcessOutcome(returncode=1, timed_out=False)

    assert (
        execute_tuning_identity(
            locked,
            tuning,
            candidate,
            identity,
            tmp_path,
            artifact_root,
            isaaclab_head="f" * 40,
            resume=True,
            executor=crash,
        )
        is TerminalState.FAILED
    )
    retry = TuningIdentity("wave1", candidate.name, 42, 4096, 40, 1)
    assert (raw_path / "partial.log").read_text(encoding="utf-8") == "raw evidence"
    assert (artifact_root / retry.run_id / "manifest.json").is_file()


def _patch_successful_trace(monkeypatch, tmp_path: Path, identity: TuningIdentity) -> Path:
    event_path = tmp_path / "events.out.tfevents.test"
    event_path.write_bytes(b"event")
    monkeypatch.setattr("benchmarks.kamino_dvi.tune.locate_rsl_rl_events", lambda bundle, logs: event_path)
    monkeypatch.setattr(
        "benchmarks.kamino_dvi.tune.parse_training_trace", lambda bundle, event: matching_trace(identity)
    )
    return event_path


def _crashing_executor(command, stdout_path, stderr_path, *, timeout_s):
    del command, timeout_s
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("RuntimeError: crash\n", encoding="utf-8")
    return ProcessOutcome(returncode=1, timed_out=False)


def test_resume_does_not_accept_older_completion_after_latest_failure(tmp_path, monkeypatch):
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    locked = load_matrix(DEFAULT_MATRIX_PATH)
    candidate = tuning.candidate("cr_iterations_3")
    identity = TuningIdentity("wave1", candidate.name, 42, 4096, 40)
    _patch_successful_trace(monkeypatch, tmp_path, identity)
    artifact_root = tmp_path / "artifacts"
    execute_tuning_identity(
        locked,
        tuning,
        candidate,
        identity,
        tmp_path,
        artifact_root,
        isaaclab_head="f" * 40,
        resume=False,
        executor=successful_executor,
    )
    execute_tuning_identity(
        locked,
        tuning,
        candidate,
        identity,
        tmp_path,
        artifact_root,
        isaaclab_head="f" * 40,
        resume=False,
        executor=_crashing_executor,
    )
    calls = 0

    def executor(*args, **kwargs):
        nonlocal calls
        calls += 1
        return successful_executor(*args, **kwargs)

    assert (
        execute_tuning_identity(
            locked,
            tuning,
            candidate,
            identity,
            tmp_path,
            artifact_root,
            isaaclab_head="f" * 40,
            resume=True,
            executor=executor,
        )
        is TerminalState.COMPLETED
    )
    assert calls == 1
    assert (artifact_root / replace(identity, attempt=2).run_id / "manifest.json").is_file()


@pytest.mark.parametrize("mutation", ("artifact_root", "missing_hash"))
def test_resume_rejects_copied_root_and_incomplete_hash_set(tmp_path, monkeypatch, mutation):
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    locked = load_matrix(DEFAULT_MATRIX_PATH)
    candidate = tuning.candidate("cr_iterations_3")
    identity = TuningIdentity("wave1", candidate.name, 42, 4096, 40)
    _patch_successful_trace(monkeypatch, tmp_path, identity)
    artifact_root = tmp_path / "artifacts"
    execute_tuning_identity(
        locked,
        tuning,
        candidate,
        identity,
        tmp_path,
        artifact_root,
        isaaclab_head="f" * 40,
        resume=False,
        executor=successful_executor,
    )
    original_path = artifact_root / identity.run_id / "manifest.json"
    original = read_tuning_manifest(original_path)
    if mutation == "artifact_root":
        copied_identity = replace(identity, attempt=1)
        copied_path = artifact_root / copied_identity.run_id / "manifest.json"
        copied_path.parent.mkdir()
        write_tuning_manifest(
            copied_path,
            replace(
                original,
                run_id=copied_identity.run_id,
                identity=copied_identity,
                retry=RetryLineage(attempt=1, parent_run_id=original.run_id),
            ),
        )
        expected_attempt = 2
    else:
        hashes = dict(original.artifact_hashes)
        hashes.pop("stderr.log")
        write_tuning_manifest(original_path, replace(original, artifact_hashes=hashes))
        expected_attempt = 1
    calls = 0

    def executor(*args, **kwargs):
        nonlocal calls
        calls += 1
        return successful_executor(*args, **kwargs)

    execute_tuning_identity(
        locked,
        tuning,
        candidate,
        identity,
        tmp_path,
        artifact_root,
        isaaclab_head="f" * 40,
        resume=True,
        executor=executor,
    )
    assert calls == 1
    assert (artifact_root / replace(identity, attempt=expected_attempt).run_id / "manifest.json").is_file()


def test_new_attempt_is_above_highest_occupied_and_uses_latest_valid_parent(tmp_path):
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    locked = load_matrix(DEFAULT_MATRIX_PATH)
    candidate = tuning.candidate("cr_iterations_3")
    identity = TuningIdentity("wave1", candidate.name, 42, 4096, 40)
    artifact_root = tmp_path / "artifacts"
    execute_tuning_identity(
        locked,
        tuning,
        candidate,
        identity,
        tmp_path,
        artifact_root,
        isaaclab_head="f" * 40,
        resume=False,
        executor=_crashing_executor,
    )
    (artifact_root / replace(identity, attempt=2).run_id).mkdir()

    execute_tuning_identity(
        locked,
        tuning,
        candidate,
        identity,
        tmp_path,
        artifact_root,
        isaaclab_head="f" * 40,
        resume=True,
        executor=_crashing_executor,
    )

    latest = read_tuning_manifest(artifact_root / replace(identity, attempt=3).run_id / "manifest.json")
    assert latest.retry == RetryLineage(attempt=3, parent_run_id=identity.run_id)


@pytest.mark.parametrize(
    ("failure_point", "category"),
    (
        ("execute", FailureCategory.CRASH),
        ("bundle", FailureCategory.ARTIFACT),
        ("locate", FailureCategory.ARTIFACT),
        ("parse", FailureCategory.ARTIFACT),
        ("hash", FailureCategory.ARTIFACT),
    ),
)
def test_post_running_exceptions_always_persist_failed_manifest(tmp_path, monkeypatch, failure_point, category):
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    locked = load_matrix(DEFAULT_MATRIX_PATH)
    candidate = tuning.candidate("cr_iterations_3")
    identity = TuningIdentity("wave1", candidate.name, 42, 4096, 40)
    event_path = tmp_path / "events.out.tfevents.test"
    event_path.write_bytes(b"event")
    monkeypatch.setattr("benchmarks.kamino_dvi.tune.locate_rsl_rl_events", lambda bundle, logs: event_path)
    monkeypatch.setattr(
        "benchmarks.kamino_dvi.tune.parse_training_trace", lambda bundle, event: matching_trace(identity)
    )
    executor = successful_executor

    def raise_error(*args, **kwargs):
        del args, kwargs
        raise RuntimeError(f"{failure_point} failed")

    if failure_point == "execute":
        executor = raise_error
    elif failure_point == "bundle":
        monkeypatch.setattr("benchmarks.kamino_dvi.tune.inspect_bundle", raise_error)
    elif failure_point == "locate":
        monkeypatch.setattr("benchmarks.kamino_dvi.tune.locate_rsl_rl_events", raise_error)
    elif failure_point == "parse":
        monkeypatch.setattr("benchmarks.kamino_dvi.tune.parse_training_trace", raise_error)
    else:
        monkeypatch.setattr("benchmarks.kamino_dvi.tune.sha256_file", raise_error)

    state = execute_tuning_identity(
        locked,
        tuning,
        candidate,
        identity,
        tmp_path,
        tmp_path / "artifacts",
        isaaclab_head="f" * 40,
        resume=False,
        executor=executor,
    )

    manifest = read_tuning_manifest(tmp_path / "artifacts" / identity.run_id / "manifest.json")
    assert state is TerminalState.FAILED
    assert manifest.state is TerminalState.FAILED
    assert manifest.failure_category is category


@pytest.mark.parametrize(
    ("filename", "stage", "count", "seeds"),
    (
        ("stage2.json", "halve", 2, 2),
        ("finalists.json", "final", 1, 3),
    ),
)
def test_adaptive_stage_schedules_accept_fewer_survivors(tmp_path, filename, stage, count, seeds):
    """Adaptive schedules accept every nonempty survivor set up to its cap."""
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    write_candidates_decision(tmp_path / filename, tuning.wave1[:count])
    args = build_parser().parse_args(["--stage", stage, "--decision-root", str(tmp_path), "--measured-only"])

    assert len(select_tuning_identities(tuning, args)) == count * seeds


@pytest.mark.parametrize(
    ("filename", "stage", "count"),
    (
        ("stage2.json", "halve", 0),
        ("stage2.json", "halve", 9),
        ("finalists.json", "final", 0),
        ("finalists.json", "final", 4),
    ),
)
def test_adaptive_stage_schedules_reject_empty_or_above_cap(tmp_path, filename, stage, count):
    """Adaptive decisions cannot schedule zero candidates or exceed their cap."""
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    write_candidates_decision(tmp_path / filename, tuning.wave1[:count])
    args = build_parser().parse_args(["--stage", stage, "--decision-root", str(tmp_path), "--measured-only"])

    with pytest.raises(ValueError, match="between 1 and"):
        select_tuning_identities(tuning, args)


def test_runner_rejects_wrong_decision_schema_or_action(tmp_path):
    """Adaptive schedules never trust a candidate file for the wrong action."""
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    write_candidates_decision(tmp_path / "stage2.json", tuning.wave1[:2])
    data = json.loads((tmp_path / "stage2.json").read_text(encoding="utf-8"))
    data["action"] = "promote-finalists"
    write_json_atomic(tmp_path / "stage2.json", data)
    args = build_parser().parse_args(["--stage", "halve", "--decision-root", str(tmp_path), "--measured-only"])

    with pytest.raises(ValueError, match="schema/action"):
        select_tuning_identities(tuning, args)


def test_tampered_adaptive_decision_is_rejected_before_source_probe(tmp_path, monkeypatch):
    """Strict raw-evidence recomputation must gate every adaptive launch."""
    source_probed = False

    def reject(*_args, **_kwargs):
        raise ValueError("persisted decision does not match recomputed raw evidence")

    def source_probe(*_args, **_kwargs):
        nonlocal source_probed
        source_probed = True
        raise AssertionError("source probe must not run")

    monkeypatch.setattr(tune_module, "_validate_adaptive_decisions", reject)
    monkeypatch.setattr(tune_module, "_validated_environment", source_probe)

    with pytest.raises(ValueError, match="does not match recomputed raw evidence"):
        main(
            [
                "--stage",
                "wave2",
                "--artifact-root",
                str(tmp_path / "artifacts"),
                "--decision-root",
                str(tmp_path / "decisions"),
            ]
        )

    assert source_probed is False


def test_exact_adaptive_decision_reaches_source_probe(tmp_path, monkeypatch):
    """An exact recomputed decision may proceed to environment validation."""
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    decisions = tmp_path / "decisions"
    write_candidates_decision(decisions / "wave2.json", tuning.wave1[:6])
    monkeypatch.setattr(tune_module, "_validate_adaptive_decisions", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        tune_module,
        "_validated_environment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("source probe reached")),
    )

    with pytest.raises(RuntimeError, match="source probe reached"):
        main(["--stage", "wave2", "--decision-root", str(decisions)])
