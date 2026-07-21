# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for strict staged tuning analysis and canonical decisions."""

from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path

import pytest

from benchmarks.kamino_dvi import analyze_tuning as analysis_module
from benchmarks.kamino_dvi.analyze_tuning import (
    TuningRecord,
    _canonical_comparison,
    _qualification_dict,
    build_parser,
    derive_stage2_baseline,
    load_decision,
    load_tuning_records,
    main,
    reconcile_decision_candidates,
    validate_tuning_records,
)
from benchmarks.kamino_dvi.manifests import command_hash, sha256_file
from benchmarks.kamino_dvi.matrix import DEFAULT_MATRIX_PATH, load_matrix
from benchmarks.kamino_dvi.parsing import TrainingTrace
from benchmarks.kamino_dvi.statistics import Estimate
from benchmarks.kamino_dvi.tune import (
    ResolvedTuningCandidate,
    TuningIdentity,
    _command_for_candidate,
    build_tuning_command,
)
from benchmarks.kamino_dvi.tuning import (
    DEFAULT_TUNING_MATRIX_PATH,
    FinalQualification,
    TuningRunMetrics,
    config_hash,
    load_tuning_matrix,
    resolve_config,
)


def _trace(identity: TuningIdentity, *, offset: float = 0.0) -> TrainingTrace:
    """Build an aligned finite trace for one synthetic completed run."""
    values = tuple(offset + float(index) for index in range(identity.max_iterations))
    return TrainingTrace(
        task="Isaac-Velocity-Flat-AnymalD",
        seed=identity.seed,
        num_envs=identity.num_envs,
        iterations=identity.max_iterations,
        iteration_time_s=tuple(0.1 + value / 10000 for value in values),
        collection_fps=(1.0,) * identity.max_iterations,
        total_fps=(1.0,) * identity.max_iterations,
        reward=values,
        ep_length=tuple(100.0 + value for value in values),
        success_rate=tuple(0.5 + value / 1000 for value in values),
        success_schema_mismatch=False,
        resources={},
    )


def _write_completed_artifact(tmp_path: Path, monkeypatch, candidate_name: str = "integrator_euler") -> Path:
    """Write one complete provenance-consistent Wave 1 artifact."""
    repo_root = Path(__file__).resolve().parents[3]
    artifact_root = tmp_path / "artifacts"
    logs_root = tmp_path / "logs"
    logs_root.mkdir()
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    candidate = tuning.candidate(candidate_name)
    identity = TuningIdentity("wave1", candidate.name, 42, 4096, 40)
    run_dir = artifact_root / identity.run_id
    run_dir.mkdir(parents=True)
    command = tuple(build_tuning_command(matrix, tuning, candidate, identity, repo_root, run_dir))
    event_path = logs_root / "events.out.tfevents.test"
    event_path.write_bytes(b"event")
    bundle_path = run_dir / "benchmark_training_test.json"
    source_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    bundle_path.write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "run": {
                    "status": "completed",
                    "task": tuning.task,
                    "seed": identity.seed,
                    "num_envs": identity.num_envs,
                    "max_iterations": identity.max_iterations,
                    "end_time_utc": "2026-07-21T12:00:00+00:00",
                },
                "runtime": {"iterations_completed": identity.max_iterations},
                "versions": {"git_commit": source_head, "git_dirty": False},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    for name, content in (("stdout.log", "ok\n"), ("stderr.log", "")):
        (run_dir / name).write_text(content, encoding="utf-8")
    resolved = resolve_config(tuning, candidate)
    manifest = {
        "run_id": identity.run_id,
        "identity": dataclasses.asdict(identity),
        "config_hash": config_hash(resolved),
        "resolved_config": resolved,
        "command": command,
        "command_hash": command_hash(command),
        "revisions": dataclasses.asdict(matrix.revisions),
        "schema_version": "1.1",
        "isaaclab_head": source_head,
        "artifact_root": str(run_dir),
        "tensorboard_event_path": str(event_path.resolve()),
        "tensorboard_event_hash": sha256_file(event_path),
        "artifact_hashes": {
            name: sha256_file(run_dir / name) for name in ("stdout.log", "stderr.log", bundle_path.name)
        },
        "state": "completed",
        "failure_category": None,
        "retry": {"attempt": 0, "parent_run_id": None},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr("benchmarks.kamino_dvi.analyze_tuning.parse_training_trace", lambda *_: _trace(identity))
    return artifact_root


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("config_hash", "0" * 64, "config hash"),
        ("command_hash", "0" * 64, "command hash"),
        ("tensorboard_event_hash", "0" * 64, "event hash"),
        ("isaaclab_head", "0" * 40, "source HEAD"),
    ),
)
def test_loader_rejects_mismatched_provenance(tmp_path, monkeypatch, field: str, value: str, message: str):
    """A completed tuning record is accepted only with exact retained provenance."""
    artifact_root = _write_completed_artifact(tmp_path, monkeypatch)
    manifest_path = next(artifact_root.glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_tuning_records(artifact_root, tmp_path / "logs", expected_stage="wave1")


def test_loader_rejects_incomplete_expected_candidate_seed_set():
    """Coverage validation rejects a partial decision input before aggregation."""
    metric = TuningRunMetrics("integrator_euler", "wave1", 42, 4096, (1.0,) * 40, (1.0,) * 40, (1.0,) * 40, (1.0,) * 40)
    record = TuningRecord(metric, "run", Path("manifest.json"), "a" * 64, Path("event"), "b" * 64, "c" * 64, {})
    with pytest.raises(ValueError, match="missing tuning record"):
        validate_tuning_records([record], (("integrator_euler", 42), ("cr_iterations_3", 42)))


def test_stage2_baseline_uses_first_hundred_aligned_points():
    """The Stage 2 comparator uses iterations 1--100, not the final 100."""
    metric = TuningRunMetrics(
        "baseline", "baseline", 42, 4096, tuple(range(300)), tuple(range(300)), tuple(range(300)), tuple(range(300))
    )
    source = TuningRecord(metric, "run", Path("manifest.json"), "a" * 64, Path("event"), "b" * 64, "c" * 64, {})
    derived, provenance = derive_stage2_baseline([source], 100)
    assert derived[0].stage == "halve"
    assert derived[0].reward == tuple(range(100))
    assert derived[0].final_mean(derived[0].reward, 20) == pytest.approx(89.5)
    assert provenance[0]["derivation"] == "first 100 aligned iterations of clean 300-iteration baseline"
    assert provenance[0]["source_manifest_hash"] == "a" * 64


def test_resolve_wave2_writes_resolved_configs_and_canonical_hashes(tmp_path, monkeypatch):
    """Wave 2 decisions retain every literal cumulative configuration."""
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    records = []
    for index, candidate in enumerate(tuning.wave1):
        metric = TuningRunMetrics(
            candidate.name,
            "wave1",
            42,
            4096,
            (1.0,) * 10 + (0.1 + index / 1000,) * 30,
            (1.0,) * 40,
            (1.0,) * 40,
            (1.0,) * 40,
        )
        resolved = resolve_config(tuning, candidate)
        records.append(
            TuningRecord(
                metric,
                candidate.name,
                Path(candidate.name),
                f"{index:064x}",
                Path("event"),
                "e" * 64,
                config_hash(resolved),
                resolved,
            )
        )
    monkeypatch.setattr("benchmarks.kamino_dvi.analyze_tuning.load_tuning_records", lambda *_args, **_kwargs: records)
    output = tmp_path / "wave2.json"
    argv = ["resolve-wave2", "--artifact-root", str(tmp_path), "--logs-root", str(tmp_path), "--output", str(output)]
    assert main(argv) == 0
    decision = json.loads(output.read_text(encoding="utf-8"))
    assert decision["schema_version"] == "1.0"
    assert decision["action"] == "resolve-wave2"
    assert decision["source_stage"] == "wave1"
    assert len(decision["resolved_candidates"]) == 6
    for item in decision["resolved_candidates"]:
        assert item["config_hash"] == config_hash(item["resolved_config"])
        assert item["resolved_config"] == {**tuning.baseline, **item["overrides"]}
    first = output.read_bytes()
    main(argv)
    assert output.read_bytes() == first


def test_failed_qualification_serializes_without_nonfinite_json():
    """Rejected finalist estimates use JSON null while retaining the failure reason."""
    invalid = Estimate(float("nan"), float("nan"), 0)
    qualification = FinalQualification("failed", False, "seed 42: numerical", invalid, invalid, invalid, invalid)

    data = _qualification_dict({"failed": qualification})

    assert data["failed"]["runtime"] is None
    json.dumps(data, allow_nan=False)


def test_loader_rejects_undeclared_artifact_directory(tmp_path):
    """An unexpected child directory cannot be silently excluded from the audit."""
    artifact_root = tmp_path / "artifacts"
    (artifact_root / "mystery").mkdir(parents=True)

    with pytest.raises(ValueError, match="undeclared tuning directory"):
        load_tuning_records(artifact_root, tmp_path / "logs", expected_stage="wave1")


@pytest.mark.parametrize(("field", "value"), (("seed", 42.0), ("attempt", False)))
def test_loader_rejects_noninteger_identity_fields(tmp_path, monkeypatch, field, value):
    """Identity integers reject float and bool values before semantic validation."""
    artifact_root = _write_completed_artifact(tmp_path, monkeypatch)
    manifest_path = next(artifact_root.glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"][field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="typed identity"):
        load_tuning_records(artifact_root, tmp_path / "logs", expected_stage="wave1")


def test_loader_rejects_wrong_stage_iteration_count(tmp_path, monkeypatch):
    """Stage protocol iterations are checked independently from trace lengths."""
    artifact_root = _write_completed_artifact(tmp_path, monkeypatch)
    manifest_path = next(artifact_root.glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["max_iterations"] = 41
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="wave1 requires exactly 40 iterations"):
        load_tuning_records(artifact_root, tmp_path / "logs", expected_stage="wave1")


def test_loader_rejects_retry_attempt_mismatch(tmp_path, monkeypatch):
    """Retry lineage attempt must equal the immutable identity attempt."""
    artifact_root = _write_completed_artifact(tmp_path, monkeypatch)
    manifest_path = next(artifact_root.glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["retry"]["attempt"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="retry attempt"):
        load_tuning_records(artifact_root, tmp_path / "logs", expected_stage="wave1")


def test_loader_rejects_run_directory_symlink_escape(tmp_path, monkeypatch):
    """A tuning run directory must resolve beneath the declared artifact root."""
    artifact_root = _write_completed_artifact(tmp_path, monkeypatch)
    run_dir = next(path for path in artifact_root.iterdir() if path.is_dir())
    escaped = tmp_path / "escaped"
    run_dir.rename(escaped)
    run_dir.symlink_to(escaped, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes artifact root"):
        load_tuning_records(artifact_root, tmp_path / "logs", expected_stage="wave1")


def test_strict_decision_parser_rejects_tampered_resolved_provenance(tmp_path):
    """Persisted decision candidates are never trusted without recomputation."""
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    candidate = tuning.wave1[0]
    resolved = resolve_config(tuning, candidate)
    path = tmp_path / "decision.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "action": "promote-stage2",
                "source_stage": "stage1",
                "source_artifact_root": str(tmp_path),
                "source_manifests": [],
                "timestamp_utc": "2026-07-21T00:00:00+00:00",
                "revisions": dataclasses.asdict(load_matrix(DEFAULT_MATRIX_PATH).revisions),
                "selected": [candidate.name],
                "rejected": {},
                "resolved_candidates": [
                    {
                        "name": candidate.name,
                        "overrides": candidate.overrides,
                        "resolved_config": {**resolved, "dvi_block_iterations": 999},
                        "config_hash": config_hash(resolved),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="decision candidate provenance"):
        load_decision(path, "promote-stage2", tuning, minimum_count=1, maximum_count=8)


def test_decision_candidates_reconcile_with_downstream_manifests():
    """A downstream manifest config must equal its persisted candidate decision."""
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    candidate = tuning.wave1[0]
    resolved = resolve_config(tuning, candidate)
    metric = TuningRunMetrics(candidate.name, "halve", 42, 4096, (1.0,) * 100, (1.0,) * 100, (1.0,) * 100, (1.0,) * 100)
    record = TuningRecord(metric, "run", Path("manifest"), "a" * 64, Path("event"), "b" * 64, "c" * 64, resolved)
    decision = {
        "resolved_candidates": [
            {
                "name": candidate.name,
                "overrides": candidate.overrides,
                "resolved_config": resolved,
                "config_hash": config_hash(resolved),
            }
        ]
    }

    with pytest.raises(ValueError, match="downstream manifest config"):
        reconcile_decision_candidates(decision, [record])


def _three_seed_metrics(candidate: str, stage: str, runtime: float, reward: float, success: float):
    return tuple(
        TuningRunMetrics(
            candidate, stage, seed, 4096, (runtime,) * 300, (reward,) * 300, (success,) * 300, (100.0,) * 300
        )
        for seed in (42, 43, 44)
    )


def test_canonical_gate_requires_overlapping_runtime_reward_and_success(matrix=None):
    """Canonical preset validation must reproduce the override-based winner intervals."""
    del matrix
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    final = _three_seed_metrics("winner", "final", 0.1, 10.0, 0.8)
    canonical = _three_seed_metrics("canonical_winner", "canonical", 0.1, 10.0, 0.8)

    comparison = _canonical_comparison(tuning, final, canonical)
    assert comparison["qualified"] is True

    disjoint = _three_seed_metrics("canonical_winner", "canonical", 0.5, 1.0, 0.1)
    with pytest.raises(ValueError, match="canonical.*runtime"):
        _canonical_comparison(tuning, final, disjoint)


def test_report_parser_exposes_tuning_matrix_for_raw_chain_recomputation(tmp_path):
    """The report command accepts the matrix needed to recompute every decision."""
    matrix_path = tmp_path / "tuning.yaml"

    args = build_parser().parse_args(
        [
            "report",
            "--artifact-root",
            str(tmp_path),
            "--decision-root",
            str(tmp_path),
            "--tuning-matrix",
            str(matrix_path),
        ]
    )

    assert args.tuning_matrix == matrix_path


def _copy_retry_attempt(artifact_root: Path, attempt: int, parent_run_id: str) -> Path:
    """Copy the single synthetic run into a provenance-consistent retry attempt."""
    import shutil

    source_dir = next(path for path in artifact_root.iterdir() if path.is_dir())
    source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    identity = TuningIdentity(**source_manifest["identity"])
    retry_identity = dataclasses.replace(identity, attempt=attempt)
    retry_dir = artifact_root / retry_identity.run_id
    shutil.copytree(source_dir, retry_dir)
    manifest_path = retry_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_id"] = retry_identity.run_id
    manifest["identity"] = dataclasses.asdict(retry_identity)
    manifest["artifact_root"] = str(retry_dir)
    manifest["retry"] = {"attempt": attempt, "parent_run_id": parent_run_id}
    manifest["state"] = "completed"
    manifest["failure_category"] = None
    command = list(manifest["command"])
    command = [argument.replace(str(source_dir), str(retry_dir)) for argument in command]
    manifest["command"] = command
    manifest["command_hash"] = command_hash(command)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return retry_dir


def test_loader_selects_only_highest_terminal_retry_attempt(tmp_path, monkeypatch):
    """A failed attempt followed by a completed retry yields one completed record."""
    artifact_root = _write_completed_artifact(tmp_path, monkeypatch)
    attempt0_dir = next(path for path in artifact_root.iterdir() if path.is_dir())
    attempt0_path = attempt0_dir / "manifest.json"
    attempt0 = json.loads(attempt0_path.read_text(encoding="utf-8"))
    attempt0["state"] = "failed"
    attempt0["failure_category"] = "numerical"
    attempt0_path.write_text(json.dumps(attempt0, sort_keys=True), encoding="utf-8")
    attempt1_dir = _copy_retry_attempt(artifact_root, 1, attempt0["run_id"])

    records = load_tuning_records(artifact_root, tmp_path / "logs", expected_stage="wave1")

    assert len(records) == 1
    assert records[0].run_id == attempt1_dir.name
    assert records[0].metrics.failure is None


def test_loader_rejects_retry_with_wrong_parent(tmp_path, monkeypatch):
    """Every later readable attempt must name the preceding attempt as parent."""
    artifact_root = _write_completed_artifact(tmp_path, monkeypatch)
    _copy_retry_attempt(artifact_root, 1, "wrong-parent")

    with pytest.raises(ValueError, match="retry parent"):
        load_tuning_records(artifact_root, tmp_path / "logs", expected_stage="wave1")


def test_stage_funnel_separates_wave1_and_wave2_origins():
    """Wave 2 promotion counts only selected candidates originating in Wave 2."""

    def record(name: str, stage: str, failure: str | None = None) -> TuningRecord:
        metrics = TuningRunMetrics(name, stage, 42, 4096, (1.0,), (1.0,), (1.0,), (1.0,), failure)
        return TuningRecord(metrics, name, Path(name), "a" * 64, None, None, "b" * 64, {})

    chain = {
        "baseline": [record("baseline", "baseline") for _ in range(3)],
        "wave1": [record("w1_good", "wave1"), record("w1_bad", "wave1", "numerical")],
        "wave2": [record("w2_good", "wave2"), record("w2_bad", "wave2", "timeout")],
        "halve": [record("w1_good", "halve"), record("w2_good", "halve")],
        "final": [record("w2_good", "final")],
        "wave2_decision": {"selected": [f"derived_{index}" for index in range(6)]},
        "stage2_decision": {"selected": ["w1_good", "w2_good"]},
        "finalists_decision": {"selected": ["w2_good"], "rejected": {"w1_good": "slower"}},
    }
    winner = {"rejected": {}}

    rows = {
        row["stage"]: row
        for row in analysis_module._stage_funnel(chain, [record("canonical_winner", "canonical")], winner)
    }

    assert rows["Wave 1"] == {"stage": "Wave 1", "attempted": 2, "valid": 1, "rejected": 1, "promoted": 6}
    assert rows["Wave 2"]["rejected"] == 1
    assert rows["Wave 2"]["promoted"] == 1
    assert rows["Wave 2"]["selected_from_wave1"] == 1


def _add_canonical_artifact(source_dir: Path, identity: TuningIdentity) -> Path:
    """Copy a fixture into one exact canonical preflight or measured artifact."""
    import shutil

    artifact_root = source_dir.parent
    run_dir = artifact_root / identity.run_id
    shutil.copytree(source_dir, run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    resolved = dict(manifest["resolved_config"])
    candidate = ResolvedTuningCandidate("canonical_winner", {}, resolved, config_hash(resolved))
    repo_root = Path(__file__).resolve().parents[3]
    command = tuple(_command_for_candidate(matrix, tuning, candidate, identity, repo_root, run_dir))
    bundle_path = next(run_dir.glob("benchmark_training_*.json"))
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["run"].update(seed=identity.seed, max_iterations=identity.max_iterations)
    bundle["runtime"]["iterations_completed"] = identity.max_iterations
    bundle_path.write_text(json.dumps(bundle, sort_keys=True), encoding="utf-8")
    manifest.update(
        run_id=identity.run_id,
        identity=dataclasses.asdict(identity),
        config_hash=config_hash(resolved),
        resolved_config=resolved,
        command=command,
        command_hash=command_hash(command),
        artifact_root=str(run_dir),
        artifact_hashes={name: sha256_file(run_dir / name) for name in ("stdout.log", "stderr.log", bundle_path.name)},
        state="completed",
        failure_category=None,
        retry={"attempt": 0, "parent_run_id": None},
    )
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return run_dir


def test_canonical_preflight_is_excluded_before_report_ci_gate(tmp_path, monkeypatch):
    """Normal canonical preflight plus three measured seeds reaches the CI gate."""
    artifact_root = _write_completed_artifact(tmp_path, monkeypatch)
    source_dir = next(path for path in artifact_root.iterdir() if path.is_dir())
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    identities = [
        TuningIdentity("canonical", "canonical_winner", 42, tuning.num_envs, tuning.preflight_iterations),
        *[
            TuningIdentity("canonical", "canonical_winner", seed, tuning.num_envs, tuning.final_iterations)
            for seed in tuning.seeds
        ],
    ]
    for identity in identities:
        _add_canonical_artifact(source_dir, identity)

    def parse(bundle_path: Path, _event_path: Path) -> TrainingTrace:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        run = bundle["run"]
        identity = TuningIdentity("canonical", "canonical_winner", run["seed"], run["num_envs"], run["max_iterations"])
        return _trace(identity)

    monkeypatch.setattr("benchmarks.kamino_dvi.analyze_tuning.parse_training_trace", parse)
    records = load_tuning_records(artifact_root, tmp_path / "logs", expected_stage="canonical")

    validate_tuning_records(records, (("canonical_winner", seed) for seed in tuning.seeds))
    comparison = _canonical_comparison(
        tuning, [record.metrics for record in records], [record.metrics for record in records]
    )
    assert comparison["qualified"] is True
    assert {record.metrics.seed for record in records} == set(tuning.seeds)

    def as_stage(record: TuningRecord, candidate: str, stage: str) -> TuningRecord:
        source = record.metrics
        metrics = TuningRunMetrics(
            candidate,
            stage,
            source.seed,
            source.num_envs,
            source.iteration_time_s,
            source.reward,
            source.success_rate,
            source.ep_length,
        )
        return dataclasses.replace(record, metrics=metrics, run_id=f"{stage}-{source.seed}")

    baseline = [as_stage(record, "baseline", "baseline") for record in records]
    final = [as_stage(record, "winner", "final") for record in records]
    estimates = comparison["override_final"]
    qualification = {
        "qualified": True,
        "reason": None,
        "runtime": estimates["runtime"],
        "reward": estimates["reward"],
        "success_rate": estimates["success"],
        "ep_length": estimates["episode_length"],
    }
    winner = {
        "candidate": "winner",
        "resolved_config": records[0].resolved_config,
        "config_hash": records[0].config_hash,
        "qualifications": {"winner": qualification},
        "rejected": {},
    }
    chain = {
        "matrix": tuning,
        "baseline": baseline,
        "wave1": [],
        "wave2": [],
        "halve": [],
        "final": final,
        "wave2_decision": {"selected": [], "rejected": {}},
        "stage2_decision": {"selected": [], "rejected": {}},
        "finalists_decision": {"selected": ["winner"], "rejected": {}},
        "winner_decision": winner,
    }
    comparison_summary = tmp_path / "comparison.json"
    comparison_summary.write_text(
        json.dumps(
            [
                {"task": tuning.task, "variant": "mjwarp", "iteration_time_s": {"mean": 0.2}},
                {"task": tuning.task, "variant": "physx", "iteration_time_s": {"mean": 0.3}},
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(analysis_module, "_recompute_chain", lambda *_args: chain)
    monkeypatch.setattr(analysis_module, "_require_persisted", lambda *_args: winner)
    output_dir = tmp_path / "report"

    assert (
        main(
            [
                "report",
                "--artifact-root",
                str(artifact_root),
                "--logs-root",
                str(tmp_path / "logs"),
                "--decision-root",
                str(tmp_path),
                "--comparison-summary",
                str(comparison_summary),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["canonical_comparison"]["qualified"] is True


@pytest.mark.parametrize(("field", "value"), (("candidate", "wrong"), ("seed", 43)))
def test_loader_rejects_malformed_five_iteration_canonical_identity(tmp_path, monkeypatch, field, value):
    """Only canonical_winner seed 42 is the canonical five-step preflight."""
    artifact_root = _write_completed_artifact(tmp_path, monkeypatch)
    source_dir = next(path for path in artifact_root.iterdir() if path.is_dir())
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    preflight = TuningIdentity("canonical", "canonical_winner", 42, tuning.num_envs, tuning.preflight_iterations)
    run_dir = _add_canonical_artifact(source_dir, preflight)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"][field] = value
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid canonical preflight identity"):
        load_tuning_records(artifact_root, tmp_path / "logs", expected_stage="canonical")
