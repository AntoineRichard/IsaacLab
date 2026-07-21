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
    _compute_finalists,
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
    stage_candidates,
)
from benchmarks.kamino_dvi.tuning import (
    DEFAULT_TUNING_MATRIX_PATH,
    FinalQualification,
    TuningRunMetrics,
    config_hash,
    load_tuning_matrix,
    resolve_config,
)
from benchmarks.kamino_dvi.tuning_reporting import _paginate_text


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


def _synthetic_record(
    candidate: str,
    stage: str,
    seed: int,
    iterations: int,
    resolved_config: dict,
    *,
    reward: float = 10.0,
    success: float = 0.9,
    episode_length: float = 100.0,
    runtime: float = 0.1,
) -> TuningRecord:
    """Build one finite record for decision and report unit tests."""
    metrics = TuningRunMetrics(
        candidate,
        stage,
        seed,
        4096,
        (runtime,) * iterations,
        (reward,) * iterations,
        (success,) * iterations,
        (episode_length,) * iterations,
    )
    return TuningRecord(
        metrics,
        f"{stage}-{candidate}-{seed}",
        Path(f"{stage}-{candidate}-{seed}.json"),
        "a" * 64,
        Path(f"events-{stage}-{candidate}-{seed}"),
        "b" * 64,
        config_hash(resolved_config),
        resolved_config,
        source_head="c" * 40,
        bundle_git_dirty=False,
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
        "schema_version": "1.2",
        "isaaclab_head": source_head,
        "isaaclab_newton": {
            "module_path": str(repo_root / "source/isaaclab_newton/isaaclab_newton/__init__.py"),
            "distribution_path": str(repo_root / "source/isaaclab_newton"),
            "direct_url": {
                "url": (repo_root / "source/isaaclab_newton").as_uri(),
                "dir_info": {"editable": True},
            },
        },
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


def _add_preflight_artifact(
    artifact_root: Path,
    monkeypatch,
    *,
    state: str,
    failure_category: str | None,
    keep_measured: bool,
) -> Path:
    """Transform or copy the synthetic measured artifact into an exact preflight."""
    import shutil

    measured_dir = next(path for path in artifact_root.iterdir() if path.name.startswith("wave1__"))
    manifest = json.loads((measured_dir / "manifest.json").read_text(encoding="utf-8"))
    measured_identity = TuningIdentity(**manifest["identity"])
    identity = dataclasses.replace(measured_identity, stage="preflight", max_iterations=5)
    preflight_dir = artifact_root / identity.run_id
    if keep_measured:
        shutil.copytree(measured_dir, preflight_dir)
    else:
        measured_dir.rename(preflight_dir)
    manifest_path = preflight_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    candidate = tuning.candidate(identity.candidate)
    resolved = resolve_config(tuning, candidate)
    command = tuple(
        build_tuning_command(matrix, tuning, candidate, identity, Path(__file__).resolve().parents[3], preflight_dir)
    )
    manifest.update(
        run_id=identity.run_id,
        identity=dataclasses.asdict(identity),
        artifact_root=str(preflight_dir),
        command=command,
        command_hash=command_hash(command),
        config_hash=config_hash(resolved),
        resolved_config=resolved,
        state=state,
        failure_category=failure_category,
    )
    bundle_path = preflight_dir / "benchmark_training_test.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["run"]["max_iterations"] = 5
    bundle["runtime"]["iterations_completed"] = 5
    bundle_path.write_text(json.dumps(bundle, sort_keys=True), encoding="utf-8")
    manifest["artifact_hashes"][bundle_path.name] = sha256_file(bundle_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    def parse(bundle_file: Path, _event_file: Path) -> TrainingTrace:
        parsed = json.loads(bundle_file.read_text(encoding="utf-8"))
        run = parsed["run"]
        stage = bundle_file.parent.name.split("__", maxsplit=1)[0]
        parsed_identity = TuningIdentity(stage, identity.candidate, run["seed"], run["num_envs"], run["max_iterations"])
        return _trace(parsed_identity)

    monkeypatch.setattr("benchmarks.kamino_dvi.analyze_tuning.parse_training_trace", parse)
    return preflight_dir


def test_loader_derives_failed_preflight_for_missing_wave1_measurement(tmp_path, monkeypatch):
    """An exact failed preflight is the immutable rejection for its withheld measured run."""
    artifact_root = _write_completed_artifact(tmp_path, monkeypatch)
    preflight_dir = _add_preflight_artifact(
        artifact_root,
        monkeypatch,
        state="failed",
        failure_category="numerical",
        keep_measured=False,
    )

    records = load_tuning_records(artifact_root, tmp_path / "logs", expected_stage="wave1")

    assert len(records) == 1
    record = records[0]
    assert record.run_id == preflight_dir.name
    assert record.metrics.stage == "wave1"
    assert record.metrics.seed == 42
    assert record.metrics.failure == "preflight:numerical"
    assert record.metrics.iteration_time_s == record.metrics.reward == ()
    assert record.derived_from_preflight is True
    assert record.preflight_identity == TuningIdentity("preflight", "integrator_euler", 42, 4096, 5)
    assert record.bundle_git_dirty is None


def test_completed_preflight_without_measurement_remains_missing(tmp_path, monkeypatch):
    """A successful preflight never substitutes for absent measured evidence."""
    artifact_root = _write_completed_artifact(tmp_path, monkeypatch)
    _add_preflight_artifact(
        artifact_root,
        monkeypatch,
        state="completed",
        failure_category=None,
        keep_measured=False,
    )

    records = load_tuning_records(artifact_root, tmp_path / "logs", expected_stage="wave1")

    with pytest.raises(ValueError, match="missing tuning record"):
        validate_tuning_records(records, (("integrator_euler", 42),))


def test_measured_record_supersedes_completed_preflight(tmp_path, monkeypatch):
    """Measured evidence remains authoritative when its exact preflight also exists."""
    artifact_root = _write_completed_artifact(tmp_path, monkeypatch)
    _add_preflight_artifact(
        artifact_root,
        monkeypatch,
        state="completed",
        failure_category=None,
        keep_measured=True,
    )

    records = load_tuning_records(artifact_root, tmp_path / "logs", expected_stage="wave1")

    assert len(records) == 1
    assert records[0].metrics.stage == "wave1"
    assert records[0].derived_from_preflight is False
    assert records[0].preflight_identity is None


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


def _set_bundle_git_dirty(artifact_root: Path, value) -> None:
    """Set and re-hash the synthetic completed bundle dirty flag."""
    run_dir = next(path for path in artifact_root.iterdir() if path.is_dir())
    bundle_path = next(run_dir.glob("benchmark_training_*.json"))
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["versions"]["git_dirty"] = value
    bundle_path.write_text(json.dumps(bundle, sort_keys=True), encoding="utf-8")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes"][bundle_path.name] = sha256_file(bundle_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def test_loader_rejects_tampered_package_location(tmp_path, monkeypatch):
    """Strict analysis rejects a manifest repointed to a stale package checkout."""
    artifact_root = _write_completed_artifact(tmp_path, monkeypatch)
    manifest_path = next(artifact_root.glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["isaaclab_newton"]["module_path"] = "/stale/isaaclab_newton/__init__.py"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="isaaclab_newton import"):
        load_tuning_records(artifact_root, tmp_path / "logs", expected_stage="wave1")


@pytest.mark.parametrize("value", (False, True))
def test_loader_preserves_boolean_bundle_git_dirty(tmp_path, monkeypatch, value):
    """Both broad boolean bundle flags load and remain available for audit."""
    artifact_root = _write_completed_artifact(tmp_path, monkeypatch)
    _set_bundle_git_dirty(artifact_root, value)

    records = load_tuning_records(artifact_root, tmp_path / "logs", expected_stage="wave1")

    assert len(records) == 1
    assert records[0].bundle_git_dirty is value


def test_loader_rejects_nonboolean_bundle_git_dirty(tmp_path, monkeypatch):
    """The broad bundle dirty flag remains a strictly typed JSON boolean."""
    artifact_root = _write_completed_artifact(tmp_path, monkeypatch)
    _set_bundle_git_dirty(artifact_root, "false")

    with pytest.raises(ValueError, match="git_dirty must be a boolean"):
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
    source = TuningRecord(
        metric,
        "run",
        Path("manifest.json"),
        "a" * 64,
        Path("event"),
        "b" * 64,
        "c" * 64,
        {},
        bundle_git_dirty=True,
    )
    derived, provenance = derive_stage2_baseline([source], 100)
    assert derived[0].stage == "halve"
    assert derived[0].reward == tuple(range(100))
    assert derived[0].final_mean(derived[0].reward, 20) == pytest.approx(89.5)
    assert provenance[0]["derivation"] == "first 100 aligned iterations of clean 300-iteration baseline"
    assert provenance[0]["source_manifest_hash"] == "a" * 64
    assert provenance[0]["source_bundle_git_dirty"] is True


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
    records[0] = dataclasses.replace(records[0], bundle_git_dirty=True)
    derived_identity = TuningIdentity("preflight", records[1].metrics.candidate, 42, 4096, 5)
    records[1] = dataclasses.replace(
        records[1],
        metrics=TuningRunMetrics(
            records[1].metrics.candidate,
            "wave1",
            42,
            4096,
            (),
            (),
            (),
            (),
            "preflight:numerical",
        ),
        run_id=derived_identity.run_id,
        manifest_path=Path(derived_identity.run_id) / "manifest.json",
        event_path=None,
        event_hash=None,
        source_head="d" * 40,
        bundle_git_dirty=None,
        derived_from_preflight=True,
        preflight_identity=derived_identity,
    )
    monkeypatch.setattr("benchmarks.kamino_dvi.analyze_tuning.load_tuning_records", lambda *_args, **_kwargs: records)
    output = tmp_path / "wave2.json"
    argv = ["resolve-wave2", "--artifact-root", str(tmp_path), "--logs-root", str(tmp_path), "--output", str(output)]
    assert main(argv) == 0
    decision = json.loads(output.read_text(encoding="utf-8"))
    assert decision["schema_version"] == "1.1"
    assert decision["action"] == "resolve-wave2"
    assert decision["source_stage"] == "wave1"
    assert decision["bundle_git_dirty"]["count"] == 1
    assert decision["bundle_git_dirty"]["run_ids"] == [records[0].run_id]
    assert "tracked-only cleanliness" in decision["bundle_git_dirty"]["advisory"]
    assert (
        next(item for item in decision["source_manifests"] if item["run_id"] == records[0].run_id)["bundle_git_dirty"]
        is True
    )
    disclosure = decision["derived_preflight_rejections"]
    assert disclosure["count"] == 1
    assert disclosure["records"][0]["run_id"] == derived_identity.run_id
    assert disclosure["records"][0]["failure"] == "preflight:numerical"
    assert disclosure["records"][0]["derived_from_preflight"] is True
    assert disclosure["records"][0]["preflight_identity"] == dataclasses.asdict(derived_identity)
    derived_source = next(item for item in decision["source_manifests"] if item["run_id"] == derived_identity.run_id)
    assert derived_source["config_hash"] == records[1].config_hash
    assert derived_source["source_head"] == "d" * 40
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


def test_validate_ignores_only_configured_decision_directory(tmp_path, monkeypatch):
    """The configured decision directory is metadata, while any other unknown child remains invalid."""
    artifact_root = tmp_path / "artifacts"
    decision_root = artifact_root / "decisions"
    decision_root.mkdir(parents=True)
    monkeypatch.setattr(analysis_module, "_validation_stage", lambda *_args: ((), None))
    argv = [
        "validate",
        "--stages",
        "wave2",
        "--artifact-root",
        str(artifact_root),
        "--decision-root",
        str(decision_root),
    ]

    assert main(argv) == 0

    (artifact_root / "mystery").mkdir()
    with pytest.raises(ValueError, match=r"undeclared tuning directory: .*mystery"):
        main(argv)


def test_promote_stage2_derives_decision_root_from_output(tmp_path, monkeypatch):
    """The documented staged command treats the output parent as its decision directory."""
    artifact_root = tmp_path / "artifacts"
    decision_root = artifact_root / "decisions"
    decision_root.mkdir(parents=True)
    output = decision_root / "stage2.json"

    def recompute(args, through):
        assert through == "stage2"
        assert (
            load_tuning_records(
                args.artifact_root,
                args.logs_root,
                expected_stage="wave1",
                decision_root=args.decision_root,
            )
            == []
        )
        return {"stage2_decision": {"action": "promote-stage2"}}

    monkeypatch.setattr(analysis_module, "_recompute_chain", recompute)
    argv = [
        "promote-stage2",
        "--artifact-root",
        str(artifact_root),
        "--output",
        str(output),
    ]

    assert main(argv) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == {"action": "promote-stage2"}

    (artifact_root / "mystery").mkdir()
    with pytest.raises(ValueError, match=r"undeclared tuning directory: .*mystery"):
        main(argv)


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
                "schema_version": "1.1",
                "action": "promote-stage2",
                "source_stage": "stage1",
                "source_artifact_root": str(tmp_path),
                "source_manifests": [],
                "bundle_git_dirty": {
                    "count": 0,
                    "run_ids": [],
                    "advisory": analysis_module.BUNDLE_GIT_DIRTY_ADVISORY,
                },
                "derived_preflight_rejections": {"count": 0, "records": []},
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


@pytest.mark.parametrize(
    "tampering",
    (
        "missing_manifest_flag",
        "missing_disclosure",
        "nonboolean_manifest_flag",
        "unsorted_run_ids",
        "duplicate_run_ids",
        "incorrect_count",
        "altered_advisory",
        "manifest_mismatch",
    ),
)
def test_strict_decision_parser_rejects_tampered_bundle_dirty_disclosure(tmp_path, tampering):
    """Persisted bundle dirty provenance must be complete and internally exact."""
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    candidate = tuning.wave1[0]
    resolved = resolve_config(tuning, candidate)
    data = {
        "schema_version": "1.1",
        "action": "promote-stage2",
        "source_stage": "stage1",
        "source_artifact_root": str(tmp_path),
        "source_manifests": [
            {"run_id": "run-a", "path": "a.json", "sha256": "a" * 64, "bundle_git_dirty": True},
            {"run_id": "run-b", "path": "b.json", "sha256": "b" * 64, "bundle_git_dirty": True},
            {"run_id": "run-c", "path": "c.json", "sha256": "c" * 64, "bundle_git_dirty": False},
            {"run_id": "run-d", "path": "d.json", "sha256": "d" * 64, "bundle_git_dirty": None},
        ],
        "bundle_git_dirty": {
            "count": 2,
            "run_ids": ["run-a", "run-b"],
            "advisory": analysis_module.BUNDLE_GIT_DIRTY_ADVISORY,
        },
        "timestamp_utc": "2026-07-21T00:00:00+00:00",
        "revisions": dataclasses.asdict(load_matrix(DEFAULT_MATRIX_PATH).revisions),
        "selected": [candidate.name],
        "rejected": {},
        "resolved_candidates": [
            {
                "name": candidate.name,
                "overrides": candidate.overrides,
                "resolved_config": resolved,
                "config_hash": config_hash(resolved),
            }
        ],
    }
    if tampering == "missing_manifest_flag":
        del data["source_manifests"][0]["bundle_git_dirty"]
    elif tampering == "missing_disclosure":
        del data["bundle_git_dirty"]
    elif tampering == "nonboolean_manifest_flag":
        data["source_manifests"][0]["bundle_git_dirty"] = 1
    elif tampering == "unsorted_run_ids":
        data["bundle_git_dirty"]["run_ids"] = ["run-b", "run-a"]
    elif tampering == "duplicate_run_ids":
        data["bundle_git_dirty"]["run_ids"] = ["run-a", "run-a"]
    elif tampering == "incorrect_count":
        data["bundle_git_dirty"]["count"] = 1
    elif tampering == "altered_advisory":
        data["bundle_git_dirty"]["advisory"] = "wrong"
    elif tampering == "manifest_mismatch":
        data["bundle_git_dirty"]["run_ids"] = ["run-a", "run-c"]
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="decision (source manifests|bundle dirty disclosure) (are|is) invalid"):
        load_decision(path, "promote-stage2", tuning, minimum_count=1, maximum_count=8)


@pytest.mark.parametrize(
    "tampering",
    (
        "missing_derived_flag",
        "nonboolean_derived_flag",
        "missing_preflight_identity",
        "mismatched_preflight_identity",
        "measured_with_preflight_identity",
        "malformed_package_location",
        "disclosure_mismatch",
    ),
)
def test_strict_decision_parser_rejects_tampered_preflight_provenance(tmp_path, tampering):
    """Persisted derived rejection provenance must exactly match its source manifest."""
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    candidate = tuning.wave1[0]
    resolved = resolve_config(tuning, candidate)
    identity = TuningIdentity("preflight", candidate.name, 42, tuning.num_envs, tuning.preflight_iterations)
    source = {
        "run_id": identity.run_id,
        "path": str(tmp_path / identity.run_id / "manifest.json"),
        "sha256": "a" * 64,
        "config_hash": config_hash(resolved),
        "source_head": "b" * 40,
        "event_path": None,
        "event_hash": None,
        "bundle_git_dirty": None,
        "isaaclab_newton": {
            "module_path": str(tmp_path / "source/isaaclab_newton/isaaclab_newton/__init__.py"),
            "distribution_path": str(tmp_path / "source/isaaclab_newton"),
            "direct_url": {},
        },
        "derived_from_preflight": True,
        "preflight_identity": dataclasses.asdict(identity),
        "failure": "preflight:numerical",
    }
    data = {
        "schema_version": "1.1",
        "action": "promote-stage2",
        "source_stage": "stage1",
        "source_artifact_root": str(tmp_path),
        "source_manifests": [source],
        "bundle_git_dirty": {
            "count": 0,
            "run_ids": [],
            "advisory": analysis_module.BUNDLE_GIT_DIRTY_ADVISORY,
        },
        "derived_preflight_rejections": {"count": 1, "records": [dict(source)]},
        "timestamp_utc": "2026-07-21T00:00:00+00:00",
        "revisions": dataclasses.asdict(load_matrix(DEFAULT_MATRIX_PATH).revisions),
        "selected": [candidate.name],
        "rejected": {candidate.name: "preflight:numerical"},
        "resolved_candidates": [
            {
                "name": candidate.name,
                "overrides": candidate.overrides,
                "resolved_config": resolved,
                "config_hash": config_hash(resolved),
            }
        ],
    }
    if tampering == "missing_derived_flag":
        del source["derived_from_preflight"]
    elif tampering == "nonboolean_derived_flag":
        source["derived_from_preflight"] = 1
    elif tampering == "missing_preflight_identity":
        source["preflight_identity"] = None
    elif tampering == "mismatched_preflight_identity":
        source["preflight_identity"]["candidate"] = "wrong"
    elif tampering == "measured_with_preflight_identity":
        source["derived_from_preflight"] = False
    elif tampering == "malformed_package_location":
        source["isaaclab_newton"]["module_path"] = 1
    elif tampering == "disclosure_mismatch":
        data["derived_preflight_rejections"]["count"] = 0
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="decision preflight provenance is invalid"):
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


def test_loader_uses_latest_failed_preflight_retry(tmp_path, monkeypatch):
    """The highest contiguous failed preflight attempt supplies the rejection evidence."""
    artifact_root = _write_completed_artifact(tmp_path, monkeypatch)
    attempt0_dir = _add_preflight_artifact(
        artifact_root,
        monkeypatch,
        state="failed",
        failure_category="numerical",
        keep_measured=False,
    )
    attempt1_dir = _copy_retry_attempt(artifact_root, 1, attempt0_dir.name)
    manifest_path = attempt1_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["state"] = "failed"
    manifest["failure_category"] = "crash"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    records = load_tuning_records(artifact_root, tmp_path / "logs", expected_stage="wave1")

    assert len(records) == 1
    assert records[0].run_id == attempt1_dir.name
    assert records[0].metrics.failure == "preflight:crash"
    assert records[0].preflight_identity.attempt == 1


def test_loader_rejects_preflight_config_mismatch(tmp_path, monkeypatch):
    """A stale preflight configuration cannot reject the current candidate."""
    artifact_root = _write_completed_artifact(tmp_path, monkeypatch)
    preflight_dir = _add_preflight_artifact(
        artifact_root,
        monkeypatch,
        state="failed",
        failure_category="numerical",
        keep_measured=False,
    )
    manifest_path = preflight_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["resolved_config"]["dvi_block_iterations"] = 999
    manifest["config_hash"] = config_hash(manifest["resolved_config"])
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="preflight config"):
        load_tuning_records(artifact_root, tmp_path / "logs", expected_stage="wave1")


def test_loader_rejects_nonterminal_preflight(tmp_path, monkeypatch):
    """A planned preflight is neither success nor immutable rejection evidence."""
    artifact_root = _write_completed_artifact(tmp_path, monkeypatch)
    preflight_dir = _add_preflight_artifact(
        artifact_root,
        monkeypatch,
        state="failed",
        failure_category="numerical",
        keep_measured=False,
    )
    manifest_path = preflight_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["state"] = "planned"
    manifest["failure_category"] = None
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest is incomplete"):
        load_tuning_records(artifact_root, tmp_path / "logs", expected_stage="wave1")


def test_loader_rejects_failed_manifest_without_failure_category(tmp_path, monkeypatch):
    """A failed lifecycle state must retain its real failure category."""
    artifact_root = _write_completed_artifact(tmp_path, monkeypatch)
    _add_preflight_artifact(
        artifact_root,
        monkeypatch,
        state="failed",
        failure_category=None,
        keep_measured=False,
    )

    with pytest.raises(ValueError, match="failed manifest requires a failure category"):
        load_tuning_records(artifact_root, tmp_path / "logs", expected_stage="wave1")


@pytest.mark.parametrize("state", ("planned", "running", "completed", "invalidated"))
def test_loader_rejects_nonfailed_manifest_with_failure_category(tmp_path, monkeypatch, state):
    """Only the failed lifecycle state may carry a failure category."""
    artifact_root = _write_completed_artifact(tmp_path, monkeypatch)
    _add_preflight_artifact(
        artifact_root,
        monkeypatch,
        state=state,
        failure_category="crash",
        keep_measured=False,
    )

    with pytest.raises(ValueError, match="failure category is only valid for a failed manifest"):
        load_tuning_records(artifact_root, tmp_path / "logs", expected_stage="wave1")


def test_loader_rejects_preflight_without_hashed_standard_logs(tmp_path, monkeypatch):
    """A failed preflight must retain hashes for both standard output streams."""
    artifact_root = _write_completed_artifact(tmp_path, monkeypatch)
    preflight_dir = _add_preflight_artifact(
        artifact_root,
        monkeypatch,
        state="failed",
        failure_category="numerical",
        keep_measured=False,
    )
    manifest_path = preflight_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["artifact_hashes"]["stdout.log"]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="hashed stdout.log and stderr.log"):
        load_tuning_records(artifact_root, tmp_path / "logs", expected_stage="wave1")


def test_loader_rejects_preflight_source_head_mismatch_with_measurement(tmp_path, monkeypatch):
    """Measured evidence cannot silently hide a stale preflight source revision."""
    artifact_root = _write_completed_artifact(tmp_path, monkeypatch)
    preflight_dir = _add_preflight_artifact(
        artifact_root,
        monkeypatch,
        state="failed",
        failure_category="numerical",
        keep_measured=True,
    )
    manifest_path = preflight_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["isaaclab_head"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="one exact source HEAD"):
        load_tuning_records(artifact_root, tmp_path / "logs", expected_stage="wave1")


def test_loader_rejects_noncontiguous_preflight_retry(tmp_path, monkeypatch):
    """Preflight retry attempts must advance contiguously."""
    artifact_root = _write_completed_artifact(tmp_path, monkeypatch)
    attempt0_dir = _add_preflight_artifact(
        artifact_root,
        monkeypatch,
        state="failed",
        failure_category="numerical",
        keep_measured=False,
    )
    _copy_retry_attempt(artifact_root, 2, attempt0_dir.name)

    with pytest.raises(ValueError, match="contiguous"):
        load_tuning_records(artifact_root, tmp_path / "logs", expected_stage="wave1")


def test_loader_rejects_preflight_retry_with_wrong_parent(tmp_path, monkeypatch):
    """Preflight retry lineage must name the immediately preceding attempt."""
    artifact_root = _write_completed_artifact(tmp_path, monkeypatch)
    _add_preflight_artifact(
        artifact_root,
        monkeypatch,
        state="failed",
        failure_category="numerical",
        keep_measured=False,
    )
    _copy_retry_attempt(artifact_root, 1, "wrong-parent")

    with pytest.raises(ValueError, match="retry parent"):
        load_tuning_records(artifact_root, tmp_path / "logs", expected_stage="wave1")


def test_loader_derives_failed_wave2_preflight_for_seed42(tmp_path, monkeypatch):
    """Adaptive Wave 2 uses the decision-resolved config for seed-42 preflight rejection."""
    artifact_root = _write_completed_artifact(tmp_path, monkeypatch)
    _add_preflight_artifact(
        artifact_root,
        monkeypatch,
        state="failed",
        failure_category="crash",
        keep_measured=False,
    )
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    resolved = resolve_config(tuning, tuning.candidate("integrator_euler"))

    records = load_tuning_records(
        artifact_root,
        tmp_path / "logs",
        expected_stage="wave2",
        expected_candidate_configs={"integrator_euler": resolved},
    )

    assert [(record.metrics.stage, record.metrics.seed, record.metrics.failure) for record in records] == [
        ("wave2", 42, "preflight:crash")
    ]


def test_loader_never_derives_later_stage_seeds_from_preflight(tmp_path, monkeypatch):
    """One seed-42 preflight failure cannot fabricate multi-seed final evidence."""
    artifact_root = _write_completed_artifact(tmp_path, monkeypatch)
    _add_preflight_artifact(
        artifact_root,
        monkeypatch,
        state="failed",
        failure_category="crash",
        keep_measured=False,
    )

    records = load_tuning_records(artifact_root, tmp_path / "logs", expected_stage="final")

    assert records == []
    with pytest.raises(ValueError, match="missing tuning record"):
        validate_tuning_records(records, (("integrator_euler", seed) for seed in (42, 43, 44)))


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


def test_loader_rejects_noncontiguous_retry_attempt(tmp_path, monkeypatch):
    """Attempt numbers must advance contiguously even when the parent is correct."""
    artifact_root = _write_completed_artifact(tmp_path, monkeypatch)
    attempt0_dir = next(path for path in artifact_root.iterdir() if path.is_dir())
    _copy_retry_attempt(artifact_root, 2, attempt0_dir.name)

    with pytest.raises(ValueError, match="contiguous"):
        load_tuning_records(artifact_root, tmp_path / "logs", expected_stage="wave1")


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
        "wave2": [
            record("w2_good", "wave2"),
            record("w2_slow", "wave2"),
            record("w2_bad", "wave2", "timeout"),
        ],
        "halve": [record("w1_good", "halve"), record("w2_good", "halve")],
        "final": [record("winner", "final"), record("qualified_loser", "final"), record("rejected", "final")],
        "wave2_decision": {"selected": [f"derived_{index}" for index in range(6)]},
        "stage2_decision": {
            "selected": ["w1_good", "w2_good"],
            "rejected": {"w2_slow": "slower", "w2_bad": "timeout"},
        },
        "finalists_decision": {"selected": ["w2_good"], "rejected": {"w1_good": "slower"}},
    }
    winner = {
        "candidate": "winner",
        "rejected": {"qualified_loser": "tie-break", "rejected": "learning threshold"},
        "qualifications": {
            "winner": {"qualified": True},
            "qualified_loser": {"qualified": True},
            "rejected": {"qualified": False},
        },
    }

    rows = {
        row["stage"]: row
        for row in analysis_module._stage_funnel(chain, [record("canonical_winner", "canonical")], winner)
    }

    assert rows["Wave 1"] == {
        "stage": "Wave 1",
        "attempted_runs": 2,
        "valid_runs": 1,
        "terminal_rejected_runs": 1,
        "learning_rejected_candidates": 0,
        "promoted_candidates": 6,
    }
    assert rows["Wave 2"]["terminal_rejected_runs"] == 1
    assert rows["Wave 2"]["learning_rejected_candidates"] == 1
    assert rows["Wave 2"]["promoted_candidates"] == 1
    assert rows["final"]["learning_rejected_candidates"] == 1
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
    assert summary["terminal_status"] == "completed"
    assert summary["canonical_comparison"]["qualified"] is True
    assert summary["bundle_git_dirty"]["count"] == 0
    assert summary["bundle_git_dirty"]["run_ids"] == []
    assert "includes untracked paths" in summary["bundle_git_dirty"]["advisory"]
    assert all(record["bundle_git_dirty"] is False for record in summary["canonical_provenance"]["records"])


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


def _validation_record(name: str, stage: str, seed: int, *, failure: str | None = None) -> TuningRecord:
    """Build one terminal record for validate-action summary tests."""
    values = () if failure else (1.0,)
    metrics = TuningRunMetrics(name, stage, seed, 4096, values, values, values, values, failure)
    return TuningRecord(
        metrics,
        f"{stage}__{name}__seed{seed}",
        Path(f"{stage}-{name}-{seed}.json"),
        "a" * 64,
        None,
        None,
        "b" * 64,
        {},
    )


def _add_baseline_artifact(source_dir: Path, identity: TuningIdentity) -> Path:
    """Copy a fixture into one provenance-consistent baseline artifact."""
    import shutil

    artifact_root = source_dir.parent
    run_dir = artifact_root / identity.run_id
    shutil.copytree(source_dir, run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    resolved = dict(tuning.baseline)
    candidate = ResolvedTuningCandidate("baseline", {}, resolved, config_hash(resolved))
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


def test_validate_action_accepts_baseline_only_completed_artifacts(tmp_path, monkeypatch, capsys):
    """The exact Task 5 baseline-only command succeeds through the strict loader."""
    artifact_root = _write_completed_artifact(tmp_path, monkeypatch)
    source_dir = next(path for path in artifact_root.iterdir() if path.is_dir())
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    for seed in tuning.seeds:
        identity = TuningIdentity("baseline", "baseline", seed, tuning.num_envs, tuning.final_iterations)
        _add_baseline_artifact(source_dir, identity)

    def parse(bundle_path: Path, _event_path: Path) -> TrainingTrace:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        run = bundle["run"]
        identity = TuningIdentity("baseline", "baseline", run["seed"], run["num_envs"], run["max_iterations"])
        return _trace(identity)

    monkeypatch.setattr(analysis_module, "parse_training_trace", parse)

    assert (
        main(
            [
                "validate",
                "--stages",
                "baseline",
                "--artifact-root",
                str(artifact_root),
                "--logs-root",
                str(tmp_path / "logs"),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["stages"] == [
        {
            "stage": "baseline",
            "expected": 3,
            "terminal": 3,
            "valid": 3,
            "rejected": 0,
            "run_ids": sorted(path.name for path in artifact_root.glob("baseline__*")),
            "rejection_reasons": [],
            "bundle_git_dirty": {
                "count": 0,
                "run_ids": [],
                "advisory": analysis_module.BUNDLE_GIT_DIRTY_ADVISORY,
            },
            "derived_preflight_rejections": {"count": 0, "records": []},
        }
    ]


def test_validate_action_emits_deterministic_stage_order_and_counts(tmp_path, monkeypatch, capsys):
    """Validate emits standard JSON with requested order, terminal counts, and sorted details."""
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    baseline = [_validation_record("baseline", "baseline", seed) for seed in reversed(tuning.seeds)]
    baseline[0] = dataclasses.replace(baseline[0], bundle_git_dirty=True)
    wave1 = [
        _validation_record(candidate.name, "wave1", 42, failure="numerical" if index == 0 else None)
        for index, candidate in enumerate(reversed(tuning.wave1))
    ]
    preflight_identity = TuningIdentity("preflight", wave1[0].metrics.candidate, 42, 4096, 5)
    wave1[0] = dataclasses.replace(
        wave1[0],
        metrics=dataclasses.replace(wave1[0].metrics, failure="preflight:numerical"),
        run_id=preflight_identity.run_id,
        manifest_path=Path(preflight_identity.run_id) / "manifest.json",
        source_head="d" * 40,
        derived_from_preflight=True,
        preflight_identity=preflight_identity,
    )
    by_stage = {"baseline": baseline, "wave1": wave1}
    monkeypatch.setattr(
        analysis_module,
        "load_tuning_records",
        lambda _artifacts, _logs, expected_stage, **_kwargs: by_stage[expected_stage],
    )

    assert main(["validate", "--stages", "baseline", "wave1", "--artifact-root", str(tmp_path)]) == 0

    output = json.loads(capsys.readouterr().out, parse_constant=lambda value: pytest.fail(value))
    assert output["schema_version"] == "1.0"
    assert [item["stage"] for item in output["stages"]] == ["baseline", "wave1"]
    assert output["stages"][0] == {
        "stage": "baseline",
        "expected": 3,
        "terminal": 3,
        "valid": 3,
        "rejected": 0,
        "run_ids": sorted(record.run_id for record in baseline),
        "rejection_reasons": [],
        "bundle_git_dirty": {
            "count": 1,
            "run_ids": [baseline[0].run_id],
            "advisory": analysis_module.BUNDLE_GIT_DIRTY_ADVISORY,
        },
        "derived_preflight_rejections": {"count": 0, "records": []},
    }
    assert output["stages"][1]["expected"] == 18
    assert output["stages"][1]["terminal"] == 18
    assert output["stages"][1]["valid"] == 17
    assert output["stages"][1]["rejected"] == 1
    assert output["stages"][1]["run_ids"] == sorted(record.run_id for record in wave1)
    assert output["stages"][1]["rejection_reasons"] == [{"run_id": wave1[0].run_id, "reason": "preflight:numerical"}]
    disclosure = output["stages"][1]["derived_preflight_rejections"]
    assert disclosure["count"] == 1
    assert disclosure["records"][0]["run_id"] == preflight_identity.run_id
    assert disclosure["records"][0]["sha256"] == wave1[0].manifest_hash


def test_validate_action_passes_wave2_configs_to_preflight_loader(tmp_path, monkeypatch, capsys):
    """Adaptive validation supplies exact decision configs for failed preflight reconciliation."""
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    candidate = tuning.wave1[0]
    resolved = resolve_config(tuning, candidate)
    decision = {
        "resolved_candidates": [
            {"name": candidate.name, "resolved_config": resolved, "config_hash": config_hash(resolved)}
        ]
    }
    record = dataclasses.replace(
        _validation_record(candidate.name, "wave2", 42, failure="preflight:numerical"),
        config_hash=config_hash(resolved),
        resolved_config=resolved,
        derived_from_preflight=True,
        preflight_identity=TuningIdentity("preflight", candidate.name, 42, 4096, 5),
    )
    captured = {}

    def load(_artifacts, _logs, expected_stage, **kwargs):
        captured.update(kwargs)
        assert expected_stage == "wave2"
        return [record]

    monkeypatch.setattr(analysis_module, "_validation_stage", lambda *_args: (((candidate.name, 42),), decision))
    monkeypatch.setattr(analysis_module, "load_tuning_records", load)

    assert main(["validate", "--stages", "wave2", "--artifact-root", str(tmp_path)]) == 0
    capsys.readouterr()
    assert captured["expected_candidate_configs"] == {candidate.name: resolved}


@pytest.mark.parametrize(
    ("stage", "candidate_names", "seeds"),
    (
        ("wave2", ("w2_a", "w2_b"), (42,)),
        ("halve", ("halve_a", "halve_b"), (42, 43)),
        ("final", ("final_a",), (42, 43, 44)),
    ),
)
def test_validate_action_resolves_adaptive_identities_from_strict_decision(
    tmp_path, monkeypatch, stage, candidate_names, seeds
):
    """Adaptive validation derives exact identities from its strict upstream decision."""
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    decision = {
        "resolved_candidates": [
            {"name": name, "resolved_config": dict(tuning.baseline), "config_hash": config_hash(tuning.baseline)}
            for name in candidate_names
        ]
    }
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return decision

    monkeypatch.setattr(analysis_module, "load_decision", load)
    args = build_parser().parse_args(
        ["validate", "--stages", stage, "--artifact-root", str(tmp_path), "--decision-root", str(tmp_path)]
    )

    expected, resolved_decision = analysis_module._validation_stage(args, tuning, stage)

    assert expected == tuple((name, seed) for name in candidate_names for seed in seeds)
    assert resolved_decision is decision
    assert calls[0][0][0].parent == tmp_path
    assert calls[0][1]["matrix_path"] == DEFAULT_MATRIX_PATH


def test_validate_action_fails_when_requested_wave1_is_missing(tmp_path, monkeypatch, capsys):
    """Baseline plus missing Wave 1 fails exact coverage without requiring decisions."""
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    baseline = [_validation_record("baseline", "baseline", seed) for seed in tuning.seeds]
    monkeypatch.setattr(
        analysis_module,
        "load_tuning_records",
        lambda _artifacts, _logs, expected_stage, **_kwargs: baseline if expected_stage == "baseline" else [],
    )

    with pytest.raises(ValueError, match="missing tuning record"):
        main(["validate", "--stages", "baseline", "wave1", "--artifact-root", str(tmp_path)])
    assert capsys.readouterr().out == ""


def test_promote_finalists_persists_terminal_zero_survivor_decision(tmp_path):
    """All Stage-2 guardrail rejections remain auditable in finalists.json."""
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    baseline = [
        _synthetic_record("baseline", "baseline", seed, tuning.final_iterations, dict(tuning.baseline))
        for seed in tuning.seeds
    ]
    candidates = list(tuning.wave1[:8])
    stage2 = {
        "resolved_candidates": [
            {
                "name": candidate.name,
                "overrides": candidate.overrides,
                "resolved_config": resolve_config(tuning, candidate),
                "config_hash": config_hash(resolve_config(tuning, candidate)),
            }
            for candidate in candidates
        ]
    }
    halve = [
        _synthetic_record(
            candidate.name,
            "halve",
            seed,
            tuning.halve_iterations,
            resolve_config(tuning, candidate),
            reward=7.0,
        )
        for candidate in candidates
        for seed in (42, 43)
    ]

    decision = _compute_finalists(tuning, tmp_path, baseline, halve, stage2)

    assert decision["selected"] == []
    assert decision["resolved_candidates"] == []
    assert decision["terminal_status"] == "stopped_no_safe_finalist"
    assert decision["stop_reason"] == "No Stage-2 candidate satisfied every per-seed learning guardrail."
    assert decision["rejected"] == {candidate.name: "seed 42: reward below 80% of baseline" for candidate in candidates}
    assert len(decision["baseline_prefix_provenance"]) == 3


def test_final_runner_rejects_terminal_zero_survivor_decision(tmp_path):
    """A scientific early stop cannot schedule a zero-candidate final stage."""
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    (tmp_path / "finalists.json").write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "action": "promote-finalists",
                "terminal_status": "stopped_no_safe_finalist",
                "stop_reason": "No Stage-2 candidate satisfied every per-seed learning guardrail.",
                "selected": [],
                "resolved_candidates": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="between 1 and 3 candidates"):
        stage_candidates(tuning, "final", tmp_path)


def test_select_winner_refuses_terminal_zero_survivor_without_writing(tmp_path, monkeypatch):
    """Winner selection explains the early stop and leaves winner.json absent."""
    output = tmp_path / "winner.json"
    chain = {
        "finalists_decision": {
            "selected": [],
            "resolved_candidates": [],
            "terminal_status": "stopped_no_safe_finalist",
        }
    }
    monkeypatch.setattr(analysis_module, "_recompute_chain", lambda *_args: chain)

    with pytest.raises(ValueError, match="no safe finalists survived Stage 2"):
        main(["select-winner", "--artifact-root", str(tmp_path / "artifacts"), "--output", str(output)])

    assert not output.exists()


def test_report_writes_exact_early_stop_outputs_without_final_or_canonical(tmp_path, monkeypatch):
    """Zero-survivor evidence produces the complete five-file terminal report."""
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    baseline = [
        _synthetic_record("baseline", "baseline", seed, tuning.final_iterations, dict(tuning.baseline))
        for seed in tuning.seeds
    ]
    candidate = tuning.wave1[0]
    resolved = resolve_config(tuning, candidate)
    wave1 = [_synthetic_record(candidate.name, "wave1", 42, tuning.screen_iterations, resolved)]
    wave2 = [_synthetic_record(candidate.name, "wave2", 42, tuning.screen_iterations, resolved)]
    halve = [
        _synthetic_record(candidate.name, "halve", seed, tuning.halve_iterations, resolved, reward=7.0)
        for seed in (42, 43)
    ]
    finalists = {
        "selected": [],
        "resolved_candidates": [],
        "rejected": {candidate.name: "seed 42: reward below 80% of baseline"},
        "terminal_status": "stopped_no_safe_finalist",
        "stop_reason": "No Stage-2 candidate satisfied every per-seed learning guardrail.",
    }
    chain = {
        "matrix": tuning,
        "baseline": baseline,
        "wave1": wave1,
        "wave2": wave2,
        "halve": halve,
        "wave2_decision": {"selected": [], "rejected": {}},
        "stage2_decision": {"selected": [candidate.name], "rejected": {}},
        "finalists_decision": finalists,
    }
    monkeypatch.setattr(analysis_module, "_recompute_chain", lambda *_args: chain)
    monkeypatch.setattr(analysis_module, "_require_persisted", lambda *_args: finalists)
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps(
            [
                {"task": tuning.task, "variant": "mjwarp", "iteration_time_s": {"mean": 0.2}},
                {"task": tuning.task, "variant": "physx", "iteration_time_s": {"mean": 0.3}},
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report"

    assert (
        main(
            [
                "report",
                "--artifact-root",
                str(tmp_path / "artifacts"),
                "--decision-root",
                str(tmp_path),
                "--comparison-summary",
                str(comparison),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    assert {path.name for path in output.iterdir()} == {
        "summary.json",
        "runtime.png",
        "learning.png",
        "anymal_d_dvi_tuning.md",
        "anymal_d_dvi_tuning.pdf",
    }
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["terminal_status"] == "stopped_no_safe_finalist"
    assert summary["winner"] is None
    assert summary["winner_config"] is None
    assert summary["coverage"]["final"] == []
    assert summary["coverage"]["canonical"] == []
    assert summary["speedups"] == {}
    stage2 = summary["final_rows"][0]
    for metric in ("runtime", "reward", "success", "episode_length"):
        assert stage2[metric]["n"] == 2
        assert stage2[metric]["half_width"] == 0.0
    screening = [row for row in summary["runtime_rows"] if row["stage"] in {"wave1", "wave2"}]
    assert len(screening) == 2
    assert all(row["half_width"] is None for row in screening)
    assert all(row["interval_status"] == "single_seed_no_ci" for row in screening)
    assert summary["rejections"] == [f"{candidate.name}: seed 42: reward below 80% of baseline"]
    assert {row["variant"] for row in summary["legacy_comparison"]} == {"mjwarp", "physx"}

    markdown = (output / "anymal_d_dvi_tuning.md").read_text(encoding="utf-8")
    for text in (
        "stopped with no safe finalist",
        "Stage-2 metrics use two-sided 95% Student-t confidence intervals with n=2",
        "preset was not modified",
        "seed 42: reward below 80% of baseline",
        "mjwarp",
        "physx",
        "No apples-to-apples winner speedup",
    ):
        assert text in markdown
    assert "clean DVI:" not in markdown
    pdf_source_text = "\n".join(_paginate_text(markdown))
    assert "stopped with no safe finalist" in pdf_source_text
    assert "No apples-to-apples winner speedup" in pdf_source_text
