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

from benchmarks.kamino_dvi.analyze_tuning import (
    TuningRecord,
    _qualification_dict,
    derive_stage2_baseline,
    load_tuning_records,
    main,
    validate_tuning_records,
)
from benchmarks.kamino_dvi.manifests import command_hash, sha256_file
from benchmarks.kamino_dvi.matrix import DEFAULT_MATRIX_PATH, load_matrix
from benchmarks.kamino_dvi.parsing import TrainingTrace
from benchmarks.kamino_dvi.statistics import Estimate
from benchmarks.kamino_dvi.tune import TuningIdentity, build_tuning_command
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
