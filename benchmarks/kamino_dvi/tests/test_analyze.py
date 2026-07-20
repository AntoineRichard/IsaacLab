# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for report-level benchmark quality findings."""

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.kamino_dvi.analysis import VariantSummary, load_records
from benchmarks.kamino_dvi.analyze import main, quality_issues
from benchmarks.kamino_dvi.manifests import command_hash
from benchmarks.kamino_dvi.matrix import DEFAULT_MATRIX_PATH, load_matrix
from benchmarks.kamino_dvi.statistics import Estimate


def test_quality_issues_quantifies_schema_mismatches_per_task(tmp_path):
    """Schema findings separate task counts and distinguish value mismatches from missing data."""
    estimate = Estimate(1.0, 0.1, 3)
    summaries = [VariantSummary("Ant", "kamino_current", 4096, estimate, estimate, estimate, estimate, estimate)]
    records = [
        SimpleNamespace(
            task="Ant", success_schema_mismatch=True, success_schema_mismatch_points=2, success_rate=(1,) * 3
        ),
        SimpleNamespace(
            task="Ant", success_schema_mismatch=False, success_schema_mismatch_points=0, success_rate=(1,) * 3
        ),
        SimpleNamespace(
            task="Cartpole", success_schema_mismatch=True, success_schema_mismatch_points=1, success_rate=(1,) * 3
        ),
        SimpleNamespace(
            task="ANYmal-D", success_schema_mismatch=False, success_schema_mismatch_points=0, success_rate=(1,) * 3
        ),
    ]

    issues = quality_issues(records, summaries, tmp_path)

    assert any("Ant" in issue and "1/2 runs" in issue and "2/6 points" in issue for issue in issues)
    assert any("Cartpole" in issue and "1/1 runs" in issue and "1/3 points" in issue for issue in issues)
    assert any("ANYmal-D" in issue and "0/1 runs" in issue and "0/3 points" in issue for issue in issues)
    assert any("every required reward, episode-length, and success field exists" in issue for issue in issues)
    assert any("value mismatch, not missing data" in issue for issue in issues)


def test_quality_issues_discloses_legacy_source_and_event_provenance(tmp_path):
    """Campaign warnings distinguish bundle evidence from current clean-run enforcement."""
    for index, (commit, dirty) in enumerate((("a" * 40, True), ("b" * 40, False))):
        run_dir = tmp_path / f"full__legacy_{index}"
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "state": "completed",
                    "artifact_hashes": {"benchmark_training_test.json": "hash"},
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "benchmark_training_test.json").write_text(
            json.dumps({"versions": {"git_commit": commit, "git_dirty": dirty}}), encoding="utf-8"
        )

    issues = quality_issues([], [], tmp_path)

    assert any("2 distinct commits" in issue and "1/2" in issue and "git_dirty=true" in issue for issue in issues)
    assert any("did not pass the current clean-source check" in issue for issue in issues)
    assert any(
        "TensorBoard event files" in issue and "2/2" in issue and "not retained or hashed" in issue for issue in issues
    )


def test_quality_issues_reports_seed_sensitive_learning_for_any_variant(tmp_path):
    """A large success interval is a weak-learning warning independent of solver variant."""
    estimate = Estimate(1.0, 0.1, 3)
    summaries = [
        VariantSummary("ANYmal-D", "kamino_current", 4096, estimate, estimate, estimate, estimate, estimate),
        VariantSummary("ANYmal-D", "mjwarp", 4096, estimate, estimate, estimate, estimate, Estimate(0.75, 1.07, 3)),
    ]

    issues = quality_issues([], summaries, tmp_path)

    assert any("ANYmal-D MJWarp" in issue and "seed-sensitive weak learning" in issue for issue in issues)
    assert any("not a runtime or stability failure" in issue for issue in issues)


def _write_run_artifacts(tmp_path: Path, *, manifest_changes: dict | None = None) -> Path:
    """Write the minimal completed run artifacts needed for identity validation."""
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    run_dir = tmp_path / "full__Isaac-Cartpole-Direct__kamino_current__seed42__env4096__iter300"
    run_dir.mkdir()
    bundle = run_dir / "benchmark_training_test.json"
    bundle.write_text("{}", encoding="utf-8")
    command = [
        "/usr/bin/env",
        f"VIRTUAL_ENV={tmp_path / '.venv-current'}",
        str(tmp_path / "isaaclab.sh"),
        "-p",
        str(tmp_path / "scripts" / "benchmarks" / "training.py"),
        "--rl_library",
        "rsl_rl",
        "--task",
        "Isaac-Cartpole-Direct",
        "--num_envs",
        "4096",
        "--seed",
        "42",
        "--max_iterations",
        "300",
        "--output_path",
        str(run_dir),
        "--benchmark_formatter",
        "schema",
        "--headless",
        "presets=newton_kamino",
    ]
    manifest = {
        "artifact_hashes": {bundle.name: sha256(bundle.read_bytes()).hexdigest()},
        "artifact_root": str(run_dir),
        "command": command,
        "command_hash": command_hash(command),
        "failure_category": None,
        "identity": {
            "max_iterations": 300,
            "num_envs": 4096,
            "phase": "full",
            "seed": 42,
            "task": "Isaac-Cartpole-Direct",
            "variant": "kamino_current",
        },
        "retry": {"attempt": 0, "parent_run_id": None},
        "revisions": {
            "isaaclab": matrix.revisions.isaaclab,
            "schema": matrix.revisions.schema,
            "newton_current": matrix.revisions.newton_current,
            "newton_pr": matrix.revisions.newton_pr,
        },
        "run_id": run_dir.name,
        "schema_version": "1.1",
        "state": "completed",
    }
    for key, value in (manifest_changes or {}).items():
        if key.startswith("identity."):
            manifest["identity"][key.removeprefix("identity.")] = value
        elif key.startswith("revisions."):
            manifest["revisions"][key.removeprefix("revisions.")] = value
        else:
            manifest[key] = value
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir.parent


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"identity.max_iterations": 299}, "max_iterations"),
        ({"revisions.isaaclab": "0" * 40}, "revisions"),
        ({"command_hash": "0" * 64}, "command hash"),
        ({"artifact_hashes": {"benchmark_training_test.json": "0" * 64}}, "artifact hash"),
    ],
)
def test_load_records_rejects_stale_manifest_provenance(tmp_path, monkeypatch, changes, message):
    """Completed labels are insufficient without exact protocol provenance and hashes."""
    artifact_root = _write_run_artifacts(tmp_path, manifest_changes=changes)
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    monkeypatch.setattr("benchmarks.kamino_dvi.analysis.locate_rsl_rl_events", lambda *_: tmp_path / "events")

    with pytest.raises(ValueError, match=message):
        load_records(artifact_root, tmp_path, matrix=matrix)


def test_load_records_rejects_bundle_identity_that_disagrees_with_manifest(tmp_path, monkeypatch):
    """A stale bundle must not inherit the task, seed, or capacity label from its directory."""
    artifact_root = _write_run_artifacts(tmp_path)
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    series = (1.0,) * 300
    trace = SimpleNamespace(
        task="Isaac-Ant-Direct",
        seed=42,
        num_envs=4096,
        iterations=300,
        iteration_time_s=series,
        total_fps=series,
        reward=series,
        ep_length=series,
        success_rate=series,
        success_schema_mismatch=False,
        success_schema_mismatch_points=0,
    )
    monkeypatch.setattr("benchmarks.kamino_dvi.analysis.locate_rsl_rl_events", lambda *_: tmp_path / "events")
    monkeypatch.setattr("benchmarks.kamino_dvi.analysis.parse_training_trace", lambda *_: trace)

    with pytest.raises(ValueError, match="bundle identity"):
        load_records(artifact_root, tmp_path, matrix=matrix)


def test_analyze_rejects_partial_matrix_before_writing_report(tmp_path, monkeypatch):
    """The CLI rejects a valid three-seed group when the remaining matrix is absent."""
    series = (1.0,) * 20
    records = [
        SimpleNamespace(
            task="Isaac-Cartpole-Direct",
            variant="kamino_current",
            seed=seed,
            num_envs=4096,
            iteration_time_s=series,
            total_fps=series,
            reward=series,
            ep_length=series,
            success_rate=series,
            success_schema_mismatch=False,
            success_schema_mismatch_points=0,
        )
        for seed in (42, 43, 44)
    ]
    monkeypatch.setattr("benchmarks.kamino_dvi.analyze.load_records", lambda *_args, **_kwargs: records)

    with pytest.raises(ValueError, match="missing"):
        main(["--artifact-root", str(tmp_path), "--logs-root", str(tmp_path), "--output-dir", str(tmp_path / "out")])


def test_load_records_ignores_explicitly_invalidated_full_result(tmp_path, monkeypatch):
    """Invalidated manifests retain raw evidence but cannot become analyzer records."""
    artifact_root = _write_run_artifacts(tmp_path, manifest_changes={"state": "invalidated"})
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    monkeypatch.setattr(
        "benchmarks.kamino_dvi.analysis.parse_training_trace",
        lambda *_: pytest.fail("invalidated result was parsed"),
    )

    assert load_records(artifact_root, tmp_path, matrix=matrix) == []


def test_quality_issues_counts_legacy_full_manifests_without_event_hash(tmp_path):
    """Integrity disclosure counts completed full runs but excludes preflights."""
    for name, phase, event_hash in (
        ("full__hashed", "full", "a" * 64),
        ("full__legacy", "full", None),
        ("preflight__legacy", "preflight", None),
    ):
        run_dir = tmp_path / name
        run_dir.mkdir()
        manifest = {"state": "completed", "identity": {"phase": phase}}
        if event_hash is not None:
            manifest["tensorboard_event_path"] = "/logs/events.out.tfevents.test"
            manifest["tensorboard_event_hash"] = event_hash
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    issues = quality_issues([], [], tmp_path)

    assert any("TensorBoard event hash was not recorded" in issue and "1/2" in issue for issue in issues)


@pytest.mark.parametrize(
    ("selector", "replacement"),
    [
        ("--task", "Isaac-Ant-Direct"),
        ("--seed", "43"),
        ("--num_envs", "2048"),
        ("--max_iterations", "299"),
        ("--output_path", "/tmp/stale-output"),
        ("--rl_library", "skrl"),
        ("--benchmark_formatter", "json"),
        ("presets=", "presets=physx"),
        ("VIRTUAL_ENV=", "VIRTUAL_ENV=/tmp/.venv-pr3570"),
    ],
)
def test_load_records_rejects_self_hashed_command_with_wrong_semantics(tmp_path, monkeypatch, selector, replacement):
    """A self-consistent command hash cannot legitimize a command for another run."""
    artifact_root = _write_run_artifacts(tmp_path)
    manifest_path = next(artifact_root.glob("full__*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    command = manifest["command"]
    if selector.startswith("--"):
        command[command.index(selector) + 1] = replacement
    else:
        index = next(index for index, value in enumerate(command) if value.startswith(selector))
        command[index] = replacement
    manifest["command_hash"] = command_hash(command)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    monkeypatch.setattr("benchmarks.kamino_dvi.analysis.locate_rsl_rl_events", lambda *_: tmp_path / "events")

    with pytest.raises(ValueError, match="command semantics"):
        load_records(artifact_root, tmp_path, matrix=matrix)


def test_load_records_rejects_missing_headless_command_flag(tmp_path, monkeypatch):
    """The recorded training command must preserve headless benchmark execution."""
    artifact_root = _write_run_artifacts(tmp_path)
    manifest_path = next(artifact_root.glob("full__*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["command"].remove("--headless")
    manifest["command_hash"] = command_hash(manifest["command"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    monkeypatch.setattr("benchmarks.kamino_dvi.analysis.locate_rsl_rl_events", lambda *_: tmp_path / "events")

    with pytest.raises(ValueError, match="command semantics"):
        load_records(artifact_root, tmp_path, matrix=matrix)


def test_load_records_rejects_wrong_recorded_tensorboard_event_hash(tmp_path, monkeypatch):
    """Future manifests must authenticate the exact TensorBoard event parsed by analysis."""
    artifact_root = _write_run_artifacts(tmp_path)
    manifest_path = next(artifact_root.glob("full__*/manifest.json"))
    event_path = tmp_path / "events.out.tfevents.test"
    event_path.write_bytes(b"event data")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tensorboard_event_path"] = str(event_path)
    manifest["tensorboard_event_hash"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    monkeypatch.setattr("benchmarks.kamino_dvi.analysis.locate_rsl_rl_events", lambda *_: event_path)

    with pytest.raises(ValueError, match="TensorBoard event hash"):
        load_records(artifact_root, tmp_path, matrix=matrix)


def test_load_records_rejects_unconfigured_hydra_solver_override(tmp_path, monkeypatch):
    """A correct preset cannot be combined with an unconfigured solver override."""
    artifact_root = _write_run_artifacts(tmp_path)
    manifest_path = next(artifact_root.glob("full__*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["command"].append("env.sim.physics.solver_cfg.dynamics_solver=dvi")
    manifest["command_hash"] = command_hash(manifest["command"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    monkeypatch.setattr("benchmarks.kamino_dvi.analysis.locate_rsl_rl_events", lambda *_: tmp_path / "events")

    with pytest.raises(ValueError, match="command semantics"):
        load_records(artifact_root, tmp_path, matrix=matrix)
