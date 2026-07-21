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

from benchmarks.kamino_dvi.analysis import (
    RunMetrics,
    VariantSummary,
    load_records,
    validate_failed_preflight_omissions,
    validate_failure_omissions,
)
from benchmarks.kamino_dvi.analyze import main, quality_issues
from benchmarks.kamino_dvi.commands import build_training_command
from benchmarks.kamino_dvi.manifests import command_hash, stable_run_id
from benchmarks.kamino_dvi.matrix import DEFAULT_MATRIX_PATH, load_matrix
from benchmarks.kamino_dvi.models import Phase, RunIdentity, TaskName, Variant
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


def test_quality_issues_identifies_task_without_success_as_stack_bug(tmp_path):
    """A task with no success definition is reported explicitly instead of silently becoming zero."""
    estimate = Estimate(1.0, 0.1, 3)
    summaries = [VariantSummary("DR Legs", "kamino_pr_dvi", 4096, estimate, estimate, estimate, estimate, None)]
    records = [
        SimpleNamespace(
            task="DR Legs", success_schema_mismatch=False, success_schema_mismatch_points=0, success_rate=None
        )
        for _ in range(3)
    ]

    issues = quality_issues(records, summaries, tmp_path)

    assert any(
        "DR Legs" in issue
        and "learning.success_rate.series_per_iter" in issue
        and "TensorBoard Metrics/success_rate" in issue
        and "benchmark/task-stack bug" in issue
        and "N/A" in issue
        for issue in issues
    )
    assert not any("every required reward, episode-length, and success field exists" in issue for issue in issues)


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

    assert any("1/2 completed full runs" in issue and "git_dirty=true" in issue for issue in issues)
    assert any("2 distinct commits" in issue for issue in issues)
    assert sum("git_dirty=true" in issue for issue in issues) == 1
    assert any("did not pass the current clean-source check" in issue for issue in issues)
    assert any(
        "TensorBoard event files" in issue and "2/2" in issue and "not retained or hashed" in issue for issue in issues
    )


def test_quality_issues_counts_bundle_git_dirty_across_current_and_legacy_runs(tmp_path):
    """Bundle workspace status covers every completed full run without duplicating the legacy warning."""
    for name, head, dirty in (
        ("legacy", None, False),
        ("current", "c" * 40, True),
    ):
        run_dir = tmp_path / f"full__{name}"
        run_dir.mkdir()
        manifest = {
            "state": "completed",
            "tensorboard_event_path": f"/logs/events.out.tfevents.{name}",
            "tensorboard_event_hash": "a" * 64,
        }
        if head is not None:
            manifest["isaaclab_head"] = head
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run_dir / f"benchmark_training_{name}.json").write_text(
            json.dumps({"versions": {"git_commit": "b" * 40, "git_dirty": dirty}}), encoding="utf-8"
        )

    issues = quality_issues([], [], tmp_path)

    dirty_issues = [issue for issue in issues if "versions.git_dirty=true" in issue]
    assert len(dirty_issues) == 1
    assert "1/2 completed full runs" in dirty_issues[0]
    assert "includes untracked paths" in dirty_issues[0]
    legacy_issue = next(issue for issue in issues if issue.startswith("Legacy campaign source provenance:"))
    assert "versions.git_dirty" not in legacy_issue


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


def _complete_records_except(matrix, omitted: tuple[str, str]) -> list[RunMetrics]:
    """Build a complete synthetic matrix except for one whole task/variant cell."""
    series = (1.0,) * matrix.full_iterations
    return [
        RunMetrics(
            task.name.value,
            variant.value,
            seed,
            4096,
            series,
            series,
            series,
            series,
            series,
        )
        for task in matrix.tasks
        for variant in task.variants
        for seed in matrix.seeds
        if (task.name.value, variant.value) != omitted
    ]


def _write_failed_preflight(
    tmp_path: Path,
    *,
    manifest_changes: dict | None = None,
) -> tuple[Path, list[RunMetrics]]:
    """Write one exact retained numerical preflight and the complementary full records."""
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    identity = RunIdentity(
        TaskName.DR_LEGS_WALK,
        Variant.KAMINO_PR_DVI,
        matrix.preflight_seed,
        Phase.PREFLIGHT,
        4096,
        matrix.preflight_iterations,
    )
    artifact_root = tmp_path / "artifacts"
    run_dir = artifact_root / stable_run_id(identity)
    run_dir.mkdir(parents=True)
    stdout = run_dir / "stdout.log"
    stderr = run_dir / "stderr.log"
    stdout.write_text("Training failed: NaN\n", encoding="utf-8")
    stderr.write_text("numerical instability\n", encoding="utf-8")
    command = build_training_command(matrix, identity, tmp_path, run_dir)
    manifest = {
        "artifact_hashes": {
            stdout.name: sha256(stdout.read_bytes()).hexdigest(),
            stderr.name: sha256(stderr.read_bytes()).hexdigest(),
        },
        "artifact_root": str(run_dir),
        "command": command,
        "command_hash": command_hash(command),
        "failure_category": "numerical",
        "identity": {
            "max_iterations": identity.max_iterations,
            "num_envs": identity.num_envs,
            "phase": identity.phase.value,
            "seed": identity.seed,
            "task": identity.task.value,
            "variant": identity.variant.value,
        },
        "isaaclab_head": "f" * 40,
        "retry": {"attempt": 0, "parent_run_id": None},
        "revisions": {
            "isaaclab": matrix.revisions.isaaclab,
            "schema": matrix.revisions.schema,
            "newton_current": matrix.revisions.newton_current,
            "newton_pr": matrix.revisions.newton_pr,
        },
        "run_id": run_dir.name,
        "schema_version": "1.1",
        "state": "failed",
        "tensorboard_event_hash": None,
        "tensorboard_event_path": None,
    }
    for key, value in (manifest_changes or {}).items():
        if key.startswith("identity."):
            manifest["identity"][key.removeprefix("identity.")] = value
        elif key.startswith("revisions."):
            manifest["revisions"][key.removeprefix("revisions.")] = value
        else:
            manifest[key] = value
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    omitted = (identity.task.value, identity.variant.value)
    return artifact_root, _complete_records_except(matrix, omitted)


def test_failed_preflight_omits_exact_whole_cell_and_is_disclosed(tmp_path):
    """An exact numerical preflight authorizes omitting all full seeds for only its cell."""
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    artifact_root, records = _write_failed_preflight(tmp_path)

    omissions = validate_failed_preflight_omissions(records, artifact_root, matrix)
    issues = quality_issues(records, [], artifact_root)

    assert omissions == {("IsaacContrib-DrLegs-Walk", "kamino_pr_dvi")}
    assert any("Failed preflight" in issue and "numerical" in issue for issue in issues)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"state": "completed"}, "terminal failed preflight"),
        ({"schema_version": "1.0"}, "schema version"),
        ({"identity.seed": 43}, "identity does not match"),
        ({"identity.num_envs": 2048}, "identity does not match"),
        ({"revisions.isaaclab": "0" * 40}, "revisions"),
        ({"command_hash": "0" * 64}, "command hash"),
        ({"artifact_hashes": {}}, "retained stdout and stderr"),
    ],
)
def test_failed_preflight_omission_rejects_inexact_evidence(tmp_path, changes, message):
    """Stale, non-terminal, or wrong-identity preflights cannot excuse missing full runs."""
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    artifact_root, records = _write_failed_preflight(tmp_path, manifest_changes=changes)

    with pytest.raises(ValueError, match=message):
        validate_failed_preflight_omissions(records, artifact_root, matrix)


def _write_partial_failed_full_campaign(
    tmp_path: Path,
    *,
    completed_seeds: tuple[int, ...] = (42, 44),
    manifest_changes: dict | None = None,
) -> tuple[Path, list[RunMetrics]]:
    """Write one failed full seed plus the cell's selected successful seeds."""
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    identity = RunIdentity(
        TaskName.DR_LEGS_WALK,
        Variant.KAMINO_CURRENT,
        43,
        Phase.FULL,
        4096,
        matrix.full_iterations,
    )
    artifact_root = tmp_path / "artifacts"
    run_dir = artifact_root / stable_run_id(identity)
    run_dir.mkdir(parents=True)
    stdout = run_dir / "stdout.log"
    stderr = run_dir / "stderr.log"
    stdout.write_text("Learning iteration 172/300\n", encoding="utf-8")
    stderr.write_text("policy observations contain NaN\n", encoding="utf-8")
    command = build_training_command(matrix, identity, tmp_path, run_dir)
    manifest = {
        "artifact_hashes": {
            stdout.name: sha256(stdout.read_bytes()).hexdigest(),
            stderr.name: sha256(stderr.read_bytes()).hexdigest(),
        },
        "artifact_root": str(run_dir),
        "command": command,
        "command_hash": command_hash(command),
        "failure_category": "numerical",
        "identity": {
            "max_iterations": identity.max_iterations,
            "num_envs": identity.num_envs,
            "phase": identity.phase.value,
            "seed": identity.seed,
            "task": identity.task.value,
            "variant": identity.variant.value,
        },
        "isaaclab_head": "f" * 40,
        "retry": {"attempt": 0, "parent_run_id": None},
        "revisions": {
            "isaaclab": matrix.revisions.isaaclab,
            "schema": matrix.revisions.schema,
            "newton_current": matrix.revisions.newton_current,
            "newton_pr": matrix.revisions.newton_pr,
        },
        "run_id": run_dir.name,
        "schema_version": "1.1",
        "state": "failed",
        "tensorboard_event_hash": None,
        "tensorboard_event_path": None,
    }
    for key, value in (manifest_changes or {}).items():
        if key.startswith("identity."):
            manifest["identity"][key.removeprefix("identity.")] = value
        elif key.startswith("revisions."):
            manifest["revisions"][key.removeprefix("revisions.")] = value
        else:
            manifest[key] = value
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    omitted = (identity.task.value, identity.variant.value)
    records = _complete_records_except(matrix, omitted)
    series = (1.0,) * matrix.full_iterations
    records.extend(
        RunMetrics(
            identity.task.value,
            identity.variant.value,
            seed,
            identity.num_envs,
            series,
            series,
            series,
            series,
            series,
        )
        for seed in completed_seeds
    )
    return artifact_root, records


def test_failed_full_omission_requires_exact_failure_for_each_missing_seed(tmp_path):
    """One exact failed full seed excludes its whole partially successful cell."""
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    artifact_root, records = _write_partial_failed_full_campaign(tmp_path)

    omissions = validate_failure_omissions(records, artifact_root, matrix)

    assert omissions == {("IsaacContrib-DrLegs-Walk", "kamino_current")}


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"state": "completed"}, "terminal failed full run"),
        ({"schema_version": "1.0"}, "schema version"),
        ({"identity.seed": 42}, "identity does not match"),
        ({"identity.num_envs": 2048}, "identity does not match"),
        ({"revisions.isaaclab": "0" * 40}, "revisions"),
        ({"command_hash": "0" * 64}, "command hash"),
        ({"artifact_hashes": {}}, "retained stdout and stderr"),
    ],
)
def test_failed_full_omission_rejects_inexact_evidence(tmp_path, changes, message):
    """A stale or unauthenticated failed full run cannot excuse partial coverage."""
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    artifact_root, records = _write_partial_failed_full_campaign(tmp_path, manifest_changes=changes)

    with pytest.raises(ValueError, match=message):
        validate_failure_omissions(records, artifact_root, matrix)


def test_failed_full_omission_rejects_second_missing_seed_without_failure(tmp_path):
    """Every missing full seed needs its own exact terminal failure evidence."""
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    artifact_root, records = _write_partial_failed_full_campaign(tmp_path, completed_seeds=(42,))

    with pytest.raises(ValueError, match="missing failed full run.*seed=44"):
        validate_failure_omissions(records, artifact_root, matrix)


def test_main_generates_report_without_failed_preflight_cell(tmp_path, monkeypatch):
    """The report entry point summarizes complete groups and omits the failed preflight bar."""
    artifact_root, records = _write_failed_preflight(tmp_path)
    output_dir = tmp_path / "report"
    written: dict[str, object] = {}
    monkeypatch.setattr("benchmarks.kamino_dvi.analyze.load_records", lambda *_args, **_kwargs: records)
    monkeypatch.setattr(
        "benchmarks.kamino_dvi.analyze.plot_runtime",
        lambda _summaries, path: path.write_bytes(b"runtime"),
    )
    monkeypatch.setattr(
        "benchmarks.kamino_dvi.analyze.plot_learning",
        lambda _summaries, path: path.write_bytes(b"learning"),
    )
    monkeypatch.setattr(
        "benchmarks.kamino_dvi.analyze.write_reports",
        lambda summaries, issues, *_args: written.update(summaries=summaries, issues=issues),
    )

    assert (
        main(
            [
                "--artifact-root",
                str(artifact_root),
                "--logs-root",
                str(tmp_path),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    summaries = written["summaries"]
    assert len(summaries) == 20
    assert not any(
        summary.task == TaskName.DR_LEGS_WALK and summary.variant == Variant.KAMINO_PR_DVI for summary in summaries
    )
    assert any("Failed preflight" in issue and "numerical" in issue for issue in written["issues"])


def test_main_excludes_partial_failed_full_cell_but_keeps_quality_records(tmp_path, monkeypatch):
    """Partial successes stay in quality accounting but not summaries when another seed failed."""
    artifact_root, records = _write_partial_failed_full_campaign(tmp_path)
    output_dir = tmp_path / "report"
    written: dict[str, object] = {}
    quality_record_count: list[int] = []
    real_quality_issues = quality_issues
    monkeypatch.setattr("benchmarks.kamino_dvi.analyze.load_records", lambda *_args, **_kwargs: records)
    monkeypatch.setattr(
        "benchmarks.kamino_dvi.analyze.plot_runtime",
        lambda _summaries, path: path.write_bytes(b"runtime"),
    )
    monkeypatch.setattr(
        "benchmarks.kamino_dvi.analyze.plot_learning",
        lambda _summaries, path: path.write_bytes(b"learning"),
    )
    monkeypatch.setattr(
        "benchmarks.kamino_dvi.analyze.quality_issues",
        lambda quality_records, summaries, root: (
            quality_record_count.append(len(quality_records)) or real_quality_issues(quality_records, summaries, root)
        ),
    )
    monkeypatch.setattr(
        "benchmarks.kamino_dvi.analyze.write_reports",
        lambda summaries, issues, *_args: written.update(summaries=summaries, issues=issues),
    )

    assert (
        main(
            [
                "--artifact-root",
                str(artifact_root),
                "--logs-root",
                str(tmp_path),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    summaries = written["summaries"]
    assert len(summaries) == 20
    assert not any(
        summary.task == TaskName.DR_LEGS_WALK and summary.variant == Variant.KAMINO_CURRENT for summary in summaries
    )
    assert quality_record_count == [62]
    assert any("Failed full" in issue and "seed 43" in issue and "numerical" in issue for issue in written["issues"])


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
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    artifact_root = _write_run_artifacts(tmp_path, manifest_changes=changes)
    monkeypatch.setattr("benchmarks.kamino_dvi.analysis.locate_rsl_rl_events", lambda *_: tmp_path / "events")

    with pytest.raises(ValueError, match=message):
        load_records(artifact_root, tmp_path, matrix=matrix)


def test_load_records_rejects_bundle_identity_that_disagrees_with_manifest(tmp_path, monkeypatch):
    """A stale bundle must not inherit the task, seed, or capacity label from its directory."""
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    artifact_root = _write_run_artifacts(tmp_path)
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
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    artifact_root = _write_run_artifacts(tmp_path, manifest_changes={"state": "invalidated"})
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
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
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
    monkeypatch.setattr("benchmarks.kamino_dvi.analysis.locate_rsl_rl_events", lambda *_: tmp_path / "events")

    with pytest.raises(ValueError, match="command semantics"):
        load_records(artifact_root, tmp_path, matrix=matrix)


def test_load_records_rejects_missing_headless_command_flag(tmp_path, monkeypatch):
    """The recorded training command must preserve headless benchmark execution."""
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    artifact_root = _write_run_artifacts(tmp_path)
    manifest_path = next(artifact_root.glob("full__*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["command"].remove("--headless")
    manifest["command_hash"] = command_hash(manifest["command"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr("benchmarks.kamino_dvi.analysis.locate_rsl_rl_events", lambda *_: tmp_path / "events")

    with pytest.raises(ValueError, match="command semantics"):
        load_records(artifact_root, tmp_path, matrix=matrix)


def test_load_records_rejects_wrong_recorded_tensorboard_event_hash(tmp_path, monkeypatch):
    """Future manifests must authenticate the exact TensorBoard event parsed by analysis."""
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    artifact_root = _write_run_artifacts(tmp_path)
    manifest_path = next(artifact_root.glob("full__*/manifest.json"))
    event_path = tmp_path / "events.out.tfevents.test"
    event_path.write_bytes(b"event data")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tensorboard_event_path"] = str(event_path)
    manifest["tensorboard_event_hash"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr("benchmarks.kamino_dvi.analysis.locate_rsl_rl_events", lambda *_: event_path)

    with pytest.raises(ValueError, match="TensorBoard event hash"):
        load_records(artifact_root, tmp_path, matrix=matrix)


def test_load_records_rejects_unconfigured_hydra_solver_override(tmp_path, monkeypatch):
    """A correct preset cannot be combined with an unconfigured solver override."""
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    artifact_root = _write_run_artifacts(tmp_path)
    manifest_path = next(artifact_root.glob("full__*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["command"].append("env.sim.physics.solver_cfg.dynamics_solver=dvi")
    manifest["command_hash"] = command_hash(manifest["command"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr("benchmarks.kamino_dvi.analysis.locate_rsl_rl_events", lambda *_: tmp_path / "events")

    with pytest.raises(ValueError, match="command semantics"):
        load_records(artifact_root, tmp_path, matrix=matrix)
