# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for report input and publication path safety."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.benchmark_comparison.manifest import HostIdentity, RunSetManifest, SoftwareIdentity, write_manifest
from tools.benchmark_comparison.matrix import expand_final_matrix, load_matrix
from tools.benchmark_comparison.models import ExecutionProvenance, RunSet
from tools.benchmark_comparison.normalize import FailureRow, NormalizedRun, write_normalized_outputs
from tools.benchmark_comparison.report import write_markdown_report
from tools.benchmark_comparison.report_cli import _validate_output_directory, main


def _manifest() -> RunSetManifest:
    return RunSetManifest(
        schema_version="2.0",
        run_set=RunSet.FINAL,
        phase="measured",
        provenance=ExecutionProvenance("a" * 40, "b" * 40, "sha256:" + "c" * 64, "d" * 64),
        host=HostIdentity(
            "host",
            "Ubuntu",
            "CPU",
            32,
            "GPU",
            "590.00",
            "13.0",
            gpu_index=0,
            gpu_uuid="GPU-TEST-0000",
        ),
        lab2=SoftwareIdentity("2.3.2", "5.1", "3.11", "2.7", "5.0"),
        lab3=SoftwareIdentity("3.0.0", "6.0", "3.12", "2.8", "5.4"),
        expansion=expand_final_matrix(load_matrix()),
    )


def _attempt_directory(version: str = "lab2") -> str:
    version_order = 0 if version == "lab2" else 1
    return (
        "final--cartpole--runtime-100--steps-100--seed-42--repeat-0"
        f"--envs-4096--rsl_rl--{version}--version-order-{version_order}"
    )


def _run(artifact_path: str) -> NormalizedRun:
    software = _manifest().lab2
    return NormalizedRun(
        version="lab2",
        version_sha="a" * 40,
        environment_identity="sha256:" + "c" * 64,
        logical_task="cartpole",
        concrete_task="Isaac-Cartpole-v0",
        mode="runtime-100",
        bound=100,
        bound_unit="steps",
        seed=42,
        num_envs=4096,
        collection_fps=100.0,
        gpu_memory_mean_mib=1024.0,
        gpu_memory_peak_mib=1536.0,
        gpu_utilization_mean_pct=75.0,
        gpu_utilization_sample_count=10,
        elapsed_time_s=20.0,
        startup_total_s=4.41,
        startup_app_launch_s=2.5,
        startup_python_imports_s=0.2,
        startup_task_config_s=0.4,
        startup_env_creation_s=1.3,
        startup_first_step_s=0.01,
        artifact_path=artifact_path,
        isaac_lab_version=software.isaac_lab,
        isaac_sim_version=software.isaac_sim,
        python_version=software.python,
        pytorch_version=software.pytorch,
        rsl_rl_version=software.rsl_rl,
    )


def _write_report(tmp_path: Path, artifact_path: str, *, artifact_root: Path | None = None) -> Path:
    normalized = write_normalized_outputs(tmp_path / "normalized", (_run(artifact_path),), ())
    return write_markdown_report(
        normalized["raw_runs"],
        normalized["paired_summary"],
        normalized["failures"],
        tmp_path / "report.md",
        manifest=_manifest(),
        artifact_root=artifact_root or tmp_path / "artifacts",
    )


@pytest.mark.parametrize(
    "output",
    [
        "canary",
        "canary/report",
        ".",
        "final/success",
        "..",
    ],
)
def test_output_policy_rejects_every_artifact_root_overlap(tmp_path: Path, output: str) -> None:
    artifact_root = tmp_path / "artifacts"
    output_directory = (artifact_root / output).resolve()

    with pytest.raises(ValueError, match="overlaps benchmark artifact root"):
        _validate_output_directory(artifact_root.resolve(), RunSet.FINAL, output_directory)


@pytest.mark.parametrize("output", ["final/report", "../external-report"])
def test_output_policy_accepts_only_canonical_or_disjoint_output(tmp_path: Path, output: str) -> None:
    artifact_root = tmp_path / "artifacts"
    output_directory = (artifact_root / output).resolve()

    _validate_output_directory(artifact_root.resolve(), RunSet.FINAL, output_directory)


@pytest.mark.parametrize("output", ["canary", "canary/report"])
def test_report_cli_rejects_final_output_inside_canary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output: str,
) -> None:
    artifact_root = tmp_path / "artifacts"
    write_manifest(artifact_root / "final" / "manifest.json", _manifest())
    canary_marker = artifact_root / "canary" / "raw-marker"
    canary_marker.parent.mkdir(parents=True)
    canary_marker.write_text("must survive", encoding="utf-8")
    monkeypatch.setattr("tools.benchmark_comparison.report_cli.generate_plots", lambda *_args, **_kwargs: ())

    with pytest.raises(ValueError, match="overlaps benchmark artifact root"):
        main(
            [
                "--artifact_root",
                str(artifact_root),
                "--run_set",
                "final",
                "--phase",
                "measured",
                "--output_dir",
                str(artifact_root / output),
            ]
        )
    assert canary_marker.read_text(encoding="utf-8") == "must survive"


def test_report_cli_accepts_disjoint_external_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    write_manifest(artifact_root / "final" / "manifest.json", _manifest())
    output = tmp_path / "external-report"
    monkeypatch.setattr("tools.benchmark_comparison.report_cli.generate_plots", lambda *_args, **_kwargs: ())

    assert (
        main(
            [
                "--artifact_root",
                str(artifact_root),
                "--run_set",
                "final",
                "--phase",
                "measured",
                "--output_dir",
                str(output),
            ]
        )
        == 0
    )
    assert (output / "report.md").is_file()


@pytest.mark.parametrize(
    "artifact_path",
    [
        "",
        f"final/../canary/{_attempt_directory()}/success",
        f"final/./{_attempt_directory()}/success",
        f"final/%2e%2e/{_attempt_directory()}/success",
        f"/final/{_attempt_directory()}/success",
        f"finality/{_attempt_directory()}/success",
        f"final\\{_attempt_directory()}\\success",
        f"final//{_attempt_directory()}/success",
        f"final/{_attempt_directory()}/success/",
    ],
)
def test_report_rejects_noncanonical_or_escaping_artifact_paths(tmp_path: Path, artifact_path: str) -> None:
    with pytest.raises(ValueError, match="artifact path"):
        _write_report(tmp_path, artifact_path)


def test_report_rejects_symlink_escape_from_selected_run_set(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    outside = tmp_path / "outside"
    outside.mkdir()
    attempt = artifact_root / "final" / _attempt_directory()
    attempt.parent.mkdir(parents=True)
    attempt.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="artifact path"):
        _write_report(
            tmp_path,
            f"final/{_attempt_directory()}/success",
            artifact_root=artifact_root,
        )


@pytest.mark.parametrize(
    ("failure_kind", "terminal"),
    [
        ("nonzero_exit", "attempt-0001-nonzero_exit"),
        ("invalid_success", "corrupt-success-0001"),
    ],
)
def test_report_accepts_canonical_failure_and_quarantine_paths(
    tmp_path: Path,
    failure_kind: str,
    terminal: str,
) -> None:
    attempt = _attempt_directory()
    failure = FailureRow(
        version="lab2",
        logical_task="cartpole",
        concrete_task="Isaac-Cartpole-v0",
        mode="runtime-100",
        bound=100,
        bound_unit="steps",
        seed=42,
        num_envs=4096,
        attempt_number=1,
        failure_kind=failure_kind,
        reason="expected test failure",
        artifact_path=f"final/{attempt}/{terminal}",
    )
    normalized = write_normalized_outputs(tmp_path / "normalized", (), (failure,))

    report = write_markdown_report(
        normalized["raw_runs"],
        normalized["paired_summary"],
        normalized["failures"],
        tmp_path / "report.md",
        manifest=_manifest(),
        artifact_root=tmp_path / "artifacts",
    )

    assert report.is_file()


def test_report_accepts_canonical_success_path(tmp_path: Path) -> None:
    report = _write_report(tmp_path, f"final/{_attempt_directory()}/success")

    assert report.is_file()
