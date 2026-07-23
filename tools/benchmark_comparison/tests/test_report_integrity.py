# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for report input and output integrity."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest

from tools.benchmark_comparison.manifest import HostIdentity, RunSetManifest, SoftwareIdentity, write_manifest
from tools.benchmark_comparison.models import ExecutionProvenance, RunSet
from tools.benchmark_comparison.normalize import FailureRow, NormalizedRun, write_normalized_outputs
from tools.benchmark_comparison.report import write_markdown_report
from tools.benchmark_comparison.report_cli import main


def _manifest() -> RunSetManifest:
    return RunSetManifest(
        schema_version="1.0",
        run_set=RunSet.FINAL,
        phase="measured",
        provenance=ExecutionProvenance("a" * 40, "b" * 40, "sha256:" + "c" * 64, "d" * 64),
        host=HostIdentity("host", "Ubuntu", "CPU", 32, "GPU", "590.00", "13.0"),
        lab2=SoftwareIdentity("2.3.2", "5.1", "3.11", "2.7", "5.0"),
        lab3=SoftwareIdentity("3.0.0", "6.0", "3.12", "2.8", "5.4"),
    )


def _attempt_directory(version: str) -> str:
    version_order = 0 if version == "lab2" else 1
    return (
        "final--cartpole--runtime-100--steps-100--seed-42--repeat-0"
        f"--envs-4096--rsl_rl--{version}--version-order-{version_order}"
    )


def _run(version: str, fps: float) -> NormalizedRun:
    software = _manifest().software(version)
    return NormalizedRun(
        version=version,
        version_sha=("a" if version == "lab2" else "b") * 40,
        environment_identity="sha256:" + "c" * 64 if version == "lab2" else "uv-lock:" + "d" * 64,
        logical_task="cartpole",
        concrete_task="Isaac-Cartpole-v0" if version == "lab2" else "Isaac-Cartpole",
        mode="runtime-100",
        bound=100,
        bound_unit="steps",
        seed=42,
        num_envs=4096,
        collection_fps=fps,
        gpu_memory_mean_mib=1024.0,
        gpu_memory_peak_mib=1536.0,
        gpu_utilization_mean_pct=75.0,
        gpu_utilization_sample_count=10,
        elapsed_time_s=20.0,
        artifact_path=f"final/{_attempt_directory(version)}/success",
        isaac_lab_version=software.isaac_lab,
        isaac_sim_version=software.isaac_sim,
        python_version=software.python,
        pytorch_version=software.pytorch,
        rsl_rl_version=software.rsl_rl,
    )


def test_report_rejects_paired_summary_not_derived_from_raw_runs(tmp_path: Path) -> None:
    normalized = write_normalized_outputs(
        tmp_path / "normalized",
        (_run("lab2", 100.0), _run("lab3", 125.0)),
        (),
    )
    with normalized["paired_summary"].open("a", encoding="utf-8") as file:
        file.write("runtime-100,cartpole,collection_fps,999,0,0,0,0,0,available\n")

    with pytest.raises(ValueError, match="paired summary"):
        write_markdown_report(
            normalized["raw_runs"],
            normalized["paired_summary"],
            normalized["failures"],
            tmp_path / "report.md",
            manifest=_manifest(),
        )


def test_report_rejects_failure_from_another_run_set(tmp_path: Path) -> None:
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
        failure_kind="nonzero_exit",
        reason="failed",
        artifact_path="canary/cartpole/failure",
    )
    normalized = write_normalized_outputs(tmp_path / "normalized", (), (failure,))

    with pytest.raises(ValueError, match="artifact path"):
        write_markdown_report(
            normalized["raw_runs"],
            normalized["paired_summary"],
            normalized["failures"],
            tmp_path / "report.md",
            manifest=_manifest(),
        )


def test_report_rejects_invalid_in_memory_manifest(tmp_path: Path) -> None:
    normalized = write_normalized_outputs(tmp_path / "normalized", (), ())

    with pytest.raises(ValueError, match="schema_version"):
        write_markdown_report(
            normalized["raw_runs"],
            normalized["paired_summary"],
            normalized["failures"],
            tmp_path / "report.md",
            manifest=replace(_manifest(), schema_version="2.0"),
        )


def test_report_only_cli_rejects_output_overlapping_raw_attempts(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    manifest = replace(_manifest(), run_set=RunSet.CANARY)
    write_manifest(artifact_root / "canary" / "manifest.json", manifest)

    with pytest.raises(ValueError, match="overlaps benchmark artifact root"):
        main(
            [
                "--artifact_root",
                str(artifact_root),
                "--run_set",
                "canary",
                "--phase",
                "measured",
                "--output_dir",
                str(artifact_root / "canary" / "success"),
            ]
        )


def test_report_only_cli_rejects_output_that_contains_raw_attempts(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    manifest = replace(_manifest(), run_set=RunSet.CANARY)
    write_manifest(artifact_root / "canary" / "manifest.json", manifest)

    with pytest.raises(ValueError, match="overlaps benchmark artifact root"):
        main(
            [
                "--artifact_root",
                str(artifact_root),
                "--run_set",
                "canary",
                "--phase",
                "measured",
                "--output_dir",
                str(artifact_root),
            ]
        )


def test_report_rejects_failure_csv_with_unexpected_columns(tmp_path: Path) -> None:
    normalized = write_normalized_outputs(tmp_path / "normalized", (), ())
    rows = list(csv.reader(normalized["failures"].open(newline="", encoding="utf-8")))
    rows[0].append("untrusted")
    with normalized["failures"].open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerows(rows)

    with pytest.raises(ValueError, match="columns"):
        write_markdown_report(
            normalized["raw_runs"],
            normalized["paired_summary"],
            normalized["failures"],
            tmp_path / "report.md",
            manifest=_manifest(),
        )
