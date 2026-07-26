# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for report input and output integrity."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tools.benchmark_comparison.manifest import HostIdentity, RunSetManifest, SoftwareIdentity, write_manifest
from tools.benchmark_comparison.matrix import (
    expand_canary_matrix,
    expand_final_matrix,
    expand_legacy_schema_1_matrix,
    load_matrix,
)
from tools.benchmark_comparison.models import ExecutionProvenance, RunSet
from tools.benchmark_comparison.normalize import FailureRow, NormalizedRun, write_normalized_outputs
from tools.benchmark_comparison.plot import PLOT_BASENAMES
from tools.benchmark_comparison.report import write_markdown_report
from tools.benchmark_comparison.report_cli import main


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
            gpu_uuid="GPU-01234567-89ab-cdef-0123-456789abcdef",
        ),
        lab2=SoftwareIdentity("2.3.2", "5.1", "3.11", "2.7", "5.0"),
        lab3=SoftwareIdentity("3.0.0", "6.0", "3.12", "2.8", "5.4"),
        expansion=expand_final_matrix(load_matrix()),
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
        startup_total_s=4.41,
        startup_app_launch_s=2.5,
        startup_python_imports_s=0.2,
        startup_task_config_s=0.4,
        startup_env_creation_s=1.3,
        startup_first_step_s=0.01,
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


def test_report_omits_integrity_appendix_without_audit(tmp_path: Path) -> None:
    normalized = write_normalized_outputs(
        tmp_path / "normalized",
        (_run("lab2", 100.0), _run("lab3", 125.0)),
        (),
    )

    report_path = write_markdown_report(
        normalized["raw_runs"],
        normalized["paired_summary"],
        normalized["failures"],
        tmp_path / "report.md",
        manifest=_manifest(),
    )

    assert "## Artifact integrity" not in report_path.read_text(encoding="utf-8")


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
            manifest=replace(_manifest(), schema_version="3.0"),
        )


def test_report_rejects_run_present_in_checkout_but_absent_from_manifest_snapshot(tmp_path: Path) -> None:
    run = replace(
        _run("lab2", 100.0),
        logical_task="cartpole_direct",
        concrete_task="Isaac-Cartpole-Direct-v0",
        artifact_path=(
            "final/final--cartpole_direct--runtime-100--steps-100--seed-42--repeat-0"
            "--envs-4096--rsl_rl--lab2--version-order-0/success"
        ),
    )
    normalized = write_normalized_outputs(tmp_path / "normalized", (run,), ())
    manifest = replace(_manifest(), expansion=expand_legacy_schema_1_matrix(RunSet.FINAL))

    with pytest.raises(ValueError, match="manifest run-set identity"):
        write_markdown_report(
            normalized["raw_runs"],
            normalized["paired_summary"],
            normalized["failures"],
            tmp_path / "report.md",
            manifest=manifest,
        )


def test_report_only_cli_rejects_output_overlapping_raw_attempts(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    manifest = replace(_manifest(), run_set=RunSet.CANARY, expansion=expand_canary_matrix(load_matrix()))
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
    manifest = replace(_manifest(), run_set=RunSet.CANARY, expansion=expand_canary_matrix(load_matrix()))
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


def test_report_only_cli_hashes_all_generated_files(tmp_path: Path) -> None:
    expected_generated_files = {
        "raw_runs.csv",
        "paired_summary.csv",
        "failures.csv",
        "report.md",
        "report.pdf",
    }
    expected_generated_files.update(f"{basename}.{suffix}" for basename in PLOT_BASENAMES for suffix in ("png", "svg"))
    assert len(PLOT_BASENAMES) == 26
    assert len(expected_generated_files) == 57
    artifact_root = tmp_path / "artifacts"
    manifest = replace(_manifest(), run_set=RunSet.CANARY, expansion=expand_canary_matrix(load_matrix()))
    write_manifest(artifact_root / "canary" / "manifest.json", manifest)
    output = artifact_root / "canary" / "report"
    report_args = [
        "--artifact_root",
        str(artifact_root),
        "--run_set",
        "canary",
        "--phase",
        "measured",
        "--output_dir",
        str(output),
    ]

    assert main(report_args) == 0

    metadata_files = {"audit_summary.json", "raw_artifact_hashes.sha256", "generated_hashes.sha256"}
    assert {path.name for path in output.iterdir()} == expected_generated_files | metadata_files
    generated_manifest = (output / "generated_hashes.sha256").read_bytes()
    manifest_lines = generated_manifest.decode().splitlines()
    entries = {}
    for line in manifest_lines:
        digest, relative_path = line.split("  ", maxsplit=1)
        entries[relative_path] = digest
    assert len(manifest_lines) == len(entries) == 57
    assert set(entries) == expected_generated_files
    assert all(
        hashlib.sha256((output / relative_path).read_bytes()).hexdigest() == digest
        for relative_path, digest in entries.items()
    )
    generated_before = {relative_path: (output / relative_path).read_bytes() for relative_path in entries}
    audit = json.loads((output / "audit_summary.json").read_text(encoding="utf-8"))
    assert audit["generated_file_count"] == 57
    assert audit["generated_hash_manifest_sha256"] == hashlib.sha256(generated_manifest).hexdigest()

    assert main(report_args) == 0

    assert (output / "generated_hashes.sha256").read_bytes() == generated_manifest
    assert json.loads((output / "audit_summary.json").read_text(encoding="utf-8"))["generated_file_count"] == 57
    assert {relative_path: (output / relative_path).read_bytes() for relative_path in entries} == generated_before
