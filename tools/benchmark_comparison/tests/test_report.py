# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for informational Markdown benchmark reports."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tools.benchmark_comparison.manifest import HostIdentity, RunSetManifest, SoftwareIdentity
from tools.benchmark_comparison.matrix import expand_final_matrix, load_matrix
from tools.benchmark_comparison.models import ExecutionProvenance, RunSet
from tools.benchmark_comparison.normalize import FailureRow, NormalizedRun, write_normalized_outputs
from tools.benchmark_comparison.report import read_provenance, write_markdown_report, write_provenance


def _provenance() -> ExecutionProvenance:
    return ExecutionProvenance(
        lab2_sha="a" * 40,
        lab3_sha="b" * 40,
        lab2_image_id="sha256:" + "c" * 64,
        uv_lock_sha256="d" * 64,
    )


def _manifest() -> RunSetManifest:
    return RunSetManifest(
        schema_version="2.0",
        run_set=RunSet.FINAL,
        phase="measured",
        provenance=_provenance(),
        host=HostIdentity(
            "host",
            "Ubuntu",
            "CPU",
            32,
            "NVIDIA Test GPU",
            "590.00",
            "13.0",
            gpu_index=0,
            gpu_uuid="GPU-TEST-0000",
        ),
        lab2=SoftwareIdentity("2.3.2", "5.1", "3.11", "2.7", "5.0"),
        lab3=SoftwareIdentity("3.0.0", "6.0", "3.12", "2.8", "5.4"),
        expansion=expand_final_matrix(load_matrix()),
    )


def _attempt_directory(
    logical_task: str,
    mode: str,
    bound_unit: str,
    bound: int,
    seed: int,
    version: str,
) -> str:
    repeat = {42: 0, 43: 1, 44: 2}[seed]
    first_version = "lab3" if seed == 43 else "lab2"
    version_order = 0 if version == first_version else 1
    return (
        f"final--{logical_task}--{mode}--{bound_unit}-{bound}--seed-{seed}--repeat-{repeat}"
        f"--envs-4096--rsl_rl--{version}--version-order-{version_order}"
    )


def _run(version: str, fps: float) -> NormalizedRun:
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
        artifact_path=f"final/{_attempt_directory('cartpole', 'runtime-100', 'steps', 100, 42, version)}/success",
        isaac_lab_version="2.3.2" if version == "lab2" else "3.0.0",
        isaac_sim_version="5.1" if version == "lab2" else "6.0",
        python_version="3.11" if version == "lab2" else "3.12",
        pytorch_version="2.7" if version == "lab2" else "2.8",
        rsl_rl_version="5.0" if version == "lab2" else "5.4",
    )


def test_report_contains_methodology_inventory_mapping_modes_deltas_samples_and_failure_links(tmp_path: Path) -> None:
    runs = (_run("lab2", 100.0), _run("lab3", 125.0))
    failures = (
        FailureRow(
            version="lab3",
            logical_task="ant",
            concrete_task="Isaac-Ant",
            mode="training-100",
            bound=100,
            bound_unit="iterations",
            seed=44,
            num_envs=4096,
            attempt_number=1,
            failure_kind="out_of_memory",
            reason="benchmark ran out of memory",
            artifact_path=(
                "final/"
                + _attempt_directory("ant", "training-100", "iterations", 100, 44, "lab3")
                + "/attempt-0001-out_of_memory"
            ),
        ),
    )
    normalized = write_normalized_outputs(tmp_path / "normalized", runs, failures)
    report_path = write_markdown_report(
        normalized["raw_runs"],
        normalized["paired_summary"],
        normalized["failures"],
        tmp_path / "report" / "report.md",
        manifest=_manifest(),
    )

    text = report_path.read_text(encoding="utf-8")
    assert report_path.name == "report.md"
    for heading in (
        "# Isaac Lab Paired Benchmark Report",
        "## Methodology",
        "## Pinned revisions and execution identities",
        "## Hardware and software inventory",
        "## Task mapping",
        "## runtime-100",
        "## Failures and missing attempts",
    ):
        assert heading in text
    assert "informational" in text
    assert "Isaac-Cartpole-v0" in text and "Isaac-Cartpole" in text
    assert "+25.000%" in text
    assert "GPU utilization samples" in text
    assert "attempt-0001-out_of_memory" in text
    assert "not imputed" in text


def test_report_renders_partial_data_without_inventing_missing_values(tmp_path: Path) -> None:
    runs = (_run("lab2", 0.0),)
    failures: tuple[FailureRow, ...] = ()
    normalized = write_normalized_outputs(tmp_path / "normalized", runs, failures)

    report_path = write_markdown_report(
        normalized["raw_runs"],
        normalized["paired_summary"],
        normalized["failures"],
        tmp_path / "report.md",
        manifest=_manifest(),
    )

    text = report_path.read_text(encoding="utf-8")
    assert "No valid paired results." in text
    assert "No failed or missing attempts were recorded." in text
    assert "0.000" in text


def test_report_uses_structured_provenance_file_when_a_version_has_no_successes(tmp_path: Path) -> None:
    failures = (
        FailureRow(
            version="lab3",
            logical_task="ant",
            concrete_task="Isaac-Ant",
            mode="training-100",
            bound=100,
            bound_unit="iterations",
            seed=42,
            num_envs=4096,
            attempt_number=1,
            failure_kind="out_of_memory",
            reason="benchmark ran out of memory",
            artifact_path=(
                "final/"
                + _attempt_directory("ant", "training-100", "iterations", 100, 42, "lab3")
                + "/attempt-0001-out_of_memory"
            ),
        ),
    )
    normalized = write_normalized_outputs(tmp_path / "normalized", (), failures)
    provenance_path = write_provenance(tmp_path / "provenance.json", _provenance())

    report_path = write_markdown_report(
        normalized["raw_runs"],
        normalized["paired_summary"],
        normalized["failures"],
        tmp_path / "report.md",
        manifest=_manifest(),
    )

    assert read_provenance(provenance_path) == _provenance()
    assert provenance_path.read_text(encoding="utf-8") == (
        "{\n"
        f'  "lab2_image_id": "sha256:{"c" * 64}",\n'
        f'  "lab2_sha": "{"a" * 40}",\n'
        f'  "lab3_sha": "{"b" * 40}",\n'
        f'  "uv_lock_sha256": "{"d" * 64}"\n'
        "}\n"
    )
    text = report_path.read_text(encoding="utf-8")
    assert f"`{'a' * 40}`" in text
    assert f"`{'b' * 40}`" in text
    assert f"`sha256:{'c' * 64}`" in text
    assert f"`uv-lock:{'d' * 64}`" in text
    assert "`out_of_memory`" in text

    original = provenance_path.read_bytes()
    assert write_provenance(provenance_path, _provenance()).read_bytes() == original
    with pytest.raises(ValueError, match="different benchmark provenance"):
        write_provenance(provenance_path, replace(_provenance(), uv_lock_sha256="other-lock"))
    assert provenance_path.read_bytes() == original


@pytest.mark.parametrize(
    ("version", "field", "bad_value"),
    [
        ("lab2", "version_sha", "f" * 40),
        ("lab3", "environment_identity", "uv-lock:wrong"),
    ],
)
def test_report_rejects_normalized_rows_that_mismatch_expected_provenance(
    tmp_path: Path,
    version: str,
    field: str,
    bad_value: str,
) -> None:
    run = replace(_run(version, 100.0), **{field: bad_value})
    normalized = write_normalized_outputs(tmp_path / "normalized", (run,), ())

    with pytest.raises(ValueError, match="provenance"):
        write_markdown_report(
            normalized["raw_runs"],
            normalized["paired_summary"],
            normalized["failures"],
            tmp_path / "report.md",
            manifest=_manifest(),
        )
