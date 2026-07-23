# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for informational Markdown benchmark reports."""

from __future__ import annotations

from pathlib import Path

from tools.benchmark_comparison.normalize import FailureRow, NormalizedRun, write_normalized_outputs
from tools.benchmark_comparison.report import write_markdown_report


def _run(version: str, fps: float) -> NormalizedRun:
    return NormalizedRun(
        version=version,
        version_sha=("a" if version == "lab2" else "b") * 40,
        environment_identity="sha256:image" if version == "lab2" else "uv-lock:lock",
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
        artifact_path=f"final/{version}/success",
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
            artifact_path="final/ant/failure",
        ),
    )
    normalized = write_normalized_outputs(tmp_path / "normalized", runs, failures)
    report_path = write_markdown_report(
        normalized["raw_runs"],
        normalized["paired_summary"],
        normalized["failures"],
        tmp_path / "report" / "report.md",
        inventory={
            "Lab 2 image": "sha256:image",
            "Lab 3 uv lock": "lock",
            "GPU": "NVIDIA Test GPU",
            "Driver": "590.00",
            "CUDA": "13.0",
            "Isaac Sim Lab 2": "5.1",
            "Isaac Sim Lab 3": "6.0",
            "PyTorch": "2.8",
            "RSL-RL": "5.4",
        },
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
    assert "[final/ant/failure](../final/ant/failure)" in text
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
        inventory={},
    )

    text = report_path.read_text(encoding="utf-8")
    assert "No valid paired results." in text
    assert "No failed or missing attempts were recorded." in text
    assert "0.000" in text
