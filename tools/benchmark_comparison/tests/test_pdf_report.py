# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the deterministic paginated PDF benchmark report."""

from __future__ import annotations

import re
import subprocess
from dataclasses import replace
from pathlib import Path

from tools.benchmark_comparison.manifest import HostIdentity, RunSetManifest, SoftwareIdentity
from tools.benchmark_comparison.matrix import expand_final_matrix, load_matrix
from tools.benchmark_comparison.models import ExecutionProvenance, RunSet
from tools.benchmark_comparison.normalize import NormalizedRun, write_normalized_outputs
from tools.benchmark_comparison.pdf_report import validate_pdf, write_pdf_report
from tools.benchmark_comparison.plot import PLOT_BASENAMES
from tools.benchmark_comparison.report import ReportAudit


def _manifest() -> RunSetManifest:
    return RunSetManifest(
        schema_version="2.0",
        run_set=RunSet.FINAL,
        phase="measured",
        provenance=ExecutionProvenance(
            lab2_sha="a" * 40,
            lab3_sha="b" * 40,
            lab2_image_id="sha256:" + "c" * 64,
            uv_lock_sha256="d" * 64,
        ),
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
        artifact_path=f"final/final--cartpole--runtime-100--seed-42--{version}/success",
        isaac_lab_version="2.3.2" if version == "lab2" else "3.0.0",
        isaac_sim_version="5.1" if version == "lab2" else "6.0",
        python_version="3.11" if version == "lab2" else "3.12",
        pytorch_version="2.7" if version == "lab2" else "2.8",
        rsl_rl_version="5.0" if version == "lab2" else "5.4",
    )


def _plot_paths(directory: Path) -> tuple[Path, ...]:
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True)
    paths: list[Path] = []
    for index, basename in enumerate(PLOT_BASENAMES):
        figure, axis = plt.subplots(figsize=(2, 1), dpi=80)
        axis.plot((0, 1), (index, index + 1))
        axis.set_title(basename)
        path = directory / f"{basename}.png"
        figure.savefig(path, metadata={"Software": "Isaac Lab benchmark comparison"})
        plt.close(figure)
        paths.append(path)
    return tuple(paths)


def _inputs(
    tmp_path: Path, runs: tuple[NormalizedRun, ...] | None = None
) -> tuple[dict[str, Path], RunSetManifest, ReportAudit, tuple[Path, ...]]:
    selected_runs = runs or (_run("lab2", 100.0), _run("lab3", 125.0))
    normalized = write_normalized_outputs(tmp_path / "normalized", selected_runs, ())
    audit = ReportAudit(
        successful_attempts=len(selected_runs),
        failed_or_missing_attempts=0,
        raw_file_count=25,
        generated_file_count=17,
        raw_hash_manifest_sha256="e" * 64,
    )
    return normalized, _manifest(), audit, _plot_paths(tmp_path / "plots")


def test_pdf_contains_large_report_and_regenerates_byte_identically(tmp_path: Path) -> None:
    normalized, manifest, audit, plots = _inputs(tmp_path)
    first = write_pdf_report(
        normalized["raw_runs"],
        normalized["paired_summary"],
        normalized["failures"],
        plots,
        tmp_path / "first.pdf",
        manifest=manifest,
        audit=audit,
    )
    second = write_pdf_report(
        normalized["raw_runs"],
        normalized["paired_summary"],
        normalized["failures"],
        plots,
        tmp_path / "second.pdf",
        manifest=manifest,
        audit=audit,
    )
    assert first.read_bytes().startswith(b"%PDF-")
    assert first.stat().st_size > 10_000
    assert first.read_bytes() == second.read_bytes()
    assert not first.with_suffix(".pdf.tmp").exists()
    validate_pdf(first, ("final", "a" * 40, "b" * 40, "Startup"))


def test_pdf_paginates_large_run_table_and_contains_first_and_last_attempts(tmp_path: Path) -> None:
    runs = tuple(
        replace(
            _run("lab2" if index % 2 == 0 else "lab3", 100.0 + index),
            seed=index,
            artifact_path=f"final/attempt-{index:03d}/success",
        )
        for index in range(40)
    )
    normalized, manifest, audit, plots = _inputs(tmp_path, runs)
    report = write_pdf_report(
        normalized["raw_runs"],
        normalized["paired_summary"],
        normalized["failures"],
        plots,
        tmp_path / "large.pdf",
        manifest=manifest,
        audit=audit,
    )

    info = subprocess.run(["pdfinfo", str(report)], check=True, text=True, capture_output=True).stdout
    page_match = re.search(r"^Pages:\s+([1-9][0-9]*)$", info, re.MULTILINE)
    assert page_match is not None
    assert int(page_match.group(1)) > 1
    text = subprocess.run(["pdftotext", str(report), "-"], check=True, text=True, capture_output=True).stdout
    assert "attempt-000" in text
    assert "attempt-039" in text
