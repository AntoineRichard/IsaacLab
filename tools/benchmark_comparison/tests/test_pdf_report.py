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

import pytest

from tools.benchmark_comparison import pdf_report as pdf_report_module
from tools.benchmark_comparison.manifest import HostIdentity, RunSetManifest, SoftwareIdentity
from tools.benchmark_comparison.matrix import expand_final_matrix, load_matrix
from tools.benchmark_comparison.models import ExecutionProvenance, RunSet
from tools.benchmark_comparison.normalize import FailureRow, NormalizedRun, write_normalized_outputs
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
        generated_file_count=41,
        raw_hash_manifest_sha256="e" * 64,
    )
    return normalized, _manifest(), audit, _plot_paths(tmp_path / "plots")


def _pdf_pages(path: Path) -> tuple[str, ...]:
    text = subprocess.run(["pdftotext", str(path), "-"], check=True, text=True, capture_output=True).stdout
    return tuple(page for page in text.split("\f") if page.strip())


def _page_title(page: str) -> str:
    return next(line.strip() for line in page.splitlines() if line.strip())


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
    exact_title = "Isaac Lab Startup and Runtime Benchmark Report"
    info = subprocess.run(["pdfinfo", str(first)], check=True, text=True, capture_output=True).stdout
    text = subprocess.run(["pdftotext", str(first), "-"], check=True, text=True, capture_output=True).stdout
    assert re.search(rf"^Title:\s+{re.escape(exact_title)}$", info, re.MULTILINE)
    assert exact_title in text
    validate_pdf(first, ("final", "a" * 40, "b" * 40, "Startup"))


def test_pdf_is_independent_of_and_preserves_ambient_font_settings(tmp_path: Path) -> None:
    import matplotlib

    normalized, manifest, audit, plots = _inputs(tmp_path)
    with matplotlib.rc_context():
        matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
        sans_settings = tuple(matplotlib.rcParams["font.family"])
        first = write_pdf_report(
            normalized["raw_runs"],
            normalized["paired_summary"],
            normalized["failures"],
            plots,
            tmp_path / "sans.pdf",
            manifest=manifest,
            audit=audit,
        )
        assert tuple(matplotlib.rcParams["font.family"]) == sans_settings

        matplotlib.rcParams["font.family"] = ["DejaVu Serif"]
        serif_settings = tuple(matplotlib.rcParams["font.family"])
        second = write_pdf_report(
            normalized["raw_runs"],
            normalized["paired_summary"],
            normalized["failures"],
            plots,
            tmp_path / "serif.pdf",
            manifest=manifest,
            audit=audit,
        )
        assert tuple(matplotlib.rcParams["font.family"]) == serif_settings

    assert first.read_bytes() == second.read_bytes()


def test_pdf_paginates_large_run_table_and_uses_fixed_page_order(tmp_path: Path) -> None:
    runs = tuple(
        replace(
            _run("lab2" if index % 2 == 0 else "lab3", 100.0 + index),
            seed=index,
            artifact_path=f"final/attempt-{index:03d}/success",
        )
        for index in range(40)
    )
    normalized, manifest, audit, plots = _inputs(tmp_path, runs)
    shuffled_plots = tuple(reversed(plots))
    assert tuple(path.stem for path in shuffled_plots) != PLOT_BASENAMES
    report = write_pdf_report(
        normalized["raw_runs"],
        normalized["paired_summary"],
        normalized["failures"],
        shuffled_plots,
        tmp_path / "large.pdf",
        manifest=manifest,
        audit=audit,
    )

    pages = _pdf_pages(report)
    page_titles = tuple(_page_title(page) for page in pages)
    expected_titles = (
        "Isaac Lab Startup and Runtime Benchmark Report",
        "Methodology",
        "Pinned revisions and execution identities",
        "Hardware and software inventory",
        "Task mapping",
        "runtime-100: Startup comparison",
        "runtime-1000: Startup comparison",
        "training-100: Startup comparison",
        "runtime-100: Runtime and resource comparison",
        "runtime-1000: Runtime and resource comparison",
        "training-100: Runtime and resource comparison",
        "Successful individual-run appendix (1/3)",
        "Successful individual-run appendix (2/3)",
        "Successful individual-run appendix (3/3)",
        "Failures and missing attempts",
        "Artifact integrity audit",
        "Classic: Collection FPS",
        "Classic: Mean GPU Memory",
        "Classic: Peak GPU Memory",
        "Classic: Mean GPU Utilization",
        "Classic: Total Startup Time",
        "Classic: Startup Phase Breakdown",
        "Locomotion: Collection FPS",
        "Locomotion: Mean GPU Memory",
        "Locomotion: Peak GPU Memory",
        "Locomotion: Mean GPU Utilization",
        "Locomotion: Total Startup Time",
        "Locomotion: Startup Phase Breakdown",
        "Manipulation: Collection FPS",
        "Manipulation: Mean GPU Memory",
        "Manipulation: Peak GPU Memory",
        "Manipulation: Mean GPU Utilization",
        "Manipulation: Total Startup Time",
        "Manipulation: Startup Phase Breakdown",
    )
    assert page_titles == expected_titles

    info = subprocess.run(["pdfinfo", str(report)], check=True, text=True, capture_output=True).stdout
    page_match = re.search(r"^Pages:\s+([1-9][0-9]*)$", info, re.MULTILINE)
    assert page_match is not None
    assert int(page_match.group(1)) == len(expected_titles)

    appendix_pages = tuple(page for page in pages if _page_title(page).startswith("Successful individual-run"))
    expected_appendix_page_count = (len(runs) + 18 - 1) // 18
    assert expected_appendix_page_count == 3
    assert len(appendix_pages) == expected_appendix_page_count
    assert all("Mode" in page and "Attempt identity" in page for page in appendix_pages)
    assert "attempt-000" in appendix_pages[0]
    assert "attempt-039" in appendix_pages[-1]

    empty_summary_page = pages[6]
    assert all(
        header in empty_summary_page for header in ("Task", "Metric", "Pairs", "Lab2 mean", "Lab3 mean", "Delta")
    )
    empty_failures_page = pages[14]
    assert all(header in empty_failures_page for header in ("Mode", "Classification", "Reason", "Attempt identity"))


def test_pdf_contains_failure_and_audit_content(tmp_path: Path) -> None:
    runs = (_run("lab2", 100.0), _run("lab3", 125.0))
    failure = FailureRow(
        version="lab2",
        logical_task="cartpole",
        concrete_task="Isaac-Cartpole-v0",
        mode="runtime-100",
        bound=100,
        bound_unit="steps",
        seed=43,
        num_envs=4096,
        attempt_number=1,
        failure_kind="timeout",
        reason="forced timeout",
        artifact_path="final/failed-attempt/attempt-0001-timeout",
    )
    normalized = write_normalized_outputs(tmp_path / "normalized", runs, (failure,))
    audit = ReportAudit(
        successful_attempts=2,
        failed_or_missing_attempts=1,
        raw_file_count=31,
        generated_file_count=41,
        raw_hash_manifest_sha256="f" * 64,
    )
    report = write_pdf_report(
        normalized["raw_runs"],
        normalized["paired_summary"],
        normalized["failures"],
        _plot_paths(tmp_path / "plots"),
        tmp_path / "failures.pdf",
        manifest=_manifest(),
        audit=audit,
    )

    pages = _pdf_pages(report)
    failure_page = next(page for page in pages if _page_title(page) == "Failures and missing attempts")
    assert all(token in failure_page for token in ("timeout", "forced timeout", "failed-attempt"))
    audit_page = next(page for page in pages if _page_title(page) == "Artifact integrity audit")
    assert all(
        token in audit_page
        for token in (
            "Successful attempts: 2",
            "Failed or missing attempts: 1",
            "Raw files: 31",
            "Generated files: 41",
            "f" * 64,
        )
    )


@pytest.mark.parametrize(
    "invalid_input",
    ("missing_category_plot", "duplicate_basename", "ungrouped_basename", "non_png", "invalid_image_bytes"),
)
def test_pdf_rejects_invalid_plot_inputs_atomically(tmp_path: Path, invalid_input: str) -> None:
    normalized, manifest, audit, plots = _inputs(tmp_path)
    invalid_plots = list(plots)
    if invalid_input == "missing_category_plot":
        invalid_plots.remove(next(path for path in invalid_plots if path.stem == "locomotion_collection_fps"))
    elif invalid_input == "duplicate_basename":
        duplicate = tmp_path / "duplicate" / f"{plots[0].stem}.png"
        duplicate.parent.mkdir()
        duplicate.write_bytes(plots[0].read_bytes())
        invalid_plots.append(duplicate)
    elif invalid_input == "ungrouped_basename":
        ungrouped = tmp_path / "collection_fps.png"
        ungrouped.write_bytes(plots[0].read_bytes())
        invalid_plots[0] = ungrouped
    elif invalid_input == "non_png":
        non_png = tmp_path / f"{plots[0].stem}.jpg"
        non_png.write_bytes(plots[0].read_bytes())
        invalid_plots[0] = non_png
    else:
        invalid_plots[0].write_bytes(b"not a PNG image\n")

    destination = tmp_path / "report.pdf"
    destination.write_bytes(b"preserved report")

    with pytest.raises(ValueError):
        write_pdf_report(
            normalized["raw_runs"],
            normalized["paired_summary"],
            normalized["failures"],
            tuple(invalid_plots),
            destination,
            manifest=manifest,
            audit=audit,
        )

    assert destination.read_bytes() == b"preserved report"
    assert not destination.with_suffix(".pdf.tmp").exists()


def test_pdf_validation_failure_preserves_existing_destination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    normalized, manifest, audit, plots = _inputs(tmp_path)
    destination = tmp_path / "report.pdf"
    destination.write_bytes(b"preserved report")

    def reject_validation(_path: Path, _expected_tokens: object) -> None:
        raise ValueError("forced validation failure")

    monkeypatch.setattr(pdf_report_module, "validate_pdf", reject_validation)
    with pytest.raises(ValueError, match="forced validation failure"):
        write_pdf_report(
            normalized["raw_runs"],
            normalized["paired_summary"],
            normalized["failures"],
            plots,
            destination,
            manifest=manifest,
            audit=audit,
        )

    assert destination.read_bytes() == b"preserved report"
    assert not destination.with_suffix(".pdf.tmp").exists()


@pytest.mark.parametrize("corruption", ("metadata", "text"))
def test_pdf_title_validation_failure_preserves_existing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    normalized, manifest, audit, plots = _inputs(tmp_path)
    destination = tmp_path / "report.pdf"
    destination.write_bytes(b"preserved report")
    exact_title = "Isaac Lab Startup and Runtime Benchmark Report"
    original_text_page = pdf_report_module._text_page

    with monkeypatch.context() as patch:
        if corruption == "metadata":
            patch.setitem(pdf_report_module._PDF_METADATA, "Title", "Incorrect benchmark report")
        else:

            def write_incorrect_cover_title(pdf, title: str, lines) -> None:
                rendered_title = "Isaac Lab Startup and Runtime Benchmark Results" if title == exact_title else title
                original_text_page(pdf, rendered_title, lines)

            patch.setattr(pdf_report_module, "_text_page", write_incorrect_cover_title)

        with pytest.raises(ValueError, match=r"report PDF .*title is invalid"):
            write_pdf_report(
                normalized["raw_runs"],
                normalized["paired_summary"],
                normalized["failures"],
                plots,
                destination,
                manifest=manifest,
                audit=audit,
            )

    assert destination.read_bytes() == b"preserved report"
    assert not destination.with_suffix(".pdf.tmp").exists()
    assert pdf_report_module._PDF_METADATA["Title"] == exact_title
    assert pdf_report_module._text_page is original_text_page
