# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for report input/output transaction boundaries."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.benchmark_comparison.manifest import HostIdentity, RunSetManifest, SoftwareIdentity, write_manifest
from tools.benchmark_comparison.matrix import expand_canary_matrix, load_matrix
from tools.benchmark_comparison.models import ExecutionProvenance, RunSet
from tools.benchmark_comparison.plot import PLOT_BASENAMES
from tools.benchmark_comparison.report_cli import _publish, main

_PREVIOUS_41_PLOT_BASENAMES = (
    "classic_collection_fps",
    "classic_gpu_memory_mean_mib",
    "classic_gpu_memory_peak_mib",
    "classic_gpu_utilization_mean_pct",
    "classic_startup_total_s",
    "classic_startup_phase_breakdown",
    "locomotion_collection_fps",
    "locomotion_gpu_memory_mean_mib",
    "locomotion_gpu_memory_peak_mib",
    "locomotion_gpu_utilization_mean_pct",
    "locomotion_startup_total_s",
    "locomotion_startup_phase_breakdown",
    "manipulation_collection_fps",
    "manipulation_gpu_memory_mean_mib",
    "manipulation_gpu_memory_peak_mib",
    "manipulation_gpu_utilization_mean_pct",
    "manipulation_startup_total_s",
    "manipulation_startup_phase_breakdown",
)


def _snapshot_files(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _write_placeholder_plots(staging: Path) -> tuple[Path, ...]:
    staging.mkdir(parents=True, exist_ok=True)
    plots = []
    for basename in PLOT_BASENAMES:
        for suffix in ("png", "svg"):
            path = staging / f"{basename}.{suffix}"
            path.write_bytes(f"placeholder {basename}.{suffix}\n".encode())
            plots.append(path)
    return tuple(plots)


def _write_previous_41_file_report(output: Path) -> None:
    generated_paths = {
        "raw_runs.csv",
        "paired_summary.csv",
        "failures.csv",
        "report.md",
        "report.pdf",
        *(f"{basename}.{suffix}" for basename in _PREVIOUS_41_PLOT_BASENAMES for suffix in ("png", "svg")),
    }
    assert len(generated_paths) == 41
    output.mkdir(parents=True)
    for relative_path in sorted(generated_paths):
        (output / relative_path).write_bytes(f"previous {relative_path}\n".encode())


def _write_previous_57_file_report(output: Path) -> None:
    generated_paths = {
        "raw_runs.csv",
        "paired_summary.csv",
        "failures.csv",
        "report.md",
        "report.pdf",
        *(f"{basename}.{suffix}" for basename in PLOT_BASENAMES for suffix in ("png", "svg")),
    }
    assert len(generated_paths) == 57
    output.mkdir(parents=True)
    for relative_path in sorted(generated_paths):
        (output / relative_path).write_bytes(f"previous {relative_path}\n".encode())


def _manifest() -> RunSetManifest:
    software = SoftwareIdentity("2.3.2", "5.1", "3.11", "2.7", "5.0")
    return RunSetManifest(
        "2.0",
        RunSet.CANARY,
        "measured",
        ExecutionProvenance("a" * 40, "b" * 40, "sha256:" + "c" * 64, "d" * 64),
        HostIdentity(
            "host",
            "os",
            "cpu",
            32,
            "gpu",
            "driver",
            "cuda",
            gpu_index=0,
            gpu_uuid="GPU-TEST-0000",
        ),
        software,
        software,
        expansion=expand_canary_matrix(load_matrix()),
    )


def test_report_cli_rejects_raw_changes_during_report_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifacts"
    write_manifest(root / "canary" / "manifest.json", _manifest())

    def normalize_then_mutate(*_args):
        (root / "canary" / "late-raw-file").write_text("changed", encoding="utf-8")
        return (), ()

    def write_placeholder_pdf(*args, **_kwargs):
        output_path = args[4]
        output_path.write_bytes(b"placeholder PDF")
        return output_path

    monkeypatch.setattr("tools.benchmark_comparison.report_cli.normalize_run_set", normalize_then_mutate)
    monkeypatch.setattr(
        "tools.benchmark_comparison.report_cli.generate_plots",
        lambda _raw_runs, _aggregate_deltas, staging, **_kwargs: _write_placeholder_plots(staging),
    )
    monkeypatch.setattr("tools.benchmark_comparison.report_cli.write_pdf_report", write_placeholder_pdf)

    with pytest.raises(ValueError, match="changed during report generation"):
        main(
            [
                "--artifact_root",
                str(root),
                "--run_set",
                "canary",
                "--phase",
                "measured",
                "--output_dir",
                str(root / "canary" / "report"),
            ]
        )


def test_report_directory_publication_rolls_back_and_removes_stale_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "report"
    output.mkdir()
    (output / "report.md").write_text("old", encoding="utf-8")
    (output / "stale.txt").write_text("stale", encoding="utf-8")
    staging = tmp_path / ".report.staging"
    staging.mkdir()
    (staging / "report.md").write_text("new", encoding="utf-8")
    real_replace = os.replace

    def fail_new_generation(source, destination):
        if Path(source) == staging and Path(destination) == output:
            raise OSError("injected publication failure")
        real_replace(source, destination)

    monkeypatch.setattr("tools.benchmark_comparison.report_cli.os.replace", fail_new_generation)
    with pytest.raises(OSError, match="injected"):
        _publish(staging, output)
    assert (output / "report.md").read_text(encoding="utf-8") == "old"
    assert (output / "stale.txt").is_file()

    monkeypatch.setattr("tools.benchmark_comparison.report_cli.os.replace", real_replace)
    _publish(staging, output)
    assert (output / "report.md").read_text(encoding="utf-8") == "new"
    assert not (output / "stale.txt").exists()


@pytest.mark.parametrize(
    "invalid_inventory",
    (
        "out_of_order",
        "missing_svg",
        "reversed_suffix_pair",
        "duplicate_entry",
        "nested_paths",
        "external_paths",
    ),
)
def test_report_cli_rejects_noncanonical_plot_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_inventory: str,
) -> None:
    root = tmp_path / "artifacts"
    write_manifest(root / "canary" / "manifest.json", _manifest())
    raw_artifacts = root / "canary" / "raw-artifact"
    (raw_artifacts / "nested").mkdir(parents=True)
    (raw_artifacts / "measurements.json").write_bytes(b'{"fps": 123.0}\n')
    (raw_artifacts / "nested" / "stdout.log").write_bytes(b"raw log bytes\n")
    output = root / "canary" / "report"
    _write_previous_57_file_report(output)
    published_before = _snapshot_files(output)
    raw_before = _snapshot_files(raw_artifacts)
    writer_calls = []

    def generate_invalid_plots(_raw_runs: Path, _aggregate_deltas, staging: Path, **_kwargs):
        plots = _write_placeholder_plots(staging)
        if invalid_inventory == "out_of_order":
            return tuple((*plots[2:4], *plots[:2], *plots[4:]))
        if invalid_inventory == "missing_svg":
            return plots[:-1]
        if invalid_inventory == "reversed_suffix_pair":
            return tuple((plots[1], plots[0], *plots[2:]))
        if invalid_inventory == "duplicate_entry":
            return tuple((*plots[:-1], plots[-2]))
        if invalid_inventory == "nested_paths":
            return _write_placeholder_plots(staging / "nested")
        return _write_placeholder_plots(tmp_path / "outside")

    def write_placeholder_markdown(*args, **_kwargs):
        writer_calls.append("markdown")
        output_path = args[3]
        output_path.write_text("placeholder Markdown\n", encoding="utf-8")
        return output_path

    def write_placeholder_pdf(*args, **_kwargs):
        writer_calls.append("pdf")
        output_path = args[4]
        output_path.write_bytes(b"placeholder PDF")
        return output_path

    monkeypatch.setattr("tools.benchmark_comparison.report_cli.generate_plots", generate_invalid_plots)
    monkeypatch.setattr("tools.benchmark_comparison.report_cli.write_markdown_report", write_placeholder_markdown)
    monkeypatch.setattr("tools.benchmark_comparison.report_cli.write_pdf_report", write_placeholder_pdf)

    with pytest.raises(ValueError, match="canonical PNG/SVG pairs"):
        main(
            [
                "--artifact_root",
                str(root),
                "--run_set",
                "canary",
                "--phase",
                "measured",
                "--output_dir",
                str(output),
            ]
        )

    assert writer_calls == []
    assert _snapshot_files(output) == published_before
    assert _snapshot_files(raw_artifacts) == raw_before
    assert not tuple(output.parent.glob(f".{output.name}.*"))


@pytest.mark.parametrize("previous_file_count", (41, 57))
@pytest.mark.parametrize("failure_boundary", ("aggregate_plots", "detailed_plots", "pdf"))
def test_report_cli_preserves_previous_report_when_generation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    previous_file_count: int,
    failure_boundary: str,
) -> None:
    root = tmp_path / "artifacts"
    write_manifest(root / "canary" / "manifest.json", _manifest())
    raw_artifacts = root / "canary" / "raw-artifact"
    (raw_artifacts / "nested").mkdir(parents=True)
    (raw_artifacts / "measurements.json").write_bytes(b'{"fps": 123.0}\n')
    (raw_artifacts / "nested" / "stdout.log").write_bytes(b"raw log bytes\n")
    output = root / "canary" / "report"
    if previous_file_count == 41:
        _write_previous_41_file_report(output)
    else:
        _write_previous_57_file_report(output)
    published_before = _snapshot_files(output)
    assert len(published_before) == previous_file_count
    raw_before = _snapshot_files(raw_artifacts)

    if failure_boundary in {"aggregate_plots", "detailed_plots"}:

        def fail_plots(_raw_runs: Path, _aggregate_deltas, staging: Path, **_kwargs):
            partial_basenames = PLOT_BASENAMES[:2]
            if failure_boundary == "detailed_plots":
                partial_basenames = PLOT_BASENAMES[:3]
            for basename in partial_basenames:
                for suffix in ("png", "svg"):
                    (staging / f"{basename}.{suffix}").write_bytes(b"partial plot")
            raise RuntimeError(f"injected {failure_boundary} failure")

        monkeypatch.setattr("tools.benchmark_comparison.report_cli.generate_plots", fail_plots)
    else:

        def fail_pdf(*_args, **_kwargs):
            raise RuntimeError("injected PDF failure")

        monkeypatch.setattr("tools.benchmark_comparison.report_cli.write_pdf_report", fail_pdf)

    with pytest.raises(RuntimeError, match="injected .* failure"):
        main(
            [
                "--artifact_root",
                str(root),
                "--run_set",
                "canary",
                "--phase",
                "measured",
                "--output_dir",
                str(output),
            ]
        )

    assert _snapshot_files(output) == published_before
    assert _snapshot_files(raw_artifacts) == raw_before
    assert not tuple(output.parent.glob(f".{output.name}.*"))
