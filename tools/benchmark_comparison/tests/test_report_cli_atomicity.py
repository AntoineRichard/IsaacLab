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
from tools.benchmark_comparison.report_cli import _publish, main

_PLOT_CATEGORIES = ("classic", "locomotion_flat", "locomotion_rough", "manipulation")
_PLOT_METRICS = (
    "collection_fps",
    "gpu_memory_mean_mib",
    "gpu_memory_peak_mib",
    "gpu_utilization_mean_pct",
    "startup_total_s",
    "startup_phase_breakdown",
)


def _snapshot_files(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _write_previous_53_file_report(output: Path) -> None:
    generated_paths = {
        "raw_runs.csv",
        "paired_summary.csv",
        "failures.csv",
        "report.md",
        "report.pdf",
    }
    generated_paths.update(
        f"{category}_{metric}.{suffix}"
        for category in _PLOT_CATEGORIES
        for metric in _PLOT_METRICS
        for suffix in ("png", "svg")
    )
    assert len(generated_paths) == 53
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
    monkeypatch.setattr("tools.benchmark_comparison.report_cli.generate_plots", lambda *_args, **_kwargs: ())
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


@pytest.mark.parametrize("failure_boundary", ("grouped_plots", "pdf"))
def test_report_cli_preserves_previous_53_file_report_when_generation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_boundary: str,
) -> None:
    root = tmp_path / "artifacts"
    write_manifest(root / "canary" / "manifest.json", _manifest())
    raw_artifacts = root / "canary" / "raw-artifact"
    (raw_artifacts / "nested").mkdir(parents=True)
    (raw_artifacts / "measurements.json").write_bytes(b'{"fps": 123.0}\n')
    (raw_artifacts / "nested" / "stdout.log").write_bytes(b"raw log bytes\n")
    output = root / "canary" / "report"
    _write_previous_53_file_report(output)
    published_before = _snapshot_files(output)
    assert len(published_before) == 53
    raw_before = _snapshot_files(raw_artifacts)

    if failure_boundary == "grouped_plots":

        def fail_plots(_raw_runs: Path, staging: Path, **_kwargs):
            (staging / "classic_collection_fps.png").write_bytes(b"partial grouped plot")
            raise RuntimeError("injected grouped plots failure")

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
