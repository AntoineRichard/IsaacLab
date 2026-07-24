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


def test_report_cli_rejects_raw_changes_during_normalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifacts"
    write_manifest(root / "canary" / "manifest.json", _manifest())

    def normalize_then_mutate(*_args):
        (root / "canary" / "late-raw-file").write_text("changed", encoding="utf-8")
        return (), ()

    monkeypatch.setattr("tools.benchmark_comparison.report_cli.normalize_run_set", normalize_then_mutate)
    monkeypatch.setattr("tools.benchmark_comparison.report_cli.generate_plots", lambda *_args, **_kwargs: ())

    with pytest.raises(ValueError, match="changed during normalization"):
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
