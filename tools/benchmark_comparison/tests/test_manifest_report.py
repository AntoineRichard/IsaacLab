# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for manifest-backed benchmark reporting."""

from __future__ import annotations

from pathlib import Path

from tools.benchmark_comparison.manifest import (
    HostIdentity,
    RunSetManifest,
    SoftwareIdentity,
    write_manifest,
)
from tools.benchmark_comparison.models import ExecutionProvenance, RunSet
from tools.benchmark_comparison.normalize import write_normalized_outputs
from tools.benchmark_comparison.report import write_markdown_report


def _manifest() -> RunSetManifest:
    return RunSetManifest(
        schema_version="1.0",
        run_set=RunSet.FINAL,
        phase="measured",
        provenance=ExecutionProvenance("a" * 40, "b" * 40, "sha256:" + "c" * 64, "d" * 64),
        host=HostIdentity("host", "Ubuntu", "cpu", 32, "gpu", "590.48.01", "13.0"),
        lab2=SoftwareIdentity("2.3.2", "5.1.0", "3.11.13", "2.7.0", "5.0.1"),
        lab3=SoftwareIdentity("3.0.0", "6.0.0", "3.12.13", "2.11.0", "5.4.1"),
        cpu_power_profile="powersave",
    )


def test_zero_success_report_uses_typed_manifest_versions_and_driver(tmp_path: Path) -> None:
    normalized = write_normalized_outputs(tmp_path / "normalized", (), ())
    manifest_path = write_manifest(tmp_path / "final" / "manifest.json", _manifest())

    report = write_markdown_report(
        normalized["raw_runs"],
        normalized["paired_summary"],
        normalized["failures"],
        tmp_path / "report.md",
        manifest=manifest_path,
    )

    text = report.read_text(encoding="utf-8")
    for expected in (
        "590.48.01",
        "2.3.2",
        "3.0.0",
        "5.1.0",
        "6.0.0",
        "3.11.13",
        "3.12.13",
        "2.7.0",
        "2.11.0",
        "5.0.1",
        "5.4.1",
        "powersave",
    ):
        assert expected in text
