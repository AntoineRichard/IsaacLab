# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression test for run-set-specific report methodology."""

from __future__ import annotations

from pathlib import Path

from tools.benchmark_comparison.manifest import HostIdentity, RunSetManifest, SoftwareIdentity
from tools.benchmark_comparison.matrix import expand_canary_matrix, load_matrix
from tools.benchmark_comparison.models import ExecutionProvenance, RunSet
from tools.benchmark_comparison.normalize import write_normalized_outputs
from tools.benchmark_comparison.report import write_markdown_report


def test_canary_report_states_canary_phase_seed_and_bounds(tmp_path: Path) -> None:
    provenance = ExecutionProvenance("a" * 40, "b" * 40, "sha256:" + "c" * 64, "d" * 64)
    software = SoftwareIdentity("2.3.2", "5.1", "3.11", "2.7", "5.0")
    manifest = RunSetManifest(
        "2.0",
        RunSet.CANARY,
        "measured",
        provenance,
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
    normalized = write_normalized_outputs(tmp_path / "normalized", (), ())

    report = write_markdown_report(
        normalized["raw_runs"],
        normalized["paired_summary"],
        normalized["failures"],
        tmp_path / "report.md",
        manifest=manifest,
    )

    text = report.read_text(encoding="utf-8")
    assert "Run set: `canary`; phase: `measured`." in text
    assert "paired seed 42" in text
    assert "10 or 25 environment steps" in text
    assert "2 iterations" in text
    assert "seeds 42, 43, and 44" not in text
