# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end tests for the simulator-free report-only command."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from tools.benchmark_comparison.manifest import (
    HostIdentity,
    RunSetManifest,
    SoftwareIdentity,
    write_manifest,
)
from tools.benchmark_comparison.matrix import expand_canary_matrix, load_matrix
from tools.benchmark_comparison.models import ExecutionProvenance, RunSet
from tools.benchmark_comparison.plot import PLOT_BASENAMES
from tools.benchmark_comparison.report_cli import main


def _manifest() -> RunSetManifest:
    return RunSetManifest(
        schema_version="2.0",
        run_set=RunSet.CANARY,
        phase="measured",
        provenance=ExecutionProvenance("a" * 40, "b" * 40, "sha256:" + "c" * 64, "d" * 64),
        host=HostIdentity(
            "host",
            "Ubuntu",
            "cpu",
            32,
            "gpu",
            "590.48.01",
            "13.0",
            gpu_index=0,
            gpu_uuid="GPU-01234567-89ab-cdef-0123-456789abcdef",
        ),
        lab2=SoftwareIdentity("2.3.2", "5.1", "3.11", "2.7", "5.0"),
        lab3=SoftwareIdentity("3.0.0", "6.0", "3.12", "2.8", "5.4"),
        expansion=expand_canary_matrix(load_matrix()),
    )


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_report_only_cli_is_deterministic_self_contained_and_simulator_free(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    output = artifact_root / "canary" / "report"
    write_manifest(artifact_root / "canary" / "manifest.json", _manifest())
    argv = [
        "--artifact_root",
        str(artifact_root),
        "--run_set",
        "canary",
        "--phase",
        "measured",
        "--output_dir",
        str(output),
    ]
    simulator_modules_before = {
        name for name in sys.modules if name == "isaacsim" or name.startswith(("isaacsim.", "omni.", "docker"))
    }

    assert main(argv) == 0
    first = _snapshot(output)
    assert main(argv) == 0
    second = _snapshot(output)

    assert first == second
    assert len(PLOT_BASENAMES) == 26
    assert {
        "raw_runs.csv",
        "paired_summary.csv",
        "failures.csv",
        "report.md",
        "raw_artifact_hashes.sha256",
        "generated_hashes.sha256",
        "audit_summary.json",
    }.issubset(first)
    assert len([name for name in first if name.endswith(".png")]) == len(PLOT_BASENAMES)
    assert len([name for name in first if name.endswith(".svg")]) == len(PLOT_BASENAMES)
    assert simulator_modules_before == {
        name for name in sys.modules if name == "isaacsim" or name.startswith(("isaacsim.", "omni.", "docker"))
    }
    audit = json.loads(first["audit_summary.json"])
    assert audit["successful_attempts"] == 0
    assert audit["failed_or_missing_attempts"] == 136
    assert audit["raw_file_count"] == 0
    assert hashlib.sha256(first["generated_hashes.sha256"]).hexdigest() == audit["generated_hash_manifest_sha256"]


def test_report_only_cli_refuses_ambiguous_schema_one_artifact_identity(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    output = artifact_root / "canary" / "report"
    write_manifest(
        artifact_root / "canary" / "manifest.json",
        replace(
            _manifest(),
            schema_version="1.0",
            expansion=None,
            host=replace(_manifest().host, gpu_index=None, gpu_uuid=None),
        ),
    )

    with pytest.raises(ValueError, match="schema 1.0.*ambiguous"):
        main(
            [
                "--artifact_root",
                str(artifact_root),
                "--run_set",
                "canary",
                "--phase",
                "measured",
                "--output_dir",
                str(output),
            ]
        )
