# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end tests for the simulator-free report-only command."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from tools.benchmark_comparison.manifest import (
    HostIdentity,
    RunSetManifest,
    SoftwareIdentity,
    write_manifest,
)
from tools.benchmark_comparison.models import ExecutionProvenance, RunSet
from tools.benchmark_comparison.report_cli import main


def _manifest() -> RunSetManifest:
    return RunSetManifest(
        schema_version="1.0",
        run_set=RunSet.CANARY,
        phase="measured",
        provenance=ExecutionProvenance("a" * 40, "b" * 40, "sha256:" + "c" * 64, "d" * 64),
        host=HostIdentity("host", "Ubuntu", "cpu", 32, "gpu", "590.48.01", "13.0"),
        lab2=SoftwareIdentity("2.3.2", "5.1", "3.11", "2.7", "5.0"),
        lab3=SoftwareIdentity("3.0.0", "6.0", "3.12", "2.8", "5.4"),
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
    assert {
        "raw_runs.csv",
        "paired_summary.csv",
        "failures.csv",
        "report.md",
        "raw_artifact_hashes.sha256",
        "generated_hashes.sha256",
        "audit_summary.json",
    }.issubset(first)
    assert len([name for name in first if name.endswith(".png")]) == 4
    assert len([name for name in first if name.endswith(".svg")]) == 4
    assert simulator_modules_before == {
        name for name in sys.modules if name == "isaacsim" or name.startswith(("isaacsim.", "omni.", "docker"))
    }
    audit = json.loads(first["audit_summary.json"])
    assert audit["successful_attempts"] == 0
    assert audit["failed_or_missing_attempts"] == 76
    assert audit["raw_file_count"] == 0
    assert hashlib.sha256(first["generated_hashes.sha256"]).hexdigest() == audit["generated_hash_manifest_sha256"]
