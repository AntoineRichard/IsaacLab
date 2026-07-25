# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read-only smoke checks for the retained measured benchmark reports."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from tools.benchmark_comparison.manifest import read_manifest, resolve_manifest_expansion

_ROOT = Path("/home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/c8d672a1dd")


@pytest.mark.benchmark
def test_retained_final_report_manifest_and_hashes_are_self_consistent() -> None:
    if not _ROOT.is_dir():
        pytest.skip("measured comparison artifact root is not present")
    manifest = read_manifest(_ROOT / "final" / "manifest.json")
    expansion = resolve_manifest_expansion(manifest, _ROOT)
    report = _ROOT / "final" / "report"

    assert manifest.run_set.value == "final"
    assert manifest.phase == "measured"
    assert len(expansion.attempts) == 228
    with (report / "raw_runs.csv").open(newline="", encoding="utf-8") as file:
        rows = csv.DictReader(file)
        assert {
            "startup_total_s",
            "startup_app_launch_s",
            "startup_python_imports_s",
            "startup_task_config_s",
            "startup_env_creation_s",
            "startup_first_step_s",
        } <= set(rows.fieldnames or ())
        assert len(tuple(rows)) == len(expansion.attempts)
    with (report / "failures.csv").open(newline="", encoding="utf-8") as file:
        assert tuple(csv.DictReader(file)) == ()
    audit = json.loads((report / "audit_summary.json").read_text(encoding="utf-8"))
    assert audit["successful_attempts"] == len(expansion.attempts)
    assert audit["failed_or_missing_attempts"] == 0
    assert audit["generated_file_count"] == 17
    assert (report / "report.pdf").is_file()
    _verify_hash_manifest(report / "generated_hashes.sha256", report)
    _verify_hash_manifest(report / "raw_artifact_hashes.sha256", _ROOT)


def _verify_hash_manifest(path: Path, root: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", maxsplit=1)
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
