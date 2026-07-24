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

from tools.benchmark_comparison.manifest import read_manifest
from tools.benchmark_comparison.report_cli import main

_ROOT = Path("/home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/91631f3328")


@pytest.mark.benchmark
@pytest.mark.parametrize(("run_set", "successes", "raw_files"), [("canary", 36, 399), ("final", 108, 1192)])
def test_measured_report_manifest_and_hashes_are_self_consistent(
    run_set: str,
    successes: int,
    raw_files: int,
) -> None:
    if not _ROOT.is_dir():
        pytest.skip("measured comparison artifact root is not present")
    manifest = read_manifest(_ROOT / run_set / "manifest.json")
    report = _ROOT / run_set / "report"

    assert manifest.run_set.value == run_set
    assert manifest.phase == "measured"
    with (report / "raw_runs.csv").open(newline="", encoding="utf-8") as file:
        assert len(tuple(csv.DictReader(file))) == successes
    with (report / "failures.csv").open(newline="", encoding="utf-8") as file:
        assert tuple(csv.DictReader(file)) == ()
    assert len((report / "raw_artifact_hashes.sha256").read_text(encoding="utf-8").splitlines()) == raw_files
    _verify_hash_manifest(report / "generated_hashes.sha256", report)


@pytest.mark.benchmark
def test_retained_schema_one_final_report_regenerates_without_manufactured_failures(tmp_path: Path) -> None:
    if not _ROOT.is_dir():
        pytest.skip("measured comparison artifact root is not present")
    output = tmp_path / "regenerated-final"

    assert (
        main(
            [
                "--artifact_root",
                str(_ROOT),
                "--run_set",
                "final",
                "--phase",
                "measured",
                "--output_dir",
                str(output),
            ]
        )
        == 0
    )

    audit = json.loads((output / "audit_summary.json").read_text(encoding="utf-8"))
    assert audit["successful_attempts"] == 108
    assert audit["failed_or_missing_attempts"] == 0
    svg = (output / "collection_fps.svg").read_text(encoding="utf-8")
    assert "<!-- rgb -->" not in svg
    assert "<!-- humanoid -->" not in svg
    assert "<!-- cassie -->" not in svg
    assert "<!-- rough -->" not in svg


def _verify_hash_manifest(path: Path, root: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", maxsplit=1)
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
