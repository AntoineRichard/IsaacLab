# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the immutable, typed benchmark run-set manifest."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from tools.benchmark_comparison.manifest import (
    HostIdentity,
    RunSetManifest,
    SoftwareIdentity,
    read_manifest,
    write_manifest,
)
from tools.benchmark_comparison.models import ExecutionProvenance, RunSet


def manifest(run_set: RunSet = RunSet.FINAL) -> RunSetManifest:
    """Return a complete manifest suitable for focused tests."""
    return RunSetManifest(
        schema_version="1.0",
        run_set=run_set,
        phase="measured",
        provenance=ExecutionProvenance(
            lab2_sha="a" * 40,
            lab3_sha="b" * 40,
            lab2_image_id="sha256:" + "c" * 64,
            uv_lock_sha256="d" * 64,
        ),
        host=HostIdentity(
            hostname="benchmark-host",
            os="Ubuntu 24.04",
            cpu_model="Fixture CPU",
            logical_cpu_count=32,
            gpu_model="Fixture GPU",
            gpu_driver="590.48.01",
            cuda_version="13.0",
        ),
        lab2=SoftwareIdentity(
            isaac_lab="2.3.2",
            isaac_sim="5.1.0",
            python="3.11.13",
            pytorch="2.7.0+cu128",
            rsl_rl="5.0.1",
        ),
        lab3=SoftwareIdentity(
            isaac_lab="3.0.0",
            isaac_sim="6.0.0",
            python="3.12.13",
            pytorch="2.11.0+cu128",
            rsl_rl="5.4.1",
        ),
        cpu_power_profile="powersave",
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.pop("host"), "exactly"),
        (lambda value: value.update(extra="surprise"), "exactly"),
        (lambda value: value["host"].update(logical_cpu_count=True), "logical_cpu_count"),
        (lambda value: value["lab2"].update(pytorch=""), "pytorch"),
        (lambda value: value.update(run_set="other"), "run_set"),
    ],
)
def test_manifest_rejects_missing_extra_and_invalid_fields(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    value = manifest().to_json()
    mutation(value)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        read_manifest(path)


def test_manifest_allows_identical_rewrite_and_rejects_conflict_atomically(tmp_path: Path) -> None:
    path = tmp_path / "final" / "manifest.json"
    expected = manifest()
    write_manifest(path, expected)
    original = path.read_bytes()

    assert write_manifest(path, expected).read_bytes() == original
    with pytest.raises(ValueError, match="different benchmark manifest"):
        write_manifest(path, replace(expected, phase="rerun"))
    assert path.read_bytes() == original


def test_manifest_concurrent_publication_has_one_complete_winner(tmp_path: Path) -> None:
    path = tmp_path / "final" / "manifest.json"
    candidates = (manifest(), replace(manifest(), phase="rerun"))

    def publish(candidate: RunSetManifest) -> str:
        try:
            write_manifest(path, candidate)
        except ValueError:
            return "conflict"
        return "published"

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = tuple(pool.map(publish, candidates * 4))

    winner = read_manifest(path)
    assert winner in candidates
    assert results.count("published") == 4
    assert results.count("conflict") == 4
    assert json.loads(path.read_text(encoding="utf-8")) == winner.to_json()
