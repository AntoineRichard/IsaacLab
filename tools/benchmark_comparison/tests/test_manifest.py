# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the immutable, typed benchmark run-set manifest."""

from __future__ import annotations

import json
import os
import signal
import stat
import time
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
from tools.benchmark_comparison.matrix import expand_final_matrix, load_matrix
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


def _inventory(root: Path) -> dict[str, tuple[int, int, int, int, bytes | None]]:
    """Capture source bytes and metadata without following symlinks."""
    inventory = {}
    for path in (root, *sorted(root.rglob("*"))):
        metadata = path.lstat()
        contents = path.read_bytes() if stat.S_ISREG(metadata.st_mode) else None
        relative = path.relative_to(root).as_posix() or "."
        inventory[relative] = (
            metadata.st_mode,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ino,
            contents,
        )
    return inventory


def _fail_if_call_blocks(call) -> None:
    """Require a regular-file rejection before a short process-local deadline."""

    def deadline_expired(_signal_number, _frame) -> None:
        raise TimeoutError("special-file validation blocked")

    started = time.monotonic()
    previous_handler = signal.signal(signal.SIGALRM, deadline_expired)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.5)
    try:
        with pytest.raises(ValueError, match="regular file"):
            call()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        previous_delay, previous_interval = previous_timer
        if previous_delay > 0:
            previous_delay = max(previous_delay - (time.monotonic() - started), 1e-6)
        signal.setitimer(signal.ITIMER_REAL, previous_delay, previous_interval)
    assert time.monotonic() - started < 0.5


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


@pytest.mark.parametrize("leaf", ("lock", "manifest"))
@pytest.mark.parametrize("target_exists", (True, False))
def test_manifest_publication_rejects_symlink_leaf_without_touching_source(
    tmp_path: Path,
    leaf: str,
    target_exists: bool,
) -> None:
    canonical_path = write_manifest(tmp_path / "canonical" / "manifest.json", manifest())
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_file = source_root / "immutable.json"
    if target_exists:
        source_file.write_bytes(canonical_path.read_bytes())
    source_before = _inventory(source_root)
    destination_path = tmp_path / "destination" / "manifest.json"
    destination_path.parent.mkdir()
    leaf_path = destination_path.with_suffix(".json.lock") if leaf == "lock" else destination_path
    leaf_path.symlink_to(source_file)

    with pytest.raises(ValueError, match="regular file|symlink"):
        write_manifest(destination_path, manifest())

    assert leaf_path.is_symlink()
    assert _inventory(source_root) == source_before


@pytest.mark.parametrize("leaf", ("lock", "manifest"))
def test_manifest_publication_rejects_fifo_leaf_without_blocking(tmp_path: Path, leaf: str) -> None:
    destination_path = tmp_path / "destination" / "manifest.json"
    destination_path.parent.mkdir()
    leaf_path = destination_path.with_suffix(".json.lock") if leaf == "lock" else destination_path
    os.mkfifo(leaf_path)

    _fail_if_call_blocks(lambda: write_manifest(destination_path, manifest()))


def test_fifo_deadline_helper_restores_preexisting_alarm_timer() -> None:
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    try:
        signal.setitimer(signal.ITIMER_REAL, 10.0, 2.0)

        _fail_if_call_blocks(lambda: (_ for _ in ()).throw(ValueError("regular file")))

        delay, interval = signal.getitimer(signal.ITIMER_REAL)
        assert 9.0 < delay <= 10.0
        assert interval == 2.0
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)


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


def test_schema_two_manifest_round_trips_exact_expansion_and_selected_gpu(tmp_path: Path) -> None:
    expansion = expand_final_matrix(load_matrix())
    expected = replace(
        manifest(),
        schema_version="2.0",
        host=replace(manifest().host, gpu_index=0, gpu_uuid="GPU-01234567-89ab-cdef-0123-456789abcdef"),
        expansion=expansion,
    )

    path = write_manifest(tmp_path / "final" / "manifest.json", expected)
    actual = read_manifest(path)

    assert actual == expected
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["host"]["gpu_index"] == 0
    assert document["host"]["gpu_uuid"] == "GPU-01234567-89ab-cdef-0123-456789abcdef"
    assert len(document["run_set_identity"]["attempts"]) == 408
    assert len(document["run_set_identity"]["sha256"]) == 64


def test_schema_two_manifest_rejects_tampered_attempt_snapshot(tmp_path: Path) -> None:
    expected = replace(
        manifest(),
        schema_version="2.0",
        host=replace(manifest().host, gpu_index=0, gpu_uuid="GPU-TEST-0000"),
        expansion=expand_final_matrix(load_matrix()),
    )
    document = expected.to_json()
    document["run_set_identity"]["attempts"][0]["logical_task"] = "tampered"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="sha256 does not match"):
        read_manifest(path)
