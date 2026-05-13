# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :func:`_bucket_and_chunk` (spec §5.2)."""

from __future__ import annotations

from tools.odin.bifrost.cli import _bucket_and_chunk, _PlannedRow


def _row(
    *,
    timeout_class: str,
    task_id: str = "Isaac-Ant-Direct-v0",
    backend: str = "physx",
    seed: int = 42,
    run_id: str | None = None,
) -> _PlannedRow:
    return _PlannedRow(
        run_id=run_id or f"{backend}_{task_id}_seed{seed}",
        task_id=task_id,
        framework="rsl-rl",
        backend=backend,
        seed=seed,
        num_envs=4096,
        max_iterations=500,
        timeout_class=timeout_class,
    )


def test_empty_rows_returns_empty_list():
    assert _bucket_and_chunk([], chunk_size=25) == []


def test_single_class_three_rows_one_bucket():
    rows = [_row(timeout_class="short", seed=s) for s in (42, 43, 44)]
    buckets = _bucket_and_chunk(rows, chunk_size=25)
    assert len(buckets) == 1
    cls, idx, chunk = buckets[0]
    assert cls == "short"
    assert idx == 0
    assert [r.seed for r in chunk] == [42, 43, 44]


def test_two_classes_two_buckets_sorted_by_class_name():
    rows = [
        _row(timeout_class="medium", seed=99),
        _row(timeout_class="short", seed=42),
        _row(timeout_class="medium", seed=100),
    ]
    buckets = _bucket_and_chunk(rows, chunk_size=25)
    assert [(cls, idx) for cls, idx, _ in buckets] == [("medium", 0), ("short", 0)]
    medium = next(chunk for cls, _, chunk in buckets if cls == "medium")
    short = next(chunk for cls, _, chunk in buckets if cls == "short")
    assert [r.seed for r in medium] == [99, 100]
    assert [r.seed for r in short] == [42]


def test_fifty_rows_same_class_chunked_at_25():
    rows = [_row(timeout_class="short", seed=i) for i in range(50)]
    buckets = _bucket_and_chunk(rows, chunk_size=25)
    assert len(buckets) == 2
    assert [(cls, idx, len(chunk)) for cls, idx, chunk in buckets] == [
        ("short", 0, 25),
        ("short", 1, 25),
    ]


def test_fiftyone_rows_same_class_overflows_into_three_chunks():
    rows = [_row(timeout_class="short", seed=i) for i in range(51)]
    buckets = _bucket_and_chunk(rows, chunk_size=25)
    assert [(cls, idx, len(chunk)) for cls, idx, chunk in buckets] == [
        ("short", 0, 25),
        ("short", 1, 25),
        ("short", 2, 1),
    ]


def test_rows_within_chunk_sorted_deterministically():
    """Rows are sorted by (task_id, backend, seed) so reruns are stable.

    Submitted in a deliberately scrambled order; expect lexicographic
    output within the bucket.
    """
    rows = [
        _row(timeout_class="medium", task_id="Isaac-B", backend="newton", seed=43),
        _row(timeout_class="medium", task_id="Isaac-A", backend="physx", seed=42),
        _row(timeout_class="medium", task_id="Isaac-A", backend="physx", seed=41),
        _row(timeout_class="medium", task_id="Isaac-A", backend="newton", seed=42),
        _row(timeout_class="medium", task_id="Isaac-B", backend="newton", seed=42),
    ]
    [(cls, idx, chunk)] = _bucket_and_chunk(rows, chunk_size=25)
    assert cls == "medium" and idx == 0
    keys = [(r.task_id, r.backend, r.seed) for r in chunk]
    assert keys == [
        ("Isaac-A", "newton", 42),
        ("Isaac-A", "physx", 41),
        ("Isaac-A", "physx", 42),
        ("Isaac-B", "newton", 42),
        ("Isaac-B", "newton", 43),
    ]
