# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :func:`_bucket_and_chunk` (sort + chunk + max-of-bucket)."""

from __future__ import annotations

from tools.odin.bifrost.cli import _bucket_and_chunk, _PlannedRow


def _row(
    *,
    per_task_timeout_s: int,
    task_id: str = "Isaac-Ant-Direct-v0",
    backend: str = "physx",
    seed: int = 42,
    run_id: str | None = None,
) -> _PlannedRow:
    return _PlannedRow(
        run_id=run_id or f"{backend}_{task_id}_seed{seed}_{per_task_timeout_s}",
        task_id=task_id,
        framework="rsl-rl",
        backend=backend,
        seed=seed,
        num_envs=4096,
        max_iterations=500,
        per_task_timeout_s=per_task_timeout_s,
    )


def test_empty_rows_returns_empty_list():
    assert _bucket_and_chunk([], chunk_size=25) == []


def test_single_row_one_bucket_uses_row_timeout_as_max():
    rows = [_row(per_task_timeout_s=600)]
    [(idx, max_s, chunk)] = _bucket_and_chunk(rows, chunk_size=25)
    assert idx == 0
    assert max_s == 600
    assert chunk == rows


def test_rows_sorted_ascending_by_per_task_timeout():
    """Mixed timeouts → ascending sort so the first chunk's max is small."""
    rows = [
        _row(per_task_timeout_s=7200, seed=1),
        _row(per_task_timeout_s=300, seed=2),
        _row(per_task_timeout_s=1800, seed=3),
    ]
    [(idx, max_s, chunk)] = _bucket_and_chunk(rows, chunk_size=25)
    assert idx == 0
    assert max_s == 7200  # max of the chunk
    assert [r.per_task_timeout_s for r in chunk] == [300, 1800, 7200]


def test_chunk_size_splits_rows_and_each_bucket_max_is_its_own():
    """50 rows, chunk_size 25 → two chunks; each chunk's max is its own.

    With ascending sort, the first chunk holds the 25 smallest timeouts;
    the second holds the 25 largest. Asserting that the second chunk's
    max > the first chunk's max validates the sort.
    """
    rows = [_row(per_task_timeout_s=i * 60, seed=i) for i in range(50)]
    buckets = _bucket_and_chunk(rows, chunk_size=25)
    assert len(buckets) == 2
    (idx0, max0, chunk0), (idx1, max1, chunk1) = buckets
    assert idx0 == 0 and idx1 == 1
    assert len(chunk0) == 25 and len(chunk1) == 25
    # First chunk has 0..24, second has 25..49.
    assert max0 == 24 * 60
    assert max1 == 49 * 60


def test_fifty_one_rows_overflows_into_three_chunks():
    rows = [_row(per_task_timeout_s=i + 1, seed=i) for i in range(51)]
    buckets = _bucket_and_chunk(rows, chunk_size=25)
    assert [(idx, len(chunk)) for idx, _max, chunk in buckets] == [(0, 25), (1, 25), (2, 1)]
    # The trailing chunk's only row is the largest timeout (51).
    assert buckets[-1][1] == 51


def test_ties_break_by_task_id_then_backend_then_seed():
    """Rows with equal timeout are sorted deterministically.

    Submitted in scrambled order; expect ``(per_task_timeout_s, task_id,
    backend, seed)`` ordering so reruns and resume both see the same
    layout.
    """
    rows = [
        _row(per_task_timeout_s=1000, task_id="Isaac-B", backend="newton", seed=43),
        _row(per_task_timeout_s=1000, task_id="Isaac-A", backend="physx", seed=42),
        _row(per_task_timeout_s=1000, task_id="Isaac-A", backend="physx", seed=41),
        _row(per_task_timeout_s=1000, task_id="Isaac-A", backend="newton", seed=42),
        _row(per_task_timeout_s=1000, task_id="Isaac-B", backend="newton", seed=42),
    ]
    [(idx, max_s, chunk)] = _bucket_and_chunk(rows, chunk_size=25)
    assert idx == 0
    assert max_s == 1000
    keys = [(r.task_id, r.backend, r.seed) for r in chunk]
    assert keys == [
        ("Isaac-A", "newton", 42),
        ("Isaac-A", "physx", 41),
        ("Isaac-A", "physx", 42),
        ("Isaac-B", "newton", 42),
        ("Isaac-B", "newton", 43),
    ]


def test_chunk_max_is_row_max_not_global_max():
    """Two chunks: first chunk's max < second chunk's max (sort guarantees it).

    Exercises the "max-of-chunk" rule with realistic mixed timeouts.
    """
    rows = [
        _row(per_task_timeout_s=300, seed=1),
        _row(per_task_timeout_s=600, seed=2),
        _row(per_task_timeout_s=2100, seed=3),
        _row(per_task_timeout_s=86400, seed=4),
    ]
    buckets = _bucket_and_chunk(rows, chunk_size=2)
    assert len(buckets) == 2
    (_, max0, _), (_, max1, _) = buckets
    # First chunk: 300, 600 → max 600. Second chunk: 2100, 86400 → max 86400.
    assert max0 == 600
    assert max1 == 86400
