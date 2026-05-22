# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :func:`_bucket_and_chunk` (sort + chunk + fixed exec_timeout)."""

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
    assert _bucket_and_chunk([], chunk_size=25, exec_timeout_s=43200) == []


def test_single_row_one_bucket_uses_fixed_exec_timeout():
    rows = [_row(per_task_timeout_s=600)]
    [(idx, exec_timeout_s, chunk)] = _bucket_and_chunk(rows, chunk_size=25, exec_timeout_s=43200)
    assert idx == 0
    assert exec_timeout_s == 43200
    assert chunk == rows


def test_rows_sorted_ascending_by_per_task_timeout():
    """Sort is preserved for determinism even though it no longer affects timeout."""
    rows = [
        _row(per_task_timeout_s=7200, seed=1),
        _row(per_task_timeout_s=300, seed=2),
        _row(per_task_timeout_s=1800, seed=3),
    ]
    [(idx, exec_timeout_s, chunk)] = _bucket_and_chunk(rows, chunk_size=25, exec_timeout_s=43200)
    assert idx == 0
    assert exec_timeout_s == 43200
    assert [r.per_task_timeout_s for r in chunk] == [300, 1800, 7200]


def test_chunk_size_splits_rows_and_all_use_fixed_exec_timeout():
    """50 rows, chunk_size 25 → two chunks; both share the same fixed exec_timeout."""
    rows = [_row(per_task_timeout_s=i * 60, seed=i) for i in range(50)]
    buckets = _bucket_and_chunk(rows, chunk_size=25, exec_timeout_s=43200)
    assert len(buckets) == 2
    (idx0, t0, chunk0), (idx1, t1, chunk1) = buckets
    assert idx0 == 0 and idx1 == 1
    assert len(chunk0) == 25 and len(chunk1) == 25
    assert t0 == 43200
    assert t1 == 43200


def test_fifty_one_rows_overflows_into_three_chunks():
    rows = [_row(per_task_timeout_s=i + 1, seed=i) for i in range(51)]
    buckets = _bucket_and_chunk(rows, chunk_size=25, exec_timeout_s=43200)
    assert [(idx, len(chunk)) for idx, _t, chunk in buckets] == [(0, 25), (1, 25), (2, 1)]


def test_ties_break_by_task_id_then_backend_then_seed():
    """Deterministic sort so reruns and resume both see the same layout."""
    rows = [
        _row(per_task_timeout_s=1000, task_id="Isaac-B", backend="newton", seed=43),
        _row(per_task_timeout_s=1000, task_id="Isaac-A", backend="physx", seed=42),
        _row(per_task_timeout_s=1000, task_id="Isaac-A", backend="physx", seed=41),
        _row(per_task_timeout_s=1000, task_id="Isaac-A", backend="newton", seed=42),
        _row(per_task_timeout_s=1000, task_id="Isaac-B", backend="newton", seed=42),
    ]
    [(idx, exec_timeout_s, chunk)] = _bucket_and_chunk(rows, chunk_size=25, exec_timeout_s=43200)
    assert idx == 0
    assert exec_timeout_s == 43200
    keys = [(r.task_id, r.backend, r.seed) for r in chunk]
    assert keys == [
        ("Isaac-A", "newton", 42),
        ("Isaac-A", "physx", 41),
        ("Isaac-A", "physx", 42),
        ("Isaac-B", "newton", 42),
        ("Isaac-B", "newton", 43),
    ]


def test_exec_timeout_passes_through_unchanged_across_chunks():
    """Every chunk reports the exact exec_timeout_s the caller supplied."""
    rows = [
        _row(per_task_timeout_s=300, seed=1),
        _row(per_task_timeout_s=600, seed=2),
        _row(per_task_timeout_s=2100, seed=3),
        _row(per_task_timeout_s=86400, seed=4),
    ]
    buckets = _bucket_and_chunk(rows, chunk_size=2, exec_timeout_s=12345)
    assert len(buckets) == 2
    assert all(t == 12345 for _, t, _ in buckets)
