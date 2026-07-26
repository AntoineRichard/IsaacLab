# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for deterministic typed aggregate benchmark deltas."""

from __future__ import annotations

import csv
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

import tools.benchmark_comparison.aggregate as aggregate_module
from tools.benchmark_comparison.aggregate import (
    AGGREGATE_METRICS,
    AGGREGATE_STATISTICS,
    AggregateDelta,
    AggregateStatistic,
    aggregate_paired_summary,
)
from tools.benchmark_comparison.matrix import expand_final_matrix, load_matrix, task_aliases_by_category
from tools.benchmark_comparison.models import MatrixExpansion, TaskCategory
from tools.benchmark_comparison.normalize import PAIRED_SUMMARY_FIELDS

_TASKS = (
    "cartpole",
    "cartpole_rgb_kit",
    "cartpole_direct",
    "ant",
    "anymal_d_flat",
    "anymal_d_rough",
    "allegro_cube",
)


def _expansion(*tasks: str) -> MatrixExpansion:
    full = expand_final_matrix(load_matrix())
    selected_tasks = frozenset(tasks or _TASKS)
    pairs = tuple(pair for pair in full.pairs if pair.logical_task in selected_tasks)
    attempts = tuple(attempt for pair in pairs for attempt in pair.attempts)
    return replace(full, pairs=pairs, attempts=attempts)


def _row(
    task: str = "cartpole",
    mode: str = "runtime-100",
    metric: str = "collection_fps",
    *,
    lab2_mean: str = "100",
    lab3_mean: str = "110",
    percent_delta: str = "10",
    percent_delta_status: str = "available",
    **overrides: str,
) -> dict[str, str]:
    row = {
        "logical_task": task,
        "mode": mode,
        "metric": metric,
        "paired_seed_count": "3",
        "lab2_mean": lab2_mean,
        "lab2_std": "1",
        "lab3_mean": lab3_mean,
        "lab3_std": "2",
        "absolute_delta": str(float(lab3_mean) - float(lab2_mean)),
        "percent_delta": percent_delta,
        "percent_delta_status": percent_delta_status,
    }
    row.update(overrides)
    return row


def _write_summary(path: Path, rows: tuple[dict[str, str], ...], fields=PAIRED_SUMMARY_FIELDS) -> Path:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_raw_summary(path: Path, values: tuple[str, ...]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(PAIRED_SUMMARY_FIELDS)
        writer.writerow(values)
    return path


def _cell(
    cells: tuple[AggregateDelta, ...],
    statistic: AggregateStatistic,
    category: TaskCategory,
    mode: str,
    metric: str,
) -> AggregateDelta:
    return next(
        cell
        for cell in cells
        if (cell.statistic, cell.category, cell.mode, cell.metric) == (statistic, category, mode, metric)
    )


def test_aggregate_cells_use_exact_order_and_equal_task_weighting(tmp_path: Path) -> None:
    expansion = _expansion()
    path = _write_summary(
        tmp_path / "paired_summary.csv",
        (
            _row("cartpole", lab2_mean="100", lab3_mean="80", percent_delta="-20"),
            _row("cartpole_direct", lab2_mean="10", lab3_mean="11", percent_delta="10"),
            _row("ant", lab2_mean="1000", lab3_mean="1700", percent_delta="70"),
        ),
    )

    cells = aggregate_paired_summary(path, expansion)

    assert AGGREGATE_METRICS == (
        "collection_fps",
        "startup_total_s",
        "gpu_memory_mean_mib",
        "gpu_memory_peak_mib",
        "gpu_utilization_mean_pct",
    )
    assert AGGREGATE_STATISTICS == (AggregateStatistic.MEDIAN, AggregateStatistic.MEAN)
    assert tuple((cell.statistic, cell.category, cell.mode, cell.metric) for cell in cells) == tuple(
        (statistic, category, mode, metric)
        for statistic in AGGREGATE_STATISTICS
        for category in TaskCategory
        for mode in ("runtime-100", "runtime-1000", "training-100")
        for metric in AGGREGATE_METRICS
    )
    assert tuple(cell.statistic for cell in cells[:60]) == (AggregateStatistic.MEDIAN,) * 60
    assert tuple(cell.statistic for cell in cells[60:]) == (AggregateStatistic.MEAN,) * 60
    assert len(cells) == 2 * 4 * 3 * 5 == 120
    median = _cell(cells, AggregateStatistic.MEDIAN, TaskCategory.CLASSIC, "runtime-100", "collection_fps")
    mean = _cell(cells, AggregateStatistic.MEAN, TaskCategory.CLASSIC, "runtime-100", "collection_fps")
    assert median.value == 10.0
    assert mean.value == 20.0
    assert median.task_count == mean.task_count == 3


def test_aggregate_cells_respect_task_capabilities_and_category_boundaries(tmp_path: Path) -> None:
    path = _write_summary(
        tmp_path / "paired_summary.csv",
        (
            _row("cartpole_rgb_kit", "runtime-100", percent_delta="10"),
            _row("cartpole_rgb_kit", "runtime-1000", percent_delta="10"),
            _row("anymal_d_flat", percent_delta="20", lab3_mean="120"),
            _row("anymal_d_rough", percent_delta="30", lab3_mean="130"),
        ),
    )

    cells = aggregate_paired_summary(path, _expansion())

    assert _cell(cells, AggregateStatistic.MEAN, TaskCategory.CLASSIC, "runtime-100", "collection_fps").task_count == 1
    assert _cell(cells, AggregateStatistic.MEAN, TaskCategory.CLASSIC, "runtime-1000", "collection_fps").task_count == 1
    assert _cell(cells, AggregateStatistic.MEAN, TaskCategory.CLASSIC, "training-100", "collection_fps").task_count == 0
    flat = _cell(cells, AggregateStatistic.MEAN, TaskCategory.LOCOMOTION_FLAT, "runtime-100", "collection_fps")
    rough = _cell(cells, AggregateStatistic.MEAN, TaskCategory.LOCOMOTION_ROUGH, "runtime-100", "collection_fps")
    assert (flat.value, flat.task_count) == (20.0, 1)
    assert (rough.value, rough.task_count) == (30.0, 1)


def test_zero_baseline_is_excluded_and_empty_cells_are_explicit(tmp_path: Path) -> None:
    path = _write_summary(
        tmp_path / "paired_summary.csv",
        (
            _row("cartpole", percent_delta="10"),
            _row("cartpole_direct", percent_delta="20", lab3_mean="120"),
            _row(
                "ant",
                metric="startup_total_s",
                lab2_mean="0.0",
                lab3_mean="3.0",
                percent_delta="",
                percent_delta_status="undefined_zero_baseline",
            ),
        ),
    )

    cells = aggregate_paired_summary(path, _expansion())

    available = _cell(cells, AggregateStatistic.MEAN, TaskCategory.CLASSIC, "runtime-100", "collection_fps")
    empty_median = _cell(cells, AggregateStatistic.MEDIAN, TaskCategory.CLASSIC, "runtime-100", "startup_total_s")
    empty_mean = _cell(cells, AggregateStatistic.MEAN, TaskCategory.CLASSIC, "runtime-100", "startup_total_s")
    assert (available.value, available.task_count) == (15.0, 2)
    assert (empty_median.value, empty_median.task_count) == (None, 0)
    assert (empty_mean.value, empty_mean.task_count) == (None, 0)


def test_serialized_percentage_disagreement_names_the_row(tmp_path: Path) -> None:
    path = _write_summary(tmp_path / "paired_summary.csv", (_row(percent_delta="9"),))

    with pytest.raises(ValueError, match="cartpole.*runtime-100.*collection_fps"):
        aggregate_paired_summary(path, _expansion())


@pytest.mark.parametrize(
    ("row", "message"),
    (
        (_row(task="unknown"), "unknown task"),
        (_row(mode="unknown"), "unknown mode"),
        (_row(metric="unknown"), "unknown metric"),
        (_row(percent_delta_status="unknown"), "unknown percentage status"),
        (_row(task="cartpole_rgb_kit", mode="training-100"), "unsupported mode"),
        (_row(lab2_std="bad"), "lab2_std"),
        (_row(lab3_std="inf"), "lab3_std"),
        (_row(paired_seed_count="0"), "paired_seed_count"),
        (_row(paired_seed_count="1.5"), "paired_seed_count"),
    ),
)
def test_invalid_summary_rows_are_rejected(tmp_path: Path, row: dict[str, str], message: str) -> None:
    path = _write_summary(tmp_path / "paired_summary.csv", (row,))

    with pytest.raises(ValueError, match=message):
        aggregate_paired_summary(path, _expansion())


def test_duplicate_summary_rows_are_rejected(tmp_path: Path) -> None:
    row = _row()
    path = _write_summary(tmp_path / "paired_summary.csv", (row, row))

    with pytest.raises(ValueError, match="duplicate.*cartpole.*runtime-100.*collection_fps"):
        aggregate_paired_summary(path, _expansion())


def test_every_known_metric_is_validated_even_when_not_aggregated(tmp_path: Path) -> None:
    path = _write_summary(
        tmp_path / "paired_summary.csv",
        (_row(metric="elapsed_time_s", percent_delta="12"),),
    )

    with pytest.raises(ValueError, match="cartpole.*runtime-100.*elapsed_time_s"):
        aggregate_paired_summary(path, _expansion())


@pytest.mark.parametrize(
    "status_overrides",
    (
        {"lab2_mean": "0", "lab3_mean": "1", "percent_delta": "", "percent_delta_status": "available"},
        {
            "lab2_mean": "0",
            "lab3_mean": "1",
            "percent_delta": "1",
            "percent_delta_status": "undefined_zero_baseline",
        },
        {
            "lab2_mean": "100",
            "lab3_mean": "110",
            "percent_delta": "",
            "percent_delta_status": "undefined_zero_baseline",
        },
    ),
)
def test_inconsistent_percentage_status_is_rejected(tmp_path: Path, status_overrides: dict[str, str]) -> None:
    path = _write_summary(tmp_path / "paired_summary.csv", (_row(**status_overrides),))

    with pytest.raises(ValueError, match="percent"):
        aggregate_paired_summary(path, _expansion())


def test_exact_paired_summary_header_is_required(tmp_path: Path) -> None:
    fields = (*PAIRED_SUMMARY_FIELDS[:-1], "unexpected_status")
    path = _write_summary(tmp_path / "paired_summary.csv", (), fields)

    with pytest.raises(ValueError, match="header"):
        aggregate_paired_summary(path, _expansion())


@pytest.mark.parametrize("record_shape", ("surplus", "missing"))
def test_paired_summary_records_must_match_the_exact_header(tmp_path: Path, record_shape: str) -> None:
    row = _row()
    values = tuple(row[field] for field in PAIRED_SUMMARY_FIELDS)
    malformed = (*values, "unexpected") if record_shape == "surplus" else values[:-1]
    path = _write_raw_summary(tmp_path / "paired_summary.csv", malformed)

    with pytest.raises(ValueError, match="row does not match header"):
        aggregate_paired_summary(path, _expansion())


def test_missing_result_rows_only_reduce_contributing_count(tmp_path: Path) -> None:
    expansion = _expansion("cartpole", "cartpole_direct", "anymal_d_flat", "anymal_d_rough", "allegro_cube")
    path = _write_summary(tmp_path / "paired_summary.csv", (_row("cartpole"),))

    cells = aggregate_paired_summary(path, expansion)

    cell = _cell(cells, AggregateStatistic.MEAN, TaskCategory.CLASSIC, "runtime-100", "collection_fps")
    assert (cell.value, cell.task_count) == (10.0, 1)


def test_expansion_alias_absent_from_matrix_is_rejected_before_reading_csv(tmp_path: Path) -> None:
    expansion = _expansion("cartpole")
    pair = replace(expansion.pairs[0], logical_task="absent_from_matrix")
    invalid = replace(expansion, pairs=(pair,))

    with pytest.raises(ValueError, match="absent from the current matrix"):
        aggregate_paired_summary(tmp_path / "does-not-exist.csv", invalid)


@pytest.mark.parametrize("metadata_kind", ("missing", "duplicate"))
def test_invalid_category_metadata_is_rejected_before_reading_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metadata_kind: str
) -> None:
    expansion = _expansion()
    categories = task_aliases_by_category(expansion)
    if metadata_kind == "missing":
        invalid = dict(categories)
        invalid.pop(TaskCategory.CLASSIC)
    else:
        invalid = {**categories, TaskCategory.MANIPULATION: (*categories[TaskCategory.MANIPULATION], "cartpole")}
    monkeypatch.setattr(aggregate_module, "task_aliases_by_category", lambda _expansion: invalid)

    with pytest.raises(ValueError, match=f"{metadata_kind}.*category"):
        aggregate_paired_summary(tmp_path / "does-not-exist.csv", expansion)


def test_identical_input_bytes_return_equal_immutable_results(tmp_path: Path) -> None:
    path = _write_summary(tmp_path / "paired_summary.csv", (_row(),))
    expansion = _expansion()

    first = aggregate_paired_summary(path, expansion)
    second = aggregate_paired_summary(path, expansion)

    assert first == second
    assert isinstance(first, tuple)
    with pytest.raises(FrozenInstanceError):
        first[0].value = 1.0  # type: ignore[misc]
