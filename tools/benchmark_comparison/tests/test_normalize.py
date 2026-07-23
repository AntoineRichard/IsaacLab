# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for deterministic benchmark artifact normalization and paired summaries."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

from tools.benchmark_comparison.artifacts import finalize_attempt
from tools.benchmark_comparison.matrix import expand_final_matrix, load_matrix
from tools.benchmark_comparison.normalize import (
    RAW_RUN_FIELDS,
    NormalizedRun,
    normalize_run_set,
    summarize_pairs,
    write_normalized_outputs,
)
from tools.benchmark_comparison.validate import attempt_identity

FIXTURES = Path(__file__).parent / "fixtures"
LAB2_SHA = "a" * 40
LAB3_SHA = "b" * 40
LAB2_IMAGE_ID = "sha256:" + "c" * 64
LAB3_LOCK = "d" * 64


def _attempts():
    expansion = expand_final_matrix(load_matrix())
    selected = tuple(
        attempt
        for attempt in expansion.attempts
        if attempt.logical_task == "cartpole" and attempt.mode.id == "runtime-100" and attempt.seed in (42, 43)
    )
    return replace(
        expansion, attempts=selected, pairs=tuple(pair for pair in expansion.pairs if pair.attempts[0] in selected)
    )


def _payloads(attempt, *, collection_fps: float, utilization: float, exit_code: int | None = 0):
    identity = attempt_identity(attempt)
    schema = json.loads((FIXTURES / "schema_runtime.json").read_text(encoding="utf-8"))
    schema["run"].update(task=attempt.concrete_task, seed=attempt.seed, num_envs=attempt.num_envs)
    schema["runtime"]["iterations_completed"] = attempt.bound.value
    schema["runtime"]["collection_fps"]["mean"] = collection_fps
    schema["resources"]["gpu_mem_gb"].update(mean=1.5, peak=2.0)
    schema["resources"]["gpu_util_pct"]["mean"] = utilization
    schema["versions"]["isaaclab_release"] = "2.3.2" if attempt.version.value == "lab2" else "3.0.0"
    measurements = json.loads((FIXTURES / "generic_runtime.json").read_text(encoding="utf-8"))
    environment_identity = LAB2_IMAGE_ID if attempt.version.value == "lab2" else f"uv-lock:{LAB3_LOCK}"
    return {
        "command": {"identity": identity, "argv": ["benchmark"]},
        "environment": {
            "identity": identity,
            "environment_identity": environment_identity,
            "lab2_sha": LAB2_SHA,
            "lab3_sha": LAB3_SHA,
            "lab2_image_id": LAB2_IMAGE_ID if attempt.version.value == "lab2" else None,
            "uv_lock_sha256": LAB3_LOCK if attempt.version.value == "lab3" else None,
            "values": {
                "ISAACLAB_BENCHMARK_LAB2_SHA": LAB2_SHA,
                "ISAACLAB_BENCHMARK_LAB3_SHA": LAB3_SHA,
            },
        },
        "stdout": "",
        "stderr": "failure" if exit_code else "",
        "exit_status": {
            "exit_code": exit_code,
            "failure_stage": None,
            "timed_out": False,
            "interrupted": False,
            "out_of_memory": False,
            "wall_time_s": 12.5,
        },
        "schema": schema if exit_code == 0 else None,
        "measurements": measurements if exit_code == 0 else None,
    }


def _run(**overrides) -> NormalizedRun:
    values = {
        "version": "lab2",
        "version_sha": LAB2_SHA,
        "environment_identity": LAB2_IMAGE_ID,
        "logical_task": "cartpole",
        "concrete_task": "Isaac-Cartpole-v0",
        "mode": "runtime-100",
        "bound": 100,
        "bound_unit": "steps",
        "seed": 42,
        "num_envs": 4096,
        "collection_fps": 100.0,
        "gpu_memory_mean_mib": 1024.0,
        "gpu_memory_peak_mib": 2048.0,
        "gpu_utilization_mean_pct": 0.0,
        "gpu_utilization_sample_count": 8,
        "elapsed_time_s": 12.5,
        "artifact_path": "final/example/success",
    }
    values.update(overrides)
    return NormalizedRun(**values)


def test_normalization_writes_one_stably_ordered_row_per_success_and_preserves_failures(tmp_path: Path) -> None:
    expansion = _attempts()
    by_key = {(attempt.seed, attempt.version.value): attempt for attempt in expansion.attempts}
    finalize_attempt(
        tmp_path, by_key[(42, "lab2")], **_payloads(by_key[(42, "lab2")], collection_fps=100, utilization=40)
    )
    finalize_attempt(
        tmp_path, by_key[(42, "lab3")], **_payloads(by_key[(42, "lab3")], collection_fps=125, utilization=50)
    )
    finalize_attempt(
        tmp_path,
        by_key[(43, "lab3")],
        **_payloads(by_key[(43, "lab3")], collection_fps=0, utilization=0, exit_code=137),
    )

    runs, failures = normalize_run_set(tmp_path, expansion)
    paths = write_normalized_outputs(tmp_path / "normalized", runs, failures)

    assert [(row.seed, row.version) for row in runs] == [(42, "lab2"), (42, "lab3")]
    assert list(runs[0].to_csv_row()) == list(RAW_RUN_FIELDS)
    assert runs[0].version_sha == LAB2_SHA
    assert runs[0].environment_identity == LAB2_IMAGE_ID
    assert runs[1].version_sha == LAB3_SHA
    assert runs[1].environment_identity == f"uv-lock:{LAB3_LOCK}"
    assert runs[0].concrete_task == "Isaac-Cartpole-v0"
    assert runs[0].bound == 100
    assert runs[0].gpu_memory_mean_mib == 1536.0
    assert runs[0].gpu_memory_peak_mib == 2048.0
    assert runs[0].gpu_utilization_sample_count > 0
    assert runs[0].elapsed_time_s == 12.5
    assert runs[0].artifact_path.endswith("/success")
    assert [failure.failure_kind for failure in failures] == ["missing", "nonzero_exit"]
    assert set(paths) == {"raw_runs", "paired_summary", "failures"}
    assert {path.name for path in paths.values()} == {"raw_runs.csv", "paired_summary.csv", "failures.csv"}
    with paths["raw_runs"].open(newline="", encoding="utf-8") as file:
        assert list(csv.DictReader(file))[0]["collection_fps"] == "100"


def test_paired_summaries_use_only_valid_pairs_and_compute_exact_signed_deltas() -> None:
    runs = (
        _run(seed=42, collection_fps=100.0),
        _run(
            version="lab3",
            version_sha=LAB3_SHA,
            environment_identity=f"uv-lock:{LAB3_LOCK}",
            concrete_task="Isaac-Cartpole",
            seed=42,
            collection_fps=125.0,
            artifact_path="final/lab3/success",
        ),
        _run(seed=43, collection_fps=120.0, artifact_path="final/lab2-43/success"),
    )

    summaries = summarize_pairs(runs)
    throughput = next(row for row in summaries if row.metric == "collection_fps")

    assert throughput.paired_seed_count == 1
    assert throughput.lab2_mean == 100.0
    assert throughput.lab2_std == 0.0
    assert throughput.lab3_mean == 125.0
    assert throughput.lab3_std == 0.0
    assert throughput.absolute_delta == 25.0
    assert throughput.percent_delta == 25.0
    assert throughput.percent_delta_status == "available"


def test_zero_lab2_baseline_has_explicit_undefined_percent_semantics() -> None:
    summaries = summarize_pairs(
        (
            _run(gpu_utilization_mean_pct=0.0),
            _run(
                version="lab3",
                version_sha=LAB3_SHA,
                environment_identity=f"uv-lock:{LAB3_LOCK}",
                concrete_task="Isaac-Cartpole",
                gpu_utilization_mean_pct=10.0,
            ),
        )
    )
    utilization = next(row for row in summaries if row.metric == "gpu_utilization_mean_pct")

    assert utilization.absolute_delta == 10.0
    assert utilization.percent_delta is None
    assert utilization.percent_delta_status == "undefined_zero_baseline"
    assert utilization.to_csv_row()["percent_delta"] == ""
