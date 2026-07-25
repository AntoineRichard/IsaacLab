# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for deterministic benchmark artifact normalization and paired summaries."""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import replace
from pathlib import Path

import pytest

from tools.benchmark_comparison.artifacts import finalize_attempt
from tools.benchmark_comparison.manifest import HostIdentity, RunSetManifest, SoftwareIdentity
from tools.benchmark_comparison.matrix import expand_final_matrix, load_matrix
from tools.benchmark_comparison.models import ExecutionProvenance, RunSet
from tools.benchmark_comparison.normalize import (
    RAW_RUN_FIELDS,
    SUMMARY_METRICS,
    TASK_MODES,
    TASK_ORDER,
    NormalizedRun,
    _startup_metrics,
    normalize_run_set,
    read_raw_runs_csv,
    summarize_pairs,
    task_order_for_mode,
    write_normalized_outputs,
    write_raw_runs_csv,
)
from tools.benchmark_comparison.validate import attempt_identity

FIXTURES = Path(__file__).parent / "fixtures"
LAB2_SHA = "a" * 40
LAB3_SHA = "b" * 40
LAB2_IMAGE_ID = "sha256:" + "c" * 64
LAB3_LOCK = "d" * 64
GPU_UUID = "GPU-01234567-89ab-cdef-0123-456789abcdef"
STARTUP = {
    "app_launch": 2.5,
    "python_imports": 0.2,
    "task_config": 0.4,
    "env_creation": 1.3,
    "first_step": 0.01,
}


def test_expanded_task_order_keeps_rgb_runtime_only_and_both_anymal_terrains() -> None:
    assert TASK_ORDER == (
        "cartpole",
        "cartpole_rgb_kit",
        "cartpole_direct",
        "ant",
        "ant_direct",
        "humanoid_manager",
        "humanoid_direct",
        "anymal_d_flat",
        "anymal_d_rough",
        "g1_flat",
        "cassie_flat",
        "allegro_cube",
        "franka_reach",
    )
    assert task_order_for_mode("runtime-100") == TASK_ORDER
    assert task_order_for_mode("runtime-1000") == TASK_ORDER
    assert "cartpole_rgb_kit" not in task_order_for_mode("training-100")
    assert TASK_MODES["cartpole_rgb_kit"] == ("runtime-100", "runtime-1000")


def _provenance() -> ExecutionProvenance:
    return ExecutionProvenance(
        lab2_sha=LAB2_SHA,
        lab3_sha=LAB3_SHA,
        lab2_image_id=LAB2_IMAGE_ID,
        uv_lock_sha256=LAB3_LOCK,
    )


def _manifest() -> RunSetManifest:
    return RunSetManifest(
        schema_version="1.0",
        run_set=RunSet.FINAL,
        phase="measured",
        provenance=_provenance(),
        host=HostIdentity(
            hostname="fixture-host",
            os="Fixture OS",
            cpu_model="Fixture CPU",
            logical_cpu_count=32,
            gpu_model="Fixture GPU",
            gpu_driver="590.48.01",
            cuda_version="12.8",
        ),
        lab2=SoftwareIdentity("2.3.2", "5.1.0", "3.11.13", "2.7.0+cu128", "5.0.1"),
        lab3=SoftwareIdentity("3.0.0", "6.0.0", "3.12.13", "2.11.0+cu128", "5.4.1"),
    )


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
    software = _manifest().software(attempt.version.value)
    schema["versions"].update(
        isaaclab_release=software.isaac_lab,
        isaacsim=software.isaac_sim,
        torch=software.pytorch,
        rsl_rl=software.rsl_rl,
    )
    schema["hardware"].update(cpu_name="Fixture CPU", cpu_count=32)
    measurements = json.loads((FIXTURES / "generic_runtime.json").read_text(encoding="utf-8"))
    environment_identity = LAB2_IMAGE_ID if attempt.version.value == "lab2" else f"uv-lock:{LAB3_LOCK}"
    return {
        "command": {"identity": identity, "argv": ["benchmark"]},
        "environment": {
            "identity": identity,
            "environment_identity": environment_identity,
            "lab2_sha": LAB2_SHA,
            "lab3_sha": LAB3_SHA,
            "lab2_image_id": LAB2_IMAGE_ID,
            "uv_lock_sha256": LAB3_LOCK,
            "values": {
                "ISAACLAB_BENCHMARK_LAB2_SHA": LAB2_SHA,
                "ISAACLAB_BENCHMARK_LAB3_SHA": LAB3_SHA,
            },
        },
        "stdout": "| Driver Version: 590.48.01 | Graphics API: Vulkan\n",
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
        "startup_total_s": 4.41,
        "startup_app_launch_s": 2.5,
        "startup_python_imports_s": 0.2,
        "startup_task_config_s": 0.4,
        "startup_env_creation_s": 1.3,
        "startup_first_step_s": 0.01,
        "artifact_path": "final/example/success",
    }
    values.update(overrides)
    return NormalizedRun(**values)


def test_normalization_preserves_startup_components_and_computed_total(tmp_path: Path) -> None:
    attempt = _attempts().attempts[0]
    expansion = replace(_attempts(), attempts=(attempt,), pairs=())
    payloads = _payloads(attempt, collection_fps=100.0, utilization=40.0)
    payloads["schema"]["runtime"]["startup_time_s"] = STARTUP
    finalize_attempt(tmp_path, attempt, **payloads)

    runs, failures = normalize_run_set(tmp_path, expansion, _manifest())

    assert failures == ()
    assert len(runs) == 1
    run = runs[0]
    assert run.startup_app_launch_s == 2.5
    assert run.startup_python_imports_s == 0.2
    assert run.startup_task_config_s == 0.4
    assert run.startup_env_creation_s == 1.3
    assert run.startup_first_step_s == 0.01
    assert run.startup_total_s == pytest.approx(4.41)
    path = write_raw_runs_csv(tmp_path / "raw_runs.csv", runs)
    assert read_raw_runs_csv(path) == runs


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("startup_app_launch_s", "-0.01", "startup phase app_launch must be non-negative"),
        ("startup_total_s", "4.42", "startup_total_s does not equal serialized startup phases"),
    ],
)
def test_raw_runs_csv_rejects_tampered_startup_metrics(tmp_path: Path, field: str, value: str, reason: str) -> None:
    path = write_raw_runs_csv(tmp_path / "raw_runs.csv", (_run(),))
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    rows[0][field] = value
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RAW_RUN_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match=reason):
        read_raw_runs_csv(path)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("missing", "startup phases do not match canonical set"),
        ("unexpected", "startup phases do not match canonical set"),
        ("negative", "startup phase app_launch must be non-negative"),
    ],
)
def test_normalization_rejects_noncanonical_startup_metrics(tmp_path: Path, mutation: str, reason: str) -> None:
    attempt = _attempts().attempts[0]
    expansion = replace(_attempts(), attempts=(attempt,), pairs=())
    payloads = _payloads(attempt, collection_fps=100.0, utilization=40.0)
    startup = payloads["schema"]["runtime"]["startup_time_s"]
    if mutation == "missing":
        startup.pop("first_step")
    elif mutation == "unexpected":
        startup["other"] = 1.0
    else:
        startup["app_launch"] = -1.0
    finalize_attempt(tmp_path, attempt, **payloads)

    runs, failures = normalize_run_set(tmp_path, expansion, _manifest())

    assert runs == ()
    assert len(failures) == 1
    assert failures[0].failure_kind == "invalid_success"
    assert reason in failures[0].reason


def test_startup_metrics_rejects_non_finite_phase() -> None:
    startup = dict(STARTUP)
    startup["app_launch"] = math.inf

    with pytest.raises(ValueError, match="startup phase app_launch must be finite"):
        _startup_metrics(startup)


def test_startup_metrics_rejects_non_finite_total() -> None:
    startup = {phase: 1e308 for phase in STARTUP}

    with pytest.raises(ValueError, match="startup total must be finite"):
        _startup_metrics(startup)


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

    runs, failures = normalize_run_set(tmp_path, expansion, _manifest())
    paths = write_normalized_outputs(tmp_path / "normalized", runs, failures)

    assert [(row.seed, row.version) for row in runs] == [(42, "lab2"), (42, "lab3")]
    assert list(runs[0].to_csv_row()) == list(RAW_RUN_FIELDS)
    assert runs[0].version_sha == LAB2_SHA
    assert runs[0].environment_identity == LAB2_IMAGE_ID
    assert runs[0].isaac_sim_version == "5.1.0"
    assert runs[0].python_version == "3.11.13"
    assert runs[1].pytorch_version == "2.11.0+cu128"
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


def test_schema_two_normalization_rejects_success_from_different_gpu_uuid(tmp_path: Path) -> None:
    full_expansion = expand_final_matrix(load_matrix())
    expansion = replace(full_expansion, pairs=full_expansion.pairs[:1], attempts=full_expansion.attempts[:2])
    attempt = expansion.attempts[0]
    payloads = _payloads(attempt, collection_fps=100, utilization=40)
    payloads["environment"]["selected_gpu"] = {"physical_index": 0, "uuid": "GPU-WRONG"}
    payloads["environment"]["values"].update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "0",
            "NVIDIA_VISIBLE_DEVICES": "0",
            "ISAACLAB_BENCHMARK_GPU_INDEX": "0",
            "ISAACLAB_BENCHMARK_GPU_UUID": "GPU-WRONG",
        }
    )
    finalize_attempt(tmp_path, attempt, **payloads)
    manifest = replace(
        _manifest(),
        schema_version="2.0",
        host=replace(_manifest().host, gpu_index=0, gpu_uuid=GPU_UUID),
        expansion=expansion,
    )

    runs, failures = normalize_run_set(tmp_path, expansion, manifest)

    assert runs == ()
    invalid = next(failure for failure in failures if failure.failure_kind == "invalid_success")
    assert "selected GPU UUID" in invalid.reason


def test_written_paired_summary_is_derived_from_serialized_raw_runs(tmp_path: Path) -> None:
    runs = (
        _run(collection_fps=954293.634885703),
        _run(
            version="lab3",
            version_sha=LAB3_SHA,
            environment_identity=f"uv-lock:{LAB3_LOCK}",
            concrete_task="Isaac-Cartpole",
            collection_fps=947093.586299632,
        ),
    )

    paths = write_normalized_outputs(tmp_path, runs, ())
    with paths["paired_summary"].open(newline="", encoding="utf-8") as file:
        written = list(csv.DictReader(file))
    serialized_runs = tuple(
        NormalizedRun(
            **{
                **run.__dict__,
                "collection_fps": float(run.to_csv_row()["collection_fps"]),
            }
        )
        for run in runs
    )

    assert written == [summary.to_csv_row() for summary in summarize_pairs(serialized_runs)]


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
            startup_total_s=5.0,
            startup_app_launch_s=3.0,
            startup_python_imports_s=0.25,
            startup_task_config_s=0.5,
            startup_env_creation_s=1.2,
            startup_first_step_s=0.05,
            artifact_path="final/lab3/success",
        ),
        _run(seed=43, collection_fps=120.0, artifact_path="final/lab2-43/success"),
    )

    summaries = summarize_pairs(runs)
    throughput = next(row for row in summaries if row.metric == "collection_fps")
    startup = next(row for row in summaries if row.metric == "startup_total_s")

    assert {row.metric for row in summaries} == set(SUMMARY_METRICS)
    assert throughput.paired_seed_count == 1
    assert throughput.lab2_mean == 100.0
    assert throughput.lab2_std == 0.0
    assert throughput.lab3_mean == 125.0
    assert throughput.lab3_std == 0.0
    assert throughput.absolute_delta == 25.0
    assert throughput.percent_delta == 25.0
    assert throughput.percent_delta_status == "available"
    assert startup.absolute_delta == pytest.approx(0.59)


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


def test_quarantined_success_and_later_valid_success_are_both_visible(tmp_path: Path) -> None:
    attempt = _attempts().attempts[0]
    expansion = replace(_attempts(), attempts=(attempt,), pairs=())
    original = finalize_attempt(tmp_path, attempt, **_payloads(attempt, collection_fps=100, utilization=40))
    quarantine = original.with_name("corrupt-success-0001")
    os.rename(original, quarantine)
    finalize_attempt(tmp_path, attempt, **_payloads(attempt, collection_fps=110, utilization=42))

    runs, failures = normalize_run_set(tmp_path, expansion, _manifest())

    assert [run.collection_fps for run in runs] == [110.0]
    assert [(failure.failure_kind, failure.artifact_path) for failure in failures] == [
        ("invalid_success", quarantine.relative_to(tmp_path).as_posix())
    ]


def test_quarantined_success_without_replacement_remains_linked_failure(tmp_path: Path) -> None:
    attempt = _attempts().attempts[0]
    expansion = replace(_attempts(), attempts=(attempt,), pairs=())
    success = finalize_attempt(tmp_path, attempt, **_payloads(attempt, collection_fps=100, utilization=40))
    quarantine = success.with_name("corrupt-success-0001")
    os.rename(success, quarantine)

    runs, failures = normalize_run_set(tmp_path, expansion, _manifest())

    assert runs == ()
    assert any(failure.artifact_path == quarantine.relative_to(tmp_path).as_posix() for failure in failures)
    assert any(failure.failure_kind == "invalid_success" for failure in failures)


def test_normalization_rejects_success_with_preflight_provenance_mismatch(tmp_path: Path) -> None:
    attempt = _attempts().attempts[0]
    expansion = replace(_attempts(), attempts=(attempt,), pairs=())
    payloads = _payloads(attempt, collection_fps=100, utilization=40)
    payloads["environment"]["uv_lock_sha256"] = "e" * 64
    finalize_attempt(tmp_path, attempt, **payloads)

    runs, failures = normalize_run_set(tmp_path, expansion, _manifest())

    assert runs == ()
    assert [failure.failure_kind for failure in failures] == ["invalid_success"]
    assert "provenance uv_lock_sha256" in failures[0].reason


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bound", 1000),
        ("bound_unit", "iterations"),
        ("num_envs", 2048),
    ],
)
def test_paired_summaries_reject_mismatched_pair_invariants(field: str, value: object) -> None:
    lab3 = {
        "version": "lab3",
        "version_sha": LAB3_SHA,
        "environment_identity": f"uv-lock:{LAB3_LOCK}",
        "concrete_task": "Isaac-Cartpole",
    }
    lab3[field] = value

    with pytest.raises(ValueError, match=field):
        summarize_pairs((_run(), _run(**lab3)))


def test_paired_summaries_compute_sample_stddev_across_two_complete_seeds() -> None:
    runs = (
        _run(seed=42, collection_fps=100.0),
        _run(
            version="lab3",
            version_sha=LAB3_SHA,
            environment_identity=f"uv-lock:{LAB3_LOCK}",
            concrete_task="Isaac-Cartpole",
            seed=42,
            collection_fps=130.0,
        ),
        _run(seed=43, collection_fps=120.0),
        _run(
            version="lab3",
            version_sha=LAB3_SHA,
            environment_identity=f"uv-lock:{LAB3_LOCK}",
            concrete_task="Isaac-Cartpole",
            seed=43,
            collection_fps=150.0,
        ),
    )

    throughput = next(row for row in summarize_pairs(runs) if row.metric == "collection_fps")

    assert throughput.paired_seed_count == 2
    assert throughput.lab2_mean == 110.0
    assert throughput.lab3_mean == 140.0
    assert math.isclose(throughput.lab2_std, math.sqrt(200.0))
    assert math.isclose(throughput.lab3_std, math.sqrt(200.0))
    assert throughput.absolute_delta == 30.0
    assert math.isclose(throughput.percent_delta, 300.0 / 11.0)


def test_normalization_rejects_failed_attempt_with_manifest_provenance_mismatch(tmp_path: Path) -> None:
    attempt = _attempts().attempts[0]
    expansion = replace(_attempts(), attempts=(attempt,), pairs=())
    payloads = _payloads(attempt, collection_fps=0, utilization=0, exit_code=137)
    payloads["environment"]["uv_lock_sha256"] = "e" * 64
    finalize_attempt(tmp_path, attempt, **payloads)

    runs, failures = normalize_run_set(tmp_path, expansion, _manifest())

    assert runs == ()
    assert [failure.failure_kind for failure in failures] == ["malformed_artifact"]
    assert "provenance uv_lock_sha256" in failures[0].reason
