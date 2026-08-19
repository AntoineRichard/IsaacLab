# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mapping a schema bundle onto the OSMO results table.

The metric names are read by dashboard SQL as literal paths, so the assertions here
are deliberately about exact spelling and units rather than about shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.odin.publish import build_row, bundle_to_kpi_phases, collect_rows, run_key_for

_BUNDLE = {
    "schema_version": 1.2,
    "run": {
        "task": "Isaac-Ant",
        "framework": "rsl_rl",
        "seed": 42,
        "num_envs": 4096,
        "max_iterations": 150,
        "status": "completed",
        "start_time_utc": "2026-08-19T10:00:00+00:00",
        "config": {"physics_backend": "newton_mjwarp", "rendering_backend": "none", "presets": ["newton_mjwarp"]},
    },
    "runtime": {
        "startup_time_s": {"app_launch": 1.0, "python_imports": 2.0, "task_config": 0.5},
        "total_fps": {"mean": 100.0, "std": 5.0, "peak": 110.0},
        "iteration_time_s": {"mean": 0.25, "std": 0.01, "peak": 0.4},
        "iterations_completed": 150,
        "total_wall_time_s": 42.0,
    },
    "resources": {"gpu_mem_gb": {"mean": 1.5, "std": 0.0, "peak": 1.6}},
    "learning": {"reward": {"final_ema": 12.5}, "ep_length": {"final_ema": 900.0}},
    "hardware": {"cpu_name": "AMD EPYC", "cpu_count": 128, "ram_gb": 1000.0, "gpu_devices": [{"name": "L40"}]},
    "versions": {"git_commit": "abc123", "isaacsim": None, "kit": None, "isaaclab": "14.0.0"},
    "success_rate": 0.97,
}


def _row(**kwargs):
    return build_row(
        _BUNDLE, kind="training", dispatch_id="20260819-120000", row_key="rk", image_ref="img", pool="pool", **kwargs
    )


def test_columns_match_the_table() -> None:
    assert sorted(_row()) == ["benchmark_type", "job_id", "kpis", "meta", "result", "startup"]


def test_times_are_reported_in_milliseconds() -> None:
    # The bundle records seconds; the table's startup phase is milliseconds, and the
    # dashboards divide by 1000 for display.
    startup = bundle_to_kpi_phases(_BUNDLE, workflow_name="w")["startup"]
    assert startup["App Launch Time"] == 1000.0
    assert startup["Python Imports Time"] == 2000.0
    assert startup["Total Start Time (Launch to Train)"] == 3500.0


def test_mean_std_and_peak_become_three_named_metrics() -> None:
    # A single MeanStd maps to three columns; the mean carries no suffix.
    runtime = bundle_to_kpi_phases(_BUNDLE, workflow_name="w")["runtime"]
    assert runtime["Mean Total FPS"] == 100.0
    assert runtime["Mean Total FPS std"] == 5.0
    assert runtime["Mean Total FPS peak"] == 110.0


def test_iteration_time_is_converted_but_fps_is_not() -> None:
    runtime = bundle_to_kpi_phases(_BUNDLE, workflow_name="w")["runtime"]
    assert runtime["Mean Iteration Time"] == 250.0
    assert runtime["Mean Total FPS"] == 100.0


def test_absent_metrics_are_omitted_rather_than_null() -> None:
    # A partial phase is preferable to one full of nulls: the dashboard reads by path.
    startup = bundle_to_kpi_phases(_BUNDLE, workflow_name="w")["startup"]
    assert "Scene Creation Time" not in startup
    assert "Simulation Start Time" not in startup


def test_headless_runs_do_not_report_a_renderer() -> None:
    # Bundles spell headless as the string "none", which must not reach the run key
    # or the preset encoding.
    row = _row()
    assert "none" not in run_key_for(_BUNDLE, "training")
    assert row["meta"]["rendering_backend"] is None


def test_preset_encoding_does_not_repeat_a_mirrored_backend() -> None:
    # Backends are mirrored into `presets` on some tasks; the naive concatenation
    # produced "newton_mjwarp,newton_mjwarp".
    assert _row()["result"]["benchmark_session"]["kit_envs"] == "OMNIPERF_ISAACLAB_PRESET=newton_mjwarp"


def test_app_version_is_the_isaac_lab_commit() -> None:
    # Kitless runs have no Isaac Sim version; the table's app_version is a commit,
    # which is what makes the field meaningful for them.
    session = _row()["result"]["benchmark_session"]
    assert session["app_version"] == "abc123"
    assert _BUNDLE["versions"]["isaacsim"] is None


def test_only_completed_runs_are_marked_success() -> None:
    # The dashboards filter on omniperf_type; a mislabelled row is invisible.
    assert _row()["result"]["omniperf_type"] == "success"
    crashed = json.loads(json.dumps(_BUNDLE))
    crashed["run"]["status"] = "failed"
    row = build_row(crashed, kind="training", dispatch_id="d", row_key="r", image_ref="i", pool="p")
    assert row["result"]["omniperf_type"] == "failure"


def test_meta_carries_the_dimensions_the_session_model_has_no_column_for() -> None:
    meta = _row()["meta"]
    assert meta["physics_backend"] == "newton_mjwarp"
    assert meta["seed"] == 42
    assert meta["rl_library"] == "rsl_rl"
    assert meta["git_commit"] == "abc123"


def test_training_quality_lands_in_the_train_phase() -> None:
    train = bundle_to_kpi_phases(_BUNDLE, workflow_name="w")["train"]
    assert train["Success Rate (tail mean)"] == 0.97
    assert train["Max Rewards"] == 12.5


def test_run_key_separates_backends_of_one_task() -> None:
    # Without the backend two rows of a task collide under one key and overwrite.
    other = json.loads(json.dumps(_BUNDLE))
    other["run"]["config"]["physics_backend"] = "ovphysx"
    assert run_key_for(_BUNDLE, "training") != run_key_for(other, "training")


def test_collect_skips_unreadable_bundles(tmp_path: Path) -> None:
    # One corrupt file must not cost the whole publish.
    (tmp_path / "row_a").mkdir()
    (tmp_path / "row_a" / "benchmark_training_x.json").write_text(json.dumps(_BUNDLE))
    (tmp_path / "row_b").mkdir()
    (tmp_path / "row_b" / "benchmark_training_y.json").write_text("{ not json")

    rows = collect_rows(tmp_path, image_ref="img", pool="pool")

    assert len(rows) == 1


@pytest.mark.parametrize("phase", ["startup", "runtime", "version_info", "hardware_info"])
def test_every_phase_names_its_producing_workflow(phase: str) -> None:
    # The sample carries workflow_name on each phase; dashboards group on it.
    phases = bundle_to_kpi_phases(_BUNDLE, workflow_name="benchmark_training")
    assert phases[phase]["workflow_name"] == "benchmark_training"


def test_run_key_names_the_scope() -> None:
    # Lets a dashboard separate core from contrib without pattern-matching the task id.
    from tools.odin.publish import scope_of

    assert scope_of("Isaac-Ant") == "core"
    assert scope_of("IsaacContrib-Velocity-Flat-Spot") == "contrib"
    assert "_core_" in run_key_for(_BUNDLE, "training")


def test_presets_are_sorted_so_one_config_yields_one_key() -> None:
    # Discovery emits presets in registry order; without sorting the same run could
    # produce two keys across releases and silently split one trend line in two.
    from tools.odin.publish import preset_slug

    a = {"physics_backend": "ovphysx", "rendering_backend": "ovrtx", "presets": ["rgb", "albedo64"]}
    b = {"physics_backend": "ovphysx", "rendering_backend": "ovrtx", "presets": ["albedo64", "rgb"]}
    assert preset_slug(a) == preset_slug(b) == "albedo64-ovphysx-ovrtx-rgb"


def test_domain_presets_do_not_collide_under_one_key() -> None:
    # Naming only the backend was enough while every task was state-only; a camera
    # task varies by domain preset and those rows would overwrite each other.
    import json

    rgb = json.loads(json.dumps(_BUNDLE))
    rgb["run"]["task"] = "Isaac-Cartpole-Camera"
    rgb["run"]["config"] = {
        "physics_backend": "isaacsim_physx",
        "rendering_backend": "isaacsim_rtx",
        "presets": ["rgb"],
    }
    depth = json.loads(json.dumps(rgb))
    depth["run"]["config"]["presets"] = ["depth128"]

    assert run_key_for(rgb, "training") != run_key_for(depth, "training")
