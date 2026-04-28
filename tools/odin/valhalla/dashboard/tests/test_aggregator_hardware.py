# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests that ``aggregate_dispatch`` writes hardware.json."""

from __future__ import annotations

import json
from pathlib import Path

from tools.odin.valhalla.aggregator import aggregate_dispatch


def _write_dispatch_json(dispatch_dir: Path, jobs: list[dict]) -> None:
    payload = {
        "schema_version": "1.3",
        "dispatch_id": dispatch_dir.name,
        "started_at": "2026-04-27T14:13:02Z",
        "ended_at": "2026-04-27T14:30:00Z",
        "seeds": [42],
        "commit_sha": "abc123",
        "fleet": [{"host": "v1", "status": "idle", "current_run_id": None, "last_error": None}],
        "jobs": jobs,
        "skipped": [],
    }
    (dispatch_dir / "dispatch.json").write_text(json.dumps(payload))


def _write_bundle(dispatch_dir: Path, run_id: str, hardware: dict | None) -> None:
    bundle = dispatch_dir / run_id
    bundle.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "machine": {"hostname": "v1", "git_commit": "abc123", "git_branch": "main"},
        "framework": "rsl_rl",
        "backend": "physx",
        "task": "Isaac-Ant-Direct-v0",
        "seed": 42,
        "phases": {
            "training": {"file": "training.json", "status": "completed", "exit_code": 0, "duration_s": 10.0},
            "startup": {"file": "startup.json", "status": "completed", "exit_code": 0, "duration_s": 1.0},
        },
        "artifacts": [],
    }
    training: dict = {
        "schema_version": "1.0",
        "runtime": {
            "iterations_completed": 300,
            "total_wall_time_s": 10.0,
            "iteration_time_s": {"mean": 0.5, "std": 0.05},
            "env_steps_per_s": {"mean": 250000.0, "std": 2500.0},
            "iterations_per_s": {"mean": 2.0, "std": 0.01},
            "startup_phase_times_s": {"app_launch": 1.0, "env_creation": 2.0, "first_step": 0.01},
        },
        "resources": {
            "ram_gb": {"mean": 7.0, "peak": 8.0},
            "gpu_mem_gb": {"mean": 3.5, "peak": 4.0},
        },
        "learning": {
            "reward": {"final_raw": 100.0, "final_ema": 99.0, "series_per_iter": [0.0] * 300},
            "ep_length": {"final_raw": 920.0, "final_ema": 900.0, "series_per_iter": [0.0] * 300},
        },
    }
    if hardware is not None:
        training["hardware"] = hardware
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    (bundle / "training.json").write_text(json.dumps(training))
    (bundle / "startup.json").write_text(json.dumps({"schema_version": "1.0"}))


def test_aggregate_dispatch_writes_hardware_json(tmp_path):
    """A successful aggregate run produces hardware.json next to aggregate.json."""
    d = tmp_path / "20260427-141302"
    d.mkdir()
    _write_dispatch_json(d, [{
        "run_id": "rsl-rl_physx_Ant_seed42",
        "task_id": "Isaac-Ant-Direct-v0",
        "framework": "rsl_rl",
        "backend": "physx",
        "num_envs": 4096,
        "max_iterations": 300,
        "seed": 42,
        "bundle_dir_name": "rsl-rl_physx_Ant_seed42",
        "status": "completed",
        "assigned_to": "v1",
        "attempts": 1,
        "started_at": "2026-04-27T14:13:02Z",
        "ended_at": "2026-04-27T14:30:00Z",
        "preferred_not": [],
        "failure": None,
    }])
    _write_bundle(d, "rsl-rl_physx_Ant_seed42", hardware={
        "hostname": "Odin-Runner-5",
        "gpu_devices": [{"name": "NVIDIA L40", "mem_gb": 44.32, "compute_cap": "8.9"}],
        "cpu_name": "Intel Xeon Processor (Icelake)",
        "cpu_count": 16,
        "ram_gb": 62.79,
    })

    aggregate_dispatch(d)

    hw_path = d / "hardware.json"
    assert hw_path.exists()
    payload = json.loads(hw_path.read_text())
    assert payload["schema_version"] == "1.0"
    assert payload["dispatch_id"] == "20260427-141302"
    assert "generated_at" in payload
    assert payload["fingerprint"] == "gpu:NVIDIA-L40"
    assert "v1" in payload["hosts"]
    block = payload["hosts"]["v1"]
    assert block["hostname"] == "Odin-Runner-5"
    assert block["gpu_devices"] == [{"name": "NVIDIA L40", "mem_gb": 44.32, "compute_cap": "8.9"}]
    assert block["cpu_name"] == "Intel Xeon Processor (Icelake)"
    assert block["cpu_count"] == 16
    assert block["ram_gb"] == 62.79
    assert block["sourced_from"] == "rsl-rl_physx_Ant_seed42"


def test_aggregate_dispatch_skips_hardware_when_no_completed_bundles(tmp_path):
    """No completed bundles → no hardware.json (warning logged, aggregate still written)."""
    d = tmp_path / "20260427-141302"
    d.mkdir()
    _write_dispatch_json(d, [{
        "run_id": "rsl-rl_physx_Ant_seed42",
        "task_id": "Isaac-Ant-Direct-v0",
        "framework": "rsl_rl",
        "backend": "physx",
        "num_envs": 4096,
        "max_iterations": 300,
        "seed": 42,
        "bundle_dir_name": "rsl-rl_physx_Ant_seed42",
        "status": "failed",
        "assigned_to": "v1",
        "attempts": 1,
        "started_at": "2026-04-27T14:13:02Z",
        "ended_at": "2026-04-27T14:30:00Z",
        "preferred_not": [],
        "failure": {"kind": "hugin_crash", "message": "x", "details": {}},
    }])

    aggregate_dispatch(d)

    assert not (d / "hardware.json").exists()


def test_aggregate_dispatch_skips_hardware_when_training_lacks_block(tmp_path):
    """Bundle exists but training.json has no .hardware → no hardware.json."""
    d = tmp_path / "20260427-141302"
    d.mkdir()
    _write_dispatch_json(d, [{
        "run_id": "rsl-rl_physx_Ant_seed42",
        "task_id": "Isaac-Ant-Direct-v0",
        "framework": "rsl_rl",
        "backend": "physx",
        "num_envs": 4096,
        "max_iterations": 300,
        "seed": 42,
        "bundle_dir_name": "rsl-rl_physx_Ant_seed42",
        "status": "completed",
        "assigned_to": "v1",
        "attempts": 1,
        "started_at": "2026-04-27T14:13:02Z",
        "ended_at": "2026-04-27T14:30:00Z",
        "preferred_not": [],
        "failure": None,
    }])
    _write_bundle(d, "rsl-rl_physx_Ant_seed42", hardware=None)

    aggregate_dispatch(d)

    # Aggregator does not raise; hardware.json simply not written.
    assert not (d / "hardware.json").exists()
