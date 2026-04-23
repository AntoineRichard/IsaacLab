# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for valhalla.aggregator on synthetic dispatch directories."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.odin.valhalla.aggregator import AggregateOptions, aggregate_dispatch


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)


def _write_completed_bundle(
    dispatch_dir: Path,
    run_id: str,
    *,
    reward_final_ema: float,
    ep_length_final_ema: float = 900.0,
    iter_time_mean: float = 0.5,
    iter_time_std: float = 0.05,
    env_steps_per_s_mean: float = 250000.0,
    ram_gb_peak: float = 8.0,
    gpu_mem_gb_peak: float = 4.0,
    commit_sha: str = "abc123",
    hostname: str = "valkyrie-01.internal",
) -> Path:
    bundle = dispatch_dir / run_id
    _write_json(
        bundle / "manifest.json",
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "machine": {"hostname": hostname, "git_commit": commit_sha, "git_branch": "main"},
            "phases": {
                "startup": {"file": "startup.json", "status": "completed", "duration_s": 30.0, "exit_code": 0},
                "training": {"file": "training.json", "status": "completed", "duration_s": 150.0, "exit_code": 0},
            },
            "config": {"framework": "rsl_rl", "backend": "physx", "task": "Isaac-Ant-Direct-v0", "seed": 42, "num_envs": 4096, "max_iterations": 300},
            "run_start_time_utc": "2026-04-23T10:00:00Z",
            "run_end_time_utc": "2026-04-23T10:03:00Z",
            "run_duration_s": 180.0,
            "artifacts": ["logs", "startup.json", "training.json", "training_data"],
        },
    )
    _write_json(
        bundle / "training.json",
        {
            "schema_version": "1.0",
            "runtime": {
                "iterations_completed": 300,
                "total_wall_time_s": 150.0,
                "iteration_time_s": {"mean": iter_time_mean, "std": iter_time_std},
                "env_steps_per_s": {"mean": env_steps_per_s_mean, "std": env_steps_per_s_mean * 0.01},
                "iterations_per_s": {"mean": 1.0 / iter_time_mean, "std": 0.01},
                "startup_phase_times_s": {"app_launch": 4.5, "env_creation": 12.4, "first_step": 0.006},
            },
            "resources": {
                "ram_gb": {"mean": ram_gb_peak * 0.9, "peak": ram_gb_peak},
                "gpu_mem_gb": {"mean": gpu_mem_gb_peak * 0.9, "peak": gpu_mem_gb_peak},
            },
            "learning": {
                "reward": {"final_raw": reward_final_ema * 1.01, "final_ema": reward_final_ema, "series_per_iter": [0.0] * 300},
                "ep_length": {"final_raw": ep_length_final_ema * 1.02, "final_ema": ep_length_final_ema, "series_per_iter": [0.0] * 300},
            },
        },
    )
    return bundle


def _make_dispatch_json(dispatch_dir: Path, jobs: list[dict]) -> None:
    _write_json(
        dispatch_dir / "dispatch.json",
        {
            "schema_version": "1.0",
            "dispatch_id": dispatch_dir.name,
            "started_at": "2026-04-23T09:59:00Z",
            "ended_at": "2026-04-23T10:10:00Z",
            "seeds": [42, 43],
            "commit_sha": "abc123",
            "fleet": [{"host": "valkyrie-01.internal", "status": "idle", "current_run_id": None, "last_error": None}],
            "jobs": jobs,
        },
    )


def _job(
    run_id: str,
    *,
    task: str = "Isaac-Ant-Direct-v0",
    framework: str = "rsl_rl",
    backend: str = "physx",
    seed: int = 42,
    status: str = "completed",
    assigned_to: str = "valkyrie-01.internal",
    failure: dict | None = None,
) -> dict:
    return {
        "run_id": run_id,
        "task_id": task,
        "framework": framework,
        "backend": backend,
        "num_envs": 4096,
        "max_iterations": 300,
        "seed": seed,
        "bundle_dir_name": run_id,
        "status": status,
        "assigned_to": assigned_to,
        "attempts": 1,
        "failure": failure,
        "preferred_not": [],
        "started_at": "2026-04-23T10:00:00Z",
        "ended_at": "2026-04-23T10:03:00Z",
    }


def test_happy_path_two_completed_seeds(tmp_path: Path):
    dispatch = tmp_path / "20260423-100000"
    dispatch.mkdir()
    _write_completed_bundle(dispatch, "run42", reward_final_ema=100.0)
    _write_completed_bundle(dispatch, "run43", reward_final_ema=110.0)
    _make_dispatch_json(
        dispatch,
        [_job("run42", seed=42), _job("run43", seed=43)],
    )
    agg = aggregate_dispatch(dispatch)

    assert agg["schema_version"] == "1.0"
    assert agg["dispatch_id"] == "20260423-100000"
    assert agg["commit_sha"] == "abc123"
    assert agg["totals"] == {"tasks": 1, "runs": 2, "completed": 2, "failed": 0}
    assert len(agg["rows"]) == 1
    row = agg["rows"][0]
    assert row["task"] == "Isaac-Ant-Direct-v0"
    assert row["framework"] == "rsl_rl"
    assert row["backend"] == "physx"
    assert set(row["seeds"].keys()) == {"42", "43"}
    assert row["seeds"]["42"]["reward_final_ema"] == 100.0
    assert row["seeds"]["43"]["reward_final_ema"] == 110.0
    assert row["seeds"]["42"]["status"] == "completed"
    assert row["aggregate"]["n_seeds_completed"] == 2
    assert row["aggregate"]["n_seeds_failed"] == 0
    assert row["aggregate"]["reward_final_ema"]["mean"] == pytest.approx(105.0)
    assert row["divergent_seeds"] == []
    assert agg["failures"] == []


def test_divergent_seed_flagged(tmp_path: Path):
    # Match the stats.is_divergent canonical test shape (5 clustered + 1 outlier),
    # which comfortably exceeds z=2.0; smaller n values are below threshold.
    dispatch = tmp_path / "20260423-100000"
    dispatch.mkdir()
    _write_completed_bundle(dispatch, "run42", reward_final_ema=100.0)
    _write_completed_bundle(dispatch, "run43", reward_final_ema=100.0)
    _write_completed_bundle(dispatch, "run44", reward_final_ema=100.0)
    _write_completed_bundle(dispatch, "run45", reward_final_ema=100.0)
    _write_completed_bundle(dispatch, "run46", reward_final_ema=100.0)
    _write_completed_bundle(dispatch, "run47", reward_final_ema=1000.0)  # outlier
    _make_dispatch_json(
        dispatch,
        [_job(f"run{s}", seed=s) for s in (42, 43, 44, 45, 46, 47)],
    )
    agg = aggregate_dispatch(dispatch)
    row = agg["rows"][0]
    assert "47" in row["divergent_seeds"]
    assert "42" not in row["divergent_seeds"]


def test_mixed_completed_and_failed_seeds(tmp_path: Path):
    dispatch = tmp_path / "20260423-100000"
    dispatch.mkdir()
    _write_completed_bundle(dispatch, "run42", reward_final_ema=100.0)
    _write_completed_bundle(dispatch, "run43", reward_final_ema=110.0)
    # run44's bundle is missing entirely; job is marked failed in dispatch.json.
    _make_dispatch_json(
        dispatch,
        [
            _job("run42", seed=42),
            _job("run43", seed=43),
            _job(
                "run44",
                seed=44,
                status="failed",
                failure={"kind": "hugin_crash", "message": "RSL-RL subprocess exited 1", "details": {}},
            ),
        ],
    )
    agg = aggregate_dispatch(dispatch)
    row = agg["rows"][0]
    assert row["aggregate"]["n_seeds_completed"] == 2
    assert row["aggregate"]["n_seeds_failed"] == 1
    assert len(agg["failures"]) == 1
    f = agg["failures"][0]
    assert f["seed"] == 44
    assert f["failure_kind"] == "hugin_crash"
    assert f["failure_message"] == "RSL-RL subprocess exited 1"


def test_all_seeds_failed_row_has_null_aggregate(tmp_path: Path):
    dispatch = tmp_path / "20260423-100000"
    dispatch.mkdir()
    _make_dispatch_json(
        dispatch,
        [
            _job(
                f"run{s}",
                seed=s,
                status="failed",
                failure={"kind": "hugin_crash", "message": "boom", "details": {}},
            )
            for s in (42, 43)
        ],
    )
    agg = aggregate_dispatch(dispatch)
    assert len(agg["rows"]) == 1
    row = agg["rows"][0]
    assert row["seeds"] == {}
    assert row["aggregate"] is None
    assert row["divergent_seeds"] == []
    assert len(agg["failures"]) == 2


def test_missing_bundle_synthesizes_missing_bundle_kind(tmp_path: Path):
    dispatch = tmp_path / "20260423-100000"
    dispatch.mkdir()
    # Job marked completed in dispatch.json but no bundle dir.
    _make_dispatch_json(
        dispatch,
        [_job("run42", seed=42, status="completed")],
    )
    agg = aggregate_dispatch(dispatch)
    assert len(agg["failures"]) == 1
    assert agg["failures"][0]["failure_kind"] == "missing_bundle"
    assert len(agg["rows"]) == 1
    assert agg["rows"][0]["seeds"] == {}


def test_malformed_training_json_wrong_schema_version(tmp_path: Path):
    dispatch = tmp_path / "20260423-100000"
    dispatch.mkdir()
    bundle = _write_completed_bundle(dispatch, "run42", reward_final_ema=100.0)
    # Corrupt training.json to schema v2.0 (major bump, rejected).
    with (bundle / "training.json").open("r") as fh:
        t = json.load(fh)
    t["schema_version"] = "2.0"
    with (bundle / "training.json").open("w") as fh:
        json.dump(t, fh)
    _make_dispatch_json(
        dispatch,
        [_job("run42", seed=42, status="completed")],
    )
    agg = aggregate_dispatch(dispatch)
    assert len(agg["failures"]) == 1
    assert agg["failures"][0]["failure_kind"] == "malformed_bundle"


def test_commit_sha_mismatch_majority_wins(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    dispatch = tmp_path / "20260423-100000"
    dispatch.mkdir()
    _write_completed_bundle(dispatch, "run42", reward_final_ema=100.0, commit_sha="aaa111")
    _write_completed_bundle(dispatch, "run43", reward_final_ema=101.0, commit_sha="aaa111")
    _write_completed_bundle(dispatch, "run44", reward_final_ema=102.0, commit_sha="bbb222")
    _make_dispatch_json(
        dispatch,
        [_job(f"run{s}", seed=s) for s in (42, 43, 44)],
    )
    agg = aggregate_dispatch(dispatch)
    assert agg["commit_sha"] == "aaa111"
    captured = capsys.readouterr()
    assert "mixed commit SHAs" in captured.out
    assert "bbb222" in captured.out


def test_empty_dispatch_has_no_rows(tmp_path: Path):
    dispatch = tmp_path / "20260423-100000"
    dispatch.mkdir()
    _make_dispatch_json(dispatch, [])
    agg = aggregate_dispatch(dispatch)
    assert agg["rows"] == []
    assert agg["failures"] == []
    assert agg["totals"]["tasks"] == 0
    assert agg["totals"]["runs"] == 0


def test_missing_dispatch_json_raises(tmp_path: Path):
    dispatch = tmp_path / "20260423-100000"
    dispatch.mkdir()
    with pytest.raises(FileNotFoundError):
        aggregate_dispatch(dispatch)
