# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for tab_a.filters.filter_jobs."""

from __future__ import annotations

from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.filters import filter_jobs


def _job(*, task="Isaac-Ant-Direct-v0", status="completed", kind=None, run_id="r"):
    j = {
        "run_id": run_id,
        "task_id": task,
        "framework": "rsl_rl",
        "backend": "physx",
        "seed": 42,
        "status": status,
        "assigned_to": "v1",
        "attempts": 1,
        "failure": None,
    }
    if status == "failed" and kind is not None:
        j["failure"] = {"kind": kind, "message": "x", "details": {}}
    return j


def test_no_filters_returns_all():
    jobs = [_job(run_id=f"r{i}") for i in range(3)]
    assert filter_jobs(jobs) == jobs


def test_status_filter_single():
    jobs = [
        _job(status="completed", run_id="c"),
        _job(status="failed", kind="hugin_crash", run_id="f"),
    ]
    out = filter_jobs(jobs, status_filter=["failed"])
    assert [j["run_id"] for j in out] == ["f"]


def test_status_filter_multi():
    jobs = [
        _job(status="completed", run_id="c"),
        _job(status="failed", kind="hugin_crash", run_id="f"),
        _job(status="pending", run_id="p"),
    ]
    out = filter_jobs(jobs, status_filter=["completed", "failed"])
    assert sorted(j["run_id"] for j in out) == ["c", "f"]


def test_kind_filter_passes_failed_only():
    jobs = [
        _job(status="completed", run_id="c"),
        _job(status="failed", kind="hugin_crash", run_id="f"),
        _job(status="failed", kind="gpu_lost", run_id="g"),
    ]
    out = filter_jobs(jobs, kind_filter=["hugin_crash"])
    assert [j["run_id"] for j in out] == ["f"]


def test_task_text_substring():
    jobs = [
        _job(task="Isaac-Ant-Direct-v0", run_id="ant"),
        _job(task="Isaac-Velocity-Flat-Anymal-C-Direct-v0", run_id="anymal"),
        _job(task="Isaac-Cartpole-Direct-v0", run_id="cart"),
    ]
    # "an" matches both "Ant" and "Anymal"; "Cartpole" has no "an".
    out = filter_jobs(jobs, task_text="an")
    assert sorted(j["run_id"] for j in out) == ["ant", "anymal"]


def test_task_text_empty_string():
    jobs = [_job(task="Isaac-Ant-Direct-v0", run_id="a"), _job(task="Cartpole", run_id="b")]
    out = filter_jobs(jobs, task_text="")
    assert sorted(j["run_id"] for j in out) == ["a", "b"]


def test_task_text_case_insensitive():
    jobs = [
        _job(task="Isaac-Ant-Direct-v0", run_id="ant"),
        _job(task="Isaac-Velocity-Flat-Anymal-C-Direct-v0", run_id="anymal"),
    ]
    out = filter_jobs(jobs, task_text="AN")
    assert sorted(j["run_id"] for j in out) == ["ant", "anymal"]


def test_combined_filters_intersect():
    jobs = [
        _job(task="Isaac-Ant-Direct-v0", status="completed", run_id="ac"),
        _job(task="Isaac-Ant-Direct-v0", status="failed", kind="hugin_crash", run_id="af"),
        _job(task="Isaac-Velocity-Flat-Anymal-C-Direct-v0", status="failed", kind="hugin_crash", run_id="amf"),
        _job(task="Isaac-Cartpole-Direct-v0", status="failed", kind="hugin_crash", run_id="cf"),
    ]
    out = filter_jobs(
        jobs,
        status_filter=["failed"],
        kind_filter=["hugin_crash"],
        task_text="an",
    )
    assert sorted(j["run_id"] for j in out) == ["af", "amf"]
