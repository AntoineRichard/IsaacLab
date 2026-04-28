# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :mod:`tools.odin.valhalla.dashboard.data`."""

from __future__ import annotations

import json
from pathlib import Path

from tools.odin.valhalla.dashboard.data import DataLayer, DispatchSummary


def _write_dispatch(runs_root: Path, dispatch_id: str, *, jobs: list[dict] | None = None,
                    started_at: str = "2026-04-27T14:13:02Z", ended_at: str | None = None) -> Path:
    d = runs_root / dispatch_id
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.3",
        "dispatch_id": dispatch_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "seeds": [42],
        "commit_sha": "abc123",
        "fleet": [{"host": "v1", "status": "idle", "current_run_id": None, "last_error": None}],
        "jobs": jobs or [],
        "skipped": [],
    }
    (d / "dispatch.json").write_text(json.dumps(payload))
    return d


def test_list_dispatches_empty_runs_root(tmp_path):
    """Non-existent or empty runs_root returns []."""
    layer = DataLayer(tmp_path / "missing")
    assert layer.list_dispatches() == []

    (tmp_path / "empty").mkdir()
    layer = DataLayer(tmp_path / "empty")
    assert layer.list_dispatches() == []


def test_list_dispatches_returns_summary(tmp_path):
    """One dispatch with one completed job → one DispatchSummary."""
    _write_dispatch(
        tmp_path,
        "20260427-141302",
        ended_at="2026-04-27T14:30:00Z",
        jobs=[{"run_id": "r1", "status": "completed", "assigned_to": "v1"}],
    )
    layer = DataLayer(tmp_path)
    result = layer.list_dispatches()
    assert len(result) == 1
    s = result[0]
    assert isinstance(s, DispatchSummary)
    assert s.dispatch_id == "20260427-141302"
    assert s.started_at == "2026-04-27T14:13:02Z"
    assert s.ended_at == "2026-04-27T14:30:00Z"
    assert s.jobs_total == 1
    assert s.jobs_completed == 1
    assert s.jobs_failed == 0
    assert s.jobs_pending == 0
    assert s.skipped_total == 0
    assert s.hostnames == ["v1"]


def test_list_dispatches_excludes_loose_bundles(tmp_path):
    """Loose pre-T3.1 bundles (rsl-rl_..._seed42) are filtered out."""
    _write_dispatch(tmp_path, "20260427-141302")
    loose = tmp_path / "rsl-rl_physx_Isaac-Ant-Direct-v0_20260423-114242_seed42"
    loose.mkdir()
    (loose / "manifest.json").write_text("{}")
    layer = DataLayer(tmp_path)
    ids = [s.dispatch_id for s in layer.list_dispatches()]
    assert ids == ["20260427-141302"]


def test_list_dispatches_sorted_newest_first(tmp_path):
    """Three dispatches sorted descending by directory name."""
    _write_dispatch(tmp_path, "20260424-160119")
    _write_dispatch(tmp_path, "20260427-141302")
    _write_dispatch(tmp_path, "20260425-080000")
    layer = DataLayer(tmp_path)
    ids = [s.dispatch_id for s in layer.list_dispatches()]
    assert ids == ["20260427-141302", "20260425-080000", "20260424-160119"]


def test_list_dispatches_skips_dirs_without_dispatch_json(tmp_path):
    """A timestamp-named dir with no dispatch.json is skipped."""
    _write_dispatch(tmp_path, "20260427-141302")
    (tmp_path / "20260428-000000").mkdir()  # no dispatch.json
    layer = DataLayer(tmp_path)
    ids = [s.dispatch_id for s in layer.list_dispatches()]
    assert ids == ["20260427-141302"]


def test_list_dispatches_counts_skipped_jobs(tmp_path):
    """`skipped` array contributes to skipped_total."""
    d = _write_dispatch(tmp_path, "20260427-141302")
    payload = json.loads((d / "dispatch.json").read_text())
    payload["skipped"] = [
        {"task_id": "X", "framework": "rsl_rl", "backend": "physx", "seed": 42, "reason": "preset_unsupported"},
        {"task_id": "Y", "framework": "rsl_rl", "backend": "physx", "seed": 42, "reason": "native_backend_mismatch"},
    ]
    (d / "dispatch.json").write_text(json.dumps(payload))
    layer = DataLayer(tmp_path)
    s = layer.list_dispatches()[0]
    assert s.skipped_total == 2
