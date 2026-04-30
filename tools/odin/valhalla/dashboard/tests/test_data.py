# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :mod:`tools.odin.valhalla.dashboard.data`."""

from __future__ import annotations

import json
from pathlib import Path

from tools.odin.valhalla.dashboard.data import DataLayer, DispatchSummary


def _write_dispatch(
    runs_root: Path,
    dispatch_id: str,
    *,
    jobs: list[dict] | None = None,
    started_at: str = "2026-04-27T14:13:02Z",
    ended_at: str | None = None,
) -> Path:
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


def test_load_dispatch_returns_payload(tmp_path):
    """load_dispatch returns the parsed dispatch.json dict."""
    _write_dispatch(tmp_path, "20260427-141302", ended_at="2026-04-27T14:30:00Z")
    layer = DataLayer(tmp_path)
    payload = layer.load_dispatch("20260427-141302")
    assert payload["dispatch_id"] == "20260427-141302"
    assert payload["ended_at"] == "2026-04-27T14:30:00Z"


def test_load_dispatch_raises_when_missing(tmp_path):
    """Unknown dispatch_id raises FileNotFoundError."""
    import pytest as _pytest

    layer = DataLayer(tmp_path)
    with _pytest.raises(FileNotFoundError):
        layer.load_dispatch("does-not-exist")


def test_load_aggregate_returns_payload(tmp_path):
    d = _write_dispatch(tmp_path, "20260427-141302")
    (d / "aggregate.json").write_text(json.dumps({"schema_version": "1.0", "rows": []}))
    layer = DataLayer(tmp_path)
    payload = layer.load_aggregate("20260427-141302")
    assert payload is not None
    assert payload["schema_version"] == "1.0"


def test_load_aggregate_returns_none_when_missing(tmp_path):
    _write_dispatch(tmp_path, "20260427-141302")
    layer = DataLayer(tmp_path)
    assert layer.load_aggregate("20260427-141302") is None


def test_load_hardware_returns_payload(tmp_path):
    d = _write_dispatch(tmp_path, "20260427-141302")
    (d / "hardware.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dispatch_id": "20260427-141302",
                "fingerprint": "gpu:NVIDIA-L40",
                "hosts": {},
            }
        )
    )
    layer = DataLayer(tmp_path)
    payload = layer.load_hardware("20260427-141302")
    assert payload is not None
    assert payload["fingerprint"] == "gpu:NVIDIA-L40"


def test_load_hardware_returns_none_when_missing(tmp_path):
    _write_dispatch(tmp_path, "20260427-141302")
    layer = DataLayer(tmp_path)
    assert layer.load_hardware("20260427-141302") is None


def _write_bundle(dispatch_dir: Path, run_id: str, *, hardware: dict | None = None) -> Path:
    bundle = dispatch_dir / run_id
    bundle.mkdir(parents=True, exist_ok=True)
    training = {"schema_version": "1.0", "hardware": hardware} if hardware else {"schema_version": "1.0"}
    (bundle / "training.json").write_text(json.dumps(training))
    return bundle


def test_lookup_hardware_returns_first_hit(tmp_path):
    """First newer dispatch wins when both have a bundle for the host."""
    older = _write_dispatch(
        tmp_path,
        "20260424-160119",
        jobs=[{"run_id": "old-r1", "status": "completed", "assigned_to": "v1"}],
    )
    _write_bundle(
        older,
        "old-r1",
        hardware={
            "hostname": "Host-Old",
            "gpu_devices": [{"name": "NVIDIA L40", "mem_gb": 44.32, "compute_cap": "8.9"}],
            "cpu_name": "Xeon",
            "cpu_count": 16,
            "ram_gb": 62.0,
        },
    )
    newer = _write_dispatch(
        tmp_path,
        "20260427-141302",
        jobs=[{"run_id": "new-r1", "status": "completed", "assigned_to": "v1"}],
    )
    _write_bundle(
        newer,
        "new-r1",
        hardware={
            "hostname": "Host-New",
            "gpu_devices": [{"name": "NVIDIA L40", "mem_gb": 44.32, "compute_cap": "8.9"}],
            "cpu_name": "Xeon",
            "cpu_count": 16,
            "ram_gb": 62.79,
        },
    )
    layer = DataLayer(tmp_path)
    info = layer.lookup_hardware("v1")
    assert info is not None
    assert info.hostname == "Host-New"
    assert info.sourced_from == "20260427-141302/new-r1"
    assert info.gpu_devices[0]["name"] == "NVIDIA L40"


def test_lookup_hardware_returns_none_when_unknown(tmp_path):
    """Host that never ran any bundle → None."""
    _write_dispatch(tmp_path, "20260427-141302")
    layer = DataLayer(tmp_path)
    assert layer.lookup_hardware("never-seen") is None


def test_lookup_hardware_skips_bundles_without_hardware_block(tmp_path):
    """training.json without .hardware → skipped; falls through to next."""
    older = _write_dispatch(
        tmp_path,
        "20260424-160119",
        jobs=[{"run_id": "old-r1", "status": "completed", "assigned_to": "v1"}],
    )
    _write_bundle(
        older,
        "old-r1",
        hardware={
            "hostname": "Host-Old",
            "gpu_devices": [],
            "cpu_name": "Xeon",
            "cpu_count": 8,
            "ram_gb": 32.0,
        },
    )
    newer = _write_dispatch(
        tmp_path,
        "20260427-141302",
        jobs=[{"run_id": "new-r1", "status": "completed", "assigned_to": "v1"}],
    )
    _write_bundle(newer, "new-r1", hardware=None)  # no .hardware block

    layer = DataLayer(tmp_path)
    info = layer.lookup_hardware("v1")
    assert info is not None
    # Newer bundle had no hardware → fell through to older.
    assert info.hostname == "Host-Old"
    assert info.sourced_from == "20260424-160119/old-r1"


def _write_aggregate_with_row(dispatch_dir: Path, *, task: str, framework: str, backend: str) -> None:
    payload = {
        "schema_version": "1.0",
        "rows": [{"task": task, "framework": framework, "backend": backend, "seeds": {}}],
    }
    (dispatch_dir / "aggregate.json").write_text(json.dumps(payload))


def _write_hardware(dispatch_dir: Path, fingerprint: str) -> None:
    payload = {
        "schema_version": "1.0",
        "dispatch_id": dispatch_dir.name,
        "fingerprint": fingerprint,
        "hosts": {},
    }
    (dispatch_dir / "hardware.json").write_text(json.dumps(payload))


def test_trend_filters_by_fingerprint(tmp_path):
    """Only dispatches with matching fingerprint appear in the trend."""
    a = _write_dispatch(tmp_path, "20260424-160119")
    _write_aggregate_with_row(a, task="Isaac-Ant-Direct-v0", framework="rsl_rl", backend="physx")
    _write_hardware(a, "gpu:NVIDIA-L40")

    b = _write_dispatch(tmp_path, "20260425-080000")
    _write_aggregate_with_row(b, task="Isaac-Ant-Direct-v0", framework="rsl_rl", backend="physx")
    _write_hardware(b, "gpu:NVIDIA-A100")

    c = _write_dispatch(tmp_path, "20260427-141302")
    _write_aggregate_with_row(c, task="Isaac-Ant-Direct-v0", framework="rsl_rl", backend="physx")
    _write_hardware(c, "gpu:NVIDIA-L40")

    layer = DataLayer(tmp_path)
    ids = layer.trend_dispatches_for("20260427-141302", "Isaac-Ant-Direct-v0", "rsl_rl", "physx", n=10)
    assert ids == ["20260427-141302", "20260424-160119"]


def test_trend_filters_by_task(tmp_path):
    """Dispatches without the requested (task, framework, backend) row are excluded."""
    a = _write_dispatch(tmp_path, "20260424-160119")
    _write_aggregate_with_row(a, task="Isaac-Cartpole-Direct-v0", framework="rsl_rl", backend="physx")
    _write_hardware(a, "gpu:NVIDIA-L40")

    b = _write_dispatch(tmp_path, "20260427-141302")
    _write_aggregate_with_row(b, task="Isaac-Ant-Direct-v0", framework="rsl_rl", backend="physx")
    _write_hardware(b, "gpu:NVIDIA-L40")

    layer = DataLayer(tmp_path)
    ids = layer.trend_dispatches_for("20260427-141302", "Isaac-Ant-Direct-v0", "rsl_rl", "physx", n=10)
    assert ids == ["20260427-141302"]


def test_trend_excludes_pre_feature_dispatches(tmp_path):
    """A dispatch without hardware.json is excluded from trends."""
    a = _write_dispatch(tmp_path, "20260424-160119")
    _write_aggregate_with_row(a, task="Isaac-Ant-Direct-v0", framework="rsl_rl", backend="physx")
    # NO hardware.json

    b = _write_dispatch(tmp_path, "20260427-141302")
    _write_aggregate_with_row(b, task="Isaac-Ant-Direct-v0", framework="rsl_rl", backend="physx")
    _write_hardware(b, "gpu:NVIDIA-L40")

    layer = DataLayer(tmp_path)
    ids = layer.trend_dispatches_for("20260427-141302", "Isaac-Ant-Direct-v0", "rsl_rl", "physx", n=10)
    assert ids == ["20260427-141302"]


def test_trend_returns_empty_when_current_has_no_hardware(tmp_path):
    """If the current dispatch has no hardware.json, trend is empty."""
    a = _write_dispatch(tmp_path, "20260427-141302")
    _write_aggregate_with_row(a, task="Isaac-Ant-Direct-v0", framework="rsl_rl", backend="physx")
    layer = DataLayer(tmp_path)
    ids = layer.trend_dispatches_for("20260427-141302", "Isaac-Ant-Direct-v0", "rsl_rl", "physx")
    assert ids == []


def test_trend_trims_to_n(tmp_path):
    """N=2 → only the two newest matches."""
    for did in ["20260423-000000", "20260424-000000", "20260425-000000", "20260426-000000"]:
        d = _write_dispatch(tmp_path, did)
        _write_aggregate_with_row(d, task="Isaac-Ant-Direct-v0", framework="rsl_rl", backend="physx")
        _write_hardware(d, "gpu:NVIDIA-L40")
    layer = DataLayer(tmp_path)
    ids = layer.trend_dispatches_for("20260426-000000", "Isaac-Ant-Direct-v0", "rsl_rl", "physx", n=2)
    assert ids == ["20260426-000000", "20260425-000000"]


def test_load_training_returns_payload(tmp_path):
    d = _write_dispatch(tmp_path, "20260427-141302")
    _write_bundle(
        d,
        "r1",
        hardware={
            "hostname": "h",
            "gpu_devices": [],
            "cpu_name": "x",
            "cpu_count": 1,
            "ram_gb": 1.0,
        },
    )
    layer = DataLayer(tmp_path)
    payload = layer.load_training("20260427-141302", "r1")
    assert payload is not None
    assert payload["schema_version"] == "1.0"


def test_load_training_returns_none_when_missing(tmp_path):
    _write_dispatch(tmp_path, "20260427-141302")
    layer = DataLayer(tmp_path)
    assert layer.load_training("20260427-141302", "r-missing") is None


def test_load_startup_returns_payload(tmp_path):
    d = _write_dispatch(tmp_path, "20260427-141302")
    bundle = d / "r1"
    bundle.mkdir()
    (bundle / "startup.json").write_text(json.dumps({"schema_version": "1.0", "phases": {}}))
    layer = DataLayer(tmp_path)
    payload = layer.load_startup("20260427-141302", "r1")
    assert payload is not None
    assert payload["schema_version"] == "1.0"


def test_load_startup_returns_none_when_missing(tmp_path):
    _write_dispatch(tmp_path, "20260427-141302")
    layer = DataLayer(tmp_path)
    assert layer.load_startup("20260427-141302", "r-missing") is None


def test_invalidate_is_callable(tmp_path):
    """Smoke test: invalidate() with and without dispatch_id is a no-op
    on disk but must not raise."""
    layer = DataLayer(tmp_path)
    layer.invalidate()
    layer.invalidate("20260427-141302")


def test_read_retry_queue_empty_when_file_missing(tmp_path):
    _write_dispatch(tmp_path, "20260427-141302")
    layer = DataLayer(tmp_path)
    assert layer.read_retry_queue("20260427-141302") == set()


def test_toggle_retry_queue_adds_run_id(tmp_path):
    _write_dispatch(tmp_path, "20260427-141302")
    layer = DataLayer(tmp_path)
    out = layer.toggle_retry_queue("20260427-141302", "rsl-rl_physx_X_seed42")
    assert out == {"rsl-rl_physx_X_seed42"}
    assert layer.read_retry_queue("20260427-141302") == {"rsl-rl_physx_X_seed42"}
    assert (tmp_path / ".retry.sqlite").exists()


def test_toggle_retry_queue_removes_existing(tmp_path):
    _write_dispatch(tmp_path, "20260427-141302")
    layer = DataLayer(tmp_path)
    layer.toggle_retry_queue("20260427-141302", "x")
    out = layer.toggle_retry_queue("20260427-141302", "x")
    assert out == set()
    assert layer.read_retry_queue("20260427-141302") == set()


def test_toggle_retry_queue_multiple_round_trip(tmp_path):
    _write_dispatch(tmp_path, "20260427-141302")
    layer = DataLayer(tmp_path)
    layer.toggle_retry_queue("20260427-141302", "a")
    layer.toggle_retry_queue("20260427-141302", "b")
    assert layer.read_retry_queue("20260427-141302") == {"a", "b"}
    layer.toggle_retry_queue("20260427-141302", "a")
    assert layer.read_retry_queue("20260427-141302") == {"b"}


def test_toggle_retry_queue_does_not_write_legacy_txt(tmp_path):
    """New toggles persist in SQLite and do not rewrite legacy txt files."""
    _write_dispatch(tmp_path, "20260427-141302")
    layer = DataLayer(tmp_path)
    layer.toggle_retry_queue("20260427-141302", "first")
    assert not (tmp_path / "20260427-141302" / "retry_queue.txt").exists()
    assert layer.read_retry_queue("20260427-141302") == {"first"}
