# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :mod:`tools.odin.asgard.tracker`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.odin.asgard.tracker import (
    TRACKER_FILENAME,
    TRACKER_SCHEMA_VERSION,
    Tracker,
    read_tracker,
    validate_tracker_payload,
    write_tracker,
)


def _payload(**overrides) -> dict:
    payload = {
        "schema_version": TRACKER_SCHEMA_VERSION,
        "run_id": "rsl-rl_physx_X_seed42",
        "container_name": "isaac-lab-base",
        "host": "10.59.114.176",
        "submitted_at": "2026-04-30T11:05:34Z",
        "pid": 12345,
        "container_pid": None,
        "per_job_timeout_s": 43200,
    }
    payload.update(overrides)
    return payload


def test_tracker_round_trip(tmp_path: Path):
    bundle = tmp_path / "rsl-rl_physx_X_seed42"
    bundle.mkdir()
    tracker = Tracker(
        run_id="rsl-rl_physx_X_seed42",
        container_name="isaac-lab-base",
        host="10.59.114.176",
        submitted_at="2026-04-30T11:05:34Z",
        pid=12345,
        per_job_timeout_s=43200,
    )
    write_tracker(bundle, tracker)
    assert (bundle / TRACKER_FILENAME).exists()
    loaded = read_tracker(bundle)
    assert loaded == tracker


def test_read_tracker_returns_none_when_missing(tmp_path: Path):
    bundle = tmp_path / "no-tracker"
    bundle.mkdir()
    assert read_tracker(bundle) is None


def test_validate_tracker_rejects_missing_required_fields():
    bad = _payload()
    bad.pop("run_id")
    with pytest.raises(ValueError, match="run_id"):
        validate_tracker_payload(bad)


def test_validate_tracker_rejects_unknown_schema_major():
    with pytest.raises(ValueError, match="schema"):
        validate_tracker_payload(_payload(schema_version="2.0"))


def test_validate_tracker_accepts_minor_schema_bump():
    """Additive minor-version bumps must be tolerated by readers."""
    validate_tracker_payload(_payload(schema_version="1.99"))


def test_validate_tracker_rejects_non_int_pid():
    with pytest.raises(ValueError, match="pid"):
        validate_tracker_payload(_payload(pid="abc"))


def test_read_tracker_raises_on_invalid_json(tmp_path: Path):
    bundle = tmp_path / "bad-json"
    bundle.mkdir()
    (bundle / TRACKER_FILENAME).write_text("{not json")
    with pytest.raises(ValueError, match="JSON"):
        read_tracker(bundle)


def test_write_tracker_emits_pretty_json(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    write_tracker(
        bundle,
        Tracker(
            run_id="r",
            container_name="c",
            host="h",
            submitted_at="2026-04-30T00:00:00Z",
            pid=1,
            per_job_timeout_s=100,
        ),
    )
    raw = (bundle / TRACKER_FILENAME).read_text()
    payload = json.loads(raw)
    assert payload["schema_version"] == TRACKER_SCHEMA_VERSION
    assert payload["run_id"] == "r"
    assert payload["pid"] == 1
    assert payload["container_pid"] is None
