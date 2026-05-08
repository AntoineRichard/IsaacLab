# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for Heimdall public dataclasses."""

from __future__ import annotations

from tools.odin.asgard.heimdall import HeimdallSnapshot, HostHealth, StaleJob


def test_host_health_default_history_is_empty():
    h = HostHealth(
        name="host-a",
        healthy=True,
        last_probe_at="2026-05-08T14:32:18Z",
        consecutive_failures=0,
        failure_reason=None,
        recovery_attempts=0,
        recovery_history=[],
        quarantined=False,
    )
    assert h.recovery_history == []
    assert h.healthy is True


def test_stale_job_carries_host_health_branch():
    sj = StaleJob(
        run_id="run-1",
        host="host-a",
        last_heartbeat_at="2026-05-08T14:30:00Z",
        age_seconds=240.0,
        host_was_healthy=True,
    )
    assert sj.host_was_healthy is True


def test_heimdall_snapshot_construction():
    snap = HeimdallSnapshot(
        generated_at="2026-05-08T14:32:18Z",
        hosts={},
        stale_jobs=[],
        recent_events=[],
    )
    assert snap.hosts == {}
    assert snap.stale_jobs == []
    assert snap.recent_events == []
