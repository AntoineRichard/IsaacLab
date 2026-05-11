# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HeimdallWatcher start / stop / latest / is_alive — no probing yet."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
from tools.odin.asgard.heimdall import HeimdallWatcher
from tools.odin.asgard.state import SCHEMA_VERSION, DispatchState
from tools.odin.asgard.transport import SSHResult


@dataclass
class _NeverProbedSSH:
    """SSH runner for tests where no probe should fire."""

    calls: list[str]

    def run(self, host, cmd, *, timeout_s=None, stdout_tee=None, pty=True):
        self.calls.append(cmd)
        return SSHResult(exit_code=0, stdout="GPU 0\n", stderr="", duration_s=0.01)


def _empty_state() -> DispatchState:
    return DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id="d",
        started_at="2026-05-08T14:00:00Z",
        ended_at=None,
        seeds=[0],
        commit_sha="x",
        fleet=[],
        jobs=[],
    )


def _make_watcher(tmp_path) -> HeimdallWatcher:
    fleet = Fleet(fleet_name="t", hosts=[ValkyrieConfig(host="host-a", ssh_user="u")])
    ssh = _NeverProbedSSH(calls=[])
    return HeimdallWatcher(
        fleet=fleet,
        dispatch_dir=tmp_path,
        ssh=ssh,
        state_view=_empty_state,
        probe_interval_s=3600,
        stale_threshold_s=180,
    )


def test_watcher_starts_and_stops_cleanly(tmp_path):
    w = _make_watcher(tmp_path)
    w.start()
    assert w.is_alive()
    w.stop(timeout_s=2.0)
    assert not w.is_alive()


def test_watcher_latest_returns_none_before_first_tick(tmp_path):
    w = _make_watcher(tmp_path)
    w.start()
    try:
        assert w.latest() is None
    finally:
        w.stop(timeout_s=2.0)


def test_watcher_double_stop_is_safe(tmp_path):
    w = _make_watcher(tmp_path)
    w.start()
    w.stop(timeout_s=2.0)
    w.stop(timeout_s=2.0)


def test_watcher_start_twice_raises(tmp_path):
    w = _make_watcher(tmp_path)
    w.start()
    try:
        with pytest.raises(RuntimeError):
            w.start()
    finally:
        w.stop(timeout_s=2.0)
