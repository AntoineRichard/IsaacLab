# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Probe cycle + K-consecutive-failure flip gate."""

from __future__ import annotations

from dataclasses import dataclass

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
from tools.odin.asgard.heimdall import HeimdallWatcher, _probe_host
from tools.odin.asgard.state import SCHEMA_VERSION, DispatchState
from tools.odin.asgard.transport import SSHResult


@dataclass
class _ScriptedSSH:
    """Returns scripted SSHResults from a per-host queue, defaulting to healthy."""

    scripts: dict[str, list[SSHResult]]
    calls: list[tuple[str, str]]

    def run(self, host, cmd, *, timeout_s=None, stdout_tee=None, pty=True):
        self.calls.append((host.host, cmd))
        q = self.scripts.get(host.host, [])
        if not q:
            return SSHResult(exit_code=0, stdout="GPU 0\n", stderr="", duration_s=0.01)
        return q.pop(0)


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


def _make_watcher(tmp_path, ssh, fleet):
    return HeimdallWatcher(
        fleet=fleet,
        dispatch_dir=tmp_path,
        ssh=ssh,
        state_view=_empty_state,
        probe_interval_s=10000,
        stale_threshold_s=180,
        flip_after_k_failures=2,
        probe_timeout_s=5,
    )


def test_probe_host_success_returns_healthy():
    host = ValkyrieConfig(host="host-a", ssh_user="u")
    ssh = _ScriptedSSH(
        scripts={"host-a": [SSHResult(exit_code=0, stdout="GPU 0\n", stderr="", duration_s=0.01)]},
        calls=[],
    )
    healthy, reason = _probe_host(host, ssh=ssh, timeout_s=5)
    assert healthy is True
    assert reason is None
    assert ssh.calls[0][1].startswith("docker exec")


def test_probe_host_ssh_timeout_returns_unhealthy():
    host = ValkyrieConfig(host="host-a", ssh_user="u")
    ssh = _ScriptedSSH(
        scripts={
            "host-a": [
                SSHResult(exit_code=255, stdout="", stderr="", duration_s=15.0, timed_out=True),
            ]
        },
        calls=[],
    )
    healthy, reason = _probe_host(host, ssh=ssh, timeout_s=5)
    assert healthy is False
    assert reason == "ssh_timeout"


def test_probe_host_empty_stdout_means_nvml_missing():
    host = ValkyrieConfig(host="host-a", ssh_user="u")
    ssh = _ScriptedSSH(
        scripts={"host-a": [SSHResult(exit_code=0, stdout="", stderr="", duration_s=0.5)]},
        calls=[],
    )
    healthy, reason = _probe_host(host, ssh=ssh, timeout_s=5)
    assert healthy is False
    assert reason == "nvml_missing"


def test_watcher_k_failure_gate_does_not_flip_on_first_failure(tmp_path):
    fleet = Fleet(fleet_name="t", hosts=[ValkyrieConfig(host="host-a", ssh_user="u")])
    ssh = _ScriptedSSH(
        scripts={
            "host-a": [
                SSHResult(exit_code=255, stdout="", stderr="", duration_s=15.0, timed_out=True),
            ]
        },
        calls=[],
    )
    w = _make_watcher(tmp_path, ssh, fleet)
    w._tick_once()

    snap = w.latest()
    assert snap is not None
    assert snap.hosts["host-a"].consecutive_failures == 1
    assert snap.hosts["host-a"].healthy is True


def test_watcher_k_failure_gate_flips_on_second_failure(tmp_path):
    fleet = Fleet(fleet_name="t", hosts=[ValkyrieConfig(host="host-a", ssh_user="u")])
    ssh = _ScriptedSSH(
        scripts={
            "host-a": [
                SSHResult(exit_code=255, stdout="", stderr="", duration_s=15.0, timed_out=True),
                SSHResult(exit_code=255, stdout="", stderr="", duration_s=15.0, timed_out=True),
            ]
        },
        calls=[],
    )
    w = _make_watcher(tmp_path, ssh, fleet)
    w._tick_once()
    w._tick_once()

    snap = w.latest()
    assert snap is not None
    assert snap.hosts["host-a"].consecutive_failures == 2
    assert snap.hosts["host-a"].healthy is False
    assert snap.hosts["host-a"].failure_reason == "ssh_timeout"


def test_watcher_success_resets_consecutive_failures(tmp_path):
    fleet = Fleet(fleet_name="t", hosts=[ValkyrieConfig(host="host-a", ssh_user="u")])
    ssh = _ScriptedSSH(
        scripts={
            "host-a": [
                SSHResult(exit_code=255, stdout="", stderr="", duration_s=15.0, timed_out=True),
                SSHResult(exit_code=0, stdout="GPU 0\n", stderr="", duration_s=0.01),
            ]
        },
        calls=[],
    )
    w = _make_watcher(tmp_path, ssh, fleet)
    w._tick_once()
    w._tick_once()

    snap = w.latest()
    assert snap.hosts["host-a"].consecutive_failures == 0
    assert snap.hosts["host-a"].healthy is True


def test_watcher_writes_fleet_json_each_tick(tmp_path):
    fleet = Fleet(fleet_name="t", hosts=[ValkyrieConfig(host="host-a", ssh_user="u")])
    ssh = _ScriptedSSH(
        scripts={"host-a": [SSHResult(exit_code=0, stdout="GPU 0\n", stderr="", duration_s=0.01)]},
        calls=[],
    )
    w = _make_watcher(tmp_path, ssh, fleet)
    w._tick_once()
    assert (tmp_path / "fleet.json").exists()
