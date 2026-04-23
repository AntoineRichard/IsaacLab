# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :func:`tools.odin.asgard.preflight.preflight_valkyrie`."""

from __future__ import annotations

from dataclasses import dataclass

from tools.odin.asgard.fleet import ValkyrieConfig
from tools.odin.asgard.preflight import PreflightResult, preflight_valkyrie
from tools.odin.asgard.transport import SSHResult


@dataclass
class _FakeSSH:
    """Deterministic SSH runner: returns scripted SSHResult for each cmd substring match."""

    scripted: dict  # {cmd_substring: SSHResult}

    def run(self, host, cmd: str, *, timeout_s=None, stdout_tee=None) -> SSHResult:
        for key, result in self.scripted.items():
            if key in cmd:
                return result
        return SSHResult(exit_code=1, stdout="", stderr=f"no fake for {cmd!r}", duration_s=0.0)


def _host() -> ValkyrieConfig:
    return ValkyrieConfig(host="v1", ssh_user="odin")


def _ok() -> SSHResult:
    return SSHResult(exit_code=0, stdout="ok\n", stderr="", duration_s=0.01)


def _fail(msg: str) -> SSHResult:
    return SSHResult(exit_code=1, stdout="", stderr=msg, duration_s=0.01)


def test_all_checks_pass():
    ssh = _FakeSSH(
        scripted={
            "echo preflight-ok": _ok(),
            "docker ps": _ok(),
            "docker inspect": SSHResult(exit_code=0, stdout="running\n", stderr="", duration_s=0.01),
            "test -d": _ok(),
        }
    )
    r = preflight_valkyrie(_host(), ssh=ssh)
    assert isinstance(r, PreflightResult)
    assert r.ok is True
    assert r.checks == {
        "ssh_reach": True,
        "docker_running": True,
        "container_up": True,
        "isaaclab_present": True,
    }


def test_ssh_unreachable():
    ssh = _FakeSSH(scripted={"echo preflight-ok": _fail("connection refused")})
    r = preflight_valkyrie(_host(), ssh=ssh)
    assert r.ok is False
    assert r.checks["ssh_reach"] is False
    # Downstream checks should NOT be run / should be False.
    assert r.checks["docker_running"] is False
    assert "connection refused" in r.message


def test_docker_down():
    ssh = _FakeSSH(
        scripted={
            "echo preflight-ok": _ok(),
            "docker ps": _fail("docker daemon not responding"),
        }
    )
    r = preflight_valkyrie(_host(), ssh=ssh)
    assert r.ok is False
    assert r.checks["ssh_reach"] is True
    assert r.checks["docker_running"] is False
    assert "docker" in r.message.lower()


def test_container_down():
    ssh = _FakeSSH(
        scripted={
            "echo preflight-ok": _ok(),
            "docker ps": _ok(),
            "docker inspect": SSHResult(exit_code=0, stdout="exited\n", stderr="", duration_s=0.01),
            "test -d": _ok(),
        }
    )
    r = preflight_valkyrie(_host(), ssh=ssh)
    assert r.ok is False
    assert r.checks["container_up"] is False
    assert "container" in r.message.lower()


def test_isaaclab_missing():
    ssh = _FakeSSH(
        scripted={
            "echo preflight-ok": _ok(),
            "docker ps": _ok(),
            "docker inspect": SSHResult(exit_code=0, stdout="running\n", stderr="", duration_s=0.01),
            "test -d": _fail("no such directory"),
        }
    )
    r = preflight_valkyrie(_host(), ssh=ssh)
    assert r.ok is False
    assert r.checks["isaaclab_present"] is False
