# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :func:`tools.odin.asgard.recovery.recover_valkyrie_gpu`."""

from __future__ import annotations

from dataclasses import dataclass

from tools.odin.asgard.fleet import ValkyrieConfig
from tools.odin.asgard.recovery import RecoveryResult, recover_valkyrie_gpu
from tools.odin.asgard.transport import SSHResult


@dataclass
class _ScriptedSSH:
    """Replays a list of SSHResult in call order; raises if exhausted.

    Each call records the (host, cmd) pair into ``calls`` for assertion.
    """

    responses: list  # list[SSHResult]
    calls: list = None  # list[tuple[str, str]]

    def __post_init__(self):
        if self.calls is None:
            self.calls = []

    def run(self, host, cmd: str, *, timeout_s=None, stdout_tee=None, pty=True) -> SSHResult:
        self.calls.append((host.host, cmd))
        if not self.responses:
            return SSHResult(exit_code=255, stdout="", stderr="ssh script exhausted", duration_s=0.0)
        return self.responses.pop(0)


def _host() -> ValkyrieConfig:
    return ValkyrieConfig(host="v1", ssh_user="horde", container_name="isaac-lab-base")


def _ok(stdout: str = "ok\n") -> SSHResult:
    return SSHResult(exit_code=0, stdout=stdout, stderr="", duration_s=0.01)


def _fail(stderr: str, exit_code: int = 1) -> SSHResult:
    return SSHResult(exit_code=exit_code, stdout="", stderr=stderr, duration_s=0.01)


def test_recovery_happy_path():
    ssh = _ScriptedSSH(
        responses=[
            _ok("isaac-lab-base\n"),  # docker restart
            _ok("running\n"),  # docker inspect (1st poll)
            _ok("GPU 0: NVIDIA A100 (UUID: GPU-abc...)\n"),  # nvidia-smi -L
        ]
    )
    r = recover_valkyrie_gpu(_host(), ssh=ssh)
    assert isinstance(r, RecoveryResult)
    assert r.attempted is True
    assert r.recovered is True
    assert r.host == "v1"
    assert r.container_name == "isaac-lab-base"
    assert "recovered_via_container_restart" in r.message
    assert r.details["docker_restart"] == "ok"
    assert r.details["gpu_probe"] == "ok"


def test_recovery_docker_restart_fails():
    ssh = _ScriptedSSH(responses=[_fail("Error response from daemon: container not running")])
    r = recover_valkyrie_gpu(_host(), ssh=ssh)
    assert r.attempted is True
    assert r.recovered is False
    assert r.message.startswith("docker_restart_failed")
    assert "docker_restart" in r.details
    # Subsequent phases not called.
    assert "container_up" not in r.details
    assert "gpu_probe" not in r.details


def test_recovery_container_never_running(monkeypatch):
    monkeypatch.setattr("tools.odin.asgard.recovery.time.sleep", lambda _: None)
    # docker restart succeeds; inspect returns "created" forever.
    responses = [_ok("isaac-lab-base\n")] + [_ok("created\n")] * 20
    ssh = _ScriptedSSH(responses=responses)
    r = recover_valkyrie_gpu(_host(), ssh=ssh)
    assert r.recovered is False
    assert r.message == "container_not_running_after_restart"
    assert r.details["container_up"] == "timeout"
    assert "gpu_probe" not in r.details


def test_recovery_gpu_probe_empty():
    # docker restart ok, container running, but nvidia-smi -L returns empty stdout.
    ssh = _ScriptedSSH(
        responses=[
            _ok("isaac-lab-base\n"),
            _ok("running\n"),
            SSHResult(exit_code=0, stdout="", stderr="", duration_s=0.01),
        ]
    )
    r = recover_valkyrie_gpu(_host(), ssh=ssh)
    assert r.recovered is False
    assert r.message.startswith("gpu_probe_failed")


def test_recovery_ssh_unreachable():
    ssh = _ScriptedSSH(responses=[SSHResult(exit_code=255, stdout="", stderr="connection refused", duration_s=0.0)])
    r = recover_valkyrie_gpu(_host(), ssh=ssh)
    # ssh_unreachable is a special case: the FIRST SSH call (docker restart)
    # came back with exit 255, which we treat as an SSH-layer failure rather
    # than a docker-layer failure. attempted stays True (we tried), but the
    # detail differs from docker_restart_failed.
    assert r.recovered is False
    assert r.message == "ssh_unreachable"
