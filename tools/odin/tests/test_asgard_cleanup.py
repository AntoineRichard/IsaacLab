# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :mod:`tools.odin.asgard.cleanup`."""

from __future__ import annotations

from tools.odin.asgard.cleanup import SweepResult, sweep_orphan_trainers
from tools.odin.asgard.fleet import ValkyrieConfig
from tools.odin.asgard.transport import SSHResult


class _FakeSSH:
    def __init__(self, *, exit_code: int = 0, stdout: str = "", stderr: str = ""):
        self.calls: list[tuple] = []
        self._exit_code = exit_code
        self._stdout = stdout
        self._stderr = stderr

    def run(self, host, cmd, *, timeout_s=None, stdout_tee=None, pty=True):
        self.calls.append((host.host, cmd, timeout_s, pty))
        return SSHResult(exit_code=self._exit_code, stdout=self._stdout, stderr=self._stderr, duration_s=0.01)


def _host() -> ValkyrieConfig:
    return ValkyrieConfig(host="v1", ssh_user="odin", container_name="isaac-lab-base")


def test_sweep_orphan_trainers_uses_pkill_in_container():
    """The sweep must execute ``pkill`` inside the host's container — not on
    the bare host — so that orphan trainers from a prior dispatch get
    SIGKILLed before the next worker assigns a job here."""
    ssh = _FakeSSH(stdout="3\n")  # pkill prints kill count
    result = sweep_orphan_trainers(_host(), ssh=ssh)
    assert isinstance(result, SweepResult)
    assert result.host == "v1"
    assert result.attempted is True
    assert len(ssh.calls) == 1
    _, cmd, _, _ = ssh.calls[0]
    assert "docker exec isaac-lab-base" in cmd
    assert "pkill" in cmd


def test_sweep_orphan_trainers_pattern_targets_benchmark_scripts():
    """The sweep targets ``benchmark_rsl_rl`` and ``benchmark_skrl`` — the two
    Hugin-launched python scripts that hold the GPU. A pattern wider than
    that risks killing unrelated container processes; narrower would miss
    one of the two trainers we actually use."""
    ssh = _FakeSSH()
    sweep_orphan_trainers(_host(), ssh=ssh)
    _, cmd, _, _ = ssh.calls[0]
    assert "benchmark_rsl_rl" in cmd
    assert "benchmark_skrl" in cmd


def test_sweep_orphan_trainers_uses_pty_false_for_short_call():
    """No PTY: this is a fire-and-forget hygiene call. ``-tt`` would force
    the local SSH client to stay attached, defeating the point."""
    ssh = _FakeSSH()
    sweep_orphan_trainers(_host(), ssh=ssh)
    _, _, _, pty = ssh.calls[0]
    assert pty is False


def test_sweep_orphan_trainers_swallows_pkill_no_match_exit_code():
    """``pkill`` returns 1 when no processes match — that's the steady-state
    case (no zombies). The sweep must NOT report this as a failure: the
    point is "make sure the host is clean before assigning a job", and a
    host with zero trainers IS clean."""
    ssh = _FakeSSH(exit_code=1, stdout="")
    result = sweep_orphan_trainers(_host(), ssh=ssh)
    assert result.killed_count == 0
    # exit 1 from pkill-no-match is benign; result.ok is True.
    assert result.ok is True


def test_sweep_orphan_trainers_reports_killed_count_when_pattern_matched():
    """When pkill kills processes, the sweep records the count for the
    dispatch summary (operators want to know "was this host wedged?")."""
    # We use ``-c`` which makes pkill print the matched-process count to stdout.
    ssh = _FakeSSH(exit_code=0, stdout="3\n")
    result = sweep_orphan_trainers(_host(), ssh=ssh)
    assert result.ok is True
    assert result.killed_count == 3


def test_sweep_orphan_trainers_marks_failure_on_ssh_error():
    """SSH error (exit 255) is not an empty fleet — surface it so the runner
    can decide whether to skip the host. Don't mask infrastructure failures
    behind the swallow-no-match logic."""
    ssh = _FakeSSH(exit_code=255, stderr="ssh: connect to host v1 port 22: Connection refused")
    result = sweep_orphan_trainers(_host(), ssh=ssh)
    assert result.ok is False
    assert "Connection refused" in (result.message or "")
