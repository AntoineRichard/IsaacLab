# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for valhalla-adjacent Asgard bootstrap (bring fresh Valkyries up)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tools.odin.asgard.bootstrap import BootstrapResult, bootstrap_valkyrie
from tools.odin.asgard.fleet import ValkyrieConfig

# --- Fakes -----------------------------------------------------------------


@dataclass
class _SSHCall:
    cmd: str
    timeout_s: float | None


@dataclass
class _RsyncCall:
    local_path: Path
    remote_path: str


@dataclass
class _FakeSSH:
    """Records calls; replies with a per-call exit_code lookup.

    Default reply is exit_code=0. Override by setting ``replies[key]`` where
    ``key`` is a substring that must appear in the cmd. First match wins;
    check order follows insertion order.
    """

    calls: list[_SSHCall] = field(default_factory=list)
    replies: dict[str, int] = field(default_factory=dict)
    reply_stdout: dict[str, str] = field(default_factory=dict)
    reply_stderr: dict[str, str] = field(default_factory=dict)

    def run(self, host, cmd, *, timeout_s=None, stdout_tee=None):
        self.calls.append(_SSHCall(cmd=cmd, timeout_s=timeout_s))
        exit_code = 0
        stdout = ""
        stderr = ""
        for key, code in self.replies.items():
            if key in cmd:
                exit_code = code
                stdout = self.reply_stdout.get(key, "")
                stderr = self.reply_stderr.get(key, "")
                break

        class R:
            pass

        R.exit_code = exit_code
        R.stdout = stdout
        R.stderr = stderr
        R.duration_s = 0.01
        return R()


@dataclass
class _FakeRsync:
    calls: list[_RsyncCall] = field(default_factory=list)
    exit_code: int = 0
    stderr: str = ""

    def push(self, host, local_path, remote_path):
        self.calls.append(_RsyncCall(local_path=Path(local_path), remote_path=str(remote_path)))

        class R:
            pass

        R.exit_code = self.exit_code
        R.stdout = ""
        R.stderr = self.stderr
        R.duration_s = 0.01
        return R()

    def pull(self, host, remote_path, local_path):
        raise AssertionError("bootstrap_valkyrie must not call rsync.pull")


def _host() -> ValkyrieConfig:
    return ValkyrieConfig(
        host="v1.internal",
        ssh_user="odin",
        ssh_key=None,
        isaaclab_path="/opt/IsaacLab",
        container_name="isaac-lab-base",
    )


# --- Tests -----------------------------------------------------------------


def test_bootstrap_valkyrie_happy_path(tmp_path: Path):
    ssh = _FakeSSH(
        replies={"docker inspect": 0},
        reply_stdout={"docker inspect": "running"},
    )
    rsync = _FakeRsync()
    result = bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync)
    assert isinstance(result, BootstrapResult)
    assert result.ok is True
    assert result.host == "v1.internal"
    assert set(result.step_durations_s.keys()) == {"wipe", "rsync", "container_start", "container_verify"}
    assert all(d >= 0.0 for d in result.step_durations_s.values())


def test_bootstrap_valkyrie_ssh_unreachable(tmp_path: Path):
    ssh = _FakeSSH(replies={"echo bootstrap-ok": 255})
    rsync = _FakeRsync()
    result = bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync)
    assert result.ok is False
    assert "ssh unreachable" in result.message
    assert rsync.calls == [], "rsync.push must not run when ssh is down"


def test_bootstrap_valkyrie_docker_daemon_down(tmp_path: Path):
    ssh = _FakeSSH(
        replies={"docker ps": 1},
        reply_stderr={"docker ps": "Cannot connect to the Docker daemon"},
    )
    rsync = _FakeRsync()
    result = bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync)
    assert result.ok is False
    assert "docker daemon" in result.message
    assert rsync.calls == []


def test_bootstrap_valkyrie_rsync_failure(tmp_path: Path):
    ssh = _FakeSSH()  # all ssh ok
    rsync = _FakeRsync(exit_code=23, stderr="send_files failed")
    result = bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync)
    assert result.ok is False
    assert "rsync push failed" in result.message
    # Wipe ran; rsync ran; container_start did NOT.
    assert any("rm -rf" in c.cmd for c in ssh.calls)
    assert len(rsync.calls) == 1
    assert not any("container.py start" in c.cmd for c in ssh.calls)


def test_bootstrap_valkyrie_container_start_failure(tmp_path: Path):
    ssh = _FakeSSH(
        replies={"container.py start": 1},
        reply_stderr={"container.py start": "timeout"},
    )
    rsync = _FakeRsync()
    result = bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync)
    assert result.ok is False
    assert "container.py start" in result.message or "container start failed" in result.message
    assert not any("docker inspect" in c.cmd for c in ssh.calls), "verify must not run when start failed"


def test_bootstrap_valkyrie_container_not_running_after_start(tmp_path: Path):
    ssh = _FakeSSH(
        replies={"docker inspect": 0},
        reply_stdout={"docker inspect": "exited"},
    )
    rsync = _FakeRsync()
    result = bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync)
    assert result.ok is False
    assert "not running" in result.message
    assert "'exited'" in result.message


def test_bootstrap_valkyrie_build_timeout_passed_through(tmp_path: Path):
    ssh = _FakeSSH(
        replies={"docker inspect": 0},
        reply_stdout={"docker inspect": "running"},
    )
    rsync = _FakeRsync()
    bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync, build_timeout_s=1800)
    start_calls = [c for c in ssh.calls if "container.py start" in c.cmd]
    assert len(start_calls) == 1
    assert start_calls[0].timeout_s == 1800


def test_bootstrap_valkyrie_wipe_always_runs(tmp_path: Path):
    ssh = _FakeSSH(
        replies={"docker inspect": 0},
        reply_stdout={"docker inspect": "running"},
    )
    rsync = _FakeRsync()
    bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync)
    wipe_calls = [c for c in ssh.calls if "rm -rf" in c.cmd and "/opt/IsaacLab" in c.cmd]
    assert len(wipe_calls) == 1


def test_bootstrap_valkyrie_wipe_failure(tmp_path: Path):
    """Wipe-step failure (e.g. permission issue) halts the pipeline before rsync."""
    ssh = _FakeSSH(
        replies={"rm -rf": 1},
        reply_stderr={"rm -rf": "Permission denied"},
    )
    rsync = _FakeRsync()
    result = bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync)
    assert result.ok is False
    assert "failed to wipe" in result.message
    assert "/opt/IsaacLab" in result.message
    assert rsync.calls == [], "rsync.push must not run when wipe failed"


def test_bootstrap_valkyrie_container_inspect_fails_surfaces_stderr(tmp_path: Path):
    """When docker inspect returns non-zero (e.g. no such container), stderr must surface."""
    ssh = _FakeSSH(
        replies={"docker inspect": 1},
        reply_stderr={"docker inspect": "Error: No such object: isaac-lab-base"},
    )
    rsync = _FakeRsync()
    result = bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync)
    assert result.ok is False
    assert "docker inspect failed" in result.message
    assert "No such object" in result.message
