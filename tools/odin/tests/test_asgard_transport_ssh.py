# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :class:`tools.odin.asgard.transport.ShellSSHRunner` (mock subprocess)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.odin.asgard.fleet import ValkyrieConfig
from tools.odin.asgard.transport import ShellSSHRunner, SSHResult


def _host() -> ValkyrieConfig:
    return ValkyrieConfig(host="valkyrie-01", ssh_user="odin", ssh_key=None)


def test_build_ssh_command_minimal():
    runner = ShellSSHRunner()
    host = _host()
    argv = runner._build_ssh_argv(host, 'echo "hi"', timeout_s=None)
    # First arg is "ssh"; there should be a user@host target and a trailing cmd string.
    assert argv[0] == "ssh"
    assert f"{host.ssh_user}@{host.host}" in argv
    assert argv[-1] == 'echo "hi"'
    # Bake-in options.
    assert any("StrictHostKeyChecking=accept-new" in a for a in argv)
    assert any("ServerAliveInterval=30" in a for a in argv)
    assert any("ConnectTimeout=10" in a for a in argv)


def test_build_ssh_command_with_key(tmp_path: Path):
    key = tmp_path / "fake_key"
    key.write_text("nope")
    host = ValkyrieConfig(host="h1", ssh_user="u", ssh_key=key)
    runner = ShellSSHRunner()
    argv = runner._build_ssh_argv(host, "true", timeout_s=None)
    # -i <key> pair present.
    i = argv.index("-i")
    assert Path(argv[i + 1]) == key


def test_run_happy_path(monkeypatch):
    """Runner returns SSHResult with exit_code 0, stdout, stderr."""
    captured: dict = {}

    class _FakePopen:
        def __init__(self, argv, **kw):
            captured["argv"] = argv
            captured["kwargs"] = kw
            self.returncode = 0
            self.stdout_lines = iter(["hello\n", "world\n"])
            self.stderr_lines = iter([])

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            pass

        def kill(self):
            pass

        def communicate(self, timeout=None):
            return ("hello\nworld\n", "")

        @property
        def stdout(self):
            return _FakeStream(["hello\n", "world\n"])

        @property
        def stderr(self):
            return _FakeStream([])

    class _FakeStream:
        def __init__(self, lines):
            self._lines = list(lines)

        def readline(self):
            if not self._lines:
                return ""
            return self._lines.pop(0)

        def close(self):
            pass

    monkeypatch.setattr(subprocess, "Popen", _FakePopen)

    runner = ShellSSHRunner()
    result = runner.run(_host(), "echo hello", timeout_s=None)
    assert isinstance(result, SSHResult)
    assert result.exit_code == 0
    assert "hello" in result.stdout
    assert not result.timed_out


def test_run_writes_tee_file(tmp_path: Path, monkeypatch):
    class _FakeStream:
        def __init__(self, lines):
            self._lines = list(lines)

        def readline(self):
            if not self._lines:
                return ""
            return self._lines.pop(0)

        def close(self):
            pass

    class _FakePopen:
        def __init__(self, argv, **kw):
            self.returncode = 0
            self.stdout = _FakeStream(["line-a\n", "line-b\n"])
            self.stderr = _FakeStream([])

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            pass

        def kill(self):
            pass

        def communicate(self, timeout=None):
            return ("", "")

    monkeypatch.setattr(subprocess, "Popen", _FakePopen)

    tee_path = tmp_path / "ssh-tail.log"
    runner = ShellSSHRunner()
    runner.run(_host(), "cmd", timeout_s=None, stdout_tee=tee_path)

    content = tee_path.read_text()
    assert "line-a" in content
    assert "line-b" in content


def test_run_timeout_terminates(monkeypatch):
    """When wait() raises TimeoutExpired, runner calls terminate and reports timed_out=True."""

    class _FakeStream:
        def readline(self):
            return ""

        def close(self):
            pass

    class _FakePopen:
        def __init__(self, argv, **kw):
            self.returncode = None
            self.stdout = _FakeStream()
            self.stderr = _FakeStream()
            self._terminated = False

        def wait(self, timeout=None):
            if not self._terminated:
                raise subprocess.TimeoutExpired(cmd="ssh", timeout=timeout or 0)
            return -15

        def terminate(self):
            self._terminated = True
            self.returncode = -15

        def kill(self):
            self._terminated = True
            self.returncode = -9

        def communicate(self, timeout=None):
            return ("", "")

    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    runner = ShellSSHRunner()
    result = runner.run(_host(), "cmd", timeout_s=0.1)
    assert result.timed_out is True
    assert result.exit_code != 0
