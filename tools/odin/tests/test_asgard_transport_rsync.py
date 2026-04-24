# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :class:`tools.odin.asgard.transport.ShellRsyncRunner`."""

from __future__ import annotations

import subprocess
from pathlib import Path

from tools.odin.asgard.fleet import ValkyrieConfig
from tools.odin.asgard.transport import RsyncResult, ShellRsyncRunner


def _host(ssh_key: Path | None = None) -> ValkyrieConfig:
    return ValkyrieConfig(host="v1", ssh_user="odin", ssh_key=ssh_key)


def _fake_completed(captured: dict):
    def _run(argv, **kw):
        captured["argv"] = argv
        captured["kwargs"] = kw

        class _R:
            returncode = 0
            stdout = "sent 100 bytes\n"
            stderr = ""

        return _R()

    return _run


def test_push_argv_includes_excludes(monkeypatch, tmp_path: Path):
    captured: dict = {}
    monkeypatch.setattr(subprocess, "run", _fake_completed(captured))
    runner = ShellRsyncRunner()
    result = runner.push(_host(), tmp_path, "~/IsaacLab")
    assert isinstance(result, RsyncResult)
    argv = captured["argv"]
    assert argv[0] == "rsync"
    # Standard flags.
    assert "-avz" in argv or ("-a" in argv and "-v" in argv and "-z" in argv)
    # Excludes for push.
    excludes = [a for a in argv if a.startswith("--exclude")]
    joined = " ".join(excludes)
    assert "__pycache__" in joined
    assert ".git" in joined
    assert "odin_runs" in joined


def test_push_destination_shape(monkeypatch, tmp_path: Path):
    captured: dict = {}
    monkeypatch.setattr(subprocess, "run", _fake_completed(captured))
    runner = ShellRsyncRunner()
    runner.push(_host(), tmp_path, "~/IsaacLab")
    argv = captured["argv"]
    # Last arg must be user@host:remote_path.
    assert argv[-1] == "odin@v1:~/IsaacLab"


def test_pull_destination_shape(monkeypatch, tmp_path: Path):
    captured: dict = {}
    monkeypatch.setattr(subprocess, "run", _fake_completed(captured))
    runner = ShellRsyncRunner()
    runner.pull(_host(), "~/IsaacLab/odin_runs/x", tmp_path / "x")
    argv = captured["argv"]
    # Source must carry a trailing slash so rsync copies the remote bundle's
    # *contents* into local_path rather than creating local_path/x/.
    assert "odin@v1:~/IsaacLab/odin_runs/x/" in argv
    assert str(tmp_path / "x") in argv


def test_pull_source_trailing_slash_idempotent(monkeypatch, tmp_path: Path):
    """Caller-supplied trailing slash must not double up into ``//``."""
    captured: dict = {}
    monkeypatch.setattr(subprocess, "run", _fake_completed(captured))
    runner = ShellRsyncRunner()
    runner.pull(_host(), "~/IsaacLab/odin_runs/x/", tmp_path / "x")
    argv = captured["argv"]
    assert "odin@v1:~/IsaacLab/odin_runs/x/" in argv
    assert "odin@v1:~/IsaacLab/odin_runs/x//" not in argv


def test_pull_no_delete(monkeypatch, tmp_path: Path):
    """Pull must NOT pass --delete (we don't want to prune the controller's prior bundles)."""
    captured: dict = {}
    monkeypatch.setattr(subprocess, "run", _fake_completed(captured))
    runner = ShellRsyncRunner()
    runner.pull(_host(), "~/odin_runs/x", tmp_path / "x")
    argv = captured["argv"]
    assert "--delete" not in argv


def test_ssh_key_threaded_through(monkeypatch, tmp_path: Path):
    key = tmp_path / "id"
    key.write_text("x")
    captured: dict = {}
    monkeypatch.setattr(subprocess, "run", _fake_completed(captured))
    runner = ShellRsyncRunner()
    runner.push(_host(ssh_key=key), tmp_path / "src", "~/dst")
    argv = captured["argv"]
    # -e 'ssh -i <key> ...' form.
    e_idx = argv.index("-e")
    assert str(key) in argv[e_idx + 1]
    assert argv[e_idx + 1].startswith("ssh ")
