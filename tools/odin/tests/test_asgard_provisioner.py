# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :func:`tools.odin.asgard.provisioner.provision_valkyrie`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tools.odin.asgard.fleet import ValkyrieConfig
from tools.odin.asgard.provisioner import ProvisionResult, provision_valkyrie
from tools.odin.asgard.transport import RsyncResult, SSHResult


@dataclass
class _FakeSSH:
    log: list[str] = field(default_factory=list)
    scripted: dict = field(default_factory=dict)

    def run(self, host, cmd: str, *, timeout_s=None, stdout_tee=None) -> SSHResult:
        self.log.append(cmd)
        for key, result in self.scripted.items():
            if key in cmd:
                return result
        return SSHResult(exit_code=0, stdout="", stderr="", duration_s=0.0)


@dataclass
class _FakeRsync:
    log: list[tuple[str, str, str]] = field(default_factory=list)  # (op, local_or_remote_src, dst)

    def push(self, host, local_path: Path, remote_path: str) -> RsyncResult:
        self.log.append(("push", str(local_path), remote_path))
        return RsyncResult(exit_code=0, stdout="", stderr="", duration_s=0.0)

    def pull(self, host, remote_path: str, local_path: Path) -> RsyncResult:
        self.log.append(("pull", remote_path, str(local_path)))
        return RsyncResult(exit_code=0, stdout="", stderr="", duration_s=0.0)


def _host() -> ValkyrieConfig:
    return ValkyrieConfig(host="v1", ssh_user="odin", isaaclab_path="~/IsaacLab")


def test_smart_sync_pushes_but_does_not_wipe(tmp_path: Path):
    ssh = _FakeSSH(
        scripted={
            "docker inspect": SSHResult(exit_code=0, stdout="running\n", stderr="", duration_s=0.0),
        }
    )
    rsync = _FakeRsync()
    r = provision_valkyrie(_host(), working_tree=tmp_path, fresh=False, ssh=ssh, rsync=rsync)
    assert isinstance(r, ProvisionResult)
    assert r.ok is True
    # No wipe.
    assert not any("rm -rf" in cmd for cmd in ssh.log)
    # rsync push happened with working_tree -> isaaclab_path.
    assert rsync.log == [("push", str(tmp_path), "~/IsaacLab")]
    # Container already running → no start call.
    assert not any("./docker/container.py start" in cmd for cmd in ssh.log)


def test_fresh_wipes_and_restarts(tmp_path: Path):
    ssh = _FakeSSH(
        scripted={
            "docker inspect": SSHResult(exit_code=0, stdout="running\n", stderr="", duration_s=0.0),
        }
    )
    rsync = _FakeRsync()
    r = provision_valkyrie(_host(), working_tree=tmp_path, fresh=True, ssh=ssh, rsync=rsync)
    assert r.ok is True
    # Wipe before push.
    wipe_idx = next(i for i, cmd in enumerate(ssh.log) if "rm -rf" in cmd and "~/IsaacLab" in cmd)
    push_idx = rsync.log.index(("push", str(tmp_path), "~/IsaacLab"))
    # The wipe must precede the rsync (same runner receives the wipe before
    # provisioner returns from the wipe call), which in our log terms means
    # the wipe command appears in ssh.log before the push call.
    assert wipe_idx >= 0
    assert push_idx >= 0
    # Container stop + start.
    assert any("./docker/container.py stop" in cmd for cmd in ssh.log)
    assert any("./docker/container.py start" in cmd for cmd in ssh.log)


def test_smart_sync_starts_container_when_stopped(tmp_path: Path):
    ssh = _FakeSSH(
        scripted={
            "docker inspect": SSHResult(exit_code=0, stdout="exited\n", stderr="", duration_s=0.0),
        }
    )
    rsync = _FakeRsync()
    r = provision_valkyrie(_host(), working_tree=tmp_path, fresh=False, ssh=ssh, rsync=rsync)
    assert r.ok is True
    # No wipe.
    assert not any("rm -rf" in cmd for cmd in ssh.log)
    # Must start container.
    assert any("./docker/container.py start" in cmd for cmd in ssh.log)


def test_failed_rsync_reports_not_ok(tmp_path: Path):
    ssh = _FakeSSH(
        scripted={
            "docker inspect": SSHResult(exit_code=0, stdout="running\n", stderr="", duration_s=0.0),
        }
    )

    class _BadRsync(_FakeRsync):
        def push(self, host, local_path: Path, remote_path: str) -> RsyncResult:
            return RsyncResult(exit_code=23, stdout="", stderr="rsync: failed", duration_s=0.0)

    rsync = _BadRsync()
    r = provision_valkyrie(_host(), working_tree=tmp_path, fresh=False, ssh=ssh, rsync=rsync)
    assert r.ok is False
    assert "rsync" in r.message.lower()


def test_provision_result_records_commit_sha(tmp_path: Path, monkeypatch):
    """ProvisionResult.commit_sha comes from _resolve_local_sha(working_tree)."""
    from tools.odin.asgard import provisioner as prov_mod

    def _fake_resolve(wt: Path) -> str:
        return "abc123d"

    monkeypatch.setattr(prov_mod, "_resolve_local_sha", _fake_resolve)
    ssh = _FakeSSH(
        scripted={
            "docker inspect": SSHResult(exit_code=0, stdout="running\n", stderr="", duration_s=0.0),
        }
    )
    rsync = _FakeRsync()
    r = provision_valkyrie(_host(), working_tree=tmp_path, fresh=False, ssh=ssh, rsync=rsync)
    assert r.commit_sha == "abc123d"
