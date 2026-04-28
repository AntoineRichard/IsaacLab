# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :mod:`tools.odin.asgard.recovery_cli`."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.odin.asgard import recovery_cli
from tools.odin.asgard.recovery import RecoveryResult


def _write_fleet_yaml(tmp_path: Path, host: str = "10.0.0.1") -> Path:
    fleet = tmp_path / "fleet.yaml"
    fleet.write_text(
        f"""fleet_name: test-fleet
default_ssh_user: horde
default_ssh_key: ~/.ssh/id_ed25519
hosts:
  - host: {host}
    container_name: isaac-lab-base
"""
    )
    return fleet


def test_main_recovered_returns_zero(tmp_path, monkeypatch, capsys):
    fleet = _write_fleet_yaml(tmp_path, host="10.0.0.1")

    def _fake_recover(host, *, ssh):
        assert host.host == "10.0.0.1"
        return RecoveryResult(
            host=host.host,
            container_name=host.container_name,
            attempted=True,
            recovered=True,
            duration_s=10.0,
            message="recovered_via_container_restart",
            details={"docker_restart": "ok", "container_up": "ok", "gpu_probe": "ok"},
        )

    monkeypatch.setattr(recovery_cli, "recover_valkyrie_gpu", _fake_recover)
    rc = recovery_cli.main(["--fleet", str(fleet), "--host", "10.0.0.1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "recovered_via_container_restart" in out


def test_main_not_recovered_returns_one(tmp_path, monkeypatch, capsys):
    fleet = _write_fleet_yaml(tmp_path, host="10.0.0.2")

    def _fake_recover(host, *, ssh):
        return RecoveryResult(
            host=host.host,
            container_name=host.container_name,
            attempted=True,
            recovered=False,
            duration_s=2.0,
            message="docker_restart_failed: daemon down",
            details={},
        )

    monkeypatch.setattr(recovery_cli, "recover_valkyrie_gpu", _fake_recover)
    rc = recovery_cli.main(["--fleet", str(fleet), "--host", "10.0.0.2"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "docker_restart_failed" in out


def test_main_unknown_host_errors(tmp_path, capsys):
    fleet = _write_fleet_yaml(tmp_path, host="10.0.0.1")
    with pytest.raises(SystemExit) as ei:
        recovery_cli.main(["--fleet", str(fleet), "--host", "10.0.0.99"])
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "10.0.0.99" in err
