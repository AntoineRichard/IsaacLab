# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the odin-cuda CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.odin.asgard.cuda_install import CheckResult
from tools.odin.asgard.cuda_install_cli import main, parse_args


def _write_fleet_yaml(tmp_path: Path) -> Path:
    content = """\
fleet_name: test
default_ssh_user: odin
default_ssh_key: ~/.ssh/id_ed25519
hosts:
  - host: v1.internal
  - host: v2.internal
"""
    path = tmp_path / "fleet.yaml"
    path.write_text(content)
    return path


def test_parse_args_check_minimal(tmp_path: Path):
    fleet_path = _write_fleet_yaml(tmp_path)
    args = parse_args(["check", "--fleet", str(fleet_path)])
    assert args.subcommand == "check"
    assert args.fleet == fleet_path
    assert args.floor == "12.4"
    assert args.verbose is False


def test_parse_args_check_custom_floor(tmp_path: Path):
    fleet_path = _write_fleet_yaml(tmp_path)
    args = parse_args(["check", "--fleet", str(fleet_path), "--floor", "12.6", "--verbose"])
    assert args.floor == "12.6"
    assert args.verbose is True


def test_parse_args_no_subcommand_errors(tmp_path: Path):
    with pytest.raises(SystemExit):
        parse_args([])


def test_main_check_exit_zero_when_all_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    fleet_path = _write_fleet_yaml(tmp_path)

    def _fake_check_fleet(fleet, *, ssh, floor, parallel):
        return [
            CheckResult(host="v1.internal", status="ok", driver="575.1", cuda="12.9"),
            CheckResult(host="v2.internal", status="ok", driver="575.1", cuda="12.9"),
        ]

    monkeypatch.setattr("tools.odin.asgard.cuda_install_cli.check_fleet", _fake_check_fleet)
    code = main(["check", "--fleet", str(fleet_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "v1.internal" in out
    assert "12.9" in out
    assert "ok" in out


def test_main_check_exit_one_when_any_below_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    fleet_path = _write_fleet_yaml(tmp_path)

    def _fake_check_fleet(fleet, *, ssh, floor, parallel):
        return [
            CheckResult(host="v1", status="ok", driver="575.1", cuda="12.9"),
            CheckResult(host="v2", status="needs-upgrade", driver="535.1", cuda="12.2"),
        ]

    monkeypatch.setattr("tools.odin.asgard.cuda_install_cli.check_fleet", _fake_check_fleet)
    code = main(["check", "--fleet", str(fleet_path)])
    out = capsys.readouterr().out
    assert code == 1
    assert "needs-upgrade" in out
    assert "v2" in out
